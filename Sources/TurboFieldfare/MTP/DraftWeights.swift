import Foundation
import Darwin
import Metal
import TurboFieldfareFormat

/// Drafter geometry the runtime runs with. Every field comes from
/// `manifest.draft`, which the codec already checked against the target arch:
/// the drafter reads the target's K/V, so none of these are free parameters
/// (`docs/mtp/03-DESIGN.md` D1).
public struct DraftConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let intermediateSize: Int
    public let backboneHiddenSize: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let rmsNormEps: Double
    /// One entry per layer: 1 = full attention, 0 = sliding.
    public let fullAttentionLayerMask: [Int]
    /// Target layers whose K/V the drafter attends to.
    public let sharedSlidingKVLayer: Int
    public let sharedFullKVLayer: Int
    public let quantBits: Int
    public let quantGroupSize: Int
    /// How the drafter's own 4-bit groups encode their zero point. Independent
    /// of the target's: the drafter is a separate checkpoint packed on its own
    /// terms, and it gets a shader library specialized for this value
    /// (`DraftContextPlan`).
    public let quantScheme: Quantization.AffineScheme

    public func headDim(forLayer layer: Int) -> Int {
        fullAttentionLayerMask[layer] == 1 ? fullHeadDim : headDim
    }
}

public enum DraftError: Error, CustomStringConvertible {
    case notInstalled(path: String)
    case geometryMismatch(String)

    public var description: String {
        switch self {
        case let .notInstalled(path):
            return "no MTP drafter installed at \(path)"
        case let .geometryMismatch(detail):
            return "MTP drafter geometry mismatch: \(detail)"
        }
    }
}

/// The MTP drafter's weights, mapped out of `draft/draft_weights.bin`.
///
/// Same lazy contract as the vision tower (`PLAN_VISION.md` §4-1, MTP D1):
/// nothing here runs until the drafter is first asked for, so a run that never
/// drafts never opens the file. The resident file is the MLX conversion's own
/// int4 layout (U32 packed + BF16 scales/biases, group 64) re-indexed, so the
/// schema check validates quantized entries against their packed and companion
/// shapes rather than a plain BF16 size.
public final class DraftWeights {
    public let config: DraftConfig
    /// Path this drafter was loaded from, relative to the model directory.
    public let relativePath: String

    let residentIndex: ResidentIndex
    let residentBuffer: ResidentBuffer

    init(config: DraftConfig,
         relativePath: String,
         residentIndex: ResidentIndex,
         residentBuffer: ResidentBuffer) {
        self.config = config
        self.relativePath = relativePath
        self.residentIndex = residentIndex
        self.residentBuffer = residentBuffer
    }

    // MARK: - Loading

    public static func load(directoryURL: URL,
                            manifest: Manifest,
                            arch: ArchConfig,
                            device: MTLDevice,
                            integrityPolicy: ModelIntegrityPolicy) throws -> DraftWeights {
        guard let draft = manifest.draft else {
            throw DraftError.notInstalled(path: directoryURL.path)
        }
        guard let entry = manifest.files[draft.weightsPath] else {
            throw ModelError.missingFile(name: draft.weightsPath)
        }

        let modelDirectory = try GTurboModelDirectory(rootURL: directoryURL)
        let fd = try modelDirectory.openFile(draft.weightsPath)
        defer { close(fd) }

        let actualSize = try modelDirectory.fileSize(fileDescriptor: fd,
                                                     relativePath: draft.weightsPath)
        guard actualSize == entry.size else {
            throw ModelError.tensorSizeMismatch(name: draft.weightsPath,
                                                expected: entry.size,
                                                actual: actualSize)
        }
        if integrityPolicy == .fullSha256 {
            try Sha256Verifier.verifyFile(fileDescriptor: fd,
                                          named: draft.weightsPath,
                                          expectedHex: entry.sha256)
        }

        let residentIndex = try ResidentIndexReader.load(fileDescriptor: fd,
                                                         displayPath: draft.weightsPath)
        let (indexEnd, overflow) = residentIndex.header.indexSize
            .addingReportingOverflow(residentIndex.header.residentSize)
        guard !overflow, indexEnd == actualSize else {
            throw ModelError.indexCorrupt(detail: """
                \(draft.weightsPath) size \(actualSize) != indexSize \
                \(residentIndex.header.indexSize) + residentSize \
                \(residentIndex.header.residentSize)
                """)
        }

        let config = DraftConfig(manifest: draft)
        try config.crossCheck(against: arch)
        try validateSchema(residentIndex: residentIndex,
                           config: config,
                           declared: draft)

        let buffer = try ResidentBuffer(
            fileURL: directoryURL.appendingPathComponent(draft.weightsPath),
            fileOffset: residentIndex.header.indexSize,
            residentSize: residentIndex.header.residentSize,
            device: device,
            fileDescriptor: fd)

        return DraftWeights(config: config,
                            relativePath: draft.weightsPath,
                            residentIndex: residentIndex,
                            residentBuffer: buffer)
    }

    // MARK: - Schema

    public struct ExpectedEntry {
        public let name: String
        public let shape: [Int]
        public let quantized: Bool
    }

    /// `3 + 11 * numLayers + 2` entries — 48 for the pinned drafter. Derived
    /// from the config the same way `DraftRepackPlanner` derives its plan, so
    /// the two derivations exist to disagree.
    public static func expectedInventory(config: DraftConfig) -> [ExpectedEntry] {
        func dims(_ values: Int...) -> [Int] { values }
        let hidden = config.hiddenSize
        var out: [ExpectedEntry] = [
            ExpectedEntry(name: "embed_tokens.weight",
                          shape: dims(config.vocabSize, hidden), quantized: true),
            ExpectedEntry(name: "pre_projection.weight",
                          shape: dims(hidden, 2 * config.backboneHiddenSize), quantized: true),
        ]
        for layer in 0..<config.numLayers {
            let prefix = "layers.\(layer)."
            let headDim = config.headDim(forLayer: layer)
            let qOut = config.numHeads * headDim
            out.append(ExpectedEntry(name: prefix + "self_attn.q_proj.weight",
                                     shape: dims(qOut, hidden), quantized: true))
            out.append(ExpectedEntry(name: prefix + "self_attn.q_norm.weight",
                                     shape: dims(headDim), quantized: false))
            out.append(ExpectedEntry(name: prefix + "self_attn.o_proj.weight",
                                     shape: dims(hidden, qOut), quantized: true))
            out.append(ExpectedEntry(name: prefix + "mlp.gate_proj.weight",
                                     shape: dims(config.intermediateSize, hidden), quantized: true))
            out.append(ExpectedEntry(name: prefix + "mlp.up_proj.weight",
                                     shape: dims(config.intermediateSize, hidden), quantized: true))
            out.append(ExpectedEntry(name: prefix + "mlp.down_proj.weight",
                                     shape: dims(hidden, config.intermediateSize), quantized: true))
            for norm in ["input_layernorm", "post_attention_layernorm",
                         "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                out.append(ExpectedEntry(name: prefix + norm + ".weight",
                                         shape: dims(hidden), quantized: false))
            }
            out.append(ExpectedEntry(name: prefix + "layer_scalar",
                                     shape: dims(1), quantized: false))
        }
        out.append(ExpectedEntry(name: "norm.weight",
                                 shape: dims(hidden), quantized: false))
        out.append(ExpectedEntry(name: "post_projection.weight",
                                 shape: dims(config.backboneHiddenSize, hidden), quantized: true))
        return out
    }

    static func validateSchema(residentIndex: ResidentIndex,
                               config: DraftConfig,
                               declared: ManifestDraft) throws {
        let inventory = expectedInventory(config: config)
        guard inventory.count == declared.tensorCount else {
            throw ModelError.indexCorrupt(detail: """
                drafter inventory derived from the manifest config has \
                \(inventory.count) tensors, manifest declares \(declared.tensorCount)
                """)
        }
        guard residentIndex.entries.count == inventory.count else {
            let expectedNames = Set(inventory.map(\.name))
            let extra = residentIndex.entries.keys
                .filter { !expectedNames.contains($0) }.sorted()
            let missing = expectedNames.subtracting(residentIndex.entries.keys).sorted()
            throw ModelError.indexCorrupt(detail: """
                drafter index has \(residentIndex.entries.count) tensors, expected \
                \(inventory.count); unexpected: \(extra.prefix(4).joined(separator: ", ")); \
                missing: \(missing.prefix(4).joined(separator: ", "))
                """)
        }

        let group = config.quantGroupSize
        let bits = config.quantBits
        var payloadBytes: UInt64 = 0
        for item in inventory {
            guard let entry = residentIndex.entries[item.name] else {
                throw ModelError.tensorNotFound(name: item.name)
            }
            var padded = item.shape.map { UInt32($0) }
            while padded.count < 4 { padded.append(0) }
            guard padded.count == 4,
                  entry.shape.0 == padded[0], entry.shape.1 == padded[1],
                  entry.shape.2 == padded[2], entry.shape.3 == padded[3] else {
                throw ModelError.indexCorrupt(
                    detail: "drafter tensor \(item.name) has shape "
                            + "[\(entry.shape.0),\(entry.shape.1),\(entry.shape.2),\(entry.shape.3)], "
                            + "expected \(item.shape)")
            }
            let rows = item.shape.first ?? 0
            let cols = item.shape.count > 1 ? item.shape[1] : 0
            if item.quantized {
                guard cols % group == 0, entry.dtype == GTurboFormatV1.DType.u32.rawValue,
                      entry.fileOffset % 2 == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "drafter tensor \(item.name) does not match the required "
                                + "U32 affine schema")
                }
                let wordsPerRow = cols / (32 / bits)
                let groupsPerRow = cols / group
                let weightBytes = UInt64(rows) * UInt64(wordsPerRow) * 4
                let companionBytes = UInt64(rows) * UInt64(groupsPerRow) * 2
                guard entry.sizeBytes == weightBytes,
                      entry.scaleSize == companionBytes, entry.biasSize == companionBytes,
                      entry.scaleOffset % 2 == 0, entry.biasOffset % 2 == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "drafter tensor \(item.name) has quantized sizes "
                                + "w=\(entry.sizeBytes) s=\(entry.scaleSize) b=\(entry.biasSize), "
                                + "expected w=\(weightBytes) s=b=\(companionBytes)")
                }
                payloadBytes += weightBytes + 2 * companionBytes
            } else {
                let elements = item.shape.reduce(1, *)
                let expectedBytes = UInt64(elements) * UInt64(MemoryLayout<UInt16>.size)
                guard entry.dtype == GTurboFormatV1.DType.bf16.rawValue,
                      entry.sizeBytes == expectedBytes,
                      entry.scaleOffset == 0, entry.scaleSize == 0,
                      entry.biasOffset == 0, entry.biasSize == 0,
                      entry.fileOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "drafter tensor \(item.name) does not match the required "
                                + "BF16 schema")
                }
                payloadBytes += expectedBytes
            }
        }

        guard payloadBytes == declared.payloadBytes,
              payloadBytes == residentIndex.header.residentSize else {
            throw ModelError.indexCorrupt(detail: """
                drafter payload is \(residentIndex.header.residentSize) bytes, the schema \
                needs \(payloadBytes), the manifest declares \(declared.payloadBytes)
                """)
        }
    }

    // MARK: - Accessors

    /// `[262144, 1024]` tied lm head (and the drafter's own token table, which
    /// the forward never reads — the input embedding comes from the target).
    public var embedTokens: TensorView { get throws { try tensor(named: "embed_tokens.weight") } }
    /// `[1024, 5632]` — input is `concat(target_embed * √2816, target_hidden)`.
    public var preProjection: TensorView { get throws { try tensor(named: "pre_projection.weight") } }
    /// `[2816, 1024]` — next round's target-hidden stand-in.
    public var postProjection: TensorView { get throws { try tensor(named: "post_projection.weight") } }
    public var norm: TensorView { get throws { try tensor(named: "norm.weight") } }

    public func qProj(layer L: Int) throws -> TensorView { try layerTensor(L, "self_attn.q_proj.weight") }
    public func qNorm(layer L: Int) throws -> TensorView { try layerTensor(L, "self_attn.q_norm.weight") }
    public func oProj(layer L: Int) throws -> TensorView { try layerTensor(L, "self_attn.o_proj.weight") }
    public func gateProj(layer L: Int) throws -> TensorView { try layerTensor(L, "mlp.gate_proj.weight") }
    public func upProj(layer L: Int) throws -> TensorView { try layerTensor(L, "mlp.up_proj.weight") }
    public func downProj(layer L: Int) throws -> TensorView { try layerTensor(L, "mlp.down_proj.weight") }
    public func inputNorm(layer L: Int) throws -> TensorView { try layerTensor(L, "input_layernorm.weight") }
    public func postAttentionNorm(layer L: Int) throws -> TensorView { try layerTensor(L, "post_attention_layernorm.weight") }
    public func preFeedForwardNorm(layer L: Int) throws -> TensorView { try layerTensor(L, "pre_feedforward_layernorm.weight") }
    public func postFeedForwardNorm(layer L: Int) throws -> TensorView { try layerTensor(L, "post_feedforward_layernorm.weight") }

    /// The per-layer scale, read once on the CPU: it is a single BF16 value the
    /// kernels take as a `float` argument.
    public func layerScalar(layer L: Int) throws -> Float {
        let view = try layerTensor(L, "layer_scalar")
        precondition(view.length == 2, "layer_scalar is one BF16 value")
        let bits = view.buffer.contents()
            .advanced(by: Int(view.offset))
            .assumingMemoryBound(to: UInt16.self)[0]
        return Quantization.bf16ToFloat(bits)
    }

    private func layerTensor(_ layer: Int, _ suffix: String) throws -> TensorView {
        guard layer >= 0 && layer < config.numLayers else {
            throw ModelError.tensorNotFound(name: "layers.\(layer).\(suffix)")
        }
        return try tensor(named: "layers.\(layer).\(suffix)")
    }

    /// Same offset arithmetic as `VisionWeights.tensor`: index offsets are
    /// absolute file offsets, the buffer starts at the payload.
    func tensor(named name: String) throws -> TensorView {
        guard let entry = residentIndex.entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        let base = residentIndex.header.indexSize
        guard entry.fileOffset >= base else {
            throw ModelError.indexCorrupt(detail: "\(name) precedes the drafter payload")
        }
        let relative = entry.fileOffset - base
        let scaleRelative = entry.scaleSize > 0 ? entry.scaleOffset - base : 0
        let biasRelative = entry.biasSize > 0 ? entry.biasOffset - base : 0
        func fits(_ offset: UInt64, _ length: UInt64) -> Bool {
            length == 0 || (offset <= residentIndex.header.residentSize &&
                length <= residentIndex.header.residentSize - offset)
        }
        guard fits(relative, entry.sizeBytes),
              fits(scaleRelative, entry.scaleSize),
              fits(biasRelative, entry.biasSize) else {
            throw ModelError.indexCorrupt(detail: "\(name) exceeds the drafter payload")
        }
        return TensorView(buffer: residentBuffer.buffer,
                          offset: relative,
                          length: entry.sizeBytes,
                          scaleOffset: scaleRelative,
                          scaleLength: entry.scaleSize,
                          biasOffset: biasRelative,
                          biasLength: entry.biasSize,
                          shape: entry.shape,
                          dtype: entry.dtype)
    }
}

extension DraftConfig {
    init(manifest: ManifestDraft) {
        self.init(hiddenSize: manifest.hiddenSize,
                  numLayers: manifest.numLayers,
                  numHeads: manifest.numHeads,
                  numKVHeads: manifest.numKVHeads,
                  numFullKVHeads: manifest.numFullKVHeads,
                  headDim: manifest.headDim,
                  fullHeadDim: manifest.fullHeadDim,
                  intermediateSize: manifest.intermediateSize,
                  backboneHiddenSize: manifest.backboneHiddenSize,
                  vocabSize: manifest.vocabSize,
                  slidingWindow: manifest.slidingWindow,
                  ropeTheta: manifest.ropeTheta,
                  fullRopeTheta: manifest.fullRopeTheta,
                  partialRotaryFactor: manifest.partialRotaryFactor,
                  rmsNormEps: manifest.rmsNormEps,
                  fullAttentionLayerMask: manifest.fullAttentionLayerMask,
                  sharedSlidingKVLayer: manifest.sharedSlidingKVLayer,
                  sharedFullKVLayer: manifest.sharedFullKVLayer,
                  quantBits: manifest.quant.weightBits,
                  quantGroupSize: manifest.quant.groupSize,
                  quantScheme: manifest.quant.storesBias ? .affine : .sym)
    }

    /// The codec checked the manifest against *some* arch; this checks it
    /// against the one this process actually loaded — the same re-assertion
    /// `resolveGeometry` makes for the vision tower.
    func crossCheck(against arch: ArchConfig) throws {
        guard backboneHiddenSize == arch.hiddenSize,
              vocabSize == arch.vocabSize,
              slidingWindow == arch.slidingWindow,
              headDim == arch.headDim,
              fullHeadDim == arch.fullHeadDim,
              numKVHeads == arch.numKVHeads,
              numFullKVHeads == arch.numFullKVHeads,
              ropeTheta == arch.ropeTheta,
              fullRopeTheta == arch.fullRopeTheta,
              partialRotaryFactor == arch.partialRotaryFactor,
              fullAttentionLayerMask.last == 1,
              sharedSlidingKVLayer == arch.fullAttentionLayerMask.lastIndex(where: { $0 == 0 }),
              sharedFullKVLayer == arch.fullAttentionLayerMask.lastIndex(of: 1) else {
            throw DraftError.geometryMismatch("""
                drafter does not fit arch \(arch.hiddenSize)/\(arch.numLayers) \
                (shared KV \(sharedSlidingKVLayer)/\(sharedFullKVLayer))
                """)
        }
    }
}
