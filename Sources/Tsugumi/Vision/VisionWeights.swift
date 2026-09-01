import Foundation
import Darwin
import Metal
import MoEPackFormat

/// The vision tower's weights, mapped out of `vision/vision_weights.bin`.
///
/// The tower lives in its own file rather than in `model_weights.bin`
/// (`PLAN_VISION.md` §4-1) precisely so that a text-only run never pays for it:
/// `Model.load` hashes `model_weights.bin` eagerly, and folding 1.15 GB into
/// that hash would slow down every launch whether or not an image is ever
/// passed. Nothing here runs until the first image arrives.
///
/// The file is the same resident-index v1 container the text weights use, so
/// the reader is the same one; what differs is the schema it is checked
/// against, which is derived from `VisionConfig` rather than trusted from the
/// file (§0-D-2 — the installer derives the same table independently, and the
/// two derivations exist to disagree).
public final class VisionWeights {
    public let config: VisionConfig
    /// Hidden size of the text model the projector emits into.
    public let textHiddenSize: Int
    /// Path this tower was loaded from, relative to the model directory.
    public let relativePath: String

    let residentIndex: ResidentIndex
    let residentBuffer: ResidentBuffer

    init(config: VisionConfig,
         textHiddenSize: Int,
         relativePath: String,
         residentIndex: ResidentIndex,
         residentBuffer: ResidentBuffer) {
        self.config = config
        self.textHiddenSize = textHiddenSize
        self.relativePath = relativePath
        self.residentIndex = residentIndex
        self.residentBuffer = residentBuffer
    }

    // MARK: - Loading

    /// Map and validate the tower declared by `manifest.vision`.
    ///
    /// `integrityPolicy` decides whether the payload is hashed here. Under
    /// `.fullSha256` it is — 1.15 GB, about a second — and that cost lands on
    /// the first image rather than on load. Under `.sizeCheckTrustedReceipt`
    /// the receipt already covers the file and only its size is re-checked,
    /// exactly as the text weights are treated.
    public static func load(directoryURL: URL,
                            manifest: Manifest,
                            config: ArchConfig,
                            device: MTLDevice,
                            integrityPolicy: ModelIntegrityPolicy) throws -> VisionWeights {
        guard let vision = manifest.vision else {
            throw VisionError.towerNotInstalled(path: directoryURL.path)
        }
        guard let entry = manifest.files[vision.weightsPath] else {
            throw ModelError.missingFile(name: vision.weightsPath)
        }

        let modelDirectory = try MoEPackModelDirectory(rootURL: directoryURL)
        let fd = try modelDirectory.openFile(vision.weightsPath)
        defer { close(fd) }

        let actualSize = try modelDirectory.fileSize(fileDescriptor: fd,
                                                     relativePath: vision.weightsPath)
        guard actualSize == entry.size else {
            throw ModelError.tensorSizeMismatch(name: vision.weightsPath,
                                                expected: entry.size,
                                                actual: actualSize)
        }
        if integrityPolicy == .fullSha256 {
            try Sha256Verifier.verifyFile(fileDescriptor: fd,
                                          named: vision.weightsPath,
                                          expectedHex: entry.sha256)
        }

        let residentIndex = try ResidentIndexReader.load(fileDescriptor: fd,
                                                         displayPath: vision.weightsPath)
        let (indexEnd, overflow) = residentIndex.header.indexSize
            .addingReportingOverflow(residentIndex.header.residentSize)
        guard !overflow, indexEnd == actualSize else {
            throw ModelError.indexCorrupt(detail: """
                \(vision.weightsPath) size \(actualSize) != indexSize \
                \(residentIndex.header.indexSize) + residentSize \
                \(residentIndex.header.residentSize)
                """)
        }

        // `manifest.vision` was already checked field-by-field against
        // `VisionConfig.gemma4Vision` by `ManifestReader`; the runtime schema is
        // therefore derived from the compile-time config, not from the manifest.
        let expected = VisionConfig.gemma4Vision
        try validateSchema(residentIndex: residentIndex,
                           config: expected,
                           textHiddenSize: config.hiddenSize,
                           declared: vision)

        let buffer = try ResidentBuffer(
            fileURL: directoryURL.appendingPathComponent(vision.weightsPath),
            fileOffset: residentIndex.header.indexSize,
            residentSize: residentIndex.header.residentSize,
            device: device,
            fileDescriptor: fd)

        return VisionWeights(config: expected,
                             textHiddenSize: config.hiddenSize,
                             relativePath: vision.weightsPath,
                             residentIndex: residentIndex,
                             residentBuffer: buffer)
    }

    // MARK: - Schema

    /// Every tensor the tower must contain, with its shape, derived from the
    /// config. `2 + 13 * numLayers + 2 + 1` entries — 356 for Gemma 4.
    public static func expectedInventory(config: VisionConfig,
                                         textHiddenSize: Int) -> [(name: String, shape: [Int])] {
        let hidden = config.hiddenSize
        let queryDim = config.numHeads * config.headDim
        let kvDim = config.numKVHeads * config.headDim
        var out: [(String, [Int])] = [
            ("vision_tower.patch_embedder.input_proj.weight",
             [hidden, config.patchSize * config.patchSize * 3]),
            ("vision_tower.patch_embedder.position_embedding_table",
             [2, config.positionEmbeddingSize, hidden]),
        ]
        for layer in 0..<config.numLayers {
            let prefix = "vision_tower.encoder.layers.\(layer)."
            out.append((prefix + "self_attn.q_proj.linear.weight", [queryDim, hidden]))
            out.append((prefix + "self_attn.k_proj.linear.weight", [kvDim, hidden]))
            out.append((prefix + "self_attn.v_proj.linear.weight", [kvDim, hidden]))
            out.append((prefix + "self_attn.o_proj.linear.weight", [hidden, queryDim]))
            out.append((prefix + "self_attn.q_norm.weight", [config.headDim]))
            out.append((prefix + "self_attn.k_norm.weight", [config.headDim]))
            out.append((prefix + "mlp.gate_proj.linear.weight", [config.intermediateSize, hidden]))
            out.append((prefix + "mlp.up_proj.linear.weight", [config.intermediateSize, hidden]))
            out.append((prefix + "mlp.down_proj.linear.weight", [hidden, config.intermediateSize]))
            for norm in ["input_layernorm", "post_attention_layernorm",
                         "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                out.append((prefix + norm + ".weight", [hidden]))
            }
        }
        out.append(("vision_tower.std_scale", [hidden]))
        out.append(("vision_tower.std_bias", [hidden]))
        out.append(("embed_vision.embedding_projection.weight", [textHiddenSize, hidden]))
        return out.map { (name: $0.0, shape: $0.1) }
    }

    static func validateSchema(residentIndex: ResidentIndex,
                               config: VisionConfig,
                               textHiddenSize: Int,
                               declared: ManifestVision) throws {
        let inventory = expectedInventory(config: config, textHiddenSize: textHiddenSize)
        guard inventory.count == declared.tensorCount else {
            throw ModelError.indexCorrupt(detail: """
                vision inventory derived from the runtime config has \(inventory.count) \
                tensors, manifest declares \(declared.tensorCount)
                """)
        }
        // An extra tensor is as much a schema break as a missing one: it means
        // the installed tower is not the one this runtime knows how to run.
        guard residentIndex.entries.count == inventory.count else {
            let expectedNames = Set(inventory.map(\.name))
            let extra = residentIndex.entries.keys
                .filter { !expectedNames.contains($0) }.sorted()
            let missing = expectedNames.subtracting(residentIndex.entries.keys).sorted()
            throw ModelError.indexCorrupt(detail: """
                vision index has \(residentIndex.entries.count) tensors, expected \
                \(inventory.count); unexpected: \(extra.prefix(4).joined(separator: ", ")); \
                missing: \(missing.prefix(4).joined(separator: ", "))
                """)
        }

        var payloadBytes: UInt64 = 0
        for (name, shape) in inventory {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.tensorNotFound(name: name)
            }
            var padded = shape.map { UInt32($0) }
            while padded.count < 4 { padded.append(0) }
            guard padded.count == 4,
                  entry.shape.0 == padded[0], entry.shape.1 == padded[1],
                  entry.shape.2 == padded[2], entry.shape.3 == padded[3] else {
                throw ModelError.indexCorrupt(
                    detail: "vision tensor \(name) has shape "
                            + "[\(entry.shape.0),\(entry.shape.1),\(entry.shape.2),\(entry.shape.3)], "
                            + "expected \(shape)")
            }
            let elements = shape.reduce(UInt64(1)) { $0 * UInt64($1) }
            let expectedBytes = elements * UInt64(MemoryLayout<UInt16>.size)
            guard entry.dtype == MoEPackFormatV1.DType.bf16.rawValue,
                  entry.sizeBytes == expectedBytes,
                  entry.scaleOffset == 0, entry.scaleSize == 0,
                  entry.biasOffset == 0, entry.biasSize == 0,
                  entry.fileOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "vision tensor \(name) does not match the required BF16 schema")
            }
            payloadBytes += expectedBytes
        }

        guard payloadBytes == declared.payloadBytes,
              payloadBytes == residentIndex.header.residentSize else {
            throw ModelError.indexCorrupt(detail: """
                vision payload is \(residentIndex.header.residentSize) bytes, the schema \
                needs \(payloadBytes), the manifest declares \(declared.payloadBytes)
                """)
        }
    }

    // MARK: - Accessors

    /// `[hidden, 768]` patch projection.
    public var patchProjection: TensorView {
        get throws { try tensor(named: "vision_tower.patch_embedder.input_proj.weight") }
    }
    /// `[2, positionEmbeddingSize, hidden]`: row 0 is indexed by x, row 1 by y.
    public var positionTable: TensorView {
        get throws { try tensor(named: "vision_tower.patch_embedder.position_embedding_table") }
    }
    public var stdScale: TensorView {
        get throws { try tensor(named: "vision_tower.std_scale") }
    }
    public var stdBias: TensorView {
        get throws { try tensor(named: "vision_tower.std_bias") }
    }
    /// `[textHiddenSize, hidden]` multimodal projector.
    public var projector: TensorView {
        get throws { try tensor(named: "embed_vision.embedding_projection.weight") }
    }

    public func qProj(layer L: Int) throws -> TensorView { try layerTensor(L, "self_attn.q_proj.linear.weight") }
    public func kProj(layer L: Int) throws -> TensorView { try layerTensor(L, "self_attn.k_proj.linear.weight") }
    public func vProj(layer L: Int) throws -> TensorView { try layerTensor(L, "self_attn.v_proj.linear.weight") }
    public func oProj(layer L: Int) throws -> TensorView { try layerTensor(L, "self_attn.o_proj.linear.weight") }
    public func qNorm(layer L: Int) throws -> TensorView { try layerTensor(L, "self_attn.q_norm.weight") }
    public func kNorm(layer L: Int) throws -> TensorView { try layerTensor(L, "self_attn.k_norm.weight") }
    public func gateProj(layer L: Int) throws -> TensorView { try layerTensor(L, "mlp.gate_proj.linear.weight") }
    public func upProj(layer L: Int) throws -> TensorView { try layerTensor(L, "mlp.up_proj.linear.weight") }
    public func downProj(layer L: Int) throws -> TensorView { try layerTensor(L, "mlp.down_proj.linear.weight") }
    public func inputNorm(layer L: Int) throws -> TensorView { try layerTensor(L, "input_layernorm.weight") }
    public func postAttentionNorm(layer L: Int) throws -> TensorView { try layerTensor(L, "post_attention_layernorm.weight") }
    public func preFeedForwardNorm(layer L: Int) throws -> TensorView { try layerTensor(L, "pre_feedforward_layernorm.weight") }
    public func postFeedForwardNorm(layer L: Int) throws -> TensorView { try layerTensor(L, "post_feedforward_layernorm.weight") }

    private func layerTensor(_ layer: Int, _ suffix: String) throws -> TensorView {
        guard layer >= 0 && layer < config.numLayers else {
            throw ModelError.tensorNotFound(
                name: "vision_tower.encoder.layers.\(layer).\(suffix)")
        }
        return try tensor(named: "vision_tower.encoder.layers.\(layer).\(suffix)")
    }

    /// Resolve a name against the tower's own resident buffer. Offsets in the
    /// index are absolute file offsets, so the index region is subtracted the
    /// same way `Model.resident` does it for the text weights.
    func tensor(named name: String) throws -> TensorView {
        guard let entry = residentIndex.entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        let base = residentIndex.header.indexSize
        guard entry.fileOffset >= base else {
            throw ModelError.indexCorrupt(detail: "\(name) precedes the vision payload")
        }
        let relative = entry.fileOffset - base
        guard relative <= residentIndex.header.residentSize,
              entry.sizeBytes <= residentIndex.header.residentSize - relative else {
            throw ModelError.indexCorrupt(detail: "\(name) exceeds the vision payload")
        }
        return TensorView(buffer: residentBuffer.buffer,
                          offset: relative,
                          length: entry.sizeBytes,
                          scaleOffset: 0, scaleLength: 0,
                          biasOffset: 0, biasLength: 0,
                          shape: entry.shape,
                          dtype: entry.dtype)
    }
}
