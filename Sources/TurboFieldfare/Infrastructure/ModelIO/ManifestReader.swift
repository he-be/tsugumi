import Foundation
import TurboFieldfareFormat

public struct ManifestFileEntry: Decodable, Equatable, Sendable {
    public let size: UInt64
    public let sha256: String
}

public struct ManifestArch: Decodable, Equatable, Sendable {
    public let hiddenSize: Int
    public let ffnIntermediate: Int
    public let moeIntermediateSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let hiddenActivation: String
    public let fullAttentionLayerMask: [Int]
}

public struct ManifestQuantSlot: Decodable, Equatable, Sendable {
    public let weightBits: Int
    public let scheme: String
    public let scaleType: String
    public let biasType: String
    public let groupSize: Int
    /// Whether this slot stores a per-group zero point. `sym` does not: the
    /// bias is `-8 * scale` and the shader derives it.
    public var storesBias: Bool { scheme.lowercased() != "sym" }

}

public struct ManifestQuant: Decodable, Equatable, Sendable {
    public let embedding: ManifestQuantSlot
    public let attention: ManifestQuantSlot
    public let router: ManifestQuantSlot
    public let sharedExpert: ManifestQuantSlot
    public let routedExpert: ManifestQuantSlot
}

public struct ManifestVision: Decodable, Equatable, Sendable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let positionEmbeddingSize: Int
    public let ropeTheta: Double
    public let rmsNormEps: Double
    public let hiddenActivation: String
    public let standardize: Bool
    public let maxSoftTokens: Int
    public let weightDType: String
    public let imageTokenID: Int
    public let boiTokenID: Int
    public let eoiTokenID: Int
    public let weightsPath: String
    public let tensorCount: Int
    public let payloadBytes: UInt64
    public let sourceRepo: String
    public let sourceRevision: String
}

/// MTP drafter description as the runtime reads it. The codec already
/// validated every field against the target arch (`validateDraftSection`):
/// the drafter has no K/V of its own, so its head geometry, window and RoPE
/// constants are re-statements of the target's rather than free parameters.
public struct ManifestDraft: Decodable, Equatable, Sendable {
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
    public let hiddenActivation: String
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [Int]
    public let sharedSlidingKVLayer: Int
    public let sharedFullKVLayer: Int
    public let quant: ManifestQuantSlot
    public let weightsPath: String
    public let tensorCount: Int
    public let payloadBytes: UInt64
    public let sourceRepo: String
    public let sourceRevision: String
}

public struct Manifest: Decodable, Equatable, Sendable {
    public let magic: String
    public let versionMajor: Int
    public let versionMinor: Int
    public let flags: [String: Bool]
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let arch: ManifestArch
    public let quant: ManifestQuant?
    public let vision: ManifestVision?
    public let draft: ManifestDraft?
    public let files: [String: ManifestFileEntry]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64
}

public enum ManifestReader {
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    /// Recognized flag keys. Anything else in `manifest.flags` is an error.
    public static let knownFlags: Set<String> = GTurboFormatV1.knownFlags

    /// Fixed required entries. Packed-layer filenames come from layout.json and
    /// are cross-validated only after that document is decoded.
    public static let requiredFiles: [String] = [
        "model_weights.bin",
        "packed_experts/layout.json",
    ]

    public static func load(directoryURL: URL,
                            expecting: ArchConfig,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> Manifest {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        return try decode(data: data, expecting: expecting)
    }

    package static func decode(data: Data,
                               expecting: ArchConfig) throws -> Manifest {
        let manifest: Manifest
        do {
            let wire = try GTurboManifestCodec.decodeUnchecked(data)
            guard wire.magic == GTurboFormatV1.magic else {
                throw ModelError.notAGTurboDirectory
            }
            guard wire.versionMajor == GTurboFormatV1.versionMajor,
                  wire.versionMinor >= 0 else {
                throw ModelError.unsupportedVersion(major: wire.versionMajor,
                                                    minor: wire.versionMinor)
            }
            for key in wire.flags.keys where !GTurboFormatV1.knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
            if wire.expertStride % GTurboFormatV1.alignmentBytes != 0 {
                throw ModelError.expertStrideNotPageAligned(
                    stride: wire.expertStride,
                    pageSize: Int(GTurboFormatV1.alignmentBytes))
            }
            try GTurboManifestCodec.validate(wire)
            manifest = Manifest(wire: wire)
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }

        try validate(manifest, against: expecting)
        return manifest
    }

    static func validate(_ m: Manifest,
                         against expected: ArchConfig) throws {
        if m.flags["turboQuantKV"] == true {
            throw ModelError.indexCorrupt(
                detail: "manifest requests removed TurboQuant KV runtime support")
        }
        try validateArch(m.arch, expected: expected)
        if let vision = m.vision {
            try validateVision(vision, expected: .gemma4Vision)
            if m.files[vision.weightsPath] == nil {
                throw ModelError.missingFile(name: vision.weightsPath)
            }
        }
        if let draft = m.draft, m.files[draft.weightsPath] == nil {
            throw ModelError.missingFile(name: draft.weightsPath)
        }
        if let quant = m.quant {
            try validateQuant(quant)
        } else if expected.numLayers == ArchConfig.gemma4_26B_A4B.numLayers,
                  expected.hiddenSize == ArchConfig.gemma4_26B_A4B.hiddenSize {
            throw ModelError.indexCorrupt(detail: "manifest.quant is required for the production architecture")
        }
        for f in requiredFiles {
            if m.files[f] == nil { throw ModelError.missingFile(name: f) }
        }
    }

    private static func validateQuant(_ quant: ManifestQuant) throws {
        let slots: [(String, ManifestQuantSlot, Set<Int>)] = [
            ("embedding", quant.embedding, [4]),
            ("attention", quant.attention, [4]),
            ("router", quant.router, [8, 16]),
            ("sharedExpert", quant.sharedExpert, [4, 8]),
            ("routedExpert", quant.routedExpert, [4]),
        ]
        for (name, slot, allowedBits) in slots {
            guard allowedBits.contains(slot.weightBits) else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
            if slot.weightBits == 16 {
                // Unquantized BF16 — the QAT checkpoints ship the router this
                // way. No affine companions, so scheme/scale/bias must say so.
                guard slot.scheme.lowercased() == "bf16",
                      slot.scaleType.lowercased() == "none",
                      slot.biasType.lowercased() == "none" else {
                    throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
                }
            } else if slot.scheme.lowercased() == "sym" {
                // `sym` drops the bias array: the checkpoint satisfies
                // `bias == -8 * scale` in every group, so the shader library --
                // compiled with `TURBO_AFFINE_SYMMETRIC` -- derives it
                // (`docs/mtp/44-W1-WEIGHT-DIET.md`). Only the 4-bit lattice has
                // that property; an INT8 group's zero point is real data.
                guard slot.weightBits == 4,
                      slot.scaleType.lowercased() == "bf16",
                      slot.biasType.lowercased() == "none" else {
                    throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
                }
            } else {
                guard slot.scheme.lowercased() == "affine",
                      slot.scaleType.lowercased() == "bf16",
                      slot.biasType.lowercased() == "bf16" else {
                    throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
                }
            }
        }
        // The scheme is a whole-model property for the same reason the group
        // size is: one compiled shader library serves every 4-bit slot. Slots
        // that carry no affine metadata (the BF16 router, an INT8 shared
        // expert) are exempt and keep their own scheme string.
        let symmetricSlots = slots.filter { $0.1.weightBits == 4 }
        let symmetric = symmetricSlots.map { $0.1.scheme.lowercased() == "sym" }
        guard symmetric.allSatisfy({ $0 == symmetric[0] }) else {
            throw ModelError.indexCorrupt(
                detail: "4-bit slots disagree on the affine scheme: "
                    + symmetricSlots.map { "\($0.0)=\($0.1.scheme)" }.joined(separator: ", "))
        }
        // The affine group size is a whole-model property: the shader library is
        // compiled with one baked-in value (`MetalContext.affineGroupSize`), and
        // `Model.affineGroupSize` reads the embedding slot on behalf of all of
        // them. Reject a manifest whose slots disagree. The BF16 router carries
        // the model's base group size but does not use it.
        let groupSize = quant.embedding.groupSize
        guard Quantization.supportedGroupSizes.contains(groupSize) else {
            throw ModelError.indexCorrupt(
                detail: "unsupported affine group size \(groupSize)")
        }
        for (name, slot, _) in slots where slot.groupSize != groupSize {
            throw ModelError.indexCorrupt(
                detail: "\(name) group size \(slot.groupSize) disagrees with the model's \(groupSize)")
        }
    }

    /// Field-by-field, in the same shape as `validateArch`. A tower whose
    /// geometry disagrees with the compiled kernels would not fail loudly — it
    /// would produce plausible-looking soft tokens — so the mismatch is caught
    /// here rather than in the first image.
    private static func validateVision(_ v: ManifestVision,
                                       expected e: VisionConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: "vision.\(field)",
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        try check("hiddenSize",            v.hiddenSize,            e.hiddenSize)
        try check("numLayers",             v.numLayers,             e.numLayers)
        try check("numHeads",              v.numHeads,              e.numHeads)
        try check("numKVHeads",            v.numKVHeads,            e.numKVHeads)
        try check("headDim",               v.headDim,               e.headDim)
        try check("intermediateSize",      v.intermediateSize,      e.intermediateSize)
        try check("patchSize",             v.patchSize,             e.patchSize)
        try check("poolingKernelSize",     v.poolingKernelSize,     e.poolingKernelSize)
        try check("positionEmbeddingSize", v.positionEmbeddingSize, e.positionEmbeddingSize)
        try check("ropeTheta",             v.ropeTheta,             e.ropeTheta)
        try check("rmsNormEps",            v.rmsNormEps,            e.rmsNormEps)
        try check("hiddenActivation",      v.hiddenActivation,      e.hiddenActivation)
        try check("standardize",           v.standardize,           e.standardize)
        try check("maxSoftTokens",         v.maxSoftTokens,         e.maxSoftTokens)
        try check("imageTokenID",          v.imageTokenID,          e.imageTokenID)
        try check("boiTokenID",            v.boiTokenID,            e.boiTokenID)
        try check("eoiTokenID",            v.eoiTokenID,            e.eoiTokenID)
        try check("weightDType",           v.weightDType.lowercased(), "bf16")
    }

    private static func validateArch(_ a: ManifestArch,
                                     expected e: ArchConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field,
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        try check("hiddenSize",          a.hiddenSize,          e.hiddenSize)
        try check("ffnIntermediate",     a.ffnIntermediate,     e.intermediateSize)
        try check("moeIntermediateSize", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("numHeads",            a.numHeads,            e.numHeads)
        try check("numKVHeads",          a.numKVHeads,          e.numKVHeads)
        try check("numFullKVHeads",      a.numFullKVHeads,      e.numFullKVHeads)
        try check("headDim",             a.headDim,             e.headDim)
        try check("fullHeadDim",         a.fullHeadDim,         e.fullHeadDim)
        try check("vocabSize",           a.vocabSize,           e.vocabSize)
        try check("slidingWindow",       a.slidingWindow,       e.slidingWindow)
        try check("finalLogitSoftcap",   a.finalLogitSoftcap,   e.finalLogitSoftcap)
        try check("ropeTheta",           a.ropeTheta,           e.ropeTheta)
        try check("fullRopeTheta",       a.fullRopeTheta,       e.fullRopeTheta)
        try check("partialRotaryFactor", a.partialRotaryFactor, e.partialRotaryFactor)
        try check("numLayers",           a.numLayers,           e.numLayers)
        try check("numExperts",          a.numExperts,          e.numExperts)
        try check("topKExperts",         a.topKExperts,         e.topKExperts)
        try check("tieWordEmbeddings",   a.tieWordEmbeddings,   e.tieWordEmbeddings)
        try check("attentionKEqV",       a.attentionKEqV,       e.attentionKEqV)
        try check("hiddenActivation",    a.hiddenActivation,    e.hiddenActivation)
        let actualMask = a.fullAttentionLayerMask.map { UInt8($0) }
        try check("fullAttentionLayerMask",
                  actualMask.description,
                  e.fullAttentionLayerMask.description)
    }
}

private extension ManifestFileEntry {
    init(wire: GTurboManifestFileV1) {
        self.init(size: wire.size, sha256: wire.sha256)
    }
}

private extension ManifestArch {
    init(wire: GTurboManifestArchV1) {
        self.init(hiddenSize: wire.hiddenSize,
                  ffnIntermediate: wire.ffnIntermediate,
                  moeIntermediateSize: wire.moeIntermediateSize,
                  numHeads: wire.numHeads,
                  numKVHeads: wire.numKVHeads,
                  numFullKVHeads: wire.numFullKVHeads,
                  headDim: wire.headDim,
                  fullHeadDim: wire.fullHeadDim,
                  vocabSize: wire.vocabSize,
                  slidingWindow: wire.slidingWindow,
                  finalLogitSoftcap: wire.finalLogitSoftcap,
                  ropeTheta: wire.ropeTheta,
                  fullRopeTheta: wire.fullRopeTheta,
                  partialRotaryFactor: wire.partialRotaryFactor,
                  numLayers: wire.numLayers,
                  numExperts: wire.numExperts,
                  topKExperts: wire.topKExperts,
                  tieWordEmbeddings: wire.tieWordEmbeddings,
                  attentionKEqV: wire.attentionKEqV,
                  hiddenActivation: wire.hiddenActivation,
                  fullAttentionLayerMask: wire.fullAttentionLayerMask)
    }
}

private extension ManifestQuantSlot {
    init(wire: GTurboManifestQuantSlotV1) {
        self.init(weightBits: wire.weightBits, scheme: wire.scheme,
                  scaleType: wire.scaleType, biasType: wire.biasType,
                  groupSize: wire.groupSize)
    }
}

private extension ManifestQuant {
    init(wire: GTurboManifestQuantV1) {
        self.init(embedding: ManifestQuantSlot(wire: wire.embedding),
                  attention: ManifestQuantSlot(wire: wire.attention),
                  router: ManifestQuantSlot(wire: wire.router),
                  sharedExpert: ManifestQuantSlot(wire: wire.sharedExpert),
                  routedExpert: ManifestQuantSlot(wire: wire.routedExpert))
    }
}

private extension ManifestVision {
    init(wire: GTurboManifestVisionV1) {
        self.init(hiddenSize: wire.hiddenSize,
                  numLayers: wire.numLayers,
                  numHeads: wire.numHeads,
                  numKVHeads: wire.numKVHeads,
                  headDim: wire.headDim,
                  intermediateSize: wire.intermediateSize,
                  patchSize: wire.patchSize,
                  poolingKernelSize: wire.poolingKernelSize,
                  positionEmbeddingSize: wire.positionEmbeddingSize,
                  ropeTheta: wire.ropeTheta,
                  rmsNormEps: wire.rmsNormEps,
                  hiddenActivation: wire.hiddenActivation,
                  standardize: wire.standardize,
                  maxSoftTokens: wire.maxSoftTokens,
                  weightDType: wire.weightDType,
                  imageTokenID: wire.imageTokenID,
                  boiTokenID: wire.boiTokenID,
                  eoiTokenID: wire.eoiTokenID,
                  weightsPath: wire.weightsPath,
                  tensorCount: wire.tensorCount,
                  payloadBytes: wire.payloadBytes,
                  sourceRepo: wire.sourceRepo,
                  sourceRevision: wire.sourceRevision)
    }
}

private extension ManifestDraft {
    init(wire: GTurboManifestDraftV1) {
        self.init(hiddenSize: wire.hiddenSize,
                  numLayers: wire.numLayers,
                  numHeads: wire.numHeads,
                  numKVHeads: wire.numKVHeads,
                  numFullKVHeads: wire.numFullKVHeads,
                  headDim: wire.headDim,
                  fullHeadDim: wire.fullHeadDim,
                  intermediateSize: wire.intermediateSize,
                  backboneHiddenSize: wire.backboneHiddenSize,
                  vocabSize: wire.vocabSize,
                  slidingWindow: wire.slidingWindow,
                  ropeTheta: wire.ropeTheta,
                  fullRopeTheta: wire.fullRopeTheta,
                  partialRotaryFactor: wire.partialRotaryFactor,
                  rmsNormEps: wire.rmsNormEps,
                  hiddenActivation: wire.hiddenActivation,
                  tieWordEmbeddings: wire.tieWordEmbeddings,
                  attentionKEqV: wire.attentionKEqV,
                  fullAttentionLayerMask: wire.fullAttentionLayerMask,
                  sharedSlidingKVLayer: wire.sharedSlidingKVLayer,
                  sharedFullKVLayer: wire.sharedFullKVLayer,
                  quant: ManifestQuantSlot(wire: wire.quant),
                  weightsPath: wire.weightsPath,
                  tensorCount: wire.tensorCount,
                  payloadBytes: wire.payloadBytes,
                  sourceRepo: wire.sourceRepo,
                  sourceRevision: wire.sourceRevision)
    }
}

private extension Manifest {
    init(wire: GTurboManifestV1) {
        self.init(magic: wire.magic,
                  versionMajor: wire.versionMajor,
                  versionMinor: wire.versionMinor,
                  flags: wire.flags,
                  modelID: wire.modelID,
                  sourceSnapshotHash: wire.sourceSnapshotHash,
                  arch: ManifestArch(wire: wire.arch),
                  quant: wire.quant.map(ManifestQuant.init(wire:)),
                  vision: wire.vision.map(ManifestVision.init(wire:)),
                  draft: wire.draft.map(ManifestDraft.init(wire:)),
                  files: wire.files.mapValues(ManifestFileEntry.init(wire:)),
                  expertsPerLayer: wire.expertsPerLayer,
                  numLayers: wire.numLayers,
                  expertStride: wire.expertStride)
    }
}
