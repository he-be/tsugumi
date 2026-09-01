import Foundation

package struct MoEPackManifestFileV1: Codable, Equatable, Sendable {
    package let size: UInt64
    package let sha256: String

    package init(size: UInt64, sha256: String) {
        self.size = size
        self.sha256 = sha256
    }
}

/// Facts about a family whose layers keep a fixed-size recurrent state instead
/// of a KV cache (Qwen3.5-MoE's Gated DeltaNet). Absent for Gemma 4, whose every
/// layer attends. `docs/qwen35moe/01-MODEL.md` §1.
package struct MoEPackManifestLinearAttentionV1: Codable, Equatable, Sendable {
    package let numKeyHeads: Int
    package let numValueHeads: Int
    package let keyHeadDim: Int
    package let valueHeadDim: Int
    package let convKernelDim: Int
    /// How many of `arch.numLayers` are linear-attention layers. The per-layer
    /// truth is `arch.layerKinds`; this is the count a reader can budget from
    /// without walking it.
    package let layerCount: Int

    package init(numKeyHeads: Int, numValueHeads: Int,
                 keyHeadDim: Int, valueHeadDim: Int,
                 convKernelDim: Int, layerCount: Int) {
        self.numKeyHeads = numKeyHeads
        self.numValueHeads = numValueHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
        self.convKernelDim = convKernelDim
        self.layerCount = layerCount
    }
}

package struct MoEPackManifestArchV1: Codable, Equatable, Sendable {
    package let hiddenSize: Int
    package let ffnIntermediate: Int
    package let moeIntermediateSize: Int
    package let numHeads: Int
    package let numKVHeads: Int
    package let numFullKVHeads: Int
    package let headDim: Int
    package let fullHeadDim: Int
    package let vocabSize: Int
    package let slidingWindow: Int
    package let finalLogitSoftcap: Double
    package let ropeTheta: Double
    package let fullRopeTheta: Double
    package let partialRotaryFactor: Double
    package let numLayers: Int
    package let numExperts: Int
    package let topKExperts: Int
    package let tieWordEmbeddings: Bool
    package let attentionKEqV: Bool
    package let hiddenActivation: String
    package let fullAttentionLayerMask: [Int]
    /// Which architecture wrote this model. Absent means Gemma 4: every
    /// manifest written before the Qwen work omits the key, and a reader that
    /// sees no family must keep reading it as Gemma.
    package let family: String?
    /// Per-layer kind: `full_attention`, `sliding_attention` or
    /// `linear_attention`. `fullAttentionLayerMask` stays the compatibility
    /// surface (1 where this reads `full_attention`); this says what the zeros
    /// are, which Gemma and Qwen disagree about.
    package let layerKinds: [String]?
    package let linearAttention: MoEPackManifestLinearAttentionV1?

    package init(hiddenSize: Int, ffnIntermediate: Int, moeIntermediateSize: Int,
                 numHeads: Int, numKVHeads: Int, numFullKVHeads: Int,
                 headDim: Int, fullHeadDim: Int, vocabSize: Int,
                 slidingWindow: Int, finalLogitSoftcap: Double,
                 ropeTheta: Double, fullRopeTheta: Double,
                 partialRotaryFactor: Double, numLayers: Int, numExperts: Int,
                 topKExperts: Int, tieWordEmbeddings: Bool, attentionKEqV: Bool,
                 hiddenActivation: String, fullAttentionLayerMask: [Int],
                 family: String? = nil,
                 layerKinds: [String]? = nil,
                 linearAttention: MoEPackManifestLinearAttentionV1? = nil) {
        self.hiddenSize = hiddenSize
        self.ffnIntermediate = ffnIntermediate
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.hiddenActivation = hiddenActivation
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.family = family
        self.layerKinds = layerKinds
        self.linearAttention = linearAttention
    }
}

package struct MoEPackManifestQuantSlotV1: Codable, Equatable, Sendable {
    package let weightBits: Int
    package let scheme: String
    package let scaleType: String
    package let biasType: String
    package let groupSize: Int

    package init(weightBits: Int, scheme: String, scaleType: String,
                 biasType: String, groupSize: Int) {
        self.weightBits = weightBits
        self.scheme = scheme
        self.scaleType = scaleType
        self.biasType = biasType
        self.groupSize = groupSize
    }
}

package struct MoEPackManifestQuantV1: Codable, Equatable, Sendable {
    package let embedding: MoEPackManifestQuantSlotV1
    package let attention: MoEPackManifestQuantSlotV1
    package let router: MoEPackManifestQuantSlotV1
    package let sharedExpert: MoEPackManifestQuantSlotV1
    package let routedExpert: MoEPackManifestQuantSlotV1

    package init(embedding: MoEPackManifestQuantSlotV1,
                 attention: MoEPackManifestQuantSlotV1,
                 router: MoEPackManifestQuantSlotV1,
                 sharedExpert: MoEPackManifestQuantSlotV1,
                 routedExpert: MoEPackManifestQuantSlotV1) {
        self.embedding = embedding
        self.attention = attention
        self.router = router
        self.sharedExpert = sharedExpert
        self.routedExpert = routedExpert
    }
}

/// Vision tower description. Optional and additive: a text-only manifest omits
/// the whole section, so its bytes are unchanged by this schema. A manifest
/// that carries it also carries `flags.visionTower`, which an older runtime
/// rejects outright rather than silently ignoring the images.
package struct MoEPackManifestVisionV1: Codable, Equatable, Sendable {
    package let hiddenSize: Int
    package let numLayers: Int
    package let numHeads: Int
    package let numKVHeads: Int
    package let headDim: Int
    package let intermediateSize: Int
    package let patchSize: Int
    package let poolingKernelSize: Int
    package let positionEmbeddingSize: Int
    package let ropeTheta: Double
    package let rmsNormEps: Double
    package let hiddenActivation: String
    package let standardize: Bool
    package let maxSoftTokens: Int
    package let weightDType: String
    package let imageTokenID: Int
    package let boiTokenID: Int
    package let eoiTokenID: Int
    /// Relative path of the tower weights inside the model directory. Also
    /// present in `files`, which is what carries its size and digest.
    package let weightsPath: String
    package let tensorCount: Int
    package let payloadBytes: UInt64
    /// The tower's own provenance. `sourceSnapshotHash` describes the text
    /// checkpoint only, and the two come from different repositories.
    package let sourceRepo: String
    package let sourceRevision: String

    package init(hiddenSize: Int, numLayers: Int, numHeads: Int, numKVHeads: Int,
                 headDim: Int, intermediateSize: Int, patchSize: Int,
                 poolingKernelSize: Int, positionEmbeddingSize: Int,
                 ropeTheta: Double, rmsNormEps: Double,
                 hiddenActivation: String, standardize: Bool,
                 maxSoftTokens: Int, weightDType: String,
                 imageTokenID: Int, boiTokenID: Int, eoiTokenID: Int,
                 weightsPath: String, tensorCount: Int, payloadBytes: UInt64,
                 sourceRepo: String, sourceRevision: String) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.patchSize = patchSize
        self.poolingKernelSize = poolingKernelSize
        self.positionEmbeddingSize = positionEmbeddingSize
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
        self.hiddenActivation = hiddenActivation
        self.standardize = standardize
        self.maxSoftTokens = maxSoftTokens
        self.weightDType = weightDType
        self.imageTokenID = imageTokenID
        self.boiTokenID = boiTokenID
        self.eoiTokenID = eoiTokenID
        self.weightsPath = weightsPath
        self.tensorCount = tensorCount
        self.payloadBytes = payloadBytes
        self.sourceRepo = sourceRepo
        self.sourceRevision = sourceRevision
    }
}

/// MTP drafter description. Optional and additive in the same way the vision
/// section is: a manifest without a drafter omits it entirely, and one that
/// carries it also carries `flags.mtpDraft` so a runtime that predates
/// speculation refuses the model instead of ignoring the extra file.
///
/// Most of these fields are not free parameters — the drafter has no K/V
/// projections of its own and reads the target's, so its head geometry, window
/// and RoPE constants have to equal the target's. `validateDraftSection` checks
/// that against `arch` rather than trusting the installer that wrote it.
package struct MoEPackManifestDraftV1: Codable, Equatable, Sendable {
    package let hiddenSize: Int
    package let numLayers: Int
    package let numHeads: Int
    package let numKVHeads: Int
    package let numFullKVHeads: Int
    package let headDim: Int
    package let fullHeadDim: Int
    package let intermediateSize: Int
    /// Hidden size of the target this drafter was trained against. Equals
    /// `arch.hiddenSize`; the `pre_projection` input is twice this.
    package let backboneHiddenSize: Int
    package let vocabSize: Int
    package let slidingWindow: Int
    package let ropeTheta: Double
    package let fullRopeTheta: Double
    package let partialRotaryFactor: Double
    package let rmsNormEps: Double
    package let hiddenActivation: String
    package let tieWordEmbeddings: Bool
    package let attentionKEqV: Bool
    /// One entry per drafter layer: 1 = full attention, 0 = sliding.
    package let fullAttentionLayerMask: [Int]
    /// Target layers whose K/V the drafter attends to. Its sliding layers read
    /// the target's last sliding layer, its full layer the last full one.
    package let sharedSlidingKVLayer: Int
    package let sharedFullKVLayer: Int
    /// Quantization of the drafter's own weights. It is converted separately
    /// from the text checkpoint, so this need not match `quant`.
    package let quant: MoEPackManifestQuantSlotV1
    /// Relative path of the drafter weights inside the model directory. Also
    /// present in `files`, which is what carries its size and digest.
    package let weightsPath: String
    package let tensorCount: Int
    package let payloadBytes: UInt64
    /// The drafter's own provenance; it comes from a different repository than
    /// the text weights, so `sourceSnapshotHash` does not describe it.
    package let sourceRepo: String
    package let sourceRevision: String

    package init(hiddenSize: Int, numLayers: Int, numHeads: Int, numKVHeads: Int,
                 numFullKVHeads: Int, headDim: Int, fullHeadDim: Int,
                 intermediateSize: Int, backboneHiddenSize: Int, vocabSize: Int,
                 slidingWindow: Int, ropeTheta: Double, fullRopeTheta: Double,
                 partialRotaryFactor: Double, rmsNormEps: Double,
                 hiddenActivation: String, tieWordEmbeddings: Bool,
                 attentionKEqV: Bool, fullAttentionLayerMask: [Int],
                 sharedSlidingKVLayer: Int, sharedFullKVLayer: Int,
                 quant: MoEPackManifestQuantSlotV1, weightsPath: String,
                 tensorCount: Int, payloadBytes: UInt64,
                 sourceRepo: String, sourceRevision: String) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.intermediateSize = intermediateSize
        self.backboneHiddenSize = backboneHiddenSize
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.rmsNormEps = rmsNormEps
        self.hiddenActivation = hiddenActivation
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.sharedSlidingKVLayer = sharedSlidingKVLayer
        self.sharedFullKVLayer = sharedFullKVLayer
        self.quant = quant
        self.weightsPath = weightsPath
        self.tensorCount = tensorCount
        self.payloadBytes = payloadBytes
        self.sourceRepo = sourceRepo
        self.sourceRevision = sourceRevision
    }
}

package struct MoEPackManifestV1: Codable, Equatable, Sendable {
    package let magic: String
    package let versionMajor: Int
    package let versionMinor: Int
    package let flags: [String: Bool]
    package let modelID: String
    package let sourceSnapshotHash: String?
    package let arch: MoEPackManifestArchV1
    package let quant: MoEPackManifestQuantV1?
    package let vision: MoEPackManifestVisionV1?
    package let draft: MoEPackManifestDraftV1?
    package let files: [String: MoEPackManifestFileV1]
    package let expertsPerLayer: Int
    package let numLayers: Int
    package let expertStride: UInt64
    package let bitWidthOverridesHonored: Int?

    package init(magic: String = MoEPackFormatV1.magic,
                 versionMajor: Int = MoEPackFormatV1.versionMajor,
                 versionMinor: Int = MoEPackFormatV1.versionMinor,
                 flags: [String: Bool], modelID: String,
                 sourceSnapshotHash: String?, arch: MoEPackManifestArchV1,
                 quant: MoEPackManifestQuantV1?,
                 vision: MoEPackManifestVisionV1? = nil,
                 draft: MoEPackManifestDraftV1? = nil,
                 files: [String: MoEPackManifestFileV1],
                 expertsPerLayer: Int, numLayers: Int, expertStride: UInt64,
                 bitWidthOverridesHonored: Int?) {
        self.magic = magic
        self.versionMajor = versionMajor
        self.versionMinor = versionMinor
        self.flags = flags
        self.modelID = modelID
        self.sourceSnapshotHash = sourceSnapshotHash
        self.arch = arch
        self.quant = quant
        self.vision = vision
        self.draft = draft
        self.files = files
        self.expertsPerLayer = expertsPerLayer
        self.numLayers = numLayers
        self.expertStride = expertStride
        self.bitWidthOverridesHonored = bitWidthOverridesHonored
    }
}

package enum MoEPackManifestCodec {
    package static func decode(_ data: Data) throws -> MoEPackManifestV1 {
        let manifest = try decodeUnchecked(data)
        try validate(manifest)
        return manifest
    }

    package static func decodeUnchecked(_ data: Data) throws -> MoEPackManifestV1 {
        let manifest: MoEPackManifestV1
        do { manifest = try JSONDecoder().decode(MoEPackManifestV1.self, from: data) }
        catch { throw MoEPackFormatError.invalid(field: "manifest.json", reason: "\(error)") }
        return manifest
    }

    package static func encode(_ manifest: MoEPackManifestV1) throws -> Data {
        try validate(manifest)
        let encoder = JSONEncoder()
        do {
            let object = try JSONSerialization.jsonObject(with: encoder.encode(manifest))
            return try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        } catch {
            throw MoEPackFormatError.invalid(field: "manifest.json", reason: "\(error)")
        }
    }

    package static func validate(_ manifest: MoEPackManifestV1) throws {
        guard manifest.magic == MoEPackFormatV1.magic
                || manifest.magic == MoEPackFormatV1.legacyMagic else {
            throw MoEPackFormatError.invalid(
                field: "manifest.magic",
                reason: "expected \(MoEPackFormatV1.magic) or \(MoEPackFormatV1.legacyMagic)")
        }
        guard manifest.versionMajor == MoEPackFormatV1.versionMajor,
              manifest.versionMinor >= 0 else {
            throw MoEPackFormatError.invalid(field: "manifest.version", reason: "unsupported version")
        }
        for flag in manifest.flags.keys where !MoEPackFormatV1.knownFlags.contains(flag) {
            throw MoEPackFormatError.invalid(field: "manifest.flags.\(flag)", reason: "unknown v1 flag")
        }
        guard !manifest.modelID.isEmpty,
              manifest.numLayers > 0, manifest.expertsPerLayer > 0,
              manifest.expertStride > 0,
              manifest.expertStride % MoEPackFormatV1.alignmentBytes == 0 else {
            throw MoEPackFormatError.invalid(field: "manifest", reason: "invalid dimensions or stride")
        }
        guard manifest.arch.numLayers == manifest.numLayers,
              manifest.arch.numExperts == manifest.expertsPerLayer else {
            throw MoEPackFormatError.invalid(
                field: "manifest.arch", reason: "dimensions disagree with streaming metadata")
        }
        let arch = manifest.arch
        guard arch.hiddenSize > 0, arch.ffnIntermediate > 0,
              arch.moeIntermediateSize > 0, arch.numHeads > 0,
              arch.numKVHeads > 0, arch.numFullKVHeads > 0,
              arch.headDim > 0, arch.fullHeadDim > 0,
              arch.vocabSize > 0,
              // A family whose non-full layers are recurrent has no window at
              // all; every attending-everywhere family still must name one.
              arch.slidingWindow > 0 || arch.linearAttention != nil,
              arch.slidingWindow >= 0,
              arch.topKExperts > 0, arch.topKExperts <= arch.numExperts,
              arch.finalLogitSoftcap.isFinite,
              arch.ropeTheta.isFinite, arch.ropeTheta > 0,
              arch.fullRopeTheta.isFinite, arch.fullRopeTheta > 0,
              arch.partialRotaryFactor.isFinite,
              arch.partialRotaryFactor >= 0, arch.partialRotaryFactor <= 1,
              !arch.hiddenActivation.isEmpty,
              arch.fullAttentionLayerMask.count == arch.numLayers,
              arch.fullAttentionLayerMask.allSatisfy({ $0 == 0 || $0 == 1 }) else {
            throw MoEPackFormatError.invalid(
                field: "manifest.arch", reason: "invalid architecture values")
        }
        if let quant = manifest.quant {
            for (name, slot) in [
                ("embedding", quant.embedding),
                ("attention", quant.attention),
                ("router", quant.router),
                ("sharedExpert", quant.sharedExpert),
                ("routedExpert", quant.routedExpert),
            ] {
                guard slot.weightBits > 0, slot.weightBits <= 32,
                      slot.groupSize > 0,
                      !slot.scheme.isEmpty, !slot.scaleType.isEmpty,
                      !slot.biasType.isEmpty else {
                    throw MoEPackFormatError.invalid(
                        field: "manifest.quant.\(name)", reason: "invalid quantization values")
                }
            }
        }
        try validateLinearAttentionSection(manifest)
        try validateVisionSection(manifest)
        try validateDraftSection(manifest)
        let reservedFiles: Set<String> = ["manifest.json", "verified-install.json"]
        let filePaths = manifest.files.keys.sorted()
        var canonicalPaths: [String: String] = [:]
        for path in filePaths {
            try MoEPackPathValidator.validateRelativePath(path, field: "manifest.files.\(path)")
            let key = MoEPackPathValidator.appleFilesystemKey(path)
            guard canonicalPaths.updateValue(path, forKey: key) == nil else {
                throw MoEPackFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "filesystem-equivalent duplicate path")
            }
            guard key != "tokenizer",
                  !reservedFiles.contains(key),
                  !reservedFiles.contains(where: { key.hasPrefix("\($0)/") }) else {
                throw MoEPackFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "reserved artifact filename")
            }
            let entry = manifest.files[path]!
            guard entry.sha256.count == 64,
                  entry.sha256.unicodeScalars.allSatisfy({ scalar in
                      ("0"..."9").contains(Character(String(scalar)))
                          || ("a"..."f").contains(Character(String(scalar)))
                          || ("A"..."F").contains(Character(String(scalar)))
                  }) else {
                throw MoEPackFormatError.invalid(
                    field: "manifest.files.\(path).sha256", reason: "expected 64 hexadecimal characters")
            }
        }
        for (key, path) in canonicalPaths {
            var components = key.split(separator: "/").map(String.init)
            while components.count > 1 {
                _ = components.removeLast()
                let ancestor = components.joined(separator: "/")
                if canonicalPaths[ancestor] != nil {
                    throw MoEPackFormatError.invalid(
                        field: "manifest.files.\(path)",
                        reason: "file path collides with a directory prefix")
                }
            }
        }
    }

    /// Layer kinds a v1 manifest may name. `sliding_attention` and
    /// `full_attention` are Gemma's; `linear_attention` is Qwen3.5-MoE's
    /// recurrent layer, which holds a fixed-size state instead of a KV cache.
    package static let knownLayerKinds: Set<String> = [
        "full_attention", "sliding_attention", "linear_attention",
    ]

    /// The linear-attention section and its flag are one fact written twice,
    /// for the same reason the tower's are: a runtime that predates recurrent
    /// layers must reject the model rather than read its zeros as sliding
    /// windows and quietly produce nonsense.
    private static func validateLinearAttentionSection(_ manifest: MoEPackManifestV1) throws {
        let arch = manifest.arch
        if let family = arch.family, family.isEmpty {
            throw MoEPackFormatError.invalid(
                field: "manifest.arch.family", reason: "empty family name")
        }
        if let kinds = arch.layerKinds {
            guard kinds.count == arch.numLayers else {
                throw MoEPackFormatError.invalid(
                    field: "manifest.arch.layerKinds",
                    reason: "expected \(arch.numLayers) entries, got \(kinds.count)")
            }
            guard kinds.allSatisfy({ knownLayerKinds.contains($0) }) else {
                throw MoEPackFormatError.invalid(
                    field: "manifest.arch.layerKinds", reason: "unknown v1 layer kind")
            }
            // The mask is the compatibility surface: it must agree with the
            // list, or a reader that only knows the mask sees a different model.
            let maskFromKinds = kinds.map { $0 == "full_attention" ? 1 : 0 }
            guard maskFromKinds == arch.fullAttentionLayerMask else {
                throw MoEPackFormatError.invalid(
                    field: "manifest.arch.layerKinds",
                    reason: "disagrees with fullAttentionLayerMask")
            }
        }
        let flagged = manifest.flags["linearAttention"] == true
        guard let linear = arch.linearAttention else {
            guard !flagged else {
                throw MoEPackFormatError.invalid(
                    field: "manifest.arch.linearAttention",
                    reason: "flags.linearAttention is set but the section is absent")
            }
            return
        }
        guard flagged else {
            throw MoEPackFormatError.invalid(
                field: "manifest.flags.linearAttention",
                reason: "linearAttention section present but the flag is not set")
        }
        guard manifest.versionMinor >= MoEPackFormatV1.versionMinorLinearAttention else {
            throw MoEPackFormatError.invalid(
                field: "manifest.version",
                reason: "linear attention requires minor >= "
                    + "\(MoEPackFormatV1.versionMinorLinearAttention)")
        }
        guard let kinds = arch.layerKinds else {
            throw MoEPackFormatError.invalid(
                field: "manifest.arch.layerKinds",
                reason: "a linear-attention model must say which layers are which")
        }
        guard linear.numKeyHeads > 0, linear.numValueHeads > 0,
              linear.keyHeadDim > 0, linear.valueHeadDim > 0,
              linear.convKernelDim > 0,
              linear.layerCount > 0, linear.layerCount <= arch.numLayers else {
            throw MoEPackFormatError.invalid(
                field: "manifest.arch.linearAttention", reason: "invalid linear-attention values")
        }
        let counted = kinds.filter { $0 == "linear_attention" }.count
        guard counted == linear.layerCount else {
            throw MoEPackFormatError.invalid(
                field: "manifest.arch.linearAttention.layerCount",
                reason: "layerKinds has \(counted) linear layers, section says \(linear.layerCount)")
        }
    }

    /// The vision section and its flag are one fact written twice; a manifest
    /// that carries only one of them would let a reader disagree with the
    /// installer about whether this model can see.
    private static func validateVisionSection(_ manifest: MoEPackManifestV1) throws {
        let flagged = manifest.flags["visionTower"] == true
        guard let vision = manifest.vision else {
            guard !flagged else {
                throw MoEPackFormatError.invalid(
                    field: "manifest.vision",
                    reason: "flags.visionTower is set but the vision section is absent")
            }
            return
        }
        guard flagged else {
            throw MoEPackFormatError.invalid(
                field: "manifest.flags.visionTower",
                reason: "vision section present but the flag is not set")
        }
        guard manifest.versionMinor >= MoEPackFormatV1.versionMinorVision else {
            throw MoEPackFormatError.invalid(
                field: "manifest.version",
                reason: "vision requires minor >= \(MoEPackFormatV1.versionMinorVision)")
        }
        guard vision.hiddenSize > 0, vision.numLayers > 0,
              vision.numHeads > 0, vision.numKVHeads > 0,
              vision.headDim > 0, vision.intermediateSize > 0,
              vision.patchSize > 0, vision.poolingKernelSize > 0,
              vision.positionEmbeddingSize > 0,
              vision.maxSoftTokens > 0, vision.tensorCount > 0,
              vision.payloadBytes > 0,
              vision.numHeads * vision.headDim == vision.hiddenSize,
              vision.ropeTheta.isFinite, vision.ropeTheta > 0,
              vision.rmsNormEps.isFinite, vision.rmsNormEps > 0,
              !vision.hiddenActivation.isEmpty,
              !vision.sourceRepo.isEmpty, !vision.sourceRevision.isEmpty else {
            throw MoEPackFormatError.invalid(
                field: "manifest.vision", reason: "invalid vision values")
        }
        guard vision.weightDType.lowercased() == "bf16" else {
            throw MoEPackFormatError.invalid(
                field: "manifest.vision.weightDType", reason: "expected bf16")
        }
        let ids = [vision.imageTokenID, vision.boiTokenID, vision.eoiTokenID]
        guard ids.allSatisfy({ $0 >= 0 }), Set(ids).count == ids.count else {
            throw MoEPackFormatError.invalid(
                field: "manifest.vision", reason: "image marker token ids must be distinct")
        }
        try MoEPackPathValidator.validateRelativePath(vision.weightsPath,
                                                     field: "manifest.vision.weightsPath")
        guard let entry = manifest.files[vision.weightsPath] else {
            throw MoEPackFormatError.invalid(
                field: "manifest.vision.weightsPath",
                reason: "\(vision.weightsPath) is not declared in manifest.files")
        }
        guard entry.size > vision.payloadBytes else {
            // The file is an index page plus the payload, so it is strictly
            // larger than the payload it declares.
            throw MoEPackFormatError.invalid(
                field: "manifest.vision.payloadBytes",
                reason: "payload \(vision.payloadBytes) does not fit in \(entry.size) bytes")
        }
    }

    /// The drafter section and its flag are one fact written twice, exactly as
    /// the vision pair is. Beyond that pairing, the drafter borrows the target's
    /// K/V instead of computing its own, so a mismatch in head geometry, window
    /// or RoPE constants is not a slower model but a wrong one — those fields
    /// are checked against `arch` here rather than at first use.
    private static func validateDraftSection(_ manifest: MoEPackManifestV1) throws {
        let flagged = manifest.flags["mtpDraft"] == true
        guard let draft = manifest.draft else {
            guard !flagged else {
                throw MoEPackFormatError.invalid(
                    field: "manifest.draft",
                    reason: "flags.mtpDraft is set but the draft section is absent")
            }
            return
        }
        guard flagged else {
            throw MoEPackFormatError.invalid(
                field: "manifest.flags.mtpDraft",
                reason: "draft section present but the flag is not set")
        }
        guard manifest.versionMinor >= MoEPackFormatV1.versionMinorDraft else {
            throw MoEPackFormatError.invalid(
                field: "manifest.version",
                reason: "draft requires minor >= \(MoEPackFormatV1.versionMinorDraft)")
        }
        guard draft.hiddenSize > 0, draft.numLayers > 0, draft.numHeads > 0,
              draft.intermediateSize > 0, draft.tensorCount > 0,
              draft.payloadBytes > 0,
              draft.rmsNormEps.isFinite, draft.rmsNormEps > 0,
              !draft.hiddenActivation.isEmpty,
              !draft.sourceRepo.isEmpty, !draft.sourceRevision.isEmpty,
              draft.fullAttentionLayerMask.count == draft.numLayers,
              draft.fullAttentionLayerMask.allSatisfy({ $0 == 0 || $0 == 1 }) else {
            throw MoEPackFormatError.invalid(
                field: "manifest.draft", reason: "invalid draft values")
        }
        let arch = manifest.arch
        // The drafter reads the target's K/V, so these are not independent
        // settings: they are the target's, restated.
        let shared: [(String, Bool)] = [
            ("backboneHiddenSize", draft.backboneHiddenSize == arch.hiddenSize),
            ("vocabSize", draft.vocabSize == arch.vocabSize),
            ("slidingWindow", draft.slidingWindow == arch.slidingWindow),
            ("headDim", draft.headDim == arch.headDim),
            ("fullHeadDim", draft.fullHeadDim == arch.fullHeadDim),
            ("numKVHeads", draft.numKVHeads == arch.numKVHeads),
            ("numFullKVHeads", draft.numFullKVHeads == arch.numFullKVHeads),
            ("ropeTheta", draft.ropeTheta == arch.ropeTheta),
            ("fullRopeTheta", draft.fullRopeTheta == arch.fullRopeTheta),
            ("partialRotaryFactor",
             draft.partialRotaryFactor == arch.partialRotaryFactor),
            ("attentionKEqV", draft.attentionKEqV == arch.attentionKEqV),
        ]
        for (field, agrees) in shared where !agrees {
            throw MoEPackFormatError.invalid(
                field: "manifest.draft.\(field)",
                reason: "drafter shares the target's K/V but disagrees with manifest.arch.\(field)")
        }
        guard draft.tieWordEmbeddings else {
            throw MoEPackFormatError.invalid(
                field: "manifest.draft.tieWordEmbeddings",
                reason: "the drafter's LM head is tied to its embedding table")
        }
        guard draft.sharedSlidingKVLayer >= 0,
              draft.sharedSlidingKVLayer < arch.numLayers,
              draft.sharedFullKVLayer >= 0,
              draft.sharedFullKVLayer < arch.numLayers,
              arch.fullAttentionLayerMask[draft.sharedSlidingKVLayer] == 0,
              arch.fullAttentionLayerMask[draft.sharedFullKVLayer] == 1 else {
            throw MoEPackFormatError.invalid(
                field: "manifest.draft.sharedFullKVLayer",
                reason: "shared K/V layers must name a sliding and a full target layer")
        }
        guard draft.quant.weightBits > 0, draft.quant.weightBits <= 32,
              draft.quant.groupSize > 0,
              !draft.quant.scheme.isEmpty, !draft.quant.scaleType.isEmpty,
              !draft.quant.biasType.isEmpty else {
            throw MoEPackFormatError.invalid(
                field: "manifest.draft.quant", reason: "invalid quantization values")
        }
        try MoEPackPathValidator.validateRelativePath(draft.weightsPath,
                                                     field: "manifest.draft.weightsPath")
        guard let entry = manifest.files[draft.weightsPath] else {
            throw MoEPackFormatError.invalid(
                field: "manifest.draft.weightsPath",
                reason: "\(draft.weightsPath) is not declared in manifest.files")
        }
        guard entry.size > draft.payloadBytes else {
            throw MoEPackFormatError.invalid(
                field: "manifest.draft.payloadBytes",
                reason: "payload \(draft.payloadBytes) does not fit in \(entry.size) bytes")
        }
    }
}
