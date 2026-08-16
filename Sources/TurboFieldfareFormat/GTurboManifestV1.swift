import Foundation

package struct GTurboManifestFileV1: Codable, Equatable, Sendable {
    package let size: UInt64
    package let sha256: String

    package init(size: UInt64, sha256: String) {
        self.size = size
        self.sha256 = sha256
    }
}

package struct GTurboManifestArchV1: Codable, Equatable, Sendable {
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

    package init(hiddenSize: Int, ffnIntermediate: Int, moeIntermediateSize: Int,
                 numHeads: Int, numKVHeads: Int, numFullKVHeads: Int,
                 headDim: Int, fullHeadDim: Int, vocabSize: Int,
                 slidingWindow: Int, finalLogitSoftcap: Double,
                 ropeTheta: Double, fullRopeTheta: Double,
                 partialRotaryFactor: Double, numLayers: Int, numExperts: Int,
                 topKExperts: Int, tieWordEmbeddings: Bool, attentionKEqV: Bool,
                 hiddenActivation: String, fullAttentionLayerMask: [Int]) {
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
    }
}

package struct GTurboManifestQuantSlotV1: Codable, Equatable, Sendable {
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

package struct GTurboManifestQuantV1: Codable, Equatable, Sendable {
    package let embedding: GTurboManifestQuantSlotV1
    package let attention: GTurboManifestQuantSlotV1
    package let router: GTurboManifestQuantSlotV1
    package let sharedExpert: GTurboManifestQuantSlotV1
    package let routedExpert: GTurboManifestQuantSlotV1

    package init(embedding: GTurboManifestQuantSlotV1,
                 attention: GTurboManifestQuantSlotV1,
                 router: GTurboManifestQuantSlotV1,
                 sharedExpert: GTurboManifestQuantSlotV1,
                 routedExpert: GTurboManifestQuantSlotV1) {
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
package struct GTurboManifestVisionV1: Codable, Equatable, Sendable {
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

package struct GTurboManifestV1: Codable, Equatable, Sendable {
    package let magic: String
    package let versionMajor: Int
    package let versionMinor: Int
    package let flags: [String: Bool]
    package let modelID: String
    package let sourceSnapshotHash: String?
    package let arch: GTurboManifestArchV1
    package let quant: GTurboManifestQuantV1?
    package let vision: GTurboManifestVisionV1?
    package let files: [String: GTurboManifestFileV1]
    package let expertsPerLayer: Int
    package let numLayers: Int
    package let expertStride: UInt64
    package let bitWidthOverridesHonored: Int?

    package init(magic: String = GTurboFormatV1.magic,
                 versionMajor: Int = GTurboFormatV1.versionMajor,
                 versionMinor: Int = GTurboFormatV1.versionMinor,
                 flags: [String: Bool], modelID: String,
                 sourceSnapshotHash: String?, arch: GTurboManifestArchV1,
                 quant: GTurboManifestQuantV1?,
                 vision: GTurboManifestVisionV1? = nil,
                 files: [String: GTurboManifestFileV1],
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
        self.files = files
        self.expertsPerLayer = expertsPerLayer
        self.numLayers = numLayers
        self.expertStride = expertStride
        self.bitWidthOverridesHonored = bitWidthOverridesHonored
    }
}

package enum GTurboManifestCodec {
    package static func decode(_ data: Data) throws -> GTurboManifestV1 {
        let manifest = try decodeUnchecked(data)
        try validate(manifest)
        return manifest
    }

    package static func decodeUnchecked(_ data: Data) throws -> GTurboManifestV1 {
        let manifest: GTurboManifestV1
        do { manifest = try JSONDecoder().decode(GTurboManifestV1.self, from: data) }
        catch { throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)") }
        return manifest
    }

    package static func encode(_ manifest: GTurboManifestV1) throws -> Data {
        try validate(manifest)
        let encoder = JSONEncoder()
        do {
            let object = try JSONSerialization.jsonObject(with: encoder.encode(manifest))
            return try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        } catch {
            throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)")
        }
    }

    package static func validate(_ manifest: GTurboManifestV1) throws {
        guard manifest.magic == GTurboFormatV1.magic else {
            throw GTurboFormatError.invalid(field: "manifest.magic", reason: "expected GTURBO")
        }
        guard manifest.versionMajor == GTurboFormatV1.versionMajor,
              manifest.versionMinor >= 0 else {
            throw GTurboFormatError.invalid(field: "manifest.version", reason: "unsupported version")
        }
        for flag in manifest.flags.keys where !GTurboFormatV1.knownFlags.contains(flag) {
            throw GTurboFormatError.invalid(field: "manifest.flags.\(flag)", reason: "unknown v1 flag")
        }
        guard !manifest.modelID.isEmpty,
              manifest.numLayers > 0, manifest.expertsPerLayer > 0,
              manifest.expertStride > 0,
              manifest.expertStride % GTurboFormatV1.alignmentBytes == 0 else {
            throw GTurboFormatError.invalid(field: "manifest", reason: "invalid dimensions or stride")
        }
        guard manifest.arch.numLayers == manifest.numLayers,
              manifest.arch.numExperts == manifest.expertsPerLayer else {
            throw GTurboFormatError.invalid(
                field: "manifest.arch", reason: "dimensions disagree with streaming metadata")
        }
        let arch = manifest.arch
        guard arch.hiddenSize > 0, arch.ffnIntermediate > 0,
              arch.moeIntermediateSize > 0, arch.numHeads > 0,
              arch.numKVHeads > 0, arch.numFullKVHeads > 0,
              arch.headDim > 0, arch.fullHeadDim > 0,
              arch.vocabSize > 0, arch.slidingWindow > 0,
              arch.topKExperts > 0, arch.topKExperts <= arch.numExperts,
              arch.finalLogitSoftcap.isFinite,
              arch.ropeTheta.isFinite, arch.ropeTheta > 0,
              arch.fullRopeTheta.isFinite, arch.fullRopeTheta > 0,
              arch.partialRotaryFactor.isFinite,
              arch.partialRotaryFactor >= 0, arch.partialRotaryFactor <= 1,
              !arch.hiddenActivation.isEmpty,
              arch.fullAttentionLayerMask.count == arch.numLayers,
              arch.fullAttentionLayerMask.allSatisfy({ $0 == 0 || $0 == 1 }) else {
            throw GTurboFormatError.invalid(
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
                    throw GTurboFormatError.invalid(
                        field: "manifest.quant.\(name)", reason: "invalid quantization values")
                }
            }
        }
        try validateVisionSection(manifest)
        let reservedFiles: Set<String> = ["manifest.json", "verified-install.json"]
        let filePaths = manifest.files.keys.sorted()
        var canonicalPaths: [String: String] = [:]
        for path in filePaths {
            try GTurboPathValidator.validateRelativePath(path, field: "manifest.files.\(path)")
            let key = GTurboPathValidator.appleFilesystemKey(path)
            guard canonicalPaths.updateValue(path, forKey: key) == nil else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "filesystem-equivalent duplicate path")
            }
            guard key != "tokenizer",
                  !reservedFiles.contains(key),
                  !reservedFiles.contains(where: { key.hasPrefix("\($0)/") }) else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "reserved artifact filename")
            }
            let entry = manifest.files[path]!
            guard entry.sha256.count == 64,
                  entry.sha256.unicodeScalars.allSatisfy({ scalar in
                      ("0"..."9").contains(Character(String(scalar)))
                          || ("a"..."f").contains(Character(String(scalar)))
                          || ("A"..."F").contains(Character(String(scalar)))
                  }) else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path).sha256", reason: "expected 64 hexadecimal characters")
            }
        }
        for (key, path) in canonicalPaths {
            var components = key.split(separator: "/").map(String.init)
            while components.count > 1 {
                _ = components.removeLast()
                let ancestor = components.joined(separator: "/")
                if canonicalPaths[ancestor] != nil {
                    throw GTurboFormatError.invalid(
                        field: "manifest.files.\(path)",
                        reason: "file path collides with a directory prefix")
                }
            }
        }
    }

    /// The vision section and its flag are one fact written twice; a manifest
    /// that carries only one of them would let a reader disagree with the
    /// installer about whether this model can see.
    private static func validateVisionSection(_ manifest: GTurboManifestV1) throws {
        let flagged = manifest.flags["visionTower"] == true
        guard let vision = manifest.vision else {
            guard !flagged else {
                throw GTurboFormatError.invalid(
                    field: "manifest.vision",
                    reason: "flags.visionTower is set but the vision section is absent")
            }
            return
        }
        guard flagged else {
            throw GTurboFormatError.invalid(
                field: "manifest.flags.visionTower",
                reason: "vision section present but the flag is not set")
        }
        guard manifest.versionMinor >= GTurboFormatV1.versionMinorVision else {
            throw GTurboFormatError.invalid(
                field: "manifest.version",
                reason: "vision requires minor >= \(GTurboFormatV1.versionMinorVision)")
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
            throw GTurboFormatError.invalid(
                field: "manifest.vision", reason: "invalid vision values")
        }
        guard vision.weightDType.lowercased() == "bf16" else {
            throw GTurboFormatError.invalid(
                field: "manifest.vision.weightDType", reason: "expected bf16")
        }
        let ids = [vision.imageTokenID, vision.boiTokenID, vision.eoiTokenID]
        guard ids.allSatisfy({ $0 >= 0 }), Set(ids).count == ids.count else {
            throw GTurboFormatError.invalid(
                field: "manifest.vision", reason: "image marker token ids must be distinct")
        }
        try GTurboPathValidator.validateRelativePath(vision.weightsPath,
                                                     field: "manifest.vision.weightsPath")
        guard let entry = manifest.files[vision.weightsPath] else {
            throw GTurboFormatError.invalid(
                field: "manifest.vision.weightsPath",
                reason: "\(vision.weightsPath) is not declared in manifest.files")
        }
        guard entry.size > vision.payloadBytes else {
            // The file is an index page plus the payload, so it is strictly
            // larger than the payload it declares.
            throw GTurboFormatError.invalid(
                field: "manifest.vision.payloadBytes",
                reason: "payload \(vision.payloadBytes) does not fit in \(entry.size) bytes")
        }
    }
}
