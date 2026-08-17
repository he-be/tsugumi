import Foundation
import TurboFieldfareFormat

/// JSON encoders for `manifest.json` and `packed_experts/layout.json`. The
/// files are small (kilobytes), so we use Foundation's `JSONSerialization`
/// rather than streaming.
enum GTurboJSON {

    static let magic = GTurboFormatV1.magic
    static let versionMajor = GTurboFormatV1.versionMajor
    static let versionMinor = GTurboFormatV1.versionMinor

    struct FileEntry {
        let size: UInt64
        let sha256: String
    }

    struct QuantBitWidths {
        var embedding: Int
        var attention: Int
        var router: Int
        var sharedExpert: Int
        var routedExpert: Int
    }

    static func encodeManifest(plan: RepackPlan,
                                      modelID: String,
                                      sourceSnapshotHash: String,
                                      files: [(relativePath: String, info: FileEntry)],
                                      expertsPerLayer: Int,
                                      numLayers: Int,
                                      expertStride: UInt64,
                                      bitWidths: QuantBitWidths) throws -> Data {
        let arch = plan.arch
        let bitWidthsByQuantSlot = [
            "embedding": bitWidths.embedding,
            "attention": bitWidths.attention,
            "router": bitWidths.router,
            "sharedExpert": bitWidths.sharedExpert,
            "routedExpert": bitWidths.routedExpert,
        ]
        let wireArch = GTurboManifestArchV1(
            hiddenSize: arch.hiddenSize,
            ffnIntermediate: arch.intermediateSize,
            moeIntermediateSize: arch.moeIntermediateSize,
            numHeads: arch.numHeads,
            numKVHeads: arch.numKVHeads,
            numFullKVHeads: arch.numFullKVHeads,
            headDim: arch.headDim,
            fullHeadDim: arch.fullHeadDim,
            vocabSize: arch.vocabSize,
            slidingWindow: arch.slidingWindow,
            finalLogitSoftcap: arch.finalLogitSoftcap,
            ropeTheta: arch.ropeTheta,
            fullRopeTheta: arch.fullRopeTheta,
            partialRotaryFactor: arch.partialRotaryFactor,
            numLayers: arch.numLayers,
            numExperts: arch.numExperts,
            topKExperts: arch.topKExperts,
            tieWordEmbeddings: arch.tieWordEmbeddings,
            attentionKEqV: arch.attentionKEqV,
            hiddenActivation: arch.hiddenActivation,
            fullAttentionLayerMask: arch.fullAttentionLayerMask.map(Int.init))
        func slot(_ name: String) throws -> GTurboManifestQuantSlotV1 {
            guard let weightBits = bitWidthsByQuantSlot[name] else {
                throw RepackError.configurationInvalid(
                    detail: "missing manifest quant slot bit width for \(name)")
            }
            guard weightBits != 16 else {
                // Unquantized BF16 slot (the QAT router): no affine companions,
                // so the scheme and the companion types have to say "none"
                // rather than describe scales that were never written. The
                // group size still carries the model's base value; nothing
                // reads it for this slot.
                return GTurboManifestQuantSlotV1(
                    weightBits: 16,
                    scheme: "bf16",
                    scaleType: "none",
                    biasType: "none",
                    groupSize: plan.baseGroupSize)
            }
            return GTurboManifestQuantSlotV1(
                weightBits: weightBits,
                scheme: plan.baseMode,
                scaleType: "BF16",
                biasType: "BF16",
                groupSize: plan.baseGroupSize)
        }
        let quant = GTurboManifestQuantV1(
            embedding: try slot("embedding"),
            attention: try slot("attention"),
            router: try slot("router"),
            sharedExpert: try slot("sharedExpert"),
            routedExpert: try slot("routedExpert"))
        var wireFiles: [String: GTurboManifestFileV1] = [:]
        wireFiles.reserveCapacity(files.count)
        for file in files {
            guard wireFiles.updateValue(
                GTurboManifestFileV1(size: file.info.size, sha256: file.info.sha256),
                forKey: file.relativePath) == nil else {
                throw RepackError.configurationInvalid(
                    detail: "duplicate manifest file entry \(file.relativePath)")
            }
        }
        var flags = [
            "streamingPresent": true,
            "turboQuantKV": false,
            "aneSharedExpert": false,
        ]
        var wireVision: GTurboManifestVisionV1?
        if let vision = plan.vision {
            // The flag is the compatibility gate: a runtime that predates vision
            // rejects the whole model rather than quietly ignoring images.
            flags["visionTower"] = true
            let config = vision.source.config
            wireVision = GTurboManifestVisionV1(
                hiddenSize: config.hiddenSize,
                numLayers: config.numLayers,
                numHeads: config.numHeads,
                numKVHeads: config.numKVHeads,
                headDim: config.headDim,
                intermediateSize: config.intermediateSize,
                patchSize: config.patchSize,
                poolingKernelSize: config.poolingKernelSize,
                positionEmbeddingSize: config.positionEmbeddingSize,
                ropeTheta: config.ropeTheta,
                rmsNormEps: config.rmsNormEps,
                hiddenActivation: config.hiddenActivation,
                standardize: config.standardize,
                maxSoftTokens: config.maxSoftTokens,
                weightDType: "bf16",
                imageTokenID: config.imageTokenID,
                boiTokenID: config.boiTokenID,
                eoiTokenID: config.eoiTokenID,
                weightsPath: GTurboFormatV1.visionWeightsPath,
                tensorCount: vision.tensorCount,
                payloadBytes: vision.payloadBytes,
                sourceRepo: vision.source.repoID,
                sourceRevision: vision.source.revision)
        }
        var wireDraft: GTurboManifestDraftV1?
        if let draft = plan.draft {
            // Same compatibility gate as the tower's: a runtime that predates
            // speculation rejects the model rather than ignoring the drafter.
            flags["mtpDraft"] = true
            wireDraft = try draftSection(
                draft, fullAttentionLayerMask: wireArch.fullAttentionLayerMask)
        }
        var versionMinor = GTurboFormatV1.versionMinor
        if wireVision != nil {
            versionMinor = max(versionMinor, GTurboFormatV1.versionMinorVision)
        }
        if wireDraft != nil {
            versionMinor = max(versionMinor, GTurboFormatV1.versionMinorDraft)
        }
        return try GTurboManifestCodec.encode(GTurboManifestV1(
            versionMinor: versionMinor,
            flags: flags,
            modelID: modelID,
            sourceSnapshotHash: sourceSnapshotHash,
            arch: wireArch,
            quant: quant,
            vision: wireVision,
            draft: wireDraft,
            files: wireFiles,
            expertsPerLayer: expertsPerLayer,
            numLayers: numLayers,
            expertStride: expertStride,
            bitWidthOverridesHonored: plan.bitsOverrideCount))
    }

    /// The manifest's `draft` section for a planned drafter. Shared by the
    /// install and the append paths so both write the same bytes.
    ///
    /// The drafter has no K/V of its own: its sliding layers read the target's
    /// last sliding layer and its full layer the target's last full one
    /// (`docs/mtp/01-CHECKPOINT.md` §2). Those two indices are derived from the
    /// target's own layer mask here rather than pinned, so a target with a
    /// different layer pattern names its own layers.
    static func draftSection(_ plan: DraftFilePlan,
                             fullAttentionLayerMask: [Int])
        throws -> GTurboManifestDraftV1 {
        guard let sharedFull = fullAttentionLayerMask.lastIndex(of: 1),
              let sharedSliding = fullAttentionLayerMask.lastIndex(of: 0) else {
            throw RepackError.configurationInvalid(detail: """
                the drafter shares K/V with the target's last sliding and last full \
                attention layer, and this target has no layer of one of those kinds
                """)
        }
        let config = plan.source.config
        return GTurboManifestDraftV1(
            hiddenSize: config.hiddenSize,
            numLayers: config.numLayers,
            numHeads: config.numHeads,
            numKVHeads: config.numKVHeads,
            numFullKVHeads: config.numFullKVHeads,
            headDim: config.headDim,
            fullHeadDim: config.fullHeadDim,
            intermediateSize: config.intermediateSize,
            backboneHiddenSize: config.backboneHiddenSize,
            vocabSize: config.vocabSize,
            slidingWindow: config.slidingWindow,
            ropeTheta: config.ropeTheta,
            fullRopeTheta: config.fullRopeTheta,
            partialRotaryFactor: config.partialRotaryFactor,
            rmsNormEps: config.rmsNormEps,
            hiddenActivation: config.hiddenActivation,
            tieWordEmbeddings: config.tieWordEmbeddings,
            attentionKEqV: config.attentionKEqV,
            fullAttentionLayerMask: config.fullAttentionLayerMask,
            sharedSlidingKVLayer: sharedSliding,
            sharedFullKVLayer: sharedFull,
            quant: GTurboManifestQuantSlotV1(weightBits: config.quantBits,
                                             scheme: config.quantMode,
                                             scaleType: "BF16",
                                             biasType: "BF16",
                                             groupSize: config.quantGroupSize),
            weightsPath: GTurboFormatV1.draftWeightsPath,
            tensorCount: plan.tensorCount,
            payloadBytes: plan.payloadBytes,
            sourceRepo: plan.source.repoID,
            sourceRevision: plan.source.revision)
    }

    static func encodeLayout(plan: RepackPlan,
                                    expertStride: UInt64) throws -> Data {
        let arch = plan.arch
        var layers: [GTurboLayerV1] = []
        layers.reserveCapacity(plan.layers.count)
        for lp in plan.layers {
            let layerFile = (lp.path as NSString).lastPathComponent
            var experts: [GTurboExpertV1] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                let physicalRank = lp.physicalRank(for: e)
                let base = UInt64(physicalRank) * lp.expertStride
                var tensors: [String: GTurboSubTensorV1] = [:]
                for slice in lp.subTensors {
                    let key: String
                    switch slice.component {
                    case "weights": key = slice.role
                    case "scales":  key = slice.role + "_scales"
                    case "biases":  key = slice.role + "_biases"
                    default:        key = slice.role + "_" + slice.component
                    }
                    guard slice.dtype == GTurboFormatV1.DType.u32.rawValue
                            || slice.dtype == GTurboFormatV1.DType.bf16.rawValue else {
                        throw RepackError.configurationInvalid(
                            detail: "unsupported packed expert dtype \(slice.dtype) for \(key)")
                    }
                    let shape = try slice.logicalShape.enumerated().map { index, value in
                        guard value <= UInt64(UInt32.max) else {
                            throw RepackError.configurationInvalid(
                                detail: "packed expert shape[\(index)] exceeds UInt32")
                        }
                        return UInt32(value)
                    }
                    let previous = tensors.updateValue(GTurboSubTensorV1(
                        offset: slice.offsetInExpertBlob,
                        size: slice.sizeInExpertBlob,
                        dtype: slice.dtype == GTurboFormatV1.DType.u32.rawValue ? "U32" : "BF16",
                        shape: shape,
                        bits: slice.bitsForWeights), forKey: key)
                    guard previous == nil else {
                        throw RepackError.configurationInvalid(
                            detail: "duplicate packed expert tensor key \(key)")
                    }
                }
                experts.append(GTurboExpertV1(
                    expert: e,
                    physicalRank: nil,
                    offset: base,
                    size: lp.expertStride,
                    tensors: tensors))
            }
            layers.append(GTurboLayerV1(layer: lp.layerIndex,
                                        file: layerFile,
                                        experts: experts))
        }
        return try GTurboPackedExpertsLayoutCodec.encode(
            GTurboPackedExpertsLayoutV1(
                expertStride: expertStride,
                numLayers: arch.numLayers,
                expertsPerLayer: plan.layers.first?.expertsPerLayer ?? 0,
                layers: layers))
    }
}
