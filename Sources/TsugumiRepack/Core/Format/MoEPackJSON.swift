import Foundation
import MoEPackFormat

/// JSON encoders for `manifest.json` and `packed_experts/layout.json`. The
/// files are small (kilobytes), so we use Foundation's `JSONSerialization`
/// rather than streaming.
enum MoEPackJSON {

    static let magic = MoEPackFormatV1.magic
    static let versionMajor = MoEPackFormatV1.versionMajor
    static let versionMinor = MoEPackFormatV1.versionMinor

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
        let wireArch = MoEPackManifestArchV1(
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
            fullAttentionLayerMask: arch.fullAttentionLayerMask.map(Int.init),
            // Gemma 4 is the family this format was written for, and its
            // manifests must stay byte-for-byte what they have always been:
            // the three keys below are written only for a family that needs
            // them, and an absent family means Gemma.
            family: arch.isGemma4 ? nil : arch.family,
            layerKinds: arch.isGemma4 ? nil : arch.layerKinds,
            linearAttention: arch.linearAttention.map {
                MoEPackManifestLinearAttentionV1(numKeyHeads: $0.numKeyHeads,
                                                numValueHeads: $0.numValueHeads,
                                                keyHeadDim: $0.keyHeadDim,
                                                valueHeadDim: $0.valueHeadDim,
                                                convKernelDim: $0.convKernelDim,
                                                layerCount: $0.layerCount)
            })
        func slot(_ name: String) throws -> MoEPackManifestQuantSlotV1 {
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
                return MoEPackManifestQuantSlotV1(
                    weightBits: 16,
                    scheme: "bf16",
                    scaleType: "none",
                    biasType: "none",
                    groupSize: plan.baseGroupSize)
            }
            // Only the 4-bit lattice can be symmetric: an INT8 group's zero
            // point is real data, so an INT8 slot keeps its bias array even in
            // a model whose 4-bit slots dropped theirs.
            let symmetric = plan.symmetric && weightBits == 4
            return MoEPackManifestQuantSlotV1(
                weightBits: weightBits,
                scheme: symmetric ? "sym" : plan.baseMode,
                scaleType: "BF16",
                biasType: symmetric ? "none" : "BF16",
                groupSize: plan.baseGroupSize)
        }
        let quant = MoEPackManifestQuantV1(
            embedding: try slot("embedding"),
            attention: try slot("attention"),
            router: try slot("router"),
            sharedExpert: try slot("sharedExpert"),
            routedExpert: try slot("routedExpert"))
        var wireFiles: [String: MoEPackManifestFileV1] = [:]
        wireFiles.reserveCapacity(files.count)
        for file in files {
            guard wireFiles.updateValue(
                MoEPackManifestFileV1(size: file.info.size, sha256: file.info.sha256),
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
        if wireArch.linearAttention != nil {
            // Same compatibility gate as the tower's and the drafter's: a
            // runtime that predates recurrent layers must refuse this model,
            // not read its zeros as sliding windows.
            flags["linearAttention"] = true
        }
        var wireVision: MoEPackManifestVisionV1?
        if let vision = plan.vision {
            // The flag is the compatibility gate: a runtime that predates vision
            // rejects the whole model rather than quietly ignoring images.
            flags["visionTower"] = true
            let config = vision.source.config
            wireVision = MoEPackManifestVisionV1(
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
                weightsPath: MoEPackFormatV1.visionWeightsPath,
                tensorCount: vision.tensorCount,
                payloadBytes: vision.payloadBytes,
                sourceRepo: vision.source.repoID,
                sourceRevision: vision.source.revision)
        }
        var wireDraft: MoEPackManifestDraftV1?
        if let draft = plan.draft {
            // Same compatibility gate as the tower's: a runtime that predates
            // speculation rejects the model rather than ignoring the drafter.
            flags["mtpDraft"] = true
            wireDraft = try draftSection(
                draft, fullAttentionLayerMask: wireArch.fullAttentionLayerMask)
        }
        var versionMinor = MoEPackFormatV1.versionMinor
        if wireVision != nil {
            versionMinor = max(versionMinor, MoEPackFormatV1.versionMinorVision)
        }
        if wireDraft != nil {
            versionMinor = max(versionMinor, MoEPackFormatV1.versionMinorDraft)
        }
        if wireArch.linearAttention != nil {
            versionMinor = max(versionMinor, MoEPackFormatV1.versionMinorLinearAttention)
        }
        return try MoEPackManifestCodec.encode(MoEPackManifestV1(
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
        throws -> MoEPackManifestDraftV1 {
        guard let sharedFull = fullAttentionLayerMask.lastIndex(of: 1),
              let sharedSliding = fullAttentionLayerMask.lastIndex(of: 0) else {
            throw RepackError.configurationInvalid(detail: """
                the drafter shares K/V with the target's last sliding and last full \
                attention layer, and this target has no layer of one of those kinds
                """)
        }
        let config = plan.source.config
        return MoEPackManifestDraftV1(
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
            quant: MoEPackManifestQuantSlotV1(weightBits: config.quantBits,
                                             scheme: config.quantMode,
                                             scaleType: "BF16",
                                             biasType: "BF16",
                                             groupSize: config.quantGroupSize),
            weightsPath: MoEPackFormatV1.draftWeightsPath,
            tensorCount: plan.tensorCount,
            payloadBytes: plan.payloadBytes,
            sourceRepo: plan.source.repoID,
            sourceRevision: plan.source.revision)
    }

    static func encodeLayout(plan: RepackPlan,
                                    expertStride: UInt64) throws -> Data {
        let arch = plan.arch
        var layers: [MoEPackLayerV1] = []
        layers.reserveCapacity(plan.layers.count)
        for lp in plan.layers {
            let layerFile = (lp.path as NSString).lastPathComponent
            var experts: [MoEPackExpertV1] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                let physicalRank = lp.physicalRank(for: e)
                let base = UInt64(physicalRank) * lp.expertStride
                var tensors: [String: MoEPackSubTensorV1] = [:]
                for slice in lp.subTensors {
                    let key: String
                    switch slice.component {
                    case "weights": key = slice.role
                    case "scales":  key = slice.role + "_scales"
                    case "biases":  key = slice.role + "_biases"
                    default:        key = slice.role + "_" + slice.component
                    }
                    guard slice.dtype == MoEPackFormatV1.DType.u32.rawValue
                            || slice.dtype == MoEPackFormatV1.DType.bf16.rawValue else {
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
                    let previous = tensors.updateValue(MoEPackSubTensorV1(
                        offset: slice.offsetInExpertBlob,
                        size: slice.sizeInExpertBlob,
                        dtype: slice.dtype == MoEPackFormatV1.DType.u32.rawValue ? "U32" : "BF16",
                        shape: shape,
                        bits: slice.bitsForWeights), forKey: key)
                    guard previous == nil else {
                        throw RepackError.configurationInvalid(
                            detail: "duplicate packed expert tensor key \(key)")
                    }
                }
                experts.append(MoEPackExpertV1(
                    expert: e,
                    physicalRank: nil,
                    offset: base,
                    size: lp.expertStride,
                    tensors: tensors))
            }
            layers.append(MoEPackLayerV1(layer: lp.layerIndex,
                                        file: layerFile,
                                        experts: experts))
        }
        return try MoEPackPackedExpertsLayoutCodec.encode(
            MoEPackPackedExpertsLayoutV1(
                expertStride: expertStride,
                numLayers: arch.numLayers,
                expertsPerLayer: plan.layers.first?.expertsPerLayer ?? 0,
                layers: layers))
    }
}
