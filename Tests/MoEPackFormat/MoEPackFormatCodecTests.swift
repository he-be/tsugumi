import Foundation
import Testing
@testable import MoEPackFormat

private enum FormatFixture {
    static let zeroSHA = String(repeating: "0", count: 64)

    static func manifest(sourceSnapshotHash: String? = "snapshot",
                         quant: MoEPackManifestQuantV1? = FormatFixture.quant,
                         bitWidthOverridesHonored: Int? = 120,
                         minor: Int = 0) -> MoEPackManifestV1 {
        MoEPackManifestV1(
            versionMinor: minor,
            flags: [
                "streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false,
            ],
            modelID: "fixture/model",
            sourceSnapshotHash: sourceSnapshotHash,
            arch: MoEPackManifestArchV1(
                hiddenSize: 64, ffnIntermediate: 128, moeIntermediateSize: 32,
                numHeads: 4, numKVHeads: 2, numFullKVHeads: 1,
                headDim: 16, fullHeadDim: 32, vocabSize: 1024,
                slidingWindow: 128, finalLogitSoftcap: 30,
                ropeTheta: 10_000, fullRopeTheta: 1_000_000,
                partialRotaryFactor: 0.25, numLayers: 1, numExperts: 2,
                topKExperts: 1, tieWordEmbeddings: true, attentionKEqV: true,
                hiddenActivation: "gelu_pytorch_tanh", fullAttentionLayerMask: [0]),
            quant: quant,
            files: [
                "model_weights.bin": MoEPackManifestFileV1(size: 16_448, sha256: zeroSHA),
                "packed_experts/layout.json": MoEPackManifestFileV1(size: 1, sha256: zeroSHA),
                "packed_experts/layer_00.bin": MoEPackManifestFileV1(
                    size: 2 * MoEPackFormatV1.alignmentBytes, sha256: zeroSHA),
            ],
            expertsPerLayer: 2,
            numLayers: 1,
            expertStride: MoEPackFormatV1.alignmentBytes,
            bitWidthOverridesHonored: bitWidthOverridesHonored)
    }

    /// A manifest that also carries the vision tower. The extra file entry is
    /// what the section's `weightsPath` has to resolve to.
    static func visionManifest(
        vision: MoEPackManifestVisionV1? = FormatFixture.vision(),
        flagged: Bool = true,
        minor: Int = MoEPackFormatV1.versionMinorVision
    ) -> MoEPackManifestV1 {
        let base = manifest(minor: minor)
        var flags = base.flags
        if flagged { flags["visionTower"] = true }
        var files = base.files
        files[MoEPackFormatV1.visionWeightsPath] =
            MoEPackManifestFileV1(size: 16_384 + 4_096, sha256: zeroSHA)
        return MoEPackManifestV1(
            versionMinor: minor,
            flags: flags,
            modelID: base.modelID,
            sourceSnapshotHash: base.sourceSnapshotHash,
            arch: base.arch,
            quant: base.quant,
            vision: vision,
            files: files,
            expertsPerLayer: base.expertsPerLayer,
            numLayers: base.numLayers,
            expertStride: base.expertStride,
            bitWidthOverridesHonored: base.bitWidthOverridesHonored)
    }

    static func vision(hiddenSize: Int = 1152,
                       numHeads: Int = 16,
                       headDim: Int = 72,
                       weightDType: String = "bf16",
                       weightsPath: String = MoEPackFormatV1.visionWeightsPath,
                       payloadBytes: UInt64 = 4_096,
                       imageTokenID: Int = 258880) -> MoEPackManifestVisionV1 {
        MoEPackManifestVisionV1(
            hiddenSize: hiddenSize, numLayers: 27, numHeads: numHeads,
            numKVHeads: 16, headDim: headDim, intermediateSize: 4304,
            patchSize: 16, poolingKernelSize: 3, positionEmbeddingSize: 10240,
            ropeTheta: 100, rmsNormEps: 1e-6,
            hiddenActivation: "gelu_pytorch_tanh", standardize: true,
            maxSoftTokens: 280, weightDType: weightDType,
            imageTokenID: imageTokenID, boiTokenID: 255999, eoiTokenID: 258882,
            weightsPath: weightsPath, tensorCount: 356,
            payloadBytes: payloadBytes,
            sourceRepo: "google/fixture", sourceRevision: "f1e06dc5")
    }

    /// A manifest that also carries the MTP drafter. The target arch gets a
    /// second layer so it has both a sliding and a full attention layer for the
    /// drafter to share K/V with.
    static func draftManifest(
        draft: MoEPackManifestDraftV1? = FormatFixture.draft(),
        flagged: Bool = true,
        minor: Int = MoEPackFormatV1.versionMinorDraft
    ) -> MoEPackManifestV1 {
        let base = manifest()
        var flags = base.flags
        if flagged { flags["mtpDraft"] = true }
        var files = base.files
        files[MoEPackFormatV1.draftWeightsPath] =
            MoEPackManifestFileV1(size: 16_384 + 4_096, sha256: zeroSHA)
        let arch = base.arch
        return MoEPackManifestV1(
            versionMinor: minor,
            flags: flags,
            modelID: base.modelID,
            sourceSnapshotHash: base.sourceSnapshotHash,
            arch: MoEPackManifestArchV1(
                hiddenSize: arch.hiddenSize, ffnIntermediate: arch.ffnIntermediate,
                moeIntermediateSize: arch.moeIntermediateSize,
                numHeads: arch.numHeads, numKVHeads: arch.numKVHeads,
                numFullKVHeads: arch.numFullKVHeads, headDim: arch.headDim,
                fullHeadDim: arch.fullHeadDim, vocabSize: arch.vocabSize,
                slidingWindow: arch.slidingWindow,
                finalLogitSoftcap: arch.finalLogitSoftcap,
                ropeTheta: arch.ropeTheta, fullRopeTheta: arch.fullRopeTheta,
                partialRotaryFactor: arch.partialRotaryFactor,
                numLayers: 2, numExperts: arch.numExperts,
                topKExperts: arch.topKExperts,
                tieWordEmbeddings: arch.tieWordEmbeddings,
                attentionKEqV: arch.attentionKEqV,
                hiddenActivation: arch.hiddenActivation,
                fullAttentionLayerMask: [0, 1]),
            quant: base.quant,
            draft: draft,
            files: files,
            expertsPerLayer: base.expertsPerLayer,
            numLayers: 2,
            expertStride: base.expertStride,
            bitWidthOverridesHonored: base.bitWidthOverridesHonored)
    }

    /// Geometry the drafter has to restate from the target arch above.
    static func draft(headDim: Int = 16,
                      backboneHiddenSize: Int = 64,
                      tieWordEmbeddings: Bool = true,
                      sharedSlidingKVLayer: Int = 0,
                      sharedFullKVLayer: Int = 1,
                      weightsPath: String = MoEPackFormatV1.draftWeightsPath,
                      payloadBytes: UInt64 = 4_096) -> MoEPackManifestDraftV1 {
        MoEPackManifestDraftV1(
            hiddenSize: 1024, numLayers: 4, numHeads: 16, numKVHeads: 2,
            numFullKVHeads: 1, headDim: headDim, fullHeadDim: 32,
            intermediateSize: 8192, backboneHiddenSize: backboneHiddenSize,
            vocabSize: 1024, slidingWindow: 128,
            ropeTheta: 10_000, fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25, rmsNormEps: 1e-6,
            hiddenActivation: "gelu_pytorch_tanh",
            tieWordEmbeddings: tieWordEmbeddings, attentionKEqV: true,
            fullAttentionLayerMask: [0, 0, 0, 1],
            sharedSlidingKVLayer: sharedSlidingKVLayer,
            sharedFullKVLayer: sharedFullKVLayer,
            quant: quantSlot, weightsPath: weightsPath, tensorCount: 48,
            payloadBytes: payloadBytes,
            sourceRepo: "mlx-community/fixture", sourceRevision: "bb94eae1")
    }

    static let quantSlot = MoEPackManifestQuantSlotV1(
        weightBits: 4, scheme: "affine", scaleType: "BF16",
        biasType: "BF16", groupSize: 64)

    static let quant = MoEPackManifestQuantV1(
        embedding: quantSlot, attention: quantSlot, router: quantSlot,
        sharedExpert: quantSlot, routedExpert: quantSlot)

    static func layout(explicitIDs: Bool = true,
                       explicitRanks: Bool = true) -> MoEPackPackedExpertsLayoutV1 {
        let tensor = MoEPackSubTensorV1(
            offset: 0, size: 32, dtype: "U32", shape: [8, 8], bits: 4)
        return MoEPackPackedExpertsLayoutV1(
            expertStride: MoEPackFormatV1.alignmentBytes,
            numLayers: 1,
            expertsPerLayer: 2,
            layers: [MoEPackLayerV1(
                layer: 0,
                file: "layer_00.bin",
                experts: (0..<2).map { expert in
                    MoEPackExpertV1(
                        expert: explicitIDs ? expert : nil,
                        physicalRank: explicitRanks ? expert : nil,
                        offset: UInt64(expert) * MoEPackFormatV1.alignmentBytes,
                        size: MoEPackFormatV1.alignmentBytes,
                        tensors: ["gate": tensor])
                })])
    }

    static func residentBytes(nameOffset: UInt32 = 96,
                              nameBytes: [UInt8] = Array("weight".utf8),
                              dtype: UInt8 = MoEPackFormatV1.DType.u32.rawValue,
                              reserved: UInt8 = 0,
                              shape: [UInt32] = [1, 1, 0, 0],
                              fileOffset: UInt64 = MoEPackFormatV1.alignmentBytes,
                              sizeBytes: UInt64 = 16,
                              scaleOffset: UInt64 = MoEPackFormatV1.alignmentBytes + 16,
                              scaleSize: UInt64 = 8,
                              biasOffset: UInt64 = MoEPackFormatV1.alignmentBytes + 24,
                              biasSize: UInt64 = 8,
                              header: MoEPackResidentIndexHeaderV1 = .init(
                                indexSize: MoEPackFormatV1.alignmentBytes,
                                residentSize: 64, entryCount: 1)) -> Data {
        var bytes = Data(repeating: 0, count: Int(header.indexSize))
        bytes.withUnsafeMutableBytes { raw in
            MoEPackResidentIndexCodec.writeHeader(into: raw.baseAddress!, header: header)
            guard raw.count >= 96 else { return }
            let entry = MoEPackResidentIndexEntryV1(
                name: String(decoding: nameBytes, as: UTF8.self), dtype: dtype,
                fileOffset: fileOffset, sizeBytes: sizeBytes, shape: shape,
                scaleOffset: scaleOffset, scaleSize: scaleSize,
                biasOffset: biasOffset, biasSize: biasSize)
            MoEPackResidentIndexCodec.writeEntry(
                into: raw.baseAddress!.advanced(by: MoEPackFormatV1.residentHeaderBytes),
                entry: entry, nameOffset: nameOffset)
            raw[MoEPackFormatV1.residentHeaderBytes + 7] = reserved
            if Int(nameOffset) + nameBytes.count <= raw.count {
                for (index, byte) in nameBytes.enumerated() {
                    raw[Int(nameOffset) + index] = byte
                }
            }
        }
        return bytes
    }
}

@Suite struct MoEPackManifestCodecTests {
    @Test func roundTripPreservesOptionalPresenceAndWriterField() throws {
        let manifest = FormatFixture.manifest()
        let encoded = try MoEPackManifestCodec.encode(manifest)
        #expect(try MoEPackManifestCodec.decode(encoded) == manifest)
        let root = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(root["bitWidthOverridesHonored"] as? Int == 120)
    }

    @Test func acceptsAdditiveMinorAndUnknownTopLevelKey() throws {
        let data = try MoEPackManifestCodec.encode(FormatFixture.manifest(minor: 9))
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root["futureMetadata"] = ["ignored": true]
        let changed = try JSONSerialization.data(withJSONObject: root)
        #expect(try MoEPackManifestCodec.decode(changed).versionMinor == 9)
    }

    @Test(arguments: ["sourceSnapshotHash", "quant", "bitWidthOverridesHonored"])
    func acceptsAbsentLegacyOptionalField(_ key: String) throws {
        let data = try MoEPackManifestCodec.encode(FormatFixture.manifest())
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root.removeValue(forKey: key)
        let changed = try JSONSerialization.data(withJSONObject: root)
        let decoded = try MoEPackManifestCodec.decode(changed)
        if key == "sourceSnapshotHash" { #expect(decoded.sourceSnapshotHash == nil) }
        if key == "quant" { #expect(decoded.quant == nil) }
        if key == "bitWidthOverridesHonored" { #expect(decoded.bitWidthOverridesHonored == nil) }
    }

    @Test func rejectsUnsafeManifestPaths() throws {
        var root = try #require(JSONSerialization.jsonObject(
            with: MoEPackManifestCodec.encode(FormatFixture.manifest())) as? [String: Any])
        var files = try #require(root["files"] as? [String: Any])
        files["../escape.bin"] = files.removeValue(forKey: "model_weights.bin")
        root["files"] = files
        let data = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: MoEPackFormatError.self) { try MoEPackManifestCodec.decode(data) }
    }

    @Test(arguments: ["manifest.json", "verified-install.json", "tokenizer"])
    func rejectsReservedManifestPaths(_ reserved: String) throws {
        var root = try #require(JSONSerialization.jsonObject(
            with: MoEPackManifestCodec.encode(FormatFixture.manifest())) as? [String: Any])
        var files = try #require(root["files"] as? [String: Any])
        files[reserved] = files["model_weights.bin"]
        root["files"] = files
        let data = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: MoEPackFormatError.self) { try MoEPackManifestCodec.decode(data) }
    }

    @Test func textOnlyManifestOmitsTheVisionSectionEntirely() throws {
        let encoded = try MoEPackManifestCodec.encode(FormatFixture.manifest())
        let root = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        // Byte-identity of a text-only install depends on this key never
        // appearing, so assert its absence rather than that it decodes to nil.
        #expect(root["vision"] == nil)
        #expect(try MoEPackManifestCodec.decode(encoded).vision == nil)
    }

    @Test func roundTripsTheVisionSection() throws {
        let manifest = FormatFixture.visionManifest()
        let encoded = try MoEPackManifestCodec.encode(manifest)
        let decoded = try MoEPackManifestCodec.decode(encoded)
        #expect(decoded == manifest)
        #expect(decoded.vision?.tensorCount == 356)
        #expect(decoded.flags["visionTower"] == true)
    }

    @Test func rejectsVisionSectionWithoutItsFlag() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.visionManifest(flagged: false))
        }
    }

    @Test func rejectsVisionFlagWithoutASection() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.visionManifest(vision: nil))
        }
    }

    @Test func rejectsVisionAtTheTextOnlyMinorVersion() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.visionManifest(minor: 0))
        }
    }

    @Test func rejectsVisionWeightsPathThatIsNotDeclaredInFiles() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.visionManifest(
                vision: FormatFixture.vision(weightsPath: "vision/other.bin")))
        }
    }

    @Test func rejectsVisionPayloadLargerThanItsFile() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.visionManifest(
                vision: FormatFixture.vision(payloadBytes: 1 << 40)))
        }
    }

    @Test func rejectsNonBF16VisionWeights() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.visionManifest(
                vision: FormatFixture.vision(weightDType: "fp16")))
        }
    }

    @Test func rejectsVisionHeadGeometryThatDoesNotFillTheHiddenSize() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.visionManifest(
                vision: FormatFixture.vision(numHeads: 15)))
        }
    }

    @Test func rejectsDuplicateImageMarkerTokenIDs() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.visionManifest(
                vision: FormatFixture.vision(imageTokenID: 255999)))
        }
    }

    @Test func textOnlyManifestOmitsTheDraftSectionEntirely() throws {
        let encoded = try MoEPackManifestCodec.encode(FormatFixture.manifest())
        let root = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        // Byte-identity of an install without a drafter depends on this key
        // never appearing (`docs/mtp/04-PHASES.md` M1).
        #expect(root["draft"] == nil)
        #expect(try MoEPackManifestCodec.decode(encoded).draft == nil)
        let flags = try #require(root["flags"] as? [String: Any])
        #expect(flags["mtpDraft"] == nil)
    }

    @Test func roundTripsTheDraftSection() throws {
        let manifest = FormatFixture.draftManifest()
        let encoded = try MoEPackManifestCodec.encode(manifest)
        let decoded = try MoEPackManifestCodec.decode(encoded)
        #expect(decoded == manifest)
        #expect(decoded.draft?.tensorCount == 48)
        #expect(decoded.flags["mtpDraft"] == true)
    }

    @Test func rejectsDraftSectionWithoutItsFlag() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.draftManifest(flagged: false))
        }
    }

    @Test func rejectsDraftFlagWithoutASection() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.draftManifest(draft: nil))
        }
    }

    @Test func rejectsDraftBelowItsMinorVersion() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(
                FormatFixture.draftManifest(minor: MoEPackFormatV1.versionMinorVision))
        }
    }

    @Test func rejectsDraftWeightsPathThatIsNotDeclaredInFiles() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.draftManifest(
                draft: FormatFixture.draft(weightsPath: "draft/other.bin")))
        }
    }

    @Test func rejectsDraftPayloadLargerThanItsFile() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.draftManifest(
                draft: FormatFixture.draft(payloadBytes: 1 << 40)))
        }
    }

    /// The drafter reads the target's K/V, so a head dimension of its own is not
    /// a tuning choice — it is a contradiction the manifest must not carry.
    @Test func rejectsDraftGeometryThatDisagreesWithTheTargetArch() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.draftManifest(
                draft: FormatFixture.draft(headDim: 17)))
        }
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.draftManifest(
                draft: FormatFixture.draft(backboneHiddenSize: 65)))
        }
    }

    @Test func rejectsSharedKVLayersOfTheWrongKind() throws {
        // Layer 0 of the fixture target is sliding, layer 1 is full.
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.draftManifest(
                draft: FormatFixture.draft(sharedSlidingKVLayer: 1)))
        }
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.draftManifest(
                draft: FormatFixture.draft(sharedFullKVLayer: 0)))
        }
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.draftManifest(
                draft: FormatFixture.draft(sharedFullKVLayer: 7)))
        }
    }

    @Test func rejectsADraftWhoseLMHeadIsNotTied() throws {
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackManifestCodec.encode(FormatFixture.draftManifest(
                draft: FormatFixture.draft(tieWordEmbeddings: false)))
        }
    }

    /// The gate a runtime that predates MTP relies on: `mtpDraft` is a known v1
    /// flag now, so a build that does not know it rejects the whole model rather
    /// than running without the drafter it advertises.
    @Test func draftFlagIsAKnownV1Flag() {
        #expect(MoEPackFormatV1.knownFlags.contains("mtpDraft"))
    }

    @Test func rejectsUnknownFlagsIncludingTypos() throws {
        var root = try #require(JSONSerialization.jsonObject(
            with: MoEPackManifestCodec.encode(FormatFixture.manifest())) as? [String: Any])
        var flags = try #require(root["flags"] as? [String: Any])
        flags["visionTowers"] = true
        root["flags"] = flags
        let data = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: MoEPackFormatError.self) { try MoEPackManifestCodec.decode(data) }
    }

    @Test func rejectsFileDirectoryPrefixCollision() throws {
        var root = try #require(JSONSerialization.jsonObject(
            with: MoEPackManifestCodec.encode(FormatFixture.manifest())) as? [String: Any])
        var files = try #require(root["files"] as? [String: Any])
        files["packed_experts"] = files["model_weights.bin"]
        root["files"] = files
        let data = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: MoEPackFormatError.self) { try MoEPackManifestCodec.decode(data) }
    }
}

@Suite struct MoEPackPackedExpertsLayoutCodecTests {
    @Test func roundTripPreservesIdentityFallback() throws {
        let layout = FormatFixture.layout(explicitIDs: false, explicitRanks: false)
        let encoded = try MoEPackPackedExpertsLayoutCodec.encode(layout)
        #expect(try MoEPackPackedExpertsLayoutCodec.decode(encoded) == layout)
    }

    @Test func rejectsMixedExplicitAndPositionalIDs() throws {
        let base = FormatFixture.layout()
        let experts = [
            base.layers[0].experts[0],
            MoEPackExpertV1(
                expert: nil, physicalRank: 1,
                offset: MoEPackFormatV1.alignmentBytes,
                size: MoEPackFormatV1.alignmentBytes,
                tensors: base.layers[0].experts[1].tensors),
        ]
        let layout = MoEPackPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 2,
            layers: [MoEPackLayerV1(layer: 0, file: "layer_00.bin", experts: experts)])
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackPackedExpertsLayoutCodec.encode(layout)
        }
    }

    @Test func rejectsSlashBearingLayerBasename() throws {
        let base = FormatFixture.layout()
        let layout = MoEPackPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 2,
            layers: [MoEPackLayerV1(
                layer: 0, file: "nested/layer.bin", experts: base.layers[0].experts)])
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackPackedExpertsLayoutCodec.encode(layout)
        }
    }

    @Test func rejectsReservedLayoutBasename() throws {
        let base = FormatFixture.layout()
        let layout = MoEPackPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 2,
            layers: [MoEPackLayerV1(
                layer: 0, file: "layout.json", experts: base.layers[0].experts)])
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackPackedExpertsLayoutCodec.encode(layout)
        }
    }

    @Test func crossValidationRequiresExactLayerManifestEntry() throws {
        var sizes = FormatFixture.manifest().files.mapValues(\.size)
        sizes["packed_experts/layer_00.bin"] = nil
        sizes["packed_experts/./layer_00.bin"] = 2 * MoEPackFormatV1.alignmentBytes
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackV1StructuralValidator.crossValidate(
                manifestNumLayers: 1, manifestExpertsPerLayer: 2,
                manifestExpertStride: MoEPackFormatV1.alignmentBytes,
                manifestFileSizes: sizes, layout: FormatFixture.layout())
        }
    }
}

@Suite struct MoEPackResidentIndexCodecTests {
    @Test func decodesValidRegion() throws {
        let header = MoEPackResidentIndexHeaderV1(
            indexSize: MoEPackFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(header: header)
        let entries = try bytes.withUnsafeBytes {
            try MoEPackResidentIndexCodec.decodeRegion($0, header: header)
        }
        #expect(entries.map(\.name) == ["weight"])
    }

    @Test func rejectsOverlappingResidentPayloadRanges() throws {
        let header = MoEPackResidentIndexHeaderV1(
            indexSize: MoEPackFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(
            scaleOffset: MoEPackFormatV1.alignmentBytes + 8,
            header: header)
        #expect(throws: MoEPackFormatError.self) {
            try bytes.withUnsafeBytes {
                try MoEPackResidentIndexCodec.decodeRegion($0, header: header)
            }
        }
    }

    @Test func rejectsNameInsideEntryTable() throws {
        let header = MoEPackResidentIndexHeaderV1(
            indexSize: MoEPackFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(
            nameOffset: UInt32(MoEPackFormatV1.residentHeaderBytes), header: header)
        #expect(throws: MoEPackFormatError.self) {
            try bytes.withUnsafeBytes { try MoEPackResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsInvalidUTF8Name() throws {
        let header = MoEPackResidentIndexHeaderV1(
            indexSize: MoEPackFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(nameBytes: [0xFF], header: header)
        #expect(throws: MoEPackFormatError.self) {
            try bytes.withUnsafeBytes { try MoEPackResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsNonzeroReservedByte() throws {
        let header = MoEPackResidentIndexHeaderV1(
            indexSize: MoEPackFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(reserved: 1, header: header)
        #expect(throws: MoEPackFormatError.self) {
            try bytes.withUnsafeBytes { try MoEPackResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsUnknownDType() throws {
        let header = MoEPackResidentIndexHeaderV1(
            indexSize: MoEPackFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(dtype: 255, header: header)
        #expect(throws: MoEPackFormatError.self) {
            try bytes.withUnsafeBytes { try MoEPackResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsEntryTableOverflow() throws {
        let header = MoEPackResidentIndexHeaderV1(
            indexSize: MoEPackFormatV1.alignmentBytes,
            residentSize: 64, entryCount: UInt64.max)
        let bytes = FormatFixture.residentBytes(header: header)
        #expect(throws: MoEPackFormatError.self) {
            try bytes.withUnsafeBytes { try MoEPackResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsPayloadRangeOutsideResidentRegion() throws {
        let header = MoEPackResidentIndexHeaderV1(
            indexSize: MoEPackFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(
            fileOffset: MoEPackFormatV1.alignmentBytes + 56,
            sizeBytes: 16, header: header)
        #expect(throws: MoEPackFormatError.self) {
            try bytes.withUnsafeBytes { try MoEPackResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsUnalignedIndexRegion() throws {
        let header = MoEPackResidentIndexHeaderV1(indexSize: 128, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(
            fileOffset: 128, scaleOffset: 144, biasOffset: 152, header: header)
        #expect(throws: MoEPackFormatError.self) {
            try bytes.withUnsafeBytes { try MoEPackResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsEmptyPrimaryPayload() throws {
        let header = MoEPackResidentIndexHeaderV1(
            indexSize: MoEPackFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(
            fileOffset: 0, sizeBytes: 0,
            scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0,
            header: header)
        #expect(throws: MoEPackFormatError.self) {
            try bytes.withUnsafeBytes { try MoEPackResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsNoncanonicalShape() throws {
        let header = MoEPackResidentIndexHeaderV1(
            indexSize: MoEPackFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(shape: [1, 0, 1, 0], header: header)
        #expect(throws: MoEPackFormatError.self) {
            try bytes.withUnsafeBytes { try MoEPackResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsAbsentCompanionWithNonzeroOffset() throws {
        let header = MoEPackResidentIndexHeaderV1(
            indexSize: MoEPackFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(
            scaleOffset: MoEPackFormatV1.alignmentBytes + 16, scaleSize: 0,
            biasOffset: 0, biasSize: 0, header: header)
        #expect(throws: MoEPackFormatError.self) {
            try bytes.withUnsafeBytes { try MoEPackResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }
}

@Suite struct MoEPackV1StructuralValidatorTests {
    @Test func acceptsMatchingDocuments() throws {
        try MoEPackV1StructuralValidator.crossValidate(
            manifest: FormatFixture.manifest(), layout: FormatFixture.layout())
    }

    @Test func rejectsTensorOutsideExpertBlob() throws {
        let base = FormatFixture.layout()
        let invalid = MoEPackSubTensorV1(
            offset: base.expertStride - 8, size: 16,
            dtype: "U32", shape: [1], bits: 4)
        let expert = MoEPackExpertV1(
            expert: 0, physicalRank: 0, offset: 0, size: base.expertStride,
            tensors: ["gate": invalid])
        let layout = MoEPackPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 1,
            layers: [MoEPackLayerV1(layer: 0, file: "layer_00.bin", experts: [expert])])
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackV1StructuralValidator.validate(layout)
        }
    }

    @Test func rejectsDuplicateLayerIDs() throws {
        let base = FormatFixture.layout()
        let layout = MoEPackPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 2, expertsPerLayer: 2,
            layers: [
                base.layers[0],
                MoEPackLayerV1(
                    layer: 0, file: "other.bin", experts: base.layers[0].experts),
            ])
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackV1StructuralValidator.validate(layout)
        }
    }

    @Test func rejectsDuplicateLogicalExpertIDs() throws {
        let base = FormatFixture.layout()
        let experts = base.layers[0].experts.enumerated().map { position, expert in
            MoEPackExpertV1(
                expert: 0, physicalRank: position,
                offset: UInt64(position) * base.expertStride, size: expert.size,
                tensors: expert.tensors)
        }
        let layout = MoEPackPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 2,
            layers: [MoEPackLayerV1(layer: 0, file: "layer_00.bin", experts: experts)])
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackV1StructuralValidator.validate(layout)
        }
    }

    @Test func rejectsDuplicatePhysicalRanks() throws {
        let base = FormatFixture.layout()
        let experts = base.layers[0].experts.enumerated().map { position, expert in
            MoEPackExpertV1(
                expert: position, physicalRank: 0,
                offset: 0, size: expert.size, tensors: expert.tensors)
        }
        let layout = MoEPackPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 2,
            layers: [MoEPackLayerV1(layer: 0, file: "layer_00.bin", experts: experts)])
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackV1StructuralValidator.validate(layout)
        }
    }

    @Test func rejectsOverlappingTensorRanges() throws {
        let base = FormatFixture.layout()
        let tensor = MoEPackSubTensorV1(
            offset: 16, size: 32, dtype: "U32", shape: [8, 8], bits: 4)
        let expert = MoEPackExpertV1(
            expert: 0, physicalRank: 0, offset: 0, size: base.expertStride,
            tensors: [
                "gate": base.layers[0].experts[0].tensors["gate"]!,
                "up": tensor,
            ])
        let layout = MoEPackPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 1,
            layers: [MoEPackLayerV1(layer: 0, file: "layer_00.bin", experts: [expert])])
        #expect(throws: MoEPackFormatError.self) {
            try MoEPackV1StructuralValidator.validate(layout)
        }
    }
}
