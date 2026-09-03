import Foundation
import Testing
@testable import MoEPackFormat
@testable import TsugumiRepackCore

@Suite(.serialized, .fakeHFServer)
struct DraftInstallTests {

    /// A refusal is only worth having if it is the refusal we meant, so the
    /// negative cases assert the reason and not merely that something threw.
    private func expectInstallRefusal(_ options: RemoteStreamingRepackOptions,
                                      because reason: String) async {
        do {
            _ = try await RemoteStreamingRepacker(options: options).run()
            Issue.record("install succeeded; expected a refusal mentioning \(reason)")
        } catch {
            #expect("\(error)".contains(reason), "unexpected refusal: \(error)")
        }
    }

    private func options(snapshotDir: String,
                         outputDir: String,
                         includeDraft: Bool,
                         pin: DraftSourcePin) -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            sourceSnapshotDirectory: snapshotDir,
            outputDir: outputDir,
            requireKnownSource: false,
            rangeChunkBytes: 4096,
            minFreeReserveBytes: 0,
            overwrite: true,
            includeDraft: includeDraft,
            draftPin: pin,
            downloadSession: fakeHFSession(),
            baseURL: URL(string: "https://hf.test")!,
            retryBaseDelayNs: 0)
    }

    @Test func installsTheDrafterAlongsideTheTextWeights() async throws {
        let snapshotDir = tmpDirForRemote("draft-snapshot")
        let outputDir = tmpPathForRemote("draft-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpRemote([outputDir])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticDraftRepo.build()
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: options(snapshotDir: snapshotDir,
                             outputDir: outputDir,
                             includeDraft: true,
                             pin: repo.pin)).run()

        // 1. The manifest declares the drafter, its flag, and the bumped minor.
        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (outputDir as NSString).appendingPathComponent("manifest.json")))
        let manifest = try MoEPackManifestCodec.decode(manifestData)
        #expect(manifest.flags["mtpDraft"] == true)
        #expect(manifest.versionMinor == MoEPackFormatV1.versionMinorDraft)
        let draft = try #require(manifest.draft)
        #expect(draft.tensorCount == repo.pin.expectedTensorCount)
        #expect(draft.payloadBytes == repo.pin.expectedPayloadBytes)
        #expect(draft.sourceRepo == repo.repoID)
        #expect(draft.sourceRevision == repo.revision)
        #expect(draft.weightsPath == MoEPackFormatV1.draftWeightsPath)
        #expect(draft.hiddenSize == SyntheticDraftRepo.config.hiddenSize)
        #expect(draft.quant.weightBits == SyntheticDraftRepo.config.quantBits)
        #expect(draft.quant.groupSize == SyntheticDraftRepo.config.quantGroupSize)
        // The synthetic target is layer 0 sliding, layer 1 full, and the drafter
        // reads the last of each.
        #expect(draft.sharedSlidingKVLayer == 0)
        #expect(draft.sharedFullKVLayer == 1)

        // 2. The file exists at the declared size.
        let weightsPath = (outputDir as NSString)
            .appendingPathComponent(MoEPackFormatV1.draftWeightsPath)
        let entry = try #require(manifest.files[MoEPackFormatV1.draftWeightsPath])
        let weights = try Data(contentsOf: URL(fileURLWithPath: weightsPath))
        #expect(UInt64(weights.count) == entry.size)

        // 3. Its index names every drafter tensor, and the bytes behind one of
        //    them — weight, scales and biases alike — are the source bytes, not
        //    merely the right length.
        let indexEntries = try weights.withUnsafeBytes { raw -> [MoEPackResidentIndexEntryV1] in
            let header = try MoEPackResidentIndexCodec.decodeHeader(raw)
            return try MoEPackResidentIndexCodec.decodeRegion(raw, header: header)
        }
        #expect(indexEntries.count == repo.pin.expectedTensorCount)
        let byName = Dictionary(uniqueKeysWithValues: indexEntries.map { ($0.name, $0) })
        for name in ["embed_tokens.weight", "pre_projection.weight",
                     "post_projection.weight", "layers.1.self_attn.q_proj.weight"] {
            let e = try #require(byName[name])
            #expect(e.dtype == MoEPackFormatV1.DType.u32.rawValue)
            let base = String(name.dropLast(".weight".count))
            for (offset, size, expectedBytes) in [
                (e.fileOffset, e.sizeBytes, repo.expectedTensorBytes[name]),
                (e.scaleOffset, e.scaleSize, repo.expectedTensorBytes[base + ".scales"]),
                (e.biasOffset, e.biasSize, repo.expectedTensorBytes[base + ".biases"]),
            ] {
                let expectedBytes = try #require(expectedBytes)
                #expect(UInt64(expectedBytes.count) == size)
                let start = Int(offset)
                #expect(weights.subdata(in: start..<(start + expectedBytes.count))
                        == expectedBytes)
            }
        }
        let norm = try #require(byName["norm.weight"])
        #expect(norm.dtype == MoEPackFormatV1.DType.bf16.rawValue)
        #expect(norm.scaleSize == 0 && norm.biasSize == 0)
        let normStart = Int(norm.fileOffset)
        #expect(weights.subdata(in: normStart..<(normStart + Int(norm.sizeBytes)))
                == repo.expectedTensorBytes["norm.weight"])

        // 4. The whole install still verifies, drafter included.
        let result = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputMoEPack: outputDir))
        #expect(result.unexpectedEntries.isEmpty)
        #expect(result.bytesVerified > entry.size)
    }

    /// The property M1 exists to protect: without the flag, nothing about the
    /// output moves — no file, no manifest key, not one request.
    @Test func textOnlyInstallWritesNoDraftArtifacts() async throws {
        let snapshotDir = tmpDirForRemote("draft-off-snapshot")
        let outputDir = tmpPathForRemote("draft-off-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpRemote([outputDir])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticDraftRepo.build()
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: options(snapshotDir: snapshotDir,
                             outputDir: outputDir,
                             includeDraft: false,
                             pin: repo.pin)).run()

        #expect(!FileManager.default.fileExists(
            atPath: (outputDir as NSString).appendingPathComponent("draft")))
        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (outputDir as NSString).appendingPathComponent("manifest.json")))
        let root = try #require(
            JSONSerialization.jsonObject(with: manifestData) as? [String: Any])
        #expect(root["draft"] == nil)
        #expect(root["versionMinor"] as? Int == MoEPackFormatV1.versionMinor)
        let flags = try #require(root["flags"] as? [String: Any])
        #expect(flags["mtpDraft"] == nil)
        // Not one request went to the drafter repository.
        #expect(FakeHFURLProtocol.requestCounts.isEmpty)
    }

    /// The frozen-fixture check of `docs/mtp/04-PHASES.md` M1, stated as an A/B:
    /// the same snapshot installed twice, once by a build that knows about
    /// drafters and once asked for one, has to produce identical text-side bytes
    /// and an identical text-only manifest.
    @Test func addingTheFlagChangesNothingOnTheTextSide() async throws {
        let snapshotDir = tmpDirForRemote("draft-ab-snapshot")
        let plain = tmpPathForRemote("draft-ab-plain")
        let withDraft = tmpPathForRemote("draft-ab-with")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpRemote([plain, withDraft])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticDraftRepo.build()
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: options(snapshotDir: snapshotDir,
                             outputDir: plain,
                             includeDraft: false,
                             pin: repo.pin)).run()
        _ = try await RemoteStreamingRepacker(
            options: options(snapshotDir: snapshotDir,
                             outputDir: withDraft,
                             includeDraft: true,
                             pin: repo.pin)).run()

        func read(_ dir: String, _ relativePath: String) throws -> Data {
            try Data(contentsOf: URL(fileURLWithPath:
                (dir as NSString).appendingPathComponent(relativePath)))
        }
        for relativePath in ["model_weights.bin",
                             "packed_experts/layer_00.bin",
                             "packed_experts/layer_01.bin",
                             "packed_experts/layout.json"] {
            #expect(try read(plain, relativePath) == (try read(withDraft, relativePath)),
                    "\(relativePath) differs when a drafter is installed")
        }
        // The manifests differ only by the drafter: same arch, same quant, same
        // digests for every text-side file.
        let a = try MoEPackManifestCodec.decode(try read(plain, "manifest.json"))
        let b = try MoEPackManifestCodec.decode(try read(withDraft, "manifest.json"))
        #expect(a.arch == b.arch)
        #expect(a.quant == b.quant)
        #expect(a.expertStride == b.expertStride)
        for (path, entry) in a.files {
            #expect(b.files[path] == entry, "\(path) changed")
        }
        #expect(b.files.count == a.files.count + 1)
    }

    @Test func refusesADrafterRepositoryWhoseIndexIsNotThePinnedOne() async throws {
        let snapshotDir = tmpDirForRemote("draft-pin-snapshot")
        let outputDir = tmpPathForRemote("draft-pin-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpRemote([outputDir])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticDraftRepo.build()
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files
        let wrongPin = DraftSourcePin(
            repoID: repo.pin.repoID,
            revision: repo.pin.revision,
            indexSha256Hex: String(repeating: "a", count: 64),
            displayName: repo.pin.displayName,
            strippedNamePrefix: repo.pin.strippedNamePrefix,
            expectedTensorCount: repo.pin.expectedTensorCount,
            expectedPayloadBytes: repo.pin.expectedPayloadBytes,
            provenanceTensors: repo.pin.provenanceTensors,
            config: repo.pin.config)

        await expectInstallRefusal(options(snapshotDir: snapshotDir,
                                           outputDir: outputDir,
                                           includeDraft: true,
                                           pin: wrongPin),
                                   because: "source fingerprint")
    }

    /// The evidence that this is Google's QAT assistant and not some other
    /// 4-bit drafter: the tensors MLX left unquantized still hash to what the
    /// BF16 release produces.
    @Test func refusesADrafterWhoseUnquantizedTensorsAreNotTheQATAssistants() async throws {
        let snapshotDir = tmpDirForRemote("draft-prov-snapshot")
        let outputDir = tmpPathForRemote("draft-prov-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpRemote([outputDir])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticDraftRepo.build()
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files
        // Internally consistent — its own index and config match — but built
        // from different weights, which only the pinned digest can see.
        let wrongPin = DraftSourcePin(
            repoID: repo.pin.repoID,
            revision: repo.pin.revision,
            indexSha256Hex: repo.pin.indexSha256Hex,
            displayName: repo.pin.displayName,
            strippedNamePrefix: repo.pin.strippedNamePrefix,
            expectedTensorCount: repo.pin.expectedTensorCount,
            expectedPayloadBytes: repo.pin.expectedPayloadBytes,
            provenanceTensors: repo.pin.provenanceTensors.map {
                DraftSourcePin.ProvenanceTensor(repoName: $0.repoName,
                                                sha256: String(repeating: "b", count: 64))
            },
            config: repo.pin.config)

        await expectInstallRefusal(options(snapshotDir: snapshotDir,
                                           outputDir: outputDir,
                                           includeDraft: true,
                                           pin: wrongPin),
                                   because: "this is not that checkpoint")
        #expect(!FileManager.default.fileExists(atPath: outputDir))
    }

    /// The drafter borrows the target's K/V. A drafter built for a different
    /// head geometry has to be refused, not merely produce bad proposals.
    @Test func refusesADrafterWhoseGeometryDoesNotFitTheTarget() async throws {
        let snapshotDir = tmpDirForRemote("draft-fit-snapshot")
        let outputDir = tmpPathForRemote("draft-fit-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpRemote([outputDir])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let base = SyntheticDraftRepo.config
        let mismatched = DraftSourceConfig(
            hiddenSize: base.hiddenSize,
            numLayers: base.numLayers,
            numHeads: base.numHeads,
            numKVHeads: base.numKVHeads,
            numFullKVHeads: base.numFullKVHeads,
            headDim: base.headDim,
            fullHeadDim: base.fullHeadDim,
            intermediateSize: base.intermediateSize,
            // The target is 128 wide; this drafter was trained against a wider one.
            backboneHiddenSize: 256,
            vocabSize: base.vocabSize,
            slidingWindow: base.slidingWindow,
            ropeTheta: base.ropeTheta,
            fullRopeTheta: base.fullRopeTheta,
            partialRotaryFactor: base.partialRotaryFactor,
            rmsNormEps: base.rmsNormEps,
            hiddenActivation: base.hiddenActivation,
            tieWordEmbeddings: base.tieWordEmbeddings,
            attentionKEqV: base.attentionKEqV,
            fullAttentionLayerMask: base.fullAttentionLayerMask,
            quantBits: base.quantBits,
            quantGroupSize: base.quantGroupSize,
            quantMode: base.quantMode)
        let repo = try SyntheticDraftRepo.build(config: mismatched)
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        await expectInstallRefusal(options(snapshotDir: snapshotDir,
                                           outputDir: outputDir,
                                           includeDraft: true,
                                           pin: repo.pin),
                                   because: "backbone_hidden_size is 256")
    }

    /// The inventory is derived from the pinned config, so a source that is
    /// short one tensor cannot be installed with a plausible-looking index.
    @Test func refusesADrafterMissingATensor() async throws {
        let snapshotDir = tmpDirForRemote("draft-partial-snapshot")
        let outputDir = tmpPathForRemote("draft-partial-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpRemote([outputDir])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticDraftRepo.build()
        resetFakeHF()
        defer { resetFakeHF() }
        var files = repo.files
        files["model.safetensors"] = try SyntheticDraftRepo.shardDropping(
            tensor: "model.layers.1.mlp.down_proj.scales",
            from: try #require(files["model.safetensors"]))
        FakeHFURLProtocol.repoFiles[repo.repoID] = files

        await expectInstallRefusal(options(snapshotDir: snapshotDir,
                                           outputDir: outputDir,
                                           includeDraft: true,
                                           pin: repo.pin),
                                   because: "missing: layers.1.mlp.down_proj.scales")
    }
}
