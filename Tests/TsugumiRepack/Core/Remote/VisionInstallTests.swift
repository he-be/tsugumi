import Foundation
import Testing
@testable import MoEPackFormat
@testable import TsugumiRepackCore

@Suite(.serialized)
struct VisionInstallTests {

    /// The staged-snapshot path copies tokenizer sidecars, so a snapshot under
    /// test needs the required ones on disk.
    static func buildStagedSnapshot(at dir: String) throws {
        _ = try SyntheticSnapshot.build(at: dir)
        for (name, contents) in [
            ("tokenizer.json", #"{"model":{"type":"BPE"}}"#),
            ("tokenizer_config.json", #"{"tokenizer_class":"PreTrainedTokenizerFast"}"#),
        ] {
            try Data(contents.utf8).write(
                to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent(name)))
        }
    }

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
                         includeVision: Bool,
                         pin: VisionSourcePin) -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            sourceSnapshotDirectory: snapshotDir,
            outputDir: outputDir,
            requireKnownSource: false,
            rangeChunkBytes: 4096,
            minFreeReserveBytes: 0,
            overwrite: true,
            includeVision: includeVision,
            visionPin: pin,
            downloadSession: fakeHFSession(),
            baseURL: URL(string: "https://hf.test")!,
            retryBaseDelayNs: 0)
    }

    @Test func installsTheVisionTowerAlongsideTheTextWeights() async throws {
        let snapshotDir = tmpDirForRemote("vision-snapshot")
        let outputDir = tmpPathForRemote("vision-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpRemote([outputDir])
        }
        try Self.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticVisionRepo.build(textSnapshotDir: snapshotDir)
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: options(snapshotDir: snapshotDir,
                             outputDir: outputDir,
                             includeVision: true,
                             pin: repo.pin)).run()

        // 1. The manifest declares the tower, its flag, and the bumped minor.
        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (outputDir as NSString).appendingPathComponent("manifest.json")))
        let manifest = try MoEPackManifestCodec.decode(manifestData)
        #expect(manifest.flags["visionTower"] == true)
        #expect(manifest.versionMinor == MoEPackFormatV1.versionMinorVision)
        let vision = try #require(manifest.vision)
        #expect(vision.tensorCount == repo.pin.expectedTensorCount)
        #expect(vision.payloadBytes == repo.pin.expectedPayloadBytes)
        #expect(vision.sourceRepo == repo.repoID)
        #expect(vision.sourceRevision == repo.revision)
        #expect(vision.weightsPath == MoEPackFormatV1.visionWeightsPath)
        #expect(vision.hiddenSize == SyntheticVisionRepo.config.hiddenSize)

        // 2. The file exists at the declared size and digest.
        let weightsPath = (outputDir as NSString)
            .appendingPathComponent(MoEPackFormatV1.visionWeightsPath)
        let entry = try #require(manifest.files[MoEPackFormatV1.visionWeightsPath])
        let weights = try Data(contentsOf: URL(fileURLWithPath: weightsPath))
        #expect(UInt64(weights.count) == entry.size)

        // 3. Its index names every tower tensor, and the bytes behind one of
        //    them are the source bytes — not merely the right length.
        let indexEntries = try weights.withUnsafeBytes { raw -> [MoEPackResidentIndexEntryV1] in
            let header = try MoEPackResidentIndexCodec.decodeHeader(raw)
            return try MoEPackResidentIndexCodec.decodeRegion(raw, header: header)
        }
        #expect(indexEntries.count == repo.pin.expectedTensorCount)
        let byName = Dictionary(uniqueKeysWithValues: indexEntries.map { ($0.name, $0) })
        #expect(byName["vision_tower.patch_embedder.input_proj.weight"] != nil)
        #expect(byName["embed_vision.embedding_projection.weight"] != nil)
        for name in ["vision_tower.encoder.layers.1.mlp.down_proj.linear.weight",
                     "vision_tower.std_scale",
                     "embed_vision.embedding_projection.weight"] {
            let e = try #require(byName[name])
            let expectedBytes = try #require(repo.expectedTensorBytes[name])
            #expect(UInt64(expectedBytes.count) == e.sizeBytes)
            let start = Int(e.fileOffset)
            #expect(weights.subdata(in: start..<(start + expectedBytes.count))
                    == expectedBytes)
            #expect(e.dtype == MoEPackFormatV1.DType.bf16.rawValue)
        }

        // 4. The whole install still verifies, tower included.
        let result = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputMoEPack: outputDir))
        #expect(result.unexpectedEntries.isEmpty)
        #expect(result.bytesVerified > entry.size)
    }

    @Test func textOnlyInstallWritesNoVisionArtifacts() async throws {
        let snapshotDir = tmpDirForRemote("vision-off-snapshot")
        let outputDir = tmpPathForRemote("vision-off-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpRemote([outputDir])
        }
        try Self.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticVisionRepo.build(textSnapshotDir: snapshotDir)
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: options(snapshotDir: snapshotDir,
                             outputDir: outputDir,
                             includeVision: false,
                             pin: repo.pin)).run()

        #expect(!FileManager.default.fileExists(
            atPath: (outputDir as NSString).appendingPathComponent("vision")))
        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (outputDir as NSString).appendingPathComponent("manifest.json")))
        let root = try #require(
            JSONSerialization.jsonObject(with: manifestData) as? [String: Any])
        #expect(root["vision"] == nil)
        #expect(root["versionMinor"] as? Int == MoEPackFormatV1.versionMinor)
        let flags = try #require(root["flags"] as? [String: Any])
        #expect(flags["visionTower"] == nil)
        // Not one request went to the vision repository.
        #expect(FakeHFURLProtocol.requestCounts.isEmpty)
    }

    /// The guarantee from `PLAN_VISION.md` §1-2: a tower may only be paired
    /// with the text weights it derives from. Here the tower is internally
    /// consistent — it matches its own pin — but was built from a different
    /// checkpoint, which only the comparison against the text weights can see.
    @Test func refusesATowerWhoseCheckpointDisagreesWithTheTextWeights() async throws {
        let snapshotDir = tmpDirForRemote("vision-parity-snapshot")
        let otherSnapshotDir = tmpDirForRemote("vision-parity-other")
        let outputDir = tmpPathForRemote("vision-parity-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: otherSnapshotDir)
            cleanUpRemote([outputDir])
        }
        try Self.buildStagedSnapshot(at: snapshotDir)
        _ = try SyntheticSnapshot.build(at: otherSnapshotDir, seed: 0x0123_4567_89AB_CDEF)
        let repo = try SyntheticVisionRepo.build(textSnapshotDir: otherSnapshotDir)
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        await expectInstallRefusal(options(snapshotDir: snapshotDir,
                                           outputDir: outputDir,
                                           includeVision: true,
                                           pin: repo.pin),
                                   because: "the two checkpoints differ")
        #expect(!FileManager.default.fileExists(atPath: outputDir))
    }

    @Test func refusesAVisionRepositoryWhoseIndexIsNotThePinnedOne() async throws {
        let snapshotDir = tmpDirForRemote("vision-pin-snapshot")
        let outputDir = tmpPathForRemote("vision-pin-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpRemote([outputDir])
        }
        try Self.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticVisionRepo.build(textSnapshotDir: snapshotDir)
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files
        let wrongPin = VisionSourcePin(
            repoID: repo.pin.repoID,
            revision: repo.pin.revision,
            indexSha256Hex: String(repeating: "a", count: 64),
            displayName: repo.pin.displayName,
            tensorPrefixes: repo.pin.tensorPrefixes,
            strippedNamePrefix: repo.pin.strippedNamePrefix,
            expectedTensorCount: repo.pin.expectedTensorCount,
            expectedPayloadBytes: repo.pin.expectedPayloadBytes,
            parityTensors: repo.pin.parityTensors,
            config: repo.pin.config)

        await expectInstallRefusal(options(snapshotDir: snapshotDir,
                                           outputDir: outputDir,
                                           includeVision: true,
                                           pin: wrongPin),
                                   because: "source fingerprint")
    }

    /// The tower inventory is derived from the pinned config, so a source that
    /// is short one tensor cannot be installed with a plausible-looking index.
    @Test func refusesAnIncompleteTower() async throws {
        let snapshotDir = tmpDirForRemote("vision-partial-snapshot")
        let outputDir = tmpPathForRemote("vision-partial-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpRemote([outputDir])
        }
        try Self.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticVisionRepo.build(textSnapshotDir: snapshotDir)
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files
        // Claim one more layer than the repository actually ships.
        let stretched = VisionSourceConfig(
            hiddenSize: SyntheticVisionRepo.config.hiddenSize,
            numLayers: SyntheticVisionRepo.config.numLayers + 1,
            numHeads: SyntheticVisionRepo.config.numHeads,
            numKVHeads: SyntheticVisionRepo.config.numKVHeads,
            headDim: SyntheticVisionRepo.config.headDim,
            intermediateSize: SyntheticVisionRepo.config.intermediateSize,
            patchSize: SyntheticVisionRepo.config.patchSize,
            poolingKernelSize: SyntheticVisionRepo.config.poolingKernelSize,
            positionEmbeddingSize: SyntheticVisionRepo.config.positionEmbeddingSize,
            ropeTheta: SyntheticVisionRepo.config.ropeTheta,
            rmsNormEps: SyntheticVisionRepo.config.rmsNormEps,
            hiddenActivation: SyntheticVisionRepo.config.hiddenActivation,
            standardize: SyntheticVisionRepo.config.standardize,
            maxSoftTokens: SyntheticVisionRepo.config.maxSoftTokens,
            imageTokenID: SyntheticVisionRepo.config.imageTokenID,
            boiTokenID: SyntheticVisionRepo.config.boiTokenID,
            eoiTokenID: SyntheticVisionRepo.config.eoiTokenID)
        let wrongPin = VisionSourcePin(
            repoID: repo.pin.repoID,
            revision: repo.pin.revision,
            indexSha256Hex: repo.pin.indexSha256Hex,
            displayName: repo.pin.displayName,
            tensorPrefixes: repo.pin.tensorPrefixes,
            strippedNamePrefix: repo.pin.strippedNamePrefix,
            expectedTensorCount: repo.pin.expectedTensorCount + 13,
            expectedPayloadBytes: repo.pin.expectedPayloadBytes,
            parityTensors: repo.pin.parityTensors,
            config: stretched)

        // The config it declares is checked against the repository's own
        // config.json first, which is where a stretched pin gets caught.
        await expectInstallRefusal(options(snapshotDir: snapshotDir,
                                           outputDir: outputDir,
                                           includeVision: true,
                                           pin: wrongPin),
                                   because: "does not match the pinned tower")
    }

    /// Same shortfall, but with the pin and the repository agreeing on the
    /// config: the missing tensors have to be caught by the inventory itself.
    @Test func refusesATowerWhoseTensorsDoNotMatchItsDeclaredConfig() async throws {
        let snapshotDir = tmpDirForRemote("vision-inventory-snapshot")
        let outputDir = tmpPathForRemote("vision-inventory-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpRemote([outputDir])
        }
        try Self.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticVisionRepo.build(textSnapshotDir: snapshotDir)
        resetFakeHF()
        defer { resetFakeHF() }
        var files = repo.files
        // Drop one tower tensor from the shard's header, leaving everything
        // else — including config.json and the index digest — intact.
        files["model-00001-of-00001.safetensors"] = try SyntheticVisionRepo.shardDropping(
            tensor: "model.vision_tower.std_bias",
            from: try #require(files["model-00001-of-00001.safetensors"]))
        FakeHFURLProtocol.repoFiles[repo.repoID] = files

        await expectInstallRefusal(options(snapshotDir: snapshotDir,
                                           outputDir: outputDir,
                                           includeVision: true,
                                           pin: repo.pin),
                                   because: "missing: vision_tower.std_bias")
    }
}
