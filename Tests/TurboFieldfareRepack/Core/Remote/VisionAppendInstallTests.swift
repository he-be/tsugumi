import Darwin
import Foundation
import Testing
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

/// `--add-vision` on a model that is already installed. The property that makes
/// it worth having is not "a tower appears" but "nothing else moves": the text
/// weights are not re-downloaded, not rewritten, and not even reopened for
/// writing, and what is left behind is indistinguishable from an install that
/// had `--include-vision` from the start.
@Suite(.serialized)
struct VisionAppendInstallTests {

    private func installOptions(snapshotDir: String,
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

    private func addOptions(outputDir: String,
                            pin: VisionSourcePin) -> AddVisionOptions {
        AddVisionOptions(
            inputGTurbo: outputDir,
            requireKnownSource: false,
            rangeChunkBytes: 4096,
            minFreeReserveBytes: 0,
            downloadSession: fakeHFSession(),
            baseURL: URL(string: "https://hf.test")!,
            retryBaseDelayNs: 0,
            visionPin: pin)
    }

    private func expectRefusal(_ options: AddVisionOptions,
                               because reason: String) async {
        do {
            _ = try await VisionAppendInstaller(options: options).run()
            Issue.record("add-vision succeeded; expected a refusal mentioning \(reason)")
        } catch {
            #expect("\(error)".contains(reason), "unexpected refusal: \(error)")
        }
    }

    private func read(_ dir: String, _ relativePath: String) throws -> Data {
        try Data(contentsOf: URL(fileURLWithPath:
            (dir as NSString).appendingPathComponent(relativePath)))
    }

    /// Identity of the file on disk, not just its contents: a re-copied
    /// `model_weights.bin` would have the same bytes but a new inode.
    private func identity(of path: String)
        throws -> (ino: UInt64, seconds: Int, nanoseconds: Int, size: Int64) {
        var info = stat()
        guard stat(path, &info) == 0 else {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        return (info.st_ino, info.st_mtimespec.tv_sec, info.st_mtimespec.tv_nsec,
                info.st_size)
    }

    private func cleanUpAppend(_ paths: [String]) {
        cleanUpRemote(paths)
        for path in paths {
            try? FileManager.default.removeItem(atPath: path + ".vision.partial")
            try? FileManager.default.removeItem(atPath: path + ".resume.json")
        }
    }

    @Test func addingTheTowerMatchesAFreshVisionInstall() async throws {
        let snapshotDir = tmpDirForRemote("append-snapshot")
        let appended = tmpPathForRemote("append-target")
        let fresh = tmpPathForRemote("append-reference")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpAppend([appended, fresh])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticVisionRepo.build(textSnapshotDir: snapshotDir)
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: installOptions(snapshotDir: snapshotDir,
                                    outputDir: appended,
                                    includeVision: false,
                                    pin: repo.pin)).run()
        _ = try await RemoteStreamingRepacker(
            options: installOptions(snapshotDir: snapshotDir,
                                    outputDir: fresh,
                                    includeVision: true,
                                    pin: repo.pin)).run()

        let textPath = (appended as NSString).appendingPathComponent("model_weights.bin")
        let expertPath = (appended as NSString)
            .appendingPathComponent("packed_experts/layout.json")
        let textBefore = try identity(of: textPath)
        let expertsBefore = try identity(of: expertPath)

        let result = try await VisionAppendInstaller(
            options: addOptions(outputDir: appended, pin: repo.pin)).run()

        // 1. The tower is there, and it is the same tower.
        #expect(result.tensorCount == repo.pin.expectedTensorCount)
        #expect(result.payloadBytes == repo.pin.expectedPayloadBytes)
        #expect(result.visionRepoID == repo.repoID)
        #expect(try read(appended, GTurboFormatV1.visionWeightsPath)
                == (try read(fresh, GTurboFormatV1.visionWeightsPath)))

        // 2. The manifest is byte-for-byte the one a fresh --include-vision
        //    install writes. That covers the flag, the minor version, the
        //    vision section and the files table in one comparison.
        #expect(try read(appended, "manifest.json") == (try read(fresh, "manifest.json")))
        let manifest = try GTurboManifestCodec.decode(try read(appended, "manifest.json"))
        #expect(manifest.flags["visionTower"] == true)
        #expect(manifest.versionMinor == GTurboFormatV1.versionMinorVision)
        #expect(manifest.files[GTurboFormatV1.visionWeightsPath]?.size
                == result.weightsFileBytes)

        // 3. Nothing on the text side was rewritten — same inode, same mtime,
        //    same size. This is the whole point of the mode.
        let textAfter = try identity(of: textPath)
        #expect(textAfter == textBefore)
        #expect(try identity(of: expertPath) == expertsBefore)
        // …and that comparison can tell a rewrite from an untouched file: the
        // reference install wrote the same bytes and gets a different identity.
        #expect(try identity(of: (fresh as NSString)
            .appendingPathComponent("model_weights.bin")).ino != textBefore.ino)
        #expect(try read(appended, "model_weights.bin")
                == (try read(fresh, "model_weights.bin")))

        // 4. The install still verifies, and the receipt it leaves covers the
        //    tower, so `--verification trusted-install` keeps working.
        #expect(result.unexpectedEntries.isEmpty)
        let receipt = try JSONSerialization.jsonObject(
            with: try read(appended, VerifiedInstallReceiptWriter.fileName)) as! [String: Any]
        let receiptFiles = try #require(receipt["files"] as? [String: Any])
        #expect(receiptFiles[GTurboFormatV1.visionWeightsPath] != nil)
        let manifestSha = try #require(receipt["manifestSha256"] as? String)
        let actualManifestSha = try Sha256Stream.hashFile(path:
            (appended as NSString).appendingPathComponent("manifest.json"))
        #expect(manifestSha == actualManifestSha)
        let verified = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: appended))
        #expect(verified.unexpectedEntries.isEmpty)

        // 5. Nothing was staged inside the model directory, and the staging
        //    directory beside it is gone.
        try assertNoInternalRemoteDirs(outputDir: appended)
        #expect(!FileManager.default.fileExists(atPath: appended + ".vision.partial"))
    }

    @Test func refusesAModelThatAlreadyHasATower() async throws {
        let snapshotDir = tmpDirForRemote("append-twice-snapshot")
        let outputDir = tmpPathForRemote("append-twice-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpAppend([outputDir])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticVisionRepo.build(textSnapshotDir: snapshotDir)
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: installOptions(snapshotDir: snapshotDir,
                                    outputDir: outputDir,
                                    includeVision: true,
                                    pin: repo.pin)).run()
        let before = try read(outputDir, "manifest.json")

        await expectRefusal(addOptions(outputDir: outputDir, pin: repo.pin),
                            because: "already has a vision tower")
        #expect(try read(outputDir, "manifest.json") == before)
    }

    /// The parity evidence of `PLAN_VISION.md` §1-2 has to survive the loss of
    /// the source snapshot: here it is measured against the installed
    /// `model_weights.bin` instead, and still catches a tower from a different
    /// checkpoint — one that agrees with its own pin.
    @Test func refusesATowerWhoseCheckpointDisagreesWithTheInstalledWeights() async throws {
        let snapshotDir = tmpDirForRemote("append-parity-snapshot")
        let otherSnapshotDir = tmpDirForRemote("append-parity-other")
        let outputDir = tmpPathForRemote("append-parity-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: otherSnapshotDir)
            cleanUpAppend([outputDir])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        _ = try SyntheticSnapshot.build(at: otherSnapshotDir, seed: 0x0123_4567_89AB_CDEF)
        let repo = try SyntheticVisionRepo.build(textSnapshotDir: otherSnapshotDir)
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: installOptions(snapshotDir: snapshotDir,
                                    outputDir: outputDir,
                                    includeVision: false,
                                    pin: repo.pin)).run()
        let before = try read(outputDir, "manifest.json")

        await expectRefusal(addOptions(outputDir: outputDir, pin: repo.pin),
                            because: "the two checkpoints differ")

        // The model is exactly as text-only as it was.
        #expect(try read(outputDir, "manifest.json") == before)
        #expect(!FileManager.default.fileExists(
            atPath: (outputDir as NSString).appendingPathComponent("vision")))
        #expect(!FileManager.default.fileExists(atPath: outputDir + ".vision.partial"))
        let verified = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: outputDir))
        #expect(verified.unexpectedEntries.isEmpty)
    }

    /// A refusal after the staging directory exists must not leave debris —
    /// neither inside the model (where `--verify-install` would report it) nor
    /// beside it (where the next run would trip over it).
    @Test func aRefusedAppendLeavesNothingBehind() async throws {
        let snapshotDir = tmpDirForRemote("append-clean-snapshot")
        let outputDir = tmpPathForRemote("append-clean-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpAppend([outputDir])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticVisionRepo.build(textSnapshotDir: snapshotDir)
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: installOptions(snapshotDir: snapshotDir,
                                    outputDir: outputDir,
                                    includeVision: false,
                                    pin: repo.pin)).run()
        let before = try read(outputDir, "manifest.json")
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

        await expectRefusal(addOptions(outputDir: outputDir, pin: wrongPin),
                            because: "source fingerprint")

        #expect(try read(outputDir, "manifest.json") == before)
        #expect(!FileManager.default.fileExists(atPath: outputDir + ".vision.partial"))
        let verified = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: outputDir))
        #expect(verified.unexpectedEntries.isEmpty)
    }

    /// The tower is added to a specific model, so an install that is mid-flight
    /// under the same name has to be settled first — otherwise the resumed
    /// install would promote text weights the tower was never checked against.
    @Test func refusesWhileAnInterruptedInstallOwnsTheName() async throws {
        let snapshotDir = tmpDirForRemote("append-busy-snapshot")
        let outputDir = tmpPathForRemote("append-busy-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpAppend([outputDir])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticVisionRepo.build(textSnapshotDir: snapshotDir)
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: installOptions(snapshotDir: snapshotDir,
                                    outputDir: outputDir,
                                    includeVision: false,
                                    pin: repo.pin)).run()
        try FileManager.default.createDirectory(atPath: outputDir + ".partial",
                                                withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: URL(fileURLWithPath: outputDir + ".resume.json"))

        await expectRefusal(addOptions(outputDir: outputDir, pin: repo.pin),
                            because: "an interrupted install exists")
    }
}
