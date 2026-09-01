import Darwin
import Foundation
import Testing
@testable import MoEPackFormat
@testable import TsugumiRepackCore

/// `--add-draft` on a model that is already installed. As with `--add-vision`,
/// the property worth having is not "a drafter appears" but "nothing else
/// moves": the 15 GB of text weights are not re-downloaded, not rewritten, and
/// not even reopened for writing.
@Suite(.serialized)
struct DraftAppendInstallTests {

    private func installOptions(snapshotDir: String,
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

    private func addOptions(outputDir: String,
                            pin: DraftSourcePin) -> AddDraftOptions {
        AddDraftOptions(
            inputMoEPack: outputDir,
            requireKnownSource: false,
            rangeChunkBytes: 4096,
            minFreeReserveBytes: 0,
            downloadSession: fakeHFSession(),
            baseURL: URL(string: "https://hf.test")!,
            retryBaseDelayNs: 0,
            draftPin: pin)
    }

    private func expectRefusal(_ options: AddDraftOptions,
                               because reason: String) async {
        do {
            _ = try await DraftAppendInstaller(options: options).run()
            Issue.record("add-draft succeeded; expected a refusal mentioning \(reason)")
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
            try? FileManager.default.removeItem(atPath: path + ".draft.partial")
            try? FileManager.default.removeItem(atPath: path + ".resume.json")
        }
    }

    @Test func addingTheDrafterMatchesAFreshDraftInstall() async throws {
        let snapshotDir = tmpDirForRemote("draft-append-snapshot")
        let appended = tmpPathForRemote("draft-append-target")
        let fresh = tmpPathForRemote("draft-append-reference")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpAppend([appended, fresh])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticDraftRepo.build()
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: installOptions(snapshotDir: snapshotDir,
                                    outputDir: appended,
                                    includeDraft: false,
                                    pin: repo.pin)).run()
        _ = try await RemoteStreamingRepacker(
            options: installOptions(snapshotDir: snapshotDir,
                                    outputDir: fresh,
                                    includeDraft: true,
                                    pin: repo.pin)).run()

        let textPath = (appended as NSString).appendingPathComponent("model_weights.bin")
        let expertPath = (appended as NSString)
            .appendingPathComponent("packed_experts/layout.json")
        let textBefore = try identity(of: textPath)
        let expertsBefore = try identity(of: expertPath)

        let result = try await DraftAppendInstaller(
            options: addOptions(outputDir: appended, pin: repo.pin)).run()

        // 1. The drafter is there, and it is the same drafter.
        #expect(result.tensorCount == repo.pin.expectedTensorCount)
        #expect(result.payloadBytes == repo.pin.expectedPayloadBytes)
        #expect(result.draftRepoID == repo.repoID)
        #expect(try read(appended, MoEPackFormatV1.draftWeightsPath)
                == (try read(fresh, MoEPackFormatV1.draftWeightsPath)))

        // 2. The manifest is byte-for-byte the one a fresh --include-draft
        //    install writes: flag, minor version, draft section and files table
        //    in one comparison.
        #expect(try read(appended, "manifest.json") == (try read(fresh, "manifest.json")))
        let manifest = try MoEPackManifestCodec.decode(try read(appended, "manifest.json"))
        #expect(manifest.flags["mtpDraft"] == true)
        #expect(manifest.versionMinor == MoEPackFormatV1.versionMinorDraft)
        #expect(manifest.files[MoEPackFormatV1.draftWeightsPath]?.size
                == result.weightsFileBytes)

        // 3. Nothing on the text side was rewritten — same inode, same mtime,
        //    same size. This is the whole point of the mode.
        let textAfter = try identity(of: textPath)
        #expect(textAfter == textBefore)
        #expect(try identity(of: expertPath) == expertsBefore)
        #expect(try identity(of: (fresh as NSString)
            .appendingPathComponent("model_weights.bin")).ino != textBefore.ino)
        #expect(try read(appended, "model_weights.bin")
                == (try read(fresh, "model_weights.bin")))

        // 4. The install still verifies, and the receipt covers the drafter.
        #expect(result.unexpectedEntries.isEmpty)
        let receipt = try JSONSerialization.jsonObject(
            with: try read(appended, VerifiedInstallReceiptWriter.fileName)) as! [String: Any]
        let receiptFiles = try #require(receipt["files"] as? [String: Any])
        #expect(receiptFiles[MoEPackFormatV1.draftWeightsPath] != nil)
        let manifestSha = try #require(receipt["manifestSha256"] as? String)
        #expect(manifestSha == (try Sha256Stream.hashFile(path:
            (appended as NSString).appendingPathComponent("manifest.json"))))
        let verified = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputMoEPack: appended))
        #expect(verified.unexpectedEntries.isEmpty)

        // 5. Nothing was staged inside the model directory, and the staging
        //    directory beside it is gone.
        try assertNoInternalRemoteDirs(outputDir: appended)
        #expect(!FileManager.default.fileExists(atPath: appended + ".draft.partial"))
    }

    /// A tower and a drafter have to be able to coexist: they are separate
    /// files, separate sections and separate flags, appended in either order.
    @Test func aDrafterCanBeAddedToAModelThatAlreadyHasATower() async throws {
        let snapshotDir = tmpDirForRemote("draft-with-vision-snapshot")
        let outputDir = tmpPathForRemote("draft-with-vision-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpAppend([outputDir])
            try? FileManager.default.removeItem(atPath: outputDir + ".vision.partial")
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let vision = try SyntheticVisionRepo.build(textSnapshotDir: snapshotDir)
        let draft = try SyntheticDraftRepo.build()
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[vision.repoID] = vision.files
        FakeHFURLProtocol.repoFiles[draft.repoID] = draft.files

        _ = try await RemoteStreamingRepacker(options: RemoteStreamingRepackOptions(
            sourceSnapshotDirectory: snapshotDir,
            outputDir: outputDir,
            requireKnownSource: false,
            rangeChunkBytes: 4096,
            minFreeReserveBytes: 0,
            overwrite: true,
            includeVision: true,
            visionPin: vision.pin,
            downloadSession: fakeHFSession(),
            baseURL: URL(string: "https://hf.test")!,
            retryBaseDelayNs: 0)).run()

        _ = try await DraftAppendInstaller(
            options: addOptions(outputDir: outputDir, pin: draft.pin)).run()

        let manifest = try MoEPackManifestCodec.decode(try read(outputDir, "manifest.json"))
        #expect(manifest.flags["visionTower"] == true)
        #expect(manifest.flags["mtpDraft"] == true)
        #expect(manifest.vision != nil)
        #expect(manifest.draft != nil)
        #expect(manifest.versionMinor == MoEPackFormatV1.versionMinorDraft)
        let verified = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputMoEPack: outputDir))
        #expect(verified.unexpectedEntries.isEmpty)
    }

    @Test func refusesAModelThatAlreadyHasADrafter() async throws {
        let snapshotDir = tmpDirForRemote("draft-twice-snapshot")
        let outputDir = tmpPathForRemote("draft-twice-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpAppend([outputDir])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticDraftRepo.build()
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: installOptions(snapshotDir: snapshotDir,
                                    outputDir: outputDir,
                                    includeDraft: true,
                                    pin: repo.pin)).run()
        let before = try read(outputDir, "manifest.json")

        await expectRefusal(addOptions(outputDir: outputDir, pin: repo.pin),
                            because: "already has an MTP drafter")
        #expect(try read(outputDir, "manifest.json") == before)
    }

    /// A refusal after the staging directory exists must not leave debris —
    /// neither inside the model (where `--verify-install` would report it) nor
    /// beside it (where the next run would trip over it).
    @Test func aRefusedAppendLeavesNothingBehind() async throws {
        let snapshotDir = tmpDirForRemote("draft-clean-snapshot")
        let outputDir = tmpPathForRemote("draft-clean-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpAppend([outputDir])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticDraftRepo.build()
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: installOptions(snapshotDir: snapshotDir,
                                    outputDir: outputDir,
                                    includeDraft: false,
                                    pin: repo.pin)).run()
        let before = try read(outputDir, "manifest.json")
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

        await expectRefusal(addOptions(outputDir: outputDir, pin: wrongPin),
                            because: "source fingerprint")

        #expect(try read(outputDir, "manifest.json") == before)
        #expect(!FileManager.default.fileExists(atPath: outputDir + ".draft.partial"))
        #expect(!FileManager.default.fileExists(
            atPath: (outputDir as NSString).appendingPathComponent("draft")))
        let verified = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputMoEPack: outputDir))
        #expect(verified.unexpectedEntries.isEmpty)
    }

    /// The drafter is added to a specific model, so an install that is mid-flight
    /// under the same name has to be settled first — otherwise the resumed
    /// install would promote text weights the drafter was never checked against.
    @Test func refusesWhileAnInterruptedInstallOwnsTheName() async throws {
        let snapshotDir = tmpDirForRemote("draft-busy-snapshot")
        let outputDir = tmpPathForRemote("draft-busy-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            cleanUpAppend([outputDir])
        }
        try VisionInstallTests.buildStagedSnapshot(at: snapshotDir)
        let repo = try SyntheticDraftRepo.build()
        resetFakeHF()
        defer { resetFakeHF() }
        FakeHFURLProtocol.repoFiles[repo.repoID] = repo.files

        _ = try await RemoteStreamingRepacker(
            options: installOptions(snapshotDir: snapshotDir,
                                    outputDir: outputDir,
                                    includeDraft: false,
                                    pin: repo.pin)).run()
        try FileManager.default.createDirectory(atPath: outputDir + ".partial",
                                                withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: URL(fileURLWithPath: outputDir + ".resume.json"))

        await expectRefusal(addOptions(outputDir: outputDir, pin: repo.pin),
                            because: "an interrupted install exists")
    }
}
