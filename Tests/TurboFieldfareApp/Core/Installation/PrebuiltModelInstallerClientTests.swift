import CryptoKit
import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// The prebuilt installer against a synthetic two-file source: reuse of
/// verified files, download with hash verification, range resume, corrupt
/// downloads, and the receipt it leaves behind.
@Suite struct PrebuiltModelInstallerClientTests {
    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static let manifestData = Data("{\"arch\": {}}".utf8)
    private static let weightsData = Data(repeating: 0xAB, count: 1_000)

    private static func makeSource() -> PrebuiltModelSource {
        PrebuiltModelSource(
            kind: .gemmaQATSym,
            repoID: "example/prebuilt",
            revision: "main",
            sourceIndexSHA256: String(repeating: "a", count: 64),
            files: [
                PrebuiltFileEntry(path: "manifest.json",
                                  bytes: UInt64(manifestData.count),
                                  sha256: sha256(manifestData)),
                PrebuiltFileEntry(path: "weights/model.bin",
                                  bytes: UInt64(weightsData.count),
                                  sha256: sha256(weightsData)),
            ])
    }

    private static func makeOpener(
        bodies: [String: Data],
        onOpen: (@Sendable (URL, UInt64) -> Void)? = nil
    ) -> PrebuiltModelInstallerClient.DownloadOpener {
        { url, resumeFrom in
            onOpen?(url, resumeFrom)
            guard let full = bodies[url.lastPathComponent] else {
                throw PrebuiltInstallError.badResponse(
                    path: url.lastPathComponent, detail: "no fixture")
            }
            let body = full.dropFirst(Int(resumeFrom))
            let (stream, continuation) = AsyncThrowingStream<Data, Error>.makeStream()
            continuation.yield(Data(body.prefix(body.count / 2)))
            continuation.yield(Data(body.dropFirst(body.count / 2)))
            continuation.finish()
            return (resumeFrom > 0 ? 206 : 200, stream)
        }
    }

    private func temporaryDirectory(_ label: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("prebuilt-\(label)-\(UUID().uuidString).gturbo")
    }

    private func runInstall(_ client: PrebuiltModelInstallerClient,
                            outputDirectory: URL) async throws -> [AppModelInstallEvent] {
        var events: [AppModelInstallEvent] = []
        for try await event in client.installDefaultModel(outputDirectory: outputDirectory) {
            events.append(event)
        }
        return events
    }

    @Test func downloadsVerifiesRenamesAndWritesReceipt() async throws {
        let source = Self.makeSource()
        let directory = temporaryDirectory("fresh")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = PrebuiltModelInstallerClient(
            source: source,
            openDownload: Self.makeOpener(bodies: [
                "manifest.json": Self.manifestData,
                "model.bin": Self.weightsData,
            ]))

        let events = try await runInstall(client, outputDirectory: directory)

        guard case .installed(let installed) = events.last else {
            Issue.record("expected installed, got \(String(describing: events.last))")
            return
        }
        #expect(installed == directory.standardizedFileURL)
        #expect(try Data(contentsOf: directory.appendingPathComponent("manifest.json"))
            == Self.manifestData)
        #expect(try Data(contentsOf: directory.appendingPathComponent("weights/model.bin"))
            == Self.weightsData)
        #expect(!client.hasPartialInstall(outputDirectory: directory))

        let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
        #expect(receipt.manifestSha256 == Self.sha256(Self.manifestData))
        #expect(receipt.modelDirectoryPath == directory.standardizedFileURL.path)
        #expect(receipt.sourceRepoID == "example/prebuilt")
        #expect(receipt.files.count == 2)
        #expect(receipt.files["weights/model.bin"]?.sha256
            == Self.sha256(Self.weightsData))

        let payloads = events.compactMap { event -> UInt64? in
            guard case .copyingPayload(_, let downloaded, let total) = event else { return nil }
            #expect(total == source.totalBytes)
            return downloaded
        }
        #expect(payloads.last == source.totalBytes)
    }

    @Test func reusesCompleteFilesWithoutDownloading() async throws {
        let source = Self.makeSource()
        let directory = temporaryDirectory("reuse")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("weights"),
            withIntermediateDirectories: true)
        try Self.manifestData.write(to: directory.appendingPathComponent("manifest.json"))
        try Self.weightsData.write(to: directory.appendingPathComponent("weights/model.bin"))

        let client = PrebuiltModelInstallerClient(
            source: source,
            openDownload: { url, _ in
                Issue.record("unexpected download of \(url)")
                throw PrebuiltInstallError.badResponse(path: url.path, detail: "offline")
            })

        let events = try await runInstall(client, outputDirectory: directory)
        guard case .installed = events.last else {
            Issue.record("expected installed, got \(String(describing: events.last))")
            return
        }
        let reused = events.compactMap { event -> UInt64? in
            guard case .copyingPayload(let reused, _, _) = event else { return nil }
            return reused
        }
        #expect(reused.first == source.totalBytes)
    }

    @Test func resumeFoldsPartIntoHashAndRequestsRange() async throws {
        let source = Self.makeSource()
        let directory = temporaryDirectory("resume")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("weights"),
            withIntermediateDirectories: true)
        try Self.manifestData.write(to: directory.appendingPathComponent("manifest.json"))
        let half = Self.weightsData.prefix(400)
        try Data(half).write(
            to: directory.appendingPathComponent("weights/model.bin.part"))

        let observed = ObservedOpens()
        let client = PrebuiltModelInstallerClient(
            source: source,
            openDownload: Self.makeOpener(
                bodies: ["model.bin": Self.weightsData],
                onOpen: { url, resumeFrom in
                    observed.append((url.lastPathComponent, resumeFrom))
                }))
        #expect(client.hasPartialInstall(outputDirectory: directory))

        let events = try await runInstall(client, outputDirectory: directory)
        guard case .installed = events.last else {
            Issue.record("expected installed")
            return
        }
        #expect(observed.snapshot() == [Pair("model.bin", 400)])
        #expect(try Data(contentsOf: directory.appendingPathComponent("weights/model.bin"))
            == Self.weightsData)
    }

    @Test func corruptDownloadFailsAndDropsThePart() async throws {
        let source = Self.makeSource()
        let directory = temporaryDirectory("corrupt")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
        try Self.manifestData.write(to: directory.appendingPathComponent("manifest.json"))
        var corrupted = Self.weightsData
        corrupted[0] = 0xCD

        let client = PrebuiltModelInstallerClient(
            source: source,
            openDownload: Self.makeOpener(bodies: ["model.bin": corrupted]))

        await #expect(throws: PrebuiltInstallError.hashMismatch(path: "weights/model.bin")) {
            _ = try await runInstall(client, outputDirectory: directory)
        }
        #expect(!FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("weights/model.bin.part").path))
        #expect(!FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("weights/model.bin").path))
    }

    @Test func discardRemovesPartsAndKeepsVerifiedFiles() async throws {
        let source = Self.makeSource()
        let directory = temporaryDirectory("discard")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("weights"),
            withIntermediateDirectories: true)
        try Self.manifestData.write(to: directory.appendingPathComponent("manifest.json"))
        try Data([1, 2, 3]).write(
            to: directory.appendingPathComponent("weights/model.bin.part"))

        let client = PrebuiltModelInstallerClient(
            source: source,
            openDownload: Self.makeOpener(bodies: [:]))
        try await client.discardPartialInstall(outputDirectory: directory)

        #expect(!client.hasPartialInstall(outputDirectory: directory))
        #expect(FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("manifest.json").path))
    }

    @Test func requirementCountsOnlyMissingBytes() throws {
        let source = Self.makeSource()
        let directory = temporaryDirectory("requirement")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
        try Self.manifestData.write(to: directory.appendingPathComponent("manifest.json"))

        let client = PrebuiltModelInstallerClient(
            source: source,
            openDownload: Self.makeOpener(bodies: [:]))
        let requirement = try client.checkInstallRequirement(outputDirectory: directory)
        #expect(requirement.requiredBytes
            == UInt64(Self.weightsData.count) + client.descriptor.reserveBytes)
    }
}

private struct Pair: Equatable {
    let name: String
    let resumeFrom: UInt64

    init(_ name: String, _ resumeFrom: UInt64) {
        self.name = name
        self.resumeFrom = resumeFrom
    }
}

private final class ObservedOpens: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Pair] = []

    func append(_ value: (String, UInt64)) {
        lock.lock()
        values.append(Pair(value.0, value.1))
        lock.unlock()
    }

    func snapshot() -> [Pair] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}
