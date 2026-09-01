import CryptoKit
import Foundation
import Synchronization
import Tsugumi
import TsugumiRepackCore

/// Downloads a finished install published on Hugging Face, file for file.
///
/// Layout on disk while installing: every file streams into `<path>.part`
/// beside its final name, is SHA-256 verified against the pin, and only then
/// renamed into place — `manifest.json` last, because its presence is what
/// the installation probe treats as "there is a model here". A file already
/// at its final name with the pinned size is reused without re-download; the
/// artifacts this app measured were built in place on this machine, so a
/// developer checkout passes through this path with zero network traffic.
public final class PrebuiltModelInstallerClient: AppModelInstallerClient, Sendable {
    private struct ActiveInstall: Sendable {
        let id: UUID
        let task: Task<Void, Never>
    }

    private final class InstallTaskState: Sendable {
        let value = Mutex<ActiveInstall?>(nil)
    }

    public let source: PrebuiltModelSource
    public let descriptor: AppModelInstallDescriptor
    private let taskState = InstallTaskState()
    /// Injectable for tests: given a download URL and the byte to resume
    /// from, returns the HTTP status and the response body as Data chunks.
    typealias DownloadOpener = @Sendable (URL, UInt64) async throws
        -> (statusCode: Int, body: AsyncThrowingStream<Data, Error>)
    private let openDownload: DownloadOpener

    public convenience init(kind: AppModelKind) {
        self.init(source: PrebuiltModelSource.source(for: kind))
    }

    public convenience init(source: PrebuiltModelSource) {
        self.init(source: source, openDownload: { url, resumeFrom in
            var request = URLRequest(url: url)
            if resumeFrom > 0 {
                request.setValue("bytes=\(resumeFrom)-", forHTTPHeaderField: "Range")
            }
            if let token = ProcessInfo.processInfo.environment["HF_TOKEN"], !token.isEmpty {
                request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
            return try await HTTPChunkDownloader.open(request: request)
        })
    }

    init(source: PrebuiltModelSource, openDownload: @escaping DownloadOpener) {
        self.source = source
        self.descriptor = .descriptor(for: source.kind)
        self.openDownload = openDownload
    }

    // MARK: - Requirement

    public func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement {
        let reused = Self.reusableBytes(source: source, outputDirectory: outputDirectory)
        let remaining = source.totalBytes > reused ? source.totalBytes - reused : 0
        let requirement = try DiskSpaceChecker.assess(
            path: outputDirectory.path,
            bytes: remaining,
            reserveBytes: descriptor.reserveBytes)
        return AppModelInstallRequirement(probePath: requirement.path,
                                          requiredBytes: requirement.requiredBytes,
                                          availableBytes: requirement.availableBytes)
    }

    /// Bytes that will not have to be downloaded again: complete files at
    /// their final names with the pinned size, plus partial `.part` files.
    static func reusableBytes(source: PrebuiltModelSource, outputDirectory: URL) -> UInt64 {
        var reused: UInt64 = 0
        for entry in source.files {
            let final = outputDirectory.appendingPathComponent(entry.path)
            if let size = fileSize(at: final), size == entry.bytes {
                reused += entry.bytes
                continue
            }
            if let size = fileSize(at: partURL(for: final)), size <= entry.bytes {
                reused += size
            }
        }
        return reused
    }

    // MARK: - Install

    public func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        AsyncThrowingStream { continuation in
            let id = UUID()
            let task = Task { [source, openDownload] in
                do {
                    continuation.yield(.checking)
                    try await Self.run(source: source,
                                       outputDirectory: outputDirectory,
                                       openDownload: openDownload) { event in
                        continuation.yield(event)
                    }
                    try Task.checkCancellation()
                    continuation.yield(.installed(outputDirectory.standardizedFileURL))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            let previous = taskState.value.withLock { active in
                let previous = active?.task
                active = ActiveInstall(id: id, task: task)
                return previous
            }
            previous?.cancel()

            continuation.onTermination = { [taskState] _ in
                let task = taskState.value.withLock { active -> Task<Void, Never>? in
                    guard active?.id == id else { return nil }
                    defer { active = nil }
                    return active?.task
                }
                task?.cancel()
            }
        }
    }

    private static func run(source: PrebuiltModelSource,
                            outputDirectory: URL,
                            openDownload: DownloadOpener,
                            emit: @Sendable (AppModelInstallEvent) -> Void) async throws {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        // `manifest.json` renames into place last: its presence is the
        // "installed" signal, so it must not appear before every payload
        // file it describes has been verified.
        let ordered = source.files.sorted { a, b in
            (a.path == "manifest.json" ? 1 : 0, a.path) < (b.path == "manifest.json" ? 1 : 0, b.path)
        }
        let total = source.totalBytes
        var reused: UInt64 = 0
        var pending: [PrebuiltFileEntry] = []
        for entry in ordered {
            let final = outputDirectory.appendingPathComponent(entry.path)
            if let size = fileSize(at: final), size == entry.bytes {
                reused += entry.bytes
            } else {
                pending.append(entry)
            }
        }
        emit(.copyingPayload(reusedBytes: reused, downloadedThisRunBytes: 0, totalBytes: total))

        var downloadedThisRun: UInt64 = 0
        for entry in pending {
            try Task.checkCancellation()
            let final = outputDirectory.appendingPathComponent(entry.path)
            try fileManager.createDirectory(at: final.deletingLastPathComponent(),
                                            withIntermediateDirectories: true)
            let reusedSoFar = reused
            let downloadedBefore = downloadedThisRun
            let downloaded = try await download(entry: entry,
                                                source: source,
                                                to: final,
                                                openDownload: openDownload) { bytes in
                emit(.copyingPayload(reusedBytes: reusedSoFar,
                                     downloadedThisRunBytes: downloadedBefore + bytes,
                                     totalBytes: total))
            }
            downloadedThisRun += downloaded
            emit(.copyingPayload(reusedBytes: reused,
                                 downloadedThisRunBytes: downloadedThisRun,
                                 totalBytes: total))
        }

        emit(.finalizing)
        try writeReceipt(source: source, outputDirectory: outputDirectory)
    }

    /// Streams one file into `<final>.part`, hashing as it goes, resuming
    /// from whatever a previous run left. Returns the bytes fetched this run.
    private static func download(entry: PrebuiltFileEntry,
                                 source: PrebuiltModelSource,
                                 to final: URL,
                                 openDownload: DownloadOpener,
                                 onProgress: @Sendable (UInt64) -> Void) async throws -> UInt64 {
        let fileManager = FileManager.default
        let part = partURL(for: final)
        var hasher = SHA256()
        var resumeFrom: UInt64 = 0
        if let existing = fileSize(at: part), existing > 0, existing <= entry.bytes {
            // The pinned hash covers the whole file, so the bytes already on
            // disk are folded into the running hash before the range request.
            let handle = try FileHandle(forReadingFrom: part)
            defer { try? handle.close() }
            while let chunk = try handle.read(upToCount: 8 * 1_048_576), !chunk.isEmpty {
                hasher.update(data: chunk)
                resumeFrom += UInt64(chunk.count)
            }
        } else {
            try? fileManager.removeItem(at: part)
            fileManager.createFile(atPath: part.path, contents: nil)
        }

        if resumeFrom < entry.bytes {
            let (statusCode, body) = try await openDownload(
                source.downloadURL(for: entry), resumeFrom)
            switch (statusCode, resumeFrom) {
            case (200, 0), (206, _):
                break
            case (200, _):
                // The server ignored the range; start the file over.
                hasher = SHA256()
                resumeFrom = 0
                try? fileManager.removeItem(at: part)
                fileManager.createFile(atPath: part.path, contents: nil)
            default:
                throw PrebuiltInstallError.badResponse(
                    path: entry.path, detail: "HTTP \(statusCode)")
            }

            let handle = try FileHandle(forWritingTo: part)
            defer { try? handle.close() }
            try handle.seekToEnd()
            var written: UInt64 = 0
            var lastReport: UInt64 = 0
            for try await chunk in body {
                try Task.checkCancellation()
                hasher.update(data: chunk)
                try handle.write(contentsOf: chunk)
                written += UInt64(chunk.count)
                if written - lastReport >= 16 * 1_048_576 {
                    lastReport = written
                    onProgress(written)
                }
            }
            resumeFrom += written
        }

        guard resumeFrom == entry.bytes else {
            throw PrebuiltInstallError.sizeMismatch(
                path: entry.path, expected: entry.bytes, actual: resumeFrom)
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == entry.sha256.lowercased() else {
            // A corrupt partial must not survive to poison the next resume.
            try? fileManager.removeItem(at: part)
            throw PrebuiltInstallError.hashMismatch(path: entry.path)
        }
        try? fileManager.removeItem(at: final)
        try fileManager.moveItem(at: part, to: final)
        let fetched = entry.bytes
        return fetched
    }

    /// The trusted-install receipt, written from the pinned table rather than
    /// by re-hashing: every file below was verified against these hashes on
    /// its way in. The set is the manifest's own files plus `manifest.json` —
    /// the sidecar's files are not in the manifest and must not be in the
    /// receipt (`VerifiedInstallReceiptReader.validate` checks set equality).
    private static func writeReceipt(source: PrebuiltModelSource,
                                     outputDirectory: URL) throws {
        var files: [String: VerifiedInstallReceipt.FileEntry] = [:]
        for entry in source.files
        where !entry.path.hasPrefix("\(AppModelKind.mtpSidecarDirectoryName)/") {
            files[entry.path] = VerifiedInstallReceipt.FileEntry(
                size: entry.bytes, sha256: entry.sha256)
        }
        guard let manifest = source.file(at: "manifest.json") else {
            throw PrebuiltInstallError.badResponse(
                path: "manifest.json", detail: "source table has no manifest entry")
        }
        let receipt = VerifiedInstallReceipt(
            manifestSha256: manifest.sha256,
            modelDirectoryPath: outputDirectory.standardizedFileURL.path,
            sourceRepoID: source.repoID,
            sourceRevision: source.revision,
            verificationTimestamp: ISO8601DateFormatter().string(from: Date()),
            toolVersion: "TsugumiMac prebuilt-installer/1",
            files: files)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(receipt)
        try data.write(
            to: outputDirectory.appendingPathComponent(VerifiedInstallReceiptReader.fileName),
            options: .atomic)
    }

    // MARK: - Partial state

    public func hasPartialInstall(outputDirectory: URL) -> Bool {
        source.files.contains { entry in
            Self.fileSize(at: Self.partURL(
                for: outputDirectory.appendingPathComponent(entry.path))) != nil
        }
    }

    public func discardPartialInstall(outputDirectory: URL) async throws {
        // Only the unverified halves go: a `.part` never passed its hash.
        // Files already renamed into place did, and deleting 20 verified
        // gigabytes because one download was interrupted would be spite.
        let fileManager = FileManager.default
        for entry in source.files {
            let part = Self.partURL(for: outputDirectory.appendingPathComponent(entry.path))
            if fileManager.fileExists(atPath: part.path) {
                try fileManager.removeItem(at: part)
            }
        }
    }

    public func cancel() {
        let task = taskState.value.withLock { active -> Task<Void, Never>? in
            defer { active = nil }
            return active?.task
        }
        task?.cancel()
    }

    // MARK: - Helpers

    private static func partURL(for final: URL) -> URL {
        final.deletingLastPathComponent()
            .appendingPathComponent(final.lastPathComponent + ".part")
    }

    private static func fileSize(at url: URL) -> UInt64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? UInt64)
            .flatMap { $0 }
    }
}

public enum PrebuiltInstallError: Error, CustomStringConvertible, Equatable, Sendable {
    case badResponse(path: String, detail: String)
    case sizeMismatch(path: String, expected: UInt64, actual: UInt64)
    case hashMismatch(path: String)

    public var description: String {
        switch self {
        case .badResponse(let path, let detail):
            return "download of \(path) failed: \(detail)"
        case .sizeMismatch(let path, let expected, let actual):
            return "\(path) is \(actual) bytes, expected \(expected)"
        case .hashMismatch(let path):
            return "\(path) failed SHA-256 verification"
        }
    }
}
