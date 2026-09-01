import Darwin
import Foundation

public protocol SourceByteProvider {
    func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws
}

public final class HTTPRangeSourceByteProvider: SourceByteProvider {
    private let remote: HuggingFaceRemoteSource
    private let files: [String: RemoteFileInfo]
    private let writeTileBytes: Int
    /// Namespace stripped from a shard ID to recover the remote filename. A
    /// vision install copies from two repositories, and their shard filenames
    /// would otherwise collide in one copy plan.
    private let shardIDPrefix: String

    public init(remote: HuggingFaceRemoteSource,
                files: [String: RemoteFileInfo],
                writeTileBytes: Int = WriterCore.tileBytes,
                shardIDPrefix: String = "") {
        self.remote = remote
        self.files = files
        self.writeTileBytes = writeTileBytes
        self.shardIDPrefix = shardIDPrefix
    }

    public func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws {
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: writeTileBytes,
            alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)

        var outputFDs: [String: Int32] = [:]
        defer { outputFDs.values.forEach { close($0) } }
        var downloaded: UInt64 = 0

        for copy in copies where !completedRangeIDs.contains(copy.id) {
            try Task.checkCancellation()
            guard let info = files[copy.shardID] else {
                throw RepackError.configurationInvalid(
                    detail: "missing remote info for \(copy.shardID)")
            }
            if try Posix.entryKind(temporaryPath) != .absent {
                try FileManager.default.removeItem(atPath: temporaryPath)
            }
            let base = downloaded
            let filename = copy.shardID.hasPrefix(shardIDPrefix)
                ? String(copy.shardID.dropFirst(shardIDPrefix.count))
                : copy.shardID
            let temporary = try await remote.downloadRangeToTempFile(
                filename: filename,
                info: info,
                offset: copy.sourceOffset,
                length: Int(copy.size),
                targetPath: temporaryPath,
                progress: { bytes in progress(base + bytes) },
                audit: audit)

            audit.remoteRangeRequests += 1
            audit.remoteBytesDownloaded += temporary.byteCount
            audit.largestRemoteTransferBytes = max(
                audit.largestRemoteTransferBytes,
                Int(temporary.byteCount))
            downloaded += temporary.byteCount
            progress(downloaded)

            let sourceFD = try Posix.openReadNoFollow(temporary.path)
            var touched = Set<String>()
            do {
                for destination in copy.destinations {
                    let destinationFD: Int32
                    if let existing = outputFDs[destination.destinationPath] {
                        destinationFD = existing
                    } else {
                        destinationFD = try Posix.openExistingRW(
                            destination.destinationPath)
                        outputFDs[destination.destinationPath] = destinationFD
                    }
                    touched.insert(destination.destinationPath)
                    try RangeCopyIO.copyBytes(
                        sourceFD: sourceFD,
                        sourcePath: temporary.path,
                        destinationFD: destinationFD,
                        destinationPath: destination.destinationPath,
                        sourceOffset: destination.sourceOffset - copy.sourceOffset,
                        destinationOffset: destination.destinationOffset,
                        size: destination.size,
                        scratch: scratch,
                        audit: audit)
                }
                close(sourceFD)
            } catch {
                close(sourceFD)
                throw error
            }

            try Task.checkCancellation()
            for path in touched {
                if let descriptor = outputFDs[path] {
                    try Posix.fsync(descriptor, path: path)
                }
            }
            let digest = try RangeCopyIO.destinationDigest(
                copy,
                partialDirectory: partialDirectory,
                scratch: scratch)
            try commit(RemoteCompletedRange(
                id: copy.id,
                destinationDigest: digest,
                sourceBytes: copy.size,
                destinationBytes: copy.destinations.reduce(0) { $0 + $1.size }))
            progress(downloaded)
            try? FileManager.default.removeItem(atPath: temporary.path)
            try Task.checkCancellation()
        }
    }

    public static func destinationDigest(
        _ copy: CoalescedRangeCopy,
        partialDirectory: String,
        scratch: UnsafeMutableRawBufferPointer? = nil
    ) throws -> String {
        try RangeCopyIO.destinationDigest(copy,
                                          partialDirectory: partialDirectory,
                                          scratch: scratch)
    }
}

/// Splits one copy plan across two sources by shard-ID namespace. A vision
/// install takes its text weights from the checkpoint being installed and its
/// tower from the pinned upstream repository, but both write into the same
/// partial directory under one checkpoint, so the batch stays a single unit of
/// work.
public final class RoutingSourceByteProvider: SourceByteProvider {
    private let prefix: String
    private let prefixed: SourceByteProvider
    private let rest: SourceByteProvider

    public init(prefix: String,
                prefixed: SourceByteProvider,
                rest: SourceByteProvider) {
        self.prefix = prefix
        self.prefixed = prefixed
        self.rest = rest
    }

    public func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws {
        let groups: [(SourceByteProvider, [CoalescedRangeCopy])] = [
            (rest, copies.filter { !$0.shardID.hasPrefix(prefix) }),
            (prefixed, copies.filter { $0.shardID.hasPrefix(prefix) }),
        ]
        // Sub-providers report bytes within their own batch; the caller wants
        // one rising total.
        var completedBefore: UInt64 = 0
        for (provider, group) in groups where !group.isEmpty {
            var groupBytes: UInt64 = 0
            let base = completedBefore
            try await provider.copyBatch(
                group,
                completedRangeIDs: completedRangeIDs,
                partialDirectory: partialDirectory,
                temporaryPath: temporaryPath,
                audit: audit,
                progress: { bytes in progress(base + bytes) },
                commit: { completed in
                    groupBytes += completed.sourceBytes
                    try commit(completed)
                })
            completedBefore = base + groupBytes
        }
    }
}

/// Reads the source bytes straight out of a staged snapshot. Same copy plan,
/// same destination digests and same checkpoint as the streaming provider —
/// only the fetch is replaced, so a coalesced range becomes a `pread` at its
/// absolute offset instead of an HTTP range into a temporary file.
public final class LocalSnapshotByteProvider: SourceByteProvider {
    private let shardPaths: [String: String]
    private let writeTileBytes: Int

    public init(shardPaths: [String: String],
                writeTileBytes: Int = WriterCore.tileBytes) {
        self.shardPaths = shardPaths
        self.writeTileBytes = writeTileBytes
    }

    public func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws {
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: writeTileBytes,
            alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)

        var sourceFDs: [String: Int32] = [:]
        var outputFDs: [String: Int32] = [:]
        defer {
            sourceFDs.values.forEach { close($0) }
            outputFDs.values.forEach { close($0) }
        }
        var copied: UInt64 = 0

        for copy in copies where !completedRangeIDs.contains(copy.id) {
            try Task.checkCancellation()
            guard let sourcePath = shardPaths[copy.shardID] else {
                throw RepackError.configurationInvalid(
                    detail: "missing snapshot shard for \(copy.shardID)")
            }
            let sourceFD: Int32
            if let existing = sourceFDs[sourcePath] {
                sourceFD = existing
            } else {
                sourceFD = try Posix.openReadNoFollow(sourcePath)
                sourceFDs[sourcePath] = sourceFD
            }

            var touched = Set<String>()
            for destination in copy.destinations {
                let destinationFD: Int32
                if let existing = outputFDs[destination.destinationPath] {
                    destinationFD = existing
                } else {
                    destinationFD = try Posix.openExistingRW(destination.destinationPath)
                    outputFDs[destination.destinationPath] = destinationFD
                }
                touched.insert(destination.destinationPath)
                // Snapshot offsets are absolute in the shard, so no rebasing
                // onto a per-range temporary file.
                try RangeCopyIO.copyBytes(
                    sourceFD: sourceFD,
                    sourcePath: sourcePath,
                    destinationFD: destinationFD,
                    destinationPath: destination.destinationPath,
                    sourceOffset: destination.sourceOffset,
                    destinationOffset: destination.destinationOffset,
                    size: destination.size,
                    scratch: scratch,
                    audit: audit)
            }

            try Task.checkCancellation()
            for path in touched {
                if let descriptor = outputFDs[path] {
                    try Posix.fsync(descriptor, path: path)
                }
            }
            let digest = try RangeCopyIO.destinationDigest(
                copy,
                partialDirectory: partialDirectory,
                scratch: scratch)
            try commit(RemoteCompletedRange(
                id: copy.id,
                destinationDigest: digest,
                sourceBytes: copy.size,
                destinationBytes: copy.destinations.reduce(0) { $0 + $1.size }))
            copied += copy.size
            progress(copied)
            try Task.checkCancellation()
        }
    }
}
