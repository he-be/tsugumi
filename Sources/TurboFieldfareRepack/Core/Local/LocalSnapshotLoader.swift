import Darwin
import Foundation

/// A checkpoint that already sits on disk in its distributed form: the shards,
/// `model.safetensors.index.json`, `config.json` and the tokenizer sidecars,
/// exactly as published. The repacker reads it with `pread` instead of ranged
/// HTTP, so there is no download, no resume window and no temporary staging.
struct LocalSnapshot {
    let metadata: IndexLoader.SourceMetadata
    let arch: ArchInfo
    let shardHeaders: [Safetensors.Header]
    /// `shard filename -> absolute path`. Keyed by filename so the copy plan's
    /// canonical fingerprint stays independent of where the snapshot lives.
    let shardPaths: [String: String]
    /// The pin this snapshot's index digest matched.
    let source: SourceFingerprint.KnownSource
    let directory: String
}

enum LocalSnapshotLoader {
    /// Loads and fingerprints a staged snapshot.
    ///
    /// Unlike the remote loader this always requires a known source: the
    /// install checkpoint and the receipt both record a 40-character revision,
    /// and a directory of files carries none. The index digest supplies it.
    static func load(directory: String, audit: RepackAudit? = nil) throws -> LocalSnapshot {
        guard try Posix.entryKind(directory) == .directory else {
            throw RepackError.configurationInvalid(
                detail: "source snapshot is not a directory: \(directory)")
        }
        let root = try Posix.physicalPath(directory)
        let metadata = try IndexLoader.load(snapshotDir: root)
        guard let source = SourceFingerprint.source(
            forIndexSha256: metadata.indexSha256Hex) else {
            throw RepackError.sourceFingerprintRejected(path: metadata.indexPath,
                                                        sha256: metadata.indexSha256Hex)
        }
        let arch = try ArchInfo.load(configPath: metadata.configPath)

        var shardPaths: [String: String] = [:]
        var headers: [Safetensors.Header] = []
        headers.reserveCapacity(metadata.shardFilenames.count)
        for shard in metadata.shardFilenames {
            let path = try resolvedShardPath(shard, root: root)
            headers.append(try readHeader(shard: shard, path: path, audit: audit))
            shardPaths[shard] = path
        }
        return LocalSnapshot(metadata: metadata,
                             arch: arch,
                             shardHeaders: headers,
                             shardPaths: shardPaths,
                             source: source,
                             directory: root)
    }

    /// Resolves a shard named by the index against the snapshot root. The index
    /// is untrusted input, so a name that escapes the directory is rejected
    /// rather than followed.
    static func resolvedShardPath(_ shard: String, root: String) throws -> String {
        guard !shard.isEmpty, !shard.hasPrefix("/"), !shard.contains("/"),
              shard != ".", shard != ".." else {
            throw RepackError.indexJsonInvalid(
                path: (root as NSString).appendingPathComponent("model.safetensors.index.json"),
                detail: "shard name is not a plain filename: \(shard)")
        }
        let path = (root as NSString).appendingPathComponent(shard)
        guard try Posix.entryKind(path) == .regular else {
            throw RepackError.configurationInvalid(
                detail: "source snapshot is missing shard \(shard)")
        }
        return path
    }

    private static func readHeader(shard: String,
                                   path: String,
                                   audit: RepackAudit?) throws -> Safetensors.Header {
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        let fileSize = try Posix.fileSize(fd: fd, path: path)
        guard fileSize > 8 else {
            throw RepackError.safetensorsHeaderInvalid(path: shard, detail: "short header prefix")
        }
        var prefix = [UInt8](repeating: 0, count: 8)
        try prefix.withUnsafeMutableBytes { raw in
            try Posix.preadAll(fd: fd, path: path, buf: raw.baseAddress!, count: 8, offset: 0)
        }
        var headerSize: UInt64 = 0
        for i in 0..<8 { headerSize |= UInt64(prefix[i]) << UInt64(i * 8) }
        guard headerSize <= Safetensors.maxHeaderBytes, headerSize <= fileSize - 8 else {
            throw RepackError.safetensorsHeaderTooLarge(path: shard, size: headerSize)
        }
        var headerBytes = Data(count: Int(headerSize))
        try headerBytes.withUnsafeMutableBytes { raw in
            try Posix.preadAll(fd: fd, path: path, buf: raw.baseAddress!,
                               count: Int(headerSize), offset: 8)
        }
        audit?.recordRead(bytes: 8 + Int(headerSize))
        return try Safetensors.parseHeaderBytes(path: shard,
                                                fileSize: fileSize,
                                                headerBytes: headerBytes)
    }
}
