import Foundation

struct RemoteSnapshot {
    let metadata: IndexLoader.SourceMetadata
    let arch: ArchInfo
    let shardHeaders: [Safetensors.Header]
    let remoteFiles: [String: RemoteFileInfo]
    let resolvedCommit: String
    let metadataDirectory: String
}

enum RemoteSnapshotLoader {
    static func load(remote: HuggingFaceRemoteSource,
                     requireKnownSource: Bool,
                     metadataDirectory: String,
                     audit: RepackAudit? = nil) async throws -> RemoteSnapshot {
        try Posix.mkdirP(metadataDirectory)

        let indexInfo = try await remote.resolveFileInfo(filename: "model.safetensors.index.json",
                                                         audit: audit)
        let pinned = remote.pinned(commit: indexInfo.resolvedCommit)
        let configInfo = try await pinned.resolveFileInfo(filename: "config.json",
                                                          audit: audit)
        guard configInfo.resolvedCommit == indexInfo.resolvedCommit else {
            throw RepackError.remoteProtocolInvalid(detail: "config commit differs from index commit")
        }

        try await pinned.fetchSmallFile(filename: "model.safetensors.index.json",
                                        info: indexInfo,
                                        capBytes: 4 * 1024 * 1024,
                                        outputPath: (metadataDirectory as NSString)
                                            .appendingPathComponent("model.safetensors.index.json"),
                                        audit: audit)
        try await pinned.fetchSmallFile(filename: "config.json",
                                        info: configInfo,
                                        capBytes: 1024 * 1024,
                                        outputPath: (metadataDirectory as NSString)
                                            .appendingPathComponent("config.json"),
                                        audit: audit)

        let metadata = try IndexLoader.load(snapshotDir: metadataDirectory)
        if requireKnownSource && SourceFingerprint.modelID(forIndexSha256: metadata.indexSha256Hex) == nil {
            throw RepackError.sourceFingerprintRejected(path: metadata.indexPath,
                                                        sha256: metadata.indexSha256Hex)
        }
        let arch = try ArchInfo.load(configPath: metadata.configPath)

        var files: [String: RemoteFileInfo] = [
            indexInfo.filename: indexInfo,
            configInfo.filename: configInfo,
        ]
        var headers: [Safetensors.Header] = []
        headers.reserveCapacity(metadata.shardFilenames.count)
        for shard in metadata.shardFilenames {
            let info = try await pinned.resolveFileInfo(filename: shard, audit: audit)
            guard info.resolvedCommit == indexInfo.resolvedCommit else {
                throw RepackError.remoteProtocolInvalid(detail: "shard \(shard) commit differs from index commit")
            }
            guard info.acceptsRanges else {
                throw RepackError.remoteProtocolInvalid(detail: "shard \(shard) does not advertise byte ranges")
            }
            files[shard] = info
            headers.append(try await RemoteShardHeader.load(remote: pinned,
                                                            filename: shard,
                                                            info: info,
                                                            shardID: shard,
                                                            audit: audit))
        }

        return RemoteSnapshot(metadata: metadata,
                              arch: arch,
                              shardHeaders: headers,
                              remoteFiles: files,
                              resolvedCommit: indexInfo.resolvedCommit,
                              metadataDirectory: metadataDirectory)
    }
}
