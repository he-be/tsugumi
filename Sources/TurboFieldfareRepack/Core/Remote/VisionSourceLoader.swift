import Foundation

/// Bound vision source: the pinned repository, resolved to a commit, with the
/// shard headers the tower tensors live in and the parity evidence measured on
/// its side. Shard IDs are prefixed so a copy plan can carry ranges from two
/// different repositories without their filenames colliding.
struct BoundVisionSource {
    static let shardIDPrefix = "vision:"

    let pin: VisionSourcePin
    let remote: HuggingFaceRemoteSource
    let resolvedCommit: String
    /// Keyed by prefixed shard ID, matching `SourceTensor.shardPath`.
    let files: [String: RemoteFileInfo]
    let shardHeaders: [Safetensors.Header]
    /// `parityTensor.textName -> SHA-256 measured in the vision repository`.
    let parityDigests: [String: String]
}

enum VisionSourceLoader {

    static func load(pin: VisionSourcePin,
                     token: String?,
                     downloadSession: RemoteDownloadSession,
                     baseURL: URL,
                     tempDirectory: String,
                     retryPolicy: RemoteRetryPolicy,
                     metadataDirectory: String,
                     audit: RepackAudit) async throws -> BoundVisionSource {
        try Posix.mkdirP(metadataDirectory)
        let remote = HuggingFaceRemoteSource(repoID: pin.repoID,
                                             requestedRevision: pin.revision,
                                             token: token,
                                             downloadSession: downloadSession,
                                             baseURL: baseURL,
                                             tempDirectory: tempDirectory,
                                             retryPolicy: retryPolicy)
        let indexInfo = try await remote.resolveFileInfo(
            filename: "model.safetensors.index.json", audit: audit)
        // The pin is an immutable commit, so anything else means the request
        // was served from a different state of the repository.
        guard indexInfo.resolvedCommit == pin.revision else {
            throw RepackError.remoteProtocolInvalid(detail: """
                vision repository \(pin.repoID) resolved to \
                \(indexInfo.resolvedCommit), expected the pinned \(pin.revision)
                """)
        }
        let pinned = remote.pinned(commit: indexInfo.resolvedCommit)

        let indexPath = (metadataDirectory as NSString)
            .appendingPathComponent("vision-index.json")
        try await pinned.fetchSmallFile(filename: "model.safetensors.index.json",
                                        info: indexInfo,
                                        capBytes: 4 * 1024 * 1024,
                                        outputPath: indexPath,
                                        audit: audit)
        let indexSha = try Sha256Stream.hashFile(path: indexPath)
        guard indexSha.lowercased() == pin.indexSha256Hex.lowercased() else {
            throw RepackError.sourceFingerprintRejected(path: pin.repoID, sha256: indexSha)
        }
        let weightMap = try weightMap(indexPath: indexPath)

        let configInfo = try await pinned.resolveFileInfo(filename: "config.json",
                                                          audit: audit)
        let configPath = (metadataDirectory as NSString)
            .appendingPathComponent("vision-config.json")
        try await pinned.fetchSmallFile(filename: "config.json",
                                        info: configInfo,
                                        capBytes: 1024 * 1024,
                                        outputPath: configPath,
                                        audit: audit)
        let declared = try VisionSourceConfigLoader.load(configPath: configPath)
        guard declared == pin.config else {
            throw RepackError.configurationInvalid(detail: """
                vision config in \(pin.repoID) does not match the pinned tower \
                geometry: \(declared) != \(pin.config)
                """)
        }

        // Only the shards that actually hold something we copy: the tower
        // tensors and the parity tensors. On the pinned source that is one 46 GB
        // shard for the tower plus, at most, one more for the parity norms —
        // and only their headers are fetched.
        var neededShards: [String] = []
        var seen = Set<String>()
        var missing: [String] = []
        for name in weightMap.keys.sorted() where
            pin.tensorPrefixes.contains(where: { name.hasPrefix($0) }) {
            let shard = weightMap[name]!
            if seen.insert(shard).inserted { neededShards.append(shard) }
        }
        for parity in pin.parityTensors {
            guard let shard = weightMap[parity.visionRepoName] else {
                missing.append(parity.visionRepoName)
                continue
            }
            if seen.insert(shard).inserted { neededShards.append(shard) }
        }
        guard missing.isEmpty else {
            throw RepackError.configurationInvalid(
                detail: "vision source is missing parity tensors \(missing.joined(separator: ", "))")
        }

        var files: [String: RemoteFileInfo] = [:]
        var headers: [Safetensors.Header] = []
        for shard in neededShards {
            let info = try await pinned.resolveFileInfo(filename: shard, audit: audit)
            guard info.resolvedCommit == indexInfo.resolvedCommit else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "vision shard \(shard) commit differs from index commit")
            }
            guard info.acceptsRanges else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "vision shard \(shard) does not advertise byte ranges")
            }
            files[BoundVisionSource.shardIDPrefix + shard] = info
            headers.append(try await RemoteShardHeader.load(
                remote: pinned,
                filename: shard,
                info: info,
                shardID: BoundVisionSource.shardIDPrefix + shard,
                audit: audit))
        }

        var parityDigests: [String: String] = [:]
        var byName: [String: SourceTensor] = [:]
        for header in headers {
            for tensor in header.tensors { byName[tensor.name] = tensor }
        }
        for parity in pin.parityTensors {
            guard let tensor = byName[parity.visionRepoName] else {
                throw RepackError.missingTensor(name: parity.visionRepoName)
            }
            guard let info = files[tensor.shardPath] else {
                throw RepackError.configurationInvalid(
                    detail: "no remote info for \(tensor.shardPath)")
            }
            guard tensor.sizeBytes <= 1024 * 1024 else {
                throw RepackError.configurationInvalid(
                    detail: "parity tensor \(parity.visionRepoName) is unexpectedly large")
            }
            let temporary = try await pinned.downloadRangeToTempFile(
                filename: String(tensor.shardPath
                    .dropFirst(BoundVisionSource.shardIDPrefix.count)),
                info: info,
                offset: tensor.absoluteOffset,
                length: Int(tensor.sizeBytes),
                audit: audit)
            defer { try? FileManager.default.removeItem(atPath: temporary.path) }
            audit.remoteRangeRequests += 1
            audit.remoteBytesDownloaded += temporary.byteCount
            let digest = try Sha256Stream.hashFile(path: temporary.path)
            guard digest.lowercased() == parity.sha256.lowercased() else {
                throw RepackError.configurationInvalid(detail: """
                    vision source tensor \(parity.visionRepoName) hashes to \
                    \(digest), expected \(parity.sha256)
                    """)
            }
            parityDigests[parity.textName] = digest
        }

        return BoundVisionSource(pin: pin,
                                 remote: pinned,
                                 resolvedCommit: indexInfo.resolvedCommit,
                                 files: files,
                                 shardHeaders: headers,
                                 parityDigests: parityDigests)
    }

    private static func weightMap(indexPath: String) throws -> [String: String] {
        let data = try Data(contentsOf: URL(fileURLWithPath: indexPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let map = root["weight_map"] as? [String: String] else {
            throw RepackError.indexJsonInvalid(path: indexPath, detail: "no weight_map")
        }
        return map
    }
}

/// Fetches one shard's safetensors header over ranges: eight bytes of length,
/// then the header itself. `shardID` is what the resulting tensors report as
/// their shard, which is how a copy plan addresses them later.
enum RemoteShardHeader {
    static func load(remote: HuggingFaceRemoteSource,
                     filename: String,
                     info: RemoteFileInfo,
                     shardID: String,
                     audit: RepackAudit?) async throws -> Safetensors.Header {
        let prefix = try await remote.downloadRangeToTempFile(filename: filename,
                                                             info: info,
                                                             offset: 0,
                                                             length: 8,
                                                             audit: audit)
        defer { try? FileManager.default.removeItem(atPath: prefix.path) }
        let prefixData = try Data(contentsOf: URL(fileURLWithPath: prefix.path))
        guard prefixData.count == 8 else {
            throw RepackError.safetensorsHeaderInvalid(path: filename,
                                                       detail: "short header prefix")
        }
        let headerSize = prefixData.withUnsafeBytes { raw -> UInt64 in
            var value: UInt64 = 0
            for i in 0..<8 { value |= UInt64(raw[i]) << UInt64(i * 8) }
            return value
        }
        guard headerSize <= Safetensors.maxHeaderBytes, headerSize <= info.size - 8 else {
            throw RepackError.safetensorsHeaderTooLarge(path: filename, size: headerSize)
        }
        let headerFile = try await remote.downloadRangeToTempFile(filename: filename,
                                                                  info: info,
                                                                  offset: 8,
                                                                  length: Int(headerSize),
                                                                  audit: audit)
        defer { try? FileManager.default.removeItem(atPath: headerFile.path) }
        let headerData = try Data(contentsOf: URL(fileURLWithPath: headerFile.path))
        return try Safetensors.parseHeaderBytes(path: shardID,
                                                fileSize: info.size,
                                                headerBytes: headerData)
    }
}

/// Reads the tower geometry out of a source repository's `config.json`.
enum VisionSourceConfigLoader {
    static func load(configPath: String) throws -> VisionSourceConfig {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        guard let vision = root["vision_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no vision_config")
        }
        func int(_ dict: [String: Any], _ key: String) throws -> Int {
            guard let value = (dict[key] as? Int) ?? (dict[key] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(key)")
            }
            return value
        }
        func double(_ dict: [String: Any], _ key: String) throws -> Double {
            guard let value = (dict[key] as? Double)
                ?? (dict[key] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(key)")
            }
            return value
        }
        guard let rope = vision["rope_parameters"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath,
                                                detail: "vision_config has no rope_parameters")
        }
        // A clipped linear is a different operator, not a different constant, so
        // it cannot be represented by the manifest and must be refused here.
        if let clipped = vision["use_clipped_linears"] as? Bool, clipped {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "vision_config.use_clipped_linears is not supported")
        }
        let softTokens = try int(root, "vision_soft_tokens_per_image")
        if let defaultLength = (vision["default_output_length"] as? Int)
            ?? (vision["default_output_length"] as? NSNumber)?.intValue,
           defaultLength != softTokens {
            throw RepackError.configJsonInvalid(path: configPath, detail: """
                vision_config.default_output_length \(defaultLength) disagrees with \
                vision_soft_tokens_per_image \(softTokens)
                """)
        }
        guard let activation = vision["hidden_activation"] as? String else {
            throw RepackError.configJsonInvalid(path: configPath,
                                                detail: "missing hidden_activation")
        }
        guard let standardize = vision["standardize"] as? Bool else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing standardize")
        }
        return VisionSourceConfig(
            hiddenSize: try int(vision, "hidden_size"),
            numLayers: try int(vision, "num_hidden_layers"),
            numHeads: try int(vision, "num_attention_heads"),
            numKVHeads: try int(vision, "num_key_value_heads"),
            headDim: try int(vision, "head_dim"),
            intermediateSize: try int(vision, "intermediate_size"),
            patchSize: try int(vision, "patch_size"),
            poolingKernelSize: try int(vision, "pooling_kernel_size"),
            positionEmbeddingSize: try int(vision, "position_embedding_size"),
            ropeTheta: try double(rope, "rope_theta"),
            rmsNormEps: try double(vision, "rms_norm_eps"),
            hiddenActivation: activation,
            standardize: standardize,
            maxSoftTokens: softTokens,
            imageTokenID: try int(root, "image_token_id"),
            boiTokenID: try int(root, "boi_token_id"),
            eoiTokenID: try int(root, "eoi_token_id"))
    }
}
