import Foundation

/// Bound drafter source: the pinned repository, resolved to a commit, with the
/// shard headers its tensors live in and the provenance evidence measured on its
/// side. Shard IDs are prefixed so one copy plan can carry ranges from the text
/// repository, the vision repository and this one without collisions.
struct BoundDraftSource {
    static let shardIDPrefix = "draft:"

    let pin: DraftSourcePin
    let remote: HuggingFaceRemoteSource
    let resolvedCommit: String
    /// Keyed by prefixed shard ID, matching `SourceTensor.shardPath`.
    let files: [String: RemoteFileInfo]
    let shardHeaders: [Safetensors.Header]
    /// `provenanceTensor.repoName -> SHA-256 measured in the drafter repository`.
    let provenanceDigests: [String: String]
}

enum DraftSourceLoader {

    static func load(pin: DraftSourcePin,
                     token: String?,
                     downloadSession: RemoteDownloadSession,
                     baseURL: URL,
                     tempDirectory: String,
                     retryPolicy: RemoteRetryPolicy,
                     metadataDirectory: String,
                     audit: RepackAudit) async throws -> BoundDraftSource {
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
        // The pin is an immutable commit, so anything else means the request was
        // served from a different state of the repository.
        guard indexInfo.resolvedCommit == pin.revision else {
            throw RepackError.remoteProtocolInvalid(detail: """
                drafter repository \(pin.repoID) resolved to \
                \(indexInfo.resolvedCommit), expected the pinned \(pin.revision)
                """)
        }
        let pinned = remote.pinned(commit: indexInfo.resolvedCommit)

        let indexPath = (metadataDirectory as NSString)
            .appendingPathComponent("draft-index.json")
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
            .appendingPathComponent("draft-config.json")
        try await pinned.fetchSmallFile(filename: "config.json",
                                        info: configInfo,
                                        capBytes: 1024 * 1024,
                                        outputPath: configPath,
                                        audit: audit)
        let declared = try DraftSourceConfigLoader.load(configPath: configPath)
        guard declared == pin.config else {
            throw RepackError.configurationInvalid(detail: """
                drafter config in \(pin.repoID) does not match the pinned drafter \
                geometry: \(declared) != \(pin.config)
                """)
        }

        // Every tensor in this repository belongs to the drafter, so the shard
        // set is simply the one the index names — on the pinned source, a single
        // 236 MB file whose header is all that is read here.
        var neededShards: [String] = []
        var seen = Set<String>()
        for name in weightMap.keys.sorted() {
            let shard = weightMap[name]!
            if seen.insert(shard).inserted { neededShards.append(shard) }
        }
        for provenance in pin.provenanceTensors {
            guard weightMap[provenance.repoName] != nil else {
                throw RepackError.configurationInvalid(detail: """
                    drafter source is missing provenance tensor \(provenance.repoName)
                    """)
            }
        }

        var files: [String: RemoteFileInfo] = [:]
        var headers: [Safetensors.Header] = []
        for shard in neededShards {
            let info = try await pinned.resolveFileInfo(filename: shard, audit: audit)
            guard info.resolvedCommit == indexInfo.resolvedCommit else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "drafter shard \(shard) commit differs from index commit")
            }
            guard info.acceptsRanges else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "drafter shard \(shard) does not advertise byte ranges")
            }
            files[BoundDraftSource.shardIDPrefix + shard] = info
            headers.append(try await RemoteShardHeader.load(
                remote: pinned,
                filename: shard,
                info: info,
                shardID: BoundDraftSource.shardIDPrefix + shard,
                audit: audit))
        }

        var byName: [String: SourceTensor] = [:]
        for header in headers {
            for tensor in header.tensors { byName[tensor.name] = tensor }
        }
        var provenanceDigests: [String: String] = [:]
        for provenance in pin.provenanceTensors {
            guard let tensor = byName[provenance.repoName] else {
                throw RepackError.missingTensor(name: provenance.repoName)
            }
            guard let info = files[tensor.shardPath] else {
                throw RepackError.configurationInvalid(
                    detail: "no remote info for \(tensor.shardPath)")
            }
            guard tensor.sizeBytes <= 1024 * 1024 else {
                throw RepackError.configurationInvalid(
                    detail: "provenance tensor \(provenance.repoName) is unexpectedly large")
            }
            let temporary = try await pinned.downloadRangeToTempFile(
                filename: String(tensor.shardPath
                    .dropFirst(BoundDraftSource.shardIDPrefix.count)),
                info: info,
                offset: tensor.absoluteOffset,
                length: Int(tensor.sizeBytes),
                audit: audit)
            defer { try? FileManager.default.removeItem(atPath: temporary.path) }
            audit.remoteRangeRequests += 1
            audit.remoteBytesDownloaded += temporary.byteCount
            let digest = try Sha256Stream.hashFile(path: temporary.path)
            guard digest.lowercased() == provenance.sha256.lowercased() else {
                throw RepackError.configurationInvalid(detail: """
                    drafter tensor \(provenance.repoName) hashes to \(digest), but the \
                    pinned QAT assistant expects \(provenance.sha256); this is not that \
                    checkpoint
                    """)
            }
            provenanceDigests[provenance.repoName] = digest
        }

        return BoundDraftSource(pin: pin,
                                remote: pinned,
                                resolvedCommit: indexInfo.resolvedCommit,
                                files: files,
                                shardHeaders: headers,
                                provenanceDigests: provenanceDigests)
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

/// Reads the drafter geometry out of a source repository's `config.json`.
/// Anything the runtime cannot express — centroid embeddings, a logit softcap on
/// the drafter, layers that compute their own K/V — is refused here rather than
/// silently dropped (`docs/mtp/01-CHECKPOINT.md` §2, §5).
enum DraftSourceConfigLoader {
    static func load(configPath: String) throws -> DraftSourceConfig {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        guard let text = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
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
        func bool(_ dict: [String: Any], _ key: String) throws -> Bool {
            guard let value = dict[key] as? Bool else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(key)")
            }
            return value
        }

        // The 26B drafter does not use ordered (centroid) embeddings; a
        // checkpoint that does needs a re-ordering step this runtime has not got.
        if let ordered = root["use_ordered_embeddings"] as? Bool, ordered {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "use_ordered_embeddings is not supported")
        }
        // The drafter's LM head is a plain tied GEMV. A softcap would change the
        // proposal distribution and is not applied by any reference implementation.
        if let softcap = text["final_logit_softcapping"], !(softcap is NSNull) {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "text_config.final_logit_softcapping must be null for a drafter")
        }
        let numLayers = try int(text, "num_hidden_layers")
        // All layers share the target's K/V. A drafter with layers of its own
        // would need K/V projections this format does not carry.
        guard try int(text, "num_kv_shared_layers") == numLayers else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "text_config.num_kv_shared_layers must equal num_hidden_layers")
        }
        if let moe = text["enable_moe_block"] as? Bool, moe {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "text_config.enable_moe_block is not supported")
        }
        guard let layerTypes = text["layer_types"] as? [String],
              layerTypes.count == numLayers else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "text_config.layer_types must list \(numLayers) entries")
        }
        var mask: [Int] = []
        for type in layerTypes {
            switch type {
            case "full_attention": mask.append(1)
            case "sliding_attention": mask.append(0)
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer type \(type)")
            }
        }
        guard let rope = text["rope_parameters"] as? [String: Any],
              let sliding = rope["sliding_attention"] as? [String: Any],
              let full = rope["full_attention"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "text_config has no per-type rope_parameters")
        }
        guard let quant = root["quantization"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "no quantization slot")
        }
        guard let mode = quant["mode"] as? String else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "quantization has no mode")
        }
        guard let activation = text["hidden_activation"] as? String else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing hidden_activation")
        }
        return DraftSourceConfig(
            hiddenSize: try int(text, "hidden_size"),
            numLayers: numLayers,
            numHeads: try int(text, "num_attention_heads"),
            numKVHeads: try int(text, "num_key_value_heads"),
            numFullKVHeads: try int(text, "num_global_key_value_heads"),
            headDim: try int(text, "head_dim"),
            fullHeadDim: try int(text, "global_head_dim"),
            intermediateSize: try int(text, "intermediate_size"),
            backboneHiddenSize: try int(root, "backbone_hidden_size"),
            vocabSize: try int(text, "vocab_size"),
            slidingWindow: try int(text, "sliding_window"),
            ropeTheta: try double(sliding, "rope_theta"),
            fullRopeTheta: try double(full, "rope_theta"),
            partialRotaryFactor: try double(full, "partial_rotary_factor"),
            rmsNormEps: try double(text, "rms_norm_eps"),
            hiddenActivation: activation,
            tieWordEmbeddings: try bool(text, "tie_word_embeddings"),
            attentionKEqV: try bool(text, "attention_k_eq_v"),
            fullAttentionLayerMask: mask,
            quantBits: try int(quant, "bits"),
            quantGroupSize: try int(quant, "group_size"),
            quantMode: mode)
    }
}
