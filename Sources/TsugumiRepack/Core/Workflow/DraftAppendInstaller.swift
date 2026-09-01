import Darwin
import Foundation
import MoEPackFormat

/// Adds the MTP drafter to a model that is already installed.
///
/// `--include-draft` builds a whole `.moepack` from scratch, which for an
/// existing install means re-copying ~15 GB of text weights the drafter does not
/// touch — and holding two copies while it runs. The drafter is a separate file
/// (`docs/mtp/03-DESIGN.md` D1) whose presence changes not one byte of
/// `model_weights.bin` or `packed_experts/`, so it can be appended instead:
/// download 236 MB, write `draft/draft_weights.bin`, rewrite the two JSON files
/// that describe the install.
public struct AddDraftOptions: Sendable {
    public let inputMoEPack: String
    public let token: String?
    /// Refuse to append the drafter to anything but the text checkpoint it is
    /// pinned to. Google trained it against the QAT target; the geometry check
    /// would let a different Gemma 4 through, so the repository is named too.
    public let requireKnownSource: Bool
    public let rangeChunkBytes: Int
    public let writeTileBytes: Int
    public let minFreeReserveBytes: UInt64
    public let downloadSession: RemoteDownloadSession
    public let baseURL: URL
    public let rangeRetryAttempts: Int
    public let retryBaseDelayNs: UInt64
    public let draftPin: DraftSourcePin
    public let copyAuditPath: String?

    public init(inputMoEPack: String,
                token: String? = nil,
                requireKnownSource: Bool = true,
                rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
                writeTileBytes: Int = WriterCore.tileBytes,
                minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024,
                downloadSession: RemoteDownloadSession = RemoteDownloadSession(),
                baseURL: URL = URL(string: "https://huggingface.co")!,
                rangeRetryAttempts: Int = 4,
                retryBaseDelayNs: UInt64 = 1_000_000_000,
                draftPin: DraftSourcePin = DraftModelSource.pin,
                copyAuditPath: String? = nil) {
        self.inputMoEPack = inputMoEPack
        self.token = token
        self.requireKnownSource = requireKnownSource
        self.rangeChunkBytes = rangeChunkBytes
        self.writeTileBytes = writeTileBytes
        self.minFreeReserveBytes = minFreeReserveBytes
        self.downloadSession = downloadSession
        self.baseURL = baseURL
        self.rangeRetryAttempts = rangeRetryAttempts
        self.retryBaseDelayNs = retryBaseDelayNs
        self.draftPin = draftPin
        self.copyAuditPath = copyAuditPath
    }
}

public struct AddDraftResult: Sendable {
    public let modelDirectory: String
    public let draftRepoID: String
    public let draftResolvedCommit: String
    public let tensorCount: Int
    public let payloadBytes: UInt64
    public let weightsFileBytes: UInt64
    public let downloadedBytes: UInt64
    /// Result of the full re-verification that runs after the drafter lands, so
    /// the receipt this leaves behind is evidence and not a copied assertion.
    public let verifiedFileCount: Int
    public let verifiedBytes: UInt64
    public let unexpectedEntries: [String]
}

public final class DraftAppendInstaller {
    private let options: AddDraftOptions
    private let audit: RepackAudit
    private let startTime = Date()

    public init(options: AddDraftOptions, audit: RepackAudit = RepackAudit()) {
        self.options = options
        self.audit = audit
    }

    public func run(progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }) async throws
        -> AddDraftResult {
        try validateOptions()
        let installLock = try InstallLock.acquire(outputDirectory: options.inputMoEPack)
        defer { withExtendedLifetime(installLock) {} }
        let paths = installLock.paths
        guard try Posix.entryKind(paths.finalDirectory) == .directory else {
            throw RepackError.configurationInvalid(
                detail: "no installed model at \(paths.finalDirectory)")
        }
        // An interrupted install owns this model's name; appending to the half
        // of it that is already promoted would leave the resumed install and the
        // drafter describing different models.
        guard try Posix.entryKind(paths.partialDirectory) == .absent,
              try Posix.entryKind(paths.checkpointFile) == .absent else {
            throw RepackError.installStateIncompatible(detail: """
                an interrupted install exists for this model; finish it with \
                --resume or remove it with --discard-partial before adding the drafter
                """)
        }
        let staging = paths.finalDirectory + ".draft.partial"
        do {
            let result = try await runPrepared(paths: paths,
                                               staging: staging,
                                               progress: progress)
            try? FileManager.default.removeItem(atPath: staging)
            return result
        } catch {
            // Everything downloaded so far lives outside the model directory, so
            // discarding it cannot damage the install we were adding to.
            try? FileManager.default.removeItem(atPath: staging)
            throw error
        }
    }

    private func runPrepared(paths: RemoteInstallPaths,
                             staging: String,
                             progress: @escaping @Sendable (ModelInstallProgress) -> Void) async throws
        -> AddDraftResult {
        try Task.checkCancellation()
        let root = URL(fileURLWithPath: options.inputMoEPack).standardizedFileURL.path
        let access = try MoEPackDirectoryAccess(rootPath: root)
        let manifest = try loadManifest(access: access)
        try requireNoDrafter(manifest)
        if options.requireKnownSource {
            guard manifest.modelID == DraftModelSource.requiredTextRepoID else {
                throw RepackError.configurationInvalid(detail: """
                    --add-draft requires the \(DraftModelSource.requiredTextRepoID) \
                    text checkpoint; this model is \(manifest.modelID)
                    """)
            }
        }
        guard manifest.files[Self.textWeightsPath] != nil else {
            throw RepackError.configurationInvalid(
                detail: "manifest does not declare \(Self.textWeightsPath)")
        }
        // Costs nothing and needs no network, so it runs before the download
        // rather than after it.
        try requireDraftFitsTarget(pin: options.draftPin, arch: manifest.arch)

        if try Posix.entryKind(staging) != .absent {
            try FileManager.default.removeItem(atPath: staging)
        }
        try Posix.mkdirP(staging)
        let metadataDirectory = (staging as NSString)
            .appendingPathComponent(".remote-metadata")
        let rangeTemporaryFile = (staging as NSString).appendingPathComponent(".range.tmp")

        progress(.downloadingMetadata)
        let draft = try await DraftSourceLoader.load(
            pin: options.draftPin,
            token: options.token,
            downloadSession: options.downloadSession,
            baseURL: options.baseURL,
            tempDirectory: staging,
            retryPolicy: RemoteRetryPolicy(attempts: options.rangeRetryAttempts,
                                           baseDelayNs: options.retryBaseDelayNs),
            metadataDirectory: metadataDirectory,
            audit: audit)
        try Task.checkCancellation()

        let weightsPath = (staging as NSString)
            .appendingPathComponent((MoEPackFormatV1.draftWeightsPath as NSString)
                .lastPathComponent)
        let plan = try DraftRepackPlanner.plan(path: weightsPath,
                                               pin: draft.pin,
                                               shardHeaders: draft.shardHeaders)

        let copies = plan.resident.entries.flatMap { entry -> [RangeCopy] in
            var out = [RangeCopy(shardID: entry.sourceWeight.shardPath,
                                 sourceOffset: entry.sourceWeight.absoluteOffset,
                                 size: entry.sizeBytes,
                                 destinationPath: plan.resident.path,
                                 destinationOffset: entry.fileOffset)]
            if let scales = entry.sourceScales {
                out.append(RangeCopy(shardID: scales.shardPath,
                                     sourceOffset: scales.absoluteOffset,
                                     size: entry.scaleSize,
                                     destinationPath: plan.resident.path,
                                     destinationOffset: entry.scaleOffset))
            }
            if let biases = entry.sourceBiases {
                out.append(RangeCopy(shardID: biases.shardPath,
                                     sourceOffset: biases.absoluteOffset,
                                     size: entry.biasSize,
                                     destinationPath: plan.resident.path,
                                     destinationOffset: entry.biasOffset))
            }
            return out
        }
        try RangeCopyPlanner.validateDestinationIntervals(copies, outputRoot: staging)
        let coalesced = try RangeCopyPlanner.coalesce(copies: copies,
                                                      rangeChunkBytes: options.rangeChunkBytes)
        let downloadBytes = coalesced.reduce(UInt64(0)) { $0 + $1.size }
        progress(.planning(downloadBytes: downloadBytes,
                           outputBytes: plan.resident.totalSize))

        let requirement = try DiskSpaceChecker.requireAvailable(
            path: paths.parentDirectory,
            bytes: plan.resident.totalSize + UInt64(options.rangeChunkBytes),
            reserveBytes: options.minFreeReserveBytes)
        progress(.checkingDisk(requirement))
        try Task.checkCancellation()

        progress(.reservingOutput(bytes: plan.resident.totalSize))
        let descriptor = try ResidentWriter.createAndWriteIndex(plan: plan.resident,
                                                                audit: audit)
        try Posix.fsync(descriptor, path: plan.resident.path)
        close(descriptor)
        try Posix.fsyncDirectory(staging)

        let downloadStart = audit.remoteBytesDownloaded
        progress(.copyingPayload(reusedBytes: 0,
                                 downloadedThisRunBytes: 0,
                                 totalBytes: downloadBytes))
        let provider = HTTPRangeSourceByteProvider(
            remote: draft.remote,
            files: draft.files,
            writeTileBytes: options.writeTileBytes,
            shardIDPrefix: BoundDraftSource.shardIDPrefix)
        try await provider.copyBatch(
            coalesced,
            completedRangeIDs: [],
            partialDirectory: staging,
            temporaryPath: rangeTemporaryFile,
            audit: audit,
            progress: { downloaded in
                progress(.copyingPayload(reusedBytes: 0,
                                         downloadedThisRunBytes: downloaded,
                                         totalBytes: downloadBytes))
            },
            commit: { _ in })

        progress(.hashingOutput(MoEPackFormatV1.draftWeightsPath))
        let weightsFD = try Posix.openRead(plan.resident.path)
        let weightsSize: UInt64
        do {
            weightsSize = try Posix.fileSize(fd: weightsFD, path: plan.resident.path)
            close(weightsFD)
        } catch {
            close(weightsFD)
            throw error
        }
        let weightsSha = try WriterCore.hashEntireFile(
            path: plan.resident.path,
            size: weightsSize,
            audit: audit,
            cancellationCheck: Task.checkCancellation)

        try? FileManager.default.removeItem(atPath: metadataDirectory)
        try? FileManager.default.removeItem(atPath: rangeTemporaryFile)
        progress(.finalizing)
        try Task.checkCancellation()

        // Encode the new manifest before anything enters the model directory:
        // if the drafter cannot be described (a geometry the codec refuses, say),
        // the refusal happens while the only thing on disk is staging.
        let updatedManifest = try Self.manifestAddingDraft(
            manifest,
            plan: plan,
            weights: MoEPackManifestFileV1(size: weightsSize, sha256: weightsSha))
        let manifestData = try MoEPackManifestCodec.encode(updatedManifest)

        // The drafter lands before the manifest names it. In the window between
        // the two, the model is the text-only model it already was plus a file
        // nothing reads; the reverse order would advertise a drafter that is not
        // there yet.
        let draftDirectory = (paths.finalDirectory as NSString)
            .appendingPathComponent((MoEPackFormatV1.draftWeightsPath as NSString)
                .deletingLastPathComponent)
        try Posix.mkdirP(draftDirectory)
        let installedWeightsPath = (paths.finalDirectory as NSString)
            .appendingPathComponent(MoEPackFormatV1.draftWeightsPath)
        try Posix.rename(from: plan.resident.path, to: installedWeightsPath)
        try Posix.fsyncDirectory(draftDirectory)
        try Posix.fsyncDirectory(paths.finalDirectory)

        try access.atomicWrite(manifestData, to: "manifest.json")

        // Re-verify from the manifest rather than trusting the digests it
        // already carried: the receipt this writes claims every file was
        // checked, and the drafter has to be inside that claim.
        let verification = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputMoEPack: options.inputMoEPack))

        audit.draftRepoID = draft.pin.repoID
        audit.draftResolvedCommit = draft.resolvedCommit
        audit.draftTensorCount = plan.tensorCount
        audit.draftPayloadBytes = plan.payloadBytes
        audit.draftProvenanceTensors = draft.pin.provenanceTensors
            .map(\.repoName).sorted()
        audit.outputFiles.append(.init(relativePath: MoEPackFormatV1.draftWeightsPath,
                                       size: weightsSize,
                                       sha256: weightsSha))
        audit.wallTimeSeconds = Date().timeIntervalSince(startTime)
        audit.wholeFileHeapBuffers = false
        if let auditPath = options.copyAuditPath {
            let data = try audit.toJSONData(outputDir: root)
            try Posix.mkdirP((auditPath as NSString).deletingLastPathComponent)
            try data.write(to: URL(fileURLWithPath: auditPath))
        }

        return AddDraftResult(
            modelDirectory: root,
            draftRepoID: draft.pin.repoID,
            draftResolvedCommit: draft.resolvedCommit,
            tensorCount: plan.tensorCount,
            payloadBytes: plan.payloadBytes,
            weightsFileBytes: weightsSize,
            downloadedBytes: audit.remoteBytesDownloaded - downloadStart,
            verifiedFileCount: verification.fileCount,
            verifiedBytes: verification.bytesVerified,
            unexpectedEntries: verification.unexpectedEntries)
    }

    // MARK: - Manifest

    private static let textWeightsPath = "model_weights.bin"

    private func loadManifest(access: MoEPackDirectoryAccess) throws -> MoEPackManifestV1 {
        let data = try access.readMetadata("manifest.json",
                                           maxBytes: VerifiedInstallTool.manifestMaxBytes)
        do { return try MoEPackManifestCodec.decode(data) }
        catch {
            throw RepackError.configurationInvalid(detail: "manifest.json invalid: \(error)")
        }
    }

    private func requireNoDrafter(_ manifest: MoEPackManifestV1) throws {
        guard manifest.draft == nil,
              manifest.flags["mtpDraft"] != true,
              manifest.files[MoEPackFormatV1.draftWeightsPath] == nil else {
            throw RepackError.configurationInvalid(detail: """
                \(options.inputMoEPack) already has an MTP drafter; reinstall the \
                model to change it
                """)
        }
    }

    /// The drafter has no K/V projections and reads the target's, so a
    /// disagreement here is not a slower model but a wrong one. The manifest
    /// codec enforces the same conditions when the updated manifest is encoded;
    /// checking first means the refusal names the mismatch and costs no bytes.
    private func requireDraftFitsTarget(pin: DraftSourcePin,
                                        arch: MoEPackManifestArchV1) throws {
        let config = pin.config
        let checks: [(String, Int, Int)] = [
            ("backbone_hidden_size", config.backboneHiddenSize, arch.hiddenSize),
            ("vocab_size", config.vocabSize, arch.vocabSize),
            ("sliding_window", config.slidingWindow, arch.slidingWindow),
            ("head_dim", config.headDim, arch.headDim),
            ("global_head_dim", config.fullHeadDim, arch.fullHeadDim),
            ("num_key_value_heads", config.numKVHeads, arch.numKVHeads),
            ("num_global_key_value_heads", config.numFullKVHeads, arch.numFullKVHeads),
        ]
        for (field, drafter, target) in checks where drafter != target {
            throw RepackError.configurationInvalid(detail: """
                drafter \(field) is \(drafter) but this model uses \(target); the \
                drafter reads the target's K/V and cannot be paired with it
                """)
        }
        let ropes: [(String, Double, Double)] = [
            ("rope_theta", config.ropeTheta, arch.ropeTheta),
            ("full rope_theta", config.fullRopeTheta, arch.fullRopeTheta),
            ("partial_rotary_factor", config.partialRotaryFactor, arch.partialRotaryFactor),
        ]
        for (field, drafter, target) in ropes where drafter != target {
            throw RepackError.configurationInvalid(detail: """
                drafter \(field) is \(drafter) but this model uses \(target); the \
                drafter reads the target's K/V and cannot be paired with it
                """)
        }
    }

    /// The same manifest with the drafter declared. Every other field is carried
    /// through untouched, so this produces the identical bytes a fresh
    /// `--include-draft` install of the same checkpoint would write.
    static func manifestAddingDraft(_ manifest: MoEPackManifestV1,
                                    plan: DraftFilePlan,
                                    weights: MoEPackManifestFileV1) throws -> MoEPackManifestV1 {
        var flags = manifest.flags
        flags["mtpDraft"] = true
        var files = manifest.files
        guard files.updateValue(weights,
                                forKey: MoEPackFormatV1.draftWeightsPath) == nil else {
            throw RepackError.configurationInvalid(
                detail: "manifest already declares \(MoEPackFormatV1.draftWeightsPath)")
        }
        return MoEPackManifestV1(
            magic: manifest.magic,
            versionMajor: manifest.versionMajor,
            versionMinor: max(manifest.versionMinor, MoEPackFormatV1.versionMinorDraft),
            flags: flags,
            modelID: manifest.modelID,
            sourceSnapshotHash: manifest.sourceSnapshotHash,
            arch: manifest.arch,
            quant: manifest.quant,
            vision: manifest.vision,
            draft: try MoEPackJSON.draftSection(
                plan, fullAttentionLayerMask: manifest.arch.fullAttentionLayerMask),
            files: files,
            expertsPerLayer: manifest.expertsPerLayer,
            numLayers: manifest.numLayers,
            expertStride: manifest.expertStride,
            bitWidthOverridesHonored: manifest.bitWidthOverridesHonored)
    }

    private func validateOptions() throws {
        guard options.rangeChunkBytes > 0,
              options.rangeChunkBytes <= RemoteChunkPolicy.maxBytes else {
            throw RepackError.configurationInvalid(
                detail: "bad range chunk bytes \(options.rangeChunkBytes)")
        }
        guard options.writeTileBytes > 0,
              options.writeTileBytes <= BoundedScratch.defaultLimitBytes else {
            throw RepackError.configurationInvalid(
                detail: "bad write tile bytes \(options.writeTileBytes)")
        }
        guard options.rangeRetryAttempts >= 0 else {
            throw RepackError.configurationInvalid(
                detail: "bad range retry attempts \(options.rangeRetryAttempts)")
        }
    }
}
