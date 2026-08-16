import Darwin
import Foundation
import TurboFieldfareFormat

/// Adds the vision tower to a model that is already installed.
///
/// `--include-vision` builds a whole `.gturbo` from scratch, which for an
/// existing install means re-copying ~15 GB of text weights that the tower does
/// not touch — and holding two copies while it runs. The tower is a separate
/// file (`PLAN_VISION.md` §4-1) whose presence changes not one byte of
/// `model_weights.bin` or `packed_experts/`, so it can be appended instead:
/// download 1.15 GB, write `vision/vision_weights.bin`, rewrite the two JSON
/// files that describe the install.
public struct AddVisionOptions: Sendable {
    public let inputGTurbo: String
    public let token: String?
    /// Refuse to append the tower to anything but the text checkpoint it is
    /// pinned to. The parity check would catch a foreign checkpoint anyway;
    /// this makes the refusal happen before the download.
    public let requireKnownSource: Bool
    public let rangeChunkBytes: Int
    public let writeTileBytes: Int
    public let minFreeReserveBytes: UInt64
    public let downloadSession: RemoteDownloadSession
    public let baseURL: URL
    public let rangeRetryAttempts: Int
    public let retryBaseDelayNs: UInt64
    public let visionPin: VisionSourcePin
    public let copyAuditPath: String?

    public init(inputGTurbo: String,
                token: String? = nil,
                requireKnownSource: Bool = true,
                rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
                writeTileBytes: Int = WriterCore.tileBytes,
                minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024,
                downloadSession: RemoteDownloadSession = RemoteDownloadSession(),
                baseURL: URL = URL(string: "https://huggingface.co")!,
                rangeRetryAttempts: Int = 4,
                retryBaseDelayNs: UInt64 = 1_000_000_000,
                visionPin: VisionSourcePin = VisionModelSource.pin,
                copyAuditPath: String? = nil) {
        self.inputGTurbo = inputGTurbo
        self.token = token
        self.requireKnownSource = requireKnownSource
        self.rangeChunkBytes = rangeChunkBytes
        self.writeTileBytes = writeTileBytes
        self.minFreeReserveBytes = minFreeReserveBytes
        self.downloadSession = downloadSession
        self.baseURL = baseURL
        self.rangeRetryAttempts = rangeRetryAttempts
        self.retryBaseDelayNs = retryBaseDelayNs
        self.visionPin = visionPin
        self.copyAuditPath = copyAuditPath
    }
}

public struct AddVisionResult: Sendable {
    public let modelDirectory: String
    public let visionRepoID: String
    public let visionResolvedCommit: String
    public let tensorCount: Int
    public let payloadBytes: UInt64
    public let weightsFileBytes: UInt64
    public let downloadedBytes: UInt64
    /// Result of the full re-verification that runs after the tower lands, so
    /// the receipt this leaves behind is evidence and not a copied assertion.
    public let verifiedFileCount: Int
    public let verifiedBytes: UInt64
    public let unexpectedEntries: [String]
}

public final class VisionAppendInstaller {
    private let options: AddVisionOptions
    private let audit: RepackAudit
    private let startTime = Date()

    public init(options: AddVisionOptions, audit: RepackAudit = RepackAudit()) {
        self.options = options
        self.audit = audit
    }

    public func run(progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }) async throws
        -> AddVisionResult {
        try validateOptions()
        let installLock = try InstallLock.acquire(outputDirectory: options.inputGTurbo)
        defer { withExtendedLifetime(installLock) {} }
        let paths = installLock.paths
        guard try Posix.entryKind(paths.finalDirectory) == .directory else {
            throw RepackError.configurationInvalid(
                detail: "no installed model at \(paths.finalDirectory)")
        }
        // An interrupted install owns this model's name; appending to the half
        // of it that is already promoted would leave the resumed install and
        // the tower describing different models.
        guard try Posix.entryKind(paths.partialDirectory) == .absent,
              try Posix.entryKind(paths.checkpointFile) == .absent else {
            throw RepackError.installStateIncompatible(detail: """
                an interrupted install exists for this model; finish it with \
                --resume or remove it with --discard-partial before adding the tower
                """)
        }
        let staging = paths.finalDirectory + ".vision.partial"
        do {
            let result = try await runPrepared(paths: paths,
                                               staging: staging,
                                               progress: progress)
            try? FileManager.default.removeItem(atPath: staging)
            return result
        } catch {
            // Everything downloaded so far lives outside the model directory,
            // so discarding it cannot damage the install we were adding to.
            try? FileManager.default.removeItem(atPath: staging)
            throw error
        }
    }

    private func runPrepared(paths: RemoteInstallPaths,
                             staging: String,
                             progress: @escaping @Sendable (ModelInstallProgress) -> Void) async throws
        -> AddVisionResult {
        try Task.checkCancellation()
        let root = URL(fileURLWithPath: options.inputGTurbo).standardizedFileURL.path
        let access = try GTurboDirectoryAccess(rootPath: root)
        let manifest = try loadManifest(access: access)
        try requireNoTower(manifest)
        if options.requireKnownSource {
            guard manifest.modelID == VisionModelSource.requiredTextRepoID else {
                throw RepackError.configurationInvalid(detail: """
                    --add-vision requires the \(VisionModelSource.requiredTextRepoID) \
                    text checkpoint; this model is \(manifest.modelID)
                    """)
            }
        }
        guard manifest.files[Self.textWeightsPath] != nil else {
            throw RepackError.configurationInvalid(
                detail: "manifest does not declare \(Self.textWeightsPath)")
        }

        if try Posix.entryKind(staging) != .absent {
            try FileManager.default.removeItem(atPath: staging)
        }
        try Posix.mkdirP(staging)
        let metadataDirectory = (staging as NSString)
            .appendingPathComponent(".remote-metadata")
        let rangeTemporaryFile = (staging as NSString).appendingPathComponent(".range.tmp")

        progress(.downloadingMetadata)
        let vision = try await VisionSourceLoader.load(
            pin: options.visionPin,
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
            .appendingPathComponent((GTurboFormatV1.visionWeightsPath as NSString)
                .lastPathComponent)
        let plan = try VisionRepackPlanner.plan(path: weightsPath,
                                                pin: vision.pin,
                                                textHiddenSize: manifest.arch.hiddenSize,
                                                shardHeaders: vision.shardHeaders)
        try verifyParity(vision: vision, access: access)

        let copies = plan.resident.entries.map {
            RangeCopy(shardID: $0.sourceWeight.shardPath,
                      sourceOffset: $0.sourceWeight.absoluteOffset,
                      size: $0.sizeBytes,
                      destinationPath: plan.resident.path,
                      destinationOffset: $0.fileOffset)
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
            remote: vision.remote,
            files: vision.files,
            writeTileBytes: options.writeTileBytes,
            shardIDPrefix: BoundVisionSource.shardIDPrefix)
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

        progress(.hashingOutput(GTurboFormatV1.visionWeightsPath))
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

        // The tower lands before the manifest names it. In the window between
        // the two, the model is the text-only model it already was plus a file
        // nothing reads; the reverse order would advertise a tower that is not
        // there yet.
        let visionDirectory = (paths.finalDirectory as NSString)
            .appendingPathComponent((GTurboFormatV1.visionWeightsPath as NSString)
                .deletingLastPathComponent)
        try Posix.mkdirP(visionDirectory)
        let installedWeightsPath = (paths.finalDirectory as NSString)
            .appendingPathComponent(GTurboFormatV1.visionWeightsPath)
        try Posix.rename(from: plan.resident.path, to: installedWeightsPath)
        try Posix.fsyncDirectory(visionDirectory)
        try Posix.fsyncDirectory(paths.finalDirectory)

        let updated = try Self.manifestAddingVision(
            manifest,
            plan: plan,
            weights: GTurboManifestFileV1(size: weightsSize, sha256: weightsSha))
        try access.atomicWrite(try GTurboManifestCodec.encode(updated), to: "manifest.json")

        // Re-verify from the manifest rather than trusting the digests it
        // already carried: the receipt this writes claims every file was
        // checked, and the tower has to be inside that claim.
        let verification = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: options.inputGTurbo))

        audit.visionRepoID = vision.pin.repoID
        audit.visionResolvedCommit = vision.resolvedCommit
        audit.visionTensorCount = plan.tensorCount
        audit.visionPayloadBytes = plan.payloadBytes
        audit.outputFiles.append(.init(relativePath: GTurboFormatV1.visionWeightsPath,
                                       size: weightsSize,
                                       sha256: weightsSha))
        audit.wallTimeSeconds = Date().timeIntervalSince(startTime)
        audit.wholeFileHeapBuffers = false
        if let auditPath = options.copyAuditPath {
            let data = try audit.toJSONData(outputDir: root)
            try Posix.mkdirP((auditPath as NSString).deletingLastPathComponent)
            try data.write(to: URL(fileURLWithPath: auditPath))
        }

        return AddVisionResult(
            modelDirectory: root,
            visionRepoID: vision.pin.repoID,
            visionResolvedCommit: vision.resolvedCommit,
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

    private func loadManifest(access: GTurboDirectoryAccess) throws -> GTurboManifestV1 {
        let data = try access.readMetadata("manifest.json",
                                           maxBytes: VerifiedInstallTool.manifestMaxBytes)
        do { return try GTurboManifestCodec.decode(data) }
        catch {
            throw RepackError.configurationInvalid(detail: "manifest.json invalid: \(error)")
        }
    }

    private func requireNoTower(_ manifest: GTurboManifestV1) throws {
        guard manifest.vision == nil,
              manifest.flags["visionTower"] != true,
              manifest.files[GTurboFormatV1.visionWeightsPath] == nil else {
            throw RepackError.configurationInvalid(detail: """
                \(options.inputGTurbo) already has a vision tower; reinstall the \
                model to change it
                """)
        }
    }

    /// The same manifest with the tower declared. Every other field is carried
    /// through untouched, so this produces the identical bytes a fresh
    /// `--include-vision` install of the same checkpoint would write.
    static func manifestAddingVision(_ manifest: GTurboManifestV1,
                                     plan: VisionFilePlan,
                                     weights: GTurboManifestFileV1) throws -> GTurboManifestV1 {
        var flags = manifest.flags
        flags["visionTower"] = true
        var files = manifest.files
        guard files.updateValue(weights,
                                forKey: GTurboFormatV1.visionWeightsPath) == nil else {
            throw RepackError.configurationInvalid(
                detail: "manifest already declares \(GTurboFormatV1.visionWeightsPath)")
        }
        let config = plan.source.config
        return GTurboManifestV1(
            magic: manifest.magic,
            versionMajor: manifest.versionMajor,
            versionMinor: max(manifest.versionMinor, GTurboFormatV1.versionMinorVision),
            flags: flags,
            modelID: manifest.modelID,
            sourceSnapshotHash: manifest.sourceSnapshotHash,
            arch: manifest.arch,
            quant: manifest.quant,
            vision: GTurboManifestVisionV1(
                hiddenSize: config.hiddenSize,
                numLayers: config.numLayers,
                numHeads: config.numHeads,
                numKVHeads: config.numKVHeads,
                headDim: config.headDim,
                intermediateSize: config.intermediateSize,
                patchSize: config.patchSize,
                poolingKernelSize: config.poolingKernelSize,
                positionEmbeddingSize: config.positionEmbeddingSize,
                ropeTheta: config.ropeTheta,
                rmsNormEps: config.rmsNormEps,
                hiddenActivation: config.hiddenActivation,
                standardize: config.standardize,
                maxSoftTokens: config.maxSoftTokens,
                weightDType: "bf16",
                imageTokenID: config.imageTokenID,
                boiTokenID: config.boiTokenID,
                eoiTokenID: config.eoiTokenID,
                weightsPath: GTurboFormatV1.visionWeightsPath,
                tensorCount: plan.tensorCount,
                payloadBytes: plan.payloadBytes,
                sourceRepo: plan.source.repoID,
                sourceRevision: plan.source.revision),
            files: files,
            expertsPerLayer: manifest.expertsPerLayer,
            numLayers: manifest.numLayers,
            expertStride: manifest.expertStride,
            bitWidthOverridesHonored: manifest.bitWidthOverridesHonored)
    }

    // MARK: - Parity

    /// `PLAN_VISION.md` §1-2, run against the installed model rather than the
    /// source checkpoint: the tensors that survive quantization unchanged are
    /// in `model_weights.bin` under their original names, so both halves can
    /// still be hashed and compared without the snapshot they came from.
    private func verifyParity(vision: BoundVisionSource,
                              access: GTurboDirectoryAccess) throws {
        let entries = try residentIndex(access: access)
        var byName: [String: GTurboResidentIndexEntryV1] = [:]
        for entry in entries { byName[entry.name] = entry }

        let descriptor = try access.openFile(Self.textWeightsPath)
        defer { close(descriptor) }
        for parity in vision.pin.parityTensors {
            try Task.checkCancellation()
            guard let entry = byName[parity.textName] else {
                throw RepackError.configurationInvalid(detail: """
                    installed model has no \(parity.textName); it cannot be paired \
                    with the \(vision.pin.repoID) vision tower
                    """)
            }
            guard entry.dtype == GTurboFormatV1.DType.bf16.rawValue,
                  entry.scaleSize == 0, entry.biasSize == 0 else {
                throw RepackError.configurationInvalid(detail: """
                    parity tensor \(parity.textName) is not stored unquantized in \
                    \(Self.textWeightsPath)
                    """)
            }
            guard entry.sizeBytes <= 1024 * 1024 else {
                throw RepackError.configurationInvalid(
                    detail: "parity tensor \(parity.textName) is unexpectedly large")
            }
            let digest = try hashRange(descriptor: descriptor,
                                       offset: entry.fileOffset,
                                       size: Int(entry.sizeBytes))
            guard digest.lowercased() == parity.sha256.lowercased() else {
                throw RepackError.configurationInvalid(detail: """
                    text tensor \(parity.textName) hashes to \(digest), but the pinned \
                    vision tower expects \(parity.sha256); the two checkpoints differ
                    """)
            }
            guard vision.parityDigests[parity.textName]?.lowercased()
                    == digest.lowercased() else {
                throw RepackError.configurationInvalid(
                    detail: "vision source disagrees with the installed model on \(parity.textName)")
            }
        }
        audit.visionParityTensors = vision.pin.parityTensors.map(\.textName).sorted()
    }

    private func residentIndex(access: GTurboDirectoryAccess) throws
        -> [GTurboResidentIndexEntryV1] {
        let descriptor = try access.openFile(Self.textWeightsPath)
        defer { close(descriptor) }
        let path = "\(access.rootPath)/\(Self.textWeightsPath)"
        var headerBytes = [UInt8](repeating: 0, count: GTurboFormatV1.residentHeaderBytes)
        try headerBytes.withUnsafeMutableBytes {
            try Posix.preadAll(fd: descriptor, path: path,
                               buf: $0.baseAddress!, count: $0.count, offset: 0)
        }
        let header = try headerBytes.withUnsafeBytes {
            try GTurboResidentIndexCodec.decodeHeader($0)
        }
        guard header.indexSize <= GTurboFormatV1.residentIndexMaxBytes,
              header.indexSize >= UInt64(GTurboFormatV1.residentHeaderBytes) else {
            throw RepackError.configurationInvalid(
                detail: "\(Self.textWeightsPath) has an out-of-range resident index")
        }
        var indexBytes = [UInt8](repeating: 0, count: Int(header.indexSize))
        try indexBytes.withUnsafeMutableBytes {
            try Posix.preadAll(fd: descriptor, path: path,
                               buf: $0.baseAddress!, count: $0.count, offset: 0)
        }
        audit.recordRead(bytes: indexBytes.count)
        do {
            return try indexBytes.withUnsafeBytes {
                try GTurboResidentIndexCodec.decodeRegion($0, header: header)
            }
        } catch {
            throw RepackError.configurationInvalid(
                detail: "\(Self.textWeightsPath) resident index invalid: \(error)")
        }
    }

    private func hashRange(descriptor: Int32,
                           offset: UInt64,
                           size: Int) throws -> String {
        let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: max(size, 1),
                                                            alignment: 16)
        defer { buffer.deallocate() }
        try Posix.preadAll(fd: descriptor,
                           path: Self.textWeightsPath,
                           buf: buffer.baseAddress!,
                           count: size,
                           offset: offset)
        audit.recordRead(bytes: size)
        var stream = Sha256Stream()
        stream.update(UnsafeRawBufferPointer(rebasing: buffer[0..<size]))
        return stream.finalizeHexString()
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
