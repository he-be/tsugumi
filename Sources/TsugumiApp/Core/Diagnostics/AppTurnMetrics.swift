import Darwin
import Foundation

/// One line per generation round, written when the round ends: what the
/// machine looked like when it started (the speedometer, and how much of
/// the weights were really in memory) next to what the round cost. Ordinary
/// use then accumulates the evidence for "more memory, faster" as a
/// scatter plot instead of a lab table (docs/MAC_APP.md §4e).
public struct AppTurnMetricsRecord: Codable, Equatable, Sendable {
    public var recordedAt: String
    public var turnID: String
    public var round: Int
    public var chatID: String
    public var outcome: String
    public var model: String
    public var contextTokens: Int
    public var slots: Int
    public var mtp: Bool
    public var thinking: Bool
    public var network: String
    public var directive: String?

    // The machine at the round's start.
    public var headroomLevel: Double?
    public var borrowableBytes: UInt64?
    public var wantedBytes: UInt64?
    public var appBytes: UInt64?
    public var wiredBytes: UInt64?
    public var compressedBytes: UInt64?
    public var cachedFileBytes: UInt64?
    /// What the runtime held for itself (KV cache, resident weights and experts).
    public var runtimeOwnBytes: UInt64?
    /// `mincore` over the expert files: the share of pages in memory.
    public var weightsResidentFraction: Double?

    // What the round cost.
    public var promptTokens: Int?
    public var cachedPromptTokens: Int?
    public var prefillSeconds: Double?
    public var timeToFirstTokenSeconds: Double?
    public var generatedTokens: Int
    public var decodeSeconds: Double
    public var tokensPerSecond: Double
    public var draftProposed: Int?
    public var draftAccepted: Int?
    public var stopReason: String
    public var toolCalls: Int
    public var ioMillisecondsPerToken: Double?
    public var cb1MillisecondsPerToken: Double?
    public var cb2MillisecondsPerToken: Double?
    public var rdadviseMegabytesPerToken: Double?
    public var peakMemoryBytes: UInt64?

    public init(recordedAt: String, turnID: String, round: Int, chatID: String,
                outcome: String, model: String, contextTokens: Int, slots: Int,
                mtp: Bool, thinking: Bool, network: String, directive: String?,
                headroom: AppMachineHeadroom?, weightsResidentFraction: Double?,
                diagnostics: AppDiagnostics, toolCalls: Int) {
        self.recordedAt = recordedAt
        self.turnID = turnID
        self.round = round
        self.chatID = chatID
        self.outcome = outcome
        self.model = model
        self.contextTokens = contextTokens
        self.slots = slots
        self.mtp = mtp
        self.thinking = thinking
        self.network = network
        self.directive = directive
        self.headroomLevel = headroom?.level
        self.borrowableBytes = headroom?.borrowableBytes
        self.wantedBytes = headroom?.wantedBytes
        self.appBytes = headroom?.host.appBytes
        self.wiredBytes = headroom?.host.wiredBytes
        self.compressedBytes = headroom?.host.compressedBytes
        self.cachedFileBytes = headroom?.host.cachedFileBytes
        self.runtimeOwnBytes = headroom?.ownBytes
        self.weightsResidentFraction = weightsResidentFraction
        self.promptTokens = diagnostics.promptTokenCount
        self.cachedPromptTokens = diagnostics.cachedPromptTokens
        self.prefillSeconds = diagnostics.prefillSeconds
        self.timeToFirstTokenSeconds = diagnostics.timeToFirstTokenSeconds
        self.generatedTokens = diagnostics.generatedTokens
        self.decodeSeconds = diagnostics.decodeSeconds
        self.tokensPerSecond = diagnostics.tokensPerSecond
        self.draftProposed = diagnostics.speculative?.proposed
        self.draftAccepted = diagnostics.speculative?.accepted
        self.stopReason = diagnostics.stopReason.rawValue
        self.toolCalls = toolCalls
        self.ioMillisecondsPerToken = diagnostics.runner?.ioMillisecondsPerToken
        self.cb1MillisecondsPerToken = diagnostics.runner?.cb1MillisecondsPerToken
        self.cb2MillisecondsPerToken = diagnostics.runner?.cb2MillisecondsPerToken
        self.rdadviseMegabytesPerToken = diagnostics.runner?.rdadviseMegabytesPerToken
        self.peakMemoryBytes = diagnostics.peakMemoryBytes
    }
}

/// Appends records as JSON Lines. One file for the app; nothing is ever
/// rewritten, so a reader can tail it while the app runs.
public final class AppTurnMetricsLog: @unchecked Sendable {
    public let fileURL: URL
    private let lock = NSLock()
    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }()

    public init(fileURL: URL) {
        self.fileURL = fileURL
    }

    /// `~/Library/Application Support/Tsugumi/turn-metrics.jsonl`.
    public static var defaultFileURL: URL {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return base
            .appendingPathComponent("Tsugumi", isDirectory: true)
            .appendingPathComponent("turn-metrics.jsonl")
    }

    public func append(_ record: AppTurnMetricsRecord) {
        guard var line = try? encoder.encode(record) else { return }
        line.append(0x0A)
        lock.lock()
        defer { lock.unlock() }
        let directory = fileURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if let handle = try? FileHandle(forWritingTo: fileURL) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: line)
        } else {
            try? line.write(to: fileURL)
        }
    }

    /// Every record in the file, for tests and for tools that read it back.
    public static func read(from fileURL: URL) -> [AppTurnMetricsRecord] {
        guard let data = try? Data(contentsOf: fileURL) else { return [] }
        let decoder = JSONDecoder()
        return data.split(separator: 0x0A).compactMap { try? decoder.decode(AppTurnMetricsRecord.self, from: $0) }
    }

    public static func isoTimestamp(_ date: Date = Date()) -> String {
        ISO8601DateFormatter().string(from: date)
    }
}

/// How much of the streamed weights is in memory right now, measured
/// rather than predicted: each expert file is mapped read-only (nothing is
/// touched, so the mapping costs no memory) and `mincore` counts the pages
/// the page cache holds. Nil when the directory has no expert files.
public enum AppWeightResidencyProbe {
    public static func residentFraction(modelDirectory: URL) -> Double? {
        let experts = modelDirectory.appendingPathComponent("packed_experts", isDirectory: true)
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: experts.path) else { return nil }
        let pageSize = Int(sysconf(_SC_PAGESIZE))
        var totalPages = 0
        var residentPages = 0
        for name in names.sorted() where name.hasSuffix(".bin") {
            let path = experts.appendingPathComponent(name).path
            let fd = open(path, O_RDONLY)
            guard fd >= 0 else { continue }
            defer { close(fd) }
            var info = stat()
            guard fstat(fd, &info) == 0, info.st_size > 0 else { continue }
            let length = Int(info.st_size)
            guard let base = mmap(nil, length, PROT_READ, MAP_SHARED, fd, 0), base != MAP_FAILED else { continue }
            defer { munmap(base, length) }
            let pages = (length + pageSize - 1) / pageSize
            var vec = [CChar](repeating: 0, count: pages)
            guard mincore(base, length, &vec) == 0 else { continue }
            totalPages += pages
            residentPages += vec.reduce(0) { $0 + (($1 & CChar(MINCORE_INCORE)) != 0 ? 1 : 0) }
        }
        guard totalPages > 0 else { return nil }
        return Double(residentPages) / Double(totalPages)
    }
}
