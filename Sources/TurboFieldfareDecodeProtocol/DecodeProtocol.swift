import Foundation

public struct DecodeRuntimeOptions: Codable, Sendable, Equatable {
    public var expertCacheSlots: Int
    public var expertCachePolicy: String
    public var prefillEnabled: Bool
    public var prefillChunkTokens: Int
    public var rdadvisePolicy: String
    public var modelVerification: String
    public var mtpEnabled: Bool

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: String = "lfu",
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 2048,
                rdadvisePolicy: String = "off",
                modelVerification: String = "full-sha256",
                mtpEnabled: Bool = true) {
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.prefillEnabled = prefillEnabled
        self.prefillChunkTokens = prefillChunkTokens
        self.rdadvisePolicy = rdadvisePolicy
        self.modelVerification = modelVerification
        self.mtpEnabled = mtpEnabled
    }

    private enum CodingKeys: String, CodingKey {
        case expertCacheSlots, expertCachePolicy, prefillEnabled
        case prefillChunkTokens, rdadvisePolicy, modelVerification, mtpEnabled
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        expertCacheSlots = try container.decode(Int.self, forKey: .expertCacheSlots)
        expertCachePolicy = try container.decode(String.self, forKey: .expertCachePolicy)
        prefillEnabled = try container.decode(Bool.self, forKey: .prefillEnabled)
        prefillChunkTokens = try container.decode(Int.self, forKey: .prefillChunkTokens)
        rdadvisePolicy = try container.decode(String.self, forKey: .rdadvisePolicy)
        modelVerification = try container.decode(String.self, forKey: .modelVerification)
        mtpEnabled = try container.decodeIfPresent(Bool.self, forKey: .mtpEnabled) ?? true
    }
}

public struct DecodeLoadRequest: Codable, Sendable {
    public var modelPath: String
    public var maxContextTokens: Int
    public var runtimeOptions: DecodeRuntimeOptions
    public var forceLogitsHead: Bool
    public var requestID: UUID

    public init(modelPath: String, maxContextTokens: Int,
                runtimeOptions: DecodeRuntimeOptions = DecodeRuntimeOptions(),
                forceLogitsHead: Bool = false,
                requestID: UUID = UUID()) {
        self.modelPath = modelPath
        self.maxContextTokens = maxContextTokens
        self.runtimeOptions = runtimeOptions
        self.forceLogitsHead = forceLogitsHead
        self.requestID = requestID
    }
}

/// One completed conversation turn carried over the wire. `reasoningText`
/// rides along on assistant turns so the service can redraw them exactly as
/// they were generated.
public struct DecodeChatTurn: Codable, Sendable, Equatable {
    public var role: String
    public var text: String
    public var reasoningText: String?
    public var imagePaths: [String]

    public init(role: String,
                text: String,
                reasoningText: String? = nil,
                imagePaths: [String] = []) {
        self.role = role
        self.text = text
        self.reasoningText = reasoningText
        self.imagePaths = imagePaths
    }

    private enum CodingKeys: String, CodingKey {
        case role, text, reasoningText, imagePaths
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        text = try container.decode(String.self, forKey: .text)
        reasoningText = try container.decodeIfPresent(String.self, forKey: .reasoningText)
        imagePaths = try container.decodeIfPresent([String].self, forKey: .imagePaths) ?? []
    }
}

public struct DecodeGenerationRequest: Codable, Sendable {
    public var history: [DecodeChatTurn]
    public var prompt: String
    public var maxNewTokens: Int
    public var maxContextTokens: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var enableThinking: Bool
    public var imagePaths: [String]
    public var runtimeOptions: DecodeRuntimeOptions
    public var generationID: UUID

    public init(history: [DecodeChatTurn] = [],
                prompt: String, maxNewTokens: Int, maxContextTokens: Int,
                temperature: Float, topK: Int? = nil, topP: Float? = nil,
                repetitionPenalty: Float = 1,
                enableThinking: Bool = false,
                imagePaths: [String] = [],
                runtimeOptions: DecodeRuntimeOptions = DecodeRuntimeOptions(),
                generationID: UUID = UUID()) {
        self.history = history
        self.prompt = prompt
        self.maxNewTokens = maxNewTokens
        self.maxContextTokens = maxContextTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.enableThinking = enableThinking
        self.imagePaths = imagePaths
        self.runtimeOptions = runtimeOptions
        self.generationID = generationID
    }

    private enum CodingKeys: String, CodingKey {
        case history, prompt, maxNewTokens, maxContextTokens, temperature, topK, topP
        case repetitionPenalty, enableThinking, imagePaths
        case runtimeOptions, generationID
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        history = try container.decodeIfPresent([DecodeChatTurn].self, forKey: .history) ?? []
        prompt = try container.decode(String.self, forKey: .prompt)
        maxNewTokens = try container.decode(Int.self, forKey: .maxNewTokens)
        maxContextTokens = try container.decode(Int.self, forKey: .maxContextTokens)
        temperature = try container.decode(Float.self, forKey: .temperature)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        topP = try container.decodeIfPresent(Float.self, forKey: .topP)
        repetitionPenalty = try container.decode(Float.self, forKey: .repetitionPenalty)
        enableThinking = try container.decodeIfPresent(Bool.self, forKey: .enableThinking) ?? false
        imagePaths = try container.decodeIfPresent([String].self, forKey: .imagePaths) ?? []
        runtimeOptions = try container.decode(DecodeRuntimeOptions.self, forKey: .runtimeOptions)
        generationID = try container.decode(UUID.self, forKey: .generationID)
    }
}

public enum DecodeServiceCommand: Codable, Sendable {
    case load(DecodeLoadRequest)
    case generate(DecodeGenerationRequest)
    case cancel
    case unload(UUID)
    case shutdown
}

public enum DecodeServiceEventKind: String, Codable, Sendable {
    case loading
    case ready
    case prefill
    case snapshot
    case finished
    case cancelled
    case failed
    case unloaded
}

public struct DecodeRunnerDiagnostics: Codable, Sendable, Equatable {
    public var cb1MillisecondsPerToken: Double
    public var ioMillisecondsPerToken: Double
    public var cb2MillisecondsPerToken: Double
    public var headMillisecondsPerToken: Double
    public var rdadviseMillisecondsPerToken: Double
    public var rdadviseCallsPerToken: Double
    public var rdadviseMegabytesPerToken: Double
    public var rdadviseSkippedPerToken: Double
    public var rdadviseFailures: UInt64

    public init(cb1MillisecondsPerToken: Double,
                ioMillisecondsPerToken: Double,
                cb2MillisecondsPerToken: Double,
                headMillisecondsPerToken: Double,
                rdadviseMillisecondsPerToken: Double,
                rdadviseCallsPerToken: Double,
                rdadviseMegabytesPerToken: Double,
                rdadviseSkippedPerToken: Double,
                rdadviseFailures: UInt64) {
        self.cb1MillisecondsPerToken = cb1MillisecondsPerToken
        self.ioMillisecondsPerToken = ioMillisecondsPerToken
        self.cb2MillisecondsPerToken = cb2MillisecondsPerToken
        self.headMillisecondsPerToken = headMillisecondsPerToken
        self.rdadviseMillisecondsPerToken = rdadviseMillisecondsPerToken
        self.rdadviseCallsPerToken = rdadviseCallsPerToken
        self.rdadviseMegabytesPerToken = rdadviseMegabytesPerToken
        self.rdadviseSkippedPerToken = rdadviseSkippedPerToken
        self.rdadviseFailures = rdadviseFailures
    }
}

public struct DecodePrefillDiagnostics: Codable, Sendable, Equatable {
    public var requestedMode: String
    public var executedMode: String
    public var kvStorageMode: String?
    public var chunkCompleteness: String
    public var unsupportedReason: String?

    public init(requestedMode: String, executedMode: String,
                kvStorageMode: String?, chunkCompleteness: String,
                unsupportedReason: String?) {
        self.requestedMode = requestedMode
        self.executedMode = executedMode
        self.kvStorageMode = kvStorageMode
        self.chunkCompleteness = chunkCompleteness
        self.unsupportedReason = unsupportedReason
    }
}

public struct DecodeServiceEvent: Codable, Sendable {
    public var kind: DecodeServiceEventKind
    public var generationID: UUID
    public var sequence: UInt64
    public var textDelta: String
    /// Thought-channel text, kept apart from `textDelta` so the app can
    /// render reasoning as reasoning. Optional for wire compatibility.
    public var reasoningDelta: String?
    public var tokenCount: Int
    public var promptTokenCount: Int?
    /// Prompt tokens served from the session's prompt cache.
    public var cachedPromptTokens: Int?
    /// Speculative-loop counters, present only when MTP actually ran.
    public var draftBlockTokens: Int?
    public var draftProposed: Int?
    public var draftAccepted: Int?
    public var prefillDone: Int?
    public var prefillTotal: Int?
    public var prefillSeconds: Double?
    public var timeToFirstTokenSeconds: Double?
    public var decodeSeconds: Double
    public var tokensPerSecond: Double
    public var stopReason: String?
    public var error: String?
    public var currentMemoryBytes: UInt64?
    public var peakMemoryBytes: UInt64?
    public var prefill: DecodePrefillDiagnostics?
    public var runner: DecodeRunnerDiagnostics?

    public init(kind: DecodeServiceEventKind, generationID: UUID,
                sequence: UInt64 = 0, textDelta: String = "",
                reasoningDelta: String? = nil,
                tokenCount: Int = 0, promptTokenCount: Int? = nil,
                cachedPromptTokens: Int? = nil,
                draftBlockTokens: Int? = nil,
                draftProposed: Int? = nil,
                draftAccepted: Int? = nil,
                prefillDone: Int? = nil, prefillTotal: Int? = nil,
                prefillSeconds: Double? = nil,
                timeToFirstTokenSeconds: Double? = nil,
                decodeSeconds: Double = 0, tokensPerSecond: Double = 0,
                stopReason: String? = nil, error: String? = nil,
                currentMemoryBytes: UInt64? = nil, peakMemoryBytes: UInt64? = nil,
                prefill: DecodePrefillDiagnostics? = nil,
                runner: DecodeRunnerDiagnostics? = nil) {
        self.kind = kind
        self.generationID = generationID
        self.sequence = sequence
        self.textDelta = textDelta
        self.reasoningDelta = reasoningDelta
        self.tokenCount = tokenCount
        self.promptTokenCount = promptTokenCount
        self.cachedPromptTokens = cachedPromptTokens
        self.draftBlockTokens = draftBlockTokens
        self.draftProposed = draftProposed
        self.draftAccepted = draftAccepted
        self.prefillDone = prefillDone
        self.prefillTotal = prefillTotal
        self.prefillSeconds = prefillSeconds
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.decodeSeconds = decodeSeconds
        self.tokensPerSecond = tokensPerSecond
        self.stopReason = stopReason
        self.error = error
        self.currentMemoryBytes = currentMemoryBytes
        self.peakMemoryBytes = peakMemoryBytes
        self.prefill = prefill
        self.runner = runner
    }
}

public enum DecodeFrameCodec {
    public static let maximumPayloadBytes = 4 * 1_024 * 1_024

    public static func encode<T: Encodable>(_ value: T) throws -> Data {
        let payload = try JSONEncoder().encode(value)
        guard payload.count <= maximumPayloadBytes else { throw DecodeFrameError.oversized }
        var length = UInt32(payload.count).littleEndian
        var frame = withUnsafeBytes(of: &length) { Data($0) }
        frame.append(payload)
        return frame
    }

    public static func read<T: Decodable>(_ type: T.Type, from handle: FileHandle) throws -> T {
        let header = try readExactly(4, from: handle)
        let count = header.withUnsafeBytes { raw -> UInt32 in
            raw.loadUnaligned(as: UInt32.self).littleEndian
        }
        guard count <= maximumPayloadBytes else { throw DecodeFrameError.oversized }
        let payload = try readExactly(Int(count), from: handle)
        return try JSONDecoder().decode(type, from: payload)
    }

    private static func readExactly(_ count: Int, from handle: FileHandle) throws -> Data {
        var result = Data()
        result.reserveCapacity(count)
        while result.count < count {
            guard let chunk = try handle.read(upToCount: count - result.count), !chunk.isEmpty else {
                throw DecodeFrameError.unexpectedEOF
            }
            result.append(chunk)
        }
        return result
    }
}

public enum DecodeFrameError: Error {
    case oversized
    case unexpectedEOF
}
