import CryptoKit
import Foundation
import TurboFieldfare

public enum ServerPromptCacheMode: String, Sendable, Equatable {
    case off
    case singlePrefix = "single-prefix"
}

struct ServerPromptCacheDomain: Sendable, Equatable {
    let modelID: String
    let sourceSnapshotHash: String?
    let runtimeProfileHash: String
    let maximumContext: Int
    let kvStorage: String
    let fp16RingEnabled: Bool
    let templateSHA256: String
}

struct CachedAssistantTurn: Sendable, Equatable {
    let message: GFTokenizer.Message
    let rawStopReason: StopReason
}

struct ServerPromptCacheEntry: Sendable, Equatable {
    let domain: ServerPromptCacheDomain
    let inputMessages: [GFTokenizer.Message]
    let tools: [GFTokenizer.FunctionDefinition]
    /// The thought-channel mode this KV was built with. A later request that
    /// disagrees is a miss: the cached system turn carries (or lacks) the
    /// `<|think|>` marker, and the bridge that continues it opens (or closes)
    /// the channel to match, so the two modes cannot share a prefix.
    let enableThinking: Bool
    /// One digest per image already in this KV, in prompt order.
    ///
    /// The rest of this entry is keyed on the *text* of the conversation, which
    /// cannot tell two pictures apart: without this, a second request with the
    /// same words and a different photograph would resume from a prefix that
    /// holds the first photograph's soft tokens and answer about it.
    let imageDigests: [String]
    let assistantTurn: CachedAssistantTurn
    let kvBackedTokenIDs: [Int32]
    let uncommittedBoundaryTokenIDs: [Int32]
    let kvPosition: Int
}

enum ServerPromptCacheMatch: Sendable, Equatable {
    case miss
    /// `vision` is the image side of the tokens still to be prefilled — the
    /// pictures this turn adds, with offsets into the continuation rather than
    /// into the whole prompt. nil when the continuation adds no picture, which
    /// includes every text-only session.
    case hit(effectivePromptIDs: [Int32],
             cachedPromptTokens: Int,
             vision: VisionPrefillInput? = nil)
}

struct ServerPromptCache: Sendable {
    private(set) var entry: ServerPromptCacheEntry?

    mutating func invalidate() {
        entry = nil
    }

    /// The per-image digests of a request, in prompt order.
    static func imageDigests(_ request: ValidatedChatRequest) -> [String] {
        (request.vision?.images ?? []).map { attachment in
            SHA256.hash(data: attachment.data)
                .map { String(format: "%02x", $0) }
                .joined()
        }
    }

    mutating func publish(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        stopStringFiltered: Bool = false
    ) {
        guard result.kvPosition == result.kvBackedTokenIDs.count,
              !result.kvBackedTokenIDs.isEmpty,
              result.uncommittedBoundaryTokenIDs.count == 1,
              !stopStringFiltered,
              result.reason == .endOfTurn
                || result.reason == .toolCalls
                || result.reason == .maxTokens else {
            entry = nil
            return
        }
        let historicalCalls = calls.map {
            GFTokenizer.HistoricalToolCall(
                id: $0.id,
                name: $0.name,
                arguments: $0.arguments)
        }
        let assistant = GFTokenizer.Message(
            role: .assistant,
            content: calls.isEmpty ? content : nil,
            toolCalls: historicalCalls)
        entry = ServerPromptCacheEntry(
            domain: domain,
            inputMessages: request.messages,
            tools: request.tools,
            enableThinking: request.enableThinking,
            imageDigests: Self.imageDigests(request),
            assistantTurn: CachedAssistantTurn(
                message: assistant,
                rawStopReason: result.reason),
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs,
            kvPosition: result.kvPosition)
    }

    /// - Parameter vision: the image side of the freshly rendered prompt, whose
    ///   `images` are the preprocessed pictures of the *whole* conversation in
    ///   order. A continuation takes the trailing ones — the pictures the new
    ///   turn adds — and rebases them onto the bridge.
    func match(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: GFTokenizer,
        vision: VisionPrefillInput? = nil
    ) -> ServerPromptCacheMatch {
        let digests = Self.imageDigests(request)
        guard let entry,
              entry.domain == domain,
              entry.tools == request.tools,
              entry.enableThinking == request.enableThinking,
              entry.kvPosition == entry.kvBackedTokenIDs.count,
              entry.kvPosition > 0,
              entry.uncommittedBoundaryTokenIDs.count == 1,
              // The pictures this KV holds have to be the first pictures of the
              // conversation being continued, in the same order.
              digests.count >= entry.imageDigests.count,
              digests.starts(with: entry.imageDigests) else {
            return .miss
        }

        if renderedPromptIDs.count > entry.kvPosition,
           renderedPromptIDs.prefix(entry.kvPosition)
            .elementsEqual(entry.kvBackedTokenIDs) {
            // The rendered prompt already has its placeholders widened, so the
            // spans are offsets into the whole prompt: only a request whose
            // pictures all sit inside the served prefix can take this branch.
            guard (vision?.spans ?? []).allSatisfy({ $0.tokenEnd <= entry.kvPosition }) else {
                return .miss
            }
            return .hit(
                effectivePromptIDs: renderedPromptIDs,
                cachedPromptTokens: entry.kvPosition)
        }

        let inputCount = entry.inputMessages.count
        guard request.messages.count > inputCount + 1,
              request.messages.prefix(inputCount)
                .elementsEqual(entry.inputMessages),
              assistantMatches(
                request.messages[inputCount],
                entry.assistantTurn.message) else {
            return .miss
        }
        let continuation = Array(request.messages.dropFirst(inputCount + 1))

        if entry.assistantTurn.message.toolCalls.isEmpty {
            return matchTextContinuation(
                entry: entry,
                request: request,
                continuation: continuation,
                tokenizer: tokenizer,
                vision: vision,
                addedImages: digests.count - entry.imageDigests.count)
        }
        // A tool result never carries a picture (images are user-turn only), so
        // a tool continuation that added one is not something this bridge can
        // describe.
        guard digests.count == entry.imageDigests.count else { return .miss }
        return matchToolContinuation(
            entry: entry,
            request: request,
            continuation: continuation,
            tokenizer: tokenizer,
            enableThinking: request.enableThinking)
    }

    private func assistantMatches(
        _ incoming: GFTokenizer.Message,
        _ cached: GFTokenizer.Message
    ) -> Bool {
        guard incoming.role == .assistant,
              cached.role == .assistant,
              incoming.toolCalls == cached.toolCalls,
              incoming.toolCallID == cached.toolCallID,
              incoming.name == cached.name else {
            return false
        }
        if !cached.toolCalls.isEmpty {
            return (incoming.content ?? "").isEmpty
                && (cached.content ?? "").isEmpty
        }
        return incoming.content == cached.content
    }

    private func matchTextContinuation(
        entry: ServerPromptCacheEntry,
        request: ValidatedChatRequest,
        continuation: [GFTokenizer.Message],
        tokenizer: GFTokenizer,
        vision: VisionPrefillInput?,
        addedImages: Int
    ) -> ServerPromptCacheMatch {
        guard continuation.count == 1,
              continuation[0].role == .user,
              continuation[0].content != nil,
              continuation[0].toolCalls.isEmpty,
              continuation[0].toolCallID == nil,
              entry.assistantTurn.rawStopReason == .endOfTurn
                || entry.assistantTurn.rawStopReason == .maxTokens else {
            return .miss
        }
        // The bodies with their image parts, when the request carried any. The
        // text projection is what everything else here compares, but the bridge
        // has to render what the turn actually holds.
        let parts: [GFTokenizer.ContentPart]
        if let multimodal = request.vision?.messages, multimodal.count == request.messages.count {
            parts = multimodal[multimodal.count - 1].parts
        } else if addedImages == 0, let content = continuation[0].content {
            parts = [.text(content)]
        } else {
            return .miss
        }
        guard parts.filter({ $0 == .image }).count == addedImages else { return .miss }
        // A rejected continuation (today: literal media markers in the text) is
        // a cache miss, not an error here. The request still has to go through
        // the normal encode path, which raises the typed error with the context
        // the caller needs; failing inside the cache probe would report it as a
        // caching problem instead.
        guard var bridge = try? tokenizer.encodeContinuation(
            parts: parts, enableThinking: entry.enableThinking) else {
            return .miss
        }
        if entry.assistantTurn.rawStopReason == .maxTokens {
            bridge = entry.uncommittedBoundaryTokenIDs + bridge
        } else if bridge.first != entry.uncommittedBoundaryTokenIDs.first {
            return .miss
        }
        guard addedImages > 0 else {
            return .hit(
                effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
                cachedPromptTokens: entry.kvPosition)
        }
        // Widen this turn's placeholders inside the bridge, so the spans that
        // come back are offsets into the slice the resumed prefill will run.
        guard let images = vision?.images, images.count >= addedImages,
              let ids = try? VisionMediaTokenIDs(tokenizer: tokenizer),
              let expanded = try? VisionPromptAssembler.makePrefillPrompt(
                  tokens: bridge,
                  images: Array(images.suffix(addedImages)),
                  ids: ids) else {
            return .miss
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + expanded.tokens,
            cachedPromptTokens: entry.kvPosition,
            vision: expanded.vision)
    }

    private func matchToolContinuation(
        entry: ServerPromptCacheEntry,
        request: ValidatedChatRequest,
        continuation: [GFTokenizer.Message],
        tokenizer: GFTokenizer,
        enableThinking: Bool
    ) -> ServerPromptCacheMatch {
        let calls = entry.assistantTurn.message.toolCalls
        guard entry.assistantTurn.rawStopReason == .toolCalls,
              continuation.count == calls.count,
              zip(continuation, calls).allSatisfy({ message, call in
                  message.role == .tool
                    && message.toolCallID == call.id
                    && (message.name == nil || message.name == call.name)
                    && message.content != nil
                    && message.toolCalls.isEmpty
              }) else {
            return .miss
        }
        guard let bridge = try? tokenizer.encodeToolResultContinuation(
            cachedMessages: entry.inputMessages,
            assistant: entry.assistantTurn.message,
            incomingMessages: request.messages,
            tools: request.tools,
            enableThinking: enableThinking),
              bridge.first == entry.uncommittedBoundaryTokenIDs.first else {
            return .miss
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }
}
