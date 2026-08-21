import Foundation

/// The Ornith (Qwen 3.5-MoE) sibling of `StructuredAssistantDecoder`: one
/// generated token at a time in, reasoning / content / tool calls out.
///
/// Three things make it a sibling rather than a branch of the Gemma decoder.
///
/// **The reasoning block is already open.** Gemma generates `<|channel>` and
/// the label that follows it. Ornith's template writes `<think>\n` into the
/// *generation prompt* when thinking is on, so the first generated token is
/// already reasoning and `</think>` is the only marker that will be seen —
/// hence `startsInReasoning`, read off the prompt by the caller.
///
/// **The markers are not special tokens.** `<think>`, `</think>`,
/// `<tool_call>`, `</tool_call>` are declared `special: false` in
/// `tokenizer.json`, so `QwenDetokenizer` emits their spelling as text instead
/// of dropping them. A marker's delta is therefore *held-back bytes, then the
/// marker's own spelling* — the spelling is stripped here and the bytes in
/// front of it are routed through the channel that was in effect before the
/// marker changed it.
///
/// **The body is XML.** It is collected as token ids and decoded once at
/// `</tool_call>`, the way Gemma's is, so a codepoint split across the closing
/// marker still lands inside the payload.
public final class QwenStructuredAssistantDecoder: @unchecked Sendable {
    private let tokenizer: QwenTokenizer
    private let parser: QwenToolCallParser
    private let emitsReasoning: Bool
    private let idGenerator: @Sendable () -> String
    /// Marker id → the text `QwenDetokenizer` appends for it.
    private let markerText: [Int32: String]

    private var insideReasoning: Bool
    private var toolTokens: [Int32]?
    private var emittedCalls = 0
    private var failed = false

    public init(tokenizer: QwenTokenizer,
                tools: [GFTokenizer.FunctionDefinition],
                emitsReasoning: Bool = false,
                startsInReasoning: Bool,
                idGenerator: @escaping @Sendable () -> String = {
                    "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                }) {
        self.tokenizer = tokenizer
        self.parser = QwenToolCallParser(tools: tools)
        self.emitsReasoning = emitsReasoning
        self.insideReasoning = startsInReasoning
        self.idGenerator = idGenerator
        var text: [Int32: String] = [:]
        for id in [tokenizer.thinkStartID, tokenizer.thinkEndID,
                   tokenizer.toolCallStartID, tokenizer.toolCallEndID,
                   tokenizer.toolResponseStartID, tokenizer.toolResponseEndID] {
            text[id] = tokenizer.token(for: id) ?? ""
        }
        self.markerText = text
    }

    /// Whether the prompt leaves the reasoning block open. The template opens
    /// it when thinking is on and closes it again when it is off, so the last
    /// marker in the prompt decides — a `<think>` quoted in an earlier turn
    /// cannot.
    public static func promptEndsInsideReasoning(_ ids: [Int32],
                                                 tokenizer: QwenTokenizer) -> Bool {
        for id in ids.reversed() {
            if id == tokenizer.thinkStartID { return true }
            if id == tokenizer.thinkEndID { return false }
        }
        return false
    }

    public func consume(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw QwenToolCallParserError.malformed }

        guard let marker = markerText[tokenID] else {
            if var tokens = toolTokens {
                tokens.append(tokenID)
                guard tokens.count * MemoryLayout<Int32>.size <= QwenToolCallParser.maximumBytes else {
                    failed = true
                    throw QwenToolCallParserError.oversized
                }
                toolTokens = tokens
                return []
            }
            return routeText(delta)
        }

        // Everything in front of the marker's own spelling is text the
        // detokenizer held back from *before* this token, so it belongs to the
        // channel in effect now — route it before the marker changes that.
        // Inside a tool body it is part of the payload, which is re-decoded
        // from its ids, so nothing is routed there.
        var events: [StructuredAssistantEvent] = []
        if toolTokens == nil {
            let held = delta.hasSuffix(marker) ? String(delta.dropLast(marker.count)) : delta
            events = routeText(held)
        }

        switch tokenID {
        case tokenizer.thinkStartID:
            insideReasoning = true
        case tokenizer.thinkEndID:
            insideReasoning = false
        case tokenizer.toolCallStartID:
            guard toolTokens == nil else {
                failed = true
                throw QwenToolCallParserError.malformed
            }
            toolTokens = []
        case tokenizer.toolCallEndID:
            guard let tokens = toolTokens else {
                failed = true
                throw QwenToolCallParserError.malformed
            }
            toolTokens = nil
            let body = tokenizer.decode(tokens, skipSpecialTokens: false)
            do {
                let call = try parser.parse(body, id: idGenerator())
                emittedCalls += 1
                return events + [.toolCall(call)]
            } catch {
                failed = true
                throw error
            }
        // A tool *response* is the client's turn, not the model's: seeing one
        // here means the model wrote the framing of a turn it does not own.
        case tokenizer.toolResponseStartID, tokenizer.toolResponseEndID:
            failed = true
            throw QwenToolCallParserError.malformed
        default:
            break
        }
        return events
    }

    /// Text the detokenizer held back at a stop boundary. It is not tied to a
    /// token id, so it cannot go through `consume`; without this a generation
    /// cut off inside the reasoning block would leak its held bytes into the
    /// answer.
    public func consumeTail(_ text: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw QwenToolCallParserError.malformed }
        guard toolTokens == nil, !text.isEmpty else { return [] }
        return routeText(text)
    }

    public func finish() throws {
        guard !failed, toolTokens == nil else {
            throw QwenToolCallParserError.malformed
        }
    }

    public var hasToolCalls: Bool { emittedCalls > 0 }
    /// Whether the reasoning block is currently open (the caller's flush needs
    /// to know which channel the tail belongs to).
    public var isInsideReasoning: Bool { insideReasoning }

    private func routeText(_ delta: String) -> [StructuredAssistantEvent] {
        guard !delta.isEmpty else { return [] }
        if insideReasoning {
            return emitsReasoning ? [.reasoning(delta)] : []
        }
        return [.content(delta)]
    }
}
