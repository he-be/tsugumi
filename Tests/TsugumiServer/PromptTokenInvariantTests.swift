import Foundation
import Testing
@testable import Tsugumi
@testable import TsugumiServerCore

/// C2 (CONFORMANCE §1): INV-1, **describing a turn again == generating it**.
///
/// A conversation that already holds a finished assistant turn, drawn again by
/// the template, must come out as the tokens that were in the KV when that turn
/// finished — prompt plus generation. When it does not, the prompt cache's
/// longest common prefix stops early and the session re-prefills work it
/// already did, every turn, for the rest of the session.
///
/// These need the tokenizer and nothing else: no weights, no Metal. The claim
/// is only ever about two token arrays.
@Suite("C2 prompt token invariants")
struct PromptTokenInvariantTests {
    struct Shape: CustomStringConvertible {
        let thinking: Bool
        let tools: Bool

        var description: String {
            "thinking=\(thinking) tools=\(tools)"
        }

        static let all = [Shape(thinking: false, tools: false),
                          Shape(thinking: false, tools: true),
                          Shape(thinking: true, tools: false),
                          Shape(thinking: true, tools: true)]
    }

    private static let answer = "Paris is the capital of France."
    private static let reasoning = "The user is asking about France."

    private static let toolsJSON = #"""
    ,"tools":[{"type":"function","function":{
      "name":"lookup","description":"Look something up",
      "parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}]
    """#

    private static func request(_ shape: Shape,
                                messages: String) throws -> ValidatedChatRequest {
        let body = #"{"model":"m","messages":["# + messages + "]"
            + (shape.tools ? toolsJSON : "")
            + #","chat_template_kwargs":{"enable_thinking":"# + "\(shape.thinking)}}"
        return try ChatRequestParser.parse(Data(body.utf8))
    }

    /// The tokens the model itself emits for the answer above, which is what
    /// the KV holds once the turn is done: the thought channel it opened (when
    /// it was reasoning), the answer, and the turn marker it stopped on.
    private static func generated(_ shape: Shape, _ tokenizer: GFTokenizer) -> [Int32] {
        var ids: [Int32] = []
        if shape.thinking {
            ids += [tokenizer.channelStartID]
            ids += tokenizer.encode("thought\n\(reasoning)\n", addBOS: false)
            ids += [tokenizer.channelEndID]
        }
        ids += tokenizer.encode(answer, addBOS: false)
        ids += [tokenizer.endOfTurnID]
        return ids
    }

    @Test("INV-1: a finished turn drawn again is the turn that was generated",
          arguments: Shape.all)
    func INV_1_redraw_equals_generation(shape: Shape) async throws {
        let tokenizer = try await GFTokenizer.load()
        let renderer = ServerPromptRenderer(tokenizer: tokenizer)

        let prompt = try renderer.promptIDs(Self.request(
            shape, messages: #"{"role":"user","content":"What is the capital of France?"}"#))
        let afterGenerating = prompt + Self.generated(shape, tokenizer)

        let reasoningField = shape.thinking
            ? #","reasoning_content":"# + "\"\(Self.reasoning)\""
            : ""
        let redrawn = try renderer.promptIDs(Self.request(shape, messages: """
            {"role":"user","content":"What is the capital of France?"},
            {"role":"assistant","content":"\(Self.answer)"\(reasoningField)},
            {"role":"user","content":"And of Italy?"}
            """))

        #expect(commonPrefixLength(afterGenerating, redrawn) == afterGenerating.count,
                "\(shape): the redraw diverges from the generated tokens")
    }

    /// The same claim stated the way the cache will read it: the second request
    /// of a session prefills only the turn it added.
    @Test("CACHE-1: the second turn's LCP covers everything the first turn left",
          arguments: Shape.all)
    func CACHE_1_second_turn_reuses_the_whole_first(shape: Shape) async throws {
        let tokenizer = try await GFTokenizer.load()
        let renderer = ServerPromptRenderer(tokenizer: tokenizer)

        let prompt = try renderer.promptIDs(Self.request(
            shape, messages: #"{"role":"user","content":"What is the capital of France?"}"#))
        let kv = prompt + Self.generated(shape, tokenizer)

        let reasoningField = shape.thinking
            ? #","reasoning_content":"# + "\"\(Self.reasoning)\""
            : ""
        let next = try renderer.promptIDs(Self.request(shape, messages: """
            {"role":"user","content":"What is the capital of France?"},
            {"role":"assistant","content":"\(Self.answer)"\(reasoningField)},
            {"role":"user","content":"And of Italy?"}
            """))

        let reused = commonPrefixLength(kv, next)
        let report = Comment(rawValue: "\(shape): reused \(reused) of \(kv.count) tokens, "
            + "so \(next.count - reused) of \(next.count) are prefilled again")
        #expect(reused == kv.count, report)
    }
}
