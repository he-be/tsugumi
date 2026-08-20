import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// C2 (CONFORMANCE §1) for SPEC §6: the one half of the wiring that needs real
/// tokens but no weights — the grammar a realistic `tools` request plans really
/// loads, and a canonical tool call really walks through the constraint the
/// server builds from it, token by token.
///
/// Everything below this (the sampler, the rejection redraw, the KV) needs a
/// model and belongs to C3.
///
/// The first run fetches the Gemma 4 IT tokenizer (~32 MB) into
/// `~/.cache/huggingface/`; after that it is offline.
@Suite("C2 server grammar wiring")
struct ServerGrammarWiringTests {
    let tok: GFTokenizer
    let vocab: GrammarVocabulary
    let markers: ChatGrammarMarkers

    init() async throws {
        self.tok = try await GFTokenizer.load()
        self.vocab = GrammarVocabulary.shared(for: tok)
        self.markers = ChatGrammarMarkers(tokenizer: tok)
    }

    /// `get_weather(city: string, days: integer)`, both required — the shape an
    /// agent client actually sends.
    private static let toolsField = #""tools":[{"type":"function","function":{"#
        + #""name":"get_weather","description":"current weather","parameters":{"#
        + #""type":"object","properties":{"city":{"type":"string"},"#
        + #""days":{"type":"integer"}},"required":["city","days"]}}}]"#

    private func makePlan(_ parts: String...) throws -> ServerGenerationPlan {
        let tail = parts.filter { !$0.isEmpty }.map { "," + $0 }.joined()
        let body = Data((#"{"model":"m","messages":[{"role":"user","content":"hi"}]"#
            + tail + "}").utf8)
        return ServerGenerationPlan(request: try ChatRequestParser.parse(body),
                                    markers: markers)
    }

    @Test("GEN-5: the markers name the tokenizer's tool-call start token")
    func GEN_5_the_trigger_is_the_tokenizers_tool_call_start_token() throws {
        let plan = try makePlan(Self.toolsField)
        #expect(plan.isLazy)
        let trigger = try #require(plan.trigger)
        #expect(trigger.tokenID == tok.toolCallStartID)
        #expect(trigger.text == ChatGrammarMarkers.gemmaToolCallStart)
    }

    @Test("GEN-1: a realistic tools request plans a grammar that loads")
    func GEN_1_a_realistic_tools_request_plans_a_grammar_that_loads() throws {
        let plan = try makePlan(Self.toolsField)
        let text = try #require(plan.grammar)
        let grammar = try GBNFGrammar(text)
        #expect(!grammar.rules.isEmpty)
        // Same for the response-format half of GEN-3, which is the other
        // grammar the server hands to the same constructor.
        let json = try makePlan(
            #""response_format":{"type":"json_schema","json_schema":{"name":"r","#
            + #""schema":{"type":"object","properties":{"answer":{"type":"string"}},"#
            + #""required":["answer"]}}}"#)
        _ = try GBNFGrammar(try #require(json.grammar))
    }

    @Test("GEN-1/GEN-5/GEN-8: a canonical tool call walks through the planned constraint")
    func GEN_1_a_canonical_tool_call_walks_through_the_planned_constraint() throws {
        let plan = try makePlan(Self.toolsField)
        let trigger = try #require(plan.trigger)
        let constraint = try GrammarTokenConstraint(
            try #require(plan.grammar),
            vocabulary: vocab,
            trigger: .token(trigger.tokenID))

        // GEN-5: nothing is constrained until the trigger fires.
        #expect(!constraint.isArmed)
        for id in tok.encode("let me look that up", addBOS: false) {
            #expect(constraint.allows(tokenID: id))
            try constraint.accept(tokenID: id)
            #expect(!constraint.isArmed)
        }

        try constraint.accept(tokenID: tok.toolCallStartID)
        #expect(constraint.isArmed)
        #expect(!constraint.mayEndHere)

        let quote = try #require(singleToken(#"<|"|>"#),
                                 #"the vocabulary has no single <|"|> token"#)
        // `<|tool_call>call:get_weather{city:<|"|>Kyoto<|"|>,days:3}<tool_call|>`
        // — GEN-8's canonical form: no whitespace, bare keys, ascending order.
        var walked: [Int32] = []
        walked += tok.encode("call:get_weather{city:", addBOS: false)
        walked.append(quote)
        walked += tok.encode("Kyoto", addBOS: false)
        walked.append(quote)
        walked += tok.encode(",days:3}", addBOS: false)
        for id in walked {
            #expect(constraint.allows(tokenID: id),
                    "the grammar rejected \(id) (\(tok.decode([id], skipSpecialTokens: false)))")
            try constraint.accept(tokenID: id)
        }
        #expect(constraint.allows(tokenID: tok.toolCallEndID))
        try constraint.accept(tokenID: tok.toolCallEndID)
        // parallel_tool_calls defaults to true, so a second call may follow —
        // but the sequence may also stop here.
        #expect(constraint.mayEndHere)
    }

    @Test("GEN-6: a trigger inside the thought block never arms the planned constraint")
    func GEN_6_a_trigger_inside_the_thought_block_never_arms() throws {
        let plan = try makePlan(Self.toolsField)
        let trigger = try #require(plan.trigger)
        let constraint = try GrammarTokenConstraint(
            try #require(plan.grammar),
            vocabulary: vocab,
            trigger: .token(trigger.tokenID))
        let watch = ServerThoughtSuppression(tokenizer: tok)

        // The token stream a thinking turn writes, driven exactly as
        // `ServerModelSession` drives it: the gate accepts, then the decoder's
        // verdict for that token moves the suppression.
        func step(_ tokenID: Int32, _ events: [StructuredAssistantEvent]) throws {
            try constraint.accept(tokenID: tokenID)
            constraint.setSuppressed(watch.observe(tokenID: tokenID, events: events))
        }
        try step(tok.channelStartID, [])
        #expect(constraint.isSuppressed)
        try step(tok.toolCallStartID, [])
        #expect(!constraint.isArmed, "a trigger inside the thought block armed the grammar")
        // The close releases, and the call right after it is constrained.
        try step(tok.channelEndID, [.reasoning("…so I will look it up")])
        #expect(!constraint.isSuppressed)
        try step(tok.toolCallStartID, [])
        #expect(constraint.isArmed)
        #expect(!constraint.mayEndHere)
    }

    private func singleToken(_ marker: String) -> Int32? {
        guard let id = tok.tokenizer.convertTokenToId(marker),
              tok.tokenizer.convertIdToToken(id) == marker,
              let value = Int32(exactly: id) else { return nil }
        return value
    }
}
