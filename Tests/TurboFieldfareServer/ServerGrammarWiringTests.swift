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

    /// The GEN-4-required failure this rule exists for: the grammar used to
    /// spell the section markers as **text**, and their spelling is also
    /// reachable as a run of ordinary tokens. Asked for a tool call on a turn
    /// where it wanted to answer in prose, the model took that door — it wrote
    /// `<`, `|`, `tool`, … for the opener and the real `<tool_call|>` token for
    /// the closer, and `StructuredAssistantDecoder`, which knows a call by its
    /// token ids, saw a section end with no section start and failed the
    /// request with a 500. With the markers as `TOKEN` elements the door is
    /// gone: after the thought block the marker token is the only move.
    @Test("GEN-4/GEN-8: required leaves the marker token as the only way to open a call")
    func GEN_4_required_can_only_open_a_call_with_the_marker_token() throws {
        let plan = try makePlan(Self.toolsField, #""tool_choice":"required""#)
        #expect(!plan.isLazy)
        let constraint = try GrammarTokenConstraint(
            try #require(plan.grammar), vocabulary: vocab)

        // GEN-13: the thought block the model opens for itself when thinking
        // is on, then its close — after which the call body is due.
        try constraint.accept(tokenID: tok.channelStartID)
        for id in tok.encode("thought\nThe user only said hello.", addBOS: false) {
            try constraint.accept(tokenID: id)
        }
        try constraint.accept(tokenID: tok.channelEndID)

        var allowed = [Bool](repeating: false, count: tok.vocabSize)
        try allowed.withUnsafeMutableBufferPointer { try constraint.fillAllowedMask($0) }
        let ids = (0..<tok.vocabSize).filter { allowed[$0] }.map(Int32.init)
        #expect(ids == [tok.toolCallStartID],
                "\(ids.map { "\($0)=\(tok.decode([$0], skipSpecialTokens: false))" })")
        // The single-token probe agrees, and the first token of the marker's
        // spelling — the one the model actually drew — is not among them.
        #expect(constraint.allows(tokenID: tok.toolCallStartID))
        for id in tok.encode("<", addBOS: false) {
            #expect(!constraint.allows(tokenID: id))
        }
        // The closer is the marker token too, never its spelling.
        try constraint.accept(tokenID: tok.toolCallStartID)
        let quote = try #require(singleToken(#"<|"|>"#))
        var body: [Int32] = tok.encode("call:get_weather{city:", addBOS: false)
        body.append(quote)
        body += tok.encode("Kyoto", addBOS: false)
        body.append(quote)
        body += tok.encode(",days:3}", addBOS: false)
        for id in body { try constraint.accept(tokenID: id) }
        for id in tok.encode("<", addBOS: false) {
            #expect(!constraint.allows(tokenID: id))
        }
        #expect(constraint.allows(tokenID: tok.toolCallEndID))
        try constraint.accept(tokenID: tok.toolCallEndID)
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
