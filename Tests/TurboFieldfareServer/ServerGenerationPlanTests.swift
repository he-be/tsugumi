import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// C0/C1 (CONFORMANCE §1) for SPEC §6: the decision `ServerInference` makes
/// before it touches the model — is this request constrained, by which grammar,
/// lazily or not, and may the speculative loop still run (DEV-14).
///
/// Everything here is a pure function of a `ValidatedChatRequest` plus the
/// tokenizer's markers, so no weights, no Metal and no tokenizer are involved.
/// The requests are built by the real parser rather than by hand, which is also
/// what makes GEN-12's refusal observable from here: the collision never
/// reaches the plan because the parser refuses it first.
@Suite("C0 server generation plan")
struct ServerGenerationPlanTests {
    // MARK: - Fixtures

    /// The markers as this model writes them, injected so the suite needs no
    /// tokenizer. `7` and `8` stand in for `<|tool_call>` / `<tool_call|>`.
    private static let markers = ChatGrammarMarkers(
        toolCallStart: "<|tool_call>",
        toolCallEnd: "<tool_call|>",
        toolCallStartTokenID: 7,
        toolCallEndTokenID: 8)

    private static let declaredTools = #""tools":[{"type":"function","function":{"#
        + #""name":"lookup","description":"","parameters":{"type":"object","#
        + #""properties":{"q":{"type":"string"}}}}},"#
        + #"{"type":"function","function":{"name":"clock","description":"",""#
        + #"parameters":{"type":"object","properties":{}}}}]"#

    private static func body(_ parts: [String]) -> Data {
        let tail = parts.filter { !$0.isEmpty }.map { "," + $0 }.joined()
        return Data((#"{"model":"m","messages":[{"role":"user","content":"hi"}]"#
            + tail + "}").utf8)
    }

    private static func plan(_ parts: String...) throws -> ServerGenerationPlan {
        let request = try ChatRequestParser.parse(body(parts))
        return ServerGenerationPlan(request: request, markers: markers)
    }

    // MARK: - GEN-4: the four tool_choice values

    @Test("GEN-4: none plans no constraint at all")
    func GEN_4_tool_choice_none_plans_no_constraint() throws {
        let plan = try Self.plan(Self.declaredTools, #""tool_choice":"none""#)
        #expect(plan.grammar == nil)
        #expect(!plan.isConstrained)
        #expect(plan.trigger == nil)
    }

    @Test("GEN-4/GEN-5: auto plans a lazy grammar triggered by the tool-call start token")
    func GEN_4_GEN_5_tool_choice_auto_plans_a_lazy_grammar() throws {
        let plan = try Self.plan(Self.declaredTools)
        let grammar = try #require(plan.grammar)
        #expect(grammar.contains(#"<[7]> "call:lookup""#))
        #expect(plan.isLazy)
        #expect(plan.trigger == ChatGrammarTrigger(tokenID: 7, text: "<|tool_call>"))
    }

    @Test("GEN-4: required plans a grammar that is armed from the first token")
    func GEN_4_tool_choice_required_plans_a_non_lazy_grammar() throws {
        let plan = try Self.plan(Self.declaredTools, #""tool_choice":"required""#)
        let grammar = try #require(plan.grammar)
        #expect(grammar.contains(#"<[7]> "call:lookup""#))
        #expect(!plan.isLazy)
        #expect(plan.trigger == nil)
    }

    @Test("GEN-4/DEV-17: a named tool_choice plans only that function")
    func GEN_4_DEV_17_named_tool_choice_pins_that_function() throws {
        let plan = try Self.plan(
            Self.declaredTools,
            #""tool_choice":{"type":"function","function":{"name":"clock"}}"#)
        let grammar = try #require(plan.grammar)
        #expect(grammar.contains(#"<[7]> "call:clock""#))
        #expect(!grammar.contains("lookup"))
        #expect(!plan.isLazy)
    }

    @Test("GEN-4: a request with no tools at all plans no constraint")
    func GEN_4_no_tools_plans_no_constraint() throws {
        #expect(try Self.plan().grammar == nil)
    }

    // MARK: - GEN-3: the three response_format types

    @Test("GEN-3: text plans no constraint")
    func GEN_3_response_format_text_plans_no_constraint() throws {
        let plan = try Self.plan(#""response_format":{"type":"text"}"#)
        #expect(plan.grammar == nil)
        #expect(plan.allowsSpeculativeDecoding)
    }

    @Test("GEN-3: json_schema plans a non-lazy grammar built from its schema")
    func GEN_3_response_format_json_schema_plans_a_non_lazy_grammar() throws {
        let plan = try Self.plan(
            #""response_format":{"type":"json_schema","json_schema":{"name":"r","#
            + #""schema":{"type":"object","properties":{"answer":{"type":"string"}},"#
            + #""required":["answer"]}}}"#)
        let grammar = try #require(plan.grammar)
        #expect(grammar.contains("root ::="))
        #expect(grammar.contains("answer-kv ::="))
        #expect(!plan.isLazy)
        #expect(plan.trigger == nil)
    }

    @Test("GEN-3/DEV-18: json_object without a schema still plans a grammar")
    func GEN_3_DEV_18_json_object_without_a_schema_still_plans_a_grammar() throws {
        let plan = try Self.plan(#""response_format":{"type":"json_object"}"#)
        let grammar = try #require(plan.grammar)
        #expect(grammar.contains("root ::="))
        #expect(!plan.isLazy)
    }

    @Test("GEN-3: json_object with a schema uses it")
    func GEN_3_json_object_with_a_schema_uses_it() throws {
        let plan = try Self.plan(
            #""response_format":{"type":"json_object","schema":{"type":"object","#
            + #""properties":{"city":{"type":"string"}},"required":["city"]}}"#)
        let grammar = try #require(plan.grammar)
        #expect(grammar.contains("city-kv ::="))
    }

    // MARK: - GEN-12: the collision never reaches the plan

    @Test("GEN-12: a response format with a forced tool choice is a 400 before the plan")
    func GEN_12_a_forced_tool_choice_with_a_response_format_never_reaches_the_plan() throws {
        for choice in [#""tool_choice":"required""#,
                       #""tool_choice":{"type":"function","function":{"name":"lookup"}}"#] {
            let data = Self.body([Self.declaredTools,
                                  choice,
                                  #""response_format":{"type":"json_object"}"#])
            let error = #expect(throws: ServerRequestError.self) {
                _ = try ChatRequestParser.parse(data)
            }
            #expect(try #require(error).envelope.error.code
                == "response_format_conflicts_with_tool_choice")
        }
    }

    @Test("GEN-12: with auto the response format wins and the tool grammar is dropped")
    func GEN_12_auto_lets_the_response_format_win() throws {
        let plan = try Self.plan(Self.declaredTools,
                                 #""response_format":{"type":"json_object"}"#)
        let grammar = try #require(plan.grammar)
        #expect(!grammar.contains("tool_call"))
        #expect(!plan.isLazy, "a response-format grammar is never lazy")
    }

    // MARK: - GEN-14: a constrained request keeps the speculative path

    /// SPEC §6 **GEN-14**. The grammar left DEV-14's list on 2026-08-21: the
    /// reference verifies each position with the grammar applied and accepts
    /// as it goes, so a constraint is no longer a reason to drop the whole
    /// request onto the plain path. The everyday client declares `tools` on every
    /// request, so this line is what decides whether that client gets MTP at
    /// all (CONFORMANCE §5).
    @Test("GEN-14: a constrained plan keeps the speculative path")
    func GEN_14_a_constrained_plan_keeps_the_speculative_path() throws {
        for parts in [[Self.declaredTools],
                      [Self.declaredTools, #""tool_choice":"required""#],
                      [Self.declaredTools, #""tool_choice":{"type":"function","function":{"name":"lookup"}}"#],
                      [#""response_format":{"type":"json_object"}"#]] {
            let request = try ChatRequestParser.parse(Self.body(parts))
            let plan = ServerGenerationPlan(request: request, markers: Self.markers)
            #expect(plan.isConstrained)
            #expect(plan.allowsSpeculativeDecoding, "\(parts)")
        }
    }

    @Test("GEN-14: an unconstrained plan keeps it too")
    func GEN_14_an_unconstrained_plan_keeps_the_speculative_path() throws {
        for parts in [[String](),
                      [Self.declaredTools, #""tool_choice":"none""#],
                      [#""response_format":{"type":"text"}"#]] {
            let request = try ChatRequestParser.parse(Self.body(parts))
            let plan = ServerGenerationPlan(request: request, markers: Self.markers)
            #expect(!plan.isConstrained)
            #expect(plan.allowsSpeculativeDecoding, "\(parts)")
        }
    }

    // MARK: - GEN-7: real logits

    @Test("GEN-7: only a constrained plan needs the logits head")
    func GEN_7_only_a_constrained_plan_needs_the_logits_head() throws {
        #expect(try Self.plan(Self.declaredTools).requiresLogitsHead)
        #expect(try !Self.plan().requiresLogitsHead)
    }

    // MARK: - GEN-2 / DEV-16: what was given up reaches the log

    /// `allOf` is not a shape the template can render, so the declaration side
    /// records the drop (`GemmaToolSchemaResult.simplifications`).
    @Test("GEN-2: tool-declaration simplifications reach the plan's approximations")
    func GEN_2_tool_declaration_simplifications_reach_the_plan() throws {
        let tools = #""tools":[{"type":"function","function":{"name":"lookup","#
            + #""description":"","parameters":{"type":"object","properties":{"#
            + #""q":{"allOf":[{"type":"string"}]}}}}}]"#
        let plan = try Self.plan(tools)
        #expect(plan.approximations.contains { $0.contains("unrepresentable-all-of") })
        let field = try #require(ServerApproximationLog.field(plan.approximations))
        #expect(field.contains("unrepresentable-all-of"))
    }

    /// A `pattern` the converter cannot express is approximated by the
    /// grammar stage, not refused (GEN-2). The reference implementation throws
    /// on this input (DEV-16).
    @Test("GEN-2: grammar-stage approximations reach the plan's approximations")
    func GEN_2_grammar_stage_approximations_reach_the_plan() throws {
        let plan = try Self.plan(
            #""response_format":{"type":"json_schema","json_schema":{"name":"r","#
            + #""schema":{"type":"object","properties":{"id":{"type":"string","#
            + #""pattern":"a+"}},"required":["id"]}}}"#)
        #expect(plan.grammar != nil, "GEN-2: an unanchored pattern is approximated, not refused")
        #expect(plan.approximations.contains { $0.hasPrefix("grammar/") })
        #expect(plan.approximations.contains { $0.contains("unanchored-pattern") })
    }

    @Test("GEN-2: a plan that gave nothing up logs nothing")
    func GEN_2_a_plan_that_gave_nothing_up_logs_nothing() throws {
        let plan = try Self.plan(Self.declaredTools)
        #expect(plan.approximations.isEmpty)
        #expect(ServerApproximationLog.field(plan.approximations) == nil)
    }

    /// The log line carries token counts and timings, never prompt text
    /// (docs/OPENAI_SERVER.md). The approximations come from `tools` and
    /// `response_format`, so the field has to survive newlines and stay bounded.
    @Test("GEN-2: the approximation field is one bounded line")
    func GEN_2_the_approximation_field_is_one_bounded_line() {
        // A non-empty list always has a field; `??` keeps the expectations
        // below readable rather than asserting that again.
        let value = ServerApproximationLog.field([
            "unknown-type: #/properties/x\n(money)",
            String(repeating: "y", count: 4_000),
        ]) ?? ""
        #expect(!value.contains("\n"))
        #expect(!value.contains("\r"))
        #expect(value.count <= 512)
        #expect(value.contains("unknown-type"))
    }

    @Test("GEN-2: the plan names the request shape without any prompt text")
    func GEN_2_the_plan_names_the_request_shape() throws {
        let plan = try Self.plan(Self.declaredTools, #""tool_choice":"required""#)
        #expect(plan.shape.contains("tool_choice=required"))
        #expect(plan.shape.contains("response_format=text"))
        #expect(plan.shape.contains("tools=2"))
        #expect(!plan.shape.contains("hi"))
    }

    // MARK: - GEN-6: the thought block suppresses the lazy grammar

    private static let channelStart: Int32 = 100
    private static let channelEnd: Int32 = 101

    private static func suppression() -> ServerThoughtSuppression {
        ServerThoughtSuppression(channelStartID: channelStart, channelEndID: channelEnd)
    }

    @Test("GEN-6: an open thought channel suppresses and its close releases")
    func GEN_6_the_thought_block_suppresses_and_its_close_releases() {
        let watch = Self.suppression()
        #expect(!watch.isSuppressed, "generation starts outside the thought block")
        #expect(watch.observe(tokenID: Self.channelStart, events: []))
        #expect(watch.observe(tokenID: 42, events: [.reasoning("weighing it up")]))
        #expect(!watch.observe(tokenID: Self.channelEnd, events: []))
    }

    /// The close is the case an event alone cannot express: the detokenizer
    /// flushes the thought text held back *before* `<channel|>`, so the token
    /// that leaves the block carries a `.reasoning` event with it. The template
    /// writes `<|tool_call>` immediately after that close, so reading the event
    /// instead of the token would keep the grammar asleep through the very call
    /// it exists to constrain.
    @Test("GEN-6: the channel close releases even when it flushes thought text")
    func GEN_6_a_channel_close_releases_even_when_it_flushes_thought_text() {
        let watch = Self.suppression()
        watch.observe(tokenID: Self.channelStart, events: [])
        watch.observe(tokenID: 42, events: [.reasoning("weighing it up")])
        #expect(!watch.observe(tokenID: Self.channelEnd,
                               events: [.reasoning(" one last thought")]))
        #expect(!watch.isSuppressed)
    }

    @Test("GEN-6: visible text never suppresses")
    func GEN_6_visible_text_never_suppresses() {
        let watch = Self.suppression()
        #expect(!watch.observe(tokenID: 42, events: [.content("Kyoto")]))
        #expect(!watch.observe(tokenID: 43, events: []))
        // A visible channel is released again by the first text the decoder
        // routes to content.
        watch.observe(tokenID: Self.channelStart, events: [])
        #expect(!watch.observe(tokenID: 44, events: [.content("the answer")]))
    }
}
