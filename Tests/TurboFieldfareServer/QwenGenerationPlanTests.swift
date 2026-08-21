import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// C0 (CONFORMANCE §1) for SPEC §6 on the Ornith (Qwen 3.5-MoE) family: the
/// decision `QwenServerSession` makes before it touches the model.
///
/// `ServerGenerationPlanTests`' sibling, and the same shape: the requests are
/// built by the **real** parser, so GEN-12's refusal is observable from here
/// (the collision never reaches a plan), and the markers are injected so no
/// weights, no Metal and no tokenizer are involved.
///
/// What is new against the Gemma suite is the third tag. This family draws with
/// a fused head that writes no logits, so the sampler a client asked for is
/// accepted and ignored (R3) — and *named*, which is what the `sampling/` line
/// is for (`docs/qwen35moe/26-PHASE8-SERVER.md` §4).
@Suite("C0 Qwen generation plan")
struct QwenGenerationPlanTests {
    // MARK: - Fixtures

    /// The real ids, as `QwenChatGrammarBuilderTests` uses them.
    private static let markers = QwenToolCallMarkers(
        toolCallStart: "<tool_call>",
        toolCallEnd: "</tool_call>",
        toolCallStartTokenID: 248_058,
        toolCallEndTokenID: 248_059,
        thinkEndTokenID: 248_069)

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

    private static func plan(_ parts: String...) throws -> QwenGenerationPlan {
        let request = try ChatRequestParser.parse(body(parts))
        return QwenGenerationPlan(request: request, markers: markers)
    }

    // MARK: - GEN-4 / GEN-5: what a tool_choice plans

    @Test("GEN-4: none plans no constraint at all")
    func GEN_4_tool_choice_none_plans_no_constraint() throws {
        let plan = try Self.plan(Self.declaredTools, #""tool_choice":"none""#)
        #expect(plan.grammar == nil)
        #expect(!plan.isConstrained)
        #expect(plan.trigger == nil)
    }

    @Test("GEN-5: auto plans a lazy grammar triggered by <tool_call>")
    func GEN_5_tool_choice_auto_plans_a_lazy_grammar() throws {
        let plan = try Self.plan(Self.declaredTools)
        let grammar = try #require(plan.grammar)
        // The name lives inside the XML the checkpoint's own template writes,
        // not behind a `call:` marker — that is the whole difference from the
        // Gemma plan, and the reason this is a sibling type.
        #expect(grammar.contains("<function=lookup>"))
        #expect(plan.isLazy)
        #expect(plan.trigger == ChatGrammarTrigger(tokenID: 248_058, text: "<tool_call>"))
    }

    @Test("GEN-4: required plans an eager grammar with no trigger")
    func GEN_4_tool_choice_required_plans_an_eager_grammar() throws {
        let plan = try Self.plan(Self.declaredTools, #""tool_choice":"required""#)
        #expect(plan.isConstrained)
        #expect(!plan.isLazy)
        #expect(plan.trigger == nil)
    }

    @Test("DEV-17: a named choice pins that one function")
    func DEV_17_named_choice_pins_one_function() throws {
        let plan = try Self.plan(
            Self.declaredTools,
            #""tool_choice":{"type":"function","function":{"name":"clock"}}"#)
        let grammar = try #require(plan.grammar)
        #expect(grammar.contains("<function=clock>"))
        #expect(!grammar.contains("<function=lookup>"))
    }

    @Test("no tools is no grammar, whatever the choice says")
    func no_tools_is_no_grammar() throws {
        #expect(try Self.plan().grammar == nil)
        #expect(try Self.plan(#""tool_choice":"auto""#).grammar == nil)
    }

    // MARK: - Sampling (SPEC §12 R3, this family's own line)

    @Test("the default request is served greedily, and says so")
    func default_request_records_the_greedy_approximation() throws {
        // REQ-temp: a client that never mentioned `temperature` still asks for
        // 1.0, so the note fires on the everyday request. That is the point —
        // the alternative is a 400 on every default client.
        let plan = try Self.plan()
        #expect(plan.approximations == ["sampling/greedy-only: temperature=1.0 ignored"])
    }

    @Test("temperature 0 with no penalty is greedy, so nothing is approximated")
    func pure_greedy_request_records_nothing() throws {
        let plan = try Self.plan(#""temperature":0"#)
        #expect(plan.approximations.isEmpty)
    }

    @Test("every sampler the client named is in the one note")
    func named_samplers_are_all_reported() throws {
        let plan = try Self.plan(#""temperature":0.7"#, #""top_p":0.9"#,
                                 #""repeat_penalty":1.1"#)
        let note = try #require(plan.approximations.first)
        #expect(plan.approximations.count == 1)
        #expect(note.hasPrefix("sampling/greedy-only:"))
        #expect(note.contains("temperature=0.7"))
        #expect(note.contains("top_p=0.9"))
        #expect(note.contains("repeat_penalty=1.1"))
    }

    @Test("a greedy request that declares tools still carries only the tool tags")
    func greedy_tool_request_has_no_sampling_note() throws {
        let plan = try Self.plan(Self.declaredTools, #""temperature":0"#)
        #expect(!plan.approximations.contains { $0.hasPrefix("sampling/") })
    }

    // MARK: - GEN-3 / GEN-12

    @Test("GEN-3: a response format constrains generation and is not lazy")
    func GEN_3_response_format_constrains_eagerly() throws {
        let plan = try Self.plan(
            #""response_format":{"type":"json_schema","json_schema":{"schema":"#
            + #"{"type":"object","properties":{"a":{"type":"integer"}}}}}"#)
        #expect(plan.isConstrained)
        #expect(!plan.isLazy)
        #expect(plan.trigger == nil)
    }

    @Test("GEN-12: the collision is a 400 and never reaches a plan")
    func GEN_12_collision_is_refused_by_the_parser() throws {
        #expect(throws: ServerRequestError.self) {
            _ = try Self.plan(Self.declaredTools,
                              #""tool_choice":"required""#,
                              #""response_format":{"type":"json_object"}"#)
        }
    }

    // MARK: - The shape an error may name

    @Test("the shape names the request and never the conversation")
    func shape_names_the_request_only() throws {
        let plan = try Self.plan(Self.declaredTools, #""tool_choice":"required""#)
        #expect(plan.shape == "tool_choice=required response_format=text "
                + "tools=2 parallel_tool_calls=true")
        #expect(!plan.shape.contains("hi"))
    }
}
