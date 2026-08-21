import Foundation
import TurboFieldfare

/// SPEC §6 for the Ornith (Qwen 3.5-MoE) family: one request's *decision* about
/// how generation is constrained and what it had to give up, with none of the
/// inference loop in it.
///
/// `ServerGenerationPlan`'s sibling, not a branch of it, for the reason
/// `QwenChatGrammarBuilder` is `ChatGrammarBuilder`'s: the two families spell a
/// call differently, and the Gemma type's markers (`ChatGrammarMarkers`) are
/// read off a `GFTokenizer` that this family never loads. What the two share is
/// the shape of the answer — a grammar, whether it is lazy, what triggers it,
/// and the list of approximations the server logs — so a route reads either the
/// same way.
///
/// It exists apart from `QwenServerSession` for the same reason as the Gemma
/// one: a session cannot be built without weights, and everything here is a
/// pure function of the validated request and the checkpoint's marker ids.
struct QwenGenerationPlan: Equatable, Sendable {
    /// GBNF text rooted at `root`, or `nil` when this request asks for no
    /// constraint at all.
    let grammar: String?
    /// GEN-5: `true` means the grammar is not applied until `trigger` fires.
    let isLazy: Bool
    /// GEN-5. Non-nil exactly when `isLazy`.
    let trigger: ChatGrammarTrigger?
    /// GEN-2 / DEV-16, and this family's own §12 entries. Three tags appear,
    /// in this order, because they degrade independently: `tools/` is what the
    /// declaration lost on the way into the prompt, `grammar/` is what the
    /// GBNF could not constrain, and `sampling/` is what this family cannot do
    /// at all yet.
    let approximations: [String]
    /// The request shape an error message may name, in the vocabulary of the
    /// request rather than of the conversation. **Never prompt text.**
    let shape: String

    var isConstrained: Bool { grammar != nil }

    /// SPEC §12 DEV-5's rule, applied to a whole sampler rather than to one
    /// field: this family's head writes no logits, so there is nothing to
    /// shape a distribution out of and every draw is the argmax
    /// (`docs/qwen35moe/19-LM-HEAD-INT8.md`). The request is **accepted and
    /// the sampler ignored** (R3), which is what the reference does with every
    /// sampler it has not implemented — refusing instead would fail the
    /// default request, whose `temperature` is 1.0 because the client never
    /// mentioned it (REQ-temp).
    ///
    /// Named here rather than swallowed: a client that asked for `temperature:
    /// 0.7` and got greedy text has a right to see that in the log.
    static func samplingApproximations(_ config: GenerationConfig) -> [String] {
        guard !config.isPureGreedy else { return [] }
        var asked: [String] = []
        if config.temperature != 0 { asked.append("temperature=\(config.temperature)") }
        if let topK = config.topK { asked.append("top_k=\(topK)") }
        if let topP = config.topP { asked.append("top_p=\(topP)") }
        if config.repetitionPenalty != 1 {
            asked.append("repeat_penalty=\(config.repetitionPenalty)")
        }
        return ["greedy-only: " + asked.joined(separator: " ") + " ignored"]
    }

    init(request: ValidatedChatRequest, markers: QwenToolCallMarkers) {
        // GEN-12 is settled before this point, by the same `ChatRequestParser`
        // check the Gemma path relies on: a constraining `response_format`
        // beside a `required` or named `tool_choice` is a 400 and never
        // reaches a plan.
        let constraint = QwenChatGrammarBuilder.constraint(
            tools: request.tools,
            toolChoice: request.toolChoice,
            parallelToolCalls: request.parallelToolCalls,
            responseFormat: Self.responseFormat(request.responseFormat),
            markers: markers)
        self.grammar = constraint?.grammar
        self.isLazy = constraint?.isLazy ?? false
        self.trigger = constraint?.trigger
        self.approximations =
            request.toolSchemaSimplifications.map { "tools/" + $0 }
            + (constraint?.approximations ?? []).map { "grammar/" + $0 }
            + Self.samplingApproximations(request.generationConfig)
                .map { "sampling/" + $0 }
        self.shape = Self.shape(request)
    }

    private static func responseFormat(
        _ format: ChatResponseFormat
    ) -> ChatGrammarBuilder.ResponseFormat {
        switch format {
        case .text: return .text
        case .jsonObject(let schema): return .jsonObject(schema: schema)
        case .jsonSchema(let schema): return .jsonSchema(schema: schema)
        }
    }

    private static func shape(_ request: ValidatedChatRequest) -> String {
        let choice: String
        switch request.toolChoice {
        case .auto: choice = "auto"
        case .none: choice = "none"
        case .required: choice = "required"
        case .function(let name): choice = "function:\(name)"
        }
        let format: String
        switch request.responseFormat {
        case .text: format = "text"
        case .jsonObject: format = "json_object"
        case .jsonSchema: format = "json_schema"
        }
        return "tool_choice=\(choice) response_format=\(format) "
            + "tools=\(request.tools.count) "
            + "parallel_tool_calls=\(request.parallelToolCalls)"
    }
}
