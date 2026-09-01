import Foundation
import Tsugumi

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
    /// GBNF could not constrain, and `sampling/` is what the request asked for
    /// and the official settings overrode (S1).
    let approximations: [String]
    /// The request shape an error message may name, in the vocabulary of the
    /// request rather than of the conversation. **Never prompt text.**
    let shape: String
    /// The sampler the run uses — always the official three (S1), whatever the
    /// request asked for. See `officialSampling`.
    let sampling: GenerationConfig

    var isConstrained: Bool { grammar != nil }

    /// The official recommended sampler for `Ornith-1.5-35B-A3B`, and the only
    /// one this server runs (`docs/qwen35moe/42-SAMPLING.md` §0 S1).
    static let officialTemperature: Float = 0.6
    static let officialTopP: Float = 0.95
    static let officialTopK = 20

    /// The sampler the run will actually use, and the list of what it
    /// overrode.
    ///
    /// **The values are not negotiable and the override is not silent.** S1
    /// says only the official settings may be used; the decision on a request
    /// that asks for something else is to *override and record*, not to refuse
    /// (2026-08-22). So `temperature`, `top_p` and `top_k` are set to the
    /// official three whatever arrived, and the completion line names every
    /// field whose requested value differed.
    ///
    /// This replaces the `greedy-only: … ignored` note, which said the request
    /// was accepted and thrown away — the state
    /// `docs/qwen35moe/42-SAMPLING.md` §1 exists to end. Two things are still
    /// dropped rather than honoured, and both are named: `repeat_penalty`,
    /// which is not part of the official recommendation and which this
    /// family's sampler does not implement, and `seed`, which the server does
    /// not take from a request.
    static func officialSampling(
        _ requested: GenerationConfig
    ) -> (config: GenerationConfig, approximations: [String]) {
        var config = requested
        var overridden: [String] = []
        if requested.temperature != officialTemperature {
            overridden.append("temperature=\(requested.temperature)→\(officialTemperature)")
        }
        if requested.topK != officialTopK {
            let asked = requested.topK.map { "\($0)" } ?? "none"
            overridden.append("top_k=\(asked)→\(officialTopK)")
        }
        if requested.topP != officialTopP {
            let asked = requested.topP.map { "\($0)" } ?? "none"
            overridden.append("top_p=\(asked)→\(officialTopP)")
        }
        if requested.repetitionPenalty != 1 {
            overridden.append("repeat_penalty=\(requested.repetitionPenalty)→1")
        }
        config.temperature = officialTemperature
        config.topK = officialTopK
        config.topP = officialTopP
        config.repetitionPenalty = 1
        config.seed = nil
        guard !overridden.isEmpty else { return (config, []) }
        return (config, ["official-override: " + overridden.joined(separator: " ")])
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
        let sampling = Self.officialSampling(request.generationConfig)
        self.sampling = sampling.config
        self.approximations =
            request.toolSchemaSimplifications.map { "tools/" + $0 }
            + (constraint?.approximations ?? []).map { "grammar/" + $0 }
            + sampling.approximations.map { "sampling/" + $0 }
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
