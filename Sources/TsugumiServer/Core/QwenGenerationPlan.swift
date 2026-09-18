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
    /// grammar (GEN-4's `none` forbids a token instead).
    let grammar: String?
    /// GEN-4's `none`: token ids that may never be drawn (the tool-call start
    /// token). Empty whenever `grammar` is set.
    let forbiddenTokenIDs: [Int32]
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

    var isConstrained: Bool { grammar != nil || !forbiddenTokenIDs.isEmpty }

    /// GEN-4's runtime half. The grammar is built by the session, which owns
    /// the vocabulary and turns a parse failure into a 500; this one has
    /// nothing to fail.
    func forbiddenTokensConstraint() -> ForbiddenTokensConstraint? {
        forbiddenTokenIDs.isEmpty ? nil : ForbiddenTokensConstraint(forbiddenTokenIDs: Set(forbiddenTokenIDs))
    }

    /// The official recommended sampler of a checkpoint, and the only one the
    /// server runs for it.
    struct OfficialSampler: Equatable, Sendable {
        let temperature: Float
        let topP: Float
        let topK: Int

        /// `Ornith-1.5-35B-A3B` (`docs/qwen35moe/42-SAMPLING.md` §0 S1).
        static let ornith = OfficialSampler(temperature: 0.6, topP: 0.95, topK: 20)
        /// Qwen3.8-Flash-Next's non-thinking settings, which `Qwen38Sampler`
        /// runs (its presence penalty 1.5 has no field here and is not named).
        static let qwen38 = OfficialSampler(temperature: 0.7, topP: 0.8, topK: 20)
    }

    static let officialTemperature: Float = OfficialSampler.ornith.temperature
    static let officialTopP: Float = OfficialSampler.ornith.topP
    static let officialTopK = OfficialSampler.ornith.topK

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
        _ requested: GenerationConfig,
        official: OfficialSampler = .ornith
    ) -> (config: GenerationConfig, approximations: [String]) {
        let officialTemperature = official.temperature
        let officialTopK = official.topK
        let officialTopP = official.topP
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

    init(request: ValidatedChatRequest, markers: QwenToolCallMarkers,
         official: OfficialSampler = .ornith) {
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
        if let grammar = request.grammar {
            precondition(request.tools.isEmpty, "an in-process grammar goes with no tools")
            self.grammar = grammar
            self.forbiddenTokenIDs = []
            self.isLazy = false
            self.trigger = nil
        } else {
            self.grammar = constraint?.grammar?.grammar
            self.forbiddenTokenIDs = constraint?.forbiddenTokenIDs ?? []
            self.isLazy = constraint?.grammar?.isLazy ?? false
            self.trigger = constraint?.grammar?.trigger
        }
        let sampling = Self.officialSampling(request.generationConfig, official: official)
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
