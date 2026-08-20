import Foundation
import TurboFieldfare

/// REQ-tool-choice's four shapes (GEN-4).
///
/// All four are the grammar's business: `none` emits no grammar, `auto` a lazy
/// one, and `required` / `function` constrain from the first token. The named
/// shape pins the function name itself (DEV-17).
public enum ChatToolChoice: Equatable, Sendable {
    case auto
    case none
    case required
    case function(name: String)
}

/// REQ-response-format's three types, already unwrapped (GEN-3).
///
/// The schema is carried as it was sent, from the place each type keeps it:
/// `json_schema` from `response_format.json_schema.schema`, `json_object` from
/// `response_format.schema` (`server-common.cpp:1150`). A `nil` schema is not
/// an error and not "unconstrained" — it is "no schema was sent", and what
/// that constrains to is DEV-18, decided by the grammar stage.
public enum ChatResponseFormat: Equatable, Sendable {
    case text
    case jsonObject(schema: JSONValue?)
    case jsonSchema(schema: JSONValue?)

    /// Whether this format constrains generation at all. `text` does not, so
    /// it never collides with a forced tool call (GEN-12).
    public var isConstraining: Bool {
        if case .text = self { return false }
        return true
    }
}

/// REQ-reasoning-format / RSN-3.
public enum ReasoningFormat: String, Equatable, Sendable {
    /// Split the thought channel out into `reasoning_content`.
    case auto
    /// Leave it in the answer as raw text.
    case none
}

/// The process-level defaults a request falls back to when it says nothing.
public struct ChatRequestDefaults: Equatable, Sendable {
    public var thinking: ServerThinkingPolicy

    public init(thinking: ServerThinkingPolicy = .off) {
        self.thinking = thinking
    }
}

/// Turns a request body into a `ValidatedChatRequest` through the SPEC §4
/// table (`ChatRequestSchema`).
///
/// Replaces `OpenAIRequestValidator`, whose per-field `guard`s decided the
/// acceptance rules on their own. Everything this type refuses is a row in the
/// table or a line of SPEC §5/§6; there is nowhere else for a rule to hide.
public enum ChatRequestParser {
    public static func parse(
        _ body: Data,
        imagePolicy: ServerImagePolicy = .default,
        defaults: ChatRequestDefaults = ChatRequestDefaults()
    ) throws -> ValidatedChatRequest {
        let json: JSONValue
        do {
            json = try JSONDecoder().decode(JSONValue.self, from: body)
        } catch {
            // ERR-3: the body really was unparseable here, which is the only
            // place that sentence is true.
            throw ServerRequestError.invalid(
                message: "the request body is not valid JSON",
                param: nil, code: "invalid_json")
        }
        return try parse(json, imagePolicy: imagePolicy, defaults: defaults)
    }

    public static func parse(
        _ body: JSONValue,
        imagePolicy: ServerImagePolicy = .default,
        defaults: ChatRequestDefaults = ChatRequestDefaults()
    ) throws -> ValidatedChatRequest {
        let request = try ChatRequestSchema.normalize(body)

        let toolChoice = Self.toolChoice(request["tool_choice"])
        let responseFormat = Self.responseFormat(request["response_format"])
        let messages = try decode([OpenAIChatMessage].self,
                                  from: request["messages"] ?? .array([]),
                                  param: "messages")
        let validated = try ChatMessageValidator.validateMessages(messages,
                                                                  imagePolicy: imagePolicy)
        // `tool_choice: none` means the model may not call a tool, which the
        // template expresses by not being told the tools exist.
        let declaredTools = toolChoice == .none
            ? []
            : try decode([OpenAITool].self,
                         from: request["tools"] ?? .array([]),
                         param: "tools")
        let tools = try declaredTools.map(ChatMessageValidator.validateTool)
        try Self.checkConstraintsAreSatisfiable(toolChoice: toolChoice,
                                                responseFormat: responseFormat,
                                                tools: tools)

        let maximumCompletionTokens = request.int("max_tokens") ?? -1
        let reasoningEffort = request.string("reasoning_effort")
        let kwargs = request.object("chat_template_kwargs") ?? [:]
        let reasoningBudgetTokens = request.int("reasoning_budget_tokens") ?? -1
        return ValidatedChatRequest(
            messages: validated.messages,
            tools: tools,
            stream: request.bool("stream") ?? false,
            includeUsage: request.object("stream_options")?["include_usage"] == .bool(true),
            generationConfig: try generationConfig(request),
            maximumCompletionTokens: maximumCompletionTokens,
            vision: validated.vision,
            enableThinking: try enableThinking(kwargs: kwargs,
                                               effort: reasoningEffort,
                                               budget: reasoningBudgetTokens,
                                               defaults: defaults),
            model: request.string("model") ?? "",
            toolChoice: toolChoice,
            responseFormat: responseFormat,
            parallelToolCalls: request.bool("parallel_tool_calls") ?? true,
            cachePrompt: request.bool("cache_prompt") ?? true,
            reasoningEffort: reasoningEffort,
            chatTemplateKwargs: kwargs,
            reasoningBudgetTokens: reasoningBudgetTokens,
            reasoningFormat: ReasoningFormat(
                rawValue: request.string("reasoning_format") ?? "auto") ?? .auto,
            timingsPerToken: request.bool("timings_per_token") ?? false)
    }

    /// REQ-tool-choice → GEN-4's four shapes. The table has already refused
    /// anything that is not one of them, so every branch here is reachable and
    /// the fallback is the default.
    private static func toolChoice(_ value: JSONValue?) -> ChatToolChoice {
        switch value {
        case .string("none"):
            return .none
        case .string("required"):
            return .required
        case .object(let choice):
            guard case .object(let function)? = choice["function"],
                  case .string(let name)? = function["name"] else { return .auto }
            // DEV-17: OpenAI's meaning — call this one function. The reference
            // implementation drops the object shape on the floor and behaves
            // as `auto`; that is its defect, not the norm.
            return .function(name: name)
        default:
            return .auto
        }
    }

    /// REQ-response-format → GEN-3's three types, each with the schema from
    /// the place that type keeps it (`server-common.cpp:1146-1159`).
    ///
    /// Nothing about the schema's *content* is checked: GEN-2 forbids a 400
    /// for schema content, and what a missing schema constrains to is the
    /// grammar stage's decision (DEV-18), not this layer's.
    private static func responseFormat(_ value: JSONValue?) -> ChatResponseFormat {
        guard case .object(let format)? = value else { return .text }
        switch format["type"] {
        case .string("json_object"):
            return .jsonObject(schema: sent(format["schema"]))
        case .string("json_schema"):
            guard case .object(let wrapper)? = format["json_schema"] else {
                return .jsonSchema(schema: nil)
            }
            return .jsonSchema(schema: sent(wrapper["schema"]))
        default:
            return .text
        }
    }

    /// R2 again, one level down: a `null` schema is a schema that was not sent.
    private static func sent(_ value: JSONValue?) -> JSONValue? {
        value == .null ? nil : value
    }

    /// The two contract parameters read together (GEN-4, GEN-12).
    ///
    /// Both refusals here exist because the alternative is worse: a grammar
    /// that cannot be written would otherwise be written as *no* grammar, and
    /// the request would get a free-form completion with a 200 — exactly the
    /// answer-in-the-wrong-shape R4 forbids. They cannot live in the §4 table
    /// because each one needs a second field beside `tool_choice`.
    private static func checkConstraintsAreSatisfiable(
        toolChoice: ChatToolChoice,
        responseFormat: ChatResponseFormat,
        tools: [GFTokenizer.FunctionDefinition]
    ) throws {
        // GEN-12: "always call a tool" and "answer in this JSON shape" are two
        // constraints on the same tokens, and honoring either one silently
        // drops the other.
        if responseFormat.isConstraining {
            switch toolChoice {
            case .required, .function:
                throw ServerRequestError.invalid(
                    message: "tool_choice forces a tool call and response_format forces "
                        + "a JSON answer; the two cannot both be honored",
                    param: "response_format",
                    code: "response_format_conflicts_with_tool_choice")
            case .auto, .none:
                break
            }
        }
        // GEN-4: the grammar has to have an alternative to spell.
        switch toolChoice {
        case .required where tools.isEmpty:
            throw ServerRequestError.invalid(
                message: "tool_choice \"required\" needs at least one tool in \"tools\"",
                param: "tool_choice", code: "invalid_tool_choice")
        case .function(let name) where !tools.contains(where: { $0.name == name }):
            throw ServerRequestError.invalid(
                message: "tool_choice names \(name), which is not declared in \"tools\"",
                param: "tool_choice", code: "invalid_tool_choice")
        default:
            break
        }
    }

    /// The clamped table values as the sampler takes them.
    ///
    /// Two of the mappings are approximations the engine forces, both
    /// registered in SPEC §12 (DEV-10): this sampler cannot run nucleus
    /// sampling over the full vocabulary, so a `top_p` below 1 borrows the
    /// widest top-k it has, and a `top_p` of 0 — an empty nucleus — is served
    /// as the greedy draw it describes. Neither is a refusal: R3 says a tuning
    /// parameter is clamped into something this machine can run.
    private static func generationConfig(
        _ request: NormalizedChatRequest
    ) throws -> GenerationConfig {
        let temperature = Float(request.double("temperature") ?? 1.0)
        let topP = request.double("top_p") ?? 1.0
        var topK = request.int("top_k") ?? 0
        var nucleus: Float? = topP < 1 ? Float(topP) : nil
        if topP <= 0 {
            nucleus = nil
            topK = 1
        } else if nucleus != nil, topK == 0 {
            topK = ChatRequestSchema.topKCeiling
        }
        let seed = request.int64("seed") ?? -1
        return GenerationConfig(
            // The real ceiling is the context left after the prompt, which
            // only the session knows; it recomputes this before generating.
            maxNewTokens: max(request.int("max_tokens") ?? -1, 1),
            temperature: temperature,
            topK: topK == 0 ? nil : topK,
            topP: nucleus,
            repetitionPenalty: Float(request.double("repeat_penalty") ?? 1.0),
            seed: seed == -1 ? nil : UInt64(bitPattern: seed),
            stopStrings: stopStrings(request["stop"]))
    }

    private static func stopStrings(_ value: JSONValue?) -> [String] {
        switch value {
        case .string(let one): [one]
        case .array(let many): many.compactMap { if case .string(let s) = $0 { s } else { nil } }
        default: []
        }
    }

    /// RSN-2. Both spellings are read, and the reference implementation's
    /// order of resolution is kept: `chat_template_kwargs.enable_thinking`
    /// says what the template does, and `reasoning_effort: "none"` overrides it
    /// afterwards (`server-common.cpp:1278-1304`). A budget of zero says the
    /// same thing a third way (RSN-1).
    private static func enableThinking(kwargs: [String: JSONValue],
                                       effort: String?,
                                       budget: Int,
                                       defaults: ChatRequestDefaults) throws -> Bool {
        var enabled = defaults.thinking.isEnabled
        if let requested = kwargs["enable_thinking"], requested != .null {
            guard case .bool(let value) = requested else {
                throw ServerRequestError.invalid(
                    message: "chat_template_kwargs.enable_thinking must be a boolean",
                    param: "chat_template_kwargs", code: "invalid_type")
            }
            enabled = value
        } else if let effort, !effort.isEmpty, effort != "none" {
            // This template has one thought channel and no budget, so a level
            // has nothing to select but the channel itself. Which levels exist
            // is not this server's business to enumerate (REQ-reasoning-effort).
            enabled = true
        }
        if effort == "none" { enabled = false }
        if budget == 0 { enabled = false }
        return enabled
    }

    private static func decode<T: Decodable>(_ type: T.Type,
                                             from value: JSONValue,
                                             param: String) throws -> T {
        do {
            return try JSONDecoder().decode(type, from: try JSONEncoder().encode(value))
        } catch let error as DecodingError {
            // ERR-3 again: name the field inside `messages` or `tools` that
            // did not fit, rather than calling the whole body malformed.
            throw ServerRequestError.invalid(
                message: "\(param) is not shaped as expected: \(Self.describe(error))",
                param: param, code: "invalid_type")
        } catch {
            throw ServerRequestError.invalid(
                message: "\(param) could not be read",
                param: param, code: "invalid_type")
        }
    }

    private static func describe(_ error: DecodingError) -> String {
        func path(_ context: DecodingError.Context) -> String {
            let steps = context.codingPath.map { key in
                key.intValue.map { "[\($0)]" } ?? ".\(key.stringValue)"
            }.joined()
            return steps.isEmpty ? "the value" : String(steps.dropFirst(steps.first == "." ? 1 : 0))
        }
        switch error {
        case .keyNotFound(let key, let context):
            return "\(path(context)) is missing \"\(key.stringValue)\""
        case .typeMismatch(_, let context), .valueNotFound(_, let context):
            return "\(path(context)) has the wrong type"
        case .dataCorrupted(let context):
            return context.debugDescription
        @unknown default:
            return "the value could not be read"
        }
    }
}
