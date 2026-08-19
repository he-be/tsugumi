import Foundation
import TurboFieldfare

public struct OpenAIErrorEnvelope: Codable, Equatable, Sendable {
    public struct Detail: Codable, Equatable, Sendable {
        public let message: String
        public let type: String
        public let param: String?
        public let code: String
    }

    public let error: Detail

    public init(message: String, param: String? = nil, code: String,
                type: String = "invalid_request_error") {
        error = Detail(message: message,
                       type: type,
                       param: param,
                       code: code)
    }
}

public struct OpenAIImageURL: Codable, Equatable, Sendable {
    public let url: String
    /// OpenAI's resolution hint. Accepted and ignored: the soft-token count
    /// follows the image's aspect ratio here (PLAN_VISION §2-1), so there is
    /// nothing for `low`/`high` to select. Rejecting it would break clients that
    /// send the field by default.
    public let detail: String?

    public init(url: String, detail: String? = nil) {
        self.url = url
        self.detail = detail
    }
}

public struct OpenAIContentPart: Codable, Equatable, Sendable {
    public let type: String
    public let text: String?
    public let imageURL: OpenAIImageURL?

    enum CodingKeys: String, CodingKey {
        case type, text
        case imageURL = "image_url"
    }

    public init(type: String, text: String? = nil, imageURL: OpenAIImageURL? = nil) {
        self.type = type
        self.text = text
        self.imageURL = imageURL
    }
}

public enum OpenAIMessageContent: Codable, Equatable, Sendable {
    case text(String)
    case parts([OpenAIContentPart])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([OpenAIContentPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text): try container.encode(text)
        case .parts(let parts): try container.encode(parts)
        }
    }

    /// The parts of this body, in order, with each image resolved to bytes.
    ///
    /// `imageIndex` is the running per-request image number so the error
    /// messages can name which image was refused; it is advanced by one per
    /// image part.
    func resolvedParts(policy: ServerImagePolicy,
                       imageIndex: inout Int,
                       images: inout [ServerImageAttachment]) throws -> [GFTokenizer.ContentPart] {
        switch self {
        case .text(let text):
            return [.text(text)]
        case .parts(let parts):
            var resolved: [GFTokenizer.ContentPart] = []
            for part in parts {
                switch part.type {
                case "text":
                    guard let text = part.text else {
                        throw ServerRequestError.invalid(
                            message: "text content part is missing \"text\"",
                            param: "messages",
                            code: "unsupported_content")
                    }
                    resolved.append(.text(text))
                case "image_url":
                    guard let imageURL = part.imageURL else {
                        throw ServerRequestError.invalid(
                            message: "image_url content part is missing \"image_url\"",
                            param: "messages",
                            code: "unsupported_content")
                    }
                    guard images.count < policy.maxImagesPerRequest else {
                        throw ServerRequestError.payloadTooLarge(
                            message: "a request may attach at most "
                                + "\(policy.maxImagesPerRequest) images",
                            param: "messages",
                            code: "too_many_images")
                    }
                    images.append(try ServerImageDecoder.attachment(fromImageURL: imageURL.url,
                                                                    policy: policy,
                                                                    index: imageIndex))
                    imageIndex += 1
                    resolved.append(.image)
                default:
                    throw ServerRequestError.invalid(
                        message: "unsupported content part type \(part.type)",
                        param: "messages",
                        code: "unsupported_content")
                }
            }
            return resolved
        }
    }
}

public struct OpenAIFunctionCall: Codable, Equatable, Sendable {
    public let name: String
    public let arguments: String
}

public struct OpenAIToolCall: Codable, Equatable, Sendable {
    public let id: String
    public let type: String
    public let function: OpenAIFunctionCall
}

public struct OpenAIChatMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: OpenAIMessageContent?
    public let toolCalls: [OpenAIToolCall]?
    public let toolCallID: String?
    public let name: String?

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }
}

public struct OpenAIFunctionDefinition: Codable, Equatable, Sendable {
    public let name: String
    public let description: String?
    public let parameters: JSONValue
}

public struct OpenAITool: Codable, Equatable, Sendable {
    public let type: String
    public let function: OpenAIFunctionDefinition
}

public enum OpenAIStop: Codable, Equatable, Sendable {
    case one(String)
    case many([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let one = try? container.decode(String.self) {
            self = .one(one)
        } else {
            self = .many(try container.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .one(let value): try container.encode(value)
        case .many(let value): try container.encode(value)
        }
    }

    var values: [String] {
        switch self {
        case .one(let value): [value]
        case .many(let value): value
        }
    }
}

public struct OpenAIStreamOptions: Codable, Equatable, Sendable {
    public let includeUsage: Bool?

    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

public struct OpenAIChatRequest: Codable, Equatable, Sendable {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let stream: Bool?
    public let streamOptions: OpenAIStreamOptions?
    public let temperature: Float?
    public let topP: Float?
    public let maxTokens: Int?
    public let maxCompletionTokens: Int?
    public let stop: OpenAIStop?
    public let seed: UInt64?
    public let tools: [OpenAITool]?
    public let toolChoice: JSONValue?
    public let parallelToolCalls: Bool?
    public let topK: Int?
    public let repetitionPenalty: Float?
    public let n: Int?
    public let logprobs: Bool?
    public let presencePenalty: Float?
    public let frequencyPenalty: Float?
    /// OpenAI's reasoning switch. Only its on/off sense is honored: the
    /// template has one thought channel, not a budget, so there is nothing for
    /// a level to select (`OpenAIReasoning`).
    public let reasoningEffort: String?
    /// vLLM's convention, and what pi sends for `thinkingFormat`
    /// `qwen-chat-template`. Only `enable_thinking` is read; other keys (pi's
    /// `preserve_thinking`) are accepted and ignored so a client that sends
    /// its whole kwargs block is not refused for a key this template lacks.
    public let chatTemplateKwargs: JSONValue?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, stop, seed, tools, n, logprobs
        case streamOptions = "stream_options"
        case reasoningEffort = "reasoning_effort"
        case chatTemplateKwargs = "chat_template_kwargs"
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
    }
}

public struct OpenAIUsage: Codable, Equatable, Sendable {
    public struct PromptTokensDetails: Codable, Equatable, Sendable {
        public let cachedTokens: Int

        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }

        public init(cachedTokens: Int) {
            self.cachedTokens = cachedTokens
        }
    }

    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int
    public let promptTokensDetails: PromptTokensDetails

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case promptTokensDetails = "prompt_tokens_details"
    }

    public init(promptTokens: Int,
                completionTokens: Int,
                totalTokens: Int,
                cachedTokens: Int = 0) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.promptTokensDetails = PromptTokensDetails(cachedTokens: cachedTokens)
    }
}

public struct OpenAIModelList: Codable, Equatable, Sendable {
    public struct Model: Codable, Equatable, Sendable {
        public let id: String
        public let object: String
        public let created: Int
        public let ownedBy: String

        enum CodingKeys: String, CodingKey {
            case id, object, created
            case ownedBy = "owned_by"
        }
    }

    public let object: String
    public let data: [Model]
}

public enum ServerRequestError: Error, Equatable, Sendable {
    case invalid(message: String, param: String?, code: String)
    /// Refused for size rather than for shape — answered with 413, not 400.
    case payloadTooLarge(message: String, param: String?, code: String)
    case unknownModel
    case queueFull

    public var envelope: OpenAIErrorEnvelope {
        switch self {
        case .invalid(let message, let param, let code):
            OpenAIErrorEnvelope(message: message, param: param, code: code)
        case .payloadTooLarge(let message, let param, let code):
            OpenAIErrorEnvelope(message: message, param: param, code: code)
        case .unknownModel:
            OpenAIErrorEnvelope(message: "requested model is not available",
                                param: "model", code: "model_not_found")
        case .queueFull:
            OpenAIErrorEnvelope(message: "generation queue is full",
                                code: "queue_full")
        }
    }
}

/// The image side of a validated request: the turns as parts, and the bytes.
///
/// Present only when the request actually carries an image. Its presence is what
/// turns off the prompt cache and switches rendering to the multimodal template,
/// so a text-only request keeps every byte of its old path.
public struct ValidatedVisionRequest: Sendable {
    public let messages: [GFTokenizer.MultimodalMessage]
    public let images: [ServerImageAttachment]

    public init(messages: [GFTokenizer.MultimodalMessage],
                images: [ServerImageAttachment]) {
        self.messages = messages
        self.images = images
    }
}

/// How a request asks for the thought channel.
///
/// Two spellings reach this server and both are honored, because the clients
/// that matter here disagree: pi's `openai-completions` adapter sends
/// `chat_template_kwargs.enable_thinking` (its `qwen-chat-template` thinking
/// format, the vLLM convention), while an OpenAI-shaped client sends
/// `reasoning_effort`. The template has one thought channel and no budget, so
/// an effort level is read only for its on/off sense.
enum OpenAIReasoning {
    /// The efforts that mean "do not reason". Everything else that is a known
    /// OpenAI level means "reason"; an unknown string is refused rather than
    /// guessed at.
    static let offEfforts: Set<String> = ["none", "off"]
    static let onEfforts: Set<String> = ["minimal", "low", "medium", "high", "max"]

    /// The request's answer, or nil when it did not ask either way.
    static func requested(_ request: OpenAIChatRequest) throws -> Bool? {
        let fromKwargs = try enableThinking(in: request.chatTemplateKwargs)
        let fromEffort = try enableThinking(effort: request.reasoningEffort)
        guard let fromKwargs else { return fromEffort }
        guard let fromEffort, fromEffort != fromKwargs else { return fromKwargs }
        throw ServerRequestError.invalid(
            message: "reasoning_effort and chat_template_kwargs.enable_thinking disagree",
            param: "reasoning_effort",
            code: "invalid_value")
    }

    private static func enableThinking(in kwargs: JSONValue?) throws -> Bool? {
        guard let kwargs else { return nil }
        guard let object = kwargs.objectValue else {
            throw ServerRequestError.invalid(
                message: "chat_template_kwargs must be an object",
                param: "chat_template_kwargs",
                code: "invalid_value")
        }
        // Every other key is ignored on purpose: pi sends `preserve_thinking`
        // alongside, and refusing a kwarg this template has no use for would
        // fail a request that is otherwise exactly right.
        guard let value = object["enable_thinking"] else { return nil }
        guard case .bool(let enabled) = value else {
            throw ServerRequestError.invalid(
                message: "chat_template_kwargs.enable_thinking must be a boolean",
                param: "chat_template_kwargs",
                code: "invalid_value")
        }
        return enabled
    }

    private static func enableThinking(effort: String?) throws -> Bool? {
        guard let effort else { return nil }
        let normalized = effort.lowercased()
        if offEfforts.contains(normalized) { return false }
        if onEfforts.contains(normalized) { return true }
        throw ServerRequestError.invalid(
            message: "reasoning_effort must be one of "
                + (offEfforts.union(onEfforts)).sorted().joined(separator: ", "),
            param: "reasoning_effort",
            code: "unsupported_value")
    }
}

public struct ValidatedChatRequest: Sendable {
    /// The text projection of the conversation. When `vision != nil` the image
    /// parts are *not* in here — rendering goes through `vision.messages`
    /// instead, and the prompt cache (which keys on this array) is off.
    public let messages: [GFTokenizer.Message]
    public let tools: [GFTokenizer.FunctionDefinition]
    public let stream: Bool
    public let includeUsage: Bool
    public let generationConfig: GenerationConfig
    public let maximumCompletionTokens: Int
    public let vision: ValidatedVisionRequest?
    /// What this request asked the thought channel to do, after the process
    /// default. Whether it is actually rendered is a second question the
    /// template answers (`ServerModelSession`): a request that declares tools
    /// goes through the tool-calling template, which pins thinking off.
    public let enableThinking: Bool

    public init(messages: [GFTokenizer.Message],
                tools: [GFTokenizer.FunctionDefinition],
                stream: Bool,
                includeUsage: Bool,
                generationConfig: GenerationConfig,
                maximumCompletionTokens: Int,
                vision: ValidatedVisionRequest? = nil,
                enableThinking: Bool = false) {
        self.messages = messages
        self.tools = tools
        self.stream = stream
        self.includeUsage = includeUsage
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.vision = vision
        self.enableThinking = enableThinking
    }
}

private enum OpenAIToolName {
    static let maximumLength = 64

    static func isValid(_ name: String) -> Bool {
        let bytes = name.utf8
        guard !bytes.isEmpty, bytes.count <= maximumLength else { return false }
        return bytes.allSatisfy { byte in
            switch byte {
            case 45, 48...57, 65...90, 95, 97...122:
                true
            default:
                false
            }
        }
    }

    static func validationMessage(for name: String) -> String {
        let prefix = name.prefix(maximumLength + 1)
        let displayed = String(prefix.prefix(maximumLength))
            + (prefix.count > maximumLength ? "..." : "")
        return "tool name \(String(reflecting: displayed)) must contain 1 to 64 ASCII letters, numbers, underscores, or hyphens"
    }
}

public enum OpenAIRequestValidator {
    public static func validate(_ request: OpenAIChatRequest,
                                modelID: String,
                                imagePolicy: ServerImagePolicy = .default,
                                thinkingPolicy: ServerThinkingPolicy = .off) throws -> ValidatedChatRequest {
        guard request.model == modelID else { throw ServerRequestError.unknownModel }
        guard request.n == nil || request.n == 1 else {
            throw invalid("only n=1 is supported", "n", "unsupported_value")
        }
        guard request.logprobs != true else {
            throw invalid("logprobs are not supported", "logprobs", "unsupported_value")
        }
        guard request.presencePenalty == nil || request.presencePenalty == 0 else {
            throw invalid("presence_penalty must be zero", "presence_penalty", "unsupported_value")
        }
        guard request.frequencyPenalty == nil || request.frequencyPenalty == 0 else {
            throw invalid("frequency_penalty must be zero", "frequency_penalty", "unsupported_value")
        }
        guard request.parallelToolCalls != false else {
            throw invalid("parallel_tool_calls=false is not supported",
                          "parallel_tool_calls", "unsupported_value")
        }

        let temperature = request.temperature ?? 1.0
        guard temperature >= 0, temperature <= 2 else {
            throw invalid("temperature must be between 0 and 2",
                          "temperature", "invalid_value")
        }
        let topP = request.topP ?? 0.95
        guard topP > 0, topP <= 1 else {
            throw invalid("top_p must be greater than 0 and at most 1",
                          "top_p", "invalid_value")
        }
        let topK = request.topK ?? 64
        guard (1...256).contains(topK) else {
            throw invalid("top_k must be between 1 and 256", "top_k", "invalid_value")
        }
        let repetitionPenalty = request.repetitionPenalty ?? 1
        guard repetitionPenalty > 0 else {
            throw invalid("repetition_penalty must be positive",
                          "repetition_penalty", "invalid_value")
        }
        let maximum = request.maxCompletionTokens ?? request.maxTokens ?? 4096
        guard maximum > 0 else {
            throw invalid("maximum completion tokens must be positive",
                          request.maxCompletionTokens != nil ? "max_completion_tokens" : "max_tokens",
                          "invalid_value")
        }

        let includeTools: Bool
        switch request.toolChoice {
        case nil, .some(.string("auto")):
            includeTools = true
        case .some(.string("none")):
            includeTools = false
        case .some(.string("required")):
            throw invalid("tool_choice=required is not supported",
                          "tool_choice", "unsupported_value")
        default:
            throw invalid("named tool choices are not supported",
                          "tool_choice", "unsupported_value")
        }

        let tools = try (includeTools ? request.tools ?? [] : []).map(validateTool)
        // Images and tools used to be refused together (PLAN_VISION §0-I-4),
        // on the belief that the tool-calling template could not render an
        // image content part. It can (11-S2 §1), and the marker it writes is
        // the one the vision assembler widens, so the refusal is gone.
        let validated = try validateMessages(request.messages, imagePolicy: imagePolicy)
        let config = GenerationConfig(maxNewTokens: maximum,
                                      temperature: temperature,
                                      topK: topK,
                                      topP: topP,
                                      repetitionPenalty: repetitionPenalty,
                                      seed: request.seed,
                                      stopStrings: request.stop?.values ?? [])
        let enableThinking = try OpenAIReasoning.requested(request) ?? thinkingPolicy.isEnabled
        return ValidatedChatRequest(messages: validated.messages,
                                    tools: tools,
                                    stream: request.stream ?? false,
                                    includeUsage: request.streamOptions?.includeUsage ?? false,
                                    generationConfig: config,
                                    maximumCompletionTokens: maximum,
                                    vision: validated.vision,
                                    enableThinking: enableThinking)
    }

    private static func validateTool(_ tool: OpenAITool) throws -> GFTokenizer.FunctionDefinition {
        guard tool.type == "function" else {
            throw invalid("only function tools are supported", "tools", "unsupported_tool")
        }
        let name = tool.function.name
        guard OpenAIToolName.isValid(name) else {
            throw invalid(OpenAIToolName.validationMessage(for: name),
                          "tools", "invalid_tool_name")
        }
        guard tool.function.parameters.objectValue != nil else {
            throw invalid("tool parameters must be an object schema",
                          "tools", "invalid_tool_schema")
        }
        try validateSchemaKeys(tool.function.parameters)
        let parameters = try GemmaToolSchema.adapted(
            tool.function.parameters, toolName: name)
        guard (try? parameters.jinjaSendableValue()) != nil else {
            throw invalid("tool schema contains a number that cannot be represented exactly",
                          "tools", "invalid_tool_schema")
        }
        return GFTokenizer.FunctionDefinition(name: name,
                                              description: tool.function.description ?? "",
                                              parameters: parameters)
    }

    private static func validateSchemaKeys(_ schema: JSONValue) throws {
        switch schema {
        case .object(let object):
            for (schemaKey, value) in object {
                if schemaKey == "properties" {
                    guard case .object(let definitions) = value else {
                        throw invalid("tool schema properties must be an object",
                                      "tools", "invalid_tool_schema")
                    }
                    for (key, definition) in definitions {
                        guard GemmaToolCallParser.isRepresentableObjectKey(key) else {
                            throw invalid(
                                "tool parameter names may contain only letters, numbers, _, -, ., and $",
                                "tools",
                                "invalid_tool_schema")
                        }
                        try validateSchemaKeys(definition)
                    }
                } else {
                    try validateSchemaKeys(value)
                }
            }
        case .array(let values):
            for value in values {
                try validateSchemaKeys(value)
            }
        default:
            break
        }
    }

    private struct ValidatedMessages {
        let messages: [GFTokenizer.Message]
        let vision: ValidatedVisionRequest?
    }

    private static func validateMessages(
        _ input: [OpenAIChatMessage],
        imagePolicy: ServerImagePolicy
    ) throws -> ValidatedMessages {
        guard !input.isEmpty else {
            throw invalid("messages must not be empty", "messages", "invalid_message")
        }
        var knownCalls: [String: (name: String, resolved: Bool)] = [:]
        var result: [GFTokenizer.Message] = []
        var multimodal: [GFTokenizer.MultimodalMessage] = []
        var images: [ServerImageAttachment] = []
        var imageIndex = 0
        var sawConversationMessage = false
        for message in input {
            guard let role = GFTokenizer.Role(rawValue: message.role) else {
                throw invalid("unsupported message role \(message.role)",
                              "messages", "invalid_message")
            }
            if role == .system || role == .developer {
                guard !sawConversationMessage else {
                    throw invalid("system or developer guidance must precede the conversation",
                                  "messages", "invalid_message")
                }
            } else {
                sawConversationMessage = true
            }
            let parts = try message.content?.resolvedParts(policy: imagePolicy,
                                                           imageIndex: &imageIndex,
                                                           images: &images)
            let turnImages = parts?.reduce(into: 0) { $0 += ($1 == .image ? 1 : 0) } ?? 0
            guard turnImages == 0 || role == .user else {
                throw invalid("images may only appear in user turns",
                              "messages", "unsupported_content")
            }
            multimodal.append(GFTokenizer.MultimodalMessage(role: role, parts: parts ?? []))
            let content = parts.map { parts in
                parts.compactMap { part -> String? in
                    if case .text(let text) = part { return text }
                    return nil
                }.joined()
            }
            let calls: [GFTokenizer.HistoricalToolCall] = try (message.toolCalls ?? []).map { call in
                guard role == .assistant, call.type == "function",
                      !call.id.isEmpty, knownCalls[call.id] == nil else {
                    throw invalid("invalid or duplicate historical tool call",
                                  "messages", "invalid_tool_call")
                }
                guard OpenAIToolName.isValid(call.function.name) else {
                    throw invalid(OpenAIToolName.validationMessage(for: call.function.name),
                                  "messages", "invalid_tool_call")
                }
                let data = Data(call.function.arguments.utf8)
                let arguments = try JSONDecoder().decode(JSONValue.self, from: data)
                guard arguments.objectValue != nil else {
                    throw invalid("historical tool arguments must be a JSON object",
                                  "messages", "invalid_tool_arguments")
                }
                guard (try? arguments.gemmaToolArgumentBody()) != nil,
                      (try? arguments.jinjaSendableValue()) != nil else {
                    throw invalid(
                        "historical tool arguments cannot be represented exactly",
                        "messages",
                        "invalid_tool_arguments")
                }
                knownCalls[call.id] = (call.function.name, false)
                return GFTokenizer.HistoricalToolCall(
                    id: call.id, name: call.function.name, arguments: arguments)
            }
            if role == .tool {
                guard let id = message.toolCallID,
                      let call = knownCalls[id], !call.resolved else {
                    throw invalid("tool result must reference one unresolved call",
                                  "messages", "invalid_tool_result")
                }
                knownCalls[id] = (call.name, true)
                guard content != nil else {
                    throw invalid("tool result content is required",
                                  "messages", "invalid_tool_result")
                }
            } else if content == nil && calls.isEmpty {
                throw invalid("message content is required",
                              "messages", "invalid_message")
            }
            result.append(GFTokenizer.Message(role: role,
                                              content: content,
                                              toolCalls: calls,
                                              toolCallID: message.toolCallID,
                                              name: message.name))
        }
        guard !images.isEmpty else {
            return ValidatedMessages(messages: result, vision: nil)
        }
        return ValidatedMessages(
            messages: result,
            vision: ValidatedVisionRequest(messages: multimodal, images: images))
    }

    private static func invalid(_ message: String,
                                _ param: String?,
                                _ code: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: code)
    }
}
