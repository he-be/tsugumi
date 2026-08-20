import Foundation
import TurboFieldfare

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
                        throw ServerRequestError.invalid(
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
    /// MSG-5: the thinking that produced a finished assistant turn, handed
    /// back by the client. Until it was read here it was dropped as an unknown
    /// key (R1), which is why a redraw of a reasoning turn lost the whole
    /// thought block and the common prefix stopped at the turn's first token.
    public let reasoningContent: String?

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
        case reasoningContent = "reasoning_content"
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

public struct ValidatedChatRequest: Sendable {
    /// The text projection of the conversation. When `vision != nil` the image
    /// parts are *not* in here — rendering goes through `vision.messages`
    /// instead, and the prompt cache (which keys on this array) is off.
    public let messages: [GFTokenizer.Message]
    public let tools: [GFTokenizer.FunctionDefinition]
    /// GEN-2 / DEV-16: what the declared tool schemas lost on the way into the
    /// prompt, in declaration order. Never an error — the server logs these.
    /// This is the *declaration* side; what the grammar could not constrain is
    /// a separate list on the constraint, because the two degrade
    /// independently.
    public let toolSchemaSimplifications: [String]
    public let stream: Bool
    public let includeUsage: Bool
    public let generationConfig: GenerationConfig
    /// REQ-max-tokens. **-1 means unlimited** (whatever the context has left
    /// after the prompt) and 0 means prefill only; both are values the client
    /// may send, so this is not a plain positive count.
    public let maximumCompletionTokens: Int
    public let vision: ValidatedVisionRequest?
    /// What this request asked the thought channel to do, after the process
    /// default. Whether it is actually rendered is a second question the
    /// template answers (`ServerModelSession`): a request that declares tools
    /// goes through the tool-calling template, which pins thinking off.
    public let enableThinking: Bool

    /// The conversation as the tool template takes it: the roles and tool
    /// metadata of `messages`, with the bodies of `vision.messages` when the
    /// request carried pictures. The two are built in lockstep by the
    /// validator, so they are joined by position.
    public var toolChatMessages: [GFTokenizer.ToolChatMessage] {
        guard let multimodal = vision?.messages, multimodal.count == messages.count else {
            return messages.map(GFTokenizer.ToolChatMessage.init)
        }
        return zip(messages, multimodal).map { message, parts in
            GFTokenizer.ToolChatMessage(role: message.role,
                                        parts: parts.parts,
                                        toolCalls: message.toolCalls,
                                        toolCallID: message.toolCallID,
                                        name: message.name,
                                        reasoningContent: message.reasoningContent)
        }
    }

    /// R5: the name the client asked for, written back into the response
    /// unexamined. A single-model server has nothing to check it against.
    public let model: String
    /// REQ-tool-choice, all four shapes (GEN-4). The two that force a call —
    /// `required` and a named function — are only ever carried here with a
    /// tool the grammar can actually pin (the parser refuses the rest).
    public let toolChoice: ChatToolChoice
    /// REQ-response-format (GEN-3). What the answer must be shaped like, for
    /// the grammar stage to turn into a constraint. `text` is no constraint.
    public let responseFormat: ChatResponseFormat
    public let parallelToolCalls: Bool
    /// REQ-cache-prompt. Whether this request may read from or write to the
    /// prompt cache (CACHE-5).
    public let cachePrompt: Bool
    /// REQ-reasoning-effort, verbatim. Only `"none"` has a meaning here
    /// (thinking off); every other value rides through to the template
    /// unexamined, as the reference implementation does.
    public let reasoningEffort: String?
    /// REQ-template-kwargs, verbatim.
    public let chatTemplateKwargs: [String: JSONValue]
    /// REQ-reasoning-budget. -1 is unlimited, 0 disables the thought channel.
    public let reasoningBudgetTokens: Int
    public let reasoningFormat: ReasoningFormat
    public let timingsPerToken: Bool

    public init(messages: [GFTokenizer.Message],
                tools: [GFTokenizer.FunctionDefinition],
                toolSchemaSimplifications: [String] = [],
                stream: Bool,
                includeUsage: Bool,
                generationConfig: GenerationConfig,
                maximumCompletionTokens: Int,
                vision: ValidatedVisionRequest? = nil,
                enableThinking: Bool = false,
                model: String = "",
                toolChoice: ChatToolChoice = .auto,
                responseFormat: ChatResponseFormat = .text,
                parallelToolCalls: Bool = true,
                cachePrompt: Bool = true,
                reasoningEffort: String? = nil,
                chatTemplateKwargs: [String: JSONValue] = [:],
                reasoningBudgetTokens: Int = -1,
                reasoningFormat: ReasoningFormat = .auto,
                timingsPerToken: Bool = false) {
        self.messages = messages
        self.tools = tools
        self.toolSchemaSimplifications = toolSchemaSimplifications
        self.stream = stream
        self.includeUsage = includeUsage
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.vision = vision
        self.enableThinking = enableThinking
        self.model = model
        self.toolChoice = toolChoice
        self.responseFormat = responseFormat
        self.parallelToolCalls = parallelToolCalls
        self.cachePrompt = cachePrompt
        self.reasoningEffort = reasoningEffort
        self.chatTemplateKwargs = chatTemplateKwargs
        self.reasoningBudgetTokens = reasoningBudgetTokens
        self.reasoningFormat = reasoningFormat
        self.timingsPerToken = timingsPerToken
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

/// SPEC §5 and the tool half of §6: everything about a request that is not a
/// row of the §4 table.
///
/// What used to live here as well — the per-parameter `guard`s of
/// `OpenAIRequestValidator` — is `ChatRequestSchema` now. This type only knows
/// about message and tool shapes.
public enum ChatMessageValidator {
    /// One declared tool the template can render, and what adapting its schema
    /// cost (GEN-2 / DEV-16).
    struct ValidatedTool {
        let definition: GFTokenizer.FunctionDefinition
        let simplifications: [String]
    }

    static func validateTool(_ tool: OpenAITool) throws -> ValidatedTool {
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
        // GEN-2: the schema's *content* is never a refusal. Everything the
        // declaration cannot render comes back simplified, with a note.
        let adapted = GemmaToolSchema.adapted(tool.function.parameters, toolName: name)
        guard (try? adapted.schema.jinjaSendableValue()) != nil else {
            throw invalid("tool schema contains a number that cannot be represented exactly",
                          "tools", "invalid_tool_schema")
        }
        return ValidatedTool(
            definition: GFTokenizer.FunctionDefinition(
                name: name,
                description: tool.function.description ?? "",
                parameters: adapted.schema),
            simplifications: adapted.simplifications)
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

    struct ValidatedMessages {
        let messages: [GFTokenizer.Message]
        let vision: ValidatedVisionRequest?
    }

    static func validateMessages(
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
            multimodal.append(GFTokenizer.MultimodalMessage(
                role: role, parts: parts ?? [], reasoningContent: message.reasoningContent))
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
                                              name: message.name,
                                              reasoningContent: message.reasoningContent))
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
