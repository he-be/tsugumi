import Foundation
import Hub
import Tokenizers

public enum GFTokenizerError: Error, CustomStringConvertible {
    case missingSpecialToken(String)
    case invalidChatTemplate(String)
    case missingToolTemplate
    case missingTokenizerConfig
    case unsupportedDecoder(actual: String)
    case invalidTokenID(token: String, id: Int)

    public var description: String {
        switch self {
        case .missingSpecialToken(let t): return "tokenizer missing required special token: \(t)"
        case .invalidChatTemplate(let detail): return "invalid chat messages: \(detail)"
        case .missingToolTemplate:
            return "installed tokenizer is missing chat_template.jinja; reinstall the model"
        case .missingTokenizerConfig:
            return "tokenizer_config.json is missing or unreadable"
        case .unsupportedDecoder(let actual):
            return "tokenizer decoder is not the pinned Gemma 4 sequence "
                + "Sequence[Replace(▁→␣), ByteFallback, Fuse]; found: \(actual)"
        case .invalidTokenID(let token, let id):
            return "tokenizer declares out-of-range ID \(id) for token \(token)"
        }
    }
}

/// Gemma 4 tokenizer wrapper.
///
/// Prefers tokenizer sidecars in a completed `.gturbo/tokenizer/` directory,
/// then falls back to the IT variant's Hugging Face Hub tokenizer cache. Exposes
/// typed accessors for the IDs the generator actually needs (BOS / EOS / pad /
/// end-of-turn) and adapts encode/decode to Int32 to match the buffer types
/// kernels consume.
///
/// TurboFieldfare owns the minimal chat framing because the upstream
/// `tokenizer_config.json` has no `chat_template`. Literal control-token text in
/// user content is accepted as a trusted-input research-runtime limitation —
/// with one carve-out: the multimodal markers (`<|image|>`, `<|audio|>`,
/// `<|video|>`, and their openers/closers) are rejected outright, because those
/// ids would otherwise be embedded as ordinary text and produce a fluent answer
/// about an image nobody supplied (`VisionPromptAssembler.rejectMediaMarkers`).
public struct GFTokenizer: @unchecked Sendable {
    public static let modelID = "google/gemma-4-26B-A4B-it"
    public static let chatTemplateIdentity = "gemma4-it-text-no-tools-v1"
    public static let toolChatTemplateIdentity = "gemma4-it-tools-jinja-v1"

    public let bosID: Int32
    public let eosID: Int32
    public let padID: Int32
    public let endOfTurnID: Int32
    public let toolCallStartID: Int32
    public let toolCallEndID: Int32
    public let toolResponseID: Int32
    public let toolResponseEndID: Int32
    public let channelStartID: Int32
    public let channelEndID: Int32
    public let stopTokenIDs: Set<Int32>
    public let vocabSize: Int
    /// The channel/tool markers that structure assistant output. Streaming
    /// treats them as detokenizer barriers: a byte-fallback run must not span
    /// one, or its text would surface after the marker and be routed under the
    /// wrong channel state.
    public var structuralMarkerIDs: Set<Int32> {
        [toolCallStartID, toolCallEndID, toolResponseID, toolResponseEndID,
         channelStartID, channelEndID]
    }
    /// IDs that `decode(skipSpecialTokens: true)` strips — the
    /// `added_tokens[special == true]` set from `tokenizer.json`, identical to
    /// the filter the library's own decode applies before its decoder chain.
    let specialTokenIDs: Set<Int32>

    @usableFromInline
    let tokenizer: any Tokenizer

    public static func load() async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(.pretrained(modelID))
    }

    public static func load(from folder: URL) async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(.local(folder.standardizedFileURL.path))
    }

    public static func load(forModelDirectory modelDirectory: URL,
                            environment: [String: String] = ProcessInfo.processInfo.environment) async throws -> GFTokenizer {
        if let folder = tokenizerFolder(forModelDirectory: modelDirectory, environment: environment) {
            return try await load(from: folder)
        }
        return try await load()
    }

    public static func tokenizerFolder(forModelDirectory modelDirectory: URL,
                                       environment: [String: String] = ProcessInfo.processInfo.environment,
                                       fileManager: FileManager = .default) -> URL? {
        let sidecar = modelDirectory
            .standardizedFileURL
            .appendingPathComponent("tokenizer", isDirectory: true)
        if hasTokenizerJSON(in: sidecar, fileManager: fileManager) {
            return sidecar
        }

        guard let override = environment["TURBO_FIELDFARE_TOKENIZER_DIR"], !override.isEmpty else {
            return nil
        }
        let overrideURL = URL(fileURLWithPath: override).standardizedFileURL
        return hasTokenizerJSON(in: overrideURL, fileManager: fileManager) ? overrideURL : nil
    }

    static func loadUncached(pretrained modelID: String = Self.modelID) async throws -> GFTokenizer {
        try await make(from: LanguageModelConfigurationFromHub(modelName: modelID))
    }

    static func loadUncached(from folder: URL) async throws -> GFTokenizer {
        try await make(from: LanguageModelConfigurationFromHub(modelFolder: folder))
    }

    /// Build from the raw tokenizer configs so the decoder pipeline and the
    /// special-token set can be validated against `tokenizer.json` itself.
    /// Same fetch/caching path `AutoTokenizer.from(pretrained:)` uses internally.
    private static func make(from hub: LanguageModelConfigurationFromHub) async throws -> GFTokenizer {
        guard let tokenizerConfig = try await hub.tokenizerConfig else {
            throw GFTokenizerError.missingTokenizerConfig
        }
        let tokenizerData = try await hub.tokenizerData
        let underlying = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        return try GFTokenizer(tokenizer: underlying, tokenizerData: tokenizerData)
    }

    private static func hasTokenizerJSON(in folder: URL, fileManager: FileManager) -> Bool {
        fileManager.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path)
    }

    /// Reject a tokenizer whose declared decoder is not the pinned Gemma 4
    /// sequence.
    ///
    /// `GemmaDecoding` reproduces `Sequence[Replace("▁" -> " "), ByteFallback,
    /// Fuse]` rather than calling `Tokenizers.decode`, so that decode stays
    /// lossless and per-token (see `GemmaDecoding`). The installed tokenizer is
    /// pinned and hash-validated, but `TURBO_FIELDFARE_TOKENIZER_DIR` can point
    /// at any directory. Without this check a tokenizer declaring a different
    /// decoder would decode subtly wrong text instead of failing.
    ///
    /// The check reads `tokenizer.json`'s decoder declaration structurally, so
    /// it rejects any foreign decoder — including one whose behavior happens to
    /// coincide on a handful of probe strings — without asserting the library's
    /// exact runtime output, which a benign dependency bump may change.
    /// Behavioral agreement with the library is pinned by the differential
    /// tests instead.
    static func verifyDecoderConfiguration(_ tokenizerData: Config) throws {
        let decoder = tokenizerData["decoder"]
        let steps = decoder.decoders.array(or: [])
        guard decoder.type.string() == "Sequence",
              steps.count == 3,
              steps[0].type.string() == "Replace",
              steps[0].pattern.String.string() == "▁",
              steps[0].content.string() == " ",
              steps[1].type.string() == "ByteFallback",
              steps[2].type.string() == "Fuse"
        else {
            throw GFTokenizerError.unsupportedDecoder(actual: decoder.description)
        }
    }

    /// Resolve a special token to its ID, rejecting `<unk>` substitution.
    ///
    /// BPE's `convertTokenToId` returns the unknown-token ID — not `nil` — for
    /// a token absent from the vocab, so a plain `guard let` never fires and a
    /// missing marker would silently bind to `<unk>`, colliding with every
    /// other missing marker. The round-trip through `convertIdToToken` detects
    /// any substitution.
    static func requireTokenID(_ tokenizer: any Tokenizer, _ token: String) throws -> Int32 {
        guard let id = tokenizer.convertTokenToId(token),
              tokenizer.convertIdToToken(id) == token else {
            throw GFTokenizerError.missingSpecialToken(token)
        }
        return try int32ID(token, id)
    }

    /// Config-supplied IDs are full-range `Int`; a trapping `Int32(_:)` would
    /// crash on a corrupt tokenizer instead of taking the typed error path
    /// every other malformed-config case uses.
    private static func int32ID(_ token: String, _ id: Int) throws -> Int32 {
        guard let value = Int32(exactly: id) else {
            throw GFTokenizerError.invalidTokenID(token: token, id: id)
        }
        return value
    }

    public init(tokenizer: any Tokenizer, tokenizerData: Config) throws {
        self.tokenizer = tokenizer
        try Self.verifyDecoderConfiguration(tokenizerData)

        // The same `added_tokens[special == true]` ID set the library's
        // `decode(skipSpecialTokens: true)` filters before running its decoder.
        var specials: Set<Int32> = []
        for added in tokenizerData["addedTokens"].array(or: []) {
            guard added["special"].boolean(or: false),
                  let id = added["id"].integer() else { continue }
            try specials.insert(Self.int32ID(added.content.string() ?? "added token", id))
        }
        self.specialTokenIDs = specials

        guard let bos = tokenizer.bosTokenId else {
            throw GFTokenizerError.missingSpecialToken("<bos>")
        }
        guard let eos = tokenizer.eosTokenId else {
            throw GFTokenizerError.missingSpecialToken("<eos>")
        }
        self.bosID = try Self.int32ID("<bos>", bos)
        self.eosID = try Self.int32ID("<eos>", eos)
        self.padID = try Self.requireTokenID(tokenizer, "<pad>")
        self.endOfTurnID = try Self.requireTokenID(tokenizer, "<turn|>")
        self.toolCallStartID = try Self.requireTokenID(tokenizer, "<|tool_call>")
        self.toolCallEndID = try Self.requireTokenID(tokenizer, "<tool_call|>")
        self.toolResponseID = try Self.requireTokenID(tokenizer, "<|tool_response>")
        self.toolResponseEndID = try Self.requireTokenID(tokenizer, "<tool_response|>")
        self.channelStartID = try Self.requireTokenID(tokenizer, "<|channel>")
        self.channelEndID = try Self.requireTokenID(tokenizer, "<channel|>")
        self.stopTokenIDs = [self.eosID, self.endOfTurnID, self.toolResponseID]
        self.vocabSize = 262_144
    }

    /// Encode UTF-8 text to token IDs. `addBOS = true` prepends `<bos>`.
    ///
    /// The library's `addSpecialTokens: true` flag is a no-op for the Gemma 4 IT
    /// tokenizer (its config has `add_bos_token = false`; BOS is expected to come
    /// from the chat template). We prepend manually so the kernel-facing API stays
    /// the same regardless of upstream defaults.
    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        let base = tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
        return addBOS ? [bosID] + base : base
    }

    /// Decode token IDs to text. `skipSpecialTokens` strips BOS/EOS/turn markers from the output.
    ///
    /// Runs the pinned Gemma decoder sequence directly (see `GemmaDecoding`)
    /// rather than `Tokenizers.decode`, whose trailing
    /// `clean_up_tokenization_spaces` pass defaults to on for this tokenizer and
    /// rewrites model output (`"he said ' ok ' now"` -> `"he said'ok'now"`),
    /// breaking `decode(encode(x)) == x`. Batch decode is a push-loop over
    /// `GFDetokenizer`, so batch and streaming decode agree by construction.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        var detok = GFDetokenizer(tokenizer: self, skipSpecialTokens: skipSpecialTokens)
        var text = ""
        for id in ids {
            text += detok.push(id)
        }
        return text + detok.flush()
    }

    // MARK: - Chat template

    public enum Role: String, Sendable { case system, developer, user, assistant, tool }
    public struct HistoricalToolCall: Sendable, Equatable {
        public let id: String
        public let name: String
        public let arguments: JSONValue

        public init(id: String, name: String, arguments: JSONValue) {
            self.id = id
            self.name = name
            self.arguments = arguments
        }
    }

    public struct FunctionDefinition: Sendable, Equatable {
        public let name: String
        public let description: String
        public let parameters: JSONValue

        public init(name: String, description: String, parameters: JSONValue) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }
    }

    public struct Message: Sendable, Equatable {
        public let role: Role
        public let content: String?
        public let toolCalls: [HistoricalToolCall]
        public let toolCallID: String?
        public let name: String?

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
            self.toolCalls = []
            self.toolCallID = nil
            self.name = nil
        }

        public init(role: Role,
                    content: String?,
                    toolCalls: [HistoricalToolCall] = [],
                    toolCallID: String? = nil,
                    name: String? = nil) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
            self.toolCallID = toolCallID
            self.name = name
        }
    }

    /// One piece of a message body. `image` is a placeholder the caller has
    /// attached a real image to; it renders as the single `<|image|>` token the
    /// bundled template emits for an `image` content part, and
    /// `VisionPromptAssembler.expandImagePlaceholders` later widens it to
    /// `<|image>` + n soft tokens + `<image|>`.
    public enum ContentPart: Sendable, Equatable {
        case text(String)
        case image
    }

    /// A message whose body is a sequence of parts. `Message` is the text-only
    /// special case of this and renders identically.
    public struct MultimodalMessage: Sendable, Equatable {
        public let role: Role
        public let parts: [ContentPart]

        public init(role: Role, parts: [ContentPart]) {
            self.role = role
            self.parts = parts
        }

        public var imageCount: Int {
            parts.reduce(into: 0) { $0 += ($1 == .image ? 1 : 0) }
        }
    }

    /// Text-only, no-tool rendering of the pinned IT checkpoint's bundled
    /// `chat_template.jinja`. Keeping this narrow makes unsupported tool/media
    /// behavior explicit instead of approximating it.
    private static let turnOpen    = "<|turn>"
    private static let turnClose   = "<turn|>"
    private static let bosMark     = "<bos>"
    private static let thinkMark   = "<|think|>"

    /// - Parameter enableThinking: mirrors the template's `enable_thinking`.
    ///   When false the generation prompt opens the thought channel and closes
    ///   it immediately, which tells the model to answer without reasoning.
    ///   When true a `<|think|>` marker leads the system turn and the thought
    ///   channel is left for the model to open, so it reasons first. The
    ///   reasoning is real output: it costs generated tokens, and callers that
    ///   want it hidden route it through `StructuredAssistantDecoder`.
    public func applyChatTemplate(_ messages: [Message],
                                  enableThinking: Bool = false) throws -> String {
        let multimodal = try messages.map { message -> MultimodalMessage in
            guard let content = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            return MultimodalMessage(role: message.role, parts: [.text(content)])
        }
        return try applyChatTemplate(multimodal: multimodal, enableThinking: enableThinking)
    }

    /// The same rendering with image placeholders allowed in the bodies.
    ///
    /// Each text part is trimmed on its own and the parts are concatenated with
    /// no separator, which is what the bundled template does with a sequence
    /// `content` (`chat_template.jinja`: `item['text'] | trim`, then
    /// `'<|image|>'` for an image item). A message of one text part therefore
    /// renders byte for byte as the text-only path above.
    public func applyChatTemplate(multimodal messages: [MultimodalMessage],
                                  enableThinking: Bool = false) throws -> String {
        func render(_ parts: [ContentPart]) throws -> String {
            var body = ""
            for part in parts {
                switch part {
                case .text(let text):
                    try VisionPromptAssembler.rejectMediaMarkers(in: text)
                    body += text.trimmingCharacters(in: .whitespacesAndNewlines)
                case .image:
                    body += VisionMediaTokenIDs.imageToken
                }
            }
            return body
        }

        var s = Self.bosMark
        var rest = messages[...]
        // The template emits a system turn whenever thinking is on, even with
        // no system message to put in it, because that turn carries the marker.
        if enableThinking {
            var body = ""
            if let first = messages.first, first.role == .system {
                body = try render(first.parts)
                rest = messages.dropFirst()
            }
            s += Self.turnOpen + "system\n" + Self.thinkMark + "\n" + body
                + Self.turnClose + "\n"
        }
        for (index, message) in rest.enumerated() {
            if message.role == .system && index != 0 {
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            }
            let content = try render(message.parts)
            let role = message.role == .assistant ? "model" : message.role.rawValue
            s += Self.turnOpen + role + "\n" + content + Self.turnClose + "\n"
        }
        s += Self.turnOpen + "model\n"
        if !enableThinking { s += "<|channel>thought\n<channel|>" }
        return s
    }

    /// A turn on the tool-calling path: `Message`'s metadata with a body that
    /// can hold images.
    ///
    /// The tool template renders content as either a string or a sequence of
    /// parts, and it needs the tool metadata (`tool_calls`, `tool_call_id`) that
    /// `MultimodalMessage` has no room for — so the tool path gets its own turn
    /// type rather than either of the two halves growing the other's fields.
    public struct ToolChatMessage: Sendable, Equatable {
        public let role: Role
        public let parts: [ContentPart]
        public let toolCalls: [HistoricalToolCall]
        public let toolCallID: String?
        public let name: String?

        public init(role: Role,
                    parts: [ContentPart],
                    toolCalls: [HistoricalToolCall] = [],
                    toolCallID: String? = nil,
                    name: String? = nil) {
            self.role = role
            self.parts = parts
            self.toolCalls = toolCalls
            self.toolCallID = toolCallID
            self.name = name
        }

        public init(_ message: Message) {
            self.init(role: message.role,
                      parts: message.content.map { [.text($0)] } ?? [],
                      toolCalls: message.toolCalls,
                      toolCallID: message.toolCallID,
                      name: message.name)
        }

        public var imageCount: Int {
            parts.reduce(into: 0) { $0 += ($1 == .image ? 1 : 0) }
        }
    }

    public func encodeToolChat(messages: [Message],
                               tools: [FunctionDefinition],
                               enableThinking: Bool = false) throws -> [Int32] {
        try encodeToolChat(messages: messages.map(ToolChatMessage.init),
                           tools: tools,
                           enableThinking: enableThinking)
    }

    /// The same rendering with image placeholders allowed in the bodies.
    ///
    /// A turn whose body is one text part is sent as a *string*, exactly as the
    /// text-only overload always did, so a request that carries no image
    /// renders byte for byte as before (11-S2 §2). Only a turn that actually
    /// holds an image is sent as a parts array.
    public func encodeToolChat(messages: [ToolChatMessage],
                               tools: [FunctionDefinition],
                               enableThinking: Bool = false) throws -> [Int32] {
        guard tokenizer.hasChatTemplate else {
            throw GFTokenizerError.missingToolTemplate
        }
        let upstreamMessages: [Tokenizers.Message] = try messages.map { message in
            var value: Tokenizers.Message = [
                "role": message.role.rawValue,
                "content": try Self.toolChatContent(message.parts),
            ]
            if !message.toolCalls.isEmpty {
                value["tool_calls"] = try message.toolCalls.map { call -> [String: any Sendable] in
                    [
                        "id": call.id,
                        "type": "function",
                        "function": [
                            "name": call.name,
                            "arguments": try call.arguments.jinjaSendableValue(),
                        ] as [String: any Sendable],
                    ]
                }
            }
            if let toolCallID = message.toolCallID { value["tool_call_id"] = toolCallID }
            if let name = message.name { value["name"] = name }
            return value
        }
        let upstreamTools: [ToolSpec] = try tools.map { tool in
            [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": try tool.parameters.jinjaSendableValue(),
                ] as [String: any Sendable],
            ]
        }
        return try tokenizer.applyChatTemplate(
            messages: upstreamMessages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: upstreamTools,
            additionalContext: ["enable_thinking": enableThinking]
        ).map(Int32.init)
    }

    /// The `content` value for one tool-path turn.
    ///
    /// nil for a turn that carries only tool calls, a string for a plain text
    /// body, and the template's `[{type: text|image}]` sequence when an image
    /// is present. Text parts are trimmed here because the template trims a
    /// string body but not the items of a sequence.
    private static func toolChatContent(_ parts: [ContentPart]) throws -> (any Sendable)? {
        guard parts.contains(.image) else {
            let text = parts.compactMap { part -> String? in
                if case .text(let text) = part { return text }
                return nil
            }.joined()
            return parts.isEmpty ? nil : text
        }
        return try parts.map { part -> [String: any Sendable] in
            switch part {
            case .text(let text):
                try VisionPromptAssembler.rejectMediaMarkers(in: text)
                return ["type": "text", "text": text]
            case .image:
                return ["type": "image"]
            }
        } as [any Sendable]
    }

    /// The tokens that carry a cached generation into the next user turn.
    ///
    /// `enableThinking` has to agree with the generation prompt the cached KV
    /// was built from: the thought channel is opened and closed here only when
    /// the session is not reasoning, exactly as `applyChatTemplate` does.
    public func encodeTextContinuation(userContent: String,
                                       enableThinking: Bool = false) throws -> [Int32] {
        try encodeContinuation(parts: [.text(userContent)], enableThinking: enableThinking)
    }

    /// The same bridge with image placeholders allowed in the new user turn.
    ///
    /// The `<|image|>` markers are left for `VisionPromptAssembler` to widen, so
    /// the spans it reports are offsets into this bridge — which is the slice a
    /// resumed prefill actually processes.
    public func encodeContinuation(parts: [ContentPart],
                                   enableThinking: Bool = false) throws -> [Int32] {
        var body = ""
        for part in parts {
            switch part {
            case .text(let text):
                try VisionPromptAssembler.rejectMediaMarkers(in: text)
                body += text.trimmingCharacters(in: .whitespacesAndNewlines)
            case .image:
                body += VisionMediaTokenIDs.imageToken
            }
        }
        return [endOfTurnID] + encode(
            "\n\(Self.turnOpen)user\n\(body)\(Self.turnClose)\n"
                + "\(Self.turnOpen)model\n"
                + (enableThinking ? "" : "<|channel>thought\n<channel|>"),
            addBOS: false)
    }

    public func encodeToolResultContinuation(
        cachedMessages: [Message],
        assistant: Message,
        incomingMessages: [Message],
        tools: [FunctionDefinition],
        enableThinking: Bool = false
    ) throws -> [Int32] {
        try encodeToolResultContinuation(
            cachedMessages: cachedMessages,
            assistant: assistant,
            incoming: incomingMessages.map(ToolChatMessage.init),
            tools: tools,
            enableThinking: enableThinking)
    }

    /// The same continuation with the incoming turns able to carry images.
    ///
    /// Everything after the cached assistant's tool call is the bridge —
    /// the tool responses, and whatever turn the client appended after them.
    /// `cachedMessages` is only used to locate that call, so its bodies never
    /// reach the output and may stay text-only.
    public func encodeToolResultContinuation(
        cachedMessages: [Message],
        assistant: Message,
        incoming: [ToolChatMessage],
        tools: [FunctionDefinition],
        enableThinking: Bool = false
    ) throws -> [Int32] {
        let prefix = try encodeToolChat(
            messages: cachedMessages + [assistant],
            tools: tools,
            enableThinking: enableThinking)
        let full = try encodeToolChat(messages: incoming,
                                      tools: tools,
                                      enableThinking: enableThinking)
        let callCount = assistant.toolCalls.count
        let starts = prefix.indices.filter { prefix[$0] == toolCallStartID }
        guard callCount > 0, starts.count >= callCount,
              let callEnd = prefix.lastIndex(of: toolCallEndID) else {
            throw GFTokenizerError.invalidChatTemplate(
                "cached assistant tool-call boundary is missing")
        }
        let callStart = starts[starts.count - callCount]
        let callSequence = Array(prefix[callStart...callEnd])
        let matches = full.subsequenceStartIndices(matching: callSequence)
        guard matches.count == 1 else {
            throw GFTokenizerError.invalidChatTemplate(
                "cached assistant tool-call boundary is ambiguous")
        }
        let suffixStart = matches[0] + callSequence.count
        let suffix = Array(full[suffixStart...])
        guard suffix.first == toolResponseID else {
            throw GFTokenizerError.invalidChatTemplate(
                "tool-result continuation does not begin at the KV boundary")
        }
        return suffix
    }
}

private extension Array where Element: Equatable {
    func subsequenceStartIndices(matching needle: [Element]) -> [Int] {
        guard !needle.isEmpty, needle.count <= count else { return [] }
        return indices.dropLast(needle.count - 1).filter { start in
            self[start..<(start + needle.count)].elementsEqual(needle)
        }
    }
}

private enum GFTokenizerLoadSource: Hashable {
    case pretrained(String)
    case local(String)
}

private actor GFTokenizerLoadCoordinator {
    static let shared = GFTokenizerLoadCoordinator()

    private var tasks: [GFTokenizerLoadSource: Task<GFTokenizer, Error>] = [:]

    func load(_ source: GFTokenizerLoadSource) async throws -> GFTokenizer {
        if let task = tasks[source] {
            return try await task.value
        }

        // Keep the CPU-heavy tokenizer build off the coordinator actor; callers
        // share the task result instead of owning its cancellation.
        let task = Task.detached(priority: .userInitiated) { () throws -> GFTokenizer in
            switch source {
            case .pretrained(let modelID):
                return try await GFTokenizer.loadUncached(pretrained: modelID)
            case .local(let path):
                return try await GFTokenizer.loadUncached(from: URL(fileURLWithPath: path))
            }
        }
        tasks[source] = task

        do {
            return try await task.value
        } catch {
            tasks[source] = nil
            throw error
        }
    }
}
