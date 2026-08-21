import Foundation
import Hub
import Tokenizers

public enum QwenTokenizerError: Error, CustomStringConvertible {
    case missingTokenizerConfig
    case missingSpecialToken(String)
    case invalidTokenID(token: String, id: Int)
    case unsupportedDecoder(actual: String)
    case unsupportedPreTokenizer(actual: String)
    case missingChatTemplate
    case invalidChatMessages(String)

    public var description: String {
        switch self {
        case .missingTokenizerConfig:
            return "tokenizer_config.json is missing or unreadable"
        case .missingSpecialToken(let token):
            return "tokenizer missing required special token: \(token)"
        case .invalidTokenID(let token, let id):
            return "tokenizer declares out-of-range ID \(id) for token \(token)"
        case .unsupportedDecoder(let actual):
            return "tokenizer decoder is not the pinned Ornith ByteLevel decoder; found: \(actual)"
        case .unsupportedPreTokenizer(let actual):
            return "tokenizer pre-tokenizer does not end in ByteLevel; found: \(actual)"
        case .missingChatTemplate:
            return "installed tokenizer has no chat template; reinstall the model"
        case .invalidChatMessages(let detail):
            return "invalid chat messages: \(detail)"
        }
    }
}

/// Ornith 1.5 (Qwen 3.5-MoE) tokenizer wrapper.
///
/// The sibling of `GFTokenizer`, not a generalization of it. The two
/// checkpoints agree on nothing that matters here: Ornith is a **byte-level
/// BPE** with no BOS, its markers are ChatML (`<|im_start|>` / `<|im_end|>`),
/// its stop set comes from `generation_config.json`
/// (`<|im_end|>`, `<|endoftext|>`), and — unlike Gemma — it **ships its own
/// `chat_template.jinja`**, so the framing is rendered by that template rather
/// than written out in Swift. Keeping Gemma's loader pinned to Gemma is the
/// point: its decoder check exists to reject a foreign tokenizer, and a
/// tokenizer that is one family's is the other's corruption.
///
/// What is shared is the streaming discipline: decode is per-token and
/// lossless, batch decode is a push-loop over the same streaming type, and
/// nothing runs HF's `clean_up_tokenization_spaces` pass (Ornith's config turns
/// it off anyway, unlike Gemma's, which omits the key and so defaults it on).
public struct QwenTokenizer: @unchecked Sendable {
    /// The checkpoint this loader is pinned to. Used as a cache key where a
    /// process could hold both families' tables at once
    /// (`GrammarVocabulary.shared(for:)`).
    public static let modelID = "ornith-ai/Ornith-1.5-35B-A3B"

    /// What a prompt cache would have to agree on: the framing is the
    /// checkpoint's own `chat_template.jinja`, rendered by swift-jinja.
    public static let chatTemplateIdentity = "ornith-1.5-bundled-jinja-v1"

    /// ChatML framing.
    public let imStartID: Int32
    public let imEndID: Int32
    /// `generation_config.json`'s other stop token, and the pad token.
    public let endOfTextID: Int32
    /// The reasoning block the template opens for the model to continue.
    public let thinkStartID: Int32
    public let thinkEndID: Int32
    /// Tool-call framing. The bodies are XML, not JSON (04-PHASES Phase 5).
    public let toolCallStartID: Int32
    public let toolCallEndID: Int32
    public let toolResponseStartID: Int32
    public let toolResponseEndID: Int32
    /// `eos_token_id` from `generation_config.json`: `<|im_end|>` when the turn
    /// ends, `<|endoftext|>` when the document does.
    public let stopTokenIDs: Set<Int32>
    /// The tokenizer's own vocabulary size (highest ID + 1). The LM head is
    /// wider than this — the checkpoint's `vocab_size` counts rows that no
    /// token maps to (19-LM-HEAD-INT8 §1) — so the two numbers are not
    /// interchangeable and this one is never used to size a buffer.
    public let vocabSize: Int

    /// `added_tokens[special == true]` — the set `decode(skipSpecialTokens:)`
    /// drops, identical to the filter the library applies before its decoder.
    let specialTokenIDs: Set<Int32>
    /// Every added token, special or not. The reference ByteLevel decoder emits
    /// these literally instead of byte-decoding them, and a byte run continues
    /// across the ones that are dropped.
    let addedTokenIDs: Set<Int32>

    @usableFromInline
    let tokenizer: any Tokenizer

    // MARK: - Loading

    public static func load(from folder: URL) async throws -> QwenTokenizer {
        try await make(from: LanguageModelConfigurationFromHub(
            modelFolder: folder.standardizedFileURL))
    }

    /// The tokenizer sidecar a repacked `.gturbo` carries
    /// (`<model>/tokenizer/`), or `TURBO_FIELDFARE_TOKENIZER_DIR`. There is no
    /// Hub fallback: this checkpoint is local-only.
    public static func load(forModelDirectory modelDirectory: URL,
                            environment: [String: String] = ProcessInfo.processInfo.environment)
        async throws -> QwenTokenizer
    {
        guard let folder = GFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory,
                                                       environment: environment) else {
            throw QwenTokenizerError.missingTokenizerConfig
        }
        return try await load(from: folder)
    }

    /// The checkpoint's own `chat_template.jinja`, as text.
    ///
    /// SPEC EP-4 `chat_template` is "the template the server actually renders
    /// with", and for this family that is the file the checkpoint ships — the
    /// repack copies it into the sidecar beside `tokenizer.json`, and
    /// `applyChatTemplate` renders it through swift-jinja. Read from the file
    /// rather than reconstructed, so `/props` cannot describe a template that
    /// is not the one in use.
    public static func chatTemplateJinja(
        forModelDirectory modelDirectory: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> String {
        guard let folder = GFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory,
                                                       environment: environment) else {
            throw QwenTokenizerError.missingTokenizerConfig
        }
        return try String(contentsOf: folder.appendingPathComponent("chat_template.jinja"),
                          encoding: .utf8)
    }

    private static func make(from hub: LanguageModelConfigurationFromHub) async throws -> QwenTokenizer {
        guard let tokenizerConfig = try await hub.tokenizerConfig else {
            throw QwenTokenizerError.missingTokenizerConfig
        }
        let tokenizerData = try await hub.tokenizerData
        let underlying = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig,
                                                tokenizerData: tokenizerData)
        return try QwenTokenizer(tokenizer: underlying, tokenizerData: tokenizerData)
    }

    // MARK: - Verification

    /// Reject a tokenizer whose declared decoder is not Ornith's ByteLevel one.
    ///
    /// `ByteLevelRun` reproduces that decoder rather than calling
    /// `Tokenizers.decode`, so decode stays per-token and lossless. The
    /// installed sidecar is what the repack wrote, but
    /// `TURBO_FIELDFARE_TOKENIZER_DIR` can point anywhere; without this check a
    /// metaspace tokenizer would decode every token as if its characters were
    /// bytes and produce mojibake instead of failing.
    ///
    /// Read structurally, like the Gemma check: the declaration, not the
    /// library's runtime output on a handful of probes.
    static func verifyDecoderConfiguration(_ tokenizerData: Config) throws {
        let decoder = tokenizerData["decoder"]
        guard decoder.type.string() == "ByteLevel" else {
            throw QwenTokenizerError.unsupportedDecoder(actual: decoder.description)
        }
    }

    /// The other half of the same claim: the *encoder* is byte-level too.
    ///
    /// A ByteLevel decoder over a metaspace pre-tokenizer would round-trip
    /// nothing, and the failure would look like a model bug rather than a
    /// tokenizer one. Ornith declares `Sequence[Split(regex), ByteLevel]`; the
    /// check requires a ByteLevel step to exist, in either the sequence or on
    /// its own.
    static func verifyPreTokenizerConfiguration(_ tokenizerData: Config) throws {
        let pre = tokenizerData["preTokenizer"]
        if pre.type.string() == "ByteLevel" { return }
        let steps = pre.pretokenizers.array(or: [])
        guard pre.type.string() == "Sequence",
              steps.contains(where: { $0.type.string() == "ByteLevel" })
        else {
            throw QwenTokenizerError.unsupportedPreTokenizer(actual: pre.description)
        }
    }

    /// Resolve a marker to its ID, rejecting `<unk>` substitution — the same
    /// round-trip `GFTokenizer.requireTokenID` does, for the same reason (BPE
    /// answers with the unknown ID, not `nil`). Ornith declares no unknown
    /// token at all, which makes a missing marker resolve to `nil` instead of
    /// colliding — but the checkpoint is not the only thing this can be
    /// pointed at.
    static func requireTokenID(_ tokenizer: any Tokenizer, _ token: String) throws -> Int32 {
        guard let id = tokenizer.convertTokenToId(token),
              tokenizer.convertIdToToken(id) == token else {
            throw QwenTokenizerError.missingSpecialToken(token)
        }
        guard let value = Int32(exactly: id) else {
            throw QwenTokenizerError.invalidTokenID(token: token, id: id)
        }
        return value
    }

    public init(tokenizer: any Tokenizer, tokenizerData: Config) throws {
        self.tokenizer = tokenizer
        try Self.verifyDecoderConfiguration(tokenizerData)
        try Self.verifyPreTokenizerConfiguration(tokenizerData)

        var specials: Set<Int32> = []
        var added: Set<Int32> = []
        var highestAddedID: Int32 = -1
        for token in tokenizerData["addedTokens"].array(or: []) {
            guard let rawID = token["id"].integer() else { continue }
            guard let id = Int32(exactly: rawID) else {
                throw QwenTokenizerError.invalidTokenID(
                    token: token.content.string() ?? "added token", id: rawID)
            }
            added.insert(id)
            if token["special"].boolean(or: false) { specials.insert(id) }
            highestAddedID = max(highestAddedID, id)
        }
        self.specialTokenIDs = specials
        self.addedTokenIDs = added

        self.imStartID = try Self.requireTokenID(tokenizer, "<|im_start|>")
        self.imEndID = try Self.requireTokenID(tokenizer, "<|im_end|>")
        self.endOfTextID = try Self.requireTokenID(tokenizer, "<|endoftext|>")
        self.thinkStartID = try Self.requireTokenID(tokenizer, "<think>")
        self.thinkEndID = try Self.requireTokenID(tokenizer, "</think>")
        self.toolCallStartID = try Self.requireTokenID(tokenizer, "<tool_call>")
        self.toolCallEndID = try Self.requireTokenID(tokenizer, "</tool_call>")
        self.toolResponseStartID = try Self.requireTokenID(tokenizer, "<tool_response>")
        self.toolResponseEndID = try Self.requireTokenID(tokenizer, "</tool_response>")
        self.stopTokenIDs = [self.imEndID, self.endOfTextID]
        self.vocabSize = Int(highestAddedID) + 1
    }

    // MARK: - Encode / decode

    /// Encode UTF-8 text to token IDs.
    ///
    /// No BOS: `add_bos_token` is false and `bos_token` is null in this
    /// checkpoint's `tokenizer_config.json` — the framing comes from the chat
    /// template, which opens with `<|im_start|>`.
    public func encode(_ text: String) -> [Int32] {
        tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    /// Decode token IDs to text. `skipSpecialTokens` drops the
    /// `added_tokens[special]` set, exactly as the library's own decode does
    /// before running its decoder chain — so a byte run **fuses across** a
    /// dropped marker, and `<think>` (which is not declared special) survives
    /// either way.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        var detok = QwenDetokenizer(tokenizer: self, skipSpecialTokens: skipSpecialTokens)
        var text = ""
        for id in ids { text += detok.push(id) }
        return text + detok.flush()
    }

    // MARK: - Vocabulary questions

    /// The token string an ID stands for — byte-level characters for an
    /// ordinary token, the literal content for an added one.
    public func token(for id: Int32) -> String? {
        tokenizer.convertIdToToken(Int(id))
    }

    /// An added token: the reference decoder emits these literally instead of
    /// byte-decoding them.
    public func isAddedToken(_ id: Int32) -> Bool { addedTokenIDs.contains(id) }

    /// `added_tokens[special == true]`: the set `decode(skipSpecialTokens:)`
    /// drops. A strict subset of the added tokens — `<think>` is added but not
    /// special, so it survives the filter.
    public func isSpecialToken(_ id: Int32) -> Bool { specialTokenIDs.contains(id) }

    public var addedTokenCount: Int { addedTokenIDs.count }

    /// A streaming detokenizer over this vocabulary, for generation loops.
    public func makeDetokenizer(skipSpecialTokens: Bool = true) -> QwenDetokenizer {
        QwenDetokenizer(tokenizer: self, skipSpecialTokens: skipSpecialTokens)
    }

    // MARK: - Chat template

    /// Render the checkpoint's own `chat_template.jinja` and encode it.
    ///
    /// Ornith ships the template, so — unlike Gemma, where the runtime owns the
    /// framing because upstream has none — nothing about the format is written
    /// out here. `enableThinking` is the template's own variable: false closes
    /// the reasoning block immediately (`<think>\n\n</think>\n\n`), true leaves
    /// it open after `<|im_start|>assistant\n<think>\n` for the model to
    /// continue into.
    public func applyChatTemplate(_ messages: [GFTokenizer.Message],
                                  tools: [GFTokenizer.FunctionDefinition] = [],
                                  enableThinking: Bool = true) throws -> [Int32] {
        guard tokenizer.hasChatTemplate else {
            throw QwenTokenizerError.missingChatTemplate
        }
        guard !messages.isEmpty else {
            throw QwenTokenizerError.invalidChatMessages("no messages")
        }
        let rendered: [Tokenizers.Message] = try messages.map { message in
            var value: Tokenizers.Message = ["role": message.role.rawValue]
            value["content"] = message.content ?? ""
            if !message.toolCalls.isEmpty {
                value["tool_calls"] = try message.toolCalls.map { call in
                    [
                        "id": call.id,
                        "type": "function",
                        "function": [
                            "name": call.name,
                            "arguments": try call.arguments.jinjaSendableValue(),
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                }
            }
            if let reasoning = message.reasoningContent, !reasoning.isEmpty {
                value["reasoning_content"] = reasoning
            }
            if let name = message.name { value["name"] = name }
            return value
        }
        let toolSpecs: [ToolSpec] = try tools.map { tool in
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
            messages: rendered,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: toolSpecs.isEmpty ? nil : toolSpecs,
            additionalContext: ["enable_thinking": enableThinking]
        ).map(Int32.init)
    }
}

/// Streaming detokenizer for Ornith. `QwenTokenizer.decode` is a push-loop over
/// this type, so batch and streaming decode agree by construction.
///
/// The two rules are `ByteLevelDecoding`'s: ordinary tokens are bytes and only
/// settle when a codepoint completes, and added tokens are literal text that
/// closes the byte run in front of them. A *dropped* special (skip mode) is not
/// a barrier — the library removes those IDs before decoding, so the runs on
/// either side of one are a single run.
public struct QwenDetokenizer {
    private let tokenizer: any Tokenizer
    private let skipSpecialTokens: Bool
    private let specialTokenIDs: Set<Int32>
    private let addedTokenIDs: Set<Int32>
    private var run = ByteLevelRun()

    public init(tokenizer: QwenTokenizer, skipSpecialTokens: Bool = true) {
        self.tokenizer = tokenizer.tokenizer
        self.skipSpecialTokens = skipSpecialTokens
        self.specialTokenIDs = tokenizer.specialTokenIDs
        self.addedTokenIDs = tokenizer.addedTokenIDs
    }

    /// Text contributed by `id`. `""` while a codepoint is still incomplete;
    /// those bytes come out with the token that completes them, at the next
    /// added token, or at `flush()`.
    public mutating func push(_ id: Int32) -> String {
        // An unknown ID contributes nothing and leaves the run open, matching
        // the library, whose decode compactMap-drops unresolvable IDs.
        guard let token = tokenizer.convertIdToToken(Int(id)) else { return "" }
        if skipSpecialTokens, specialTokenIDs.contains(id) { return "" }
        if addedTokenIDs.contains(id) { return run.commit() + token }
        return run.push(token)
    }

    /// Remainder held back at a stop boundary.
    public mutating func flush() -> String {
        run.commit()
    }
}
