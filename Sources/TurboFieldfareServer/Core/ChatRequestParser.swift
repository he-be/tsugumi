import Foundation
import TurboFieldfare

/// REQ-tool-choice's four shapes.
public enum ChatToolChoice: Equatable, Sendable {
    case auto
    case none
    case required
    case function(name: String)
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
        throw ServerRequestError.invalid(
            message: "ChatRequestParser is not implemented yet",
            param: nil,
            code: "not_implemented")
    }
}
