import Foundation
import Tsugumi

/// Turns a validated request into the token sequence the model is prefilled
/// with, for every path that needs nothing but the tokenizer.
///
/// Split out of `ServerModelSession` so INV-1 can be checked without weights or
/// Metal: the invariant is a statement about two token sequences, and the C2
/// tests that hold it are the reason this is its own type (CONFORMANCE §1).
/// The image path stays in the session, which owns the vision tower.
public struct ServerPromptRenderer: Sendable {
    /// SPEC INV-1: the server draws a finished assistant turn the way the
    /// model generated it, which the checkpoint's own template does not do.
    /// Everything the server prefills goes through this one variant, so the
    /// two paths below cannot drift apart.
    public static let variant = GFTokenizer.ChatTemplateVariant.serverRedraw

    public let tokenizer: GFTokenizer

    public init(tokenizer: GFTokenizer) {
        self.tokenizer = tokenizer
    }

    /// Whether this request has to go through the tool-calling template.
    public static func usesToolTemplate(_ request: ValidatedChatRequest) -> Bool {
        !request.tools.isEmpty || request.messages.contains {
            $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
        }
    }

    /// The prompt for a request that carries no image.
    public func promptIDs(_ request: ValidatedChatRequest) throws -> [Int32] {
        if Self.usesToolTemplate(request) {
            return try tokenizer.encodeToolChat(
                messages: request.messages,
                tools: request.tools,
                enableThinking: request.enableThinking,
                variant: Self.variant)
        }
        let rendered = try tokenizer.applyChatTemplate(
            request.messages,
            enableThinking: request.enableThinking,
            variant: Self.variant)
        return tokenizer.encode(rendered, addBOS: false)
    }
}
