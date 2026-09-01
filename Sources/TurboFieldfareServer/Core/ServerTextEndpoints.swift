import Foundation
import TurboFieldfare

/// SPEC §3 **EP-5**: `/tokenize`, `/detokenize` and `/apply-template`.
///
/// All three need the tokenizer and nothing else — no weights, no Metal — so
/// they live here rather than inside the session that owns the model, and the
/// tests that hold them are C2 (CONFORMANCE §1).
///
/// `/apply-template` renders with **`ServerPromptRenderer.variant`** (SPEC §12
/// **DEV-12**), which is the repo-owned variant this server actually prefills
/// with. Answering with the checkpoint's bundled template would describe a
/// prompt this server never builds, and a client that took the answer and sent
/// it back would get a different token sequence than the one it was told about.
public struct ServerTextEndpoints: Sendable {
    public let tokenizer: GFTokenizer

    public init(tokenizer: GFTokenizer) {
        self.tokenizer = tokenizer
    }

    /// EP-5 `/tokenize`. `addSpecial` is the reference's `add_special`: whether
    /// `<bos>` is prepended.
    public func tokenize(_ text: String, addSpecial: Bool) -> [Int32] {
        tokenizer.encode(text, addBOS: addSpecial)
    }

    /// EP-5 `/detokenize`, the inverse of the above — special tokens are
    /// rendered rather than dropped, or the round trip would not close.
    public func detokenize(_ tokens: [Int32]) -> String {
        tokenizer.decode(tokens, skipSpecialTokens: false)
    }

    /// EP-5 `/apply-template`: the prompt string this conversation renders to.
    ///
    /// The three template paths are the three the session prefills through, and
    /// picked by the same question (`ServerPromptRenderer.usesToolTemplate`), so
    /// the answer is about the prompt this request would actually get.
    ///
    /// The tool path has no string form of its own — `encodeToolChat` goes
    /// straight to ids because that is all a prefill ever wanted — so its text
    /// is those ids read back. The tokenizer's decode is the inverse of its
    /// encode by construction (`GFDetokenizer`), which is what the DEV-12 test
    /// checks by re-encoding the answer.
    public func applyChatTemplate(_ request: ValidatedChatRequest) throws -> String {
        if ServerPromptRenderer.usesToolTemplate(request) {
            return detokenize(try tokenizer.encodeToolChat(
                messages: request.toolChatMessages,
                tools: request.tools,
                enableThinking: request.enableThinking,
                variant: ServerPromptRenderer.variant))
        }
        if let vision = request.vision {
            // The picture is a marker here and not a span of soft tokens: what
            // widens it is the prefill assembler, and this endpoint runs no
            // prefill.
            return try tokenizer.applyChatTemplate(
                multimodal: vision.messages,
                enableThinking: request.enableThinking,
                variant: ServerPromptRenderer.variant)
        }
        return try tokenizer.applyChatTemplate(
            request.messages,
            enableThinking: request.enableThinking,
            variant: ServerPromptRenderer.variant)
    }
}
