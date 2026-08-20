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
        []
    }

    /// EP-5 `/detokenize`, the inverse of the above — special tokens are
    /// rendered rather than dropped, or the round trip would not close.
    public func detokenize(_ tokens: [Int32]) -> String {
        ""
    }

    /// EP-5 `/apply-template`: the prompt string this conversation renders to.
    public func applyChatTemplate(_ request: ValidatedChatRequest) throws -> String {
        ""
    }
}
