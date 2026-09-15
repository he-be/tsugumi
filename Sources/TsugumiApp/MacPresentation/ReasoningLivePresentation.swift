import Foundation

/// How the live thought channel's text view catches up with the reasoning text.
///
/// The live view appends to its text storage instead of laying the text out
/// again. A SwiftUI `Text` measures and draws its whole string with CoreText
/// on every token; even cut to its last 1,500 characters, that took about
/// three quarters of the main thread two and a half minutes into a Qwen3.8
/// think, and the Stop button was handled 22 s after the click — after the
/// generation had ended. Appending costs what the token added.
public enum ReasoningLivePresentation {
    public enum Edit: Equatable, Sendable {
        case unchanged
        /// The shown text is a prefix of the new one: add the rest.
        case append(String)
        /// Anything else (a new turn, a switched chat): show this instead.
        case replace(String)
    }

    public static func edit(from shown: String, to text: String) -> Edit {
        // Scalars, not Characters: `"e\u{301}".hasPrefix("e")` is false, and a
        // token can end before the combining mark that follows it.
        let shownScalars = shown.unicodeScalars
        let scalars = text.unicodeScalars
        if scalars.elementsEqual(shownScalars) { return .unchanged }
        if scalars.starts(with: shownScalars) {
            return .append(String(scalars.dropFirst(shownScalars.count)))
        }
        return .replace(text)
    }
}
