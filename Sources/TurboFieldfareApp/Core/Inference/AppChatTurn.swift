import Foundation

/// One completed turn of the app's conversation, as the next request resends
/// it. An assistant turn keeps its `reasoningText` so the chat template can
/// redraw the turn exactly as the model wrote it (SPEC MSG-5 / INV-1); that
/// exact redraw is what lets the prompt cache extend across turns with
/// thinking on or off.
public struct AppChatTurn: Equatable, Sendable {
    public enum Role: String, Equatable, Sendable {
        case user
        case assistant
    }

    public var role: Role
    public var text: String
    /// The thought channel of an assistant turn; empty when none was produced.
    public var reasoningText: String
    /// Absolute paths of images attached to a user turn. Gemma only.
    public var imagePaths: [String]

    public init(role: Role,
                text: String,
                reasoningText: String = "",
                imagePaths: [String] = []) {
        self.role = role
        self.text = text
        self.reasoningText = reasoningText
        self.imagePaths = imagePaths
    }
}
