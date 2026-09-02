import Foundation

/// One completed turn of the app's conversation, as the next request resends
/// it. An assistant turn keeps its `reasoningText` so the chat template can
/// redraw the turn exactly as the model wrote it (SPEC MSG-5 / INV-1); that
/// exact redraw is what lets the prompt cache extend across turns with
/// thinking on or off.
///
/// A tool loop adds two shapes: an assistant turn whose `toolCalls` is not
/// empty (the model asked for the calls, its `text` is whatever it said
/// alongside), and a `tool` turn carrying one call's result, named by
/// `toolCallID`. The template resolves the function name from the call.
public struct AppChatTurn: Equatable, Sendable, Codable {
    public enum Role: String, Equatable, Sendable, Codable {
        case user
        case assistant
        case tool
    }

    public var role: Role
    public var text: String
    /// The thought channel of an assistant turn; empty when none was produced.
    public var reasoningText: String
    /// Absolute paths of images attached to a user turn. Gemma only.
    public var imagePaths: [String]
    /// The function calls an assistant turn asked for.
    public var toolCalls: [AppToolCall]
    /// On a `tool` turn: which call this result answers.
    public var toolCallID: String?
    /// On a `tool` turn: the function name, for display and as the template's
    /// fallback when the call id does not resolve.
    public var toolName: String?

    public init(role: Role,
                text: String,
                reasoningText: String = "",
                imagePaths: [String] = [],
                toolCalls: [AppToolCall] = [],
                toolCallID: String? = nil,
                toolName: String? = nil) {
        self.role = role
        self.text = text
        self.reasoningText = reasoningText
        self.imagePaths = imagePaths
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.toolName = toolName
    }

    /// The result of one tool call, as the model will read it back.
    public static func toolResult(callID: String, name: String, content: String) -> AppChatTurn {
        AppChatTurn(role: .tool, text: content, toolCallID: callID, toolName: name)
    }

    private enum CodingKeys: String, CodingKey {
        case role, text, reasoningText, imagePaths, toolCalls, toolCallID, toolName
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(Role.self, forKey: .role)
        text = try container.decode(String.self, forKey: .text)
        reasoningText = try container.decodeIfPresent(String.self, forKey: .reasoningText) ?? ""
        imagePaths = try container.decodeIfPresent([String].self, forKey: .imagePaths) ?? []
        toolCalls = try container.decodeIfPresent([AppToolCall].self, forKey: .toolCalls) ?? []
        toolCallID = try container.decodeIfPresent(String.self, forKey: .toolCallID)
        toolName = try container.decodeIfPresent(String.self, forKey: .toolName)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        try container.encode(text, forKey: .text)
        try container.encode(reasoningText, forKey: .reasoningText)
        try container.encode(imagePaths, forKey: .imagePaths)
        if !toolCalls.isEmpty { try container.encode(toolCalls, forKey: .toolCalls) }
        try container.encodeIfPresent(toolCallID, forKey: .toolCallID)
        try container.encodeIfPresent(toolName, forKey: .toolName)
    }
}
