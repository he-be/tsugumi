import Foundation

public struct AppGenerationRequest: Equatable, Sendable {
    public var modelDirectory: URL
    /// Completed turns preceding `prompt`, oldest first. `prompt` and
    /// `imagePaths` stay the current user turn; history carries what earlier
    /// turns said, including assistant reasoning, for the exact redraw the
    /// prompt cache needs.
    public var history: [AppChatTurn]
    public var prompt: String
    /// A system message rendered before the history, or nil for none. The
    /// tool loop uses it to say what the tools are for and what day it is.
    public var systemPrompt: String?
    /// Turns that belong to the current user turn but come *after* `prompt`:
    /// the assistant's tool calls and their results, while a tool loop is
    /// still deciding the answer. Starts with an assistant turn; never holds
    /// a user turn.
    public var continuation: [AppChatTurn]
    /// Functions the model may call. Empty declares nothing and the request
    /// renders through the plain template exactly as before.
    public var tools: [AppToolDefinition]
    public var toolChoice: AppToolChoice
    /// Tokens the thought channel may spend before its closing tag is
    /// forced; -1 for no bound. The tool loop bounds the round that has no
    /// results to think about yet (`WebSearchConfiguration`).
    public var reasoningBudgetTokens: Int
    public var maxNewTokens: Int
    public var maxContextTokens: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var runtimeOptions: AppRuntimeOptions
    /// Whether the chat template renders the thought channel open. The
    /// default follows the model kind (`AppModelKind.thinkingDefault`); this
    /// carries what the toggle chose for one request.
    public var enableThinking: Bool
    /// Absolute paths of images attached to the prompt. Gemma only — the
    /// engine refuses them for a model with no vision tower.
    public var imagePaths: [String]

    public init(modelDirectory: URL,
                history: [AppChatTurn] = [],
                prompt: String,
                systemPrompt: String? = nil,
                continuation: [AppChatTurn] = [],
                tools: [AppToolDefinition] = [],
                toolChoice: AppToolChoice = .auto,
                reasoningBudgetTokens: Int = -1,
                maxNewTokens: Int = 4_096,
                maxContextTokens: Int = 4096,
                temperature: Float = 1.0,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                runtimeOptions: AppRuntimeOptions = AppRuntimeOptions(),
                enableThinking: Bool = false,
                imagePaths: [String] = []) {
        self.modelDirectory = modelDirectory
        self.history = history
        self.prompt = prompt
        self.systemPrompt = systemPrompt
        self.continuation = continuation
        self.tools = tools
        self.toolChoice = toolChoice
        self.reasoningBudgetTokens = reasoningBudgetTokens
        self.maxNewTokens = maxNewTokens
        self.maxContextTokens = maxContextTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.runtimeOptions = runtimeOptions
        self.enableThinking = enableThinking
        self.imagePaths = imagePaths
    }

    public var isPureGreedy: Bool {
        temperature == 0 && repetitionPenalty == 1
    }

    public func validate(fileManager: FileManager = .default,
                         requireModelDirectory: Bool = true) throws {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AppInferenceError.invalidRequest("Prompt cannot be empty.")
        }
        guard maxNewTokens > 0 else {
            throw AppInferenceError.invalidRequest("Max response length must be greater than zero.")
        }
        guard maxContextTokens > 0 else {
            throw AppInferenceError.invalidRequest("Max context must be greater than zero.")
        }
        guard temperature >= 0 else {
            throw AppInferenceError.invalidRequest("Temperature cannot be negative.")
        }
        if let topK {
            guard (1...256).contains(topK) else {
                throw AppInferenceError.invalidRequest("Top-K must be between 1 and 256.")
            }
        }
        if let topP {
            guard topP > 0, topP <= 1 else {
                throw AppInferenceError.invalidRequest("Top-P must be greater than 0 and at most 1.")
            }
            if temperature > 0, topP < 1, topK == nil {
                throw AppInferenceError.invalidRequest(
                    "Top-P below 1 requires Top-K to be enabled.")
            }
        }
        guard repetitionPenalty >= 1 else {
            throw AppInferenceError.invalidRequest("Repetition penalty must be at least 1.")
        }
        guard imagePaths.count <= 4 else {
            throw AppInferenceError.invalidRequest("At most 4 images can be attached.")
        }
        for path in imagePaths {
            guard fileManager.fileExists(atPath: path) else {
                throw AppInferenceError.invalidRequest("Attached image is missing: \(path)")
            }
        }
        if let first = history.first, first.role != .user {
            throw AppInferenceError.invalidRequest("History must begin with a user turn.")
        }
        for turn in history + continuation {
            guard turn.role == .user || turn.imagePaths.isEmpty else {
                throw AppInferenceError.invalidRequest("Images may only appear in user turns.")
            }
            guard turn.imagePaths.count <= 4 else {
                throw AppInferenceError.invalidRequest("At most 4 images can be attached.")
            }
            for path in turn.imagePaths {
                guard fileManager.fileExists(atPath: path) else {
                    throw AppInferenceError.invalidRequest("Attached image is missing: \(path)")
                }
            }
            guard turn.role == .assistant || turn.toolCalls.isEmpty else {
                throw AppInferenceError.invalidRequest("Only assistant turns may carry tool calls.")
            }
            if turn.role == .tool {
                guard let id = turn.toolCallID, !id.isEmpty else {
                    throw AppInferenceError.invalidRequest("A tool turn must name the call it answers.")
                }
            }
        }
        if let first = continuation.first, first.role != .assistant {
            throw AppInferenceError.invalidRequest("A continuation must begin with an assistant turn.")
        }
        guard !continuation.contains(where: { $0.role == .user }) else {
            throw AppInferenceError.invalidRequest("A continuation cannot hold a user turn.")
        }
        guard tools.isEmpty || tools.allSatisfy({ !$0.name.isEmpty }) else {
            throw AppInferenceError.invalidRequest("Every declared tool needs a name.")
        }
        if toolChoice == .required, tools.isEmpty {
            throw AppInferenceError.invalidRequest("Requiring a tool call needs at least one tool.")
        }
        if case .function(let name) = toolChoice, !tools.contains(where: { $0.name == name }) {
            throw AppInferenceError.invalidRequest("Forcing a call of \(name) needs that tool declared.")
        }
        try runtimeOptions.validate()

        if requireModelDirectory {
            var isDirectory = ObjCBool(false)
            guard fileManager.fileExists(atPath: modelDirectory.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                throw AppInferenceError.modelNotFound(modelDirectory.path)
            }
        }
    }
}
