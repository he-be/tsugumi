import Foundation

public struct AppGenerationRequest: Equatable, Sendable {
    public var modelDirectory: URL
    /// Completed turns preceding `prompt`, oldest first. `prompt` and
    /// `imagePaths` stay the current user turn; history carries what earlier
    /// turns said, including assistant reasoning, for the exact redraw the
    /// prompt cache needs.
    public var history: [AppChatTurn]
    public var prompt: String
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
        for turn in history {
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
