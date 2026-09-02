import Foundation

/// One function call the model asked for. `argumentsJSON` is the argument
/// object as the model wrote it, kept as text so the redraw on the next
/// round shows the model exactly what it said.
public struct AppToolCall: Equatable, Sendable, Codable, Identifiable {
    public var id: String
    public var name: String
    public var argumentsJSON: String

    public init(id: String, name: String, argumentsJSON: String) {
        self.id = id
        self.name = name
        self.argumentsJSON = argumentsJSON
    }

    /// The argument object, or nil when the model wrote something the JSON
    /// parser rejects. The executor reports that back as a tool error rather
    /// than failing the turn.
    public var arguments: [String: Any]? {
        guard let data = argumentsJSON.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) else { return nil }
        return object as? [String: Any]
    }

    public func stringArgument(_ key: String) -> String? {
        guard let value = arguments?[key] else { return nil }
        if let string = value as? String { return string }
        if let number = value as? NSNumber { return number.stringValue }
        return nil
    }
}

/// One function the request declares to the model. `parametersJSON` is the
/// JSON Schema object for the arguments, as text.
public struct AppToolDefinition: Equatable, Sendable {
    public var name: String
    public var description: String
    public var parametersJSON: String

    public init(name: String, description: String, parametersJSON: String) {
        self.name = name
        self.description = description
        self.parametersJSON = parametersJSON
    }
}

/// The `tool_choice` shapes the app path uses. `required` forces a call
/// through the grammar; `function` forces a call of one named tool;
/// `none` hides the tools. Travels as a string (`rawValue`) to the decode
/// service.
public enum AppToolChoice: Equatable, Sendable, RawRepresentable {
    case auto
    case required
    case none
    case function(name: String)

    public var rawValue: String {
        switch self {
        case .auto: return "auto"
        case .required: return "required"
        case .none: return "none"
        case .function(let name): return "function:" + name
        }
    }

    public init?(rawValue: String) {
        switch rawValue {
        case "auto": self = .auto
        case "required": self = .required
        case "none": self = .none
        default:
            guard rawValue.hasPrefix("function:"), rawValue.count > "function:".count else { return nil }
            self = .function(name: String(rawValue.dropFirst("function:".count)))
        }
    }
}

/// What a tool returned to the model, plus a short label for the trace the
/// UI shows. `content` is what the tool turn carries; `isError` only marks
/// the trace entry — the model sees the message either way.
public struct AppToolResult: Equatable, Sendable {
    public var content: String
    public var isError: Bool
    /// One short line for the trace: where the result came from and how big
    /// it is ("Serper · 8 hits", "Jina Reader · 4,120 chars").
    public var summary: String

    public init(content: String, isError: Bool = false, summary: String = "") {
        self.content = content
        self.isError = isError
        self.summary = summary
    }
}

/// One step of a tool loop as the UI shows it: the call, and once it has
/// run, what came back.
public struct AppToolTraceEntry: Equatable, Sendable, Codable, Identifiable {
    public enum Status: String, Equatable, Sendable, Codable {
        case running
        case done
        case failed
    }

    public var id: String
    public var name: String
    /// The argument the step is about — the query, the URL.
    public var subject: String
    public var status: Status
    public var summary: String

    public init(id: String, name: String, subject: String,
                status: Status = .running, summary: String = "") {
        self.id = id
        self.name = name
        self.subject = subject
        self.status = status
        self.summary = summary
    }
}

/// What the system prompt needs to know about the declared tools: whether
/// the web ones are among them, and the date of the Wikipedia copy when the
/// local ones are (nil when they are not).
public struct AppToolPromptFacts: Equatable, Sendable {
    public var web: Bool
    public var wikipediaDate: String?

    public init(web: Bool, wikipediaDate: String?) {
        self.web = web
        self.wikipediaDate = wikipediaDate
    }
}

/// A tool round the app ran on its own before the model's first round: the
/// call as the transcript will show it and what it returned. The model
/// reads it like any other tool result (the template draws a tool turn by
/// its name; the name need not be declared).
public struct AppToolLookup: Equatable, Sendable {
    public var call: AppToolCall
    public var result: AppToolResult
    /// What the trace shows the step is about (the article names).
    public var subject: String

    public init(call: AppToolCall, result: AppToolResult, subject: String) {
        self.call = call
        self.result = result
        self.subject = subject
    }
}

/// Executes the tools the app declares. The model never runs anything; the
/// app decides what each name means and what its result text is.
public protocol AppToolExecutor: Sendable {
    /// The functions declared to the model for this executor.
    var definitions: [AppToolDefinition] { get }
    /// What the prompt says about these tools. Web by default.
    var promptFacts: AppToolPromptFacts { get }
    /// Runs one call. Never throws: a failure is a result the model can read
    /// and recover from (search another way, answer without the page).
    func execute(_ call: AppToolCall) async -> AppToolResult
    /// The one argument the trace shows for a call — the query, the URL.
    func subject(of call: AppToolCall) -> String
    /// Results worth handing the model before it decides anything — the
    /// pages the prompt links to, the Wikipedia openings of the things it
    /// names. Call ids are `callIDPrefix` plus a number from 1. Empty when
    /// there is nothing to add; then the first round starts as usual.
    func lookups(prompt: String, callIDPrefix: String) async -> [AppToolLookup]
}

extension AppToolExecutor {
    public func subject(of call: AppToolCall) -> String { call.argumentsJSON }
    public func lookups(prompt: String, callIDPrefix: String) async -> [AppToolLookup] { [] }
    public var promptFacts: AppToolPromptFacts { AppToolPromptFacts(web: true, wikipediaDate: nil) }
}
