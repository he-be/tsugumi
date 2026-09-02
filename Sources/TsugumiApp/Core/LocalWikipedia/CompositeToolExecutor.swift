import Foundation

/// Several executors declared together — the web tools next to the local
/// Wikipedia — dispatched by tool name. The model sees one list of
/// functions; each call goes to the executor that declared its name.
public struct CompositeToolExecutor: AppToolExecutor {
    public let executors: [any AppToolExecutor]

    public init(_ executors: [any AppToolExecutor]) {
        self.executors = executors
    }

    public var definitions: [AppToolDefinition] {
        executors.flatMap(\.definitions)
    }

    public var promptFacts: AppToolPromptFacts {
        executors.map(\.promptFacts).reduce(AppToolPromptFacts(web: false, wikipediaDate: nil)) { merged, facts in
            AppToolPromptFacts(web: merged.web || facts.web,
                               wikipediaDate: merged.wikipediaDate ?? facts.wikipediaDate)
        }
    }

    public func execute(_ call: AppToolCall) async -> AppToolResult {
        guard let executor = owner(of: call) else {
            return AppToolResult(content: "error: unknown tool \(call.name).",
                                 isError: true, summary: "unknown tool")
        }
        return await executor.execute(call)
    }

    public func subject(of call: AppToolCall) -> String {
        owner(of: call)?.subject(of: call) ?? call.argumentsJSON
    }

    /// Everything every executor has to say, in declaration order; each
    /// executor numbers its own calls under its own prefix.
    public func lookups(prompt: String, callIDPrefix: String) async -> [AppToolLookup] {
        var lookups: [AppToolLookup] = []
        for (index, executor) in executors.enumerated() {
            lookups += await executor.lookups(prompt: prompt, callIDPrefix: "\(callIDPrefix)\(index + 1)-")
        }
        return lookups
    }

    private func owner(of call: AppToolCall) -> (any AppToolExecutor)? {
        executors.first { executor in executor.definitions.contains { $0.name == call.name } }
    }
}
