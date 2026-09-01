import Foundation

public enum QwenToolCallParserError: Error, Equatable {
    case malformed
    case unknownTool(String)
    case oversized
    /// A parameter whose schema declares a non-string type, written in a form
    /// that is not the JSON the template would have produced for it.
    case unparsableArgument(String)
}

/// Ornith 1.5 (Qwen 3.5-MoE) tool calls — the XML form, not Gemma's
/// `call:name{…}`.
///
/// The shape is fixed by the checkpoint's own `chat_template.jinja`, which is
/// both what the system prompt shows the model and what an assistant turn is
/// re-rendered as:
///
/// ```
/// <tool_call>
/// <function=get_weather>
/// <parameter=city>
/// Kyoto
/// </parameter>
/// <parameter=days>
/// 3
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// **A value's spelling depends on its declared type, so parsing needs the
/// tool's schema.** The template writes
/// `args_value | string if args_value is string else args_value | tojson`:
/// a string parameter is written **raw** (no quotes, no escapes — `Kyoto`,
/// not `"Kyoto"`), and everything else is written as JSON. Without the schema
/// there is no way to tell the string `"3"` from the integer `3`, and no way
/// to tell the string `hello` from a syntax error. That is the one structural
/// difference from `GemmaToolCallParser`, which only ever needed the set of
/// allowed names.
///
/// The bytes between the markers are all this reads; `<tool_call>` and
/// `</tool_call>` are recognised by **token id** one level up
/// (`QwenStructuredAssistantDecoder`), because both markers are also spellable
/// as a run of ordinary tokens. They are tolerated here so that a caller
/// holding a whole call as text — a test, a log line — can hand it over
/// unedited.
public struct QwenToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    /// Declared name → that tool's `parameters` schema.
    private let schemas: [String: JSONValue]

    public init(tools: [GFTokenizer.FunctionDefinition]) {
        var table: [String: JSONValue] = [:]
        for tool in tools { table[tool.name] = tool.parameters }
        self.schemas = table
    }

    public func parse(_ text: String, id: String) throws -> ParsedToolCall {
        guard text.utf8.count <= Self.maximumBytes else {
            throw QwenToolCallParserError.oversized
        }
        var rest = Self.stripMarkers(Substring(text))
        // The template writes `<tool_call>\n<function=`; a model that omits
        // that newline, or adds another, has still written the same call.
        rest = rest.drop(while: \.isWhitespace)

        let name = try Self.take(&rest, opener: "<function=")
        guard let parameters = schemas[name] else {
            throw QwenToolCallParserError.unknownTool(name)
        }
        try Self.expectNewline(&rest)

        var arguments: [String: JSONValue] = [:]
        while true {
            if rest.hasPrefix("</function>") {
                rest = rest.dropFirst("</function>".count)
                break
            }
            let key = try Self.take(&rest, opener: "<parameter=")
            try Self.expectNewline(&rest)
            // The grammar forbids a value from containing the closer, so the
            // first occurrence is the only one. A parser that searched for the
            // *last* one would accept calls the grammar cannot produce and
            // then disagree with it about where the value ended.
            guard let closer = rest.range(of: "\n</parameter>") else {
                throw QwenToolCallParserError.malformed
            }
            let raw = String(rest[..<closer.lowerBound])
            rest = rest[closer.upperBound...]
            try Self.expectNewline(&rest)
            // A repeated parameter has no rendering — `arguments` is a
            // dictionary — so it cannot be round-tripped and is not accepted.
            guard arguments[key] == nil else { throw QwenToolCallParserError.malformed }
            arguments[key] = try Self.value(raw,
                                            schema: Self.property(key, of: parameters),
                                            key: key)
        }

        guard rest.drop(while: \.isWhitespace).isEmpty else {
            throw QwenToolCallParserError.malformed
        }
        let object = JSONValue.object(arguments)
        return ParsedToolCall(id: id,
                              name: name,
                              arguments: object,
                              argumentsJSON: try object.encoded())
    }

    // MARK: - Scanning

    /// Drop the section markers if the caller left them on. Only an exact
    /// pair is removed: half a pair is a malformed call, not a call with a
    /// stray marker in its body.
    private static func stripMarkers(_ text: Substring) -> Substring {
        let body = text.trimmingCharactersInSubstring(where: \.isWhitespace)
        guard body.hasPrefix("<tool_call>"), body.hasSuffix("</tool_call>") else {
            return text
        }
        return body.dropFirst("<tool_call>".count).dropLast("</tool_call>".count)
    }

    /// `<function=NAME>` / `<parameter=KEY>` — consume the opener and return
    /// the name up to `>`.
    private static func take(_ rest: inout Substring, opener: String) throws -> String {
        guard rest.hasPrefix(opener) else { throw QwenToolCallParserError.malformed }
        rest = rest.dropFirst(opener.count)
        guard let close = rest.firstIndex(of: ">") else {
            throw QwenToolCallParserError.malformed
        }
        let name = rest[..<close]
        // A newline inside the name means the `>` belongs to a later line and
        // this one was never closed.
        guard !name.isEmpty, !name.contains("\n") else {
            throw QwenToolCallParserError.malformed
        }
        rest = rest[rest.index(after: close)...]
        return String(name)
    }

    private static func expectNewline(_ rest: inout Substring) throws {
        guard rest.hasPrefix("\n") else { throw QwenToolCallParserError.malformed }
        rest = rest.dropFirst()
    }

    // MARK: - Typing a value

    /// The schema declared for one parameter, if the tool declared one at all.
    public static func property(_ key: String, of parameters: JSONValue) -> JSONValue? {
        guard case .object(let members) = parameters,
              case .object(let properties)? = members["properties"] else {
            return nil
        }
        return properties[key]
    }

    /// Whether the template would have written this parameter raw.
    ///
    /// `is string` in Jinja is a property of the *value*, not of the schema,
    /// but the value is exactly what is being recovered — so the schema is the
    /// only thing left to ask. A union that includes a non-string is not
    /// treated as a string: `tojson` would have quoted it.
    public static func isStringSchema(_ schema: JSONValue?) -> Bool {
        guard case .object(let members)? = schema else { return false }
        switch members["type"] {
        case .string("string"): return true
        case .array(let types):
            return !types.isEmpty && types.allSatisfy { $0 == .string("string") }
        default: return false
        }
    }

    /// Whether the schema names a type at all. When it does not — `{}`,
    /// `anyOf`, a parameter the tool never declared — a value that is not JSON
    /// is taken as the string it looks like rather than rejected.
    private static func declaresType(_ schema: JSONValue?) -> Bool {
        guard case .object(let members)? = schema else { return false }
        return members["type"] != nil
    }

    static func value(_ raw: String, schema: JSONValue?, key: String) throws -> JSONValue {
        if isStringSchema(schema) { return .string(raw) }
        if let decoded = decodeJSON(raw) { return decoded }
        guard !declaresType(schema) else {
            throw QwenToolCallParserError.unparsableArgument(key)
        }
        return .string(raw)
    }

    /// One `tojson` value. Fragments are the common case (`3`, `true`), so the
    /// decode has to allow them.
    static func decodeJSON(_ raw: String) -> JSONValue? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        // `JSONValue`'s own decoder, so the numeric cases (`integer` /
        // `unsignedInteger` / `decimal`) are picked the same way they are
        // everywhere else. It accepts top-level fragments, which is the common
        // case here — most non-string arguments are one number or one bool.
        return try? JSONDecoder().decode(JSONValue.self, from: Data(trimmed.utf8))
    }
}

private extension Substring {
    func trimmingCharactersInSubstring(where predicate: (Character) -> Bool) -> Substring {
        var result = self
        while let first = result.first, predicate(first) { result = result.dropFirst() }
        while let last = result.last, predicate(last) { result = result.dropLast() }
        return result
    }
}
