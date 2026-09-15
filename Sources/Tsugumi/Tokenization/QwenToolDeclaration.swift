import Foundation

/// One tool declaration as the Qwen templates print it: `{{- tool | tojson }}`.
///
/// The template's `tojson` is Hugging Face's — `json.dumps(tool, ensure_ascii=False)`: keys in the order the client
/// wrote them, `", "` between members and `": "` after a key, non-ASCII written as itself. swift-jinja's `tojson` is
/// `JSONEncoder` with `.sortedKeys`, no spaces, `\/` for every slash and — `ensure_ascii` defaulting to true — `\uXXXX`
/// for every non-ASCII character (`docs/qwen38/17` §3-1). A Japanese description came out as a run of escapes: the
/// app's four declarations were a different, longer token sequence from anything the checkpoint was trained on.
///
/// So the declaration is spelled here and the template prints the string (`QwenTokenizer.applyChatTemplate`).
public enum QwenToolDeclaration {
    /// The line the checkpoints' templates print a declaration with, and what it is replaced by.
    public static let templateLine = "{{- tool | tojson }}"
    public static let renderedKey = "tsugumi_json"
    public static let replacementLine = "{{- tool.\(renderedKey) }}"

    /// `{"type": "function", "function": {"name": …, "description": …, "parameters": …}}`, the OpenAI shape a client
    /// hands to `apply_chat_template`. The parameters keep `parametersSource`'s key order when it describes the same
    /// schema, and fall back to ascending keys (the order a `JSONValue` can still give) when there is no source.
    public static func json(_ tool: GFTokenizer.FunctionDefinition) -> String {
        let parameters = tool.parametersSource
            .flatMap { OrderedJSON.parse($0) }
            .flatMap { $0.jsonValue == tool.parameters ? $0 : nil }
            ?? OrderedJSON(tool.parameters)
        let declaration = OrderedJSON.object([
            ("type", .string("function")),
            ("function", .object([
                ("name", .string(tool.name)),
                ("description", .string(tool.description)),
                ("parameters", parameters),
            ])),
        ])
        return declaration.pythonDumps()
    }

    /// The order a Qwen tool call writes `tool`'s parameters in (SPEC §12 DEV-15): the required ones first, then the
    /// optional ones, each in the declaration's order when `parametersSource` spells the same schema and in ascending
    /// key order when there is no source. The same rule as llama.cpp's `json-schema-to-grammar`.
    ///
    /// Ascending keys alone (the Gemma rule, whose template writes arguments through `dictsort`) closed the call to an
    /// optional parameter that sorts before a required one: a model that wrote `url` first — the order the declaration
    /// shows — could not add `sections` (or `from`) after it, and fetched the same page again (`docs/qwen38/24` §3).
    /// The Qwen templates write `arguments|items`, the order the call has.
    public static func parameterOrder(_ tool: GFTokenizer.FunctionDefinition) -> [String] {
        guard case .object(let schema) = tool.parameters,
              case .object(let properties)? = schema["properties"] else { return [] }
        var declared = properties.keys.sorted()
        if let source = tool.parametersSource.flatMap(OrderedJSON.parse), source.jsonValue == tool.parameters,
           case .object(let members) = source,
           case .object(let ordered)? = members.first(where: { $0.0 == "properties" })?.1 {
            declared = ordered.map(\.0)
        }
        var required: Set<String> = []
        if case .array(let names)? = schema["required"] {
            for case .string(let name) in names { required.insert(name) }
        }
        return declared.filter { required.contains($0) } + declared.filter { !required.contains($0) }
    }
}

/// A JSON value that remembers the order of its object members.
public indirect enum OrderedJSON: Equatable, Sendable {
    case object([(String, OrderedJSON)])
    case array([OrderedJSON])
    case string(String)
    /// The literal as written (`12`, `-0.5`, `1e3`).
    case number(String)
    case bool(Bool)
    case null

    public static func == (lhs: OrderedJSON, rhs: OrderedJSON) -> Bool {
        switch (lhs, rhs) {
        case (.object(let a), .object(let b)):
            a.count == b.count && zip(a, b).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
        case (.array(let a), .array(let b)): a == b
        case (.string(let a), .string(let b)): a == b
        case (.number(let a), .number(let b)): a == b
        case (.bool(let a), .bool(let b)): a == b
        case (.null, .null): true
        default: false
        }
    }

    /// A `JSONValue` has no member order left; ascending keys is the one order it can give.
    public init(_ value: JSONValue) {
        switch value {
        case .object(let members):
            self = .object(members.keys.sorted().map { ($0, OrderedJSON(members[$0]!)) })
        case .array(let items): self = .array(items.map(OrderedJSON.init))
        case .string(let text): self = .string(text)
        case .integer(let number): self = .number(String(number))
        case .unsignedInteger(let number): self = .number(String(number))
        case .decimal(let number): self = .number(NSDecimalNumber(decimal: number).stringValue)
        case .number(let number): self = .number(Self.pythonFloat(number))
        case .bool(let flag): self = .bool(flag)
        case .null: self = .null
        }
    }

    /// The same value without the order, to check a source against the schema it is meant to spell.
    public var jsonValue: JSONValue? {
        switch self {
        case .object(let members):
            var dictionary: [String: JSONValue] = [:]
            for (key, value) in members {
                guard let converted = value.jsonValue else { return nil }
                dictionary[key] = converted
            }
            return .object(dictionary)
        case .array(let items):
            let converted = items.compactMap(\.jsonValue)
            return converted.count == items.count ? .array(converted) : nil
        case .string(let text): return .string(text)
        case .number(let literal):
            return try? JSONDecoder().decode(JSONValue.self, from: Data(literal.utf8))
        case .bool(let flag): return .bool(flag)
        case .null: return .null
        }
    }

    // MARK: - Python's json.dumps

    /// `json.dumps(value, ensure_ascii=False)`.
    public func pythonDumps() -> String {
        var out = ""
        write(into: &out)
        return out
    }

    private func write(into out: inout String) {
        switch self {
        case .object(let members):
            out += "{"
            for (index, member) in members.enumerated() {
                if index > 0 { out += ", " }
                Self.writeString(member.0, into: &out)
                out += ": "
                member.1.write(into: &out)
            }
            out += "}"
        case .array(let items):
            out += "["
            for (index, item) in items.enumerated() {
                if index > 0 { out += ", " }
                item.write(into: &out)
            }
            out += "]"
        case .string(let text):
            Self.writeString(text, into: &out)
        case .number(let literal):
            // Python re-prints what it parsed: an integer as written, a float as its repr.
            if literal.contains(where: { ".eE".contains($0) }), let value = Double(literal) {
                out += Self.pythonFloat(value)
            } else {
                out += literal
            }
        case .bool(let flag):
            out += flag ? "true" : "false"
        case .null:
            out += "null"
        }
    }

    /// `json.encoder.py_encode_basestring`: the two JSON metacharacters, the five named controls, every other
    /// control below 0x20 as `\u00XX`, and nothing else.
    private static func writeString(_ text: String, into out: inout String) {
        out += "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        out += "\""
    }

    /// `float.__repr__`: the shortest round-trip digits, `1.0` for an integral value, exponent form below 1e-4 or
    /// from 1e16 up.
    static func pythonFloat(_ value: Double) -> String {
        guard value.isFinite else { return value.isNaN ? "NaN" : (value < 0 ? "-Infinity" : "Infinity") }
        let magnitude = abs(value)
        if magnitude != 0, magnitude < 1e-4 || magnitude >= 1e16 {
            // Swift writes `1e-05` as `1e-05` and `1e+16` as `1e+16`, which is Python's spelling too.
            return "\(value)"
        }
        let text = "\(value)"
        return text.contains(".") || text.contains("e") ? text : text + ".0"
    }

    // MARK: - Parsing

    /// Parse JSON text keeping member order. Nil for anything that is not one complete JSON value.
    public static func parse(_ text: String) -> OrderedJSON? {
        var parser = Parser(scalars: Array(text.unicodeScalars))
        guard let value = parser.value() else { return nil }
        parser.skipWhitespace()
        return parser.index == parser.scalars.count ? value : nil
    }

    private struct Parser {
        let scalars: [Unicode.Scalar]
        var index = 0

        init(scalars: [Unicode.Scalar]) { self.scalars = scalars }

        mutating func skipWhitespace() {
            while index < scalars.count, [" ", "\n", "\r", "\t"].contains(scalars[index]) { index += 1 }
        }

        mutating func consume(_ literal: String) -> Bool {
            let expected = Array(literal.unicodeScalars)
            guard index + expected.count <= scalars.count,
                  Array(scalars[index..<index + expected.count]) == expected else { return false }
            index += expected.count
            return true
        }

        mutating func value() -> OrderedJSON? {
            skipWhitespace()
            guard index < scalars.count else { return nil }
            switch scalars[index] {
            case "{":
                index += 1
                var members: [(String, OrderedJSON)] = []
                skipWhitespace()
                if consume("}") { return .object(members) }
                while true {
                    skipWhitespace()
                    guard let key = string() else { return nil }
                    skipWhitespace()
                    guard consume(":"), let item = value() else { return nil }
                    members.append((key, item))
                    skipWhitespace()
                    if consume(",") { continue }
                    return consume("}") ? .object(members) : nil
                }
            case "[":
                index += 1
                var items: [OrderedJSON] = []
                skipWhitespace()
                if consume("]") { return .array(items) }
                while true {
                    guard let item = value() else { return nil }
                    items.append(item)
                    skipWhitespace()
                    if consume(",") { continue }
                    return consume("]") ? .array(items) : nil
                }
            case "\"":
                return string().map(OrderedJSON.string)
            case "t": return consume("true") ? .bool(true) : nil
            case "f": return consume("false") ? .bool(false) : nil
            case "n": return consume("null") ? .null : nil
            default:
                let start = index
                while index < scalars.count, "+-0123456789.eE".unicodeScalars.contains(scalars[index]) { index += 1 }
                guard index > start else { return nil }
                var literal = ""
                literal.unicodeScalars.append(contentsOf: scalars[start..<index])
                return Double(literal) == nil ? nil : .number(literal)
            }
        }

        mutating func string() -> String? {
            guard consume("\"") else { return nil }
            var out = String.UnicodeScalarView()
            while index < scalars.count {
                let scalar = scalars[index]
                index += 1
                switch scalar {
                case "\"":
                    return String(out)
                case "\\":
                    guard index < scalars.count else { return nil }
                    let escape = scalars[index]
                    index += 1
                    switch escape {
                    case "\"": out.append("\"")
                    case "\\": out.append("\\")
                    case "/": out.append("/")
                    case "b": out.append("\u{08}")
                    case "f": out.append("\u{0C}")
                    case "n": out.append("\n")
                    case "r": out.append("\r")
                    case "t": out.append("\t")
                    case "u":
                        guard let high = hex4() else { return nil }
                        if (0xD800..<0xDC00).contains(high) {
                            guard consume("\\u"), let low = hex4(), (0xDC00..<0xE000).contains(low),
                                  let combined = Unicode.Scalar(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
                            else { return nil }
                            out.append(combined)
                        } else {
                            guard let single = Unicode.Scalar(high) else { return nil }
                            out.append(single)
                        }
                    default:
                        return nil
                    }
                default:
                    out.append(scalar)
                }
            }
            return nil
        }

        mutating func hex4() -> UInt32? {
            guard index + 4 <= scalars.count else { return nil }
            var text = ""
            text.unicodeScalars.append(contentsOf: scalars[index..<index + 4])
            guard let value = UInt32(text, radix: 16) else { return nil }
            index += 4
            return value
        }
    }
}
