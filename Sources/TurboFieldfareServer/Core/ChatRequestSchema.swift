import Foundation
import TurboFieldfare

/// One row of SPEC §4's request-parameter table.
///
/// The table is the schema: the reference implementation declares its request
/// fields the same way (`server-schema.cpp`), and keeping ours declarative is
/// what lets the two be diffed line by line — and what lets the C0 conformance
/// tests stand 1:1 with the rows (CONFORMANCE §1).
public struct ChatRequestField: Sendable {
    /// The JSON types this field accepts. Anything else is ERR-3: a 400 that
    /// names the field, never a bare "malformed JSON request".
    public enum ValueKind: String, Sendable, Equatable {
        case boolean, integer, number, string, array, object
    }

    /// What happens to a value outside the field's range (SPEC §4's legend).
    public enum Rule: Sendable, Equatable {
        /// Tuning parameter (R3): out of range is rounded to the nearest end.
        case clamp(lower: Double?, upper: Double?)
        /// Out of range is 400 with `param` set to this field.
        case hard(lower: Double?, upper: Double?)
        /// Used as given; the range is the type's.
        case passthrough
        /// Accepted and not honored (R3 + DEV-5). Listed so the deviation is
        /// visible here rather than hidden behind R1's unknown-key rule.
        case ignored
    }

    /// The SPEC ID this row implements, e.g. `REQ-temp`.
    public let id: String
    public let name: String
    /// Other spellings of the same field, in the order they are consulted.
    public let aliases: [String]
    public let kinds: [ValueKind]
    public let rule: Rule
    /// The effective default, which `/props.default_generation_settings`
    /// reports verbatim (EP-4). `nil` where the field has no default because
    /// it is required or has no standing value.
    public let defaultValue: JSONValue?
    /// A 400 naming this field when it is absent.
    public let isRequired: Bool
    /// The value-level rule a range cannot express — which shapes of
    /// `tool_choice` exist, which `response_format` types are implemented.
    /// The reference implementation carries the same escape hatch on its own
    /// field descriptors (`field::custom_handler`, `server-schema.h:36`).
    /// Returns the value to keep, or nil to drop it.
    public let handler: (@Sendable (JSONValue) throws -> JSONValue?)?

    public init(id: String,
                name: String,
                aliases: [String] = [],
                kinds: [ValueKind],
                rule: Rule,
                defaultValue: JSONValue? = nil,
                isRequired: Bool = false,
                handler: (@Sendable (JSONValue) throws -> JSONValue?)? = nil) {
        self.id = id
        self.name = name
        self.aliases = aliases
        self.kinds = kinds
        self.rule = rule
        self.defaultValue = defaultValue
        self.isRequired = isRequired
        self.handler = handler
    }

    /// Every spelling that maps to this row.
    public var spellings: [String] { [name] + aliases }
}

/// SPEC §4, transcribed. Nothing reads a request field that is not a row here.
public enum ChatRequestSchema {
    public static let fields: [ChatRequestField] = [
        .init(id: "REQ-model", name: "model",
              kinds: [.string], rule: .passthrough),
        .init(id: "REQ-messages", name: "messages",
              kinds: [.array], rule: .passthrough, isRequired: true,
              handler: { value in
                  guard case .array(let turns) = value, !turns.isEmpty else {
                      throw ServerRequestError.invalid(
                          message: "\"messages\" must be a non-empty array",
                          param: "messages", code: "invalid_messages")
                  }
                  return value
              }),
        .init(id: "REQ-stream", name: "stream",
              kinds: [.boolean], rule: .passthrough, defaultValue: .bool(false)),
        .init(id: "REQ-stream-usage", name: "stream_options",
              kinds: [.object], rule: .passthrough),
        .init(id: "REQ-max-tokens", name: "max_tokens",
              aliases: ["max_completion_tokens"],
              kinds: [.integer],
              rule: .hard(lower: -1, upper: nil),
              defaultValue: .integer(-1)),
        .init(id: "REQ-temp", name: "temperature",
              kinds: [.number, .integer],
              rule: .clamp(lower: 0, upper: nil),
              defaultValue: .number(1.0)),
        .init(id: "REQ-top-p", name: "top_p",
              kinds: [.number, .integer],
              rule: .clamp(lower: 0, upper: 1),
              defaultValue: .number(1.0)),
        // The upper end is the sampler's partial-sort limit, not OpenAI's or
        // the reference implementation's — a clamp, never a refusal (DEV-9).
        .init(id: "REQ-top-k", name: "top_k",
              kinds: [.integer],
              rule: .clamp(lower: 0, upper: Double(ChatRequestSchema.topKCeiling)),
              defaultValue: .integer(0)),
        .init(id: "REQ-seed", name: "seed",
              kinds: [.integer], rule: .passthrough, defaultValue: .integer(-1)),
        .init(id: "REQ-stop", name: "stop",
              kinds: [.string, .array], rule: .passthrough,
              defaultValue: .array([])),
        .init(id: "REQ-repeat-penalty", name: "repeat_penalty",
              kinds: [.number, .integer], rule: .passthrough,
              defaultValue: .number(1.0)),
        .init(id: "REQ-n", name: "n",
              kinds: [.integer], rule: .hard(lower: 1, upper: 1),
              defaultValue: .integer(1)),
        // DEV-6. A contract parameter, so an unimplemented one is an error and
        // never a silent drop (R4): a client that asked for probabilities must
        // not be handed a body without them and a 200.
        .init(id: "REQ-logprobs", name: "logprobs",
              kinds: [.boolean], rule: .passthrough,
              handler: { value in
                  guard value == .bool(true) else { return nil }
                  throw ServerRequestError.notSupported(
                      message: "logprobs are not implemented",
                      param: "logprobs", code: "logprobs_not_supported")
              }),
        .init(id: "REQ-logprobs", name: "top_logprobs",
              kinds: [.integer], rule: .passthrough,
              handler: { _ in
                  throw ServerRequestError.notSupported(
                      message: "top_logprobs are not implemented",
                      param: "top_logprobs", code: "logprobs_not_supported")
              }),
        .init(id: "REQ-tools", name: "tools",
              kinds: [.array], rule: .passthrough,
              defaultValue: .array([])),
        // GEN-4. All four shapes are the grammar's now, so this row only says
        // which four exist; a fifth is still a 400. What each one does to
        // generation — and the two refusals a grammar cannot express (a named
        // choice for an undeclared tool, `required` with no tools) — needs the
        // request's `tools` beside it, so it lives in `ChatRequestParser`.
        .init(id: "REQ-tool-choice", name: "tool_choice",
              kinds: [.string, .object], rule: .passthrough,
              defaultValue: .string("auto"),
              handler: { value in
                  switch value {
                  case .string("auto"), .string("none"), .string("required"):
                      return value
                  case .object(let choice):
                      guard choice["type"] == .string("function"),
                            case .object(let function)? = choice["function"],
                            case .string? = function["name"] else {
                          throw ServerRequestError.invalid(
                              message: "tool_choice must be \"auto\", \"none\", \"required\", "
                                  + "or {\"type\":\"function\",\"function\":{\"name\":…}}",
                              param: "tool_choice", code: "invalid_tool_choice")
                      }
                      return value
                  default:
                      throw ServerRequestError.invalid(
                          message: "tool_choice must be \"auto\", \"none\", \"required\", "
                              + "or {\"type\":\"function\",\"function\":{\"name\":…}}",
                          param: "tool_choice", code: "invalid_tool_choice")
                  }
              }),
        .init(id: "REQ-parallel", name: "parallel_tool_calls",
              kinds: [.boolean], rule: .passthrough, defaultValue: .bool(true)),
        // GEN-3. Structured output rides the same grammar machinery as tool
        // calls, and that machinery exists, so all three types pass. A fourth
        // type is still a 400 (the reference implementation refuses it at
        // `server-common.cpp:1156`), because being asked for JSON and answered
        // with prose at 200 is the one failure R4 forbids at every stage.
        // The schema inside is never inspected here: GEN-2 keeps schema
        // content out of the 400 business entirely.
        .init(id: "REQ-response-format", name: "response_format",
              kinds: [.object], rule: .passthrough,
              defaultValue: .object(["type": .string("text")]),
              handler: { value in
                  guard case .object(let format) = value else { return nil }
                  switch format["type"] {
                  case nil, .null, .string(""), .string("text"),
                       .string("json_object"), .string("json_schema"):
                      return value
                  default:
                      throw ServerRequestError.invalid(
                          message: "response_format.type must be \"text\", "
                              + "\"json_object\", or \"json_schema\"",
                          param: "response_format", code: "invalid_response_format")
                  }
              }),
        .init(id: "REQ-reasoning-effort", name: "reasoning_effort",
              kinds: [.string], rule: .passthrough),
        .init(id: "REQ-template-kwargs", name: "chat_template_kwargs",
              kinds: [.object], rule: .passthrough,
              defaultValue: .object([:])),
        .init(id: "REQ-cache-prompt", name: "cache_prompt",
              kinds: [.boolean], rule: .passthrough, defaultValue: .bool(true)),
        .init(id: "REQ-reasoning-budget", name: "reasoning_budget_tokens",
              kinds: [.integer], rule: .hard(lower: -1, upper: nil),
              defaultValue: .integer(-1)),
        .init(id: "REQ-reasoning-format", name: "reasoning_format",
              kinds: [.string], rule: .passthrough,
              defaultValue: .string("auto"),
              handler: { value in
                  guard case .string(let name) = value,
                        ReasoningFormat(rawValue: name) != nil else {
                      throw ServerRequestError.invalid(
                          message: "reasoning_format must be \"auto\" or \"none\"",
                          param: "reasoning_format", code: "invalid_reasoning_format")
                  }
                  return value
              }),
        .init(id: "REQ-timings", name: "timings_per_token",
              kinds: [.boolean], rule: .passthrough, defaultValue: .bool(false)),
    ] + ignoredNames.map {
        .init(id: "REQ-ignored", name: $0,
              kinds: ChatRequestField.ValueKind.allKinds, rule: .ignored)
    }

    /// DEV-9. The sampler's partial sort tops out here, so a larger `top_k`
    /// rounds down instead of being refused.
    public static let topKCeiling = 256

    /// DEV-5: sampling knobs the engine has no implementation for. Accepted so
    /// a client that always sends them is not refused, and listed so that
    /// "accepted and ignored" is a written-down deviation rather than a
    /// side effect of R1.
    public static let ignoredNames = [
        "min_p", "typical_p", "presence_penalty", "frequency_penalty",
        "repeat_last_n", "mirostat", "mirostat_tau", "mirostat_eta",
        "dry_multiplier", "dry_base", "dry_allowed_length",
        "dry_penalty_last_n", "dry_sequence_breakers",
        "xtc_probability", "xtc_threshold",
        "dynatemp_range", "dynatemp_exponent",
        "samplers", "logit_bias", "ignore_eos",
    ]

    public static func field(named name: String) -> ChatRequestField? {
        fields.first { $0.spellings.contains(name) }
    }

    /// Applies the table to one request body: unknown keys dropped (R1), nulls
    /// treated as absent (R2), types checked (ERR-3), ranges clamped or refused
    /// (R3), defaults filled in. What comes back is keyed by canonical name, so
    /// nothing downstream has to know an alias exists.
    ///
    /// The body is walked field by field and never key by key, which is what
    /// makes R1 true by construction rather than by a list of keys to forgive
    /// (`server-schema.cpp:545`).
    public static func normalize(_ body: JSONValue) throws -> NormalizedChatRequest {
        guard case .object(let raw) = body else {
            throw ServerRequestError.invalid(
                message: "the request body must be a JSON object",
                param: nil, code: "invalid_body")
        }
        var values: [String: JSONValue] = [:]
        for field in fields {
            guard let sent = field.sentValue(in: raw) else {
                if field.isRequired {
                    throw ServerRequestError.invalid(
                        message: "\"\(field.name)\" is required",
                        param: field.name, code: "missing_field")
                }
                if let defaultValue = field.defaultValue {
                    values[field.name] = defaultValue
                }
                continue
            }
            if case .ignored = field.rule { continue }
            let typed = try field.coerced(sent.value, spelling: sent.spelling)
            let bounded = try field.bounded(typed, spelling: sent.spelling)
            if let handler = field.handler {
                if let kept = try handler(bounded) { values[field.name] = kept }
            } else {
                values[field.name] = bounded
            }
        }
        return NormalizedChatRequest(values: values)
    }
}

extension ChatRequestField {
    /// The first spelling the request actually used. A null is not a use
    /// (R2, `server-schema.cpp:581`'s `has_value`).
    func sentValue(in body: [String: JSONValue]) -> (spelling: String, value: JSONValue)? {
        for spelling in spellings {
            guard let value = body[spelling], value != .null else { continue }
            return (spelling, value)
        }
        return nil
    }

    /// ERR-3: a wrong type is a 400 that says which field and what was wanted.
    func coerced(_ value: JSONValue, spelling: String) throws -> JSONValue {
        let wantsNumber = kinds.contains(.number)
        let wantsInteger = kinds.contains(.integer)
        switch value {
        case .bool where kinds.contains(.boolean),
             .string where kinds.contains(.string),
             .array where kinds.contains(.array),
             .object where kinds.contains(.object):
            return value
        case .integer, .unsignedInteger:
            guard wantsInteger || wantsNumber else { break }
            guard let exact = value.exactDouble else { break }
            return wantsNumber && !wantsInteger ? .number(exact) : value
        case .decimal, .number:
            guard wantsNumber || wantsInteger else { break }
            guard let exact = value.exactDouble else { break }
            // An integer field truncates a fractional value rather than
            // refusing it, as the reference implementation's `get<int32_t>()`
            // does — a client that sends 3.0 for `top_k` meant 3.
            return wantsNumber ? .number(exact) : .integer(Int64(exact))
        default:
            break
        }
        throw ServerRequestError.invalid(
            message: "\"\(spelling)\" must be \(Self.expectation(kinds)), "
                + "but the request sent \(value.kindName)",
            param: name,
            code: "invalid_type")
    }

    /// R3: clamp, or refuse and name the field.
    func bounded(_ value: JSONValue, spelling: String) throws -> JSONValue {
        let lower: Double?
        let upper: Double?
        let isHard: Bool
        switch rule {
        case .clamp(let low, let high):
            (lower, upper, isHard) = (low, high, false)
        case .hard(let low, let high):
            (lower, upper, isHard) = (low, high, true)
        case .passthrough, .ignored:
            return value
        }
        guard let number = value.exactDouble else { return value }
        let clamped = min(max(number, lower ?? -.infinity), upper ?? .infinity)
        guard clamped != number else { return value }
        guard !isHard else {
            throw ServerRequestError.invalid(
                message: "\"\(spelling)\" must be \(Self.rangeText(lower, upper)), "
                    + "but the request sent \(Self.numberText(number))",
                param: name,
                code: "out_of_range")
        }
        switch value {
        case .integer, .unsignedInteger: return .integer(Int64(clamped))
        default: return .number(clamped)
        }
    }

    private static func expectation(_ kinds: [ValueKind]) -> String {
        let names = kinds.map { kind -> String in
            switch kind {
            case .boolean: "a boolean"
            case .integer: "an integer"
            case .number: "a number"
            case .string: "a string"
            case .array: "an array"
            case .object: "an object"
            }
        }
        guard let last = names.last, names.count > 1 else { return names.first ?? "a value" }
        return names.dropLast().joined(separator: ", ") + " or " + last
    }

    private static func rangeText(_ lower: Double?, _ upper: Double?) -> String {
        switch (lower, upper) {
        case (let low?, let high?) where low == high: "exactly \(numberText(low))"
        case (let low?, let high?): "between \(numberText(low)) and \(numberText(high))"
        case (let low?, nil): "at least \(numberText(low))"
        case (nil, let high?): "at most \(numberText(high))"
        case (nil, nil): "any value"
        }
    }

    private static func numberText(_ value: Double) -> String {
        value == value.rounded() && abs(value) < 1e15
            ? String(Int64(value))
            : String(value)
    }
}

extension JSONValue {
    /// The value as a Double when it is a number at all, and nil otherwise.
    var exactDouble: Double? {
        switch self {
        case .integer(let value): Double(value)
        case .unsignedInteger(let value): Double(value)
        case .decimal(let value): Double(truncating: NSDecimalNumber(decimal: value))
        case .number(let value): value.isFinite ? value : nil
        default: nil
        }
    }

    /// What to call this value in an ERR-3 message.
    var kindName: String {
        switch self {
        case .object: "an object"
        case .array: "an array"
        case .string: "a string"
        case .integer, .unsignedInteger: "an integer"
        case .decimal, .number: "a number"
        case .bool: "a boolean"
        case .null: "null"
        }
    }
}

/// A request body after the table has been applied to it.
public struct NormalizedChatRequest: Equatable, Sendable {
    /// Canonical field name to value. Every row of the table with a default is
    /// present; a row without one is present only when the request sent it.
    public let values: [String: JSONValue]

    public init(values: [String: JSONValue]) {
        self.values = values
    }

    public subscript(name: String) -> JSONValue? { values[name] }

    public func bool(_ name: String) -> Bool? {
        guard case .bool(let value) = values[name] else { return nil }
        return value
    }

    public func string(_ name: String) -> String? {
        guard case .string(let value) = values[name] else { return nil }
        return value
    }

    public func int(_ name: String) -> Int? {
        switch values[name] {
        case .integer(let value): Int(exactly: value)
        case .unsignedInteger(let value): Int(exactly: value)
        default: nil
        }
    }

    public func int64(_ name: String) -> Int64? {
        switch values[name] {
        case .integer(let value): value
        case .unsignedInteger(let value): Int64(exactly: value)
        default: nil
        }
    }

    public func double(_ name: String) -> Double? {
        switch values[name] {
        case .integer(let value): Double(value)
        case .unsignedInteger(let value): Double(value)
        case .decimal(let value): Double(truncating: NSDecimalNumber(decimal: value))
        case .number(let value): value
        default: nil
        }
    }

    public func array(_ name: String) -> [JSONValue]? {
        guard case .array(let value) = values[name] else { return nil }
        return value
    }

    public func object(_ name: String) -> [String: JSONValue]? {
        guard case .object(let value) = values[name] else { return nil }
        return value
    }
}

extension ChatRequestField.ValueKind {
    /// Every kind, for rows that do not constrain the type because they do
    /// not read the value at all.
    static let allKinds: [Self] = [.boolean, .integer, .number, .string, .array, .object]
}
