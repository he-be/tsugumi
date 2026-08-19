import Foundation
import TurboFieldfare

/// One row of SPEC §4's request-parameter table.
///
/// The table is the schema: the reference implementation declares its request
/// fields the same way (`server-schema.cpp`), and keeping ours declarative is
/// what lets the two be diffed line by line — and what lets the C0 conformance
/// tests stand 1:1 with the rows (CONFORMANCE §1).
public struct ChatRequestField: Sendable, Equatable {
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

    public init(id: String,
                name: String,
                aliases: [String] = [],
                kinds: [ValueKind],
                rule: Rule,
                defaultValue: JSONValue? = nil) {
        self.id = id
        self.name = name
        self.aliases = aliases
        self.kinds = kinds
        self.rule = rule
        self.defaultValue = defaultValue
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
              kinds: [.array], rule: .passthrough),
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
              kinds: [.string, .array], rule: .passthrough),
        .init(id: "REQ-repeat-penalty", name: "repeat_penalty",
              kinds: [.number, .integer], rule: .passthrough,
              defaultValue: .number(1.0)),
        .init(id: "REQ-n", name: "n",
              kinds: [.integer], rule: .hard(lower: 1, upper: 1),
              defaultValue: .integer(1)),
        .init(id: "REQ-logprobs", name: "logprobs",
              kinds: [.boolean], rule: .passthrough),
        .init(id: "REQ-logprobs", name: "top_logprobs",
              kinds: [.integer], rule: .passthrough),
        .init(id: "REQ-tools", name: "tools",
              kinds: [.array], rule: .passthrough),
        .init(id: "REQ-tool-choice", name: "tool_choice",
              kinds: [.string, .object], rule: .passthrough,
              defaultValue: .string("auto")),
        .init(id: "REQ-parallel", name: "parallel_tool_calls",
              kinds: [.boolean], rule: .passthrough, defaultValue: .bool(true)),
        .init(id: "REQ-response-format", name: "response_format",
              kinds: [.object], rule: .passthrough),
        .init(id: "REQ-reasoning-effort", name: "reasoning_effort",
              kinds: [.string], rule: .passthrough),
        .init(id: "REQ-template-kwargs", name: "chat_template_kwargs",
              kinds: [.object], rule: .passthrough),
        .init(id: "REQ-cache-prompt", name: "cache_prompt",
              kinds: [.boolean], rule: .passthrough, defaultValue: .bool(true)),
        .init(id: "REQ-reasoning-budget", name: "reasoning_budget_tokens",
              kinds: [.integer], rule: .hard(lower: -1, upper: nil),
              defaultValue: .integer(-1)),
        .init(id: "REQ-reasoning-format", name: "reasoning_format",
              kinds: [.string], rule: .passthrough,
              defaultValue: .string("auto")),
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
    public static func normalize(_ body: JSONValue) throws -> NormalizedChatRequest {
        throw ServerRequestError.invalid(
            message: "ChatRequestSchema.normalize is not implemented yet",
            param: nil,
            code: "not_implemented")
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
