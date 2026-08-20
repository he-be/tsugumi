import TurboFieldfare

/// One tool's `parameters` after it has been adapted to the **tool
/// declaration** the chat template renders into the prompt, plus what had to
/// be given up to get there (GEN-2 / DEV-16).
///
/// `simplifications` is the same idea as `JSONSchemaGrammarResult`'s
/// `approximations`, and reads in the same vocabulary — `<kind>: <path>` — but
/// the two are separate lists on purpose, because they describe two different
/// things that now degrade independently. See the note on `GemmaToolSchema`.
struct GemmaToolSchemaResult: Equatable, Sendable {
    let schema: JSONValue
    let simplifications: [String]
}

/// Adapts a tool's `parameters` to what the **tool declaration** can say.
///
/// **This is not the grammar.** Two different things read a tool's schema and
/// they fail in different ways, so they are simplified separately:
///
/// - *This type* feeds the declaration the chat template renders into the
///   prompt (`GFTokenizer.encodeToolChat` → `tool.parameters`). It has to be
///   something the template can write and Jinja can carry, which is why a
///   `type` array collapses to `type` + `nullable`, why a keyword the template
///   has no spelling for is dropped, and why a number that would make the
///   redraw throw cannot survive. What the model reads is a *description* of
///   the arguments.
/// - `JSONSchemaGrammar` feeds the GBNF that constrains generation. It has to
///   be something the matcher can accept token by token, and it approximates
///   different things (a broken `pattern`, an unresolvable `$ref`, a key the
///   dialect cannot write).
///
/// A schema can be perfectly renderable and badly constrainable, or the other
/// way round. Neither is ever a 400: GEN-2 says schema *content* is not a
/// reason to refuse a request, because a client's schema is usually generated
/// and one unrenderable line at its edge would otherwise take down the whole
/// task (DEV-16). Everything given up here comes back in `simplifications`.
enum GemmaToolSchema {
    private static let types: Set<String> = [
        "array", "boolean", "integer", "null", "number", "object", "string",
    ]
    private static let annotations: Set<String> = [
        "$comment", "default", "deprecated", "description", "examples", "readOnly",
        "title", "writeOnly",
    ]

    /// GEN-2's entry point: never throws for schema content.
    static func adapted(_ schema: JSONValue, toolName: String) -> GemmaToolSchemaResult {
        let root = "tools.\(toolName).parameters"
        var adapter = Adapter()
        var object = adapter.adapt(schema, path: root).objectValue ?? [:]
        // The declaration's `parameters` is an argument list, so the only
        // shape the template can render it as is a non-null object. A schema
        // that says otherwise is moved to that shape; a schema that simply did
        // not say (`{}`, or properties without a `type`) is not contradicted
        // by it, so nothing is recorded for those.
        if let declared = object["type"], declared != .string("object") {
            adapter.note("unrepresentable-parameters", root)
        } else if object["nullable"] == .bool(true) {
            adapter.note("unrepresentable-parameters", root)
        }
        object["type"] = .string("object")
        object["nullable"] = nil
        // Last: a number the redraw cannot carry would make the *next* turn's
        // template render throw (`jinjaSendableValue`), so it cannot stay in
        // the declaration at all.
        let sanitized = adapter.sanitized(.object(object), path: root)
        return GemmaToolSchemaResult(
            schema: sanitized ?? .object(["type": .string("object")]),
            simplifications: adapter.simplifications)
    }

    /// The walk itself. Kept as a value with the record on it so that every
    /// drop has exactly one place to be written down.
    private struct Adapter {
        private(set) var simplifications: [String] = []

        mutating func note(_ kind: String, _ path: String) {
            let message = "\(kind): \(path)"
            // A branch folded into its parent is visited twice; say it once.
            guard !simplifications.contains(message) else { return }
            simplifications.append(message)
        }

        mutating func adapt(_ schema: JSONValue, path: String) -> JSONValue {
            guard case .object(var object) = schema else {
                // A boolean (or anything else) where a schema belongs. The
                // empty schema is the closest renderable thing: it names the
                // member without claiming anything about it.
                note("unrepresentable-schema", path)
                return .object([:])
            }
            if object["allOf"] != nil {
                // The template writes one schema per member, not an
                // intersection of them. Dropping the keyword keeps whatever
                // stood beside it rather than losing the member.
                note("unrepresentable-all-of", path)
                object["allOf"] = nil
            }
            if object["anyOf"] != nil || object["oneOf"] != nil {
                object = adaptUnion(object, path: path)
            }

            switch object["type"] {
            case .string(let name):
                if !types.contains(name) {
                    note("unknown-type", "\(path) (\(name))")
                    object["type"] = nil
                }
            case .array(let values):
                adaptTypeArray(values, object: &object, path: path)
            case nil:
                // An untyped schema is legal JSON Schema — it means "any" —
                // and the template renders it as it stands. Nothing lost.
                break
            default:
                note("unrepresentable-type", path)
                object["type"] = nil
            }
            let type: String? = {
                if case .string(let name)? = object["type"] { return name }
                return nil
            }()

            if let properties = object["properties"] {
                if type == nil || type == "object", case .object(let definitions) = properties {
                    adaptProperties(definitions, object: &object, path: path)
                } else {
                    // `properties` under a non-object type describes members
                    // that can never appear.
                    note("unrepresentable-properties", path)
                    object["properties"] = nil
                }
            }

            if let items = object["items"] {
                if type == nil || type == "array", case .object = items {
                    object["items"] = adapt(items, path: "\(path).items")
                } else {
                    // A tuple (`items: [...]`), a boolean, or `items` under a
                    // non-array type. Without the keyword the array simply
                    // holds anything.
                    note("unrepresentable-items", "\(path).items")
                    object["items"] = nil
                }
            }
            return .object(object)
        }

        /// DEV-15 orders the keys; GEN-2 drops the ones the tool-call dialect
        /// cannot write. A dropped property leaves `required` too — the model
        /// is never asked for a key whose value `GemmaToolCallParser` could
        /// not read back off the wire.
        private mutating func adaptProperties(_ definitions: [String: JSONValue],
                                              object: inout [String: JSONValue],
                                              path: String) {
            var adapted: [String: JSONValue] = [:]
            adapted.reserveCapacity(definitions.count)
            var dropped: Set<String> = []
            for key in definitions.keys.sorted() {
                guard GemmaToolCallParser.isRepresentableObjectKey(key) else {
                    note("unrepresentable-key", "\(path).properties.\(key)")
                    dropped.insert(key)
                    continue
                }
                adapted[key] = adapt(definitions[key]!, path: "\(path).properties.\(key)")
            }
            object["properties"] = .object(adapted)
            guard !dropped.isEmpty, case .array(let required)? = object["required"] else { return }
            object["required"] = .array(required.filter { entry in
                guard case .string(let name) = entry else { return true }
                return !dropped.contains(name)
            })
        }

        /// A `type` array. The template writes one `type`, so the only union
        /// it can spell is "one concrete type, optionally null" — which it
        /// writes as `type` + `nullable`.
        private mutating func adaptTypeArray(_ values: [JSONValue],
                                             object: inout [String: JSONValue],
                                             path: String) {
            var names: [String] = []
            var unrepresentable = false
            for value in values {
                guard case .string(let name) = value else {
                    unrepresentable = true
                    continue
                }
                guard types.contains(name) else {
                    note("unknown-type", "\(path) (\(name))")
                    unrepresentable = true
                    continue
                }
                names.append(name)
            }
            let unique = Set(names)
            let concrete = unique.subtracting(["null"])
            guard !unrepresentable, concrete.count <= 1, !unique.isEmpty else {
                // Two or more concrete types: no spelling. The member keeps
                // its other keywords and loses only the type.
                note("unrepresentable-type-union", path)
                object["type"] = nil
                return
            }
            guard let type = concrete.first else {
                object["type"] = .string("null")
                return
            }
            object["type"] = .string(type)
            guard unique.contains("null") else { return }
            if let nullable = object["nullable"], nullable != .bool(true) {
                // The two spellings disagree; the union is the one JSON Schema
                // gives a meaning to, so it wins and the disagreement is
                // recorded rather than refused.
                note("nullable-conflict", path)
            }
            object["nullable"] = .bool(true)
        }

        // MARK: - anyOf / oneOf

        private mutating func adaptUnion(_ source: [String: JSONValue],
                                         path: String) -> [String: JSONValue] {
            if let result = adaptStringConstantUnion(source, path: path) { return result }
            if let result = adaptNullableUnion(source, path: path) { return result }
            // Nothing the declaration can spell. Dropping the branches leaves
            // the member unconstrained instead of losing it.
            note("unrepresentable-union", path)
            var result = source
            result["anyOf"] = nil
            result["oneOf"] = nil
            return result
        }

        /// A union of string constants is an `enum`, which the template does
        /// write.
        private mutating func adaptStringConstantUnion(
            _ source: [String: JSONValue],
            path: String
        ) -> [String: JSONValue]? {
            guard !(source["anyOf"] != nil && source["oneOf"] != nil) else { return nil }
            let keyword = source["anyOf"] != nil ? "anyOf" : "oneOf"
            guard case .array(let branches)? = source[keyword], branches.count >= 2 else {
                return nil
            }

            var values: [JSONValue] = []
            var overlaps = false
            for branch in branches {
                guard case .object(let object) = branch,
                      Set(object.keys).isSubset(of: ["const", "type"]),
                      object["type"] == .string("string"),
                      case .string = object["const"] else {
                    return nil
                }
                let value = object["const"]!
                if values.contains(value) {
                    // A `oneOf` whose branches overlap matches nothing; as an
                    // `enum` it matches the value once. That is a changed
                    // meaning, so it is recorded.
                    overlaps = overlaps || keyword == "oneOf"
                } else {
                    values.append(value)
                }
            }

            let allowedSiblings = annotations.union([keyword])
            guard Set(source.keys).isSubset(of: allowedSiblings) else { return nil }
            if overlaps { note("overlapping-one-of", path) }
            var result = source
            result[keyword] = nil
            result["type"] = .string("string")
            result["enum"] = .array(values)
            return result
        }

        /// `[T, null]` is `type: T` + `nullable: true`, which the template
        /// does write. Anything wider is not this shape.
        private mutating func adaptNullableUnion(
            _ source: [String: JSONValue],
            path: String
        ) -> [String: JSONValue]? {
            guard !(source["anyOf"] != nil && source["oneOf"] != nil) else { return nil }
            let keyword = source["anyOf"] != nil ? "anyOf" : "oneOf"
            guard case .array(let branches)? = source[keyword], branches.count == 2 else {
                return nil
            }
            let nullIndexes = branches.indices.filter { index in
                branches[index] == .object(["type": .string("null")])
            }
            guard nullIndexes.count == 1 else { return nil }
            let concreteIndex = nullIndexes[0] == 0 ? 1 : 0
            guard case .object(let concreteSource) = branches[concreteIndex] else { return nil }
            let concreteValue = adapt(.object(concreteSource), path: path)
            guard case .object(let concrete) = concreteValue,
                  concrete["type"] != .string("null") else { return nil }
            if keyword == "oneOf", concrete["nullable"] == .bool(true) {
                // Both branches admit null, so the `oneOf` matched null
                // nowhere; folded, it matches null. Changed meaning: recorded.
                note("overlapping-one-of", path)
            }

            var result = source
            result[keyword] = nil
            if let nullable = result["nullable"], nullable != .bool(true) {
                note("nullable-conflict", path)
            }
            for (key, value) in concrete {
                if let existing = result[key], existing != value, !annotations.contains(key) {
                    // The sibling keyword and the branch disagree. The sibling
                    // is kept (it applies to the whole member) and the loss is
                    // recorded.
                    note("union-branch-conflict", "\(path).\(key)")
                }
                if result[key] == nil || annotations.contains(key) {
                    result[key] = result[key] ?? value
                }
            }
            result["nullable"] = .bool(true)
            return result
        }

        // MARK: - the redraw's arithmetic

        /// GEN-11's problem one layer up: `JSONValue.jinjaSendableValue()`
        /// throws on a `Decimal` that does not round-trip through `Double`, and
        /// the declaration is rendered through Jinja on **every** turn. A
        /// number like that in a `default` or a `minimum` would make the
        /// render throw, so the member carrying it is dropped.
        mutating func sanitized(_ value: JSONValue, path: String) -> JSONValue? {
            switch value {
            case .object(let members):
                var result: [String: JSONValue] = [:]
                for key in members.keys.sorted() {
                    if let kept = sanitized(members[key]!, path: "\(path).\(key)") {
                        result[key] = kept
                    }
                }
                return .object(result)
            case .array(let values):
                var result: [JSONValue] = []
                for (index, element) in values.enumerated() {
                    if let kept = sanitized(element, path: "\(path)[\(index)]") {
                        result.append(kept)
                    }
                }
                return .array(result)
            default:
                guard (try? value.jinjaSendableValue()) != nil else {
                    note("unrepresentable-number", path)
                    return nil
                }
                return value
            }
        }
    }
}
