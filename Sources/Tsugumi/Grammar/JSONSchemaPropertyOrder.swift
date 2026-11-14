import Foundation

/// The order a schema's author declared each object's `properties` in (SPEC §12 DEV-15).
///
/// `JSONValue.object` is a Swift Dictionary and keeps no order, so the grammar falls back to ascending keys. That is
/// right for a Gemma tool call, which the template redraws through `dictsort` (GEN-8, INV-1), and wrong for a
/// `response_format` answer: the answer is body text and is never redrawn, and the order of the keys is the order
/// the model has to think in. A schema that asks for `what it is` before `which category` gets `category` first
/// when the keys are sorted, and the model then picks a category before it has said what it is looking at.
/// llama.cpp's `json-schema-to-grammar` writes the required keys first in declaration order, then the optional ones
/// in declaration order; with this table the converter does the same.
///
/// The table is read off the request text (`OrderedJSON`), one entry per `properties` object in the schema, keyed by
/// that object's unordered value. The converter looks a `properties` object up by value; one that is not in the
/// table (an empty table, a schema built in process, an `allOf` merge) keeps ascending keys.
public struct JSONSchemaPropertyOrder: Equatable, Sendable {
    struct Entry: Equatable, Sendable {
        let properties: JSONValue
        let keys: [String]
    }

    private(set) var entries: [Entry] = []

    /// No declared order: every object keeps ascending keys.
    public init() {}

    /// Every `properties` object in `source`, at any depth (nested objects, `items`, `$defs`, `anyOf`, …).
    public init(source: OrderedJSON) {
        collect(source)
    }

    public var isEmpty: Bool { entries.isEmpty }

    /// The declared key order of a `properties` object, or nil when the table does not know it.
    public func keys(for properties: JSONValue) -> [String]? {
        entries.first { $0.properties == properties }?.keys
    }

    private mutating func collect(_ node: OrderedJSON) {
        switch node {
        case .object(let members):
            if case .object(let properties)? = members.first(where: { $0.0 == "properties" })?.1,
               let value = OrderedJSON.object(properties).jsonValue {
                entries.append(Entry(properties: value, keys: properties.map(\.0)))
            }
            for (_, child) in members { collect(child) }
        case .array(let items):
            for item in items { collect(item) }
        default:
            return
        }
    }
}
