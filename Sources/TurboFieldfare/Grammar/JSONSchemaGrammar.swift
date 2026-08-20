import Foundation

/// JSON Schema → GBNF 変換 (SPEC §6 GEN-1 / GEN-3)。
///
/// 参照実装は `~/LLM/llama.cpp` のピン `34af94cd9`、
/// `common/json-schema-to-grammar.cpp`。規則の並び・規則名・空白の入れ方まで
/// あちらに合わせる (期待値は `tests/test-json-schema-to-grammar.cpp` から取る)。
///
/// **逸脱 (SPEC §12 に登録)**: オブジェクトのプロパティは宣言順ではなく
/// **キーの昇順**で並べる。理由は 2 つあり、どちらも単独で決定的である。
/// 1. こちらのスキーマ表現 `JSONValue.object` は Swift の `Dictionary` で、
///    宣言順を保持していない。
/// 2. サーバーのチャットテンプレートは tool call の引数を Jinja の `dictsort`
///    で描く (`server_chat_template.jinja`)。生成物がキー昇順でなければ
///    「描き直し == 生成」(SPEC §7 INV-1) が破れる。
///
/// 並びについてそれ以外は参照実装のまま: 出力の規則は規則名の昇順、`enum` は
/// 与えられた順、タプルは宣言順。
public enum JSONSchemaGrammarDialect: Sendable, Equatable {
    /// 厳密な JSON。キーは `"引用符付き"`、文字列は `"…"`。`response_format` 用。
    case json
    /// サーバーのテンプレートが描き `GemmaToolCallParser` が読む引数の方言。
    /// オブジェクトのキーは裸 (`[A-Za-z0-9_\-.$]+`)、文字列値は JSON 形式
    /// `"…"` と学習形式 `<|"|>…<|"|>` のどちらでもよい。
    case gemmaToolArguments
}

/// 変換の結果。`approximations` は GEN-2 で近似に落とした箇所の記録で、
/// サーバーはこれをログに出す。空なら参照実装と同じ厳密な変換ができている。
public struct JSONSchemaGrammarResult: Sendable, Equatable {
    public let grammar: String
    public let approximations: [String]

    public init(grammar: String, approximations: [String]) {
        self.grammar = grammar
        self.approximations = approximations
    }
}

/// 厳密モード (テスト専用) が「参照実装ならここで弾いていた」と言うための型。
/// 公開の入口 (`JSONSchemaGrammar.grammar(for:dialect:)`) は
/// **スキーマの内容では決して throw しない** (SPEC §6 GEN-2)。
public struct JSONSchemaGrammarStrictError: Error, Equatable {
    public let approximations: [String]
}

public enum JSONSchemaGrammar {
    /// GEN-2 の入口。表現できないスキーマ要素は拘束できる近似に落ち、
    /// 落ちた事実は `approximations` に載る。決して throw しない。
    public static func grammar(
        for schema: JSONValue,
        dialect: JSONSchemaGrammarDialect = .json
    ) -> JSONSchemaGrammarResult {
        build(dialect: dialect) { builder in
            _ = builder.addSchema("", schema)
        }
    }

    /// 厳密モード (テスト専用)。参照実装が例外を投げていた形を「投げていた」と
    /// 言えるようにするためだけにあり、サーバーの経路からは呼ばない。
    static func strictGrammar(
        for schema: JSONValue,
        dialect: JSONSchemaGrammarDialect = .json
    ) throws -> String {
        var builder = JSONSchemaGrammarBuilder(
            converter: JSONSchemaGrammarConverter(dialect: dialect))
        _ = builder.addSchema("", schema)
        if builder.converter.hasFatalApproximation {
            throw JSONSchemaGrammarStrictError(
                approximations: builder.converter.approximations)
        }
        return builder.converter.formatGrammar()
    }

    /// 参照実装の `build_grammar` に対応する組み立て口。tool call の文法のように
    /// 複数のスキーマと手書きの規則を 1 つの文法にまとめるときに使う。
    public static func build(
        dialect: JSONSchemaGrammarDialect = .json,
        _ body: (inout JSONSchemaGrammarBuilder) -> Void
    ) -> JSONSchemaGrammarResult {
        var builder = JSONSchemaGrammarBuilder(
            converter: JSONSchemaGrammarConverter(dialect: dialect))
        body(&builder)
        return JSONSchemaGrammarResult(
            grammar: builder.converter.formatGrammar(),
            approximations: builder.converter.approximations)
    }
}

/// `build(dialect:_:)` に渡る組み立て口 (参照実装 `common_grammar_builder`)。
public struct JSONSchemaGrammarBuilder {
    var converter: JSONSchemaGrammarConverter

    public mutating func addRule(_ name: String, _ rule: String) -> String {
        converter.addRule(name, rule)
    }

    public mutating func addSchema(_ name: String, _ schema: JSONValue) -> String {
        var schema = schema
        converter.resolveRefs(&schema, url: "")
        return converter.visit(schema, name == "root" ? "" : name)
    }
}

// MARK: - 表

struct JSONSchemaGrammarBuiltinRule {
    let content: String
    let deps: [String]

    init(_ content: String, _ deps: [String] = []) {
        self.content = content
        self.deps = deps
    }
}

extension JSONSchemaGrammarDialect {
    static let spaceRule = #"| " " | "\n"{1,2} [ \t]{0,20}"#

    /// 文字列の 2 形式。`.json` は `"…"` だけ、`.gemmaToolArguments` は
    /// `"…"` と `<|"|>…<|"|>` の選択 (`<|"|>` は `"` を含むので、`char` が
    /// 生の `"` を拒む限り本文が終端子を飲み込むことはない)。
    func quoted(_ body: String) -> String {
        switch self {
        case .json:
            return #""\"" "# + body + #" "\"""#
        case .gemmaToolArguments:
            return #""\"" "# + body + #" "\"" | "<|\"|>" "# + body + #" "<|\"|>""#
        }
    }

    /// キーの字集合 (`.gemmaToolArguments` のみ)。`GemmaToolCallParser` の
    /// `objectKey()` が受ける文字と 1:1。
    static let keyRanges: [(UInt8, UInt8)] = [
        (0x30, 0x39),  // 0-9
        (0x41, 0x5A),  // A-Z
        (0x61, 0x7A),  // a-z
        (0x5F, 0x5F),  // _
        (0x2D, 0x2D),  // -
        (0x2E, 0x2E),  // .
        (0x24, 0x24),  // $
    ]

    var primitiveRules: [String: JSONSchemaGrammarBuiltinRule] {
        var table: [String: JSONSchemaGrammarBuiltinRule] = [
            "boolean": .init(#"("true" | "false")"#),
            "decimal-part": .init("[0-9]{1,16}"),
            "integral-part": .init("[0] | [1-9] [0-9]{0,15}"),
            "number": .init(
                #"("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?"#,
                ["integral-part", "decimal-part"]),
            "integer": .init(#"("-"? integral-part)"#, ["integral-part"]),
            "value": .init(
                "object | array | string | number | boolean | null",
                ["object", "array", "string", "number", "boolean", "null"]),
            "array": .init(#""[" space ( value ("," space value)* )? space "]""#, ["value"]),
            "uuid": .init(quoted(
                "[0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" "
                + "[0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12}")),
            "char": .init(#"[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})"#),
            "string": .init(quoted("char*"), ["char"]),
            "null": .init(#""null""#),
        ]
        switch self {
        case .json:
            table["object"] = .init(
                #""{" space ( string ":" space value ("," space string ":" space value)* )? space "}""#,
                ["string", "value"])
        case .gemmaToolArguments:
            table["object"] = .init(
                #""{" space ( key ":" space value ("," space key ":" space value)* )? space "}""#,
                ["key", "value"])
            table["key-char"] = .init(Self.characterClass(excluding: []))
            table["key"] = .init("key-char+", ["key-char"])
        }
        return table
    }

    var stringFormatRules: [String: JSONSchemaGrammarBuiltinRule] {
        [
            "date": .init(
                #"[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )"#),
            "time": .init(
                #"([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? "#
                + #"( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )"#),
            "date-time": .init(#"date "T" time"#, ["date", "time"]),
            "date-string": .init(quoted("date"), ["date"]),
            "time-string": .init(quoted("time"), ["time"]),
            "date-time-string": .init(quoted("date-time"), ["date-time"]),
        ]
    }

    var reservedNames: Set<String> {
        var names: Set<String> = ["root"]
        names.formUnion(primitiveRules.keys)
        names.formUnion(stringFormatRules.keys)
        return names
    }

    /// キーの字集合から `excluded` を引いた GBNF の文字クラス。
    static func characterClass(excluding excluded: Set<UInt8>) -> String {
        var out = "["
        for (low, high) in keyRanges {
            var start = low
            while start <= high {
                if excluded.contains(start) {
                    if start == 0xFF { break }
                    start += 1
                    continue
                }
                var end = start
                while end < high, !excluded.contains(end + 1) { end += 1 }
                if start == end {
                    out += JSONSchemaGrammarConverter.escapeInRange(start)
                } else {
                    out += JSONSchemaGrammarConverter.escapeInRange(start)
                    out += "-"
                    out += JSONSchemaGrammarConverter.escapeInRange(end)
                }
                if end == 0xFF { break }
                start = end + 1
            }
        }
        return out + "]"
    }
}

// MARK: - 変換器

struct JSONSchemaGrammarConverter {
    let dialect: JSONSchemaGrammarDialect
    let dotall: Bool
    let primitives: [String: JSONSchemaGrammarBuiltinRule]
    let formats: [String: JSONSchemaGrammarBuiltinRule]
    let reserved: Set<String>

    private(set) var rules: [String: String]
    private var refs: [String: JSONValue] = [:]
    private var refsBeingResolved: Set<String> = []
    /// GEN-2 の記録。`fatal` は「参照実装ならここで弾いていた」の印で、
    /// 厳密モード (テスト) だけがこれを見る。
    private(set) var diagnostics: [(message: String, fatal: Bool)] = []

    init(dialect: JSONSchemaGrammarDialect, dotall: Bool = false) {
        self.dialect = dialect
        self.dotall = dotall
        self.primitives = dialect.primitiveRules
        self.formats = dialect.stringFormatRules
        self.reserved = dialect.reservedNames
        self.rules = ["space": JSONSchemaGrammarDialect.spaceRule]
    }

    var approximations: [String] { diagnostics.map(\.message) }
    var hasFatalApproximation: Bool { diagnostics.contains { $0.fatal } }

    mutating func approximate(_ message: String, fatal: Bool = true) {
        diagnostics.append((message, fatal))
    }

    // MARK: 規則置き場

    func formatGrammar() -> String {
        rules
            .sorted { Array($0.key.utf8).lexicographicallyPrecedes(Array($1.key.utf8)) }
            .map { "\($0.key) ::= \($0.value)\n" }
            .joined()
    }

    @discardableResult
    mutating func addRule(_ name: String, _ rule: String) -> String {
        let escaped = Self.sanitizeRuleName(name)
        if rules[escaped] == nil || rules[escaped] == rule {
            rules[escaped] = rule
            return escaped
        }
        var index = 0
        while let existing = rules[escaped + String(index)], existing != rule {
            index += 1
        }
        let key = escaped + String(index)
        rules[key] = rule
        return key
    }

    /// 参照実装の `INVALID_RULE_CHARS_RE` = `[^a-zA-Z0-9-]+` → `-`。
    static func sanitizeRuleName(_ name: String) -> String {
        var out = ""
        var inRun = false
        for scalar in name.unicodeScalars {
            let allowed = (scalar >= "a" && scalar <= "z")
                || (scalar >= "A" && scalar <= "Z")
                || (scalar >= "0" && scalar <= "9")
                || scalar == "-"
            if allowed {
                out.unicodeScalars.append(scalar)
                inRun = false
            } else if !inRun {
                out += "-"
                inRun = true
            }
        }
        return out
    }

    @discardableResult
    mutating func addPrimitive(_ name: String, _ rule: JSONSchemaGrammarBuiltinRule) -> String {
        let added = addRule(name, rule.content)
        for dep in rule.deps {
            guard let depRule = primitives[dep] ?? formats[dep] else {
                approximate("unknown-primitive-dependency: \(dep)")
                continue
            }
            if rules[dep] == nil {
                addPrimitive(dep, depRule)
            }
        }
        return added
    }

    mutating func addPrimitive(_ name: String) -> String {
        guard let rule = primitives[name] else {
            approximate("unknown-primitive-dependency: \(name)")
            return name
        }
        return addPrimitive(name, rule)
    }

    // MARK: リテラル

    /// 参照実装の `GRAMMAR_LITERAL_ESCAPE_RE` = `[\r\n"\\]`。
    static func formatLiteral(_ literal: String) -> String {
        var out = "\""
        for character in literal {
            switch character {
            case "\r": out += #"\r"#
            case "\n": out += #"\n"#
            case "\"": out += #"\""#
            case "\\": out += #"\\"#
            default: out.append(character)
            }
        }
        return out + "\""
    }

    /// 文字クラスの中に生の 1 バイトを置くときの逃がし。参照実装の
    /// `GRAMMAR_RANGE_LITERAL_ESCAPE_RE` から `"` を除いたもの — 参照実装は
    /// `[^"…]` の `"` を素で書くので、`"` を逃がすと期待値が動いてしまう。
    /// `]` `-` `\` を逃がす分は参照実装のテストベクタでは差が出ず、
    /// 変な名前のときに壊れた文法が出るのを止めるだけである。
    static func escapeInRange(_ byte: UInt8) -> String {
        switch byte {
        case 0x0D: return #"\r"#
        case 0x0A: return #"\n"#
        case 0x5D: return #"\]"#
        case 0x2D: return #"\-"#
        case 0x5C: return #"\\"#
        default: return String(UnicodeScalar(byte))
        }
    }

    static func escapeInRange(_ character: Character) -> String {
        guard let ascii = character.asciiValue else { return String(character) }
        return escapeInRange(ascii)
    }

    /// 参照実装の `json(x).dump()`。オブジェクトのキーは決定 (a) により昇順。
    static func jsonDump(_ value: JSONValue) -> String {
        switch value {
        case .object(let members):
            let body = members.keys
                .sorted { Array($0.utf8).lexicographicallyPrecedes(Array($1.utf8)) }
                .map { "\(dumpString($0)):\(jsonDump(members[$0]!))" }
                .joined(separator: ",")
            return "{\(body)}"
        case .array(let items):
            return "[\(items.map(jsonDump).joined(separator: ","))]"
        case .string(let text): return dumpString(text)
        case .integer(let value): return String(value)
        case .unsignedInteger(let value): return String(value)
        case .decimal(let value): return NSDecimalNumber(decimal: value).stringValue
        case .number(let value): return String(value)
        case .bool(let value): return value ? "true" : "false"
        case .null: return "null"
        }
    }

    static func dumpString(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += #"\""#
            case "\\": out += #"\\"#
            case "\u{08}": out += #"\b"#
            case "\u{0C}": out += #"\f"#
            case "\n": out += #"\n"#
            case "\r": out += #"\r"#
            case "\t": out += #"\t"#
            default:
                if scalar.value < 0x20 {
                    out += String(format: #"\u%04x"#, scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }

    /// 定数 1 個を表す規則本体。`.json` は dump をそのままリテラルに、
    /// `.gemmaToolArguments` は文字列だけ 2 形式の選択に開く。
    mutating func constantRule(_ value: JSONValue) -> String {
        switch dialect {
        case .json:
            return Self.formatLiteral(Self.jsonDump(value))
        case .gemmaToolArguments:
            return gemmaConstantRule(value)
        }
    }

    private mutating func gemmaConstantRule(_ value: JSONValue) -> String {
        switch value {
        case .string(let text):
            let json = Self.formatLiteral(Self.dumpString(text))
            let trained = Self.formatLiteral("<|\"|>" + text + "<|\"|>")
            return "(\(json) | \(trained))"
        case .array(let items):
            let body = items.map { gemmaConstantRule($0) }.joined(separator: #" "," "#)
            return items.isEmpty ? #""[]""# : #""[" "# + body + #" "]""#
        case .object(let members):
            let keys = members.keys
                .sorted { Array($0.utf8).lexicographicallyPrecedes(Array($1.utf8)) }
            if keys.isEmpty { return #""{}""# }
            let body = keys
                .map { Self.formatLiteral($0 + ":") + " " + gemmaConstantRule(members[$0]!) }
                .joined(separator: #" "," "#)
            return #""{" "# + body + #" "}""#
        default:
            return Self.formatLiteral(Self.jsonDump(value))
        }
    }

    // MARK: 繰り返し

    static func buildRepetition(
        _ item: String,
        minItems: Int,
        maxItems: Int,
        separator: String = ""
    ) -> String {
        let hasMax = maxItems != Int.max
        if maxItems == 0 { return "" }
        if minItems == 0 && maxItems == 1 { return item + "?" }
        if separator.isEmpty {
            if minItems == 1 && !hasMax { return item + "+" }
            if minItems == 0 && !hasMax { return item + "*" }
            return item + "{" + String(minItems) + "," + (hasMax ? String(maxItems) : "") + "}"
        }
        var result = item + " " + buildRepetition(
            "(" + separator + " " + item + ")",
            minItems: minItems == 0 ? 0 : minItems - 1,
            maxItems: hasMax ? maxItems - 1 : maxItems)
        if minItems == 0 { result = "(" + result + ")?" }
        return result
    }

    // MARK: $ref

    /// 参照実装の `resolve_refs`。`https://` の取得は採らない (ローカル専用の
    /// サーバーは外へ HTTP を出さない — SPEC §12 DEV-4 と同じ理由) ので、
    /// リモート参照は解決できない参照として近似に落ちる。
    mutating func resolveRefs(_ schema: inout JSONValue, url: String) {
        let root = schema
        visitRefs(&schema, root: root, url: url)
    }

    private mutating func visitRefs(_ node: inout JSONValue, root: JSONValue, url: String) {
        switch node {
        case .array(var items):
            for index in items.indices {
                visitRefs(&items[index], root: root, url: url)
            }
            node = .array(items)
        case .object(var members):
            if case .string(let ref)? = members["$ref"] {
                if refs[ref] == nil {
                    // 未対応の形 (リモート `https://` を含む) はここでは黙って
                    // 見送り、実際に使われたときに `resolveRef` が 1 度だけ
                    // 記録する。使われない `$ref` は近似ではない。
                    guard ref.hasPrefix("#/") else { return }
                    let pointer = String(ref.dropFirst())
                    var target = root
                    for token in pointer.split(separator: "/", omittingEmptySubsequences: true) {
                        let selector = String(token)
                        if case .object(let object) = target, let next = object[selector] {
                            target = next
                        } else if case .array(let array) = target,
                                  let index = Int(selector), index < array.count {
                            target = array[index]
                        } else {
                            return
                        }
                    }
                    refs[ref] = target
                }
                return
            }
            for key in members.keys {
                var child = members[key]!
                visitRefs(&child, root: root, url: url)
                members[key] = child
            }
            node = .object(members)
        default:
            return
        }
    }

    private mutating func resolveRef(_ ref: String) -> String {
        let fragment = ref.firstIndex(of: "#").map { String(ref[ref.index(after: $0)...]) } ?? ref
        var refName = "ref" + Self.sanitizeRuleName(fragment)
        if rules[refName] == nil && !refsBeingResolved.contains(ref) {
            guard let resolved = refs[ref] else {
                // GEN-2: 解決できない参照は「何でもよい JSON 値」に落とす。
                approximate(ref.hasPrefix("#/")
                    ? "unresolvable-ref: \(ref)"
                    : "unsupported-ref: \(ref)")
                return addPrimitive("value")
            }
            refsBeingResolved.insert(ref)
            refName = visit(resolved, refName)
            refsBeingResolved.remove(ref)
        }
        return refName
    }

    // MARK: 除外キー

    private final class TrieNode {
        var children: [UInt8: TrieNode] = [:]
        var isEndOfString = false
    }

    /// 参照実装の `_not_strings`。`.gemmaToolArguments` では `["]` の囲いが
    /// 無く、`char` の代わりに `key-char`、否定クラスの代わりにキーの字集合の
    /// 差集合を使う (GBNF に交差が無いため)。空キーは書けないので参照実装が
    /// 付ける末尾の `?` も付けない。
    mutating func notStrings(_ strings: [String]) -> String {
        let trie = TrieNode()
        for string in strings {
            var node = trie
            for byte in Array(string.utf8) {
                if let next = node.children[byte] {
                    node = next
                } else {
                    let next = TrieNode()
                    node.children[byte] = next
                    node = next
                }
            }
            node.isEndOfString = true
        }

        let isGemma = dialect == .gemmaToolArguments
        let charRule = isGemma ? addPrimitive("key-char") : addPrimitive("char")

        var out = isGemma ? "( " : "[\"] ( "
        func visitNode(_ node: TrieNode) {
            var rejects: [UInt8] = []
            var first = true
            for byte in node.children.keys.sorted() {
                let child = node.children[byte]!
                rejects.append(byte)
                if first { first = false } else { out += " | " }
                out += "[" + Self.escapeInRange(byte) + "]"
                if !child.children.isEmpty {
                    out += " ("
                    visitNode(child)
                    out += ")"
                } else if child.isEndOfString {
                    out += " " + charRule + "+"
                }
            }
            if !node.children.isEmpty {
                if !first { out += " | " }
                if isGemma {
                    out += JSONSchemaGrammarDialect.characterClass(excluding: Set(rejects))
                } else {
                    out += "[^\"" + rejects.map { Self.escapeInRange($0) }.joined() + "]"
                }
                out += " " + charRule + "*"
            }
        }
        visitNode(trie)

        out += " )"
        if !isGemma {
            if !trie.isEndOfString { out += "?" }
            out += " [\"]"
        }
        return out
    }

    // MARK: オブジェクト

    mutating func buildObjectRule(
        properties: [(String, JSONValue)],
        required: Set<String>,
        name: String,
        additionalProperties: JSONValue?
    ) -> String {
        // 決定 (a): プロパティはキーの昇順。参照実装は宣言順 (SPEC §12)。
        let properties = properties.sorted {
            Array($0.0.utf8).lexicographicallyPrecedes(Array($1.0.utf8))
        }
        var kvRuleNames: [String: String] = [:]
        var propertyNames: [String] = []
        var requiredProps: [String] = []
        var optionalProps: [String] = []

        for (propertyName, propertySchema) in properties {
            let prefix = name + (name.isEmpty ? "" : "-")
            let propertyRule = visit(propertySchema, prefix + propertyName)
            kvRuleNames[propertyName] = addRule(
                prefix + propertyName + "-kv",
                keyLiteral(propertyName) + " space \":\" space " + propertyRule)
            if required.contains(propertyName) {
                requiredProps.append(propertyName)
            } else {
                optionalProps.append(propertyName)
            }
            propertyNames.append(propertyName)
        }

        let allowsAdditional: Bool
        switch additionalProperties {
        case .bool(true), .object: allowsAdditional = true
        default: allowsAdditional = false
        }
        if allowsAdditional {
            let subName = name + (name.isEmpty ? "" : "-") + "additional"
            let valueRule: String
            if case .object = additionalProperties {
                valueRule = visit(additionalProperties!, subName + "-value")
            } else {
                valueRule = addPrimitive("value")
            }
            let keyRule = propertyNames.isEmpty
                ? addPrimitive(dialect == .gemmaToolArguments ? "key" : "string")
                : addRule(subName + "-k", notStrings(propertyNames))
            let kvRule = addRule(subName + "-kv", keyRule + " \":\" space " + valueRule)
            kvRuleNames["*"] = kvRule
            optionalProps.append("*")
        }

        var rule = "\"{\" space "
        for (index, key) in requiredProps.enumerated() {
            if index > 0 { rule += " \",\" space " }
            rule += kvRuleNames[key]!
        }

        if !optionalProps.isEmpty {
            rule += " ("
            if !requiredProps.isEmpty { rule += " \",\" space ( " }

            func recursiveRefs(_ keys: ArraySlice<String>, firstIsOptional: Bool) -> String {
                guard let key = keys.first else { return "" }
                let kvRuleName = kvRuleNames[key]!
                let commaRef = "( \",\" space " + kvRuleName + " )"
                var result: String
                if firstIsOptional {
                    result = commaRef + (key == "*" ? "*" : "?")
                } else {
                    result = kvRuleName + (key == "*" ? " " + commaRef + "*" : "")
                }
                let rest = keys.dropFirst()
                if !rest.isEmpty {
                    result += " " + addRule(
                        name + (name.isEmpty ? "" : "-") + key + "-rest",
                        recursiveRefs(rest, firstIsOptional: true))
                }
                return result
            }

            for index in optionalProps.indices {
                if index > 0 { rule += " | " }
                rule += recursiveRefs(optionalProps[index...], firstIsOptional: false)
            }
            if !requiredProps.isEmpty { rule += " )" }
            rule += " )?"
        }

        rule += " space \"}\""
        return rule
    }

    /// プロパティ名をキーとして書くときのリテラル。`.json` は JSON dump を
    /// もう一度逃がす (二重の逃がし)、`.gemmaToolArguments` は裸。
    private mutating func keyLiteral(_ propertyName: String) -> String {
        switch dialect {
        case .json:
            return Self.formatLiteral(Self.dumpString(propertyName))
        case .gemmaToolArguments:
            let representable = !propertyName.isEmpty && propertyName.utf8.allSatisfy { byte in
                JSONSchemaGrammarDialect.keyRanges.contains { byte >= $0.0 && byte <= $0.1 }
            }
            if !representable {
                // この方言では書けないキー。文法は出すが、テンプレートが描いた
                // ものをパーサが読み戻せないので記録に残す (GEN-2)。
                approximate("unrepresentable-key: \(propertyName)", fatal: false)
            }
            return Self.formatLiteral(propertyName)
        }
    }

    // MARK: visit

    mutating func visit(_ schema: JSONValue, _ name: String) -> String {
        let schemaType = Self.member(schema, "type")
        let schemaFormat = Self.string(Self.member(schema, "format")) ?? ""
        let ruleName = reserved.contains(name)
            ? name + "-"
            : (name.isEmpty ? "root" : name)

        if case .string(let ref)? = Self.member(schema, "$ref") {
            return addRule(ruleName, resolveRef(ref))
        }
        if let alternatives = Self.array(Self.member(schema, "oneOf"))
            ?? Self.array(Self.member(schema, "anyOf")) {
            return addRule(ruleName, unionRule(name: name, alternatives: alternatives))
        }
        if case .array(let types)? = schemaType {
            let variants: [JSONValue] = types.map { type in
                guard case .object(var members) = schema else { return type }
                members["type"] = type
                return .object(members)
            }
            return addRule(ruleName, unionRule(name: name, alternatives: variants))
        }
        if let constant = Self.member(schema, "const") {
            return addRule(ruleName, constantRule(constant))
        }
        if case .array(let values)? = Self.member(schema, "enum") {
            let alternatives = values.map { constantRule($0) }
            return addRule(ruleName, "(" + alternatives.joined(separator: " | ") + ")")
        }
        let additionalProperties = Self.member(schema, "additionalProperties")
        let additionalIsNotTrue: Bool = {
            guard let additionalProperties else { return false }
            if case .bool(true) = additionalProperties { return false }
            return true
        }()
        if (schemaType == nil || schemaType == .string("object"))
            && (Self.member(schema, "properties") != nil || additionalIsNotTrue) {
            var required: Set<String> = []
            if case .array(let items)? = Self.member(schema, "required") {
                for item in items {
                    if case .string(let text) = item { required.insert(text) }
                }
            }
            var properties: [(String, JSONValue)] = []
            if case .object(let members)? = Self.member(schema, "properties") {
                properties = members.map { ($0.key, $0.value) }
            }
            return addRule(ruleName, buildObjectRule(
                properties: properties,
                required: required,
                name: name,
                additionalProperties: additionalProperties))
        }
        if schemaType == nil || schemaType == .string("object") || schemaType == .string("string"),
           case .array(let components)? = Self.member(schema, "allOf") {
            return addRule(ruleName, allOfRule(components, name: name))
        }
        if schemaType == nil || schemaType == .string("array"),
           let items = Self.member(schema, "items") ?? Self.member(schema, "prefixItems") {
            if case .array(let tuple) = items {
                var rule = "\"[\" space "
                for (index, item) in tuple.enumerated() {
                    if index > 0 { rule += " \",\" space " }
                    rule += visit(item, name + (name.isEmpty ? "" : "-") + "tuple-" + String(index))
                }
                rule += " space \"]\""
                return addRule(ruleName, rule)
            }
            let itemRule = visit(items, name + (name.isEmpty ? "" : "-") + "item")
            let minItems = Self.int(Self.member(schema, "minItems")) ?? 0
            let maxItems = Self.int(Self.member(schema, "maxItems")) ?? Int.max
            return addRule(ruleName, "\"[\" space " + Self.buildRepetition(
                itemRule, minItems: minItems, maxItems: maxItems,
                separator: "\",\" space") + " space \"]\"")
        }
        if schemaType == nil || schemaType == .string("string"),
           let patternRule = visitPatternIfSupported(schema, ruleName: ruleName) {
            return patternRule
        }
        if (schemaType == nil || schemaType == .string("string"))
            && Self.isUUIDFormat(schemaFormat) {
            return addPrimitive(ruleName == "root" ? "root" : schemaFormat,
                                primitives["uuid"]!)
        }
        if schemaType == nil || schemaType == .string("string"),
           let formatRule = formats[schemaFormat + "-string"] {
            return addRule(ruleName, addPrimitive(schemaFormat + "-string", formatRule))
        }
        if schemaType == .string("string"),
           Self.member(schema, "minLength") != nil || Self.member(schema, "maxLength") != nil {
            let charRule = addPrimitive("char")
            let minLength = Self.int(Self.member(schema, "minLength")) ?? 0
            let maxLength = Self.int(Self.member(schema, "maxLength")) ?? Int.max
            return addRule(ruleName, dialect.quoted(Self.buildRepetition(
                charRule, minItems: minLength, maxItems: maxLength)))
        }
        if schemaType == .string("integer"),
           let boundsRule = visitIntegerBoundsIfSupported(schema, ruleName: ruleName) {
            return boundsRule
        }
        if Self.isEmptySchema(schema) || schemaType == .string("object") {
            return addRule(ruleName, addPrimitive("object"))
        }
        if schemaType == nil, case .object = schema {
            // 型の指定も構造のキーワードも無い (`{"description": "…"}` など)。
            // JSON Schema の意味は `{}` と同じで、何でも受ける。
            return addRule(ruleName, addPrimitive("value"))
        }
        // ここから先は参照実装が例外を投げる領域。GEN-2 により 400 にはせず、
        // 拘束できるいちばん近いもの (何でもよい JSON 値) に落として記録する。
        guard case .string(let typeName) = schemaType,
              primitives[typeName] != nil else {
            approximate("unrecognized-schema: \(Self.jsonDump(schema))")
            return addRule(ruleName, addPrimitive("value"))
        }
        return addPrimitive(ruleName == "root" ? "root" : typeName,
                            primitives[typeName]!)
    }

    private mutating func unionRule(name: String, alternatives: [JSONValue]) -> String {
        var rules: [String] = []
        for (index, alternative) in alternatives.enumerated() {
            rules.append(visit(
                alternative,
                name + (name.isEmpty ? "alternative-" : "-") + String(index)))
        }
        return rules.joined(separator: " | ")
    }

    private mutating func allOfRule(_ components: [JSONValue], name: String) -> String {
        var required: Set<String> = []
        var properties: [(String, JSONValue)] = []
        var enumValues: [String: Int] = [:]

        func addComponent(_ component: JSONValue, isRequired: Bool) {
            if case .string(let ref)? = Self.member(component, "$ref") {
                guard let resolved = refs[ref] else {
                    approximate("unresolvable-ref: \(ref)")
                    return
                }
                addComponent(resolved, isRequired: isRequired)
            } else if case .object(let members)? = Self.member(component, "properties") {
                for (key, value) in members {
                    properties.append((key, value))
                    if isRequired { required.insert(key) }
                }
            } else if case .array(let values)? = Self.member(component, "enum") {
                for value in values {
                    enumValues[constantRule(value), default: 0] += 1
                }
            }
        }

        for component in components {
            if case .array(let alternatives)? = Self.member(component, "anyOf") {
                for alternative in alternatives {
                    addComponent(alternative, isRequired: false)
                }
            } else {
                addComponent(component, isRequired: true)
            }
        }

        if !enumValues.isEmpty {
            let intersection = enumValues
                .filter { $0.value == components.count }
                .keys
                .sorted { Array($0.utf8).lexicographicallyPrecedes(Array($1.utf8)) }
            if !intersection.isEmpty {
                return "(" + intersection.joined(separator: " | ") + ")"
            }
        }
        return buildObjectRule(
            properties: properties, required: required,
            name: name, additionalProperties: nil)
    }

    // MARK: 下請け

    static func member(_ schema: JSONValue, _ key: String) -> JSONValue? {
        guard case .object(let members) = schema else { return nil }
        return members[key]
    }

    static func string(_ value: JSONValue?) -> String? {
        guard case .string(let text)? = value else { return nil }
        return text
    }

    static func array(_ value: JSONValue?) -> [JSONValue]? {
        guard case .array(let items)? = value else { return nil }
        return items
    }

    static func int(_ value: JSONValue?) -> Int? {
        switch value {
        case .integer(let value): return Int(exactly: value)
        case .unsignedInteger(let value): return Int(exactly: value)
        default: return nil
        }
    }

    static func isEmptySchema(_ schema: JSONValue) -> Bool {
        switch schema {
        case .object(let members): return members.isEmpty
        case .array(let items): return items.isEmpty
        case .null: return true
        default: return false
        }
    }

    static func isUUIDFormat(_ format: String) -> Bool {
        guard format.hasPrefix("uuid") else { return false }
        let rest = format.dropFirst(4)
        if rest.isEmpty { return true }
        return rest.count == 1 && ("1"..."5").contains(String(rest))
    }
}

// MARK: - 段階的に入る枝

extension JSONSchemaGrammarConverter {
    /// `pattern` (正規表現) → GBNF。`^…$` が無い等で表現できないときは `nil` を
    /// 返し、スキーマは型どおりの原始規則 (`string`) に落ちる (GEN-2)。
    mutating func visitPatternIfSupported(_ schema: JSONValue, ruleName: String) -> String? {
        guard let pattern = Self.member(schema, "pattern") else { return nil }
        guard case .string(let text) = pattern else {
            approximate("non-string-pattern")
            return addRule(ruleName, addPrimitive("string"))
        }
        // 破れたときに途中まで足した規則 (`dot` や副規則) を残さないよう、
        // 規則置き場を巻き戻してから素の `string` に落とす (GEN-2)。
        let snapshot = rules
        let diagnosticsBefore = diagnostics.count
        if let rule = visitPattern(text, name: ruleName),
           !diagnostics[diagnosticsBefore...].contains(where: { $0.fatal }) {
            return rule
        }
        rules = snapshot
        return addRule(ruleName, addPrimitive("string"))
    }

    /// 整数の `minimum` / `maximum` → GBNF。境界が整数でなければ `nil` を返し、
    /// スキーマは `integer` の原始規則に落ちる (GEN-2)。
    mutating func visitIntegerBoundsIfSupported(
        _ schema: JSONValue, ruleName: String
    ) -> String? {
        let minimum = Self.member(schema, "minimum")
        let exclusiveMinimum = Self.member(schema, "exclusiveMinimum")
        let maximum = Self.member(schema, "maximum")
        let exclusiveMaximum = Self.member(schema, "exclusiveMaximum")
        guard minimum != nil || exclusiveMinimum != nil
                || maximum != nil || exclusiveMaximum != nil else {
            return nil
        }
        var minValue = Int64.min
        var maxValue = Int64.max
        if let minimum {
            guard let value = Self.integerBound(minimum) else {
                approximate("non-integer-bound: minimum")
                return nil
            }
            minValue = value
        } else if let exclusiveMinimum {
            guard let value = Self.integerBound(exclusiveMinimum),
                  value < Int64.max else {
                approximate("non-integer-bound: exclusiveMinimum")
                return nil
            }
            minValue = value + 1
        }
        if let maximum {
            guard let value = Self.integerBound(maximum) else {
                approximate("non-integer-bound: maximum")
                return nil
            }
            maxValue = value
        } else if let exclusiveMaximum {
            guard let value = Self.integerBound(exclusiveMaximum),
                  value > Int64.min else {
                approximate("non-integer-bound: exclusiveMaximum")
                return nil
            }
            maxValue = value - 1
        }
        return addRule(ruleName, "(" + Self.minMaxInt(minValue, maxValue) + ")")
    }
}
