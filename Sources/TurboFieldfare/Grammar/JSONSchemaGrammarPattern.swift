import Foundation

/// `pattern` (正規表現) → GBNF (SPEC §6 GEN-1 / GEN-3)。
/// 参照実装 `common/json-schema-to-grammar.cpp` の `_visit_pattern`
/// (ピン `34af94cd9`) の移植。
///
/// 参照実装と同じく **`^…$` の錨は必須**、`.` は `dot` 規則、`{m,n}` の
/// 対象が非リテラルなら **1 始まり**の連番で副規則へ持ち上げ、`(?:` は
/// ただのグループとして扱い、先読み・後読みは丸ごと捨てる。
extension JSONSchemaGrammarConverter {
    /// 参照実装の `NON_LITERAL_SET`。
    static let nonLiteralBytes: Set<UInt8> = Set("|.()[]{}*+?".utf8)
    /// 参照実装の `ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS`。
    static let escapedInRegexpsOnly: Set<UInt8> = Set("^$.[]()|{}*+?".utf8)

    private struct Piece {
        var text: String
        var isLiteral: Bool
    }

    /// 変換できたら規則名、`^…$` が無ければ `nil` (呼び出し側が近似に落とす)。
    /// 途中の破れ (括弧の不整合・`{m,n}` の数値) は診断に積んで先へ進む
    /// (参照実装と同じ)。
    mutating func visitPattern(_ pattern: String, name: String) -> String? {
        let bytes = Array(pattern.utf8)
        guard bytes.count >= 2,
              bytes.first == UInt8(ascii: "^"),
              bytes.last == UInt8(ascii: "$") else {
            approximate("unanchored-pattern: \(pattern)")
            return nil
        }
        let sub = Array(bytes[1..<(bytes.count - 1)])
        if dialect == .gemmaToolArguments, sub.contains(UInt8(ascii: "\"")) {
            // GEN-9: 正規表現が `"` を綴らせると、描き直しでずれる値が出る。
            // 文法は出す (止めるほうが害が大きい) が、記録は残す。
            approximate("unrepresentable-string-character: \(pattern)", fatal: false)
        }
        let length = sub.count
        var index = 0
        var subRuleIDs: [String: String] = [:]

        func toRule(_ piece: Piece) -> String {
            piece.isLiteral ? "\"" + piece.text + "\"" : piece.text
        }

        func transform() -> Piece {
            let start = index
            var seq: [Piece] = []

            func addDot() -> String {
                // GEN-9: gemma 方言では `"` と `\` は文字列本体に置けない。
                let rule: String
                switch (dialect, dotall) {
                case (.json, true): rule = #"[\U00000000-\U0010FFFF]"#
                case (.json, false): rule = #"[^\x0A\x0D]"#
                case (.gemmaToolArguments, true): rule = #"[^"\\]"#
                case (.gemmaToolArguments, false): rule = #"[^"\\\x0A\x0D]"#
                }
                return addRule("dot", rule)
            }

            // 続くリテラルをまとめてから並べる。
            func joinSeq() -> Piece {
                var merged: [Piece] = []
                var literal = ""
                func flush() {
                    guard !literal.isEmpty else { return }
                    merged.append(Piece(text: literal, isLiteral: true))
                    literal = ""
                }
                for piece in seq {
                    if piece.isLiteral {
                        literal += piece.text
                    } else {
                        flush()
                        merged.append(piece)
                    }
                }
                flush()
                return Piece(
                    text: merged.map(toRule).joined(separator: " "),
                    isLiteral: false)
            }

            while index < length {
                let byte = sub[index]
                switch byte {
                case UInt8(ascii: "."):
                    seq.append(Piece(text: addDot(), isLiteral: false))
                    index += 1

                case UInt8(ascii: "("):
                    index += 1
                    if index < length, sub[index] == UInt8(ascii: "?") {
                        if index + 1 < length, sub[index + 1] == UInt8(ascii: ":") {
                            index += 2  // 非捕捉グループ。ただのグループとして扱う。
                        } else {
                            // 先読み / 後読みは表現できない。参照実装と同じく
                            // 対応する ')' まで読み飛ばす (警告どまり)。
                            approximate("unsupported-pattern-syntax: \(pattern)", fatal: false)
                            var depth = 1
                            while index < length, depth > 0 {
                                if sub[index] == UInt8(ascii: "\\"), index + 1 < length {
                                    index += 2
                                } else {
                                    if sub[index] == UInt8(ascii: "(") { depth += 1 }
                                    else if sub[index] == UInt8(ascii: ")") { depth -= 1 }
                                    index += 1
                                }
                            }
                            continue
                        }
                    }
                    seq.append(Piece(text: "(" + toRule(transform()) + ")", isLiteral: false))

                case UInt8(ascii: ")"):
                    index += 1
                    if start > 0, sub[start - 1] != UInt8(ascii: "("),
                       start < 2 || sub[start - 2] != UInt8(ascii: "?")
                        || sub[start - 1] != UInt8(ascii: ":") {
                        approximate("unbalanced-parentheses: \(pattern)")
                    }
                    return joinSeq()

                case UInt8(ascii: "["):
                    var characterClass: [UInt8] = [byte]
                    index += 1
                    while index < length, sub[index] != UInt8(ascii: "]") {
                        if sub[index] == UInt8(ascii: "\\"), index + 1 < length {
                            characterClass.append(contentsOf: sub[index...(index + 1)])
                            index += 2
                        } else {
                            characterClass.append(sub[index])
                            index += 1
                        }
                    }
                    if index >= length {
                        approximate("unbalanced-square-brackets: \(pattern)")
                    }
                    characterClass.append(UInt8(ascii: "]"))
                    index += 1
                    seq.append(Piece(
                        text: String(decoding: characterClass, as: UTF8.self),
                        isLiteral: false))

                case UInt8(ascii: "|"):
                    seq.append(Piece(text: "|", isLiteral: false))
                    index += 1

                case UInt8(ascii: "*"), UInt8(ascii: "+"), UInt8(ascii: "?"):
                    guard let last = seq.last else {
                        approximate("dangling-repetition: \(pattern)")
                        index += 1
                        continue
                    }
                    seq[seq.count - 1] = Piece(
                        text: toRule(last) + String(UnicodeScalar(byte)),
                        isLiteral: false)
                    index += 1

                case UInt8(ascii: "{"):
                    var braces: [UInt8] = []
                    index += 1
                    while index < length, sub[index] != UInt8(ascii: "}") {
                        braces.append(sub[index])
                        index += 1
                    }
                    if index >= length {
                        approximate("unbalanced-curly-brackets: \(pattern)")
                    }
                    index += 1
                    let numbers = String(decoding: braces, as: UTF8.self)
                        .components(separatedBy: ",")
                    var minTimes = 0
                    var maxTimes = Int.max
                    if numbers.count == 1 {
                        guard let value = Int(numbers[0]) else {
                            approximate("invalid-repetition-count: \(pattern)")
                            return Piece(text: "", isLiteral: false)
                        }
                        minTimes = value
                        maxTimes = value
                    } else if numbers.count != 2 {
                        approximate("wrong-repetition-count: \(pattern)")
                    } else {
                        if !numbers[0].isEmpty {
                            guard let value = Int(numbers[0]) else {
                                approximate("invalid-repetition-count: \(pattern)")
                                return Piece(text: "", isLiteral: false)
                            }
                            minTimes = value
                        }
                        if !numbers[1].isEmpty {
                            guard let value = Int(numbers[1]) else {
                                approximate("invalid-repetition-count: \(pattern)")
                                return Piece(text: "", isLiteral: false)
                            }
                            maxTimes = value
                        }
                    }
                    guard let last = seq.last else {
                        approximate("dangling-repetition: \(pattern)")
                        continue
                    }
                    var target = last.text
                    if !last.isLiteral {
                        if let existing = subRuleIDs[target] {
                            target = existing
                        } else {
                            // 連番は 1 始まり (参照実装は `sub_rule_ids[sub]` を
                            // 作ってから `.size()` を読むため)。
                            let identifier = addRule(
                                name + "-" + String(subRuleIDs.count + 1), target)
                            subRuleIDs[last.text] = identifier
                            target = identifier
                        }
                    }
                    seq[seq.count - 1] = Piece(
                        text: Self.buildRepetition(
                            last.isLiteral ? "\"" + target + "\"" : target,
                            minItems: minTimes,
                            maxItems: maxTimes),
                        isLiteral: false)

                default:
                    var literal: [UInt8] = []
                    while index < length {
                        if sub[index] == UInt8(ascii: "\\"), index < length - 1 {
                            let next = sub[index + 1]
                            if Self.escapedInRegexpsOnly.contains(next) {
                                index += 1
                                literal.append(sub[index])
                                index += 1
                            } else {
                                literal.append(contentsOf: sub[index...(index + 1)])
                                index += 2
                            }
                        } else if sub[index] == UInt8(ascii: "\"") {
                            literal.append(contentsOf: Array(#"\""#.utf8))
                            index += 1
                        } else if !Self.nonLiteralBytes.contains(sub[index]),
                                  index == length - 1
                                    || literal.isEmpty
                                    || sub[index + 1] == UInt8(ascii: ".")
                                    || !Self.nonLiteralBytes.contains(sub[index + 1]) {
                            literal.append(sub[index])
                            index += 1
                        } else {
                            break
                        }
                    }
                    if !literal.isEmpty {
                        seq.append(Piece(
                            text: String(decoding: literal, as: UTF8.self),
                            isLiteral: true))
                    }
                }
            }
            return joinSeq()
        }

        return addRule(name, dialect.quoted("(" + toRule(transform()) + ")"))
    }
}
