import Foundation

/// 整数の `minimum` / `maximum` / `exclusiveMinimum` / `exclusiveMaximum`
/// → GBNF (SPEC §6 GEN-1 / GEN-3)。
/// 参照実装 `common/json-schema-to-grammar.cpp` の `build_min_max_int`
/// (ピン `34af94cd9`) の移植。桁数で場合分けし、`uniform_range` /
/// `more_digits` / `digit_range` で数字の範囲を書き下す。
extension JSONSchemaGrammarConverter {
    /// 参照実装の `decimals_left` の既定値。
    static let integerBoundsDecimals = 16

    static func minMaxInt(
        _ minimum: Int64,
        _ maximum: Int64,
        decimalsLeft: Int = integerBoundsDecimals,
        topLevel: Bool = true
    ) -> String {
        var minValue = minimum
        let maxValue = maximum
        let hasMin = minValue != Int64.min
        let hasMax = maxValue != Int64.max
        var out = ""

        let zero = UInt8(ascii: "0")
        let one = UInt8(ascii: "1")
        let nine = UInt8(ascii: "9")

        func digitRange(_ from: UInt8, _ to: UInt8) {
            out += "["
            if from == to {
                out.unicodeScalars.append(UnicodeScalar(from))
            } else {
                out.unicodeScalars.append(UnicodeScalar(from))
                out += "-"
                out.unicodeScalars.append(UnicodeScalar(to))
            }
            out += "]"
        }

        func moreDigits(_ minDigits: Int, _ maxDigits: Int) {
            out += "[0-9]"
            if minDigits == maxDigits && minDigits == 1 { return }
            out += "{"
            out += String(minDigits)
            if maxDigits != minDigits {
                out += ","
                if maxDigits != Int.max { out += String(maxDigits) }
            }
            out += "}"
        }

        func uniformRange(_ from: [UInt8], _ to: [UInt8]) {
            var index = 0
            while index < from.count, index < to.count, from[index] == to[index] {
                index += 1
            }
            if index > 0 {
                out += "\"" + String(decoding: from[0..<index], as: UTF8.self) + "\""
            }
            guard index < from.count, index < to.count else { return }
            if index > 0 { out += " " }
            let subLength = from.count - index - 1
            guard subLength > 0 else {
                out += "["
                out.unicodeScalars.append(UnicodeScalar(from[index]))
                out += "-"
                out.unicodeScalars.append(UnicodeScalar(to[index]))
                out += "]"
                return
            }
            let fromSub = Array(from[(index + 1)...])
            let toSub = Array(to[(index + 1)...])
            let subZeros = [UInt8](repeating: zero, count: subLength)
            let subNines = [UInt8](repeating: nine, count: subLength)

            var toReached = false
            out += "("
            if fromSub == subZeros {
                digitRange(from[index], to[index] - 1)
                out += " "
                moreDigits(subLength, subLength)
            } else {
                out += "["
                out.unicodeScalars.append(UnicodeScalar(from[index]))
                out += "] ("
                uniformRange(fromSub, subNines)
                out += ")"
                if from[index] < to[index] - 1 {
                    out += " | "
                    if toSub == subNines {
                        digitRange(from[index] + 1, to[index])
                        toReached = true
                    } else {
                        digitRange(from[index] + 1, to[index] - 1)
                    }
                    out += " "
                    moreDigits(subLength, subLength)
                }
            }
            if !toReached {
                out += " | "
                digitRange(to[index], to[index])
                out += " "
                uniformRange(subZeros, toSub)
            }
            out += ")"
        }

        if hasMin && hasMax {
            if minValue < 0 && maxValue < 0 {
                out += "\"-\" ("
                out += minMaxInt(-maxValue, -minValue,
                                 decimalsLeft: decimalsLeft, topLevel: true)
                out += ")"
                return out
            }
            if minValue < 0 {
                out += "\"-\" ("
                out += minMaxInt(0, -minValue, decimalsLeft: decimalsLeft, topLevel: true)
                out += ") | "
                minValue = 0
            }
            var minText = Array(String(minValue).utf8)
            let maxText = Array(String(maxValue).utf8)
            let minDigits = minText.count
            let maxDigits = maxText.count
            var digits = minDigits
            while digits < maxDigits {
                uniformRange(minText, [UInt8](repeating: nine, count: digits))
                minText = [one] + [UInt8](repeating: zero, count: digits)
                out += " | "
                digits += 1
            }
            uniformRange(minText, maxText)
            return out
        }

        let lessDecimals = max(decimalsLeft - 1, 1)

        if hasMin {
            if minValue < 0 {
                out += "\"-\" ("
                out += minMaxInt(Int64.min, -minValue,
                                 decimalsLeft: decimalsLeft, topLevel: false)
                out += ") | [0] | [1-9] "
                moreDigits(0, decimalsLeft - 1)
            } else if minValue == 0 {
                if topLevel {
                    out += "[0] | [1-9] "
                    moreDigits(0, lessDecimals)
                } else {
                    moreDigits(1, decimalsLeft)
                }
            } else if minValue <= 9 {
                let character = zero + UInt8(minValue)
                let rangeStart = topLevel ? one : zero
                if character > rangeStart {
                    digitRange(rangeStart, character - 1)
                    out += " "
                    moreDigits(1, lessDecimals)
                    out += " | "
                }
                digitRange(character, nine)
                out += " "
                moreDigits(0, lessDecimals)
            } else {
                let minText = String(minValue)
                let length = minText.count
                let character = Array(minText.utf8)[0]
                if character > one {
                    digitRange(topLevel ? one : zero, character - 1)
                    out += " "
                    moreDigits(length, lessDecimals)
                    out += " | "
                }
                digitRange(character, character)
                out += " ("
                out += minMaxInt(Int64(minText.dropFirst()) ?? 0, Int64.max,
                                 decimalsLeft: lessDecimals, topLevel: false)
                out += ")"
                if character < nine {
                    out += " | "
                    digitRange(character + 1, nine)
                    out += " "
                    moreDigits(length - 1, lessDecimals)
                }
            }
            return out
        }

        if hasMax {
            if maxValue >= 0 {
                if topLevel {
                    out += "\"-\" [1-9] "
                    moreDigits(0, lessDecimals)
                    out += " | "
                }
                out += minMaxInt(0, maxValue, decimalsLeft: decimalsLeft, topLevel: true)
            } else {
                out += "\"-\" ("
                out += minMaxInt(-maxValue, Int64.max,
                                 decimalsLeft: decimalsLeft, topLevel: false)
                out += ")"
            }
            return out
        }

        // 参照実装はここで例外を投げるが、呼び出し側が「片方はある」ことを
        // 確かめてから来るので到達しない (プログラミング誤りのみ)。
        return out
    }

    static func integerBound(_ value: JSONValue?) -> Int64? {
        switch value {
        case .integer(let value): return value
        case .unsignedInteger(let value): return Int64(exactly: value)
        default: return nil
        }
    }
}
