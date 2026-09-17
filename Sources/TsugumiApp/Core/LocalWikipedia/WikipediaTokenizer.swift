import Foundation

/// The way the local Wikipedia index cuts text into terms. The index is
/// built by `Scripts/wiki/build_jawiki_index.py`, which pre-tokenizes every
/// column with the same rules before FTS5 (`unicode61`) sees it; the query
/// side has to cut identically or nothing matches. Both sides tokenize
/// `checkSample` and the index stores its answer, so a drift is caught when
/// the file is opened rather than as silently empty searches.
///
/// Rules: NFKC, lowercase; a run of CJK characters or digits becomes its
/// character bigrams (a lone character stays as itself); a run of other
/// letters and digits is one word; everything else separates. Bigrams are
/// what makes two-character Japanese words (米国, 戦争, 首相) findable —
/// FTS5's own trigram tokenizer cannot match them at all.
public enum WikipediaTokenizer {
    public static let version = "bigram-2"
    public static let checkSample = "東京タワーは2026年9月1日に iPhone 16 Pro で撮った。Ａ１ ｱｲｳ 々 ー A・B M4"

    /// How loosely a query is matched: every term, any term, or any single
    /// bigram of any term. `search` tries them in that order.
    public enum MatchMode: Sendable, CaseIterable {
        case allTerms
        case anyTerm
        case anyToken
    }

    public static func tokenize(_ text: String) -> [String] {
        let folded = text.precomposedStringWithCompatibilityMapping.lowercased()
        var tokens: [String] = []
        var run: [Unicode.Scalar] = []
        var runKind = 0

        func flush() {
            guard !run.isEmpty else { return }
            if runKind == 1 {
                tokens.append(String(String.UnicodeScalarView(run)))
            } else if run.count == 1 {
                tokens.append(String(run[0]))
            } else {
                for index in 0..<(run.count - 1) {
                    var view = String.UnicodeScalarView()
                    view.append(run[index])
                    view.append(run[index + 1])
                    tokens.append(String(view))
                }
            }
            run.removeAll(keepingCapacity: true)
        }

        for scalar in folded.unicodeScalars {
            let k = kind(scalar)
            if k != runKind {
                flush()
                runKind = k
            }
            if k != 0 { run.append(scalar) }
        }
        flush()
        return tokens
    }

    /// The FTS5 MATCH expression for a query: each whitespace-separated
    /// term becomes a phrase of its bigrams, joined with AND, or OR for the
    /// looser retries (the loosest ORs the bigrams one by one). Nil when
    /// the query has no indexable character.
    public static func matchExpression(_ query: String, mode: MatchMode) -> String? {
        var phrases: [String] = []
        for term in query.split(whereSeparator: \.isWhitespace) {
            let tokens = tokenize(String(term))
            guard !tokens.isEmpty else { continue }
            if mode == .anyToken {
                phrases.append(contentsOf: tokens.map { "\"" + $0 + "\"" })
            } else {
                phrases.append("\"" + tokens.joined(separator: " ") + "\"")
            }
        }
        var seen = Set<String>()
        phrases = phrases.filter { seen.insert($0).inserted }
        guard !phrases.isEmpty else { return nil }
        return phrases.joined(separator: mode == .allTerms ? " AND " : " OR ")
    }

    /// The key the `titles` table is looked up by: NFKC, underscores to
    /// spaces, runs of whitespace collapsed, case-folded.
    public static func normalizeTitle(_ title: String) -> String {
        title.precomposedStringWithCompatibilityMapping
            .replacingOccurrences(of: "_", with: " ")
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
            .lowercased()
    }

    /// 2 = CJK or digit (bigrams), 1 = other letter/number (one word), 0 = separator.
    /// Mirrors `_kind` in the build script: Python's `isdigit` is the Decimal
    /// and Digit numeric types, `isalnum` the L* and N* categories.
    static func kind(_ scalar: Unicode.Scalar) -> Int {
        let properties = scalar.properties
        switch properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter,
             .decimalNumber, .letterNumber, .otherNumber:
            break
        default:
            // Punctuation inside the CJK blocks (the middle dot) separates.
            return 0
        }
        if isCJK(scalar.value) { return 2 }
        if let numeric = properties.numericType, numeric == .decimal || numeric == .digit { return 2 }
        return 1
    }

    static func isCJK(_ value: UInt32) -> Bool {
        switch value {
        case 0x3040...0x30FF, 0x31F0...0x31FF, 0x3400...0x4DBF, 0x4E00...0x9FFF,
             0xF900...0xFAFF, 0x20000...0x2FFFF, 0xAC00...0xD7AF,
             0x3005, 0x3006, 0x3007:
            return true
        default:
            return false
        }
    }
}
