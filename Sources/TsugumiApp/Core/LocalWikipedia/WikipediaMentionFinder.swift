import Foundation
import NaturalLanguage

/// The strings in a prompt that could be article names, and the rule that
/// keeps the ones that are.
///
/// Cutting a Japanese prompt into candidate spans needs a word boundary;
/// `NLTokenizer` gives one (「淀城の遺構が」→ 淀 / 城 / の / 遺構 / が).
/// Every window of up to `maxWords` words is a candidate, longest first,
/// so 「城崎シーワールド」 is tried before 「城崎」. Whether a candidate is an
/// article is the index's business (`LocalWikipediaIndex.mentions`); whether
/// an article is worth mentioning is `keeps`, an entity-linking rule of
/// thumb (Milne & Witten 2008, TagMe): a string that is usually a link when
/// it appears in Wikipedia (淀城, えきねっと) names a thing, one that is
/// almost never a link (ニュース, 工事) is a common noun that happens to
/// have an article. The index has no anchor counts, so the probability is
/// approximated as incoming links to the article over documents whose
/// head contains the string. It separates the two well; the numbers and
/// thresholds come from the app's own chat history (docs/LOCAL_WIKIPEDIA.md).
public enum WikipediaMentionFinder {
    public struct Candidate: Equatable, Sendable {
        public var text: String
        /// Which words of the prompt the candidate covers.
        public var words: Range<Int>
        /// The candidate starts with a non-CJK letter (an acronym, a brand).
        public var isLatin: Bool

        public var wordCount: Int { words.count }
    }

    public static let maxWords = 4

    /// The prompt's words, URLs removed first (a commit hash or a domain
    /// cut into "cff" and "ai" would match articles).
    public static func words(in text: String) -> [String] {
        let stripped = stripURLs(text)
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.setLanguage(.japanese)
        tokenizer.string = stripped
        var words: [String] = []
        tokenizer.enumerateTokens(in: stripped.startIndex..<stripped.endIndex) { range, _ in
            let word = stripped[range].trimmingCharacters(in: .whitespacesAndNewlines)
            if !word.isEmpty { words.append(word) }
            return true
        }
        return words
    }

    static func stripURLs(_ text: String) -> String {
        text.replacingOccurrences(of: #"[a-z][a-z0-9+.-]*://\S+"#, with: " ",
                                  options: [.regularExpression, .caseInsensitive])
    }

    /// Every window of 1...`maxWords` words, longest windows first. Windows
    /// shorter than two characters or without a letter (a year, a number)
    /// are not article names worth a lookup.
    public static func candidates(in words: [String]) -> [Candidate] {
        guard !words.isEmpty else { return [] }
        var out: [Candidate] = []
        for length in stride(from: min(maxWords, words.count), through: 1, by: -1) {
            for start in 0...(words.count - length) {
                let range = start..<(start + length)
                let text = join(words[range])
                guard text.count >= 2, text.contains(where: \.isLetter) else { continue }
                out.append(Candidate(text: text, words: range, isLatin: !startsWithCJK(text)))
            }
        }
        return out
    }

    /// Words back into a string the way the title would be written: no
    /// space where either side is CJK (城崎 + シーワールド), a space between
    /// two Latin words (Mac + mini).
    static func join(_ words: ArraySlice<String>) -> String {
        var text = ""
        var previous: String?
        for word in words {
            if let previous, !(endsWithCJK(previous) || startsWithCJK(word)) { text += " " }
            text += word
            previous = word
        }
        return text
    }

    static func startsWithCJK(_ text: String) -> Bool {
        text.unicodeScalars.first.map { WikipediaTokenizer.isCJK($0.value) } ?? false
    }

    static func endsWithCJK(_ text: String) -> Bool {
        text.unicodeScalars.last.map { WikipediaTokenizer.isCJK($0.value) } ?? false
    }

    /// Whether an article the prompt names is worth showing the model.
    /// A string that is linked at least as often as it is written is a
    /// name whatever its shape (米国, 任天堂, ベネズエラ). Below that, a
    /// single CJK word is a common noun (遺構 0.33, クーデター 0.57, ソニー 0.30
    /// are left out with 天気 and 工事); a CJK compound of several words
    /// keeps a lower bar (熊本地震 0.17); a Latin word a middle one (IBM
    /// 0.99 and GitHub 1.3 in, granite the rock at 0.12 out).
    public static func keeps(linkProbability: Double, wordCount: Int, isLatin: Bool) -> Bool {
        if linkProbability >= 1.0 { return true }
        if isLatin { return linkProbability >= 0.5 }
        return wordCount >= 2 && linkProbability >= 0.1
    }
}
