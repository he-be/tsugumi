import Foundation

/// A long page as numbered sections the model picks from (docs/qwen38/24).
///
/// Before this, `fetch_page` and `wikipedia_page` handed back the first 6,000 characters and a `from` to read on:
/// the only way to the part a question needed was to read everything before it, one clip per round (the iPhone 18
/// chat read 11,899 of 15,923 characters, 7,100 tokens, to answer from what the first clip already held). Now the
/// first read shows the outline and the opening sections, and the model asks for the sections it wants by number.
///
/// Sections come from the page's headings. Text without them — an article in the local Wikipedia index is one
/// run of plain text, and many pages mark none — is cut into parts at a line or sentence end, each named by its
/// first words. Headings with no text of their own join the next section's name; a section too short to be worth
/// a line of the outline joins the next; a long one is cut into parts. Every section fits in one read.
public struct PageOutline: Equatable, Sendable {
    public struct Section: Equatable, Sendable {
        /// The name the outline shows; empty for text before the first heading and for a part of unheaded text.
        public var heading: String
        public var text: String
        /// One of the parts a long section was cut into: the outline adds its opening words, since "XML (7/19)"
        /// says nothing about which keys the part holds.
        public var isPart: Bool

        public init(heading: String, text: String, isPart: Bool = false) {
            self.heading = heading
            self.text = text
            self.isPart = isPart
        }
    }

    public var sections: [Section]
    public var totalCharacters: Int

    /// A part's size before a page is long enough to need bigger ones.
    static let partCharacters = 1_500
    /// A section shorter than this joins the next one.
    static let mergeBelowCharacters = 150
    /// Past this many sections the parts grow and neighbours merge: the outline itself is read every time.
    static let maxSections = 40
    static let labelCharacters = 60
    static let unheadedLabelCharacters = 30
    static let partOpeningCharacters = 24
    /// How much of the opening the first read shows below the outline (at least the first section).
    static let leadCharacters = 1_500
    /// A page this short comes back whole: an outline would cost more than it saves.
    static let wholePageCharacters = 3_000

    /// `lines` is the page one block per line; `headingLines` the indices of the lines that are headings.
    /// `pageLimit` is the most one read returns, so no section is made longer.
    public init(lines: [String], headingLines: Set<Int> = [], pageLimit: Int) {
        let total = lines.reduce(0) { $0 + $1.count } + max(lines.count - 1, 0)
        totalCharacters = total
        let part = min(max(Self.partCharacters, (total + Self.maxSections - 1) / Self.maxSections),
                       max(pageLimit * 3 / 4, 1))

        var collected: [Section] = []
        var heading = ""
        var pending: [String] = []
        var body: [String] = []
        func close() {
            if !body.isEmpty { collected.append(Section(heading: heading, text: body.joined(separator: "\n"))) }
            body = []
        }
        for (index, line) in lines.enumerated() {
            if headingLines.contains(index) {
                pending.append(line)
                continue
            }
            if !pending.isEmpty {
                close()
                heading = pending.joined(separator: " / ")
                pending = []
            }
            body.append(line)
        }
        close()

        // Short sections join the next one; the next one's heading stays in the text where the page had it.
        var merged: [Section] = []
        var carry: Section?
        for section in collected {
            guard let held = carry else {
                if section.text.count < Self.mergeBelowCharacters { carry = section } else { merged.append(section) }
                continue
            }
            let joined = Section(heading: Self.join(held.heading, section.heading),
                                 text: held.text + "\n" + (section.heading.isEmpty ? "" : section.heading + "\n")
                                     + section.text)
            if joined.text.count < Self.mergeBelowCharacters {
                carry = joined
            } else {
                merged.append(joined)
                carry = nil
            }
        }
        if let held = carry {
            if var last = merged.popLast() {
                last.text += "\n" + (held.heading.isEmpty ? "" : held.heading + "\n") + held.text
                merged.append(last)
            } else {
                merged.append(held)
            }
        }

        // Long sections are cut into parts that each fit in a read.
        var cut: [Section] = []
        for section in merged {
            guard section.text.count > part * 4 / 3 else {
                cut.append(section)
                continue
            }
            let pieces = Self.split(section.text, size: part)
            for (index, piece) in pieces.enumerated() {
                let name = section.heading.isEmpty ? "" : "\(section.heading) (\(index + 1)/\(pieces.count))"
                cut.append(Section(heading: name, text: piece, isPart: true))
            }
        }

        // Too many sections: merge the smallest neighbouring pair that still fits in a read.
        while cut.count > Self.maxSections {
            var best: Int?
            for index in 0..<(cut.count - 1) {
                let size = cut[index].text.count + cut[index + 1].text.count
                guard size + 1 <= pageLimit else { continue }
                if best.map({ size < cut[$0].text.count + cut[$0 + 1].text.count }) ?? true { best = index }
            }
            guard let index = best else { break }
            let next = cut.remove(at: index + 1)
            cut[index] = Section(heading: Self.join(cut[index].heading, next.heading),
                                 text: cut[index].text + "\n" + (next.heading.isEmpty ? "" : next.heading + "\n")
                                     + next.text, isPart: cut[index].isPart)
        }
        sections = cut
    }

    /// Plain text with no headings (a Wikipedia article): parts only.
    public init(text: String, pageLimit: Int) {
        self.init(lines: text.split(separator: "\n", omittingEmptySubsequences: true).map(String.init),
                  pageLimit: pageLimit)
    }

    /// Whether the page is short enough to hand back whole, without an outline.
    public func isWhole(limit: Int) -> Bool {
        totalCharacters <= min(Self.wholePageCharacters, limit) || sections.count <= 1
    }

    /// The name the outline shows for section `index` (0-based).
    public func label(_ index: Int) -> String {
        let section = sections[index]
        let first = String(section.text.prefix { $0 != "\n" })
        guard !section.heading.isEmpty else { return Self.clip(first, to: Self.unheadedLabelCharacters) }
        let name = Self.clip(section.heading, to: Self.labelCharacters)
        return section.isPart ? "\(name) 「\(Self.clip(first, to: Self.partOpeningCharacters))」" : name
    }

    /// "[3] 価格 (1210字)"
    public func entry(_ index: Int) -> String {
        "[\(index + 1)] \(label(index)) (\(sections[index].text.count)字)"
    }

    /// How many sections from the top the first read shows.
    func leadCount(limit: Int) -> Int {
        var count = 0
        var size = 0
        for section in sections {
            if count > 0, size + section.text.count > min(Self.leadCharacters, limit) { break }
            size += section.text.count
            count += 1
        }
        return count
    }

    /// The first read: what the page is, the outline, and the opening sections.
    public func overview(tool: String, limit: Int) -> (lines: [String], shown: Int) {
        let lead = leadCount(limit: limit)
        let opening = lead == 1 ? "節 1" : "節 1〜\(lead)"
        var lines = ["本文は全 \(totalCharacters) 文字、\(sections.count) 節。目次と\(opening) の本文です。"
                     + "他の節は \(tool) の sections に番号を渡すと読めます (1 回 \(limit) 文字まで)。", "", "目次:"]
        lines.append(contentsOf: sections.indices.map(entry))
        for index in 0..<lead {
            lines.append("")
            lines.append(contentsOf: body(index, limit: limit))
        }
        if lead < sections.count {
            lines.append("")
            lines.append("(次の節: \(entry(lead)))")
        }
        return (lines, lead)
    }

    /// A read of the sections `requested` (1-based, as the outline numbers them), in page order, as many as fit
    /// in `limit`; the ones left out and the numbers that name nothing are said.
    public func read(_ requested: [Int], tool: String, limit: Int) -> (lines: [String], shown: [Int], isError: Bool) {
        let wanted = Array(Set(requested)).sorted()
        let valid = wanted.filter { (1...sections.count).contains($0) }
        let invalid = wanted.filter { !(1...sections.count).contains($0) }
        guard !valid.isEmpty else {
            var lines = ["節 \(Self.numbers(invalid)) はありません。節は 1〜\(sections.count) です。", "", "目次:"]
            lines.append(contentsOf: sections.indices.map(entry))
            return (lines, [], true)
        }
        var shown: [Int] = []
        var skipped: [Int] = []
        var size = 0
        for number in valid {
            let length = sections[number - 1].text.count
            if !shown.isEmpty, size + length > limit {
                skipped.append(number)
                continue
            }
            size += length
            shown.append(number)
        }
        var lines = ["節 \(Self.numbers(shown)) (全 \(sections.count) 節、本文は全 \(totalCharacters) 文字)"]
        for number in shown {
            lines.append("")
            lines.append(contentsOf: body(number - 1, limit: limit))
        }
        lines.append("")
        if !invalid.isEmpty {
            lines.append("(節 \(Self.numbers(invalid)) はありません。節は 1〜\(sections.count) です)")
        }
        if !skipped.isEmpty {
            lines.append("(節 \(Self.numbers(skipped)) は 1 回 \(limit) 文字の上限を超えるので出していません。"
                         + "\(tool) の sections に渡すと読めます)")
        }
        if let last = shown.last, last < sections.count, !valid.contains(last + 1) {
            lines.append("(次の節: \(entry(last)))")
        } else if shown.last == sections.count, skipped.isEmpty {
            lines.append("(これが最後の節です)")
        }
        return (lines, shown, false)
    }

    /// One section as a read shows it: its number and name, then its text.
    func body(_ index: Int, limit: Int) -> [String] {
        let (text, clipped) = HTMLTextExtractor.clip(sections[index].text, to: limit)
        var lines = ["[\(index + 1)] \(label(index))", text]
        if clipped { lines.append("…(節 \(index + 1) はここで打ち切り)") }
        return lines
    }

    /// "2, 3, 5"
    static func numbers(_ values: [Int]) -> String {
        values.map(String.init).joined(separator: ", ")
    }

    static func join(_ first: String, _ second: String) -> String {
        [first, second].filter { !$0.isEmpty }.joined(separator: " / ")
    }

    static func clip(_ text: String, to limit: Int) -> String {
        text.count > limit ? String(text.prefix(limit)) + "…" : text
    }

    /// `text` in parts of about `size` characters, cut at a line end, else a sentence end, else a space, in the last
    /// third of each window; the last part takes the tail rather than leaving a scrap of it.
    static func split(_ text: String, size: Int) -> [String] {
        let characters = Array(text)
        let sentenceEnds: Set<Character> = ["。", "．", "！", "？", "!", "?"]
        var parts: [String] = []
        var start = 0
        while start < characters.count {
            if characters.count - start <= size * 4 / 3 {
                parts.append(String(characters[start...]))
                break
            }
            let limit = start + size
            let window = (start + size * 2 / 3)..<limit
            var cut = limit
            if let index = window.reversed().first(where: { characters[$0] == "\n" }) {
                cut = index + 1
            } else if let index = window.reversed().first(where: {
                sentenceEnds.contains(characters[$0])
                    || (characters[$0] == "." && $0 + 1 < characters.count && characters[$0 + 1] == " ")
            }) {
                cut = index + 1
            } else if let index = window.reversed().first(where: { characters[$0] == " " }) {
                cut = index + 1
            }
            parts.append(String(characters[start..<cut]))
            start = cut
        }
        return parts.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
    }
}
