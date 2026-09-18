import Foundation

/// The app's half of a turn where the model does not read Wikipedia itself (docs/qwen38/40): the model writes search
/// terms, and this runs them, cuts the found articles into short chunks, and hands back the chunks closest to the
/// question and the terms, up to a fixed number of characters. The model then either answers from them or asks for
/// more terms; it never opens an article or picks a section.
///
/// One instance per turn: the chunks already handed over are not repeated, and each article is embedded once.
public final class WikipediaGatherer: @unchecked Sendable {
    /// A piece of one section: consecutive lines packed up to `chunkCharacters`, a longer line cut at "。".
    public struct Chunk: Equatable, Sendable {
        public var title: String
        public var section: Int
        public var sectionCount: Int
        public var label: String
        public var text: String
    }

    public struct Gathered: Sendable {
        /// What goes into the conversation.
        public var text: String
        public var articles: [String]
        public var chunksScored: Int
        public var chunksPicked: Int
        public var characters: Int
        public var embedSeconds: Double
    }

    let index: LocalWikipediaIndex
    let embedder: any SectionEmbedding
    let maxArticles: Int
    let characterBudget: Int
    let chunkCharacters: Int
    let pageCharacterLimit: Int
    private let question: String
    private var shown: Set<String> = []
    private var cache: [Int: (chunks: [Chunk], vectors: [[Float]])] = [:]

    public init(index: LocalWikipediaIndex, embedder: any SectionEmbedding, question: String,
                maxArticles: Int = 8, characterBudget: Int = 3_000, chunkCharacters: Int = 300,
                pageCharacterLimit: Int = 6_000) {
        self.index = index
        self.embedder = embedder
        self.question = question.trimmingCharacters(in: .whitespacesAndNewlines)
        self.maxArticles = max(1, maxArticles)
        self.characterBudget = max(1, characterBudget)
        self.chunkCharacters = max(20, chunkCharacters)
        self.pageCharacterLimit = pageCharacterLimit
    }

    /// The articles the terms find, taken in turn from each term's hits (1st of each, then 2nd of each, …) until
    /// `maxArticles`, without repeats.
    func articles(for terms: [String]) -> [LocalWikipediaIndex.Hit] {
        let lists = terms.map { index.search($0, limit: maxArticles) }
        var seen: Set<Int> = []
        var picked: [LocalWikipediaIndex.Hit] = []
        for rank in 0..<(lists.map(\.count).max() ?? 0) {
            for list in lists where rank < list.count {
                let hit = list[rank]
                if picked.count < maxArticles, seen.insert(hit.pageID).inserted { picked.append(hit) }
            }
        }
        return picked
    }

    /// A section's text as chunks of at most `size` characters: lines packed in order, a line over `size` split
    /// after "。" (and cut hard if one sentence is still over).
    static func chunks(of text: String, size: Int) -> [String] {
        var pieces: [String] = []
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = line.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }
            if line.count <= size { pieces.append(line); continue }
            var sentence = ""
            for character in line {
                sentence.append(character)
                if character == "。" { pieces.append(sentence); sentence = "" }
            }
            if !sentence.isEmpty { pieces.append(sentence) }
        }
        var out: [String] = []
        var current = ""
        for piece in pieces.flatMap({ hardCut($0, size: size) }) {
            if current.isEmpty {
                current = piece
            } else if current.count + 1 + piece.count <= size {
                current += "\n" + piece
            } else {
                out.append(current)
                current = piece
            }
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    private static func hardCut(_ text: String, size: Int) -> [String] {
        guard text.count > size else { return [text] }
        var out: [String] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            out.append(String(rest.prefix(size)))
            rest = rest.dropFirst(size)
        }
        return out
    }

    private func article(_ hit: LocalWikipediaIndex.Hit) throws -> (chunks: [Chunk], vectors: [[Float]])? {
        if let cached = cache[hit.pageID] { return cached }
        guard let page = index.page(id: hit.pageID) else { return nil }
        let outline = PageOutline(text: page.text, pageLimit: pageCharacterLimit)
        var chunks: [Chunk] = []
        for (number, section) in outline.sections.enumerated() {
            for text in Self.chunks(of: section.text, size: chunkCharacters) {
                chunks.append(Chunk(title: page.title, section: number + 1, sectionCount: outline.sections.count,
                                    label: outline.label(number), text: text))
            }
        }
        let vectors = try embedder.embed(documents: chunks.map { chunk in
            let heading = chunk.label.isEmpty ? chunk.title : chunk.title + " " + chunk.label
            return heading + "\n" + chunk.text
        })
        cache[hit.pageID] = (chunks, vectors)
        return (chunks, vectors)
    }

    /// A chunk with its score: the best cosine against the question and each term (`targets` in that order).
    public struct Scored: Sendable {
        public var chunk: Chunk
        public var score: Float
        public var targets: [Float]
    }

    /// The found articles, and their chunks not yet handed over, closest first (ties keep article, then page order).
    /// `extraPages` (by id) are scored as if found too; the tool loop never passes any (`--gather-probe`,
    /// docs/qwen38/40 §6).
    public func ranked(terms: [String], extraPages: [Int] = []) throws
        -> (articles: [LocalWikipediaIndex.Hit], chunks: [Scored]) {
        var hits = articles(for: terms)
        for id in extraPages where !hits.contains(where: { $0.pageID == id }) {
            if let page = index.page(id: id) {
                hits.append(LocalWikipediaIndex.Hit(pageID: id, title: page.title, snippet: "", incomingLinks: 0))
            }
        }
        let targets = try ([question] + terms).filter { !$0.isEmpty }.map { try embedder.embed(query: $0) }
        var scored: [Scored] = []
        for hit in hits {
            guard let (chunks, vectors) = try article(hit) else { continue }
            for (chunk, vector) in zip(chunks, vectors) where !shown.contains(Self.key(chunk)) {
                let each = targets.map { target in zip(target, vector).reduce(Float(0)) { $0 + $1.0 * $1.1 } }
                scored.append(Scored(chunk: chunk, score: each.max() ?? 0, targets: each))
            }
        }
        let order = scored.indices.sorted { a, b in
            scored[a].score != scored[b].score ? scored[a].score > scored[b].score : a < b
        }
        return (hits, order.map { scored[$0] })
    }

    /// Runs `terms`, and hands back the chunks not yet handed over, closest first by their best cosine against the
    /// question and each term, as many as fit in `characterBudget` (at least one).
    public func gather(terms: [String], dateStamp: String) throws -> Gathered {
        let started = Date()
        let (hits, candidates) = try ranked(terms: terms)
        var picked: [Chunk] = []
        var size = 0
        for candidate in candidates {
            let length = candidate.chunk.text.count
            if !picked.isEmpty, size + length > characterBudget { break }
            picked.append(candidate.chunk)
            size += length
        }
        for chunk in picked { shown.insert(Self.key(chunk)) }

        var lines = ["Wikipedia (\(dateStamp)) を「\(terms.joined(separator: "」「"))」で検索し、"
                     + "見つかった記事から質問と検索語に近い段落を選びました (近い順):"]
        if picked.isEmpty { lines.append("(新しい段落はありません)") }
        for chunk in picked {
            lines.append("")
            lines.append("[\(chunk.title) · 節 \(chunk.section)/\(chunk.sectionCount)] \(chunk.label)")
            lines.append(chunk.text)
        }
        lines.append("")
        lines.append("見つかった記事: " + (hits.isEmpty ? "なし" : hits.map(\.title).joined(separator: " / ")))
        return Gathered(text: lines.joined(separator: "\n"), articles: hits.map(\.title),
                        chunksScored: candidates.count, chunksPicked: picked.count, characters: size,
                        embedSeconds: Date().timeIntervalSince(started))
    }

    static func key(_ chunk: Chunk) -> String { "\(chunk.title)\u{1F}\(chunk.section)\u{1F}\(chunk.text)" }
}
