import Foundation
import Synchronization

/// The two tools the chat declares when a local Wikipedia index is
/// configured. `wikipedia_search` runs the full-text search and hands back
/// titles with a line of each opening; `wikipedia_page` returns one
/// article: a short one whole, a long one as its outline and opening parts,
/// with `sections` to read the parts the model picks (`PageOutline`; the
/// index keeps no headings, so the parts are cut from the plain text).
/// Nothing here touches the network: the index is a file on this Mac.
public struct WikipediaToolExecutor: AppToolExecutor {
    public static let searchToolName = "wikipedia_search"
    public static let pageToolName = "wikipedia_page"

    let index: LocalWikipediaIndex
    let maxResults: Int
    let pageCharacterLimit: Int
    /// With one, every search result carries the parts of the found articles closest to the user's question
    /// instead of the top article's opening (R1-d, docs/qwen38/39).
    let sectionEmbedder: (any SectionEmbedding)?
    let nearState = NearSectionState()

    public init(index: LocalWikipediaIndex, maxResults: Int, pageCharacterLimit: Int,
                sectionEmbedder: (any SectionEmbedding)? = nil) {
        self.index = index
        self.maxResults = max(1, maxResults)
        self.pageCharacterLimit = max(500, pageCharacterLimit)
        self.sectionEmbedder = sectionEmbedder
    }

    /// The question the near sections are picked for, and the vectors of the articles already cut this turn.
    final class NearSectionState: Sendable {
        let question = Mutex("")
        let vectors = Mutex<[Int: [[Float]]]>([:])
    }

    /// Nothing to add before the first round; the user's question is kept for the near sections.
    public func lookups(prompt: String, callIDPrefix: String) async -> [AppToolLookup] {
        if sectionEmbedder != nil {
            nearState.question.withLock { $0 = prompt }
            nearState.vectors.withLock { $0 = [:] }
        }
        return []
    }

    public var promptFacts: AppToolPromptFacts {
        AppToolPromptFacts(web: false, wikipediaDate: index.summary.dumpDateJapanese ?? "")
    }

    public var definitions: [AppToolDefinition] {
        let stamp = index.summary.dumpDateJapanese.map { "、\($0) 時点" } ?? ""
        return [
            AppToolDefinition(
                name: Self.searchToolName,
                description: "この Mac に保存された日本語版 Wikipedia (オフライン\(stamp)) を検索して、該当する記事の題名と導入部を返す。人物・組織・地名・作品・用語・歴史・科学など、百科事典にある事柄を調べるときに使う。記事名そのものか、それに近い語で検索する。",
                parametersJSON: #"{"type":"object","properties":{"query":{"type":"string","description":"検索語。記事名か、固有名詞と要点を短く並べる。"}},"required":["query"]}"#),
            AppToolDefinition(
                name: Self.pageToolName,
                description: "日本語版 Wikipedia の記事を 1 つ開いて本文を返す。wikipedia_search の結果の題名をそのまま渡す。長い記事は目次 (番号つきの節) と冒頭の節を返すので、質問に必要な節の番号を sections に渡して読む。",
                parametersJSON: #"{"type":"object","properties":{"title":{"type":"string","description":"記事の題名。"},"sections":{"type":"string","description":"読む節の番号 (目次の [ ] の数字)。1 つか、2,3 のようにカンマで区切って並べる。省略すると目次と冒頭の節を返す。"}},"required":["title"]}"#),
        ]
    }

    public func execute(_ call: AppToolCall) async -> AppToolResult {
        switch call.name {
        case Self.searchToolName:
            guard let query = call.stringArgument("query")?
                .trimmingCharacters(in: .whitespacesAndNewlines), !query.isEmpty else {
                return AppToolResult(content: "error: wikipedia_search needs a non-empty \"query\".",
                                     isError: true, summary: "missing query")
            }
            return search(query)
        case Self.pageToolName:
            guard let title = call.stringArgument("title")?
                .trimmingCharacters(in: .whitespacesAndNewlines), !title.isEmpty else {
                return AppToolResult(content: "error: wikipedia_page needs a non-empty \"title\".",
                                     isError: true, summary: "missing title")
            }
            return page(title, sections: call.integerListArgument("sections"))
        default:
            return AppToolResult(content: "error: unknown tool \(call.name).",
                                 isError: true, summary: "unknown tool")
        }
    }

    public func subject(of call: AppToolCall) -> String {
        switch call.name {
        case Self.searchToolName: call.stringArgument("query") ?? call.argumentsJSON
        case Self.pageToolName: call.stringArgument("title") ?? call.argumentsJSON
        default: call.argumentsJSON
        }
    }

    /// What every result says about the copy it comes from, so the model
    /// reads a dated encyclopedia rather than a live one (the system prompt
    /// says the same; the result is what it actually quotes from).
    var dateStamp: String {
        index.summary.dumpDateJapanese.map { "\($0) 時点" } ?? "日付不明"
    }

    /// Wikipedia's own search box "goes" straight to the article when the
    /// query is a title; this does the same inside the result. When the
    /// first hit is a title or redirect match, or the query is one term (the
    /// model named an article rather than describing one), the article's
    /// text follows the list in the same result — no second round, and no
    /// decision the model has to make to read what it asked for. The list
    /// still comes first so a wrong guess can be corrected.
    static func shouldGo(_ hits: [LocalWikipediaIndex.Hit], query: String) -> Bool {
        guard let first = hits.first else { return false }
        return first.isExactTitle || !query.contains(where: \.isWhitespace)
    }

    func search(_ query: String) -> AppToolResult {
        let hits = index.search(query, limit: maxResults)
        guard !hits.isEmpty else {
            return AppToolResult(
                content: "Wikipedia 検索: \(query) (\(dateStamp)) — 該当する記事はありません。別の語や記事名で検索してください。",
                summary: "Wikipedia · 0 hits")
        }
        var lines = ["Wikipedia 検索: \(query) (\(hits.count) 件、\(dateStamp)の複製)"]
        for (number, hit) in hits.enumerated() {
            lines.append("[\(number + 1)] \(hit.title)")
            if !hit.snippet.isEmpty { lines.append("    \(hit.snippet)") }
        }
        lines.append("")
        var summary = "Wikipedia · \(hits.count) hits"
        if let near = nearSections(hits, query: query) {
            lines.append(contentsOf: near.lines)
            lines.append("")
            lines.append("同じ記事の他の節は、wikipedia_page に題名と sections (節の番号) を渡すと読めます。")
            summary += near.summary
        } else if Self.shouldGo(hits, query: query), let page = index.page(id: hits[0].pageID) {
            let body = Self.body(of: page, sections: nil, limit: pageCharacterLimit)
            lines.append("[1] \(page.title) の本文:")
            lines.append(contentsOf: body.lines)
            lines.append("")
            lines.append("他の記事を読むには wikipedia_page に題名を渡します。")
            summary += " + \(page.title) \(body.summary)"
        } else {
            lines.append("本文を読むには wikipedia_page に題名を渡します。")
        }
        return AppToolResult(content: lines.joined(separator: "\n"), summary: summary)
    }

    /// The sections of the found articles closest to the user's question (the search query when there is none),
    /// closest first, as many as fit in one read's limit (at least one, clipped). Nil without an embedder.
    func nearSections(_ hits: [LocalWikipediaIndex.Hit], query: String) -> (lines: [String], summary: String)? {
        guard let embedder = sectionEmbedder else { return nil }
        let started = Date()
        let kept = nearState.question.withLock { $0 }.trimmingCharacters(in: .whitespacesAndNewlines)
        let question = kept.isEmpty ? query : kept
        struct Candidate { let order: Int; let title: String; let outline: PageOutline; let section: Int; let score: Float }
        var candidates: [Candidate] = []
        do {
            let target = try embedder.embed(query: question)
            for hit in hits {
                guard let page = index.page(id: hit.pageID) else { continue }
                let outline = PageOutline(text: page.text, pageLimit: pageCharacterLimit)
                let vectors: [[Float]]
                if let cached = nearState.vectors.withLock({ $0[page.pageID] }) {
                    vectors = cached
                } else {
                    vectors = try embedder.embed(documents: outline.sections.map {
                        $0.heading.isEmpty ? $0.text : $0.heading + "\n" + $0.text
                    })
                    nearState.vectors.withLock { $0[page.pageID] = vectors }
                }
                for (section, vector) in vectors.enumerated() {
                    let score = zip(target, vector).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                    candidates.append(Candidate(order: candidates.count, title: page.title, outline: outline,
                                                section: section, score: score))
                }
            }
        } catch {
            return (["(質問に近い節は選べませんでした)"], " + near failed: \(error)")
        }
        guard !candidates.isEmpty else { return nil }
        // Ties keep search order, then page order.
        candidates.sort { $0.score != $1.score ? $0.score > $1.score : $0.order < $1.order }
        var picked: [Candidate] = []
        var size = 0
        for candidate in candidates {
            let length = candidate.outline.sections[candidate.section].text.count
            if !picked.isEmpty, size + length > pageCharacterLimit { break }
            picked.append(candidate)
            size += length
        }
        var lines = ["質問に近い節 (検索結果の記事から、アプリが質問との近さで選んだもの。近い順):"]
        for candidate in picked {
            let outline = candidate.outline
            let (text, clipped) = HTMLTextExtractor.clip(outline.sections[candidate.section].text, to: pageCharacterLimit)
            lines.append("")
            lines.append("[\(candidate.title) · 節 \(candidate.section + 1)/\(outline.sections.count)] "
                         + outline.label(candidate.section))
            lines.append(text)
            if clipped { lines.append("…(節 \(candidate.section + 1) はここで打ち切り)") }
        }
        let seconds = String(format: "%.2f", Date().timeIntervalSince(started))
        return (lines, " + near \(picked.count) of \(candidates.count) sections (\(seconds) s)")
    }

    func page(_ title: String, sections: [Int]?) -> AppToolResult {
        guard let page = index.page(title: title) else {
            let near = index.search(title, limit: 5)
            var lines = ["Wikipedia に「\(title)」という記事はありません。"]
            if !near.isEmpty {
                lines.append("近い題名: " + near.map(\.title).joined(separator: " / "))
                lines.append("この中の題名をそのまま wikipedia_page に渡してください。")
            }
            return AppToolResult(content: lines.joined(separator: "\n"),
                                 isError: true, summary: "Wikipedia · not found")
        }
        let body = Self.body(of: page, sections: sections, limit: pageCharacterLimit)
        var lines = ["Wikipedia 記事: \(page.title) (\(dateStamp))", ""]
        lines.append(contentsOf: body.lines)
        return AppToolResult(content: lines.joined(separator: "\n"), isError: body.isError,
                             summary: "Wikipedia · \(body.summary)")
    }

    /// The article as a read shows it: whole when short, else the outline and opening parts, or the parts
    /// `sections` names.
    static func body(of page: LocalWikipediaIndex.Page, sections: [Int]?, limit: Int)
        -> (lines: [String], summary: String, isError: Bool) {
        let outline = PageOutline(text: page.text, pageLimit: limit)
        let total = outline.totalCharacters.formatted()
        if outline.isWhole(limit: limit) {
            let (text, clipped) = HTMLTextExtractor.clip(page.text, to: limit)
            return ([text], "\(total) chars\(clipped ? " (clipped)" : "")", false)
        }
        guard let sections, !sections.isEmpty else {
            return (outline.overview(tool: pageToolName, limit: limit).lines,
                    "\(total) chars · \(outline.sections.count) sections", false)
        }
        let read = outline.read(sections, tool: pageToolName, limit: limit)
        let shown = read.shown.isEmpty ? "none" : PageOutline.numbers(read.shown)
        return (read.lines, "sections \(shown) of \(outline.sections.count)", read.isError)
    }
}
