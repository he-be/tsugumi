import Foundation

/// The two tools the chat declares when a local Wikipedia index is
/// configured. `wikipedia_search` runs the full-text search and hands back
/// titles with a line of each opening; `wikipedia_page` returns one
/// article's text, clipped to the configured length, with `from` to read
/// on. Nothing here touches the network: the index is a file on this Mac.
public struct WikipediaToolExecutor: AppToolExecutor {
    public static let searchToolName = "wikipedia_search"
    public static let pageToolName = "wikipedia_page"
    /// The round the app runs itself before the model's first: not
    /// declared, so the model cannot call it, only read it.
    public static let lookupToolName = "wikipedia_lookup"
    /// How many articles a lookup shows, and how much of each opening.
    public static let lookupLimit = 3
    public static let lookupOpeningLimit = 400

    let index: LocalWikipediaIndex
    let maxResults: Int
    let pageCharacterLimit: Int

    public init(index: LocalWikipediaIndex, maxResults: Int, pageCharacterLimit: Int) {
        self.index = index
        self.maxResults = max(1, maxResults)
        self.pageCharacterLimit = max(500, pageCharacterLimit)
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
                description: "日本語版 Wikipedia の記事を 1 つ開いて本文を返す。wikipedia_search の結果の題名をそのまま渡す。本文が長くて打ち切られたときは、from に示された文字位置を渡すと続きが読める。",
                parametersJSON: #"{"type":"object","properties":{"title":{"type":"string","description":"記事の題名。"},"from":{"type":"integer","description":"本文を読み始める文字位置 (省略時は 0)。"}},"required":["title"]}"#),
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
            let from = Int(call.stringArgument("from") ?? "") ?? 0
            return page(title, from: max(0, from))
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

    /// The openings of the articles the prompt names — a small, dated
    /// reference the model gets for free, so that what a 4B model half
    /// remembers about 淀城 or えきねっと is corrected before it decides
    /// whether to search. The prompt's own words pick the articles
    /// (`LocalWikipediaIndex.mentions`); nothing is guessed.
    public func lookups(prompt: String, callIDPrefix: String) async -> [AppToolLookup] {
        let mentions = index.mentions(in: prompt, limit: Self.lookupLimit)
        guard !mentions.isEmpty else { return [] }
        let titles = mentions.map(\.title)
        var lines = ["参考: 質問に含まれる語を Wikipedia (\(dateStamp)の複製) で引いた記事の導入部です。質問に関係なければ無視してください。本文は wikipedia_page で読めます。"]
        for mention in mentions {
            lines.append("")
            let alias = mention.mention == mention.title ? "" : " (質問中の「\(mention.mention)」)"
            lines.append("■ \(mention.title)\(alias)")
            lines.append(Self.clip(mention.opening, to: Self.lookupOpeningLimit))
        }
        let arguments = try? JSONSerialization.data(withJSONObject: ["titles": titles], options: [.withoutEscapingSlashes])
        let call = AppToolCall(id: callIDPrefix + "1", name: Self.lookupToolName,
                               argumentsJSON: arguments.flatMap { String(data: $0, encoding: .utf8) } ?? "{}")
        return [AppToolLookup(call: call,
                              result: AppToolResult(content: lines.joined(separator: "\n"),
                                                    summary: "Wikipedia · \(titles.count) 件"),
                              subject: titles.joined(separator: " / "))]
    }

    static func clip(_ text: String, to limit: Int) -> String {
        guard text.count > limit else { return text }
        return String(text.prefix(limit)) + "…"
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
        if Self.shouldGo(hits, query: query), let page = index.page(id: hits[0].pageID) {
            let body = Self.body(of: page, from: 0, limit: pageCharacterLimit)
            lines.append("[1] \(page.title) の本文:")
            lines.append(contentsOf: body.lines)
            lines.append("")
            lines.append("他の記事を読むには wikipedia_page に題名を渡します。")
            summary += " + \(page.title) \(body.shown.formatted()) chars\(body.clipped ? " (clipped)" : "")"
        } else {
            lines.append("本文を読むには wikipedia_page に題名を渡します。")
        }
        return AppToolResult(content: lines.joined(separator: "\n"), summary: summary)
    }

    func page(_ title: String, from: Int) -> AppToolResult {
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
        let body = Self.body(of: page, from: from, limit: pageCharacterLimit)
        var lines = ["Wikipedia 記事: \(page.title) (\(dateStamp))"]
        if body.start > 0 { lines.append("(\(body.start) 文字目から)") }
        lines.append("")
        lines.append(contentsOf: body.lines)
        return AppToolResult(content: lines.joined(separator: "\n"),
                             summary: "Wikipedia · \(body.shown.formatted()) / \(page.text.count.formatted()) chars\(body.clipped ? " (clipped)" : "")")
    }

    /// The article text from `from`, clipped to `limit` characters, with
    /// the line that says how to read on.
    static func body(of page: LocalWikipediaIndex.Page, from: Int, limit: Int)
        -> (lines: [String], start: Int, shown: Int, clipped: Bool) {
        let total = page.text.count
        let start = min(max(0, from), total)
        let startIndex = page.text.index(page.text.startIndex, offsetBy: start)
        let endIndex = page.text.index(startIndex, offsetBy: limit,
                                       limitedBy: page.text.endIndex) ?? page.text.endIndex
        let clipped = endIndex < page.text.endIndex
        var lines = [String(page.text[startIndex..<endIndex])]
        if clipped {
            let next = page.text.distance(from: page.text.startIndex, to: endIndex)
            lines.append("…(本文はここで打ち切り。全 \(total) 文字。続きは wikipedia_page の from=\(next) で読めます)")
        }
        return (lines, start, page.text.distance(from: startIndex, to: endIndex), clipped)
    }
}
