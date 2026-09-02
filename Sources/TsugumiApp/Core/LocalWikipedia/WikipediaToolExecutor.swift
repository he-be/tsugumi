import Foundation

/// The two tools the chat declares when a local Wikipedia index is
/// configured. `wikipedia_search` runs the full-text search and hands back
/// titles with a line of each opening; `wikipedia_page` returns one
/// article's text, clipped to the configured length, with `from` to read
/// on. Nothing here touches the network: the index is a file on this Mac.
public struct WikipediaToolExecutor: AppToolExecutor {
    public static let searchToolName = "wikipedia_search"
    public static let pageToolName = "wikipedia_page"

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

    func search(_ query: String) -> AppToolResult {
        let hits = index.search(query, limit: maxResults)
        guard !hits.isEmpty else {
            return AppToolResult(
                content: "Wikipedia 検索: \(query) — 該当する記事はありません。別の語や記事名で検索してください。",
                summary: "Wikipedia · 0 hits")
        }
        var lines = ["Wikipedia 検索: \(query) (\(hits.count) 件)"]
        for (number, hit) in hits.enumerated() {
            lines.append("[\(number + 1)] \(hit.title)")
            if !hit.snippet.isEmpty { lines.append("    \(hit.snippet)") }
        }
        lines.append("")
        lines.append("本文を読むには wikipedia_page に題名を渡します。")
        return AppToolResult(content: lines.joined(separator: "\n"),
                             summary: "Wikipedia · \(hits.count) hits")
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
        let total = page.text.count
        let start = min(from, total)
        let startIndex = page.text.index(page.text.startIndex, offsetBy: start)
        let endIndex = page.text.index(startIndex, offsetBy: pageCharacterLimit,
                                       limitedBy: page.text.endIndex) ?? page.text.endIndex
        let clipped = endIndex < page.text.endIndex
        var lines = ["Wikipedia 記事: \(page.title)"]
        if start > 0 { lines.append("(\(start) 文字目から)") }
        lines.append("")
        lines.append(String(page.text[startIndex..<endIndex]))
        if clipped {
            let next = page.text.distance(from: page.text.startIndex, to: endIndex)
            lines.append("…(本文はここで打ち切り。全 \(total) 文字。続きは from=\(next) で読めます)")
        }
        let shown = page.text.distance(from: startIndex, to: endIndex)
        return AppToolResult(content: lines.joined(separator: "\n"),
                             summary: "Wikipedia · \(shown.formatted()) / \(total.formatted()) chars\(clipped ? " (clipped)" : "")")
    }
}
