import Foundation
import TsugumiAppCore

// `--extract-endpoint URL --extract-model ID` (Online, docs/qwen38/48): the model's web tools stop handing it pages.
// `web_search` and `fetch_page` take a `focus` — the points the model wants picked out — and the app opens the pages
// itself (the top `--extract-pages` results of a search, or the one URL) and has a second, light model (Gemma 4 E4B
// on a llama-server) read them and copy out the parts about the question and the focus. The model reads that
// extract instead of the page. Everything else is the app's loop; the other tools (the local Wikipedia) are as
// they were.

/// The light model that reads the pages: one chat completion per tool call, the question, the focus and the pages
/// in, the extract out. Its timings (llama-server's `timings`) go to `extract.jsonl`.
final class PageExtractor: @unchecked Sendable {
    let endpoint: URL
    let modelID: String
    /// Each page is cut to this many characters before the model reads it.
    let pageCharacters: Int
    let maxTokens: Int
    let log: URL
    private let session: URLSession
    private let lock = NSLock()
    /// The question of the running turn (set from `lookups`, which the app calls at the start of every turn).
    var question = ""
    /// Starts and stops the light model around each call (`--extract-launch`); nil when it stays up.
    var launcher: ExtractorLauncher?

    init(endpoint: URL, modelID: String, pageCharacters: Int, maxTokens: Int, log: URL) {
        self.endpoint = endpoint
        self.modelID = modelID
        self.pageCharacters = pageCharacters
        self.maxTokens = maxTokens
        self.log = log
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 600
        configuration.timeoutIntervalForResource = 1200
        session = URLSession(configuration: configuration)
    }

    static let system = """
    あなたは資料係です。質問に答える担当者のために、渡された Web ページから必要な部分を抜き出します。答えるのは担当者で、あなたは抜き出すだけです。
    - 質問と「拾ってほしい点」に関係する記述を、ページの文のまま抜き出してください。数字・日付・固有名詞・金額は一字も変えないでください。
    - 拾ってほしい点に日付や条件が書かれていても、それに近い記述 (日付が少し違うもの、同じ話題のもの) は日付ごと抜き出してください。合うかどうかは担当者が判断します。
    - 抜き出しごとに、先頭に出典のページ番号を [1] のように付けてください。
    - 「[番号] 関係する記述なし」と書くのは、そのページに質問の話題がまったく無いときだけです。
    - ページに書かれていないことは書かないでください。自分の知識で補ったり、質問への答えを書いたりしないでください。
    - 前置き・見出し・質問の繰り返しは書かず、最初の行から抜き出しを書いてください。全体で 1,000 字以内にしてください。
    """

    struct Page {
        var title: String
        var url: String
        var text: String
    }

    /// The extract, or an error line the model can read.
    func extract(focus: String, pages: [Page], dateStamp: String, tool: String, subject: String) async -> String {
        // ページを先に、頼みごとを最後に置く (読む前に質問を写し始めないように)
        var user = "ページ (\(dateStamp)):\n"
        for (index, page) in pages.enumerated() {
            let text = page.text.count > pageCharacters ? String(page.text.prefix(pageCharacters)) + "\n(以下略)" : page.text
            user += "\n[\(index + 1)] \(page.title)\n\(page.url)\n\(text)\n"
        }
        user += "\n---\n質問: \(question)\n拾ってほしい点: \(focus)\n上のページから、抜き出しだけを書いてください。"
        let body: [String: Any] = [
            "model": modelID,
            "messages": [["role": "system", "content": Self.system], ["role": "user", "content": user]],
            // Gemma 4 の公式サンプリング、thinking なし
            "temperature": 1.0, "top_p": 0.95, "top_k": 64,
            "max_tokens": maxTokens,
            "chat_template_kwargs": ["enable_thinking": false],
            "cache_prompt": false,
        ]
        var request = URLRequest(url: endpoint.appendingPathComponent("v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        let started = Date()
        var record: [String: Any] = ["tool": tool, "subject": subject, "focus": focus, "question": question,
                                     "pages": pages.map { ["url": $0.url, "chars": $0.text.count] },
                                     "started": started.timeIntervalSince1970]
        defer { append(record) }
        if let launcher {
            // ページの取得の後に、立ち上げの残りをここで待つ (見える待ち)
            let waitStart = Date()
            record["launch_s"] = await launcher.ready()
            record["launch_wait_s"] = Date().timeIntervalSince(waitStart)
        }
        let requestStart = Date()
        defer { record["request_s"] = Date().timeIntervalSince(requestStart) }
        do {
            let (data, response) = try await session.data(for: request)
            record["wall_s"] = Date().timeIntervalSince(started)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200,
                  let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let choices = json["choices"] as? [[String: Any]],
                  let message = choices.first?["message"] as? [String: Any],
                  let content = message["content"] as? String else {
                let text = String(data: data, encoding: .utf8) ?? ""
                record["error"] = String(text.prefix(500))
                return "error: 抜き出しに失敗しました (\(String(text.prefix(200))))"
            }
            if let timings = json["timings"] as? [String: Any] {
                record["timings"] = timings
            }
            record["finish_reason"] = choices.first?["finish_reason"] ?? ""
            record["extract"] = content
            return content.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            record["wall_s"] = Date().timeIntervalSince(started)
            record["error"] = "\(error)"
            return "error: 抜き出しに失敗しました (\(error))"
        }
    }

    private func append(_ record: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) else { return }
        lock.withLock {
            if !FileManager.default.fileExists(atPath: log.path) {
                FileManager.default.createFile(atPath: log.path, contents: nil)
            }
            guard let handle = try? FileHandle(forWritingTo: log) else { return }
            handle.seekToEndOfFile()
            handle.write(data)
            handle.write(Data("\n".utf8))
            try? handle.close()
        }
    }
}

/// `--extract-launch SCRIPT --extract-prefetch FILES`: the light model runs only while it reads. QFN and E4B do not
/// fit in 18 GB together (docs/qwen38/48 §3: with E4B left up, each QFN request paged in 5-8 GB more and swapped), so
/// each tool call starts the server and stops it after the extract. The start runs beside the page reads: the weight
/// files are read into the page cache by parallel sequential reads (4-6 GB/s, against ~1.8 GB/s for mmap faults),
/// then the server script starts (`--no-warmup`).
final class ExtractorLauncher: @unchecked Sendable {
    let script: String
    let prefetch: [String]
    let health: URL
    private let lock = NSLock()
    private var process: Process?
    private var starting: Task<Double, Never>?

    init(script: String, prefetch: [String], endpoint: URL) {
        self.script = script
        self.prefetch = prefetch
        self.health = endpoint.appendingPathComponent("health")
    }

    /// Starts the server unless it is up or starting. Returns at once; `ready()` waits.
    func begin() {
        lock.withLock {
            guard starting == nil else { return }
            starting = Task { await self.launch() }
        }
    }

    /// Seconds from `begin` until the server answered `/health` (starts it first if nobody did).
    func ready() async -> Double {
        begin()
        return await lock.withLock { starting! }.value
    }

    func stop() {
        lock.withLock {
            process?.terminate()
            process?.waitUntilExit()
            process = nil
            starting = nil
        }
    }

    private func launch() async -> Double {
        let started = Date()
        for path in prefetch {
            Self.readIntoCache(path)
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: script)
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            return -1
        }
        lock.withLock { self.process = process }
        while process.isRunning {
            var request = URLRequest(url: health)
            request.timeoutInterval = 1
            if let (_, response) = try? await URLSession.shared.data(for: request),
               (response as? HTTPURLResponse)?.statusCode == 200 {
                return Date().timeIntervalSince(started)
            }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        return -1
    }

    /// Reads the file with 4 parallel sequential readers, 4 MiB at a time, discarding the bytes.
    static func readIntoCache(_ path: String, readers: Int = 4) {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
              let size = (attributes[.size] as? NSNumber)?.intValue, size > 0 else { return }
        let part = (size + readers - 1) / readers
        DispatchQueue.concurrentPerform(iterations: readers) { index in
            guard let handle = FileHandle(forReadingAtPath: path) else { return }
            defer { try? handle.close() }
            var offset = index * part
            let end = min(size, offset + part)
            try? handle.seek(toOffset: UInt64(offset))
            while offset < end {
                let count = min(4 << 20, end - offset)
                guard let data = try? handle.read(upToCount: count), !data.isEmpty else { break }
                offset += data.count
            }
        }
    }
}

/// The web tools with `focus`, answering with extracts; any other tool goes to `base`.
struct ExtractingWebExecutor: AppToolExecutor {
    let base: (any AppToolExecutor)?
    let web: WebSearchToolExecutor
    let extractor: PageExtractor
    /// How many of a search's results the app opens.
    let pagesPerSearch: Int

    static let reading = """
    web_search を呼ぶと、アプリが検索結果の上位 {pages} ページを開き、別のモデルが focus に書いた点をページの本文から抜き出して返します (検索結果のタイトル・URL と、[番号] 付きの抜き出し)。focus には、質問に答えるためにページから拾ってほしい事実を短く書きます (例: 「発売日と価格」「廃止されたかどうかと、その日付」)。抜き出しで足りなければ、focus を変えて fetch_page で別の URL (検索結果の他の URL など) を読むか、言い換えて検索します。fetch_page も、そのページから focus の点を抜き出して返します。web_search と fetch_page が返す内容は、いま実際にインターネットから取得した現在の情報です。抜き出しに無いことを推測で埋めないでください。質問に URL が書かれていれば、最初に fetch_page としてそのページからの抜き出しが添えられています。
    """

    var definitions: [AppToolDefinition] {
        let own = [
            AppToolDefinition(
                name: WebSearchToolExecutor.searchToolName,
                description: "Web を検索する。上位のページをアプリが開き、focus に書いた点をページから抜き出して、検索結果のタイトル・URL と一緒に返す。最新の情報や、自分の知識だけでは確信が持てない事実を調べるときに使う。日本語のクエリで検索する。",
                parametersJSON: #"{"type":"object","properties":{"query":{"type":"string","description":"検索クエリ。固有名詞と要点を短く並べる。"},"focus":{"type":"string","description":"ページから拾ってほしい点。質問に答えるのに要る事実を短く書く。"}},"required":["query","focus"]}"#),
            AppToolDefinition(
                name: WebSearchToolExecutor.fetchToolName,
                description: "URL を 1 つ開き、focus に書いた点をページの本文から抜き出して返す。検索結果の他の URL や、抜き出しで足りなかったページを読むときに使う。",
                parametersJSON: #"{"type":"object","properties":{"url":{"type":"string","description":"読むページの URL (http または https)。"},"focus":{"type":"string","description":"ページから拾ってほしい点。質問に答えるのに要る事実を短く書く。"}},"required":["url","focus"]}"#),
        ]
        return (base?.definitions ?? []).filter { !Self.isWeb($0.name) } + own
    }

    var promptFacts: AppToolPromptFacts {
        var facts = base?.promptFacts ?? AppToolPromptFacts(web: true, wikipediaDate: nil)
        facts.web = true
        facts.webReading = Self.reading.replacingOccurrences(of: "{pages}", with: String(pagesPerSearch))
        facts.searchReadsPages = true
        return facts
    }

    static func isWeb(_ name: String) -> Bool {
        name == WebSearchToolExecutor.searchToolName || name == WebSearchToolExecutor.fetchToolName
    }

    func subject(of call: AppToolCall) -> String {
        switch call.name {
        case WebSearchToolExecutor.searchToolName: call.stringArgument("query") ?? call.argumentsJSON
        case WebSearchToolExecutor.fetchToolName: call.stringArgument("url") ?? call.argumentsJSON
        default: base?.subject(of: call) ?? call.argumentsJSON
        }
    }

    /// The app calls this at the start of every turn with the question: the extractor keeps it. A URL in the
    /// question is read as a `fetch_page` whose focus is the question.
    func lookups(prompt: String, callIDPrefix: String) async -> [AppToolLookup] {
        extractor.question = prompt
        var lookups = await base?.lookups(prompt: prompt, callIDPrefix: callIDPrefix + "b") ?? []
        for (index, url) in WebSearchToolExecutor.urlsInPrompt(prompt).prefix(WebSearchToolExecutor.lookupURLLimit).enumerated() {
            let result = await fetch(url, focus: "質問に答えるのに要る記述")
            let arguments = try? JSONSerialization.data(withJSONObject: ["url": url, "focus": "質問に答えるのに要る記述"],
                                                        options: [.withoutEscapingSlashes])
            let call = AppToolCall(id: "\(callIDPrefix)\(index + 1)", name: WebSearchToolExecutor.fetchToolName,
                                   argumentsJSON: arguments.flatMap { String(data: $0, encoding: .utf8) } ?? "{}")
            lookups.append(AppToolLookup(call: call, result: result, subject: url))
        }
        return lookups
    }

    func execute(_ call: AppToolCall) async -> AppToolResult {
        let focus = call.stringArgument("focus")?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let pointed = focus.isEmpty ? "質問に答えるのに要る記述" : focus
        switch call.name {
        case WebSearchToolExecutor.searchToolName:
            guard let query = call.stringArgument("query")?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !query.isEmpty else {
                return AppToolResult(content: "error: web_search needs a non-empty \"query\".",
                                     isError: true, summary: "missing query")
            }
            return await search(query, focus: pointed)
        case WebSearchToolExecutor.fetchToolName:
            guard let url = call.stringArgument("url")?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !url.isEmpty else {
                return AppToolResult(content: "error: fetch_page needs a non-empty \"url\".",
                                     isError: true, summary: "missing url")
            }
            return await fetch(url, focus: pointed)
        default:
            guard let base else {
                return AppToolResult(content: "error: unknown tool \(call.name).", isError: true, summary: "unknown tool")
            }
            return await base.execute(call)
        }
    }

    func search(_ query: String, focus: String) async -> AppToolResult {
        extractor.launcher?.begin()
        defer { extractor.launcher?.stop() }
        let response: WebSearchResponse
        switch await web.searchResponse(query) {
        case .success(let found): response = found
        case .failure(let failure): return failure.result
        }
        var lines = ["検索: \(query) (\(response.provider), \(response.hits.count) 件、\(web.dateStamp))"]
        for highlight in response.highlights {
            lines.append("★ \(highlight)")
        }
        if response.hits.isEmpty {
            lines.append("(結果なし。別の言い方で検索する)")
            return AppToolResult(content: lines.joined(separator: "\n"), summary: "\(response.provider) · 0 hits")
        }
        // 上位から開く。pagesPerSearch + 2 件を並べて取りに行き、開けたものを順位の順に pagesPerSearch 個使う
        let candidates = Array(response.hits.enumerated().prefix(pagesPerSearch + 2))
        let web = self.web
        let fetched = await withTaskGroup(of: (Int, PageExtractor.Page?).self) { group in
            for (index, hit) in candidates {
                group.addTask {
                    switch await web.page(hit.url) {
                    case .success(let (url, page)):
                        return (index, PageExtractor.Page(title: page.title.isEmpty ? hit.title : page.title,
                                                          url: url.absoluteString, text: page.text))
                    case .failure:
                        return (index, nil)
                    }
                }
            }
            var all: [(Int, PageExtractor.Page?)] = []
            for await item in group { all.append(item) }
            return all
        }
        let opened: [(index: Int, page: PageExtractor.Page)] = fetched
            .compactMap { item in item.1.map { (index: item.0, page: $0) } }
            .sorted { $0.index < $1.index }
            .prefix(pagesPerSearch).map { $0 }
        let openedIndexes = Set(opened.map(\.index))
        for (index, hit) in response.hits.enumerated() {
            lines.append("[\(index + 1)] \(hit.title)")
            lines.append("    \(hit.url)")
            if !openedIndexes.contains(index) {
                let dated = hit.date.map { "\($0) " } ?? ""
                let snippet = HTMLTextExtractor.collapseWhitespace(hit.snippet)
                if !snippet.isEmpty || !dated.isEmpty {
                    lines.append("    \(dated)\(snippet)")
                }
            }
        }
        guard !opened.isEmpty else {
            lines.append("(上位のページはどれも開けなかった)")
            return AppToolResult(content: lines.joined(separator: "\n"), summary: "\(response.provider) · no page")
        }
        // 抜き出しの [番号] は検索結果の番号にそろえる
        let extract = await extractor.extract(focus: focus, pages: opened.map(\.page), dateStamp: web.dateStamp,
                                              tool: WebSearchToolExecutor.searchToolName, subject: query)
        let renumbered = Self.renumber(extract, to: opened.map { $0.index + 1 })
        lines.append("")
        lines.append("上位のページ (\(opened.map { "[\($0.index + 1)]" }.joined(separator: " "))) から「\(focus)」について抜き出したもの:")
        lines.append(renumbered)
        return AppToolResult(content: lines.joined(separator: "\n"),
                             summary: "\(response.provider) · \(response.hits.count) hits · extract of \(opened.count)")
    }

    func fetch(_ text: String, focus: String) async -> AppToolResult {
        extractor.launcher?.begin()
        defer { extractor.launcher?.stop() }
        switch await web.page(text) {
        case .success(let (url, page)):
            let extract = await extractor.extract(
                focus: focus, pages: [PageExtractor.Page(title: page.title, url: url.absoluteString, text: page.text)],
                dateStamp: web.dateStamp, tool: WebSearchToolExecutor.fetchToolName, subject: url.absoluteString)
            var header: [String] = []
            if !page.title.isEmpty { header.append("タイトル: \(page.title)") }
            header.append("URL: \(url.absoluteString)")
            header.append(web.dateStamp)
            header.append("本文 \(page.text.count.formatted()) 字から「\(focus)」について抜き出したもの:")
            let body = extract.replacingOccurrences(of: "[1] ", with: "")
            return AppToolResult(content: (header + [body]).joined(separator: "\n"),
                                 summary: "\(page.reader) · \(page.text.count) chars · extract")
        case .failure(let failure):
            return failure.result
        }
    }

    /// `[1]`…`[n]` (the order the pages were handed over) → the search result numbers.
    static func renumber(_ text: String, to numbers: [Int]) -> String {
        var out = text
        // 大きい番号から置き換えて、[1] が [12] を壊さないようにする。一時的な印を経由する
        for (position, number) in numbers.enumerated().reversed() {
            out = out.replacingOccurrences(of: "[\(position + 1)]", with: "[#\(number)#]")
        }
        return out.replacingOccurrences(of: "[#", with: "[").replacingOccurrences(of: "#]", with: "]")
    }
}
