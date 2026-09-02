import Compression
import Foundation
import SQLite3
import Synchronization

/// A copy of Japanese Wikipedia on disk, as `Scripts/wiki/build_jawiki_index.py`
/// lays it out: one SQLite file with the articles (`pages`, body deflated),
/// the title and redirect lookup (`titles`), and a contentless FTS5 index
/// over pre-tokenized title / aliases / opening / body head (`search`).
/// Read-only; the app never writes it. macOS's own SQLite is enough — no
/// library ships with the app for this.
public final class LocalWikipediaIndex: Sendable {
    public struct Summary: Equatable, Sendable {
        public var articles: Int
        /// The dump the file was built from, as `YYYYMMDD`; empty when the
        /// build did not know.
        public var dumpDate: String
        public var builtAt: String

        /// The dump date the way the prompt says it ("2026年8月30日").
        public var dumpDateJapanese: String? {
            guard dumpDate.count == 8, let year = Int(dumpDate.prefix(4)),
                  let month = Int(dumpDate.dropFirst(4).prefix(2)),
                  let day = Int(dumpDate.suffix(2)) else { return nil }
            return "\(year)年\(month)月\(day)日"
        }
    }

    public struct Hit: Equatable, Sendable {
        public var pageID: Int
        public var title: String
        public var snippet: String
        public var incomingLinks: Int
        /// True when the article's title or one of its redirect names is
        /// the whole query — the hit `search` puts first whatever the scores.
        public var isExactTitle: Bool = false
    }

    /// An article the prompt names, with the numbers that said so.
    public struct Mention: Equatable, Sendable {
        public var pageID: Int
        public var title: String
        /// The string in the prompt (a redirect name when it differs: 米国).
        public var mention: String
        public var opening: String
        public var incomingLinks: Int
        /// Documents whose indexed head contains the mention string.
        public var documentFrequency: Int
        /// `incomingLinks / documentFrequency` — how much of the time the
        /// string is a link when it appears (§ WikipediaMentionFinder).
        public var linkProbability: Double
    }

    public struct Page: Equatable, Sendable {
        public var pageID: Int
        public var title: String
        public var text: String
    }

    public enum OpenError: Error, Equatable, CustomStringConvertible {
        case cannotOpen(String)
        case notAnIndex(String)
        case schemaVersion(String)
        case tokenizerMismatch

        public var description: String {
            switch self {
            case .cannotOpen(let message): "cannot open the index: \(message)"
            case .notAnIndex(let message): "not a Wikipedia index: \(message)"
            case .schemaVersion(let found): "index schema \(found) is not the \(LocalWikipediaIndex.schemaVersion) this build reads"
            case .tokenizerMismatch: "the index was built with different tokenizer rules than this build"
            }
        }
    }

    public static let schemaVersion = "1"
    /// bm25 column weights: title, aliases, opening, body head.
    static let bm25Weights: [Double] = [10, 6, 2, 1]
    static let candidateLimit = 60

    public let path: String
    public let summary: Summary
    private let db: Mutex<OpaquePointer?>

    public init(path: String) throws {
        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX
        guard sqlite3_open_v2(path, &handle, flags, nil) == SQLITE_OK, let handle else {
            let message = handle.map { String(cString: sqlite3_errmsg($0)) } ?? "sqlite3_open failed"
            if let handle { sqlite3_close(handle) }
            throw OpenError.cannotOpen(message)
        }
        self.path = path
        var meta: [String: String] = [:]
        do {
            meta = try Self.readMeta(handle)
        } catch {
            sqlite3_close(handle)
            throw error
        }
        guard meta["schema"] == Self.schemaVersion else {
            sqlite3_close(handle)
            throw OpenError.schemaVersion(meta["schema"] ?? "?")
        }
        guard meta["tokenizer_check"] == WikipediaTokenizer.tokenize(WikipediaTokenizer.checkSample).joined(separator: " ") else {
            sqlite3_close(handle)
            throw OpenError.tokenizerMismatch
        }
        self.summary = Summary(articles: Int(meta["articles"] ?? "") ?? 0,
                               dumpDate: meta["dump_date"] ?? "",
                               builtAt: meta["built_at"] ?? "")
        self.db = Mutex(handle)
    }

    deinit {
        db.withLock { handle in
            if let handle { sqlite3_close(handle) }
            handle = nil
        }
    }

    /// Opens the file just to read its summary — what the Inspector shows
    /// next to the path.
    public static func probe(path: String) -> Result<Summary, OpenError> {
        do {
            return .success(try LocalWikipediaIndex(path: path).summary)
        } catch let error as OpenError {
            return .failure(error)
        } catch {
            return .failure(.cannotOpen("\(error)"))
        }
    }

    // MARK: Search

    /// Full-text search: every term must match (AND); when that finds
    /// nothing, any term (OR); when that finds nothing, any bigram. An
    /// article whose title or redirect name is the whole query comes first
    /// whatever the scores ("米国" is アメリカ合衆国 before anything with 米国
    /// in its own title). The rest are the top candidates by bm25, nudged
    /// by their incoming links — additively and gently, because an article
    /// written this year has none yet and must not lose to a passing
    /// mention in an old, well-linked one.
    public func search(_ query: String, limit: Int) -> [Hit] {
        db.withLock { handle in
            guard let handle else { return [] }
            var candidates: [(id: Int, rank: Double)] = []
            for mode in WikipediaTokenizer.MatchMode.allCases {
                guard let expression = WikipediaTokenizer.matchExpression(query, mode: mode)
                else { return [] }
                candidates = Self.candidates(handle, expression: expression)
                if !candidates.isEmpty { break }
            }
            let exact = Self.pageID(handle, normalizedTitle: WikipediaTokenizer.normalizeTitle(query))
            var ids = candidates.map(\.id)
            if let exact, !ids.contains(exact) { ids.append(exact) }
            guard !ids.isEmpty else { return [] }
            let rows = Self.pageRows(handle, ids: ids)
            func hit(_ id: Int, _ row: PageRow) -> Hit {
                Hit(pageID: id, title: row.title,
                    snippet: Self.snippet(row.opening, query: query), incomingLinks: row.incoming)
            }
            var scored: [(score: Double, hit: Hit)] = []
            for candidate in candidates where candidate.id != exact {
                guard let row = rows[candidate.id] else { continue }
                scored.append((candidate.rank - 0.5 * log1p(Double(row.incoming)), hit(candidate.id, row)))
            }
            scored.sort { $0.score < $1.score }
            var hits = scored.map(\.hit)
            if let exact, let row = rows[exact] {
                var first = hit(exact, row)
                first.isExactTitle = true
                hits.insert(first, at: 0)
            }
            return Array(hits.prefix(max(limit, 1)))
        }
    }

    /// The articles `text` names, best first, at most `limit`. Candidate
    /// spans come from `WikipediaMentionFinder`; a span that is a title or
    /// redirect name claims its words (its sub-spans are not tried), and is
    /// kept when its link probability says it is a name rather than a word.
    /// Two spans naming the same article count once.
    public func mentions(in text: String, limit: Int) -> [Mention] {
        let candidates = WikipediaMentionFinder.candidates(in: WikipediaMentionFinder.words(in: text))
        return db.withLock { handle in
            guard let handle else { return [] }
            var claimed: [Range<Int>] = []
            var pages = Set<Int>()
            var found: [Mention] = []
            for candidate in candidates {
                if claimed.contains(where: { $0.lowerBound <= candidate.words.lowerBound
                                             && candidate.words.upperBound <= $0.upperBound }) { continue }
                let key = WikipediaTokenizer.normalizeTitle(candidate.text)
                guard let id = Self.pageID(handle, normalizedTitle: key) else { continue }
                claimed.append(candidate.words)
                guard let row = Self.pageRows(handle, ids: [id])[id], row.incoming > 0 else { continue }
                let frequency = Self.documentFrequency(handle, text: key)
                guard frequency > 0 else { continue }
                let probability = Double(row.incoming) / Double(frequency)
                guard WikipediaMentionFinder.keeps(linkProbability: probability,
                                                   wordCount: candidate.wordCount,
                                                   isLatin: candidate.isLatin),
                      pages.insert(id).inserted else { continue }
                found.append(Mention(pageID: id, title: row.title, mention: candidate.text,
                                     opening: row.opening, incomingLinks: row.incoming,
                                     documentFrequency: frequency, linkProbability: probability))
            }
            found.sort { $0.linkProbability > $1.linkProbability }
            return Array(found.prefix(max(limit, 0)))
        }
    }

    /// The article with this exact title or redirect name (after
    /// normalization). Nil when there is none.
    public func page(title: String) -> Page? {
        db.withLock { handle in
            guard let handle else { return nil }
            let key = WikipediaTokenizer.normalizeTitle(title)
            guard let id = Self.pageID(handle, normalizedTitle: key) else { return nil }
            return Self.page(handle, id: id)
        }
    }

    public func page(id: Int) -> Page? {
        db.withLock { handle in
            guard let handle else { return nil }
            return Self.page(handle, id: id)
        }
    }

    // MARK: SQL

    private static func readMeta(_ handle: OpaquePointer) throws -> [String: String] {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, "SELECT key, value FROM meta", -1, &statement, nil) == SQLITE_OK,
              let statement else {
            throw OpenError.notAnIndex(String(cString: sqlite3_errmsg(handle)))
        }
        defer { sqlite3_finalize(statement) }
        var meta: [String: String] = [:]
        while sqlite3_step(statement) == SQLITE_ROW {
            meta[column(statement, 0)] = column(statement, 1)
        }
        return meta
    }

    private static func candidates(_ handle: OpaquePointer, expression: String) -> [(id: Int, rank: Double)] {
        let sql = """
        SELECT rowid, bm25(search, ?, ?, ?, ?) AS rank FROM search
        WHERE search MATCH ? ORDER BY rank LIMIT \(candidateLimit)
        """
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK, let statement else { return [] }
        defer { sqlite3_finalize(statement) }
        for (index, weight) in bm25Weights.enumerated() {
            sqlite3_bind_double(statement, Int32(index + 1), weight)
        }
        sqlite3_bind_text(statement, 5, expression, -1, transient)
        var rows: [(id: Int, rank: Double)] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            rows.append((Int(sqlite3_column_int64(statement, 0)), sqlite3_column_double(statement, 1)))
        }
        return rows
    }

    private struct PageRow {
        var title: String
        var opening: String
        var incoming: Int
    }

    private static func pageRows(_ handle: OpaquePointer, ids: [Int]) -> [Int: PageRow] {
        guard !ids.isEmpty else { return [:] }
        let marks = Array(repeating: "?", count: ids.count).joined(separator: ",")
        let sql = "SELECT id, title, opening, incoming FROM pages WHERE id IN (\(marks))"
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK, let statement else { return [:] }
        defer { sqlite3_finalize(statement) }
        for (index, id) in ids.enumerated() {
            sqlite3_bind_int64(statement, Int32(index + 1), Int64(id))
        }
        var rows: [Int: PageRow] = [:]
        while sqlite3_step(statement) == SQLITE_ROW {
            rows[Int(sqlite3_column_int64(statement, 0))] = PageRow(
                title: column(statement, 1), opening: column(statement, 2),
                incoming: Int(sqlite3_column_int64(statement, 3)))
        }
        return rows
    }

    /// How many indexed documents contain `text` as a phrase.
    private static func documentFrequency(_ handle: OpaquePointer, text: String) -> Int {
        guard let expression = WikipediaTokenizer.phraseExpression(text) else { return 0 }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, "SELECT count(*) FROM search WHERE search MATCH ?",
                                 -1, &statement, nil) == SQLITE_OK, let statement else { return 0 }
        defer { sqlite3_finalize(statement) }
        sqlite3_bind_text(statement, 1, expression, -1, transient)
        guard sqlite3_step(statement) == SQLITE_ROW else { return 0 }
        return Int(sqlite3_column_int64(statement, 0))
    }

    private static func pageID(_ handle: OpaquePointer, normalizedTitle: String) -> Int? {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, "SELECT page_id FROM titles WHERE norm = ?", -1, &statement, nil) == SQLITE_OK,
              let statement else { return nil }
        defer { sqlite3_finalize(statement) }
        sqlite3_bind_text(statement, 1, normalizedTitle, -1, transient)
        guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
        return Int(sqlite3_column_int64(statement, 0))
    }

    private static func page(_ handle: OpaquePointer, id: Int) -> Page? {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, "SELECT title, text, text_bytes FROM pages WHERE id = ?", -1, &statement, nil) == SQLITE_OK,
              let statement else { return nil }
        defer { sqlite3_finalize(statement) }
        sqlite3_bind_int64(statement, 1, Int64(id))
        guard sqlite3_step(statement) == SQLITE_ROW else { return nil }
        let title = column(statement, 0)
        let byteCount = Int(sqlite3_column_int64(statement, 2))
        guard let blob = sqlite3_column_blob(statement, 1) else { return nil }
        let packed = Data(bytes: blob, count: Int(sqlite3_column_bytes(statement, 1)))
        guard let text = inflate(packed, byteCount: byteCount) else { return nil }
        return Page(pageID: id, title: title, text: text)
    }

    /// Raw DEFLATE (what `zlib.compressobj(wbits=-15)` wrote), which is what
    /// Apple's `COMPRESSION_ZLIB` reads.
    static func inflate(_ packed: Data, byteCount: Int) -> String? {
        guard byteCount > 0 else { return "" }
        var output = [UInt8](repeating: 0, count: byteCount)
        let written = packed.withUnsafeBytes { source -> Int in
            guard let base = source.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return 0 }
            return output.withUnsafeMutableBufferPointer { destination in
                compression_decode_buffer(destination.baseAddress!, byteCount,
                                          base, packed.count, nil, COMPRESSION_ZLIB)
            }
        }
        guard written == byteCount else { return nil }
        return String(decoding: output, as: UTF8.self)
    }

    private static func column(_ statement: OpaquePointer, _ index: Int32) -> String {
        guard let text = sqlite3_column_text(statement, index) else { return "" }
        return String(cString: text)
    }

    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    /// About 120 characters of the opening around the first query term
    /// that occurs in it, or its start.
    static func snippet(_ opening: String, query: String, width: Int = 120) -> String {
        let text = opening.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        let folded = text.precomposedStringWithCompatibilityMapping.lowercased()
        var start = text.startIndex
        var startOffset = 0
        for term in query.split(whereSeparator: \.isWhitespace) {
            let needle = String(term).precomposedStringWithCompatibilityMapping.lowercased()
            guard let range = folded.range(of: needle) else { continue }
            let position = folded.distance(from: folded.startIndex, to: range.lowerBound)
            startOffset = max(0, position - width / 3)
            // NFKC can change the character count; clamp to the original.
            startOffset = min(startOffset, max(0, text.count - 1))
            start = text.index(text.startIndex, offsetBy: startOffset)
            break
        }
        let end = text.index(start, offsetBy: width, limitedBy: text.endIndex) ?? text.endIndex
        var piece = String(text[start..<end])
        if startOffset > 0 { piece = "…" + piece }
        if end < text.endIndex { piece += "…" }
        return piece
    }
}
