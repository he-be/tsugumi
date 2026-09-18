import Foundation
// Usage: wiki_section_split <index.sqlite> <max results> <queries.txt> [extra titles...] > sections.jsonl (docs/qwen38/38 §2-7)
// Build: swiftc -O -o wiki_section_split Scripts/qwen38/wiki_section_split/main.swift \
//   Sources/TsugumiApp/Core/LocalWikipedia/LocalWikipediaIndex.swift Sources/TsugumiApp/Core/LocalWikipedia/WikipediaTokenizer.swift \
//   Sources/TsugumiApp/Core/WebSearch/PageOutline.swift Sources/TsugumiApp/Core/WebSearch/HTMLTextExtractor.swift
// Every article the search returns for each query (one query per line), plus the extra titles, cut as
// WikipediaToolExecutor does (PageOutline(text:pageLimit: 6,000)). One line per section; stderr lists the articles.
let args = CommandLine.arguments
let index = try LocalWikipediaIndex(path: args[1])
let limit = Int(args[2])!
let queries = try String(contentsOfFile: args[3], encoding: .utf8).split(separator: "\n").map(String.init)
var ids: [Int] = []
func add(_ id: Int) { if !ids.contains(id) { ids.append(id) } }
for q in queries { for hit in index.search(q, limit: limit) { add(hit.pageID) } }
for title in args.dropFirst(4) {
    guard let page = index.page(title: title) else { FileHandle.standardError.write("no article: \(title)\n".data(using: .utf8)!); continue }
    add(page.pageID)
}
let out = FileHandle.standardOutput
var id = 0
for pageID in ids {
    guard let page = index.page(id: pageID) else { continue }
    let outline = PageOutline(text: page.text, pageLimit: 6_000)
    let whole = outline.isWhole(limit: 6_000)
    FileHandle.standardError.write("\(page.title)\t\(outline.totalCharacters)\t\(outline.sections.count)\t\(whole)\n".data(using: .utf8)!)
    for (i, s) in outline.sections.enumerated() {
        let rec: [String: Any] = ["id": id, "page": page.title, "section": i + 1, "count": outline.sections.count,
                                  "whole": whole, "heading": s.heading, "text": s.heading + "\n" + s.text]
        out.write(try JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys]))
        out.write("\n".data(using: .utf8)!)
        id += 1
    }
}
