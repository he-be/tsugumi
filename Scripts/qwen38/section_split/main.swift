import Foundation
// Usage: section_split <recorded-web-dir> > sections.jsonl (docs/qwen38/38)
// Build: swiftc -O -o section_split Scripts/qwen38/section_split/main.swift \
//   Sources/TsugumiApp/Core/WebSearch/HTMLTextExtractor.swift Sources/TsugumiApp/Core/WebSearch/PageOutline.swift
// One line per section of every recorded direct page fetch, split as the app does (limit 6,000).
let dir = URL(fileURLWithPath: CommandLine.arguments[1])
let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
    .filter { $0.pathExtension == "json" }.sorted { $0.path < $1.path }
let out = FileHandle.standardOutput
for file in files {
    guard let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any],
          let key = obj["key"] as? String, key.hasPrefix("GET "),
          !key.contains("r.jina.ai"), "\(obj["status"] ?? "")" == "200",
          let b64 = obj["body"] as? String, let body = Data(base64Encoded: b64) else { continue }
    let headers = (obj["headers"] as? String) ?? ""
    var ct = ""
    if let r = headers.range(of: #"'Content-Type': '([^']*)'"#, options: [.regularExpression, .caseInsensitive]) {
        ct = String(headers[r]).components(separatedBy: "': '").last!.replacingOccurrences(of: "'", with: "").lowercased()
    }
    guard ct.isEmpty || ct.contains("html") else { continue }
    let ex = HTMLTextExtractor.extract(html: HTMLTextExtractor.decode(body.prefix(3_000_000), contentType: ct))
    let lines = ex.text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    let outline = PageOutline(lines: lines, headingLines: ex.headingLines, pageLimit: 6_000)
    let url = String(key.dropFirst(4))
    for (i, s) in outline.sections.enumerated() {
        let rec: [String: Any] = ["page": url, "title": ex.title, "section": i + 1, "count": outline.sections.count,
                                  "whole": outline.isWhole(limit: 6_000), "heading": s.heading, "text": s.text]
        out.write(try JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys]))
        out.write("\n".data(using: .utf8)!)
    }
}
