import Foundation

/// Turns a fetched HTML page into the text a model can read: the main
/// content region when the page marks one (`<article>` / `<main>`), with
/// scripts, styles, navigation and boilerplate removed, block tags turned
/// into line breaks, and entities decoded. A heuristic, not a browser —
/// good enough for news, documentation, blogs and wikis, which is what a
/// search result usually is.
public enum HTMLTextExtractor {
    public struct Extract: Equatable, Sendable {
        public var title: String
        public var text: String
    }

    /// Decodes the bytes with the charset the response or the page names,
    /// falling back through the encodings a Japanese page is likely to use.
    public static func decode(_ data: Data, contentType: String?) -> String {
        var candidates: [String.Encoding] = []
        if let named = charset(in: contentType ?? ""), let encoding = encoding(named: named) {
            candidates.append(encoding)
        }
        let head = String(decoding: data.prefix(4_096), as: UTF8.self)
        if let named = metaCharset(in: head), let encoding = encoding(named: named) {
            candidates.append(encoding)
        }
        candidates.append(contentsOf: [.utf8, shiftJIS, .japaneseEUC, .isoLatin1])
        for encoding in candidates {
            if let text = String(data: data, encoding: encoding) { return text }
        }
        return String(decoding: data, as: UTF8.self)
    }

    public static func extract(html: String) -> Extract {
        let title = firstMatch(#"<title[^>]*>(.*?)</title>"#, in: html)
            .map { decodeEntities(collapseWhitespace($0)) } ?? ""
        var body = html
        // Whole elements that never carry the article.
        for tag in ["script", "style", "noscript", "svg", "template", "iframe",
                    "nav", "header", "footer", "aside", "form", "select", "canvas"] {
            body = removing(#"<\#(tag)\b[^>]*>.*?</\#(tag)\s*>"#, from: body)
        }
        body = removing(#"<!--.*?-->"#, from: body)
        // The marked content region, when there is one and it is not tiny.
        if let region = largestRegion(["article", "main"], in: body),
           region.count >= 400 {
            body = region
        } else if let inner = firstMatch(#"<body\b[^>]*>(.*)</body>"#, in: body) {
            body = inner
        }
        body = replacing(#"<\s*li\b[^>]*>"#, in: body, with: "\n- ")
        body = replacing(#"<\s*(br|hr)\b[^>]*/?>"#, in: body, with: "\n")
        body = replacing(
            #"</?\s*(p|div|h[1-6]|tr|section|article|blockquote|pre|ul|ol|table|thead|tbody|dl|dt|dd|figure|figcaption|main|li)\b[^>]*>"#,
            in: body, with: "\n")
        body = replacing(#"</\s*(td|th)\b[^>]*>"#, in: body, with: "\t")
        body = replacing(#"<[^>]+>"#, in: body, with: " ")
        body = decodeEntities(body)
        return Extract(title: title, text: normalize(body))
    }

    /// Whitespace as the model should see it: single spaces within a line,
    /// trimmed lines, one line per block and no blank lines — a blank line
    /// is a token the model gains nothing from.
    static func normalize(_ text: String) -> String {
        text.split(separator: "\n", omittingEmptySubsequences: true)
            .map { collapseWhitespace(String($0)) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    /// Clips to `limit` characters on a line boundary when one is near.
    public static func clip(_ text: String, to limit: Int) -> (text: String, clipped: Bool) {
        guard text.count > limit else { return (text, false) }
        let cut = text.index(text.startIndex, offsetBy: limit)
        let head = text[..<cut]
        if let newline = head.lastIndex(of: "\n"),
           head.distance(from: newline, to: cut) < limit / 5 {
            return (String(head[..<newline]), true)
        }
        return (String(head), true)
    }

    // MARK: - Pieces

    static var shiftJIS: String.Encoding {
        String.Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(
            CFStringEncoding(CFStringEncodings.dosJapanese.rawValue)))
    }

    static func charset(in contentType: String) -> String? {
        firstMatch(#"charset\s*=\s*"?([A-Za-z0-9_\-]+)"#, in: contentType)
    }

    static func metaCharset(in head: String) -> String? {
        firstMatch(#"<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)"#, in: head)
    }

    static func encoding(named name: String) -> String.Encoding? {
        switch name.lowercased() {
        case "utf-8", "utf8": .utf8
        case "shift_jis", "shift-jis", "sjis", "x-sjis", "windows-31j", "cp932", "ms932": shiftJIS
        case "euc-jp", "eucjp", "x-euc-jp": .japaneseEUC
        case "iso-2022-jp": .iso2022JP
        case "iso-8859-1", "latin1": .isoLatin1
        case "windows-1252": .windowsCP1252
        case "utf-16": .utf16
        default: nil
        }
    }

    private static let options: NSRegularExpression.Options = [
        .caseInsensitive, .dotMatchesLineSeparators,
    ]

    static func firstMatch(_ pattern: String, in text: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else {
            return nil
        }
        let range = NSRange(text.startIndex..., in: text)
        guard let match = regex.firstMatch(in: text, range: range),
              match.numberOfRanges > 1,
              let captured = Range(match.range(at: 1), in: text) else { return nil }
        return String(text[captured])
    }

    static func largestRegion(_ tags: [String], in text: String) -> String? {
        var best: String?
        for tag in tags {
            guard let regex = try? NSRegularExpression(
                pattern: #"<\#(tag)\b[^>]*>(.*?)</\#(tag)\s*>"#, options: options) else { continue }
            let range = NSRange(text.startIndex..., in: text)
            for match in regex.matches(in: text, range: range) {
                guard let captured = Range(match.range(at: 1), in: text) else { continue }
                let region = String(text[captured])
                if region.count > (best?.count ?? 0) { best = region }
            }
        }
        return best
    }

    static func removing(_ pattern: String, from text: String) -> String {
        replacing(pattern, in: text, with: " ")
    }

    static func replacing(_ pattern: String, in text: String, with template: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else {
            return text
        }
        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, range: range, withTemplate: template)
    }

    static func collapseWhitespace(_ text: String) -> String {
        text.split(whereSeparator: { $0.isWhitespace || $0 == "\u{00A0}" })
            .joined(separator: " ")
    }

    private static let namedEntities: [String: String] = [
        "amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": "\u{00A0}",
        "copy": "©", "reg": "®", "trade": "™", "hellip": "…", "mdash": "—", "ndash": "–",
        "lsquo": "‘", "rsquo": "’", "ldquo": "“", "rdquo": "”", "yen": "¥", "middot": "·",
        "laquo": "«", "raquo": "»", "bull": "•", "deg": "°", "times": "×", "divide": "÷",
    ]

    static func decodeEntities(_ text: String) -> String {
        guard text.contains("&") else { return text }
        guard let regex = try? NSRegularExpression(
            pattern: #"&(#x[0-9A-Fa-f]{1,6}|#[0-9]{1,7}|[A-Za-z][A-Za-z0-9]{1,15});"#) else {
            return text
        }
        let nsText = text as NSString
        var result = ""
        var cursor = 0
        for match in regex.matches(in: text, range: NSRange(location: 0, length: nsText.length)) {
            result += nsText.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
            let entity = nsText.substring(with: match.range(at: 1))
            if entity.hasPrefix("#x") || entity.hasPrefix("#X"),
               let code = UInt32(entity.dropFirst(2), radix: 16),
               let scalar = Unicode.Scalar(code) {
                result.unicodeScalars.append(scalar)
            } else if entity.hasPrefix("#"),
                      let code = UInt32(entity.dropFirst()),
                      let scalar = Unicode.Scalar(code) {
                result.unicodeScalars.append(scalar)
            } else if let named = namedEntities[entity] {
                result += named
            } else {
                result += nsText.substring(with: match.range)
            }
            cursor = match.range.location + match.range.length
        }
        result += nsText.substring(from: cursor)
        return result
    }
}
