import AppKit
import Foundation
import Testing
@testable import TsugumiMacPresentation

/// Real model answers, rendered whole. `Fixtures/markdown-corpus/*.md` are
/// Gemma 4 responses captured through the server with prompts chosen to
/// exercise tables, code, HTML, images, quotes and nested lists; add a file
/// whenever a new kind of answer renders wrong, and the fix stays fixed.
///
/// Two checks run on every file:
/// - nothing falls back to raw text, and no Markdown syntax survives where
///   it should have been consumed;
/// - every word the model wrote is still on screen (the text-preservation
///   property), so a rewrite can never silently drop content.
///
/// Set `TSUGUMI_MARKDOWN_CORPUS_JSON` to a `chats.json` (the app's own
/// store) to run the same checks over a local conversation history, and
/// `TSUGUMI_MARKDOWN_PNG_DIR` to also write one PNG per answer for a visual
/// check; neither is part of the committed corpus.
@MainActor
@Suite struct MarkdownCorpusTests {
    struct Sample: CustomStringConvertible {
        let name: String
        let source: String
        var description: String { name }
    }

    nonisolated static var bundledSamples: [Sample] {
        guard let directory = Bundle.module.url(forResource: "markdown-corpus",
                                                withExtension: nil,
                                                subdirectory: "Fixtures"),
              let files = try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)
        else { return [] }
        return files
            .filter { $0.pathExtension == "md" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .compactMap { url in
                (try? String(contentsOf: url, encoding: .utf8)).map {
                    Sample(name: url.deletingPathExtension().lastPathComponent, source: $0)
                }
            }
    }

    nonisolated static var localSamples: [Sample] {
        guard let path = ProcessInfo.processInfo.environment["TSUGUMI_MARKDOWN_CORPUS_JSON"],
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let chats = root["chats"] as? [[String: Any]]
        else { return [] }
        var samples: [Sample] = []
        for (index, chat) in chats.enumerated() {
            for (turnIndex, turn) in ((chat["turns"] as? [[String: Any]]) ?? []).enumerated()
            where (turn["role"] as? String) == "assistant" {
                if let text = turn["text"] as? String, !text.isEmpty {
                    samples.append(Sample(name: "chat\(index)-turn\(turnIndex)", source: text))
                }
            }
            if let text = chat["outputText"] as? String, !text.isEmpty {
                samples.append(Sample(name: "chat\(index)-output", source: text))
            }
        }
        return samples
    }

    @Test func corpusIsPresent() {
        #expect(Self.bundledSamples.count >= 10)
    }

    @Test(arguments: bundledSamples + localSamples)
    func rendersWithoutLeavingMarkdownBehind(_ sample: Sample) {
        let result = ResponseMarkdownRenderer().render(sample.source)
        // Markdown syntax is judged outside code, where `**kwargs` or a
        // Python `# comment` are content, not leftover markup.
        let text = Self.textOutsideCode(result.attributedString)

        #expect(!result.usedFallback, "\(sample.name) fell back to raw text")
        #expect(!text.contains("```"), "\(sample.name) shows a code fence")
        #expect(!text.contains("**"), "\(sample.name) shows a bold marker")
        #expect(text.range(of: #"(?m)^\s*\|.*\|\s*$"#, options: .regularExpression) == nil,
                "\(sample.name) shows a table row as text")
        #expect(text.range(of: #"(?m)^\s*#{1,6} "#, options: .regularExpression) == nil,
                "\(sample.name) shows a heading marker")
        #expect(text.range(of: #"<br\s*/?>"#, options: [.regularExpression, .caseInsensitive]) == nil,
                "\(sample.name) shows a <br>")
    }

    /// The rendered text with every monospaced (code) run removed.
    static func textOutsideCode(_ rendered: NSAttributedString) -> String {
        var text = ""
        rendered.enumerateAttribute(.font, in: NSRange(location: 0, length: rendered.length)) {
            value, range, _ in
            if let font = value as? NSFont, font.isFixedPitch { return }
            text += (rendered.string as NSString).substring(with: range)
        }
        return text
    }

    @Test(arguments: bundledSamples + localSamples)
    func everyWordTheModelWroteIsShown(_ sample: Sample) {
        let text = ResponseMarkdownRenderer().plainText(sample.source)
        let missing = Self.contentTokens(in: sample.source).filter { !text.contains($0) }
        #expect(missing.isEmpty, "\(sample.name) lost \(missing.prefix(8))")
    }

    @Test(arguments: bundledSamples + localSamples)
    func writesPreviewWhenAsked(_ sample: Sample) throws {
        guard let directory = ProcessInfo.processInfo.environment["TSUGUMI_MARKDOWN_PNG_DIR"]
        else { return }
        let rendered = ResponseMarkdownRenderer().render(sample.source).attributedString
        let textView = NSTextView(frame: NSRect(x: 0, y: 0, width: 700, height: 10))
        _ = textView.layoutManager   // TextKit 1, as the transcript view uses
        textView.textStorage?.setAttributedString(rendered)
        textView.isVerticallyResizable = true
        textView.textContainer?.widthTracksTextView = true
        textView.layoutManager?.ensureLayout(for: textView.textContainer!)
        textView.sizeToFit()
        textView.frame = NSRect(x: 0, y: 0, width: 700, height: min(textView.frame.height, 2400))
        let rep = try #require(textView.bitmapImageRepForCachingDisplay(in: textView.bounds))
        textView.cacheDisplay(in: textView.bounds, to: rep)
        let png = try #require(rep.representation(using: .png, properties: [:]))
        try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
        try png.write(to: URL(fileURLWithPath: directory).appendingPathComponent("\(sample.name).png"))
    }

    /// The words of the source that must survive rendering: runs of two or
    /// more letters or digits, taken from everything except what Markdown
    /// legitimately consumes — link and image destinations, code-fence info
    /// strings, table alignment rows and character entities.
    nonisolated static func contentTokens(in source: String) -> [String] {
        var kept: [String] = []
        var inFence = false
        for rawLine in source.split(separator: "\n", omittingEmptySubsequences: false) {
            var line = String(rawLine)
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                inFence.toggle()
                continue
            }
            if !inFence {
                if line.range(of: #"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$"#,
                              options: .regularExpression) != nil {
                    continue
                }
                line = line.replacingOccurrences(of: #"\]\([^)]*\)"#, with: "]",
                                                 options: .regularExpression)
                line = line.replacingOccurrences(of: #"&[A-Za-z#0-9]+;"#, with: " ",
                                                 options: .regularExpression)
                line = line.replacingOccurrences(of: #"<br\s*/?>"#, with: " ",
                                                 options: [.regularExpression, .caseInsensitive])
            }
            kept.append(line)
        }
        let joined = kept.joined(separator: "\n")
        let pattern = try! NSRegularExpression(pattern: #"[\p{L}\p{N}]{2,}"#)
        let range = NSRange(joined.startIndex..., in: joined)
        var seen = Set<String>()
        return pattern.matches(in: joined, range: range).compactMap { match in
            guard let r = Range(match.range, in: joined) else { return nil }
            let token = String(joined[r])
            return seen.insert(token).inserted ? token : nil
        }
    }
}
