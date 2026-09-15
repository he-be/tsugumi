import Foundation
import Testing
@testable import TsugumiAppCore

@Suite struct HTMLTextExtractorTests {
    @Test func keepsTheArticleAndDropsChrome() {
        let html = """
        <html><head><title>見出し &amp; テスト</title><style>p{}</style>
        <script>var x = "<p>fake</p>";</script></head>
        <body><nav><a href="/">ホーム</a><a href="/news">ニュース</a></nav>
        <header>サイト名</header>
        <article><h1>本文の見出し</h1>
        <p>最初の段落です。&nbsp;続きの文。</p>
        <ul><li>項目一</li><li>項目二</li></ul>
        <p>二つ目の段落 &#12354; &#x3044;</p>
        \(String(repeating: "<p>詰め物の段落。</p>", count: 40))
        </article>
        <footer>© 2026 example</footer></body></html>
        """
        let extract = HTMLTextExtractor.extract(html: html)
        #expect(extract.title == "見出し & テスト")
        #expect(extract.text.hasPrefix("本文の見出し\n最初の段落です。 続きの文。\n- 項目一"))
        #expect(extract.text.contains("- 項目一\n- 項目二"))
        #expect(extract.text.contains("二つ目の段落 あ い"))
        #expect(!extract.text.contains("ホーム"))
        #expect(!extract.text.contains("fake"))
        #expect(!extract.text.contains("サイト名"))
        #expect(!extract.text.contains("example"))
        #expect(!extract.text.contains("\n\n"))
    }

    @Test func headingsAreKeptAsLines() {
        let html = """
        <html><body><h1>題名</h1><p>導入。</p><h2 class="x"> <span>価格</span> </h2><h3></h3>
        <p>219,800円 から。</p><h3>発売日</h3><p>9月18日。</p></body></html>
        """
        let extract = HTMLTextExtractor.extract(html: html)
        #expect(extract.text == "題名\n導入。\n価格\n219,800円 から。\n発売日\n9月18日。")
        #expect(extract.headingLines == [0, 2, 4])
        let markdown = HTMLTextExtractor.markdownHeadings(in: "# 題名\n本文\n#タグ\n## 節\n続き")
        #expect(markdown.text == "題名\n本文\n#タグ\n節\n続き")
        #expect(markdown.headingLines == [0, 3])
    }

    @Test func fallsBackToTheBodyWhenThereIsNoArticle() {
        let html = "<html><body><div>ひとつ</div><div>ふたつ</div></body></html>"
        let extract = HTMLTextExtractor.extract(html: html)
        #expect(extract.text == "ひとつ\nふたつ")
    }

    @Test func decodesShiftJISFromTheMetaTag() throws {
        let source = "<html><head><meta charset=\"Shift_JIS\"></head><body><p>日本語の本文</p></body></html>"
        let data = try #require(source.data(using: HTMLTextExtractor.shiftJIS))
        let decoded = HTMLTextExtractor.decode(data, contentType: "text/html")
        #expect(decoded.contains("日本語の本文"))
    }

    @Test func decodesTheHeaderCharsetFirst() throws {
        let data = try #require("<p>テキスト</p>".data(using: .japaneseEUC))
        let decoded = HTMLTextExtractor.decode(data, contentType: "text/html; charset=EUC-JP")
        #expect(decoded.contains("テキスト"))
    }

    @Test func clipsOnALineBoundaryWhenOneIsNear() {
        let text = String(repeating: "あ", count: 95) + "\n" + String(repeating: "い", count: 50)
        let (clipped, wasClipped) = HTMLTextExtractor.clip(text, to: 100)
        #expect(wasClipped)
        #expect(clipped == String(repeating: "あ", count: 95))
        let (short, wasShortClipped) = HTMLTextExtractor.clip("短い", to: 100)
        #expect(!wasShortClipped)
        #expect(short == "短い")
    }
}
