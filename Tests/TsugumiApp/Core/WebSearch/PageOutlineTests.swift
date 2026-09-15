import Foundation
import Testing
@testable import TsugumiAppCore

@Suite struct PageOutlineTests {
    func paragraph(_ name: String, _ characters: Int) -> String {
        String((name + String(repeating: "あ", count: characters)).prefix(characters - 1)) + "。"
    }

    /// Headings name sections; headings with no text of their own join the next name; a short section joins the next.
    @Test func headingsMakeTheSections() {
        let lines = ["ぱんくず", "題名", "章", "導入", paragraph("導入", 400), "価格", paragraph("価格", 300),
                     "発売日", "9月18日。", "スペック", paragraph("仕様", 500)]
        let outline = PageOutline(lines: lines, headingLines: [1, 2, 5, 7, 9], pageLimit: 6_000)
        #expect(outline.sections.map(\.heading) == ["題名 / 章", "価格", "発売日 / スペック"])
        #expect(outline.sections[0].text == "ぱんくず\n題名 / 章\n導入\n" + paragraph("導入", 400))
        #expect(outline.sections[2].text == "9月18日。\nスペック\n" + paragraph("仕様", 500))
        #expect(outline.entry(1) == "[2] 価格 (300字)")
        #expect(outline.totalCharacters == lines.joined(separator: "\n").count)
    }

    /// A long unheaded run is cut into parts at sentence ends, none past a read, and the parts are the text.
    @Test func unheadedTextIsCutIntoParts() {
        let text = (1...400).map { "文の \($0) 番目です。" }.joined()
        let outline = PageOutline(text: text, pageLimit: 6_000)
        #expect(outline.sections.count == 3)
        #expect(outline.sections.allSatisfy { $0.text.count <= 2_000 && $0.text.hasSuffix("。") && $0.heading.isEmpty })
        #expect(outline.sections.map(\.text).joined() == text)
        #expect(outline.label(0) == String(text.prefix(30)) + "…")
        #expect(!outline.isWhole(limit: 6_000))
        #expect(PageOutline(text: String(text.prefix(2_000)), pageLimit: 6_000).isWhole(limit: 6_000))
    }

    /// A very long page keeps its outline to 40 entries, every section still inside one read.
    @Test func aVeryLongPageKeepsTheOutlineShort() {
        var lines: [String] = []
        var headings: Set<Int> = []
        for number in 1...120 {
            headings.insert(lines.count)
            lines.append("見出し \(number)")
            lines.append(paragraph("本文\(number)", 300 + number * 7 % 200))
        }
        let outline = PageOutline(lines: lines, headingLines: headings, pageLimit: 6_000)
        #expect(outline.sections.count <= 40)
        #expect(outline.sections.allSatisfy { $0.text.count <= 6_000 })
        let huge = PageOutline(text: String(repeating: "長い文です。", count: 20_000), pageLimit: 6_000)
        #expect(huge.sections.count <= 40)
        #expect(huge.sections.allSatisfy { $0.text.count <= 4_500 })
    }

    @Test func aReadSaysWhatItLeftOut() {
        let lines = (1...6).flatMap { ["節\($0)", paragraph("本文\($0)", 900)] }
        let outline = PageOutline(lines: lines, headingLines: Set(stride(from: 0, to: 12, by: 2)), pageLimit: 2_000)
        #expect(outline.sections.count == 6)
        let read = outline.read([5, 2, 3, 9, 2], tool: "fetch_page", limit: 2_000)
        #expect(read.shown == [2, 3])
        #expect(!read.isError)
        #expect(read.lines.first == "節 2, 3 (全 6 節、本文は全 \(outline.totalCharacters) 文字)")
        #expect(read.lines.contains("(節 9 はありません。節は 1〜6 です)"))
        #expect(read.lines.contains("(節 5 は 1 回 2000 文字の上限を超えるので出していません。fetch_page の sections に渡すと読めます)"))
        #expect(read.lines.last == "(次の節: [4] 節4 (900字))")
        let last = outline.read([6], tool: "fetch_page", limit: 2_000)
        #expect(last.lines.last == "(これが最後の節です)")
        let overview = outline.overview(tool: "fetch_page", limit: 2_000)
        #expect(overview.shown == 1)
        #expect(overview.lines.last == "(次の節: [2] 節2 (900字))")
    }

    @Test func sectionNumbersAreReadInTheFormsAModelWrites() {
        func numbers(_ json: String) -> [Int]? {
            AppToolCall(id: "c", name: "fetch_page", argumentsJSON: json).integerListArgument("sections")
        }
        #expect(numbers(#"{"sections":"2,3"}"#) == [2, 3])
        #expect(numbers(#"{"sections":"[2, 3]"}"#) == [2, 3])
        #expect(numbers(#"{"sections":"2〜4"}"#) == [2, 3, 4])
        #expect(numbers(#"{"sections":"節 5 と 7"}"#) == [5, 7])
        #expect(numbers(#"{"sections":[1,2]}"#) == [1, 2])
        #expect(numbers(#"{"sections":3}"#) == [3])
        #expect(numbers(#"{"sections":""}"#) == nil)
        #expect(numbers(#"{"url":"x"}"#) == nil)
    }
}
