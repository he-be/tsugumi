import AppKit
import Foundation
import Testing
@testable import TsugumiMacPresentation

@MainActor
@Suite struct ResponseMarkdownRendererTests {
    @Test func rendersSupportedMarkdownWithNativeAttributes() throws {
        let source = """
        # Heading

        A **bold** and *italic* sentence with ~~obsolete~~ text, `inlineCode`, and a [link](https://example.com).

        - first
        - second

        > quoted text

        ```swift
        let answer = 42
        ```

        ---
        """

        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string

        #expect(!result.usedFallback)
        #expect(text.contains("Heading"))
        #expect(text.contains("bold"))
        #expect(text.contains("italic"))
        #expect(text.contains("•\tfirst\n•\tsecond"))
        #expect(text.contains("│\tquoted text"))
        #expect(text.contains("let answer = 42"))
        #expect(text.contains("────────────────"))
        #expect(!text.contains("**"))
        #expect(!text.contains("```"))

        let linkRange = (text as NSString).range(of: "link")
        #expect(result.attributedString.attribute(.link, at: linkRange.location,
                                                  effectiveRange: nil) as? URL
            == URL(string: "https://example.com"))
        let linkColor = result.attributedString.attribute(
            .foregroundColor, at: linkRange.location, effectiveRange: nil) as? NSColor
        #expect(linkColor?.isEqual(NSColor.linkColor) == true)
        #expect(result.attributedString.attribute(.underlineStyle,
                                                  at: linkRange.location,
                                                  effectiveRange: nil) as? Int
            == NSUnderlineStyle.single.rawValue)

        let codeRange = (text as NSString).range(of: "inlineCode")
        let codeFont = try #require(result.attributedString.attribute(
            .font, at: codeRange.location, effectiveRange: nil) as? NSFont)
        #expect(codeFont.fontDescriptor.symbolicTraits.contains(.monoSpace))
        #expect(result.attributedString.attribute(
            .backgroundColor, at: codeRange.location, effectiveRange: nil) != nil)

        let strikeRange = (text as NSString).range(of: "obsolete")
        #expect(result.attributedString.attribute(
            .strikethroughStyle, at: strikeRange.location, effectiveRange: nil) != nil)
    }

    @Test func bareURLsAreLinkedWithoutTheirTrailingPunctuation() throws {
        let source = "参照: https://example.jp/w、https://ja.wikipedia.org/wiki/淀城。\n\n(https://example.com/a) と `https://code.example/x`"
        let result = ResponseMarkdownRenderer().render(source)
        let string = result.attributedString
        let text = string.string

        func link(at needle: String) -> URL? {
            let range = (text as NSString).range(of: needle)
            return string.attribute(.link, at: range.location, effectiveRange: nil) as? URL
        }
        #expect(link(at: "https://example.jp/w") == URL(string: "https://example.jp/w"))
        #expect(link(at: "https://ja.wikipedia.org") == URL(string: "https://ja.wikipedia.org/wiki/淀城"))
        #expect(link(at: "https://example.com/a") == URL(string: "https://example.com/a"))
        // The punctuation after each URL is plain text.
        for needle in ["、https", "。", ") と"] {
            let range = (text as NSString).range(of: needle)
            #expect(string.attribute(.link, at: range.location, effectiveRange: nil) == nil, "\(needle)")
        }
        // Inside code nothing is linked.
        #expect(link(at: "https://code.example/x") == nil)
    }

    @Test func aLinkWithoutAUsableDestinationOnlyLooksLikeOne() throws {
        let result = ResponseMarkdownRenderer().render("[記事](淀城) と [空]()")
        let text = result.attributedString.string
        let range = (text as NSString).range(of: "記事")
        #expect(result.attributedString.attribute(.link, at: range.location, effectiveRange: nil) == nil)
        let color = result.attributedString.attribute(
            .foregroundColor, at: range.location, effectiveRange: nil) as? NSColor
        #expect(color?.isEqual(NSColor.linkColor) == true)
    }

    @Test func unfinishedFenceStillRendersAsCode() throws {
        let source = "Before\n\n```python\nprint('unfinished')"
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        let text = result.attributedString.string
        #expect(!text.contains("```"))
        let codeRange = (text as NSString).range(of: "print('unfinished')")
        #expect(codeRange.location != NSNotFound)
        let font = try #require(result.attributedString.attribute(
            .font, at: codeRange.location, effectiveRange: nil) as? NSFont)
        #expect(font.isFixedPitch)
    }

    @Test func gfmTableRendersAsTextTableBlocks() throws {
        let source = """
        ### まとめ
        | 特徴 | 内容 |
        | :--- | :--- |
        | **Tensor コア** | 行列演算専用 |
        | NVLink | GPU間通信 |

        参照:
        * [NVIDIA](https://www.nvidia.com/)
        """
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        let rendered = result.attributedString
        let text = rendered.string
        #expect(text.contains("Tensor コア"))
        #expect(text.contains("GPU間通信"))
        #expect(text.contains("参照:"))
        #expect(!text.contains("|"))
        #expect(!text.contains(":---"))

        // Every cell is its own paragraph that carries a table block; the
        // header row is bold; text after the table carries no block.
        let headerRange = (text as NSString).range(of: "特徴")
        let headerStyle = try #require(rendered.attribute(
            .paragraphStyle, at: headerRange.location, effectiveRange: nil) as? NSParagraphStyle)
        #expect(headerStyle.textBlocks.first is NSTextTableBlock)
        let headerFont = try #require(rendered.attribute(
            .font, at: headerRange.location, effectiveRange: nil) as? NSFont)
        #expect(headerFont.fontDescriptor.symbolicTraits.contains(.bold))

        let cellRange = (text as NSString).range(of: "GPU間通信")
        let cellBlock = try #require((rendered.attribute(
            .paragraphStyle, at: cellRange.location, effectiveRange: nil) as? NSParagraphStyle)?
            .textBlocks.first as? NSTextTableBlock)
        #expect(cellBlock.startingRow == 2)
        #expect(cellBlock.startingColumn == 1)
        #expect(cellBlock.table.numberOfColumns == 2)

        let afterRange = (text as NSString).range(of: "参照:")
        let afterStyle = try #require(rendered.attribute(
            .paragraphStyle, at: afterRange.location, effectiveRange: nil) as? NSParagraphStyle)
        #expect(afterStyle.textBlocks.isEmpty)
    }

    @Test func htmlStaysLiteralTextWithoutFailingTheWholeAnswer() throws {
        let renderer = ResponseMarkdownRenderer()
        let source = """
        ### 1. std::vector<int> （動的配列）
        文中の std::map<std::string, int> と `<html>` と <b>太字</b>。

        <div>Never execute this</div>
        """
        let result = renderer.render(source)

        #expect(!result.usedFallback)
        let text = result.attributedString.string
        #expect(text.contains("std::vector<int>"))
        #expect(text.contains("std::map<std::string, int>"))
        #expect(text.contains("<html>"))
        #expect(text.contains("<b>太字</b>"))
        #expect(text.contains("<div>Never execute this</div>"))
        let headingRange = (text as NSString).range(of: "std::vector<int>")
        let font = try #require(result.attributedString.attribute(
            .font, at: headingRange.location, effectiveRange: nil) as? NSFont)
        #expect(font.pointSize > NSFont.systemFontSize)
    }

    @Test func boldAroundJapaneseBracketsRendersBold() throws {
        let source = "C++における `std::vector` は**「可変長配列」**、`std::map` は**「連想コンテナ」**です。"
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        let text = result.attributedString.string
        #expect(!text.contains("**"))
        #expect(text.contains("「可変長配列」"))
        let boldRange = (text as NSString).range(of: "可変長配列")
        let font = try #require(result.attributedString.attribute(
            .font, at: boldRange.location, effectiveRange: nil) as? NSFont)
        #expect(font.fontDescriptor.symbolicTraits.contains(.bold))
    }

    @Test func bulletsNestedUnderOrderedItemsStayBullets() {
        let source = """
        1. **`<html>`**
            *   ルート要素です。
            *   `lang` を書けます。
        2. **`<head>`**
            *   設定を書く場所です。
        """
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        let text = result.attributedString.string
        #expect(text.contains("1.\t<html>"))
        #expect(text.contains("•\tルート要素です。"))
        #expect(text.contains("•\t設定を書く場所です。"))
        #expect(text.contains("2.\t<head>"))
        #expect(!text.contains("2.\t設定"))
    }

    @Test func breakTagInsideTableCellBecomesLineBreak() throws {
        let source = """
        | 項目 | Python |
        | :--- | :--- |
        | **主な用途** | AI<br>データ<br/>Web |
        """
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        let text = result.attributedString.string
        #expect(!text.contains("<br"))
        #expect(text.contains("AI\u{2028}データ\u{2028}Web"))
        let cellRange = (text as NSString).range(of: "データ")
        let style = try #require(result.attributedString.attribute(
            .paragraphStyle, at: cellRange.location, effectiveRange: nil) as? NSParagraphStyle)
        #expect(style.textBlocks.first is NSTextTableBlock)
    }

    @Test func imageSyntaxBecomesLinkAndFencedCodeIsUntouched() throws {
        let source = """
        画像 ![代替テキスト](https://example.com/a.png) と本文。

        ![](https://example.com/b.png)

        ```markdown
        ![alt](https://example.com/c.png)
        ```
        """
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        let text = result.attributedString.string
        #expect(text.contains("代替テキスト"))
        #expect(text.contains("https://example.com/b.png"))
        #expect(text.contains("![alt](https://example.com/c.png)"))
        // Links are styled, not clickable, in this renderer; the alt text
        // gets the same styling an ordinary link does.
        let linkRange = (text as NSString).range(of: "代替テキスト")
        #expect(result.attributedString.attribute(
            .underlineStyle, at: linkRange.location, effectiveRange: nil) != nil)
    }

    @Test func latexRemainsReadableText() {
        let source = "Cosine is $\\frac{u \\cdot v}{||u|| ||v||}$."
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string.contains("\\frac"))
        #expect(result.attributedString.string.contains("\\cdot"))
    }

    @Test func boldOnlyModelHeadingStaysOnItsOwnLine() {
        let source = "**Origins**\nFieldfares arrive from northern Europe."
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string
            == "Origins\n\nFieldfares arrive from northern Europe.")
    }
}

@MainActor
@Suite struct InstructionTranscriptDocumentControllerTests {
    private typealias CompletedTurn = InstructionTranscriptDocumentController.CompletedTurn

    @Test func historyRendersAboveLiveTurnInOrder() throws {
        let controller = InstructionTranscriptDocumentController()
        let storage = NSMutableAttributedString()

        controller.synchronize(
            storage: storage,
            history: [CompletedTurn(prompt: "first question", response: "first reply")],
            prompt: "second question",
            response: "second reply",
            isTerminal: false)

        let text = storage.string
        let firstPrompt = try #require(text.range(of: "first question"))
        let firstReply = try #require(text.range(of: "first reply"))
        let secondPrompt = try #require(text.range(of: "second question"))
        let secondReply = try #require(text.range(of: "second reply"))
        #expect(firstPrompt.lowerBound < firstReply.lowerBound)
        #expect(firstReply.lowerBound < secondPrompt.lowerBound)
        #expect(secondPrompt.lowerBound < secondReply.lowerBound)
    }

    @Test func streamingAppendsWithoutRebuildWhileHistoryIsUnchanged() {
        let controller = InstructionTranscriptDocumentController()
        let storage = NSMutableAttributedString()
        let history = [CompletedTurn(prompt: "q1", response: "a1")]

        controller.synchronize(
            storage: storage, history: history,
            prompt: "q2", response: "partial", isTerminal: false)
        let update = controller.synchronize(
            storage: storage, history: history,
            prompt: "q2", response: "partial answer", isTerminal: false)

        #expect(update.mutation == .appended)
        #expect(storage.string.contains("partial answer"))
        #expect(storage.string.contains("a1"))
    }

    @Test func newHistoryTurnForcesRebuild() {
        let controller = InstructionTranscriptDocumentController()
        let storage = NSMutableAttributedString()

        controller.synchronize(
            storage: storage, history: [],
            prompt: "q1", response: "a1", isTerminal: true)
        let update = controller.synchronize(
            storage: storage,
            history: [CompletedTurn(prompt: "q1", response: "a1")],
            prompt: "q2", response: "", isTerminal: false)

        #expect(update.mutation == .rebuilt)
        #expect(storage.string.contains("q2"))
    }

    @Test func historyOnlyDocumentHasNoLiveAnswerLabel() {
        let controller = InstructionTranscriptDocumentController()
        let storage = NSMutableAttributedString()

        controller.synchronize(
            storage: storage,
            history: [CompletedTurn(prompt: "q1", response: "a1")],
            prompt: "", response: "", isTerminal: true)

        let text = storage.string
        #expect(text.contains("a1"))
        let answerLabels = text.components(separatedBy: "Answer\n").count - 1
        #expect(answerLabels == 1)
    }

    @Test func appAccentMatchesProductRGB() {
        let color = TsugumiMacTheme.accentNSColor
            .usingColorSpace(.sRGB)
        #expect(color != nil)
        #expect(abs((color?.redComponent ?? 0) - 106.0 / 255.0) < 0.000_001)
        #expect(abs((color?.greenComponent ?? 0) - 186.0 / 255.0) < 0.000_001)
        #expect(abs((color?.blueComponent ?? 0) - 113.0 / 255.0) < 0.000_001)
    }

    @Test func rebuildsThenAppendsOnlyNewResponseSuffix() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()

        let first = controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: "Hel",
            isTerminal: false)
        let second = controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: "Hello",
            isTerminal: false)

        #expect(first.mutation == .rebuilt)
        #expect(second.mutation == .appended)
        #expect(storage.string == "You\nExplain this\n\nAnswer\nHello")
        #expect(storage.string.components(separatedBy: "Answer").count == 2)
        let answerRange = (storage.string as NSString).range(of: "Answer")
        let answerColor = storage.attribute(
            .foregroundColor,
            at: answerRange.location,
            effectiveRange: nil) as? NSColor
        #expect(answerColor?.isEqual(TsugumiMacTheme.accentNSColor) == true)
        #expect(controller.response == "Hello")
    }

    @Test func animatedPrefillPlaceholderIsPresentationOnlyAndFirstResponseRemovesIt() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()

        let prefilling = controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: "",
            isTerminal: false,
            showsPrefillPlaceholder: true)

        #expect(prefilling.mutation == .rebuilt)
        #expect(controller.showsPrefillPlaceholder)
        #expect(storage.string == "You\nExplain this\n\nAnswer\nProcessing your prompt")
        #expect(controller.response.isEmpty)
        #expect(controller.assistantRange.length == 0)

        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt."))
        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt.."))
        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt..."))
        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt"))

        let responding = controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: "Hello",
            isTerminal: false,
            showsPrefillPlaceholder: true)

        #expect(responding.mutation == .rebuilt)
        #expect(!controller.showsPrefillPlaceholder)
        #expect(storage.string == "You\nExplain this\n\nAnswer\nHello")
        #expect(!storage.string.contains("Processing your prompt"))
        #expect((storage.string as NSString).substring(with: responding.assistantRange)
            == "Hello")
    }

    @Test func processingAnimationPolicyStopsForTextAndTerminalStates() {
        #expect(InstructionTranscriptDocumentController.shouldRunPrefillAnimation(
            response: "", isTerminal: false, requested: true))
        #expect(!InstructionTranscriptDocumentController.shouldRunPrefillAnimation(
            response: "First token", isTerminal: false, requested: true))
        #expect(!InstructionTranscriptDocumentController.shouldRunPrefillAnimation(
            response: "", isTerminal: true, requested: true))
        #expect(!InstructionTranscriptDocumentController.shouldRunPrefillAnimation(
            response: "", isTerminal: false, requested: false))
    }

    @Test func promptChangeOrResponseResetRebuildsWithoutStaleBytes() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        _ = controller.synchronize(
            storage: storage, prompt: "Old", response: "Long response", isTerminal: false)

        let result = controller.synchronize(
            storage: storage, prompt: "New", response: "Short", isTerminal: false)

        #expect(result.mutation == .rebuilt)
        #expect(storage.string == "You\nNew\n\nAnswer\nShort")
        #expect(!storage.string.contains("Old"))
        #expect(!storage.string.contains("Long response"))
    }

    @Test func terminalUpdateFormatsOnlyAssistantRangeAndKeepsRawResponse() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        _ = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "**Bold answer**",
            isTerminal: false)

        let result = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "**Bold answer**",
            isTerminal: true)

        #expect(result.mutation == .finalized)
        #expect(controller.isFinalized)
        #expect(controller.response == "**Bold answer**")
        #expect(storage.string == "You\nQuestion\n\nAnswer\nBold answer")
        #expect((storage.string as NSString).substring(with: result.assistantRange)
            == "Bold answer")

        let unchanged = storage.copy() as! NSAttributedString
        let repeated = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "**Bold answer**",
            isTerminal: true)
        #expect(repeated.mutation == .none)
        #expect(storage.isEqual(to: unchanged))
    }

    @Test func terminalPartialOutputIsReadableAndNextRunRestoresStreamingSource() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        _ = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "Partial **answer**",
            isTerminal: true)
        #expect(storage.string.hasSuffix("Partial answer"))

        let result = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "Partial **answer**",
            isTerminal: false)
        #expect(result.mutation == .rebuilt)
        #expect(!controller.isFinalized)
        #expect(storage.string.hasSuffix("Partial **answer**"))
    }

    @Test func terminalResponseRendersAgainWhenClosingFenceArrivesLate() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        let partial = "```cpp\nkernel void matmul() {}"
        let complete = partial + "\n```"

        let first = controller.synchronize(
            storage: storage,
            prompt: "Write a Metal kernel",
            response: partial,
            isTerminal: true)
        #expect(first.mutation == .finalized)
        // The unfinished fence is closed for display, so the code already
        // shows as code; the late closing fence changes nothing visible.
        #expect(storage.string.hasSuffix("kernel void matmul() {}\n"))
        #expect(!storage.string.contains("```"))

        let updated = controller.synchronize(
            storage: storage,
            prompt: "Write a Metal kernel",
            response: complete,
            isTerminal: true)

        #expect(updated.mutation == .finalized)
        #expect(controller.response == complete)
        #expect((storage.string as NSString).substring(with: updated.assistantRange)
            == "kernel void matmul() {}\n")
        #expect(!storage.string.contains("```"))
    }

    @Test func selectionRangesClampToCurrentStorage() {
        let ranges = InstructionTranscriptDocumentController.clampedRanges([
            NSRange(location: 3, length: 20),
            NSRange(location: 50, length: 2),
        ], toLength: 10)

        #expect(ranges == [
            NSRange(location: 3, length: 7),
            NSRange(location: 10, length: 0),
        ])
    }

    @Test func terminalFormattingAlwaysScrollsToBottom() {
        #expect(InstructionTranscriptDocumentController.shouldScrollToBottom(
            wasAtBottom: false,
            mutation: .finalized))
        #expect(InstructionTranscriptDocumentController.shouldScrollToBottom(
            wasAtBottom: true,
            mutation: .appended))
        #expect(!InstructionTranscriptDocumentController.shouldScrollToBottom(
            wasAtBottom: false,
            mutation: .appended))
    }
}
