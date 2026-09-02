import AppKit
import Foundation
import Markdown

/// Renders a model answer (GitHub-flavored Markdown) into the attributed
/// string the transcript view shows.
///
/// The source is parsed into swift-markdown's AST and walked once; every
/// node kind has one place that decides how it looks, and nesting (a bullet
/// under a numbered item, a code block inside a quote) comes from the tree
/// rather than from guessing at flattened runs. Nothing the model wrote is
/// dropped: HTML stays literal text, an unfinished fence is still code, an
/// image shows its alt text (or its URL) styled like a link. The only
/// rewrites happen before parsing, in `presentationSource`, and cover the
/// two places where Gemma's Markdown and CommonMark disagree.
@MainActor
public struct ResponseMarkdownRenderer {
    public struct Result {
        public let attributedString: NSAttributedString
        public let usedFallback: Bool

        public init(attributedString: NSAttributedString, usedFallback: Bool) {
            self.attributedString = attributedString
            self.usedFallback = usedFallback
        }
    }

    public init() {}

    public func render(_ source: String) -> Result {
        guard !source.isEmpty else {
            return Result(attributedString: NSAttributedString(), usedFallback: false)
        }
        let document = Document(parsing: Self.presentationSource(for: source))
        var builder = Builder()
        builder.visit(document)
        let output = builder.finish()
        guard output.length > 0 else { return fallback(source) }
        return Result(attributedString: output, usedFallback: false)
    }

    /// The rendered text with the in-paragraph line separators the renderer
    /// uses for soft breaks and `<br>` turned back into newlines.
    public func plainText(_ source: String) -> String {
        render(source).attributedString.string
            .replacingOccurrences(of: "\u{2028}", with: "\n")
    }

    // MARK: - Source rewrites

    /// The Markdown handed to the parser. What the model wrote is kept; the
    /// rewrites only cover Markdown that the model means one way and
    /// CommonMark reads another, each seen in real Gemma output:
    ///
    /// - `は**「語」**、` and `**語（注）**に` are not emphases: an opener
    ///   followed by punctuation must itself follow whitespace or
    ///   punctuation, and a closer preceded by punctuation must be followed
    ///   by whitespace or punctuation — rules written for languages with
    ///   spaces. Japanese runs brackets straight into kana, so the `**`
    ///   would stay on screen. Punctuation at either end of the emphasized
    ///   text moves outside the markers, where the rules hold; the
    ///   brackets lose their bold and nothing else changes.
    /// - A bold-only line directly followed by text becomes its own
    ///   paragraph, the way the model meant it as a heading.
    ///
    /// Fenced code is left untouched.
    static func presentationSource(for source: String) -> String {
        var lines: [String] = []
        var inFence = false
        for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                inFence.toggle()
                lines.append(String(line))
                continue
            }
            if inFence {
                lines.append(String(line))
                continue
            }
            lines.append(String(line).replacingOccurrences(
                of: #"\*\*(\p{P}*)([^*\n\p{P}\s](?:[^*\n]*?[^*\n\p{P}\s])?)(\p{P}*)\*\*"#,
                with: "$1**$2**$3",
                options: .regularExpression))
        }
        return lines.joined(separator: "\n").replacingOccurrences(
            of: #"(?m)^([ \t]*\*\*[^*\n]+\*\*[ \t]*)\n(?=\S)"#,
            with: "$1\n\n",
            options: .regularExpression)
    }

    private func fallback(_ source: String) -> Result {
        Result(
            attributedString: NSAttributedString(
                string: source,
                attributes: Style.baseAttributes()),
            usedFallback: true)
    }
}

// MARK: - Styling

/// Fonts, colors and paragraph styles: the one place the look is decided.
@MainActor
private enum Style {
    enum BlockKind: Equatable {
        case paragraph
        case heading(Int)
        case code
        case quote
        case list(indent: Int)
        case thematicBreak
    }

    struct Inline {
        var bold = false
        var italic = false
        var code = false
        var strikethrough = false
        var link = false
        /// Where the link goes; set only for a destination that parses as
        /// a URL, so a click opens exactly what the model wrote.
        var linkURL: URL?
    }

    static func baseAttributes() -> [NSAttributedString.Key: Any] {
        [
            .font: NSFont.systemFont(ofSize: NSFont.systemFontSize),
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: paragraphStyle(for: .paragraph),
        ]
    }

    static func attributes(inline: Inline, block: BlockKind) -> [NSAttributedString.Key: Any] {
        var values = baseAttributes()
        values[.paragraphStyle] = paragraphStyle(for: block)
        values[.font] = font(for: block, inline: inline)
        if block == .quote {
            values[.foregroundColor] = NSColor.secondaryLabelColor
        }
        if block == .code || inline.code {
            values[.backgroundColor] = NSColor.controlBackgroundColor
        }
        if inline.strikethrough {
            values[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
        }
        if inline.link {
            values[.foregroundColor] = NSColor.linkColor
            values[.underlineStyle] = NSUnderlineStyle.single.rawValue
            // A click opens the URL in the browser (the text view handles
            // `.link`). References without a usable URL only look like one.
            if let url = inline.linkURL {
                values[.link] = url
            }
        }
        return values
    }

    static func font(for block: BlockKind, inline: Inline) -> NSFont {
        if block == .code || inline.code {
            return NSFont.monospacedSystemFont(
                ofSize: NSFont.systemFontSize - 0.5,
                weight: .regular)
        }
        let size: CGFloat
        let baseWeight: NSFont.Weight
        switch block {
        case .heading(let level):
            size = max(NSFont.systemFontSize + 1, 22 - CGFloat(level - 1) * 2)
            baseWeight = .semibold
        default:
            size = NSFont.systemFontSize
            baseWeight = .regular
        }
        var font = NSFont.systemFont(ofSize: size, weight: inline.bold ? .semibold : baseWeight)
        if inline.italic {
            let descriptor = font.fontDescriptor.withSymbolicTraits(.italic)
            font = NSFont(descriptor: descriptor, size: size) ?? font
        }
        return font
    }

    static func paragraphStyle(for block: BlockKind) -> NSParagraphStyle {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        style.paragraphSpacing = 6
        switch block {
        case .heading:
            style.paragraphSpacingBefore = 8
            style.paragraphSpacing = 4
        case .code:
            // Every code line is its own paragraph (real newlines, so a copy
            // pastes as code), so paragraph spacing would open up the block.
            style.firstLineHeadIndent = 10
            style.headIndent = 10
            style.tailIndent = -10
            style.paragraphSpacingBefore = 0
            style.paragraphSpacing = 0
            style.lineSpacing = 2
        case .quote:
            style.firstLineHeadIndent = 4
            style.headIndent = 20
            style.tailIndent = -8
            style.tabStops = [NSTextTab(textAlignment: .left, location: 16)]
        case .list(let indent):
            let base = CGFloat(22 + indent * 18)
            style.firstLineHeadIndent = CGFloat(indent * 18)
            style.headIndent = base
            style.tabStops = [NSTextTab(textAlignment: .left, location: base)]
            style.paragraphSpacing = 2
        case .thematicBreak:
            style.alignment = .center
            style.paragraphSpacingBefore = 8
            style.paragraphSpacing = 8
        case .paragraph:
            break
        }
        return style
    }

    /// One table cell: its block, borders and padding, header cells tinted.
    static func tableCellStyle(table: NSTextTable, row: Int, column: Int,
                               isHeader: Bool) -> NSParagraphStyle {
        let block = NSTextTableBlock(table: table, startingRow: row, rowSpan: 1,
                                     startingColumn: column, columnSpan: 1)
        block.setWidth(1, type: .absoluteValueType, for: .border)
        block.setBorderColor(NSColor.separatorColor)
        block.setWidth(4, type: .absoluteValueType, for: .padding, edge: .minY)
        block.setWidth(4, type: .absoluteValueType, for: .padding, edge: .maxY)
        block.setWidth(8, type: .absoluteValueType, for: .padding, edge: .minX)
        block.setWidth(8, type: .absoluteValueType, for: .padding, edge: .maxX)
        if isHeader {
            block.backgroundColor = NSColor.controlBackgroundColor
        }
        let style = NSMutableParagraphStyle()
        style.textBlocks = [block]
        style.lineSpacing = 2
        style.paragraphSpacing = 0
        return style
    }
}

// MARK: - AST walk

/// Walks the document once and appends to `output`. Block nodes decide
/// separators, prefixes and paragraph styles; inline nodes push and pop the
/// inline state that text runs are styled with.
@MainActor
private struct Builder: @MainActor MarkupVisitor {
    typealias Result = Void

    private var output = NSMutableAttributedString()
    private var inline = Style.Inline()
    private var quoteDepth = 0
    private var listDepth = 0
    private var previousWasListItem = false
    /// Set by a list item so the paragraph that carries its text continues
    /// the prefixed line instead of opening a new block.
    private var joinNextBlockToPrefix = false

    func finish() -> NSMutableAttributedString { output }

    // MARK: Blocks

    private var currentBlock: Style.BlockKind {
        if listDepth > 0 { return .list(indent: listDepth - 1) }
        if quoteDepth > 0 { return .quote }
        return .paragraph
    }

    /// Opens a block: a blank line between blocks, a single line break
    /// between list items, nothing before the text of a freshly prefixed
    /// list item.
    private mutating func beginBlock(isListItem: Bool = false) {
        if joinNextBlockToPrefix {
            joinNextBlockToPrefix = false
            return
        }
        defer { if listDepth == 0 || isListItem { previousWasListItem = isListItem } }
        guard output.length > 0 else { return }
        let required = (isListItem && previousWasListItem) ? 1 : 2
        let trailing = output.string.reversed().prefix { $0 == "\n" }.count
        guard trailing < required else { return }
        output.append(NSAttributedString(
            string: String(repeating: "\n", count: required - trailing),
            attributes: Style.baseAttributes()))
    }

    private mutating func append(_ text: String, block: Style.BlockKind) {
        guard !text.isEmpty else { return }
        output.append(NSAttributedString(
            string: text,
            attributes: Style.attributes(inline: inline, block: block)))
    }

    private mutating func appendQuotePrefixIfNeeded() {
        guard quoteDepth > 0, listDepth == 0 else { return }
        append("│\t", block: .quote)
    }

    mutating func defaultVisit(_ markup: Markup) {
        for child in markup.children {
            visit(child)
        }
    }

    mutating func visitParagraph(_ paragraph: Paragraph) {
        beginBlock()
        appendQuotePrefixIfNeeded()
        let block = currentBlock
        for child in paragraph.children {
            visitInline(child, block: block)
        }
    }

    mutating func visitHeading(_ heading: Heading) {
        beginBlock()
        appendQuotePrefixIfNeeded()
        for child in heading.children {
            visitInline(child, block: .heading(heading.level))
        }
    }

    mutating func visitCodeBlock(_ codeBlock: CodeBlock) {
        beginBlock()
        var code = codeBlock.code
        if !code.hasSuffix("\n") { code += "\n" }
        append(code, block: .code)
    }

    mutating func visitHTMLBlock(_ html: HTMLBlock) {
        // Raw HTML is shown as the text it is; nothing is interpreted.
        beginBlock()
        appendQuotePrefixIfNeeded()
        append(html.rawHTML.trimmingCharacters(in: .newlines), block: currentBlock)
    }

    mutating func visitThematicBreak(_ thematicBreak: ThematicBreak) {
        beginBlock()
        append("────────────────", block: .thematicBreak)
    }

    mutating func visitBlockQuote(_ blockQuote: BlockQuote) {
        quoteDepth += 1
        defer { quoteDepth -= 1 }
        for child in blockQuote.children {
            visit(child)
        }
    }

    mutating func visitUnorderedList(_ list: UnorderedList) {
        visitList(list, ordinal: nil)
    }

    mutating func visitOrderedList(_ list: OrderedList) {
        visitList(list, ordinal: Int(list.startIndex))
    }

    private mutating func visitList(_ list: Markup, ordinal: Int?) {
        listDepth += 1
        defer { listDepth -= 1 }
        var ordinal = ordinal
        for child in list.children {
            guard let item = child as? ListItem else { continue }
            visitListItem(item, ordinal: ordinal)
            if ordinal != nil { ordinal! += 1 }
        }
    }

    private mutating func visitListItem(_ item: ListItem, ordinal: Int?) {
        beginBlock(isListItem: true)
        let block = Style.BlockKind.list(indent: listDepth - 1)
        var prefix = ordinal.map { "\($0).\t" } ?? "•\t"
        if let checkbox = item.checkbox {
            prefix = (checkbox == .checked ? "☑ " : "☐ ") + prefix
        }
        append(prefix, block: block)
        joinNextBlockToPrefix = true
        for child in item.children {
            visit(child)
        }
        joinNextBlockToPrefix = false
    }

    mutating func visitTable(_ table: Table) {
        beginBlock()
        let textTable = NSTextTable()
        let columnCount = max(1, table.maxColumnCount)
        textTable.numberOfColumns = columnCount
        textTable.collapsesBorders = true
        textTable.layoutAlgorithm = .automaticLayoutAlgorithm

        var rowIndex = 0
        appendTableRow(Array(table.head.cells), table: textTable, row: rowIndex,
                       columnCount: columnCount, isHeader: true)
        rowIndex += 1
        for row in table.body.rows {
            appendTableRow(Array(row.cells), table: textTable, row: rowIndex,
                           columnCount: columnCount, isHeader: false)
            rowIndex += 1
        }
    }

    private mutating func appendTableRow(_ cells: [Table.Cell],
                                         table: NSTextTable, row: Int,
                                         columnCount: Int, isHeader: Bool) {
        for column in 0..<columnCount {
            let cell = NSMutableAttributedString()
            if column < cells.count {
                let saved = output
                output = NSMutableAttributedString()
                let savedInline = inline
                if isHeader { inline.bold = true }
                for child in cells[column].children {
                    visitInline(child, block: .paragraph)
                }
                inline = savedInline
                cell.append(output)
                output = saved
            }
            cell.append(NSAttributedString(string: "\n", attributes: Style.baseAttributes()))
            cell.addAttribute(
                .paragraphStyle,
                value: Style.tableCellStyle(table: table, row: row, column: column,
                                            isHeader: isHeader),
                range: NSRange(location: 0, length: cell.length))
            output.append(cell)
        }
    }

    // MARK: Inlines

    /// Inline nodes are visited with the block they sit in, which decides
    /// the base font and paragraph style of their text.
    private mutating func visitInline(_ markup: Markup, block: Style.BlockKind) {
        switch markup {
        case let text as Text:
            // The parser has no autolink extension attached, so a bare URL
            // (the "参照:" lines the prompt asks for) arrives as text and is
            // linked here; inside a link or code it is left alone.
            if inline.link || inline.code {
                append(text.string, block: block)
            } else {
                appendAutolinked(text.string, block: block)
            }
        case is SoftBreak, is LineBreak:
            // A line break inside the paragraph, not a new paragraph: the
            // model's line structure stays visible without the paragraph
            // spacing a newline would add.
            append("\u{2028}", block: block)
        case let code as InlineCode:
            let saved = inline
            inline.code = true
            append(code.code, block: block)
            inline = saved
        case let html as InlineHTML:
            append(Self.displayText(forInlineHTML: html.rawHTML), block: block)
        case is Strong:
            withInline({ $0.bold = true }, markup, block: block)
        case is Emphasis:
            withInline({ $0.italic = true }, markup, block: block)
        case is Strikethrough:
            withInline({ $0.strikethrough = true }, markup, block: block)
        case let link as Link:
            withInline({
                $0.link = true
                $0.linkURL = Self.url(from: link.destination)
            }, markup, block: block)
        case let image as Image:
            // Never fetched. The alt text stands in, styled like a link so
            // it reads as a reference; without alt text the URL does.
            if image.childCount > 0 {
                withInline({ $0.link = true }, image, block: block)
            } else if let source = image.source {
                let saved = inline
                inline.link = true
                append(source, block: block)
                inline = saved
            }
        default:
            for child in markup.children {
                visitInline(child, block: block)
            }
        }
    }

    /// Bare `http(s)://` runs in a text node become links. Trailing
    /// punctuation the model wrote after the URL (a full stop, a closing
    /// bracket, a Japanese comma) stays text.
    private mutating func appendAutolinked(_ text: String, block: Style.BlockKind) {
        var cursor = text.startIndex
        for match in text.matches(of: Self.bareURL) {
            append(String(text[cursor..<match.range.lowerBound]), block: block)
            var url = String(match.output)
            var trailing = ""
            while let last = url.last, Self.trailingPunctuation.contains(last) {
                trailing.insert(last, at: trailing.startIndex)
                url.removeLast()
            }
            if let parsed = Self.url(from: url) {
                let saved = inline
                inline.link = true
                inline.linkURL = parsed
                append(url, block: block)
                inline = saved
            } else {
                append(url, block: block)
            }
            append(trailing, block: block)
            cursor = match.range.upperBound
        }
        append(String(text[cursor...]), block: block)
    }

    // Japanese punctuation ends a URL (a path can carry kana and kanji,
    // so only the punctuation is excluded, not the script).
    nonisolated(unsafe) private static let bareURL = /https?:\/\/[^\s<>"'、。，．（）「」『』【】]+/
    private static let trailingPunctuation: Set<Character> = [".", ",", ";", ":", "!", "?", "。", "、", ")", "]", "）"]

    static func url(from destination: String?) -> URL? {
        guard let destination, let url = URL(string: destination),
              let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
              url.host != nil else { return nil }
        return url
    }

    private mutating func withInline(_ change: (inout Style.Inline) -> Void,
                                     _ markup: Markup, block: Style.BlockKind) {
        let saved = inline
        change(&inline)
        for child in markup.children {
            visitInline(child, block: block)
        }
        inline = saved
    }

    /// Inline HTML stays the literal text the model wrote, except `<br>`,
    /// which the model uses for a line break inside a table cell and which
    /// becomes one (a line separator, so the cell stays one paragraph in its
    /// table block).
    static func displayText(forInlineHTML raw: String) -> String {
        if raw.range(of: #"^<br\s*/?>$"#,
                     options: [.regularExpression, .caseInsensitive]) != nil {
            return "\u{2028}"
        }
        return raw
    }
}
