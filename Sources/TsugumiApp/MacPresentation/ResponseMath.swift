import AppKit
import SwiftMath

/// TeX math in an answer (`$…$`, `$$…$$`, `\(…\)`, `\[…\]`), the way Gemma
/// writes it. The spans are lifted out of the Markdown before parsing —
/// `x_1 * x_2` must not become emphasis — and drawn with SwiftMath (Latin
/// Modern Math, CoreText) as attachments the text view lays out like
/// glyphs. Copying gives back the TeX as written.
enum ResponseMath {
    struct Span: Equatable {
        /// As written, delimiters included: what a copy of the answer gets.
        let source: String
        let latex: String
        let display: Bool
    }

    /// The attribute an attachment or fallback run carries so `plainText`
    /// can put the TeX back.
    static let sourceKey = NSAttributedString.Key("TsugumiMathSource")

    private static let open: Character = "\u{E000}"
    private static let close: Character = "\u{E001}"
    nonisolated(unsafe) static let placeholder = /\u{E000}(\d+)\u{E001}/

    static func placeholder(_ index: Int) -> String {
        "\(open)\(index)\(close)"
    }

    // MARK: Extraction

    /// Replaces each math span with a placeholder the Markdown parser passes
    /// through as text. Fenced and inline code are left alone. A `$` opens
    /// inline math only when the next character is not a space and a
    /// closing `$` follows on the same line, itself not preceded by a space
    /// nor followed by a digit — so "$5 and $10" stays money.
    static func extract(from source: String) -> (text: String, spans: [Span]) {
        let chars = Array(source)
        var text = ""
        var spans: [Span] = []
        var i = 0
        var atLineStart = true
        var inFence = false

        func lineRest(_ from: Int) -> String {
            var end = from
            while end < chars.count, chars[end] != "\n" { end += 1 }
            return String(chars[from..<end])
        }
        func find(_ closing: [Character], from start: Int, sameLine: Bool) -> Int? {
            var j = start
            while j + closing.count <= chars.count {
                if sameLine, chars[j] == "\n" { return nil }
                if Array(chars[j..<(j + closing.count)]) == closing { return j }
                j += 1
            }
            return nil
        }
        func take(_ n: Int) {
            text.append(contentsOf: chars[i..<(i + n)])
            i += n
        }
        func emit(_ span: Span) {
            spans.append(span)
            text.append(placeholder(spans.count - 1))
        }

        while i < chars.count {
            let c = chars[i]
            if atLineStart, lineRest(i).trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                inFence.toggle()
            }
            atLineStart = c == "\n"
            if inFence {
                take(1)
                continue
            }
            if c == "`" {
                // A code span: copy through its closing run of the same length.
                var run = 0
                while i + run < chars.count, chars[i + run] == "`" { run += 1 }
                let closing = Array(repeating: Character("`"), count: run)
                if let end = find(closing, from: i + run, sameLine: false) {
                    take(end + run - i)
                } else {
                    take(run)
                }
                continue
            }
            if c == "\\", i + 1 < chars.count {
                let next = chars[i + 1]
                if next == "$" { take(2); continue }
                if next == "[" || next == "(" {
                    let closing: [Character] = next == "[" ? ["\\", "]"] : ["\\", ")"]
                    if let end = find(closing, from: i + 2, sameLine: false) {
                        let latex = String(chars[(i + 2)..<end])
                        emit(Span(source: String(chars[i..<(end + 2)]),
                                  latex: latex.trimmingCharacters(in: .whitespacesAndNewlines),
                                  display: next == "["))
                        i = end + 2
                        continue
                    }
                }
                take(1)
                continue
            }
            if c == "$" {
                if i + 1 < chars.count, chars[i + 1] == "$" {
                    if let end = find(["$", "$"], from: i + 2, sameLine: false),
                       end > i + 2 {
                        let latex = String(chars[(i + 2)..<end])
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        if !latex.isEmpty {
                            emit(Span(source: String(chars[i..<(end + 2)]), latex: latex, display: true))
                            i = end + 2
                            continue
                        }
                    }
                    take(2)
                    continue
                }
                if i + 1 < chars.count, !chars[i + 1].isWhitespace, chars[i + 1] != "$" {
                    // A code span outranks the math: "$10, plus `$HOME`"
                    // is not a formula.
                    var j = i + 1
                    while j < chars.count, chars[j] != "\n", chars[j] != "`" {
                        if chars[j] == "$", !chars[j - 1].isWhitespace,
                           j + 1 >= chars.count || !chars[j + 1].isNumber {
                            break
                        }
                        j += 1
                    }
                    if j < chars.count, chars[j] == "$" {
                        emit(Span(source: String(chars[i...j]),
                                  latex: String(chars[(i + 1)..<j]), display: false))
                        i = j + 1
                        continue
                    }
                }
                take(1)
                continue
            }
            take(1)
        }
        return (text, spans)
    }

    // MARK: Rendering

    /// Latin Modern has no CJK glyphs; `\text{定数項}` is drawn as text.
    static func hasCJK(_ latex: String) -> Bool {
        latex.unicodeScalars.contains { scalar in
            scalar.properties.isIdeographic
                || (0x3040...0x30FF).contains(scalar.value)
                || (0xFF00...0xFFEF).contains(scalar.value)
        }
    }

    /// The span as an attachment, or nil when SwiftMath cannot typeset it
    /// (then `fallbackText` shows it as text). SwiftMath bakes the colour
    /// in when it draws, so the math is drawn once per appearance and the
    /// attachment's image picks the one for the window at draw time.
    @MainActor
    static func attachment(for span: Span, baseFontSize: CGFloat) -> NSTextAttachment? {
        guard !hasCJK(span.latex) else { return nil }
        let size = span.display ? baseFontSize + 2 : baseFontSize
        let inset: CGFloat = span.display ? 2 : 1
        func draw(_ appearance: NSAppearance.Name) -> (NSImage, MathImage.LayoutInfo)? {
            var result: (NSImage, MathImage.LayoutInfo)?
            NSAppearance(named: appearance)?.performAsCurrentDrawingAppearance {
                var math = MathImage(latex: span.latex, fontSize: size, textColor: .labelColor,
                                     labelMode: span.display ? .display : .text, textAlignment: .left)
                math.contentInsets = MTEdgeInsets(top: 0, left: inset, bottom: 0, right: inset)
                let (error, image, layout) = math.asImage()
                if error == nil, let image, let layout { result = (image, layout) }
            }
            return result
        }
        guard let (light, layout) = draw(.aqua), let (dark, _) = draw(.darkAqua) else { return nil }
        let bounds = light.size
        let image = NSImage(size: bounds, flipped: false) { rect in
            let isDark = NSAppearance.currentDrawing().bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            (isDark ? dark : light).draw(in: rect)
            return true
        }
        let attachment = NSTextAttachment()
        attachment.image = image
        // Baseline on the text's baseline: the image's bottom edge is the
        // math's descent below it.
        attachment.bounds = CGRect(x: 0, y: -ceil(layout.descent), width: bounds.width, height: bounds.height)
        return attachment
    }

    /// Readable text for a span that is not drawn: the common commands as
    /// their symbols, `\text{…}` and braces dropped.
    static func fallbackText(for span: Span) -> String {
        var s = span.latex
        s = s.replacingOccurrences(of: #"\\text\{([^}]*)\}"#, with: "$1", options: .regularExpression)
        s = s.replacingOccurrences(of: #"\\frac\{([^{}]*)\}\{([^{}]*)\}"#, with: "($1)/($2)", options: .regularExpression)
        s = s.replacingOccurrences(of: #"\\sqrt\{([^{}]*)\}"#, with: "√($1)", options: .regularExpression)
        for (command, symbol) in symbols {
            s = s.replacingOccurrences(of: "\\\(command)", with: symbol)
        }
        s = s.replacingOccurrences(of: #"\^\{([^{}]*)\}"#, with: "^($1)", options: .regularExpression)
        s = s.replacingOccurrences(of: #"_\{([^{}]*)\}"#, with: "_($1)", options: .regularExpression)
        s = s.replacingOccurrences(of: #"\\([A-Za-z]+)"#, with: "$1", options: .regularExpression)
        s = s.replacingOccurrences(of: "{", with: "").replacingOccurrences(of: "}", with: "")
        return s.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
    }

    private static let symbols: [(String, String)] = [
        ("implies", "⟹"), ("iff", "⟺"), ("Rightarrow", "⇒"), ("Leftrightarrow", "⇔"),
        ("rightarrow", "→"), ("to", "→"), ("leftarrow", "←"),
        ("neq", "≠"), ("ne", "≠"), ("leq", "≤"), ("le", "≤"), ("geq", "≥"), ("ge", "≥"),
        ("approx", "≈"), ("equiv", "≡"), ("pm", "±"), ("mp", "∓"), ("times", "×"), ("cdot", "·"),
        ("div", "÷"), ("infty", "∞"), ("sum", "Σ"), ("prod", "Π"), ("int", "∫"), ("partial", "∂"),
        ("nabla", "∇"), ("in", "∈"), ("notin", "∉"), ("subset", "⊂"), ("cup", "∪"), ("cap", "∩"),
        ("forall", "∀"), ("exists", "∃"), ("ldots", "…"), ("cdots", "⋯"), ("dots", "…"),
        ("alpha", "α"), ("beta", "β"), ("gamma", "γ"), ("delta", "δ"), ("epsilon", "ε"),
        ("varepsilon", "ε"), ("zeta", "ζ"), ("eta", "η"), ("theta", "θ"), ("iota", "ι"),
        ("kappa", "κ"), ("lambda", "λ"), ("mu", "μ"), ("nu", "ν"), ("xi", "ξ"), ("pi", "π"),
        ("rho", "ρ"), ("sigma", "σ"), ("tau", "τ"), ("upsilon", "υ"), ("phi", "φ"), ("varphi", "φ"),
        ("chi", "χ"), ("psi", "ψ"), ("omega", "ω"), ("Gamma", "Γ"), ("Delta", "Δ"), ("Theta", "Θ"),
        ("Lambda", "Λ"), ("Xi", "Ξ"), ("Pi", "Π"), ("Sigma", "Σ"), ("Phi", "Φ"), ("Psi", "Ψ"),
        ("Omega", "Ω"), ("left", ""), ("right", ""), ("quad", " "), (",", " "), (" ", " "),
    ]
}
