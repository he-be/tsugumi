import Foundation
import TurboFieldfare

// Phase 5's tool half: the XML tool call, over the real vocabulary
// (`docs/qwen35moe/04-PHASES.md` 次の一手 #22).
//
//   swift run -c release TurboFieldfareKernelCheck \
//     --qwen-tools scratch/ornith-oq4e-g64.gturbo
//
// Three things are on trial, and the interesting cases need all three at once:
//
// 1. `GrammarVocabulary`'s piece table read with **ByteLevel** rules rather
//    than Gemma's. A table built the Gemma way does not fail — it silently
//    stops matching anything non-ASCII, so the negative control is the only
//    thing that would ever catch it.
// 2. `QwenToolCallGrammar` over that table: what the sampler would let the
//    model write.
// 3. `QwenStructuredAssistantDecoder` + `QwenToolCallParser`: what comes back.
//
// The centrepiece is one statement that only holds if all three agree —
// **the template's own rendering of a tool call is accepted by the grammar and
// parses back to the arguments it was rendered from.** The call is not written
// out here; it is produced by `chat_template.jinja` from a `Message` carrying
// `toolCalls`, which is the same rendering the server would send back on the
// next turn (INV-1).
//
// The pure parts — the grammar text, the parser's typing rules — are in
// `swift test` (`QwenChatGrammarBuilderTests`, `QwenToolCallParserTests`).
// What is here is what needs 248,070 real tokens.

private struct ToolsCase {
    let name: String
    let passed: Bool
    let detail: String
}

private func toolsCase(_ name: String, _ passed: Bool, _ detail: String = "") -> ToolsCase {
    ToolsCase(name: name, passed: passed, detail: detail)
}

// MARK: - Fixtures

/// `get_weather(city: string required, days: integer optional)`. `city` is
/// written raw and `days` as JSON — the difference that runs through the whole
/// format.
private let weatherTool = GFTokenizer.FunctionDefinition(
    name: "get_weather",
    description: "Get the weather for a city.",
    parameters: .object([
        "type": .string("object"),
        "properties": .object([
            "city": .object(["type": .string("string")]),
            "days": .object(["type": .string("integer")]),
        ]),
        "required": .array([.string("city")]),
    ]))

/// The arguments each round-trip case renders and expects back. The Japanese
/// and markup cases are the ones a byte-level table or an `[^<]*` value rule
/// would lose.
private let roundTripArguments: [(String, JSONValue)] = [
    ("ascii", .object(["city": .string("Kyoto"), "days": .integer(3)])),
    ("japanese", .object(["city": .string("東京都"), "days": .integer(1)])),
    ("markup", .object(["city": .string("<b>Kyoto</b> & co")])),
    ("partial-closer", .object(["city": .string("a\n</paramete>\nb")])),
    ("multiline", .object(["city": .string("line one\nline two")])),
    ("emoji", .object(["city": .string("Kyoto 🏯")])),
    ("string-only", .object(["city": .string("Osaka")])),
]

// MARK: - The check

func runQwenToolsCheck(modelPath: String) async throws -> Bool {
    let directoryURL = URL(fileURLWithPath: modelPath)
    print("=== ornith tool calls (docs/qwen35moe/04-PHASES 次の一手 #22) ===")
    print("  model  \(directoryURL.path)")

    let tokenizer: QwenTokenizer
    if FileManager.default.isReadableFile(
        atPath: directoryURL.appendingPathComponent("tokenizer.json").path) {
        tokenizer = try await QwenTokenizer.load(from: directoryURL)
    } else {
        tokenizer = try await QwenTokenizer.load(forModelDirectory: directoryURL)
    }

    var cases: [ToolsCase] = []
    let vocabulary = GrammarVocabulary(tokenizer)
    let markers = QwenToolCallMarkers(tokenizer: tokenizer)

    cases.append(contentsOf: pieceTableCases(tokenizer: tokenizer, vocabulary: vocabulary))
    cases.append(contentsOf: try toolsBlockCases(tokenizer: tokenizer))
    cases.append(contentsOf: try roundTripCases(tokenizer: tokenizer,
                                                vocabulary: vocabulary,
                                                markers: markers))
    cases.append(contentsOf: try maskCases(tokenizer: tokenizer,
                                           vocabulary: vocabulary,
                                           markers: markers))
    cases.append(contentsOf: try negativeCases(tokenizer: tokenizer,
                                               vocabulary: vocabulary,
                                               markers: markers))

    let failures = cases.filter { !$0.passed }
    for item in failures { print("  FAIL  \(item.name) — \(item.detail)") }
    let negatives = cases.filter { $0.name.hasPrefix("negative:") }
    if failures.isEmpty {
        print("PASS  \(cases.count) cases, \(negatives.count) of them negative controls")
        for item in negatives { print("    \(item.name): \(item.detail)") }
    } else {
        print("FAIL  \(failures.count)/\(cases.count) cases")
    }
    return failures.isEmpty
}

// MARK: - 1. The piece table

/// The pieces are what the grammar sees. Checking them against the
/// *detokenizer* rather than against the rule that built them makes it a
/// cross-check between two paths and not a restatement of one.
private func pieceTableCases(tokenizer: QwenTokenizer,
                             vocabulary: GrammarVocabulary) -> [ToolsCase] {
    var cases: [ToolsCase] = []
    cases.append(toolsCase("piece table covers the vocabulary",
                           vocabulary.count == tokenizer.vocabSize,
                           "\(vocabulary.count) != \(tokenizer.vocabSize)"))

    // The markers the grammar spells as ids still need a piece: the matcher
    // consumes the trigger's bytes when a lazy grammar fires.
    for (name, id) in [("<tool_call>", tokenizer.toolCallStartID),
                       ("</tool_call>", tokenizer.toolCallEndID),
                       ("</think>", tokenizer.thinkEndID)] {
        let piece = vocabulary.piece(for: id)
        cases.append(toolsCase("piece(\(name)) is its spelling",
                               piece == Array(name.utf8),
                               "got \(String(decoding: piece, as: UTF8.self).debugDescription)"))
    }

    // Concatenating the pieces of a token run must be the bytes the
    // detokenizer produces for that run. This is where a Gemma-rule table
    // parts company with reality.
    let probes = ["こんにちは、世界。", "The quick brown fox.",
                  "def f(x):\n    return x < 3\n", "🏯 京都 🍵", "  spaced   out  "]
    var mismatches: [String] = []
    for text in probes {
        let ids = tokenizer.encode(text)
        var bytes: [UInt8] = []
        for id in ids { bytes.append(contentsOf: vocabulary.piece(for: id)) }
        let assembled = String(decoding: bytes, as: UTF8.self)
        let decoded = tokenizer.decode(ids, skipSpecialTokens: false)
        if assembled != decoded { mismatches.append(text) }
    }
    cases.append(toolsCase("concatenated pieces == decoded text on \(probes.count) probes",
                           mismatches.isEmpty,
                           "differed on \(mismatches.map(\.debugDescription))"))

    // Negative: the same table built with Gemma's rules. It must lose at least
    // the non-ASCII probes — and it must not merely fail, which is the point.
    var gemmaRuleDisagreements = 0
    var firstExample = ""
    for text in probes {
        let ids = tokenizer.encode(text)
        var bytes: [UInt8] = []
        for id in ids {
            guard let token = tokenizer.token(for: Int32(id)) else { continue }
            bytes.append(contentsOf: gemmaRulePiece(token))
        }
        let assembled = String(decoding: bytes, as: UTF8.self)
        let decoded = tokenizer.decode(ids, skipSpecialTokens: false)
        if assembled != decoded {
            gemmaRuleDisagreements += 1
            if firstExample.isEmpty {
                firstExample = "\(assembled.prefix(20).debugDescription) != "
                    + "\(decoded.prefix(20).debugDescription)"
            }
        }
    }
    cases.append(toolsCase(
        "negative: gemmaPieceRules — metaspace rules on a ByteLevel vocabulary",
        gemmaRuleDisagreements > 0,
        gemmaRuleDisagreements > 0
            ? "differed on \(gemmaRuleDisagreements)/\(probes.count) probes, e.g. \(firstExample)"
            : "reproduced every probe — this control has no detection power"))
    return cases
}

/// `GemmaDecoding`'s two rules, reproduced here rather than reached for: a
/// `<0xXX>` token is that one byte, and anything else is its spelling with the
/// metaspace `▁` turned back into a space. Kept local so the control stays a
/// *plausible other implementation* and does not widen the runtime's API.
private func gemmaRulePiece(_ token: String) -> [UInt8] {
    if token.hasPrefix("<0x"), token.hasSuffix(">"), token.count == 6,
       let byte = UInt8(token.dropFirst(3).dropLast(), radix: 16) {
        return [byte]
    }
    return Array(token.replacingOccurrences(of: "\u{2581}", with: " ").utf8)
}

// MARK: - 1b. The `<tools>` block the template writes

/// What `{{- tool | tojson }}` produces on this host.
///
/// `docs/qwen35moe/22-PHASE5-TOKENIZER.md` §4-2 recorded that the tools block
/// differs from Python's and called swift-jinja's key order "unordered, because
/// the spec arrives as a dictionary". **That is wrong, and the correction
/// matters**: swift-jinja sorts. `Jinja.Value(any:)` sorts a Swift dictionary's
/// keys on the way in, and `tojson` encodes with `.sortedKeys` on the way out,
/// so the rendering is deterministic — it is *ascending key order with compact
/// separators*, against Python's *insertion order with `", "` / `": "`*.
///
/// Two fixed spellings, not one fixed and one arbitrary. That is what makes the
/// arguments' ordering usable at all: the grammar can emit parameters in
/// ascending key order and the re-render will agree with it (§`parameterBlocks`).
private func toolsBlockCases(tokenizer: QwenTokenizer) throws -> [ToolsCase] {
    let ids = try tokenizer.applyChatTemplate(
        [.init(role: .user, content: "What is the weather?")],
        tools: [weatherTool],
        enableThinking: false)
    let text = tokenizer.decode(ids, skipSpecialTokens: false)
    guard let open = text.range(of: "<tools>\n"),
          let close = text.range(of: "\n</tools>") else {
        return [toolsCase("the template wrote a <tools> block", false, "no block found")]
    }
    let line = String(text[open.upperBound..<close.lowerBound])
    print("  <tools> line  \(line)")
    let keysAreSorted = line.range(of: #""description":"#).map { descriptionRange in
        line.range(of: #""name":"#).map { $0.lowerBound > descriptionRange.lowerBound } ?? false
    } ?? false
    return [
        toolsCase("the <tools> line is compact (no space after `:` or `,`)",
                  !line.contains("\": ") && !line.contains("\", \""),
                  "got \(line)"),
        toolsCase("the <tools> line is in ascending key order (description before name)",
                  keysAreSorted,
                  "got \(line)"),
    ]
}

// MARK: - 2. Template → grammar → parser

/// Render one assistant turn holding a tool call, and hand the ids the template
/// produced to the grammar and to the decoder.
private func roundTripCases(tokenizer: QwenTokenizer,
                            vocabulary: GrammarVocabulary,
                            markers: QwenToolCallMarkers) throws -> [ToolsCase] {
    var cases: [ToolsCase] = []
    let grammarText = QwenToolCallGrammar.grammar(tools: [weatherTool],
                                                  parallelToolCalls: true,
                                                  withPreamble: false,
                                                  markers: markers).grammar
    let grammar = try GBNFGrammar(grammarText)

    for (name, arguments) in roundTripArguments {
        guard let span = try renderedCallSpan(tokenizer: tokenizer,
                                              arguments: arguments) else {
            cases.append(toolsCase("round-trip[\(name)] rendered a call", false,
                                   "the template wrote no <tool_call> section"))
            continue
        }
        // The rendering itself, for the record — a reader can line it up with
        // `chat_template.jinja`.
        let text = tokenizer.decode(span, skipSpecialTokens: false)
        cases.append(toolsCase("round-trip[\(name)] rendered the XML form",
                               text.hasPrefix("<tool_call>\n<function=get_weather>\n")
                                   && text.hasSuffix("</function>\n</tool_call>"),
                               "got \(text.debugDescription)"))

        // The grammar, fed the way the sampler feeds it.
        let constraint = try GrammarTokenConstraint(
            grammar: grammar,
            vocabulary: vocabulary,
            trigger: .token(markers.toolCallStartTokenID))
        var accepted = true
        var rejectedAt = -1
        for (index, id) in span.enumerated() {
            if !constraint.allows(tokenID: id) {
                accepted = false
                rejectedAt = index
                break
            }
            do { try constraint.accept(tokenID: id) } catch {
                accepted = false
                rejectedAt = index
                break
            }
        }
        cases.append(toolsCase("round-trip[\(name)] the grammar accepts the rendering",
                               accepted && constraint.mayEndHere,
                               accepted
                                   ? "accepted but the grammar is incomplete"
                                   : "rejected token \(rejectedAt) of \(span.count) "
                                       + "(\(tokenizer.token(for: span[rejectedAt]) ?? "?"))"))

        // The decoder, fed the same ids.
        let parsed = try? parseCall(span, tokenizer: tokenizer)
        cases.append(toolsCase("round-trip[\(name)] parses back to the same arguments",
                               parsed?.arguments == arguments,
                               "got \(parsed.map { "\($0.arguments)" } ?? "nothing")"))
    }
    return cases
}

/// `chat_template.jinja`'s rendering of an assistant turn holding one tool
/// call, sliced to the section the model would have generated.
private func renderedCallSpan(tokenizer: QwenTokenizer,
                              arguments: JSONValue) throws -> [Int32]? {
    let call = GFTokenizer.HistoricalToolCall(id: "call_fixed",
                                              name: weatherTool.name,
                                              arguments: arguments)
    let messages: [GFTokenizer.Message] = [
        .init(role: .user, content: "What is the weather?"),
        .init(role: .assistant, content: "", toolCalls: [call]),
    ]
    let ids = try tokenizer.applyChatTemplate(messages,
                                              tools: [weatherTool],
                                              enableThinking: false)
    // **The last pair, not the first.** With `tools` set, the template writes
    // the calling convention into the system prompt as a worked example —
    // `<tool_call>\n<function=example_function_name>\n…` — and those markers
    // tokenize as the marker tokens themselves. Slicing from the first one
    // hands the grammar the instructions instead of the call.
    guard let start = ids.lastIndex(of: tokenizer.toolCallStartID),
          let end = ids.lastIndex(of: tokenizer.toolCallEndID),
          start < end else {
        return nil
    }
    return Array(ids[start...end])
}

/// The generation-loop wiring: a streaming detokenizer feeding the structured
/// decoder, exactly as a caller would run it.
private func parseCall(_ ids: [Int32], tokenizer: QwenTokenizer) throws -> ParsedToolCall {
    let decoder = QwenStructuredAssistantDecoder(tokenizer: tokenizer,
                                                 tools: [weatherTool],
                                                 emitsReasoning: true,
                                                 startsInReasoning: false,
                                                 idGenerator: { "call_fixed" })
    var detokenizer = tokenizer.makeDetokenizer(skipSpecialTokens: true)
    var calls: [ParsedToolCall] = []
    for id in ids {
        for event in try decoder.consume(tokenID: id, delta: detokenizer.push(id)) {
            if case .toolCall(let call) = event { calls.append(call) }
        }
    }
    for event in try decoder.consumeTail(detokenizer.flush()) {
        if case .toolCall(let call) = event { calls.append(call) }
    }
    try decoder.finish()
    guard calls.count == 1 else { throw QwenToolCallParserError.malformed }
    return calls[0]
}

// MARK: - 3. The mask

/// `fillAllowedMask` is the path generation actually takes; `allows(tokenID:)`
/// is the one everything is reasoned about. They have to be the same set.
private func maskCases(tokenizer: QwenTokenizer,
                       vocabulary: GrammarVocabulary,
                       markers: QwenToolCallMarkers) throws -> [ToolsCase] {
    let grammarText = QwenToolCallGrammar.grammar(tools: [weatherTool],
                                                  parallelToolCalls: false,
                                                  withPreamble: false,
                                                  markers: markers).grammar
    guard let span = try renderedCallSpan(
        tokenizer: tokenizer,
        arguments: .object(["city": .string("Kyoto"), "days": .integer(3)])) else {
        return [toolsCase("mask agrees with allows()", false, "no rendering to walk")]
    }
    let constraint = try GrammarTokenConstraint(
        grammar: try GBNFGrammar(grammarText),
        vocabulary: vocabulary,
        trigger: .token(markers.toolCallStartTokenID))

    var allowed = [Bool](repeating: false, count: tokenizer.vocabSize)
    var disagreements = 0
    var positionsChecked = 0
    var refusedAt = -1
    // Every position is 248k `allows` calls, so a handful of them: the start,
    // the middle of a raw value, and the end.
    let sampled = Set([0, span.count / 3, span.count / 2, span.count - 2])
    for (index, id) in span.enumerated() {
        if sampled.contains(index) {
            positionsChecked += 1
            try allowed.withUnsafeMutableBufferPointer {
                try constraint.fillAllowedMask($0)
            }
            for candidate in 0..<tokenizer.vocabSize
            where allowed[candidate] != constraint.allows(tokenID: Int32(candidate)) {
                disagreements += 1
            }
        }
        guard (try? constraint.accept(tokenID: id)) != nil else {
            refusedAt = index
            break
        }
    }
    return [toolsCase(
        "fillAllowedMask == allows() over the whole vocabulary "
        + "(\(positionsChecked) positions)",
        disagreements == 0 && refusedAt < 0,
        refusedAt >= 0
            ? "the grammar refused token \(refusedAt) before the walk finished"
            : "\(disagreements) token(s) disagreed")]
}

// MARK: - 4. Negative controls

private func negativeCases(tokenizer: QwenTokenizer,
                           vocabulary: GrammarVocabulary,
                           markers: QwenToolCallMarkers) throws -> [ToolsCase] {
    var cases: [ToolsCase] = []
    // **The negatives are walked against the non-lazy grammar**, and that is
    // itself a finding rather than a convenience. A lazy grammar (GEN-5) is not
    // applied until its trigger token fires, so a run that never contains the
    // marker token is not constrained at all — `allows` returns true for
    // everything and `mayEndHere` is true. Nothing a lazy grammar can do would
    // refuse a call whose markers are spelled as text; only `tool_choice:
    // required`, which is applied from the first generated token, can.
    let lazyGrammar = try GBNFGrammar(
        QwenToolCallGrammar.grammar(tools: [weatherTool],
                                    parallelToolCalls: true,
                                    withPreamble: false,
                                    markers: markers).grammar)
    let grammar = try GBNFGrammar(
        QwenToolCallGrammar.grammar(tools: [weatherTool],
                                    parallelToolCalls: true,
                                    withPreamble: true,
                                    markers: markers).grammar)

    /// Walk a token run and say whether the grammar took all of it and could
    /// stop there.
    func accepts(_ ids: [Int32], lazy: Bool = false) -> Bool {
        guard let constraint = try? GrammarTokenConstraint(
            grammar: lazy ? lazyGrammar : grammar,
            vocabulary: vocabulary,
            trigger: lazy ? .token(markers.toolCallStartTokenID) : nil) else { return false }
        for id in ids {
            guard constraint.allows(tokenID: id),
                  (try? constraint.accept(tokenID: id)) != nil else { return false }
        }
        return constraint.mayEndHere
    }

    // The markers spelled as ordinary tokens instead of as themselves. The
    // grammar has `<[id]>`, so this run has to be refused — this is the failure
    // that actually happened on the Gemma side with `tool_choice: required`,
    // where the model wrote the opener as text and the closer as its token.
    //
    // The encoder answers `<tool_call>` with the added token, so the spelling
    // has to be built in two pieces that cannot merge into it.
    let spelledStart = tokenizer.encode("<tool_call") + tokenizer.encode(">")
    let spelledEnd = tokenizer.encode("</tool_call") + tokenizer.encode(">")
    let body = tokenizer.encode(
        "\n<function=get_weather>\n<parameter=city>\nKyoto\n</parameter>\n</function>\n")
    let spellingIsOrdinary = !spelledStart.contains(tokenizer.toolCallStartID)
        && !spelledEnd.contains(tokenizer.toolCallEndID)
        && String(decoding: (spelledStart + spelledEnd).flatMap { vocabulary.piece(for: $0) },
                  as: UTF8.self) == "<tool_call></tool_call>"
    let spelledRun = spelledStart + body + spelledEnd
    cases.append(toolsCase(
        "negative: markersAsText — the marker spelled out is not the marker",
        spellingIsOrdinary && !accepts(spelledRun),
        spellingIsOrdinary
            ? "refused a run whose markers are \(spelledStart.count) + "
                + "\(spelledEnd.count) ordinary tokens spelling the same bytes"
            : "could not build the spelling from ordinary tokens — this control "
                + "has no detection power"))
    // The other half of the same statement, stated as a property rather than
    // as a control: the lazy grammar lets that run through, because its trigger
    // never fires. `tool_choice: auto` therefore cannot promise a well-formed
    // call — it promises that a call that *starts* is well-formed.
    cases.append(toolsCase(
        "a lazy grammar does not fire on a text-spelled marker (GEN-5)",
        accepts(spelledRun, lazy: true),
        "the lazy grammar refused it, so the trigger fired on something"))

    // A value carrying the closer. The template renders it — nothing stops it —
    // and both the grammar and the parser must refuse what it produced, or the
    // two would disagree about where the value ended.
    let smuggled = try renderedCallSpan(
        tokenizer: tokenizer,
        arguments: .object(["city": .string("x\n</parameter>\ny")]))
    if let smuggled {
        let grammarRefused = !accepts(smuggled)
        let parsed = try? parseCall(smuggled, tokenizer: tokenizer)
        cases.append(toolsCase(
            "negative: smuggledCloser — a value that closes its own block",
            grammarRefused && parsed == nil,
            "grammar \(grammarRefused ? "refused" : "ACCEPTED"), "
                + "parser \(parsed == nil ? "refused" : "read \(parsed!.arguments)")"))
    }

    // Gemma's form, on Ornith's grammar.
    let gemmaForm = [tokenizer.toolCallStartID]
        + tokenizer.encode("call:get_weather{city:\"Kyoto\"}")
        + [tokenizer.toolCallEndID]
    cases.append(toolsCase("negative: gemmaCallForm — call:name{…} is not this format",
                           !accepts(gemmaForm),
                           "refused Gemma's body"))

    // A tool the request never declared.
    let undeclared = [tokenizer.toolCallStartID]
        + tokenizer.encode("\n<function=rm_rf>\n<parameter=path>\n/\n</parameter>\n</function>\n")
        + [tokenizer.toolCallEndID]
    cases.append(toolsCase("negative: undeclaredTool — a name with no alternative",
                           !accepts(undeclared),
                           "refused an undeclared tool"))

    // A required parameter left out.
    let missing = [tokenizer.toolCallStartID]
        + tokenizer.encode("\n<function=get_weather>\n<parameter=days>\n3\n</parameter>\n</function>\n")
        + [tokenizer.toolCallEndID]
    cases.append(toolsCase("negative: missingRequired — `city` is not optional",
                           !accepts(missing),
                           "refused a call without its required parameter"))

    return cases
}
