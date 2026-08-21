import Foundation
import TurboFieldfare

// Phase 5's tokenizer half: the Swift side reproduces the upstream tokenizer
// on this checkpoint's own files (`docs/qwen35moe/04-PHASES.md` Phase 5).
//
//   ~/LLM/venv/bin/python3 Scripts/qwen35/tokenizer_fixture.py \
//     --tokenizer scratch/ornith-oq4e-g64.gturbo/tokenizer \
//     --out scratch/qwen35/tokenizer-fixture.json
//
//   swift run -c release TurboFieldfareKernelCheck \
//     --qwen-tokenizer scratch/ornith-oq4e-g64.gturbo \
//     --qwen-tokenizer-fixture scratch/qwen35/tokenizer-fixture.json
//
// The fixture is `tokenizers` / `transformers` run once on this machine over
// the same `tokenizer.json` — encode, decode (special kept and dropped), and
// the rendered `chat_template.jinja`. Two implementations reading one file:
// they can disagree, and where they disagree is the finding.
//
// Decode is where the family differs most (`ByteLevelDecoding`). Every token is
// bytes, so a codepoint can — and for Japanese routinely does — straddle two
// tokens, and the fixture carries cases built to straddle: a token ending in a
// UTF-8 lead byte followed by one starting with continuations, with and without
// a marker wedged between them. Those are the cases that separate the rule the
// library actually follows (drop the special *before* decoding, so the byte run
// fuses across it) from the plausible one (close the run at every marker).
//
// The negative controls are four deliberately wrong detokenizers run over the
// same fixture. Each must disagree with it somewhere.
//
// The declaration checks (this tokenizer is not Gemma's, and Gemma's is not
// this one) need no vocabulary at all and live in the test target instead:
// `QwenDecoderConfigTests`.

private enum QwenTokenizerCheckError: Error, CustomStringConvertible {
    case badFixture(String)

    var description: String {
        switch self {
        case .badFixture(let detail): return "fixture: \(detail)"
        }
    }
}

// MARK: - Fixture

private struct QwenTokenizerFixture: Decodable {
    struct AddedToken: Decodable {
        let id: Int32
        let content: String
        let special: Bool
    }

    struct EncodeCase: Decodable {
        let text: String
        let ids: [Int32]
        let decoded: String
    }

    struct DecodeCase: Decodable {
        let ids: [Int32]
        let keep: String
        let skip: String
    }

    struct ChatCase: Decodable {
        let name: String
        let enableThinking: Bool
        let text: String
        let ids: [Int32]
    }

    let vocabSize: Int
    let baseVocabSize: Int
    let addedTokens: [AddedToken]
    let encode: [EncodeCase]
    let decode: [DecodeCase]
    let chat: [ChatCase]

    static func load(_ path: String) throws -> QwenTokenizerFixture {
        let url = URL(fileURLWithPath: path)
        guard let data = try? Data(contentsOf: url) else {
            throw QwenTokenizerCheckError.badFixture("cannot read \(url.path)")
        }
        return try JSONDecoder().decode(QwenTokenizerFixture.self, from: data)
    }
}

// MARK: - Case plumbing

private struct TextCase {
    let name: String
    let passed: Bool
    let detail: String
}

private func textCase(_ name: String, _ passed: Bool, _ detail: String = "") -> TextCase {
    TextCase(name: name, passed: passed, detail: detail)
}

/// Equality with the difference located: the first index where two strings part
/// company, printed with a little context on each side.
private func equalCase(_ name: String, _ got: String, _ want: String) -> TextCase {
    if got == want { return textCase(name, true) }
    let g = Array(got), w = Array(want)
    var i = 0
    while i < min(g.count, w.count), g[i] == w[i] { i += 1 }
    let from = max(0, i - 12)
    func window(_ s: [Character]) -> String {
        String(s[from..<min(s.count, i + 12)])
    }
    return textCase(name, false,
                    "diverged at character \(i): got …\(window(g))… want …\(window(w))…")
}

private func equalIDsCase(_ name: String, _ got: [Int32], _ want: [Int32]) -> TextCase {
    if got == want { return textCase(name, true) }
    var i = 0
    while i < min(got.count, want.count), got[i] == want[i] { i += 1 }
    return textCase(name, false,
                    "diverged at token \(i) of \(want.count): got "
                    + "\(Array(got[i..<min(got.count, i + 6)])) want "
                    + "\(Array(want[i..<min(want.count, i + 6)]))")
}

// MARK: - The deliberately wrong detokenizers

/// Ways a byte-level detokenizer plausibly goes wrong. Each one is a real
/// design decision made the other way, not noise.
private enum DetokenizerFault: String, CaseIterable {
    /// Close the byte run at every marker, including the ones being dropped.
    /// The library drops those IDs *before* the decoder runs, so the runs on
    /// either side are one run.
    case commitAtSkippedSpecial
    /// Decode each token's bytes on its own instead of holding the incomplete
    /// tail. Every multi-token codepoint becomes replacement characters.
    case perTokenBytes
    /// Take the token's characters as text — skip the byte alphabet entirely.
    case literalTokenText
    /// Map each character to a byte by masking its scalar value instead of
    /// going through the GPT-2 table. Every ASCII token — and, by coincidence,
    /// `Ġ` — comes out right, so this one is invisible in English.
    case maskedByteAlphabet

    var summary: String {
        switch self {
        case .commitAtSkippedSpecial: return "close the run at a dropped marker"
        case .perTokenBytes: return "decode each token's bytes alone"
        case .literalTokenText: return "token characters as text"
        case .maskedByteAlphabet: return "scalar & 0xFF instead of the byte table"
        }
    }
}

/// The production detokenizer with one rule broken. Kept here, not in the
/// runtime type, so the shipped path has no fault switch in it.
private func faultyDecode(_ ids: [Int32],
                          skipSpecialTokens: Bool,
                          fault: DetokenizerFault,
                          tokenizer: QwenTokenizer) -> String {
    var text = ""
    var run = ByteLevelRun()
    var masked: [UInt8] = []
    for id in ids {
        guard let token = tokenizer.token(for: id) else { continue }
        if skipSpecialTokens, tokenizer.isSpecialToken(id) {
            if fault == .commitAtSkippedSpecial { text += run.commit() }
            continue
        }
        if tokenizer.isAddedToken(id) {
            if fault == .maskedByteAlphabet {
                text += String(decoding: masked, as: UTF8.self) + token
                masked.removeAll()
            } else {
                text += run.commit() + token
            }
            continue
        }
        switch fault {
        case .commitAtSkippedSpecial:
            text += run.push(token)
        case .perTokenBytes:
            text += String(decoding: ByteLevelDecoding.bytes(of: token) ?? [], as: UTF8.self)
        case .literalTokenText:
            text += token
        case .maskedByteAlphabet:
            masked.append(contentsOf: token.unicodeScalars.map { UInt8($0.value & 0xFF) })
        }
    }
    if fault == .maskedByteAlphabet {
        text += String(decoding: masked, as: UTF8.self)
    } else {
        text += run.commit()
    }
    return text
}

// MARK: - The check

func runQwenTokenizerCheck(modelPath: String, fixturePath: String?) async throws -> Bool {
    let directoryURL = URL(fileURLWithPath: modelPath)
    print("=== ornith tokenizer (docs/qwen35moe/04-PHASES Phase 5) ===")
    print("  model  \(directoryURL.path)")

    let tokenizer: QwenTokenizer
    if directoryURL.appendingPathComponent("tokenizer.json").isReadableFile {
        tokenizer = try await QwenTokenizer.load(from: directoryURL)
    } else {
        tokenizer = try await QwenTokenizer.load(forModelDirectory: directoryURL)
    }

    var cases: [TextCase] = []

    // 1. The IDs the runtime pins. `generation_config.json` says the stop set
    //    is [248046, 248044] (`04-PHASES.md` Phase 5, 実測(上流)).
    let pinned: [(String, Int32, Int32)] = [
        ("<|im_start|>", tokenizer.imStartID, 248_045),
        ("<|im_end|>", tokenizer.imEndID, 248_046),
        ("<|endoftext|>", tokenizer.endOfTextID, 248_044),
        ("<think>", tokenizer.thinkStartID, 248_068),
        ("</think>", tokenizer.thinkEndID, 248_069),
        ("<tool_call>", tokenizer.toolCallStartID, 248_058),
        ("</tool_call>", tokenizer.toolCallEndID, 248_059),
        ("<tool_response>", tokenizer.toolResponseStartID, 248_066),
        ("</tool_response>", tokenizer.toolResponseEndID, 248_067),
    ]
    for (token, got, want) in pinned {
        cases.append(textCase("marker \(token) == \(want)", got == want,
                              got == want ? "" : "resolved to \(got)"))
    }
    cases.append(textCase("stop tokens are [<|im_end|>, <|endoftext|>]",
                          tokenizer.stopTokenIDs == [248_046, 248_044],
                          "got \(tokenizer.stopTokenIDs.sorted())"))

    // 2. The byte alphabet covers the whole vocabulary. A token outside it
    //    would decode to replacement characters at runtime; here it is a
    //    finding, not a surprise later.
    var unmapped: [Int32] = []
    for id in 0..<Int32(tokenizer.vocabSize) {
        guard let token = tokenizer.token(for: id) else { continue }
        if tokenizer.isAddedToken(id) { continue }
        if ByteLevelDecoding.bytes(of: token) == nil { unmapped.append(id) }
    }
    cases.append(textCase("every vocabulary token is byte-level", unmapped.isEmpty,
                          "\(unmapped.count) outside the alphabet, e.g. \(unmapped.prefix(4))"))

    guard let fixturePath else {
        report(cases)
        print("  (no --qwen-tokenizer-fixture: encode/decode/chat not compared)")
        return cases.allSatisfy(\.passed)
    }
    let fixture = try QwenTokenizerFixture.load(fixturePath)
    print("  fixture \(fixturePath) — encode \(fixture.encode.count) / "
          + "decode \(fixture.decode.count) / chat \(fixture.chat.count)")

    cases.append(textCase("vocabulary size == \(fixture.vocabSize)",
                          tokenizer.vocabSize == fixture.vocabSize,
                          "got \(tokenizer.vocabSize)"))
    let addedMatches = fixture.addedTokens.allSatisfy { added in
        tokenizer.isAddedToken(added.id)
            && tokenizer.isSpecialToken(added.id) == added.special
    }
    cases.append(textCase("added-token set and its special flags match",
                          addedMatches
                              && tokenizer.addedTokenCount == fixture.addedTokens.count,
                          "got \(tokenizer.addedTokenCount) added tokens"))

    // 3. Encode. The Japanese cases are the ones that matter — a byte-level BPE
    //    that got its pre-tokenizer wrong still encodes ASCII correctly.
    for (index, item) in fixture.encode.enumerated() {
        let got = tokenizer.encode(item.text)
        cases.append(equalIDsCase("encode[\(index)] \(preview(item.text))", got, item.ids))
    }

    // 4. Decode, both modes, plus streaming/batch agreement.
    var streamingDisagreements = 0
    for (index, item) in fixture.decode.enumerated() {
        let keep = tokenizer.decode(item.ids, skipSpecialTokens: false)
        let skip = tokenizer.decode(item.ids, skipSpecialTokens: true)
        cases.append(equalCase("decode[\(index)] keep specials", keep, item.keep))
        cases.append(equalCase("decode[\(index)] skip specials", skip, item.skip))
        // The streaming path is what the generator uses; a token at a time,
        // with the tail flushed at the stop boundary.
        var detok = tokenizer.makeDetokenizer(skipSpecialTokens: false)
        var streamed = ""
        for id in item.ids { streamed += detok.push(id) }
        streamed += detok.flush()
        if streamed != keep { streamingDisagreements += 1 }
    }
    cases.append(textCase("streaming decode == batch decode on every case",
                          streamingDisagreements == 0,
                          "\(streamingDisagreements) case(s) differ"))

    // 5. The chat template, rendered by swift-jinja from the checkpoint's own
    //    `chat_template.jinja`. Comparing the ids compares the template *and*
    //    the encoder; decoding them back compares the text a human can read.
    //
    //    The tool case is the one exception, and it is a real finding rather
    //    than a tolerance: `tool | tojson` is the *host's* JSON writer, and the
    //    two hosts disagree on things JSON does not define — swift-jinja emits
    //    compact separators and ascending key order (it sorts at both ends:
    //    `Value(any:)` on the way in, `.sortedKeys` on the way out); Python
    //    emits `", "` / `": "` and keeps insertion order. Both are
    //    deterministic; there are two fixed spellings, not one fixed and one
    //    arbitrary (`docs/qwen35moe/23-PHASE5-TOOLS.md` §5-1). The tokens
    //    differ, so the case compares the `<tools>` block as parsed JSON and
    //    everything around it verbatim (§`toolsBlockCases`).
    for item in fixture.chat {
        // The fixture's tool case is rendered with a tool spec; the Swift side
        // gets the same one from `toolFixture`.
        let messages = chatMessages(for: item.name)
        let tools = item.name == "tools" ? [toolFixture] : []
        let got = try tokenizer.applyChatTemplate(messages,
                                                  tools: tools,
                                                  enableThinking: item.enableThinking)
        let text = tokenizer.decode(got, skipSpecialTokens: false)
        if tools.isEmpty {
            cases.append(equalIDsCase("chat[\(item.name)] ids", got, item.ids))
            cases.append(equalCase("chat[\(item.name)] text", text, item.text))
        } else {
            cases.append(contentsOf: toolsBlockCases(name: item.name,
                                                     got: text,
                                                     want: item.text))
        }
    }

    // 6. What Phase 3 and Phase 4 already generated, read as text. The tokens
    //    are the ones `--qwen-decode` and `--qwen-prefill` match; this is the
    //    first time anything in this repo turns them back into a sentence.
    if let generated = decodeFixtureText(tokenizer: tokenizer) {
        print("  decode-fixture-55 → \(generated.debugDescription)")
        cases.append(textCase("the 55-token generation reads as the reference sentence",
                              generated.contains("日本の首都は東京です。")
                                  && generated.hasPrefix("<think>"),
                              "got \(generated.debugDescription)"))
    }

    // 7. Negative controls: four wrong detokenizers over the same fixture.
    for fault in DetokenizerFault.allCases {
        var disagreements = 0
        var firstExample = ""
        for item in fixture.decode {
            for skip in [false, true] {
                let got = faultyDecode(item.ids,
                                       skipSpecialTokens: skip,
                                       fault: fault,
                                       tokenizer: tokenizer)
                let want = skip ? item.skip : item.keep
                if got != want {
                    disagreements += 1
                    if firstExample.isEmpty {
                        firstExample = "\(got.debugDescription) != \(want.debugDescription)"
                    }
                }
            }
        }
        cases.append(textCase("negative: \(fault.rawValue) — \(fault.summary)",
                              disagreements > 0,
                              disagreements > 0
                                  ? "disagreed on \(disagreements) case(s), e.g. \(firstExample)"
                                  : "decoded the whole fixture correctly — this "
                                      + "control has no detection power"))
    }

    report(cases)
    return cases.allSatisfy(\.passed)
}

// MARK: - Helpers

private func report(_ cases: [TextCase]) {
    let failures = cases.filter { !$0.passed }
    for item in cases where !item.passed {
        print("  FAIL  \(item.name) — \(item.detail)")
    }
    let negatives = cases.filter { $0.name.hasPrefix("negative:") }
    if failures.isEmpty {
        print("PASS  \(cases.count) cases, \(negatives.count) of them negative controls")
        for item in negatives { print("    \(item.name): \(item.detail)") }
    } else {
        print("FAIL  \(failures.count)/\(cases.count) cases")
    }
}

private func preview(_ text: String) -> String {
    let flat = text.replacingOccurrences(of: "\n", with: "\\n")
    return flat.count <= 24 ? "\"\(flat)\"" : "\"\(flat.prefix(24))…\""
}

private extension URL {
    var isReadableFile: Bool { FileManager.default.isReadableFile(atPath: path) }
}


/// The tool-path rendering, compared where the two hosts can be expected to
/// agree.
///
/// Everything outside `<tools>…</tools>` is compared verbatim — that is the
/// template's own text, and a difference there would be a template bug. Inside,
/// each line is one tool spec, and the comparison is between the *parsed*
/// objects: same tools, same fields, same values, whatever the writer did with
/// separators and key order.
private func toolsBlockCases(name: String, got: String, want: String) -> [TextCase] {
    func split(_ text: String) -> (head: String, body: [String], tail: String)? {
        guard let open = text.range(of: "<tools>"),
              let close = text.range(of: "</tools>", range: open.upperBound..<text.endIndex)
        else { return nil }
        let body = text[open.upperBound..<close.lowerBound]
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map(String.init)
        return (String(text[text.startIndex..<open.upperBound]),
                body,
                String(text[close.lowerBound...]))
    }
    guard let mine = split(got), let theirs = split(want) else {
        return [textCase("chat[\(name)] has a <tools> block", false,
                         "no <tools>…</tools> in one of the two renderings")]
    }
    var cases: [TextCase] = [
        equalCase("chat[\(name)] text before <tools>", mine.head, theirs.head),
        equalCase("chat[\(name)] text after </tools>", mine.tail, theirs.tail),
        textCase("chat[\(name)] declares \(theirs.body.count) tool(s)",
                 mine.body.count == theirs.body.count,
                 "got \(mine.body.count)"),
    ]
    for (index, pair) in zip(mine.body, theirs.body).enumerated() {
        let lhs = try? JSONSerialization.jsonObject(with: Data(pair.0.utf8))
        let rhs = try? JSONSerialization.jsonObject(with: Data(pair.1.utf8))
        let equal = lhs != nil && rhs != nil
            && (lhs as? NSDictionary)?.isEqual(to: rhs as? NSDictionary ?? [:]) == true
        cases.append(textCase("chat[\(name)] tool[\(index)] spec (as JSON)", equal,
                              "got \(pair.0) want \(pair.1)"))
    }
    return cases
}

/// The messages the fixture generator used, by case name. Kept in step with
/// `Scripts/qwen35/tokenizer_fixture.py`: if the two drift apart the comparison
/// fails loudly rather than quietly comparing different conversations.
private func chatMessages(for name: String) -> [GFTokenizer.Message] {
    switch name {
    case "user-only-thinking", "user-only-no-thinking":
        return [.init(role: .user, content: "日本の首都はどこですか。一文で答えてください。")]
    case "system-user":
        return [.init(role: .system, content: "You are a terse assistant."),
                .init(role: .user, content: "Name three primes.")]
    case "multi-turn":
        return [.init(role: .user, content: "1 + 1 は?"),
                .init(role: .assistant, content: "2 です。"),
                .init(role: .user, content: "では 2 + 2 は?")]
    case "assistant-with-reasoning":
        return [.init(role: .user, content: "1 + 1 は?"),
                .init(role: .assistant,
                      content: "2 です。",
                      reasoningContent: "足し算をする。"),
                .init(role: .user, content: "では 2 + 2 は?")]
    case "tools":
        return [.init(role: .user, content: "東京の天気は?")]
    default:
        return [.init(role: .user, content: name)]
    }
}

private let toolFixture = GFTokenizer.FunctionDefinition(
    name: "get_weather",
    description: "Get the weather for a city.",
    parameters: .object([
        "type": .string("object"),
        "properties": .object(["city": .object(["type": .string("string")])]),
        "required": .array([.string("city")]),
    ]))

/// The Phase 3/4 fixture, decoded. Absent (or unreadable) means the case is
/// skipped — the tokens live in `scratch/`, which is not in the repository.
private func decodeFixtureText(tokenizer: QwenTokenizer) -> String? {
    let url = URL(fileURLWithPath: "scratch/qwen35/decode-fixture-55.json")
    guard let data = try? Data(contentsOf: url),
          let fixture = try? JSONDecoder().decode(QwenDecodeFixture.self, from: data)
    else { return nil }
    return tokenizer.decode(fixture.expected, skipSpecialTokens: false)
}
