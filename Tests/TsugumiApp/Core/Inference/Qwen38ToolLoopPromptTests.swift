import Foundation
import Testing
import Tsugumi
@testable import TsugumiAppCore
@testable import TsugumiServerCore

/// The Qwen3.8 tokenizer folder (`docs/qwen38/17` §1). Not in the repository; the suite is skipped without it.
private enum Qwen38TokenizerFolder {
    static let url = URL(fileURLWithPath: ProcessInfo.processInfo.environment["TSUGUMI_QWEN38_TOKENIZER"]
        ?? NSString(string: "~/LLM/Qwen3.8-Flash-Next-tokenizer").expandingTildeInPath)
    static var isInstalled: Bool {
        FileManager.default.fileExists(atPath: url.appendingPathComponent("tokenizer.json").path)
    }
}

/// The web tool loop as the Mac app sends it to Qwen3.8 (`docs/qwen38/20`), through the app's own request builder
/// (`RealInferenceSession.validatedChatRequest`) and the session's renderer, without weights.
///
/// Two claims:
/// 1. **The prompt is upstream's.** Each request renders to the same text as the checkpoint's template under
///    Hugging Face's jinja (`Scripts/qwen38/tool_loop_fixture.py` writes `Fixtures/qwen38-tool-loop/`). The tool
///    declarations were the gap (`docs/qwen38/17` §3-1: swift-jinja's `tojson` escaped every Japanese character).
/// 2. **INV-1 across rounds and turns.** What a round generated is a prefix of the next request's rendering, as tokens
///    after `Qwen38PromptCache.aligned`, so the next round continues from the live state instead of a checkpoint.
@Suite("Qwen3.8 tool loop prompt",
       .enabled(if: Qwen38TokenizerFolder.isInstalled, "needs ~/LLM/Qwen3.8-Flash-Next-tokenizer"))
struct Qwen38ToolLoopPromptTests {
    private static let shared = Task { try await QwenTokenizer.load(from: Qwen38TokenizerFolder.url) }
    private let tokenizer: QwenTokenizer

    init() async throws {
        tokenizer = try await Self.shared.value
    }

    // MARK: - The conversation

    /// One request of the loop, and what the model wrote in answer (nil for the last one).
    struct Step {
        let label: String
        let request: AppGenerationRequest
        let generated: String?
    }

    static let fixtureDirectory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures/qwen38-tool-loop")

    /// The declarations and system prompt an Online turn with a local Wikipedia index gets (`AppModel.makeToolExecutor`
    /// order: Wikipedia, then the web), on a fixed date.
    static func declarations() throws -> (tools: [AppToolDefinition], system: String) {
        let index = try LocalWikipediaIndex(path: #require(
            Bundle.module.url(forResource: "wikipedia-fixture", withExtension: "sqlite", subdirectory: "Fixtures")).path)
        var configuration = WebSearchConfiguration()
        configuration.serperAPIKey = "k"
        let executor = CompositeToolExecutor([
            WikipediaToolExecutor(index: index, maxResults: 8, pageCharacterLimit: 6_000),
            WebSearchToolExecutor(configuration: configuration.resolved()),
        ])
        var date = DateComponents()
        (date.year, date.month, date.day, date.hour) = (2026, 9, 14, 12)
        date.timeZone = TimeZone(identifier: "Asia/Tokyo")
        let today = Calendar(identifier: .gregorian).date(from: date)!
        let persona = AppPersona(identity: "あなたは Tsugumi。この Mac の中だけで動くローカル AI アシスタントです。")
        let system = [persona.promptSection!,
                      WebSearchPrompt.system(date: today, maxRounds: 6, tools: executor.promptFacts)]
            .joined(separator: "\n\n")
        return (executor.definitions, system)
    }

    /// Two Online turns: the app's own lookup, a forced search, prose and a forced fetch, the answer; then a follow-up
    /// that reads a Wikipedia page from a character offset (an integer argument) and answers; then a third question.
    static func steps() throws -> [Step] {
        let (tools, system) = try declarations()
        func request(history: [AppChatTurn], prompt: String, continuation: [AppChatTurn],
                     choice: AppToolChoice) -> AppGenerationRequest {
            AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/tmp"), history: history, prompt: prompt,
                                 systemPrompt: system, continuation: continuation, tools: tools, toolChoice: choice,
                                 maxContextTokens: 12_288, temperature: 0.7, topK: 20, topP: 0.8)
        }
        func call(_ id: String, _ name: String, _ arguments: String) -> AppToolCall {
            AppToolCall(id: id, name: name, argumentsJSON: arguments)
        }

        let q1 = "ツグミの渡りの時期と、2026年の観察情報を調べて"
        let lookup = call("lookup-1a2b3c4d-1", "wikipedia_lookup", #"{"titles":["ツグミ"]}"#)
        let search = call("call_1", "web_search", #"{"query":"ツグミ 2026 観察"}"#)
        let fetch = call("call_2", "fetch_page", #"{"url":"https://example.jp/birds/tsugumi?year=2026"}"#)
        let lookupResult = "参考: 質問に含まれる語を Wikipedia (2026年8月30日 時点の複製) で引いた記事の導入部です。\n\n■ ツグミ\nツグミ（鶫、学名: Turdus eunomus）は、スズメ目ツグミ科に分類される鳥類の一種。"
        let searchResult = "検索: ツグミ 2026 観察 (Serper, 2 件、取得日 2026年9月14日)\n[1] ツグミの観察記録 2026\n    https://example.jp/birds/tsugumi?year=2026\n    10月下旬から飛来…\n[2] 冬鳥カレンダー\n    https://example.org/calendar\n    ツグミ・シロハラ…\n"
        let pageResult = "URL: https://example.jp/birds/tsugumi?year=2026\n取得日 2026年9月14日\n\n# ツグミの観察記録 2026\n\n  10 月下旬にシベリアから渡来し、4 月まで滞在する。\n\n| 月 | 記録数 |\n| --- | ---: |\n| 11 | 42 |\n\n"
        let answer1 = "ツグミは冬鳥で、**10 月下旬**に渡来し 4 月ごろまで滞在します。2026 年の記録では 11 月に 42 件の観察がありました。\n\n参照:\n- https://example.jp/birds/tsugumi?year=2026"
        let t1r1Continuation = [AppChatTurn(role: .assistant, text: "", toolCalls: [lookup]),
                                .toolResult(callID: lookup.id, name: lookup.name, content: lookupResult)]
        let t1r2Continuation = t1r1Continuation
            + [AppChatTurn(role: .assistant, text: "", toolCalls: [search]),
               .toolResult(callID: search.id, name: search.name, content: searchResult)]
        let prose = "1 件目のページを読みます。"
        let t1r3Continuation = t1r2Continuation
            + [AppChatTurn(role: .assistant, text: prose, toolCalls: [fetch]),
               .toolResult(callID: fetch.id, name: fetch.name, content: pageResult)]
        let turn1 = [AppChatTurn(role: .user, text: q1)] + t1r3Continuation
            + [AppChatTurn(role: .assistant, text: answer1)]

        let q2 = "Summarize that in English, and read the Wikipedia article body from character 2000."
        let page = call("call_3", "wikipedia_page", #"{"from":2000,"title":"ツグミ"}"#)
        let wikiResult = "ツグミ (Wikipedia、2026年8月30日 時点の複製、2000 字目から)\n\n繁殖地はシベリア中部・東部。日本では冬鳥。\n"
        let answer2 = "Dusky thrushes (Turdus eunomus) winter in Japan from late October to April.\n\nSources:\n- https://example.jp/birds/tsugumi?year=2026\n- Wikipedia: ツグミ"
        let t2r2Continuation = [AppChatTurn(role: .assistant, text: "", toolCalls: [page]),
                                .toolResult(callID: page.id, name: page.name, content: wikiResult)]
        let turn2 = [AppChatTurn(role: .user, text: q2)] + t2r2Continuation
            + [AppChatTurn(role: .assistant, text: answer2)]

        func written(_ calls: [AppToolCall], prose: String = "") -> String {
            // What the grammar lets the model write for a call: parameters in ascending order, a string raw, anything
            // else as compact JSON (`QwenToolCallGrammar`).
            let body = calls.map { call -> String in
                let object = try! JSONDecoder().decode(JSONValue.self, from: Data(call.argumentsJSON.utf8))
                guard case .object(let arguments) = object else { return "" }
                let parameters = arguments.keys.sorted().map { key -> String in
                    let value: String = if case .string(let text) = arguments[key]! { text }
                        else { try! arguments[key]!.encoded() }
                    return "<parameter=\(key)>\n\(value)\n</parameter>\n"
                }.joined()
                return "<tool_call>\n<function=\(call.name)>\n\(parameters)</function>\n</tool_call>"
            }.joined(separator: "\n")
            return (prose.isEmpty ? "" : prose + "\n\n") + body
        }

        return [
            Step(label: "t1r1", request: request(history: [], prompt: q1, continuation: t1r1Continuation,
                                                 choice: .function(name: "web_search")),
                 generated: written([search])),
            Step(label: "t1r2", request: request(history: [], prompt: q1, continuation: t1r2Continuation,
                                                 choice: .function(name: "fetch_page")),
                 generated: written([fetch], prose: prose)),
            // The round budget is spent here: `none` keeps the declarations and forbids a call.
            Step(label: "t1r3", request: request(history: [], prompt: q1, continuation: t1r3Continuation,
                                                 choice: .none),
                 generated: answer1),
            Step(label: "t2r1", request: request(history: turn1, prompt: q2, continuation: [], choice: .auto),
                 generated: written([page])),
            Step(label: "t2r2", request: request(history: turn1, prompt: q2, continuation: t2r2Continuation,
                                                 choice: .auto),
                 generated: answer2),
            Step(label: "t3r1", request: request(history: turn1 + turn2, prompt: "ありがとう", continuation: [],
                                                 choice: .auto),
                 generated: nil),
        ]
    }

    // MARK: - Rendering

    private func render(_ request: AppGenerationRequest) throws -> [Int32] {
        let validated = try RealInferenceSession.validatedChatRequest(for: request, kind: .qwen38)
        return try tokenizer.applyChatTemplate(validated.messages, tools: validated.tools,
                                               enableThinking: validated.enableThinking)
    }

    /// The conversation the Python side renders: every step's request as the app's turns, and the declarations.
    static func spec(_ steps: [Step]) throws -> Data {
        func turn(_ t: AppChatTurn) -> [String: Any] {
            var object: [String: Any] = ["role": t.role.rawValue, "text": t.text, "reasoningText": t.reasoningText]
            if !t.toolCalls.isEmpty {
                object["toolCalls"] = t.toolCalls.map { ["id": $0.id, "name": $0.name, "argumentsJSON": $0.argumentsJSON] }
            }
            if let id = t.toolCallID { object["toolCallID"] = id }
            if let name = t.toolName { object["toolName"] = name }
            return object
        }
        let first = steps[0].request
        let object: [String: Any] = [
            "system": first.systemPrompt ?? "",
            "tools": first.tools.map { ["name": $0.name, "description": $0.description, "parametersJSON": $0.parametersJSON] },
            "steps": steps.map { step in
                ["label": step.label, "history": step.request.history.map(turn), "prompt": step.request.prompt,
                 "continuation": step.request.continuation.map(turn), "generated": step.generated.map { $0 as Any } ?? NSNull()]
            },
        ]
        return try JSONSerialization.data(withJSONObject: object,
                                          options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    // MARK: - Tests

    /// The spec the Python fixture was rendered from is this file's conversation and the app's current declarations.
    /// When the app's prompt changes, run `Scripts/qwen38/tool_loop_fixture.py` after writing the new spec.
    @Test("the committed spec is the app's current conversation")
    func specIsCurrent() throws {
        let current = try Self.spec(Self.steps())
        let committedURL = Self.fixtureDirectory.appendingPathComponent("spec.json")
        let committed = try? Data(contentsOf: committedURL)
        if committed != current {
            let fresh = FileManager.default.temporaryDirectory.appendingPathComponent("qwen38-tool-loop-spec.json")
            try current.write(to: fresh)
            Issue.record("spec changed: new one at \(fresh.path); copy it to \(committedURL.path) and run Scripts/qwen38/tool_loop_fixture.py")
        }
    }

    /// The request past the round budget declares the same tools (the prompt's head is unchanged) and leaves nothing
    /// for the grammar or the decoder to call.
    @Test("tool_choice none keeps the declarations and calls nothing")
    func noneKeepsTheDeclarations() throws {
        for step in try Self.steps() {
            let validated = try RealInferenceSession.validatedChatRequest(for: step.request, kind: .qwen38)
            #expect(validated.tools.map(\.name) == step.request.tools.map(\.name))
            #expect(validated.callableTools.isEmpty == (step.request.toolChoice == .none), "\(step.label)")
        }
    }

    @Test("every request renders to upstream's text")
    func rendersLikeUpstream() throws {
        for step in try Self.steps() {
            let swift = tokenizer.decode(try render(step.request), skipSpecialTokens: false)
            let python = try String(contentsOf: Self.fixtureDirectory.appendingPathComponent("\(step.label).txt"),
                                    encoding: .utf8)
            if swift != python {
                let common = zip(swift, python).prefix { $0 == $1 }.count
                let at = swift.index(swift.startIndex, offsetBy: common)
                let pyAt = python.index(python.startIndex, offsetBy: common)
                Issue.record("\(step.label) differs at character \(common): swift \(String(reflecting: String(swift[at...].prefix(80)))) python \(String(reflecting: String(python[pyAt...].prefix(80))))")
            }
        }
    }

    /// Each request continues from the live state the previous one left: the prompt, then the generated tokens up to
    /// the stop token (`<|im_end|>` is sampled, never fed).
    @Test("each round and each turn continues from the live state (INV-1)")
    func continuesFromLiveState() throws {
        let steps = try Self.steps()
        let piece = { (id: Int32) -> String? in tokenizer.isAddedToken(id) ? nil : tokenizer.token(for: id) }
        for (step, next) in zip(steps, steps.dropFirst()) {
            let prompt = try render(step.request)
            let generated = tokenizer.encode(step.generated!) + [tokenizer.imEndID]
            var cache = Qwen38PromptCache()
            cache.publish(prompt: prompt, generated: generated, kvPosition: prompt.count + generated.count - 1)
            let (aligned, _) = cache.aligned(try render(next.request), piece: piece)
            let (decision, agreed) = cache.decide(aligned, checkpoints: [])
            if decision != .live(prompt.count + generated.count - 1) {
                let held = cache.tokens
                let window = { (ids: [Int32]) in
                    String(reflecting: tokenizer.decode(Array(ids[max(agreed - 8, 0)..<min(agreed + 8, ids.count)]),
                                                        skipSpecialTokens: false))
                }
                Issue.record("\(step.label) → \(next.label): \(decision) agreed \(agreed) of \(held.count); held \(window(held)) next \(window(aligned))")
            }
        }
    }
}
