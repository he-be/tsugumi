import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// The checkpoint's tokenizer sidecar. It ships inside the repacked
/// `.gturbo`, which lives under `scratch/` and is not in the repository, so a
/// checkout that has not installed Ornith has nothing to load here.
private enum OrnithTokenizerSidecar {
    static let folder = URL(fileURLWithPath: "scratch/ornith-oq4e-g64.gturbo")
        .appendingPathComponent("tokenizer")

    /// Whether the sidecar is installed. The claim it gates is about *this*
    /// checkpoint's template, and no other tokenizer can stand in for it, so
    /// the suite is skipped rather than failed when the sidecar is absent.
    static var isInstalled: Bool {
        FileManager.default.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path)
    }
}

/// INV-1 for Ornith: **describing a finished turn again == generating it.**
///
/// `PromptTokenInvariantTests` makes this claim for Gemma and nothing made it
/// for this family — the family whose prompt cache is all-or-nothing
/// (`QwenPromptCache`: thirty recurrent layers, so the only reusable prefix is
/// the whole state). One byte of disagreement anywhere inside a finished
/// assistant turn and every later request re-prefills the whole conversation,
/// for the rest of the session.
///
/// These need the checkpoint's tokenizer and template and nothing else: no
/// weights, no Metal. The claim is only ever about two strings.
@Suite("Ornith turn redraw",
       .enabled(if: OrnithTokenizerSidecar.isInstalled,
                "needs the Ornith tokenizer sidecar under scratch/"))
struct QwenTurnRedrawTests {
    /// One load for the whole suite. swift-testing builds a fresh instance for
    /// every test, and this checkpoint's `tokenizer.json` is 12 MB: parsing it
    /// once per test was nearly all of this suite's wall clock. The tokenizer
    /// is read-only, so the tests share the one value.
    private static let shared = Task {
        try await QwenTokenizer.load(from: OrnithTokenizerSidecar.folder)
    }

    private let tokenizer: QwenTokenizer

    init() async throws {
        tokenizer = try await Self.shared.value
    }

    // MARK: - The two sides

    /// The prompt as text, through the same call generation renders.
    private func text(_ messages: [GFTokenizer.Message],
                      _ tools: [GFTokenizer.FunctionDefinition]) throws -> String {
        tokenizer.decode(try tokenizer.applyChatTemplate(messages,
                                                         tools: tools,
                                                         enableThinking: true),
                         skipSpecialTokens: false)
    }

    /// What the K/V holds when a tool-calling turn finishes: the open prompt,
    /// then everything the model wrote. `reasoning` is what the decoder routes
    /// before `</think>`, `content` everything after it that is not the call.
    ///
    /// The redraw is the same conversation with that turn described back, the
    /// way a client hands it over. INV-1 is `redrawn.hasPrefix(generated)`.
    private func roundTrip(
        tools: [GFTokenizer.FunctionDefinition],
        reasoning: String,
        content: String,
        written: String,
        calls: [GFTokenizer.HistoricalToolCall]
    ) throws -> (generated: String, redrawn: String) {
        let user = GFTokenizer.Message(role: .user, content: "go")
        let generated = try text([user], tools)
            + reasoning + "</think>" + content + written + "<|im_end|>\n"
        let assistant = GFTokenizer.Message(role: .assistant,
                                            content: content,
                                            toolCalls: calls,
                                            reasoningContent: reasoning)
        return (generated, try text([user, assistant], tools))
    }

    // MARK: - A string-only tool

    private static let readTool = GFTokenizer.FunctionDefinition(
        name: "read_file",
        description: "read a file",
        parameters: .object([
            "type": .string("object"),
            "properties": .object(["path": .object(["type": .string("string")])]),
            "required": .array([.string("path")]),
        ]))

    private static let callText = """
        <tool_call>
        <function=read_file>
        <parameter=path>
        /tmp/a
        </parameter>
        </function>
        </tool_call>
        """

    private static let call = GFTokenizer.HistoricalToolCall(
        id: "call_1", name: "read_file", arguments: .object(["path": .string("/tmp/a")]))

    private func readCall(reasoning: String,
                          content: String) throws -> (generated: String, redrawn: String) {
        try roundTrip(tools: [Self.readTool],
                      reasoning: reasoning,
                      content: content,
                      written: Self.callText,
                      calls: [Self.call])
    }

    @Test("a call with no prose in front of it")
    func bareCall() throws {
        let (generated, redrawn) = try readCall(reasoning: "I should read it.\n",
                                                content: "\n\n")
        #expect(redrawn.hasPrefix(generated))
    }

    /// The seam the template normalises at both ends: `content` is `|trim`-ed
    /// on the way in and the separator in front of the call is always `\n\n`,
    /// so the whitespace the model wrote there does not have to be guessed.
    @Test("a call with prose in front of it")
    func proseBeforeCall() throws {
        let (generated, redrawn) = try readCall(reasoning: "I should read it.\n",
                                                content: "\n\nLet me read the file.\n\n")
        #expect(redrawn.hasPrefix(generated))
    }

    /// The same turn as a client that trimmed the content before handing it
    /// back. It has to draw the same bytes, or trimming is a cache miss.
    @Test("a client that trimmed the content draws the same turn")
    func trimmedContent() throws {
        let user = GFTokenizer.Message(role: .user, content: "go")
        let generated = try text([user], [Self.readTool])
            + "I should read it.\n</think>\n\nLet me read the file.\n\n"
            + Self.callText + "<|im_end|>\n"
        let assistant = GFTokenizer.Message(role: .assistant,
                                            content: "Let me read the file.",
                                            toolCalls: [Self.call],
                                            reasoningContent: "I should read it.")
        #expect(try text([user, assistant], [Self.readTool]).hasPrefix(generated))
    }

    @Test("two calls in one turn")
    func parallelCalls() throws {
        let (generated, redrawn) = try roundTrip(
            tools: [Self.readTool],
            reasoning: "ok\n",
            content: "\n\n",
            written: Self.callText + "\n" + Self.callText,
            calls: [Self.call, Self.call])
        #expect(redrawn.hasPrefix(generated))
    }

    // MARK: - A tool with values the template writes as JSON

    /// The template writes `args_value | string if args_value is string else
    /// args_value | tojson`, so only a string parameter is written raw. Every
    /// other type goes through swift-jinja's `tojson`, and *that* spelling is
    /// what the redraw has to agree with.
    private static let todoTool = GFTokenizer.FunctionDefinition(
        name: "todo_write",
        description: "write todos",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "count": .object(["type": .string("integer")]),
                "flag": .object(["type": .string("boolean")]),
                "items": .object(["type": .string("array")]),
                "meta": .object(["type": .string("object")]),
            ]),
            "required": .array([.string("count")]),
        ]))

    private func todoCall(_ written: String,
                          _ arguments: [String: JSONValue]) throws -> (generated: String,
                                                                       redrawn: String) {
        try roundTrip(tools: [Self.todoTool],
                      reasoning: "ok\n",
                      content: "\n\n",
                      written: "<tool_call>\n<function=todo_write>\n" + written
                          + "</function>\n</tool_call>",
                      calls: [GFTokenizer.HistoricalToolCall(id: "c",
                                                             name: "todo_write",
                                                             arguments: .object(arguments))])
    }

    private static func parameter(_ key: String, _ value: String) -> String {
        "<parameter=\(key)>\n\(value)\n</parameter>\n"
    }

    @Test("an integer parameter")
    func integerParameter() throws {
        let (generated, redrawn) = try todoCall(Self.parameter("count", "3"),
                                                ["count": .integer(3)])
        #expect(redrawn.hasPrefix(generated))
    }

    @Test("a boolean parameter")
    func booleanParameter() throws {
        let (generated, redrawn) = try todoCall(
            Self.parameter("count", "1") + Self.parameter("flag", "true"),
            ["count": .integer(1), "flag": .bool(true)])
        #expect(redrawn.hasPrefix(generated))
    }

    /// GEN-8: the only spelling the grammar leaves is the one the redraw
    /// writes — `JSONValue.encoded()`, which puts no whitespace anywhere. The
    /// spaced form the checkpoint was trained on (`["a", "b"]`) is not a
    /// generation this server can produce any more (`.qwenToolArguments`), so
    /// it is not a round trip that has to hold.
    @Test("an array parameter, in the one spelling the grammar leaves")
    func arrayParameter() throws {
        let (generated, redrawn) = try todoCall(
            Self.parameter("count", "1") + Self.parameter("items", #"["a","b"]"#),
            ["count": .integer(1), "items": .array([.string("a"), .string("b")])])
        #expect(redrawn.hasPrefix(generated))
    }

    @Test("an object parameter, in the one spelling the grammar leaves")
    func objectParameter() throws {
        let (generated, redrawn) = try todoCall(
            Self.parameter("count", "1") + Self.parameter("meta", #"{"k":1}"#),
            ["count": .integer(1), "meta": .object(["k": .integer(1)])])
        #expect(redrawn.hasPrefix(generated))
    }

    /// The call that was actually observed: `pi`'s `edit` takes `edits` as an
    /// array, so every string inside it is spelled by the redraw rather than
    /// copied. Two things were wrong and one still is.
    ///
    /// **Fixed.** swift-jinja's `tojson` is `JSONEncoder()` without
    /// `.withoutEscapingSlashes`, so it wrote `\/` for every `/`: the model
    /// read its own last call back as `\/\/ fog` where it wrote `// fog`,
    /// copied that into the next `oldText`, and four `edit` calls in a row
    /// could not match the file (session `01a02a00-…`, 2026-08-22). The JSON is
    /// now spelled by `JSONValue.encoded()` instead
    /// (`jinjaToolArgumentsValue`), which does not escape slashes.
    ///
    /// **Still registered (§12 DEV-15).** The keys of a *free-form* value come
    /// back in the encoder's order, not the model's. GBNF cannot compare two
    /// strings, so a grammar cannot require ascending keys inside a value the
    /// schema does not describe; and `JSONValue.object` is a Swift dictionary,
    /// so there is no original order left to preserve either. What it costs is
    /// now one turn's re-prefill (CACHE-8), not the conversation.
    @Test("an array of objects holding text: slashes are literal, key order is the encoder's")
    func editShapedCall() throws {
        let tool = GFTokenizer.FunctionDefinition(
            name: "edit",
            description: "edit a file",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "edits": .object(["type": .string("array")]),
                    "path": .object(["type": .string("string")]),
                ]),
                "required": .array([.string("path"), .string("edits")]),
            ]))
        let call = GFTokenizer.HistoricalToolCall(
            id: "c", name: "edit",
            arguments: .object([
                "path": .string("game.html"),
                "edits": .array([.object(["oldText": .string("// fog"),
                                          "newText": .string("// rim")])]),
            ]))
        let user = GFTokenizer.Message(role: .user, content: "go")
        let assistant = GFTokenizer.Message(role: .assistant, content: "",
                                            toolCalls: [call], reasoningContent: "ok")
        let redrawn = try text([user, assistant], [tool])
        // The slashes survive. This is the byte the model was reading wrong.
        #expect(redrawn.contains("// fog"))
        #expect(!redrawn.contains("\\/"))
        // DEV-15: sorted, not written-order.
        #expect(redrawn.contains(#"[{"newText":"// rim","oldText":"// fog"}]"#))
        // And the whole call round-trips when the model writes that spelling —
        // which is now the only spelling the grammar leaves it.
        let written = Self.parameter("edits", #"[{"newText":"// rim","oldText":"// fog"}]"#)
            + Self.parameter("path", "game.html")
        let generated = try text([user], [tool])
            + "ok\n</think>\n\n<tool_call>\n<function=edit>\n" + written
            + "</function>\n</tool_call><|im_end|>\n"
        #expect(redrawn.hasPrefix(generated))
    }
}
