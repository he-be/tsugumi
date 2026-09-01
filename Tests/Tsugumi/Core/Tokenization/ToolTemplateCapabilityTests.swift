import Foundation
import Testing
import Tokenizers
@testable import Tsugumi

/// What the bundled tool-calling template can render (docs/serving §S2).
///
/// `encodeToolChat` pins `enable_thinking: false` and passes message content as
/// a string, which is why the server has neither reasoning nor images on the
/// tool path. These tests ask the template itself — the same swift-jinja engine
/// the server renders with — whether that limit is the template's or ours.
@Suite("Tool template capabilities")
struct ToolTemplateCapabilityTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load()
    }

    private var readTool: ToolSpec {
        [
            "type": "function",
            "function": [
                "name": "read",
                "description": "Read a file",
                "parameters": [
                    "type": "object",
                    "properties": ["path": ["type": "string", "description": "path"]],
                    "required": ["path"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    private func render(messages: [Tokenizers.Message],
                        tools: [ToolSpec]?,
                        enableThinking: Bool) throws -> String {
        let ids = try tok.tokenizer.applyChatTemplate(
            messages: messages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: tools,
            additionalContext: ["enable_thinking": enableThinking])
        return tok.decode(ids.map(Int32.init), skipSpecialTokens: false)
    }

    /// S2 question (a): can the template reason while tools are declared?
    @Test func toolsAndThinkingRenderTogether() throws {
        let rendered = try render(
            messages: [["role": "user", "content": "read /tmp/a"]],
            tools: [readTool],
            enableThinking: true)
        // The thought marker leads the system turn that also carries the tools.
        #expect(rendered.contains("<|think|>"))
        #expect(rendered.contains("<|tool>"))
        #expect(rendered.range(of: "<|think|>")!.lowerBound
                < rendered.range(of: "<|tool>")!.lowerBound)
        // With thinking on the generation prompt leaves the channel open.
        #expect(rendered.hasSuffix("<|turn>model\n"))
        #expect(!rendered.hasSuffix("<|channel>thought\n<channel|>"))
    }

    /// The same call with thinking off is what the server renders today.
    @Test func toolsWithoutThinkingCloseTheChannel() throws {
        let rendered = try render(
            messages: [["role": "user", "content": "read /tmp/a"]],
            tools: [readTool],
            enableThinking: false)
        #expect(!rendered.contains("<|think|>"))
        #expect(rendered.contains("<|tool>"))
        #expect(rendered.hasSuffix("<|turn>model\n<|channel>thought\n<channel|>"))
    }

    /// S2 question (b): can the template place an image while tools are
    /// declared? The content-parts branch is in the shared message loop, not in
    /// a tools-free path.
    @Test func toolsAndImagePartsRenderTogether() throws {
        let rendered = try render(
            messages: [[
                "role": "user",
                "content": [
                    ["type": "text", "text": "what is this"] as [String: any Sendable],
                    ["type": "image"] as [String: any Sendable],
                ] as [any Sendable],
            ]],
            tools: [readTool],
            enableThinking: false)
        #expect(rendered.contains("<|tool>"))
        #expect(rendered.contains("<|image|>"))
        #expect(rendered.contains("what is this"))
    }

    /// All three at once: tools, an image, and reasoning.
    @Test func toolsImageAndThinkingRenderTogether() throws {
        let rendered = try render(
            messages: [[
                "role": "user",
                "content": [
                    ["type": "text", "text": "what is this"] as [String: any Sendable],
                    ["type": "image"] as [String: any Sendable],
                ] as [any Sendable],
            ]],
            tools: [readTool],
            enableThinking: true)
        #expect(rendered.contains("<|think|>"))
        #expect(rendered.contains("<|tool>"))
        #expect(rendered.contains("<|image|>"))
        #expect(rendered.hasSuffix("<|turn>model\n"))
    }

    /// The image placeholder the template writes is the one the vision
    /// assembler widens, so a tool-path image would land in the same pipeline.
    @Test func templateImageMarkerMatchesTheVisionAssembler() throws {
        let rendered = try render(
            messages: [[
                "role": "user",
                "content": [["type": "image"] as [String: any Sendable]] as [any Sendable],
            ]],
            tools: [readTool],
            enableThinking: false)
        #expect(rendered.contains(VisionMediaTokenIDs.imageToken))
    }
}
