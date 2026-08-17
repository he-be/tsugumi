import Foundation
import Testing
@testable import TurboFieldfare

@Suite("VisionPrompt")
struct VisionPromptTests {

    /// Stand-ins for the real marker ids. The expansion is pure integer
    /// rewriting, so it can be checked without loading a 32 MB tokenizer;
    /// `VisionMediaTokenIDsTests` below pins the real values.
    private static let ids = VisionMediaTokenIDs(beginImage: 900,
                                                 image: 901,
                                                 endImage: 902,
                                                 audio: 903,
                                                 video: 904)

    @Test("One placeholder expands to opener, soft tokens, closer")
    func singleImage() throws {
        let (tokens, spans) = try VisionPromptAssembler.expandImagePlaceholders(
            tokens: [10, 901, 11], softTokenCounts: [3], ids: Self.ids)
        #expect(tokens == [10, 900, 901, 901, 901, 902, 11])
        #expect(spans == [VisionImageSpan(imageIndex: 0, tokenOffset: 2, tokenCount: 3)])
        // The span covers the soft tokens only: openers and closers are
        // ordinary causal text, matching upstream's `image_ids = [image_token_id]`.
        for offset in spans[0].tokenOffset..<spans[0].tokenEnd {
            #expect(tokens[offset] == Self.ids.image)
        }
        #expect(tokens[spans[0].tokenOffset - 1] == Self.ids.beginImage)
        #expect(tokens[spans[0].tokenEnd] == Self.ids.endImage)
    }

    /// Two images in one prompt with *different* soft-token counts — the case a
    /// "280 per image" implementation gets wrong without ever crashing.
    @Test("Multiple images keep their own lengths and offsets")
    func multipleImages() throws {
        let (tokens, spans) = try VisionPromptAssembler.expandImagePlaceholders(
            tokens: [1, 901, 2, 901, 3], softTokenCounts: [2, 4], ids: Self.ids)
        #expect(tokens == [1, 900, 901, 901, 902, 2, 900, 901, 901, 901, 901, 902, 3])
        #expect(spans == [
            VisionImageSpan(imageIndex: 0, tokenOffset: 2, tokenCount: 2),
            VisionImageSpan(imageIndex: 1, tokenOffset: 7, tokenCount: 4),
        ])
        for span in spans {
            #expect(tokens[(span.tokenOffset - 1)] == Self.ids.beginImage)
            #expect(tokens[span.tokenEnd] == Self.ids.endImage)
            #expect(tokens[span.tokenOffset..<span.tokenEnd].allSatisfy { $0 == Self.ids.image })
        }
    }

    @Test("Placeholder and image counts must agree")
    func countMismatch() {
        #expect(throws: GFTokenizerError.self) {
            try VisionPromptAssembler.expandImagePlaceholders(
                tokens: [901, 901], softTokenCounts: [3], ids: Self.ids)
        }
        #expect(throws: GFTokenizerError.self) {
            try VisionPromptAssembler.expandImagePlaceholders(
                tokens: [901], softTokenCounts: [3, 3], ids: Self.ids)
        }
        #expect(throws: GFTokenizerError.self) {
            try VisionPromptAssembler.expandImagePlaceholders(
                tokens: [901], softTokenCounts: [0], ids: Self.ids)
        }
    }

    @Test("A prompt without placeholders is returned unchanged")
    func noImages() throws {
        let (tokens, spans) = try VisionPromptAssembler.expandImagePlaceholders(
            tokens: [1, 2, 3], softTokenCounts: [], ids: Self.ids)
        #expect(tokens == [1, 2, 3])
        #expect(spans.isEmpty)
    }

    @Test("Literal media markers in user text are rejected")
    func rejectsLiteralMarkers() {
        for marker in ["<|image|>", "<|image>", "<image|>", "<|audio|>", "<|video|>"] {
            #expect(throws: VisionError.self) {
                try VisionPromptAssembler.rejectMediaMarkers(in: "look at this \(marker) please")
            }
        }
    }

    @Test("Ordinary text with angle brackets still passes")
    func allowsOrdinaryText() throws {
        for text in ["a < b and c > d", "<|turn>", "an image of a cat", "<|channel>thought", "|image|"] {
            try VisionPromptAssembler.rejectMediaMarkers(in: text)
        }
    }

    /// The error must name the marker that was found. Ordering the scan
    /// longest-first is what keeps `<|image|>` from being reported as `<|image>`.
    @Test("The rejection names the marker it found")
    func namesTheMarker() {
        do {
            try VisionPromptAssembler.rejectMediaMarkers(in: "x <|image|> y")
            Issue.record("expected a rejection")
        } catch let error as VisionError {
            #expect(error == .unattachedMediaToken("<|image|>"))
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }
}

/// `VisionPrefillInput` is the contract between the prompt and the tower: the
/// spans say where the soft tokens go, the images say how many there are. It
/// exists to refuse the combinations that would scatter the wrong rows.
@Suite("VisionPrefillInput")
struct VisionPrefillInputTests {
    private func image(_ softTokens: Int) -> VisionPreprocessedImage {
        // Only the geometry is read here; the patch payload is irrelevant.
        VisionPreprocessedImage(
            geometry: VisionImageGeometry(targetWidth: 48 * softTokens,
                                          targetHeight: 48,
                                          patchesWide: 3 * softTokens,
                                          patchesHigh: 3,
                                          softTokenCount: softTokens),
            patches: [])
    }

    private func span(_ offset: Int, _ count: Int, index: Int = 0) -> VisionImageSpan {
        VisionImageSpan(imageIndex: index, tokenOffset: offset, tokenCount: count)
    }

    @Test("Matching spans and images are accepted")
    func accepted() throws {
        let input = try VisionPrefillInput(
            spans: [span(2, 4, index: 0), span(9, 7, index: 1)],
            images: [image(4), image(7)])
        #expect(!input.isEmpty)
        #expect(input.spans.count == 2)
    }

    @Test("A span length that disagrees with the image is refused")
    func lengthMismatch() {
        #expect(throws: VisionError.self) {
            try VisionPrefillInput(spans: [span(2, 280)], images: [image(266)])
        }
    }

    @Test("Count, order, and overlap are all refused")
    func structuralMismatch() {
        #expect(throws: VisionError.self) {
            try VisionPrefillInput(spans: [span(2, 4)], images: [image(4), image(4)])
        }
        #expect(throws: VisionError.self) {
            try VisionPrefillInput(spans: [span(9, 4, index: 0), span(2, 4, index: 1)],
                                   images: [image(4), image(4)])
        }
        #expect(throws: VisionError.self) {
            try VisionPrefillInput(spans: [span(2, 4, index: 0), span(4, 4, index: 1)],
                                   images: [image(4), image(4)])
        }
    }
}

@Suite("VisionMediaTokenIDs")
struct VisionMediaTokenIDsTests {
    /// Binds the marker strings to the ids `config.json` declares. If a
    /// tokenizer change moved any of them, every image prompt would be built
    /// from wrong ids, so this is pinned rather than derived.
    @Test("Marker tokens resolve to the checkpoint's declared ids")
    func resolvesAgainstTokenizer() async throws {
        let tokenizer = try await GFTokenizer.load()
        let ids = try VisionMediaTokenIDs(tokenizer: tokenizer)
        #expect(ids.beginImage == 255_999)
        #expect(ids.image == 258_880)
        #expect(ids.endImage == 258_882)
        #expect(ids.audio == 258_881)
        #expect(ids.video == 258_884)
    }

    /// The image placeholder has to survive the round trip through the chat
    /// template and the tokenizer as *one* id. If `<|image|>` encoded as
    /// ordinary text instead, the expansion would find no placeholder and the
    /// prompt would carry an image nobody could see.
    @Test("A multimodal turn renders and encodes to one image id per image")
    func multimodalTemplateEncodesPlaceholders() async throws {
        let tokenizer = try await GFTokenizer.load()
        let ids = try VisionMediaTokenIDs(tokenizer: tokenizer)
        let rendered = try tokenizer.applyChatTemplate(multimodal: [
            GFTokenizer.MultimodalMessage(role: .user,
                                          parts: [.text("describe this"), .image]),
        ])
        #expect(rendered.contains("<|turn>user\ndescribe this<|image|><turn|>"))

        let tokens = tokenizer.encode(rendered, addBOS: false)
        #expect(tokens.filter { $0 == ids.image }.count == 1)

        let prompt = try VisionPromptAssembler.expandImagePlaceholders(
            tokens: tokens, softTokenCounts: [5], ids: ids)
        #expect(prompt.spans.count == 1)
        #expect(prompt.tokens.count == tokens.count + 6)   // 5 soft tokens + boi + eoi - 1
        #expect(prompt.tokens[prompt.spans[0].tokenOffset - 1] == ids.beginImage)
        #expect(prompt.tokens[prompt.spans[0].tokenEnd] == ids.endImage)
    }

    /// A text-only message must render exactly as it did before content parts
    /// existed — the multimodal path is the same renderer.
    @Test("Text-only messages render identically through both entry points")
    func textOnlyRenderingIsUnchanged() async throws {
        let tokenizer = try await GFTokenizer.load()
        let viaMessages = try tokenizer.applyChatTemplate([
            GFTokenizer.Message(role: .system, content: " you are terse "),
            GFTokenizer.Message(role: .user, content: "  hello\n"),
        ], enableThinking: true)
        let viaParts = try tokenizer.applyChatTemplate(multimodal: [
            GFTokenizer.MultimodalMessage(role: .system, parts: [.text(" you are terse ")]),
            GFTokenizer.MultimodalMessage(role: .user, parts: [.text("  hello\n")]),
        ], enableThinking: true)
        #expect(viaMessages == viaParts)
    }

    @Test("The chat template refuses a prompt carrying a literal marker")
    func chatTemplateRejectsMarkers() async throws {
        let tokenizer = try await GFTokenizer.load()
        #expect(throws: VisionError.self) {
            try tokenizer.applyChatTemplate([
                GFTokenizer.Message(role: .user, content: "describe <|image|>")
            ])
        }
        #expect(throws: VisionError.self) {
            try tokenizer.encodeTextContinuation(userContent: "and this <|video|>?")
        }
    }
}
