import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// Records what the HTTP layer handed the model, so the transport tests can
/// assert the image survived decoding rather than only that a 200 came back.
private actor RecordingBackend: ServerInferenceBackend {
    private(set) var imageCount = 0
    private(set) var imagePixels: [Int] = []

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        imageCount = request.vision?.images.count ?? 0
        imagePixels = request.vision?.images.map { $0.pixelWidth * $0.pixelHeight } ?? []
        onEvent(.content("hello"))
        return ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4))
    }
}

/// The `image_url` entry point (PLAN_VISION §4-6): what the server accepts as an
/// image, what it refuses, and with which status.
@Suite("Server image requests")
struct ServerImageRequestTests {

    // MARK: - Accepted

    @Test func dataURIImagePartBecomesAVisionRequest() throws {
        let request = try decode(body(parts: """
            {"type":"text","text":"what is this?"},
            {"type":"image_url","image_url":{"url":"\(dataURI(width: 64, height: 48))"}}
            """))
        let validated = try ChatRequestParser.parse(request)

        let vision = try #require(validated.vision)
        #expect(vision.images.count == 1)
        #expect(vision.images[0].pixelWidth == 64)
        #expect(vision.images[0].pixelHeight == 48)
        #expect(vision.images[0].mediaType == "image/png")
        #expect(vision.messages.count == 1)
        #expect(vision.messages[0].role == .user)
        #expect(vision.messages[0].parts == [.text("what is this?"), .image])
        #expect(vision.messages[0].imageCount == 1)
        // The text projection keeps only the words: it is what the prompt cache
        // keys on, and it is not what an image request is rendered from.
        #expect(validated.messages.map(\.content) == ["what is this?"])
    }

    @Test func openAIDetailHintIsAcceptedAndIgnored() throws {
        let request = try decode(body(parts: """
            {"type":"image_url","image_url":{"url":"\(dataURI(width: 8, height: 8))","detail":"high"}}
            """))
        let validated = try ChatRequestParser.parse(request)
        #expect(try #require(validated.vision).images.count == 1)
    }

    @Test func imagesFromSeveralTurnsAreCollectedInPromptOrder() throws {
        let request = try decode("""
            {"model":"m","messages":[
              {"role":"user","content":[
                {"type":"image_url","image_url":{"url":"\(dataURI(width: 16, height: 16))"}}]},
              {"role":"assistant","content":"a square"},
              {"role":"user","content":[
                {"type":"text","text":"and this?"},
                {"type":"image_url","image_url":{"url":"\(dataURI(width: 32, height: 16))"}}]}
            ]}
            """)
        let vision = try #require(
            try ChatRequestParser.parse(request).vision)
        #expect(vision.images.map(\.pixelWidth) == [16, 32])
        #expect(vision.messages.map(\.imageCount) == [1, 0, 1])
    }

    @Test func textOnlyPartsStillProduceNoVisionRequest() throws {
        let request = try decode(body(parts: """
            {"type":"text","text":"a"},{"type":"text","text":"b"}
            """))
        let validated = try ChatRequestParser.parse(request)
        #expect(validated.vision == nil)
        #expect(validated.messages.map(\.content) == ["ab"])
    }

    // MARK: - Refused

    /// The whole point of accepting only `data:`. Fetching a caller-supplied URL
    /// is server-side request forgery, so the refusal is by scheme, by name,
    /// before anything resolves it.
    @Test func remoteImageURLsAreRefusedByScheme() throws {
        for url in ["https://example.invalid/cat.png",
                    "http://127.0.0.1:9/cat.png",
                    "file:///etc/passwd"] {
            let request = try decode(body(parts: """
                {"type":"image_url","image_url":{"url":"\(url)"}}
                """))
            let error = #expect(throws: ServerRequestError.self) {
                try ChatRequestParser.parse(request)
            }
            #expect(error?.envelope.error.code == "unsupported_image_url")
            #expect(error?.httpStatus == .badRequest)
        }
    }

    @Test func oversizedImageIsRefused() throws {
        let request = try decode(body(parts: """
            {"type":"image_url","image_url":{"url":"\(dataURI(width: 64, height: 64))"}}
            """))
        let policy = ServerImagePolicy(maxImageBytes: 32)
        let error = #expect(throws: ServerRequestError.self) {
            try ChatRequestParser.parse(request, imagePolicy: policy)
        }
        #expect(error?.envelope.error.code == "image_too_large")
        #expect(error?.httpStatus == .badRequest)
    }

    /// Read from the container header, so a decompression bomb is refused
    /// without ever allocating its pixels.
    @Test func pixelCeilingIsCheckedBeforeDecoding() throws {
        let request = try decode(body(parts: """
            {"type":"image_url","image_url":{"url":"\(dataURI(width: 64, height: 64))"}}
            """))
        let policy = ServerImagePolicy(maxImagePixels: 4_095)
        let error = #expect(throws: ServerRequestError.self) {
            try ChatRequestParser.parse(request, imagePolicy: policy)
        }
        #expect(error?.envelope.error.code == "image_too_large")
        #expect(error?.envelope.error.message.contains("64x64") == true)
    }

    @Test func imageCountCeilingIsEnforced() throws {
        let part = """
            {"type":"image_url","image_url":{"url":"\(dataURI(width: 8, height: 8))"}}
            """
        let request = try decode(body(parts: [part, part, part].joined(separator: ",")))
        let error = #expect(throws: ServerRequestError.self) {
            try ChatRequestParser.parse(
                request, imagePolicy: ServerImagePolicy(maxImagesPerRequest: 2))
        }
        #expect(error?.envelope.error.code == "too_many_images")
        #expect(error?.httpStatus == .badRequest)
    }

    @Test func nonImageMediaTypesAndNonBase64PayloadsAreRefused() throws {
        let cases = [
            ("data:text/plain;base64,aGVsbG8=", "unsupported_image_media_type"),
            ("data:image/png,%89PNG", "invalid_image_data"),
            ("data:image/png;base64,", "invalid_image_data"),
            ("data:image/png;base64,####", "invalid_image_data"),
        ]
        for (url, code) in cases {
            let request = try decode(body(parts: """
                {"type":"image_url","image_url":{"url":"\(url)"}}
                """))
            let error = #expect(throws: ServerRequestError.self) {
                try ChatRequestParser.parse(request)
            }
            #expect(error?.envelope.error.code == code, "\(url)")
        }
    }

    /// Base64 that decodes but is not an image at all: caught by the header
    /// read, not by the tower.
    @Test func nonImageBytesAreRefusedAsInvalidImageData() throws {
        let encoded = Data("this is not an image, it is a sentence".utf8).base64EncodedString()
        let request = try decode(body(parts: """
            {"type":"image_url","image_url":{"url":"data:image/png;base64,\(encoded)"}}
            """))
        let error = #expect(throws: ServerRequestError.self) {
            try ChatRequestParser.parse(request)
        }
        #expect(error?.envelope.error.code == "invalid_image_data")
    }

    @Test func unknownContentPartTypesAreStillRefused() throws {
        let request = try decode(body(parts: """
            {"type":"input_audio","input_audio":{"data":"AA=="}}
            """))
        let error = #expect(throws: ServerRequestError.self) {
            try ChatRequestParser.parse(request)
        }
        #expect(error?.envelope.error.code == "unsupported_content")
    }

    /// Images and tools were refused together until 11-S2 showed the tool
    /// template renders an image content part like any other. Both halves of
    /// the request survive validation now: the tools are declared, and the
    /// picture is attached to the turn it came in on.
    @Test func imagesAreAcceptedAlongsideTools() throws {
        let request = try decode("""
            {"model":"m","messages":[{"role":"user","content":[
                {"type":"text","text":"look"},
                {"type":"image_url","image_url":{"url":"\(dataURI(width: 8, height: 8))"}}]}],
             "tools":[{"type":"function","function":{
                "name":"f","description":"d","parameters":{"type":"object"}}}]}
            """)
        let validated = try ChatRequestParser.parse(request)
        #expect(validated.tools.count == 1)
        let vision = try #require(validated.vision)
        #expect(vision.images.count == 1)
        #expect(vision.messages.last?.imageCount == 1)
    }

    /// The same request as pi sends it in an interactive session: tools every
    /// turn, a picture in the last user turn, and reasoning asked for.
    @Test func toolsImagesAndReasoningValidateTogether() throws {
        let request = try decode("""
            {"model":"m","messages":[{"role":"user","content":[
                {"type":"text","text":"look"},
                {"type":"image_url","image_url":{"url":"\(dataURI(width: 8, height: 8))"}}]}],
             "chat_template_kwargs":{"enable_thinking":true},
             "tools":[{"type":"function","function":{
                "name":"f","description":"d","parameters":{"type":"object"}}}]}
            """)
        let validated = try ChatRequestParser.parse(request)
        #expect(validated.enableThinking)
        #expect(validated.tools.count == 1)
        #expect(validated.vision != nil)
    }

    @Test func imagesAreRefusedOutsideUserTurns() throws {
        let request = try decode("""
            {"model":"m","messages":[
              {"role":"system","content":[
                {"type":"image_url","image_url":{"url":"\(dataURI(width: 8, height: 8))"}}]},
              {"role":"user","content":"hi"}]}
            """)
        let error = #expect(throws: ServerRequestError.self) {
            try ChatRequestParser.parse(request)
        }
        #expect(error?.envelope.error.message.contains("user turns") == true)
    }

    // MARK: - Policy

    /// A 1 MiB body cannot hold a base64 photograph. The ceiling has to move
    /// with the image limits, or every image request would be answered
    /// `request_too_large` before validation could say anything useful.
    @Test func bodyCeilingCoversTheConfiguredImages() {
        let policy = ServerImagePolicy(maxImagesPerRequest: 2, maxImageBytes: 3_000_000)
        #expect(policy.maximumBodyBytes >= ServerImagePolicy.textBodyBytes + 2 * 4_000_000)
        // Text-only servers keep exactly the ceiling they always had.
        #expect(ServerImagePolicy(maxImagesPerRequest: 0).maximumBodyBytes
                == TurboFieldfareHTTPServer.maximumBodyBytes)
    }

    @Test func serverArgumentsCarryTheImagePolicy() throws {
        let parsed = try ServerArguments.parse([
            "--model", "m.gturbo",
            "--image-tokens", "70",
            "--max-images", "1",
            "--max-image-bytes", "1024",
            "--max-image-pixels", "2048",
        ])
        #expect(parsed.imagePolicy == ServerImagePolicy(maxSoftTokens: 70,
                                                        maxImagesPerRequest: 1,
                                                        maxImageBytes: 1_024,
                                                        maxImagePixels: 2_048))
        #expect(try ServerArguments.parse(["--model", "m.gturbo"]).imagePolicy == .default)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--model", "m.gturbo", "--image-tokens", "560"])
        }
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--model", "m.gturbo", "--max-images", "0"])
        }
    }

    /// The words of two requests can be identical while the pictures differ,
    /// so the digests are what a resumed prefix is keyed on.
    @Test func imageDigestsDistinguishOtherwiseIdenticalRequests() throws {
        let first = try ChatRequestParser.parse(
            decode(body(parts: """
                {"type":"image_url","image_url":{"url":"\(dataURI(width: 8, height: 8))"}}
                """)))
        let second = try ChatRequestParser.parse(
            decode(body(parts: """
                {"type":"image_url","image_url":{"url":"\(dataURI(width: 16, height: 16))"}}
                """)))
        #expect(ServerPromptCache.imageDigests(first).count == 1)
        #expect(ServerPromptCache.imageDigests(first) != ServerPromptCache.imageDigests(second))
        #expect(ServerPromptCache.imageDigests(first) == ServerPromptCache.imageDigests(first))
    }

    /// A second picture in the same conversation resumes from the first one's
    /// KV: the bridge holds the new turn, and the spans it reports are offsets
    /// into that bridge — the slice a resumed prefill actually runs — not into
    /// the whole prompt. Before S3.6 one picture cost the conversation its
    /// prefix reuse for every turn that followed.
    @Test func aSecondImageTurnResumesFromTheFirstOnesPrefix() async throws {
        let tokenizer = try await GFTokenizer.load()
        let ids = try VisionMediaTokenIDs(tokenizer: tokenizer)
        let config = try VisionPreprocessorConfig(maxSoftTokens: 70)
        let domain = ServerPromptCacheDomain(
            modelID: "model",
            sourceSnapshotHash: "snapshot",
            runtimeProfileHash: "profile",
            maximumContext: 16_384,
            kvStorage: "fp16",
            fp16RingEnabled: true,
            templateSHA256: "template")

        func validated(_ json: String) throws -> ValidatedChatRequest {
            try ChatRequestParser.parse(decode(json))
        }
        func rendered(_ request: ValidatedChatRequest) throws
            -> (tokens: [Int32], vision: VisionPrefillInput) {
            let vision = try #require(request.vision)
            let images = try vision.images.map {
                try VisionImagePreprocessor.preprocess(data: $0.data, config: config)
            }
            let text = try tokenizer.applyChatTemplate(multimodal: vision.messages,
                                                       variant: ServerPromptRenderer.variant)
            return try VisionPromptAssembler.makePrefillPrompt(
                tokens: tokenizer.encode(text, addBOS: false), images: images, ids: ids)
        }

        let first = try validated("""
            {"model":"m","messages":[{"role":"user","content":[
              {"type":"text","text":"caption this"},
              {"type":"image_url","image_url":{"url":"\(dataURI(width: 64, height: 64))"}}]}]}
            """)
        let firstPrompt = try rendered(first)
        let generated = tokenizer.encode("a blue square", addBOS: false)
        let kvBacked = firstPrompt.tokens + generated
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: first,
            result: RawDecodeResult(
                prefillTokens: firstPrompt.tokens.count,
                cachedPromptTokens: 0,
                computedPrefillTokens: firstPrompt.tokens.count,
                prefillSeconds: 0,
                newTokens: generated.count,
                decodeSeconds: 0,
                timeToFirstTokenSeconds: 0,
                reason: .endOfTurn,
                kvPosition: kvBacked.count,
                kvBackedTokenIDs: kvBacked,
                uncommittedBoundaryTokenIDs: [tokenizer.endOfTurnID]))

        let second = try validated("""
            {"model":"m","messages":[
              {"role":"user","content":[
                {"type":"text","text":"caption this"},
                {"type":"image_url","image_url":{"url":"\(dataURI(width: 64, height: 64))"}}]},
              {"role":"assistant","content":"a blue square"},
              {"role":"user","content":[
                {"type":"text","text":"and this one"},
                {"type":"image_url","image_url":{"url":"\(dataURI(width: 96, height: 96))"}}]}]}
            """)
        let secondPrompt = try rendered(second)
        guard case .hit(let effective, let cached, let continuationVision) = cache.match(
            domain: domain,
            request: second,
            renderedPromptIDs: secondPrompt.tokens,
            vision: secondPrompt.vision,
            maximumRewind: 2048) else {
            Issue.record("expected the second image turn to resume from the first")
            return
        }
        #expect(cached == kvBacked.count)
        #expect(Array(effective.prefix(cached)) == kvBacked)

        // Exactly the picture this turn added, positioned inside the slice
        // that is still to be prefilled.
        let vision = try #require(continuationVision)
        #expect(vision.images.count == 1)
        let remainingLength = effective.count - cached
        let span = try #require(vision.spans.first)
        #expect(span.tokenOffset >= 0)
        #expect(span.tokenEnd <= remainingLength)
        #expect(span.tokenCount == vision.images[0].geometry.softTokenCount)
        // The soft tokens sit between the markers the assembler wrote.
        let bridge = Array(effective.dropFirst(cached))
        #expect(bridge[span.tokenOffset - 1] == ids.beginImage)
        #expect(bridge[span.tokenEnd] == ids.endImage)
    }

    /// A different picture behind the same words cannot ride the cached prefix.
    @Test func aDifferentPictureWithTheSameWordsMisses() async throws {
        let tokenizer = try await GFTokenizer.load()
        let ids = try VisionMediaTokenIDs(tokenizer: tokenizer)
        let config = try VisionPreprocessorConfig(maxSoftTokens: 70)
        let domain = ServerPromptCacheDomain(
            modelID: "model",
            sourceSnapshotHash: "snapshot",
            runtimeProfileHash: "profile",
            maximumContext: 16_384,
            kvStorage: "fp16",
            fp16RingEnabled: true,
            templateSHA256: "template")
        func validated(_ json: String) throws -> ValidatedChatRequest {
            try ChatRequestParser.parse(decode(json))
        }
        let first = try validated("""
            {"model":"m","messages":[{"role":"user","content":[
              {"type":"text","text":"caption this"},
              {"type":"image_url","image_url":{"url":"\(dataURI(width: 64, height: 64))"}}]}]}
            """)
        let visionFirst = try #require(first.vision)
        let images = try visionFirst.images.map {
            try VisionImagePreprocessor.preprocess(data: $0.data, config: config)
        }
        let prompt = try VisionPromptAssembler.makePrefillPrompt(
            tokens: tokenizer.encode(
                try tokenizer.applyChatTemplate(multimodal: visionFirst.messages,
                                                variant: ServerPromptRenderer.variant),
                addBOS: false),
            images: images,
            ids: ids)
        let kvBacked = prompt.tokens + tokenizer.encode("a blue square", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: first,
            result: RawDecodeResult(
                prefillTokens: prompt.tokens.count,
                cachedPromptTokens: 0,
                computedPrefillTokens: prompt.tokens.count,
                prefillSeconds: 0,
                newTokens: 3,
                decodeSeconds: 0,
                timeToFirstTokenSeconds: 0,
                reason: .endOfTurn,
                kvPosition: kvBacked.count,
                kvBackedTokenIDs: kvBacked,
                uncommittedBoundaryTokenIDs: [tokenizer.endOfTurnID]))

        // Same words, a different photograph in the first turn.
        let other = try validated("""
            {"model":"m","messages":[
              {"role":"user","content":[
                {"type":"text","text":"caption this"},
                {"type":"image_url","image_url":{"url":"\(dataURI(width: 96, height: 96))"}}]},
              {"role":"assistant","content":"a blue square"},
              {"role":"user","content":"and now?"}]}
            """)
        // Two photographs widen into the *same* soft-token ids, so the token
        // walk agrees on this whole first turn. The digests are the only thing
        // that can refuse it.
        let otherVision = try #require(other.vision)
        let otherImages = try otherVision.images.map {
            try VisionImagePreprocessor.preprocess(data: $0.data, config: config)
        }
        let otherPrompt = try VisionPromptAssembler.makePrefillPrompt(
            tokens: tokenizer.encode(
                try tokenizer.applyChatTemplate(multimodal: otherVision.messages,
                                                variant: ServerPromptRenderer.variant),
                addBOS: false),
            images: otherImages,
            ids: ids)
        #expect(commonPrefixLength(kvBacked, otherPrompt.tokens) > prompt.tokens.count / 2,
                "the two prompts have to agree on tokens for this test to mean anything")
        #expect(cache.match(domain: domain,
                            request: other,
                            renderedPromptIDs: otherPrompt.tokens,
                            vision: otherPrompt.vision,
                            maximumRewind: 2048) == .miss)
    }

    // MARK: - Over HTTP

    @Test func dataURIImageReachesTheBackendWith200() async throws {
        let backend = RecordingBackend()
        let server = TurboFieldfareHTTPServer(modelID: "test-model",
                                              queueLimit: 1,
                                              backend: backend)
        let port = try #require(try await server.start(port: 0).localAddress?.port)
        defer { Task { try? await server.shutdown() } }

        let (data, response) = try await post(port: port, body: """
            {"model":"test-model","messages":[{"role":"user","content":[
              {"type":"text","text":"describe"},
              {"type":"image_url","image_url":{"url":"\(dataURI(width: 64, height: 48))"}}]}]}
            """)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        #expect(String(decoding: data, as: UTF8.self).contains("hello"))
        #expect(await backend.imageCount == 1)
        #expect(await backend.imagePixels == [64 * 48])

        try await server.shutdown()
    }

    @Test func remoteImageURLReturns400OverHTTP() async throws {
        let server = TurboFieldfareHTTPServer(modelID: "test-model",
                                              queueLimit: 1,
                                              backend: RecordingBackend())
        let port = try #require(try await server.start(port: 0).localAddress?.port)

        let (data, response) = try await post(port: port, body: """
            {"model":"test-model","messages":[{"role":"user","content":[
              {"type":"image_url","image_url":{"url":"https://example.invalid/cat.png"}}]}]}
            """)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains("unsupported_image_url"))
        #expect(text.contains("never fetches"))

        try await server.shutdown()
    }

    @Test func oversizedImageIsRefusedOverHTTP() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: RecordingBackend(),
            imagePolicy: ServerImagePolicy(maxImageBytes: 64))
        let port = try #require(try await server.start(port: 0).localAddress?.port)

        let (data, response) = try await post(port: port, body: """
            {"model":"test-model","messages":[{"role":"user","content":[
              {"type":"image_url","image_url":{"url":"\(dataURI(width: 64, height: 64))"}}]}]}
            """)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect(String(decoding: data, as: UTF8.self).contains("image_too_large"))

        try await server.shutdown()
    }

    // MARK: - Helpers

    private func post(port: Int, body: String) async throws -> (Data, URLResponse) {
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(body.utf8)
        return try await URLSession.shared.data(for: request)
    }

    private func decode(_ json: String) throws -> Data {
        Data(json.utf8)
    }

    private func body(parts: String) -> String {
        """
        {"model":"m","messages":[{"role":"user","content":[\(parts)]}]}
        """
    }

    /// A real PNG, encoded the way a client would send it. Synthesised rather
    /// than checked in so the pixel-size checks run against a container header
    /// that actually says what the test claims it says.
    private func dataURI(width: Int, height: Int) -> String {
        "data:image/png;base64," + pngData(width: width, height: height).base64EncodedString()
    }

    private func pngData(width: Int, height: Int) -> Data {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CGContext(data: nil,
                                width: width,
                                height: height,
                                bitsPerComponent: 8,
                                bytesPerRow: width * 4,
                                space: space,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.setFillColor(CGColor(red: 0.2, green: 0.6, blue: 0.9, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = context.makeImage()!
        let data = NSMutableData()
        let destination = CGImageDestinationCreateWithData(
            data, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, image, nil)
        _ = CGImageDestinationFinalize(destination)
        return data as Data
    }
}
