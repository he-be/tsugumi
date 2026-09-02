import Foundation
import Testing
import TsugumiDecodeProtocol

@Suite struct DecodeProtocolTests {
    @Test func loadRequestRoundTripPreservesEveryPublicRuntimeOption() throws {
        let options = DecodeRuntimeOptions(
            expertCacheSlots: 32,
            expertCachePolicy: "lru",
            prefillEnabled: false,
            prefillChunkTokens: 64,
            rdadvisePolicy: "adaptive",
            modelVerification: "trusted-install")
        let request = DecodeLoadRequest(
            modelPath: "/tmp/model.moepack",
            maxContextTokens: 8192,
            runtimeOptions: options,
            forceLogitsHead: true)

        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(request))
        try pipe.fileHandleForWriting.close()
        let decoded = try DecodeFrameCodec.read(
            DecodeLoadRequest.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.modelPath == request.modelPath)
        #expect(decoded.maxContextTokens == 8192)
        #expect(decoded.runtimeOptions == options)
        #expect(decoded.forceLogitsHead)
    }

    @Test func generationRequestRoundTripPreservesHistory() throws {
        let request = DecodeGenerationRequest(
            history: [
                DecodeChatTurn(role: "user", text: "q1",
                               imagePaths: ["/tmp/a.png"]),
                DecodeChatTurn(role: "assistant", text: "a1",
                               reasoningText: "r1"),
            ],
            prompt: "q2",
            maxNewTokens: 64,
            maxContextTokens: 4096,
            temperature: 1)

        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(request))
        try pipe.fileHandleForWriting.close()
        let decoded = try DecodeFrameCodec.read(
            DecodeGenerationRequest.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.history == request.history)
        #expect(decoded.prompt == "q2")
    }

    @Test func generationRequestWithoutHistoryKeyDecodesAsEmpty() throws {
        let json = """
        {"prompt":"solo","maxNewTokens":8,"maxContextTokens":1024,
         "temperature":1,"repetitionPenalty":1,
         "runtimeOptions":{"expertCacheSlots":16,"expertCachePolicy":"lfu",
          "prefillEnabled":true,"prefillChunkTokens":2048,
          "rdadvisePolicy":"off","modelVerification":"full-sha256"},
         "generationID":"\(UUID().uuidString)"}
        """
        let decoded = try JSONDecoder().decode(
            DecodeGenerationRequest.self, from: Data(json.utf8))
        #expect(decoded.history.isEmpty)
        #expect(decoded.prompt == "solo")
    }

    @Test func terminalEventRoundTripPreservesDiagnosticsAndMemory() throws {
        let runner = DecodeRunnerDiagnostics(
            cb1MillisecondsPerToken: 0.6,
            ioMillisecondsPerToken: 12,
            cb2MillisecondsPerToken: 0.4,
            headMillisecondsPerToken: 1.7,
            rdadviseMillisecondsPerToken: 0,
            rdadviseCallsPerToken: 0,
            rdadviseMegabytesPerToken: 0,
            rdadviseSkippedPerToken: 0,
            rdadviseFailures: 0)
        let event = DecodeServiceEvent(
            kind: .finished,
            generationID: UUID(),
            tokenCount: 256,
            promptTokenCount: 1_017,
            prefillSeconds: 10.2,
            timeToFirstTokenSeconds: 0.04,
            decodeSeconds: 7.7,
            tokensPerSecond: 33.2,
            currentMemoryBytes: 2_000_000_000,
            peakMemoryBytes: 2_100_000_000,
            runner: runner)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(event))
        try pipe.fileHandleForWriting.close()
        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.tokenCount == 256)
        #expect(decoded.promptTokenCount == 1_017)
        #expect(decoded.currentMemoryBytes == 2_000_000_000)
        #expect(decoded.peakMemoryBytes == 2_100_000_000)
        #expect(decoded.runner == runner)
    }

    @Test func prefillEventRoundTripPreservesProgress() throws {
        let event = DecodeServiceEvent(
            kind: .prefill,
            generationID: UUID(),
            sequence: 7,
            prefillDone: 128,
            prefillTotal: 514)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(event))
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.kind == .prefill)
        #expect(decoded.sequence == 7)
        #expect(decoded.prefillDone == 128)
        #expect(decoded.prefillTotal == 514)
    }

    @Test func generationRequestRoundTripPreservesTwoModelFields() throws {
        let request = DecodeGenerationRequest(
            prompt: "hello",
            maxNewTokens: 128,
            maxContextTokens: 4_096,
            temperature: 0.6,
            topK: 20,
            topP: 0.95,
            enableThinking: true,
            imagePaths: ["/tmp/a.png"],
            runtimeOptions: DecodeRuntimeOptions(mtpEnabled: false))
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(request))
        try pipe.fileHandleForWriting.close()
        let decoded = try DecodeFrameCodec.read(
            DecodeGenerationRequest.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.topK == 20)
        #expect(decoded.topP == 0.95)
        #expect(decoded.enableThinking)
        #expect(decoded.imagePaths == ["/tmp/a.png"])
        #expect(!decoded.runtimeOptions.mtpEnabled)
    }

    @Test func decoderAcceptsFramesWrittenBeforeTheTwoModelFields() throws {
        // A frame from a build that predates topK/thinking/images/MTP must
        // still decode, with the additions at their defaults.
        let legacyOptions = """
        {"expertCacheSlots": 32, "expertCachePolicy": "lfu", \
        "prefillEnabled": true, "prefillChunkTokens": 2048, \
        "rdadvisePolicy": "off", "modelVerification": "full-sha256"}
        """
        let legacyRequest = """
        {"prompt": "hi", "maxNewTokens": 8, "maxContextTokens": 4096, \
        "temperature": 1, "repetitionPenalty": 1, \
        "runtimeOptions": \(legacyOptions), \
        "generationID": "\(UUID().uuidString)"}
        """
        let decoded = try JSONDecoder().decode(
            DecodeGenerationRequest.self, from: Data(legacyRequest.utf8))
        #expect(decoded.topK == nil)
        #expect(decoded.topP == nil)
        #expect(!decoded.enableThinking)
        #expect(decoded.imagePaths.isEmpty)
        #expect(decoded.runtimeOptions.mtpEnabled)
    }

    @Test func decoderAcceptsAFrameSplitAcrossSingleByteWrites() throws {
        let event = DecodeServiceEvent(
            kind: .snapshot,
            generationID: UUID(),
            sequence: 1,
            textDelta: "caf\u{00E9}",
            tokenCount: 1)
        let frame = try DecodeFrameCodec.encode(event)
        let pipe = Pipe()
        for byte in frame {
            try pipe.fileHandleForWriting.write(contentsOf: Data([byte]))
        }
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.sequence == 1)
        #expect(decoded.textDelta == "caf\u{00E9}")
    }

    @Test func oversizedPayloadIsRejectedBeforeEncoding() {
        let request = DecodeGenerationRequest(
            prompt: String(repeating: "x", count: DecodeFrameCodec.maximumPayloadBytes + 1),
            maxNewTokens: 1,
            maxContextTokens: 4_096,
            temperature: 0)

        #expect(throws: DecodeFrameError.self) {
            _ = try DecodeFrameCodec.encode(request)
        }
    }

    @Test func oversizedFrameIsRejectedBeforePayloadRead() throws {
        let pipe = Pipe()
        var count = UInt32(DecodeFrameCodec.maximumPayloadBytes + 1).littleEndian
        try pipe.fileHandleForWriting.write(contentsOf: withUnsafeBytes(of: &count) { Data($0) })
        try pipe.fileHandleForWriting.close()

        #expect(throws: DecodeFrameError.self) {
            _ = try DecodeFrameCodec.read(
                DecodeServiceEvent.self,
                from: pipe.fileHandleForReading)
        }
    }
}

@Suite struct DecodeProtocolToolTests {
    @Test func generationRequestCarriesToolsContinuationAndSystemPrompt() throws {
        let call = DecodeToolCall(id: "c1", name: "web_search", argumentsJSON: #"{"query":"q"}"#)
        let request = DecodeGenerationRequest(
            history: [DecodeChatTurn(role: "user", text: "q0"),
                      DecodeChatTurn(role: "assistant", text: "a0")],
            prompt: "q1",
            systemPrompt: "sys",
            continuation: [
                DecodeChatTurn(role: "assistant", text: "", reasoningText: "r", toolCalls: [call]),
                DecodeChatTurn(role: "tool", text: "result", toolCallID: "c1", toolName: "web_search"),
            ],
            tools: [DecodeToolDefinition(name: "web_search", description: "d",
                                         parametersJSON: #"{"type":"object"}"#)],
            toolChoice: "required",
            reasoningBudgetTokens: 512,
            maxNewTokens: 8, maxContextTokens: 64, temperature: 1)
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(DecodeGenerationRequest.self, from: data)
        #expect(decoded.systemPrompt == "sys")
        #expect(decoded.reasoningBudgetTokens == 512)
        #expect(decoded.continuation == request.continuation)
        #expect(decoded.continuation[0].toolCalls == [call])
        #expect(decoded.continuation[1].toolCallID == "c1")
        #expect(decoded.tools == request.tools)
        #expect(decoded.toolChoice == "required")
    }

    @Test func aRequestFromBeforeTheToolFieldsStillDecodes() throws {
        let json = """
        {"prompt":"p","maxNewTokens":4,"maxContextTokens":64,"temperature":1,
         "repetitionPenalty":1,"history":[{"role":"user","text":"u"}],
         "runtimeOptions":{"expertCacheSlots":16,"expertCachePolicy":"lfu","prefillEnabled":true,
         "prefillChunkTokens":2048,"rdadvisePolicy":"off","modelVerification":"full-sha256"},
         "generationID":"\(UUID().uuidString)"}
        """
        let decoded = try JSONDecoder().decode(DecodeGenerationRequest.self, from: Data(json.utf8))
        #expect(decoded.systemPrompt == nil)
        #expect(decoded.continuation.isEmpty)
        #expect(decoded.tools.isEmpty)
        #expect(decoded.toolChoice == nil)
        #expect(decoded.reasoningBudgetTokens == nil)
        #expect(decoded.history[0].toolCalls.isEmpty)
    }

    @Test func terminalEventCarriesToolCalls() throws {
        let event = DecodeServiceEvent(
            kind: .finished, generationID: UUID(), stopReason: "toolCalls",
            toolCalls: [DecodeToolCall(id: "c1", name: "fetch_page", argumentsJSON: #"{"url":"u"}"#)])
        let data = try JSONEncoder().encode(event)
        let decoded = try JSONDecoder().decode(DecodeServiceEvent.self, from: data)
        #expect(decoded.toolCalls?.first?.name == "fetch_page")
        let plain = try JSONDecoder().decode(
            DecodeServiceEvent.self,
            from: try JSONEncoder().encode(DecodeServiceEvent(kind: .finished, generationID: UUID())))
        #expect(plain.toolCalls == nil)
    }
}
