import Foundation
import Testing
@testable import TsugumiAppCore

@Suite struct AppGenerationRequestTests {
    private let existingDirectory = FileManager.default.temporaryDirectory

    @Test func defaultRequestUsesDocumentedSamplingPolicy() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory, prompt: "hello")
        #expect(request.maxNewTokens == 4_096)
        #expect(request.temperature == 1.0)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.repetitionPenalty == 1)
        #expect(!request.isPureGreedy)
    }

    @Test func temperatureZeroRemainsPureGreedyWithTruncationDefaults() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello",
                                           temperature: 0)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.isPureGreedy)
    }

    @Test func historyImagesOnAssistantTurnRejected() {
        let request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            history: [
                AppChatTurn(role: .user, text: "look"),
                AppChatTurn(role: .assistant, text: "seen",
                            imagePaths: [existingDirectory.path]),
            ],
            prompt: "next")
        #expect(throws: AppInferenceError.self) {
            try request.validate(requireModelDirectory: false)
        }
    }

    @Test func historyStartingWithAssistantTurnRejected() {
        let request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            history: [AppChatTurn(role: .assistant, text: "unprompted")],
            prompt: "next")
        #expect(throws: AppInferenceError.self) {
            try request.validate(requireModelDirectory: false)
        }
    }

    @Test func historyWithMissingImageFileRejected() {
        let request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            history: [
                AppChatTurn(role: .user, text: "look",
                            imagePaths: ["/nonexistent/image-\(UUID()).png"]),
                AppChatTurn(role: .assistant, text: "seen"),
            ],
            prompt: "next")
        #expect(throws: AppInferenceError.self) {
            try request.validate(requireModelDirectory: false)
        }
    }

    @Test func multiTurnHistoryWithReasoningValidates() throws {
        let request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            history: [
                AppChatTurn(role: .user, text: "q1"),
                AppChatTurn(role: .assistant, text: "a1",
                            reasoningText: "thinking about q1"),
            ],
            prompt: "q2")
        try request.validate(requireModelDirectory: false)
    }

    @Test func emptyPromptRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory, prompt: "   ")
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidMaxTokensRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", maxNewTokens: 0)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidSlotCountRejected() {
        var options = AppRuntimeOptions()
        options.expertCacheSlots = 7
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", runtimeOptions: options)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func repetitionPenaltyBelowOneRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", repetitionPenalty: 0.9)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidTopKRejected() {
        for topK in [0, 257] {
            let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                               prompt: "hello", topK: topK)
            #expect(throws: AppInferenceError.self) {
                try request.validate()
            }
        }
    }

    @Test func invalidTopPRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", topP: 1.1)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func stochasticTopPRequiresTopK() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", topK: nil, topP: 0.95)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func missingModelDirectoryRejected() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model.moepack"),
            prompt: "hello")
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }
}

@Suite struct AppGenerationRequestToolTests {
    private func request(continuation: [AppChatTurn] = [],
                         history: [AppChatTurn] = [],
                         tools: [AppToolDefinition] = [],
                         toolChoice: AppToolChoice = .auto) -> AppGenerationRequest {
        AppGenerationRequest(modelDirectory: FileManager.default.temporaryDirectory,
                             history: history, prompt: "p",
                             continuation: continuation, tools: tools, toolChoice: toolChoice)
    }

    @Test func aWellFormedToolLoopValidates() throws {
        let call = AppToolCall(id: "c1", name: "web_search", argumentsJSON: "{}")
        try request(
            continuation: [AppChatTurn(role: .assistant, text: "", toolCalls: [call]),
                           .toolResult(callID: "c1", name: "web_search", content: "r")],
            tools: [AppToolDefinition(name: "web_search", description: "", parametersJSON: "{}")])
            .validate(requireModelDirectory: false)
    }

    @Test func continuationMustStartWithAnAssistantTurn() {
        #expect(throws: AppInferenceError.self) {
            try request(continuation: [.toolResult(callID: "c1", name: "t", content: "r")])
                .validate(requireModelDirectory: false)
        }
        #expect(throws: AppInferenceError.self) {
            try request(continuation: [AppChatTurn(role: .assistant, text: "a"),
                                       AppChatTurn(role: .user, text: "u")])
                .validate(requireModelDirectory: false)
        }
    }

    @Test func toolTurnsNeedACallIDAndOnlyAssistantsCarryCalls() {
        #expect(throws: AppInferenceError.self) {
            try request(history: [AppChatTurn(role: .user, text: "u"),
                                  AppChatTurn(role: .tool, text: "r")])
                .validate(requireModelDirectory: false)
        }
        #expect(throws: AppInferenceError.self) {
            try request(history: [AppChatTurn(
                role: .user, text: "u",
                toolCalls: [AppToolCall(id: "c", name: "n", argumentsJSON: "{}")])])
                .validate(requireModelDirectory: false)
        }
    }

    @Test func requiredNeedsATool() {
        #expect(throws: AppInferenceError.self) {
            try request(toolChoice: .required).validate(requireModelDirectory: false)
        }
    }

    @Test func chatTurnDecodesWithoutTheToolFields() throws {
        let json = #"{"role":"assistant","text":"a","reasoningText":"","imagePaths":[]}"#
        let turn = try JSONDecoder().decode(AppChatTurn.self, from: Data(json.utf8))
        #expect(turn.toolCalls.isEmpty)
        #expect(turn.toolCallID == nil)
        let tool = AppChatTurn.toolResult(callID: "c1", name: "web_search", content: "r")
        let roundTrip = try JSONDecoder().decode(
            AppChatTurn.self, from: try JSONEncoder().encode(tool))
        #expect(roundTrip == tool)
    }
}
