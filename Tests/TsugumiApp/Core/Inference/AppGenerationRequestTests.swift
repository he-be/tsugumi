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
