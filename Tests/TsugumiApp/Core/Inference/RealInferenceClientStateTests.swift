import CoreGraphics
import Foundation
import ImageIO
import Testing
import Tsugumi
import UniformTypeIdentifiers
@testable import TsugumiAppCore

/// Model-free state coverage for the real client: load failure surfaces
/// before any network or Metal work, idle cancel is a no-op, and a bad
/// request fails the stream with a typed error.
@Suite struct RealInferenceClientStateTests {
    @Test func generationRegistryScopesTerminationToOwningID() async {
        let registry = GenerationTaskRegistry()
        let first = UUID()
        let second = UUID()
        #expect(registry.reserve(first))
        registry.clear(first)
        #expect(registry.reserve(second))
        let secondTask = Task<Void, Never> {
            do { try await Task.sleep(for: .seconds(10)) } catch {}
        }
        registry.attach(secondTask, to: second)

        #expect(registry.take(first) == nil)
        #expect(!secondTask.isCancelled)
        registry.take(second)?.cancel()
        #expect(secondTask.isCancelled)
    }

    @Test func generationRegistryRejectsConcurrentReservationAndClearsByOwner() {
        let registry = GenerationTaskRegistry()
        let first = UUID()
        let second = UUID()
        #expect(registry.reserve(first))
        #expect(!registry.reserve(second))
        registry.clear(second)
        #expect(!registry.reserve(second))
        registry.clear(first)
        #expect(registry.reserve(second))
        registry.clear(second)
    }

    @Test func generationRegistryCancelsTaskAttachedAfterReservationEnded() async {
        let registry = GenerationTaskRegistry()
        let id = UUID()
        #expect(registry.reserve(id))
        registry.clear(id)
        let task = Task<Void, Never> {
            do { try await Task.sleep(for: .seconds(10)) } catch {}
        }
        registry.attach(task, to: id)
        #expect(task.isCancelled)
        let next = UUID()
        #expect(registry.reserve(next))
        registry.clear(next)
    }

    @Test func generationRunnerPolicyKeepsFusionHeadForPureGreedyChunkedPrefill() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.moepack"),
            prompt: "hello",
            temperature: 0,
            repetitionPenalty: 1)

        #expect(!RealInferenceSession.forceLogitsHead(for: request))
    }

    @Test func generationRunnerPolicyForcesLogitsForSamplingChunkedPrefill() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.moepack"),
            prompt: "hello",
            temperature: 0.7,
            repetitionPenalty: 1)

        #expect(RealInferenceSession.forceLogitsHead(for: request))
    }

    @Test func validatedChatRequestCarriesDocumentedSamplingPolicy() throws {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.moepack"),
            prompt: "hello")

        let validated = try RealInferenceSession.validatedChatRequest(
            for: request, kind: .gemmaQATSym)
        #expect(validated.generationConfig.temperature == 1.0)
        #expect(validated.generationConfig.topK == 64)
        #expect(validated.generationConfig.topP == 0.95)
        #expect(validated.generationConfig.repetitionPenalty == 1)
        #expect(validated.messages.count == 1)
        #expect(validated.messages.first?.content == "hello")
        #expect(!validated.enableThinking)
        #expect(validated.vision == nil)
        #expect(validated.cachePrompt)
    }

    @Test func validatedChatRequestCarriesThinkingToggle() throws {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.moepack"),
            prompt: "hello",
            enableThinking: true)

        let validated = try RealInferenceSession.validatedChatRequest(
            for: request, kind: .ornith)
        #expect(validated.enableThinking)
    }

    /// MSG-5: assistant turns are redrawn with their reasoning so the
    /// template output extends the generated token sequence and the prompt
    /// cache hits with thinking on.
    @Test func validatedChatRequestRedrawsHistoryWithReasoning() throws {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.moepack"),
            history: [
                AppChatTurn(role: .user, text: "q1"),
                AppChatTurn(role: .assistant, text: "a1", reasoningText: "r1"),
            ],
            prompt: "q2")

        let validated = try RealInferenceSession.validatedChatRequest(
            for: request, kind: .ornith)
        #expect(validated.messages.count == 3)
        #expect(validated.messages[0].role == .user)
        #expect(validated.messages[0].content == "q1")
        #expect(validated.messages[0].reasoningContent == nil)
        #expect(validated.messages[1].role == .assistant)
        #expect(validated.messages[1].content == "a1")
        #expect(validated.messages[1].reasoningContent == "r1")
        #expect(validated.messages[2].role == .user)
        #expect(validated.messages[2].content == "q2")
        #expect(validated.messages[2].reasoningContent == nil)
        #expect(validated.vision == nil)
    }

    @Test func validatedChatRequestOmitsEmptyReasoningFromRedraw() throws {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.moepack"),
            history: [
                AppChatTurn(role: .user, text: "q1"),
                AppChatTurn(role: .assistant, text: "a1"),
            ],
            prompt: "q2")

        let validated = try RealInferenceSession.validatedChatRequest(
            for: request, kind: .gemmaQATSym)
        #expect(validated.messages[1].reasoningContent == nil)
    }

    @Test func validatedChatRequestRefusesHistoryImagesWithoutVision() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.moepack"),
            history: [
                AppChatTurn(role: .user, text: "look",
                            imagePaths: ["/tmp/image.png"]),
                AppChatTurn(role: .assistant, text: "seen"),
            ],
            prompt: "next")

        #expect(throws: AppInferenceError.self) {
            _ = try RealInferenceSession.validatedChatRequest(
                for: request, kind: .ornith)
        }
    }

    @Test func validatedChatRequestCollectsImagesAcrossTurns() throws {
        let first = try writeTestPNG(named: "history-image")
        let second = try writeTestPNG(named: "live-image")
        defer {
            try? FileManager.default.removeItem(at: first)
            try? FileManager.default.removeItem(at: second)
        }
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.moepack"),
            history: [
                AppChatTurn(role: .user, text: "what is this",
                            imagePaths: [first.path]),
                AppChatTurn(role: .assistant, text: "a bird"),
            ],
            prompt: "and this",
            imagePaths: [second.path])

        let validated = try RealInferenceSession.validatedChatRequest(
            for: request, kind: .gemmaQATSym)
        let vision = try #require(validated.vision)
        #expect(vision.images.count == 2)
        #expect(vision.messages.count == 3)
        #expect(vision.messages[0].parts == [.image, .text("what is this")])
        #expect(vision.messages[1].parts == [.text("a bird")])
        #expect(vision.messages[2].parts == [.image, .text("and this")])
    }

    @Test func validatedChatRequestRefusesImagesWithoutVision() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.moepack"),
            prompt: "hello",
            imagePaths: ["/tmp/image.png"])

        #expect(throws: AppInferenceError.self) {
            _ = try RealInferenceSession.validatedChatRequest(
                for: request, kind: .ornith)
        }
    }

    @Test func generateWithoutLoadedModelFailsWithoutPartialDiagnostics() async throws {
        let client = RealInferenceClient()
        let modelDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("moepack-prefill-off-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelDirectory,
                                                withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let request = AppGenerationRequest(
            modelDirectory: modelDirectory,
            prompt: "hello",
            runtimeOptions: AppRuntimeOptions(prefillEnabled: false))

        var failure: AppInferenceError?
        var partial: AppDiagnostics?
        do {
            for try await event in client.generate(request) {
                if case .failed(let error, let diagnostics) = event {
                    failure = error
                    partial = diagnostics
                }
            }
        } catch let error as AppInferenceError {
            failure = failure ?? error
        } catch {
            Issue.record("unexpected error type: \(error)")
        }

        #expect(failure != nil)
        #expect(partial == nil)
    }

    @Test func ensureLoadedFailsFastForMissingDirectory() async {
        let client = RealInferenceClient()
        var states: [AppModelLoadState] = []
        let recorder = StateRecorder()

        await #expect(throws: AppInferenceError.self) {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/nonexistent/model.moepack"),
                maxContextTokens: 1024,
                options: AppRuntimeOptions(),
                forceLogitsHead: false,
                onState: { recorder.append($0) })
        }
        states = recorder.snapshot()
        #expect(states.first == .loading(.validatingDirectory))
        #expect(states.last?.isFailed == true)
        #expect(!states.contains(.loading(.tokenizer)))
    }

    @Test func generateWithMissingDirectoryFailsStream() async {
        let client = RealInferenceClient()
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model.moepack"),
            prompt: "hello")

        var failure: AppInferenceError?
        do {
            for try await event in client.generate(request) {
                if case .failed(let error, _) = event { failure = error }
            }
        } catch let error as AppInferenceError {
            failure = failure ?? error
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(failure == .modelNotFound("/nonexistent/model.moepack"))
    }

    @Test func mtpDegradesToOffWhenNothingIsInstalled() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("no-draft-\(UUID().uuidString).moepack")
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{\"arch\": {}}".utf8).write(
            to: directory.appendingPathComponent("manifest.json"))

        #expect(RealInferenceSession.resolvedDraftBlockSize(
            kind: .gemmaQATSym, modelDirectory: directory, requested: true) == 0)
        #expect(RealInferenceSession.resolvedDraftBlockSize(
            kind: .gemmaQATSym, modelDirectory: directory, requested: false) == 0)
    }

    @Test func cancelWhenIdleIsNoOp() {
        let client = RealInferenceClient()
        client.cancel()
        client.cancel()
    }

    @Test func unloadWhenIdleIsSafe() async {
        let client = RealInferenceClient()
        await client.unload()
    }

    /// A real PNG on disk, synthesised so the attachment decoder has an
    /// actual container to open.
    private func writeTestPNG(named name: String) throws -> URL {
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let context = try #require(CGContext(
            data: nil, width: 8, height: 8,
            bitsPerComponent: 8, bytesPerRow: 8 * 4, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.setFillColor(CGColor(red: 0.2, green: 0.6, blue: 0.9, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
        let image = try #require(context.makeImage())
        let data = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(
            data, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(name)-\(UUID().uuidString).png")
        try (data as Data).write(to: url)
        return url
    }
}

private final class StateRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var states: [AppModelLoadState] = []

    func append(_ state: AppModelLoadState) {
        lock.lock()
        states.append(state)
        lock.unlock()
    }

    func snapshot() -> [AppModelLoadState] {
        lock.lock()
        defer { lock.unlock() }
        return states
    }
}
