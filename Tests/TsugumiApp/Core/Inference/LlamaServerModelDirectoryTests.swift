import Foundation
import Testing
@testable import TsugumiAppCore

@Suite struct LlamaServerModelDirectoryTests {
    private func makeDirectory(manifest: String, gguf: Bool = true) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("llama-server-dir-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(manifest.utf8).write(to: directory.appendingPathComponent("manifest.json"))
        if gguf { try Data().write(to: directory.appendingPathComponent("model.gguf")) }
        return directory
    }

    @Test func theManifestNamesTheServerTheGGUFAndTheArguments() throws {
        let directory = try makeDirectory(manifest: """
            { "arch": { "family": "qwen3_8_dense_llamacpp" }, "llama_server": "/bin/ls", "gguf": "model.gguf",
              "server_args": ["-ngl", "99", "--spec-draft-n-max", "1"] }
            """)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try LlamaServerModelDirectory(modelDirectory: directory)
        #expect(model.server.path == "/bin/ls")
        #expect(model.gguf.lastPathComponent == "model.gguf")
        #expect(model.arguments(contextTokens: 32_768, port: 5_001) == [
            "-m", model.gguf.path, "-ngl", "99", "--spec-draft-n-max", "1",
            "-c", "32768", "--host", "127.0.0.1", "--port", "5001",
        ])
        #expect(AppModelKind.probe(modelDirectory: directory) == .bonsai27b)
    }

    @Test func anArgumentTheClientSetsIsRefused() throws {
        let directory = try makeDirectory(manifest: """
            { "arch": {}, "llama_server": "/bin/ls", "gguf": "model.gguf", "server_args": ["-c", "4096"] }
            """)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(throws: AppInferenceError.self) { try LlamaServerModelDirectory(modelDirectory: directory) }
    }

    @Test func aMissingServerOrGGUFFailsTheLoad() throws {
        let noServer = try makeDirectory(manifest: """
            { "arch": {}, "llama_server": "/nonexistent/llama-server", "gguf": "model.gguf" }
            """)
        defer { try? FileManager.default.removeItem(at: noServer) }
        #expect(throws: AppInferenceError.self) { try LlamaServerModelDirectory(modelDirectory: noServer) }
        let noGGUF = try makeDirectory(manifest: """
            { "arch": {}, "llama_server": "/bin/ls", "gguf": "model.gguf" }
            """, gguf: false)
        defer { try? FileManager.default.removeItem(at: noGGUF) }
        #expect(throws: AppInferenceError.self) { try LlamaServerModelDirectory(modelDirectory: noGGUF) }
    }

    @Test func aFreePortIsOneTheServerCanTake() throws {
        let port = try LlamaServerInferenceClient.freePort()
        #expect((1_024...65_535).contains(port))
    }

    @Test func generatingBeforeALoadFails() async throws {
        let client = LlamaServerInferenceClient(
            stateDirectory: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString))
        var events: [AppInferenceEvent] = []
        for try await event in client.generate(AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/tmp"),
                                                                    prompt: "hi")) {
            events.append(event)
        }
        guard case .failed(.modelNotLoaded, _)? = events.first else {
            Issue.record("expected modelNotLoaded, got \(events)")
            return
        }
    }
}
