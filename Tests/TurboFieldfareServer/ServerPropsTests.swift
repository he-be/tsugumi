import Foundation
import NIOCore
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

private actor PropsStubBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

/// EP-4. `/props` is where a client works out what this server can do, so the
/// tests here are about the contents and not the transport.
@Suite("Server /props", .serialized)
struct ServerPropsTests {
    private static let modelPath = "/tmp/turbofieldfare-tests/gemma4.gturbo"
    private static let contextLength = 32_768

    private static func started() async throws -> (TurboFieldfareHTTPServer, Int) {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: PropsStubBackend(),
            properties: ServerProperties(
                modelPath: modelPath,
                contextLength: contextLength,
                chatTemplate: try ServerChatTemplate.jinja()))
        let channel = try await server.start(port: 0)
        guard let port = channel.localAddress?.port else {
            throw ServerRequestError.invalid(message: "no port")
        }
        return (server, port)
    }

    private static func props(port: Int) async throws -> JSONValue {
        let (data, response) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/props")!)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        return try JSONDecoder().decode(JSONValue.self, from: data)
    }

    /// EP-4: the capability answer. `total_slots` is the one generation slot
    /// this machine has (DEV-3) and `modalities` says vision, which is what a
    /// client checks before it sends an image.
    @Test func EP_4_props_reports_slots_path_template_and_modalities() async throws {
        let (server, port) = try await Self.started()
        let props = try await Self.props(port: port).objectValue ?? [:]

        #expect(props["total_slots"] == .integer(1))
        #expect(props["model_path"] == .string(Self.modelPath))
        #expect(props["modalities"] == .object(["vision": .bool(true)]))
        #expect(props["chat_template"] == .string(try ServerChatTemplate.jinja()))
        if case .string(let build)? = props["build_info"] {
            #expect(!build.isEmpty)
        } else {
            Issue.record("build_info is missing or is not a string")
        }

        try await server.shutdown()
    }

    /// EP-4: the effective `n_ctx`, which is what `--ctx-size` rounded down to
    /// (FLAG-2) — a client sizing a conversation has no other way to read it.
    @Test func EP_4_props_reports_the_effective_context_length() async throws {
        let (server, port) = try await Self.started()
        let props = try await Self.props(port: port).objectValue ?? [:]
        let settings = props["default_generation_settings"]?.objectValue ?? [:]

        #expect(settings["n_ctx"] == .integer(Int64(Self.contextLength)))

        try await server.shutdown()
    }

    /// EP-4: `default_generation_settings` is the effective value of every
    /// SPEC §4 row that has a default — read off the same declarative table the
    /// request parser uses, so the two cannot drift.
    @Test func EP_4_default_generation_settings_are_the_request_table_defaults() async throws {
        let (server, port) = try await Self.started()
        let props = try await Self.props(port: port).objectValue ?? [:]
        let settings = props["default_generation_settings"]?.objectValue ?? [:]

        var checked = 0
        for field in ChatRequestSchema.fields {
            guard let expected = field.defaultValue else { continue }
            checked += 1
            guard let reported = settings[field.name] else {
                Issue.record("\(field.id): /props does not report \(field.name)")
                continue
            }
            #expect(Self.sameJSONNumberOrValue(reported, expected),
                    "\(field.id) (\(field.name))")
        }
        // The table is not empty and the loop is not vacuous.
        #expect(checked >= 10)

        try await server.shutdown()
    }

    /// EP-4: the spot values a client is most likely to read, spelled out so a
    /// change to the table shows up here as a failure and not as a diff of two
    /// derivations of the same thing.
    @Test func EP_4_default_generation_settings_hold_the_spec_table_values() async throws {
        let (server, port) = try await Self.started()
        let props = try await Self.props(port: port).objectValue ?? [:]
        let settings = props["default_generation_settings"]?.objectValue ?? [:]

        #expect(Self.sameJSONNumberOrValue(settings["temperature"] ?? .null, .number(1.0)))
        #expect(Self.sameJSONNumberOrValue(settings["top_p"] ?? .null, .number(1.0)))
        #expect(settings["top_k"] == .integer(0))
        #expect(settings["seed"] == .integer(-1))
        #expect(settings["max_tokens"] == .integer(-1))
        #expect(settings["n"] == .integer(1))
        #expect(settings["stream"] == .bool(false))
        #expect(settings["cache_prompt"] == .bool(true))
        #expect(settings["parallel_tool_calls"] == .bool(true))
        #expect(settings["tool_choice"] == .string("auto"))
        #expect(settings["reasoning_format"] == .string("auto"))
        #expect(settings["reasoning_budget_tokens"] == .integer(-1))
        #expect(settings["timings_per_token"] == .bool(false))
        #expect(Self.sameJSONNumberOrValue(settings["repeat_penalty"] ?? .null, .number(1.0)))

        try await server.shutdown()
    }

    /// A JSON number carries no Swift type: `1.0` on the wire comes back as an
    /// integer. Compare numbers as numbers and everything else by value.
    private static func sameJSONNumberOrValue(_ lhs: JSONValue, _ rhs: JSONValue) -> Bool {
        if let left = lhs.exactDouble, let right = rhs.exactDouble {
            return left == right
        }
        return lhs == rhs
    }
}
