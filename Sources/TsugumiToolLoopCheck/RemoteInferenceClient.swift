import Foundation
import Tsugumi
import TsugumiAppCore

/// The app's inference client pointed at a llama-server behind llama-swap instead of this Mac's decode service
/// (`--endpoint`, `docs/experiments/knowledge-sources`). The tool loop around it stays the app's: `AppModel` builds the
/// requests (system prompt, tools, the Online policy, the round budget) and the app's executors run the calls here.
///
/// Three shapes of round, by `toolChoice`:
///   auto      `/v1/chat/completions`, streamed; llama-server renders the model's own template and parses the calls.
///   function  llama-server (b10825) ignores a forced `tool_choice`: `required` and a named function both came back
///   required  as plain text. The round renders the same messages with `/apply-template`, appends the opening of the
///             call (`<|tool_call>call:web_search{`), continues it with `/v1/completions` up to the closing marker, and
///             reads the call with the app's parser (`GemmaToolCallParser` / `QwenToolCallParser`). A call the parser
///             rejects fails as `structured_output_failure`, which the app retries once, as it does on the Mac.
///   none      chat completions with the call-opening token banned by `logit_bias`; the declarations stay in the
///             prompt, as the app's `ForbiddenTokensConstraint` keeps them.
final class RemoteInferenceClient: AppModelLifecycleClient, AppInferenceRuntimeReporting, @unchecked Sendable {
    enum Dialect: String {
        case gemma
        case qwen

        init(kind: AppModelKind) { self = kind == .qwen38 ? .qwen : .gemma }

        var callOpen: String { self == .gemma ? "<|tool_call>" : "<tool_call>" }
        var callClose: String { self == .gemma ? "<tool_call|>" : "</tool_call>" }

        /// What the forced round appends after the rendered prompt, and the part of it the parser reads.
        func forcedPrefix(name: String?) -> (appended: String, parsed: String) {
            switch self {
            case .gemma:
                let parsed = name.map { "call:\($0){" } ?? "call:"
                return (callOpen + parsed, parsed)
            case .qwen:
                let parsed = name.map { "<function=\($0)>\n" } ?? "<function="
                return (callOpen + "\n" + parsed, parsed)
            }
        }
    }

    let endpoint: URL
    let modelID: String
    let dialect: Dialect
    /// The per-slot context llama-server reports (`default_generation_settings.n_ctx`), read at load.
    private(set) var serverContext: Int?
    private var callOpenTokenID: Int?
    private let session: URLSession
    private let lock = NSLock()
    private var running: Task<Void, Never>?

    var loadedRuntimeOwnBytes: UInt64? { nil }

    init(endpoint: URL, modelID: String, dialect: Dialect) {
        self.endpoint = endpoint
        self.modelID = modelID
        self.dialect = dialect
        let configuration = URLSessionConfiguration.ephemeral
        // llama-swap starts the model on the first request (about 3 minutes) and a long prefill sends nothing.
        configuration.timeoutIntervalForRequest = 1_800
        configuration.timeoutIntervalForResource = 7_200
        session = URLSession(configuration: configuration)
    }

    // MARK: Lifecycle

    /// Brings the model up through llama-swap and reads what the rounds need: the slot's context (which the caller
    /// sets on `AppModel` before loading, since a changed context asks for a reload) and the call-opening token.
    func prepare() async throws {
        // One token through the router, so llama-swap has the model up before the first timed round.
        _ = try await postJSON("v1/chat/completions", [
            "model": modelID, "max_tokens": 1,
            "messages": [["role": "user", "content": "hi"]],
        ])
        let props = try await getJSON("upstream/\(modelID)/props")
        serverContext = (props["default_generation_settings"] as? [String: Any])?["n_ctx"] as? Int
        let tokens = try await postJSON("upstream/\(modelID)/tokenize",
                                        ["content": dialect.callOpen, "parse_special": true])
        if let list = tokens["tokens"] as? [Any], list.count == 1 {
            callOpenTokenID = (list[0] as? Int) ?? ((list[0] as? [String: Any])?["id"] as? Int)
        }
        guard callOpenTokenID != nil else {
            throw AppInferenceError.modelLoadFailed("\(dialect.callOpen) is not one token on \(modelID)")
        }
    }

    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int, options: AppRuntimeOptions, forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        if callOpenTokenID == nil { try await prepare() }
        onState(.ready(modelDirectory: modelDirectory, loadSeconds: 0))
    }

    func unload() async {}

    func cancel() {
        lock.lock(); defer { lock.unlock() }
        running?.cancel()
    }

    // MARK: Generation

    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    switch request.toolChoice {
                    case .function(let name): try await forcedRound(request, name: name, into: continuation)
                    case .required: try await forcedRound(request, name: nil, into: continuation)
                    case .auto, .none: try await chatRound(request, into: continuation)
                    }
                } catch is CancellationError {
                    continuation.yield(.cancelled(Self.diagnostics(request, timings: [:], stop: .cancelled)))
                } catch let error as AppInferenceError {
                    continuation.yield(.failed(error, partial: nil))
                } catch {
                    if Task.isCancelled {
                        continuation.yield(.cancelled(Self.diagnostics(request, timings: [:], stop: .cancelled)))
                    } else {
                        continuation.yield(.failed(.unknown("\(error)"), partial: nil))
                    }
                }
                continuation.finish()
            }
            lock.lock()
            running = task
            lock.unlock()
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func chatRound(_ request: AppGenerationRequest,
                           into continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation) async throws {
        var body = requestBody(request)
        body["stream"] = true
        body["stream_options"] = ["include_usage": true]
        if request.toolChoice == .none, !request.tools.isEmpty, let id = callOpenTokenID {
            body["logit_bias"] = [[id, false]]
        }
        if !request.tools.isEmpty {
            body["tool_choice"] = "auto"
            body["parallel_tool_calls"] = true
        }

        let (bytes, response) = try await session.bytes(for: urlRequest("v1/chat/completions", body))
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            var text = ""
            for try await line in bytes.lines { text += line }
            throw Self.serverError(status: http.statusCode, body: text)
        }

        let decodeStart = Date()
        var index = 0
        var finish = ""
        var timings: [String: Any] = [:]
        var calls: [Int: (id: String, name: String, arguments: String)] = [:]
        for try await line in bytes.lines {
            guard line.hasPrefix("data: ") else { continue }
            let payload = line.dropFirst("data: ".count)
            if payload == "[DONE]" { break }
            guard let object = try JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any] else {
                continue
            }
            if let error = object["error"] {
                throw AppInferenceError.unknown("server error in stream: \(error)")
            }
            if let t = object["timings"] as? [String: Any] { timings = t }
            guard let choice = (object["choices"] as? [[String: Any]])?.first else { continue }
            if let reason = choice["finish_reason"] as? String { finish = reason }
            guard let delta = choice["delta"] as? [String: Any] else { continue }
            let text = delta["content"] as? String ?? ""
            let reasoning = delta["reasoning_content"] as? String ?? ""
            if !text.isEmpty || !reasoning.isEmpty {
                continuation.yield(.token(AppTokenEvent(index: index, textDelta: text,
                                                        elapsedDecodeSeconds: Date().timeIntervalSince(decodeStart),
                                                        reasoningDelta: reasoning)))
                index += 1
            }
            for fragment in delta["tool_calls"] as? [[String: Any]] ?? [] {
                let slot = fragment["index"] as? Int ?? 0
                var call = calls[slot] ?? ("", "", "")
                if let id = fragment["id"] as? String, !id.isEmpty { call.id = id }
                if let function = fragment["function"] as? [String: Any] {
                    call.name += function["name"] as? String ?? ""
                    call.arguments += function["arguments"] as? String ?? ""
                }
                calls[slot] = call
            }
        }
        try Task.checkCancellation()

        // A call written on a `none` round would restart the loop the app has just closed; the ban should make this
        // impossible, and the round is kept as an answer if it happens.
        let parsed = request.toolChoice == .none ? [] : calls.keys.sorted().compactMap { calls[$0] }
        for call in parsed {
            continuation.yield(.toolCall(AppToolCall(id: call.id.isEmpty ? Self.callID() : call.id,
                                                     name: call.name,
                                                     argumentsJSON: call.arguments.isEmpty ? "{}" : call.arguments)))
        }
        let stop: AppStopReason = !parsed.isEmpty ? .toolCalls : finish == "length" ? .maxTokens : .endOfTurn
        continuation.yield(.finished(Self.diagnostics(request, timings: timings, stop: stop)))
    }

    private func forcedRound(_ request: AppGenerationRequest, name: String?,
                             into continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation) async throws {
        var templateBody = requestBody(request)
        templateBody.removeValue(forKey: "max_tokens")
        let rendered = try await postJSON("upstream/\(modelID)/apply-template", templateBody)
        guard let prompt = rendered["prompt"] as? String else {
            throw AppInferenceError.unknown("apply-template returned no prompt")
        }
        let prefix = dialect.forcedPrefix(name: name)
        var body: [String: Any] = [
            "model": modelID,
            "prompt": prompt + prefix.appended,
            "stop": [dialect.callClose],
            "max_tokens": 1_024,
            "temperature": request.temperature,
            "cache_prompt": true,
        ]
        if let topK = request.topK { body["top_k"] = topK }
        if let topP = request.topP { body["top_p"] = topP }
        let result = try await postJSON("v1/completions", body)
        try Task.checkCancellation()
        let timings = result["timings"] as? [String: Any] ?? [:]
        let choice = (result["choices"] as? [[String: Any]])?.first ?? [:]
        let text = choice["text"] as? String ?? ""
        let finish = choice["finish_reason"] as? String ?? ""

        let id = Self.callID()
        let allowed = request.tools.map(\.name)
        do {
            guard finish == "stop" else { throw GemmaToolCallParserError.malformed }
            let call: ParsedToolCall
            switch dialect {
            case .gemma:
                call = try GemmaToolCallParser().parse(prefix.parsed + text, allowedTools: Set(allowed), id: id)
            case .qwen:
                let definitions = try request.tools.map { tool in
                    GFTokenizer.FunctionDefinition(
                        name: tool.name, description: tool.description,
                        parameters: try JSONDecoder().decode(JSONValue.self, from: Data(tool.parametersJSON.utf8)),
                        parametersSource: tool.parametersJSON)
                }
                call = try QwenToolCallParser(tools: definitions).parse(prefix.parsed + text, id: id)
            }
            if let name, call.name != name { throw GemmaToolCallParserError.unknownTool(call.name) }
            continuation.yield(.toolCall(AppToolCall(id: call.id, name: call.name, argumentsJSON: call.argumentsJSON)))
            continuation.yield(.finished(Self.diagnostics(request, timings: timings, stop: .toolCalls)))
        } catch {
            let snippet = String((prefix.parsed + text).prefix(300))
            throw AppInferenceError.unknown("structured_output_failure: forced \(name ?? "call") (\(finish)): \(error) — \(snippet)")
        }
    }

    // MARK: Request shape

    private func requestBody(_ request: AppGenerationRequest) -> [String: Any] {
        var messages: [[String: Any]] = []
        if let system = request.systemPrompt { messages.append(["role": "system", "content": system]) }
        messages.append(contentsOf: request.history.map(Self.message))
        messages.append(["role": "user", "content": request.prompt])
        messages.append(contentsOf: request.continuation.map(Self.message))

        var body: [String: Any] = [
            "model": modelID,
            "messages": messages,
            "temperature": request.temperature,
            // The app lets a round run to the context; a runaway repetition is cut here instead.
            "max_tokens": min(request.maxNewTokens, 8_192),
            "chat_template_kwargs": ["enable_thinking": request.enableThinking],
            "cache_prompt": true,
        ]
        if let topK = request.topK { body["top_k"] = topK }
        if let topP = request.topP { body["top_p"] = topP }
        if !request.tools.isEmpty {
            body["tools"] = request.tools.map { tool -> [String: Any] in
                let parameters = (try? JSONSerialization.jsonObject(with: Data(tool.parametersJSON.utf8)))
                    ?? ["type": "object", "properties": [String: Any]()]
                return ["type": "function",
                        "function": ["name": tool.name, "description": tool.description, "parameters": parameters]]
            }
        }
        return body
    }

    private static func message(_ turn: AppChatTurn) -> [String: Any] {
        switch turn.role {
        case .user:
            return ["role": "user", "content": turn.text]
        case .assistant:
            var message: [String: Any] = ["role": "assistant", "content": turn.text]
            if !turn.reasoningText.isEmpty { message["reasoning_content"] = turn.reasoningText }
            if !turn.toolCalls.isEmpty {
                message["tool_calls"] = turn.toolCalls.map {
                    ["id": $0.id, "type": "function", "function": ["name": $0.name, "arguments": $0.argumentsJSON]]
                }
            }
            return message
        case .tool:
            var message: [String: Any] = ["role": "tool", "tool_call_id": turn.toolCallID ?? "", "content": turn.text]
            if let name = turn.toolName { message["name"] = name }
            return message
        }
    }

    private static func diagnostics(_ request: AppGenerationRequest, timings: [String: Any],
                                    stop: AppStopReason) -> AppDiagnostics {
        func number(_ key: String) -> Double? { (timings[key] as? NSNumber)?.doubleValue }
        let cached = (timings["cache_n"] as? Int) ?? 0
        let evaluated = (timings["prompt_n"] as? Int) ?? 0
        var speculative: AppSpeculativeDiagnostics?
        if let proposed = timings["draft_n"] as? Int, proposed > 0 {
            speculative = AppSpeculativeDiagnostics(blockTokens: 1, proposed: proposed,
                                                    accepted: (timings["draft_n_accepted"] as? Int) ?? 0)
        }
        return AppDiagnostics(generatedTokens: (timings["predicted_n"] as? Int) ?? 0,
                              stopReason: stop,
                              promptTokenCount: timings.isEmpty ? nil : cached + evaluated,
                              cachedPromptTokens: timings.isEmpty ? nil : cached,
                              speculative: speculative,
                              prefillSeconds: number("prompt_ms").map { $0 / 1_000 },
                              timeToFirstTokenSeconds: nil,
                              decodeSeconds: (number("predicted_ms") ?? 0) / 1_000,
                              tokensPerSecond: number("predicted_per_second") ?? 0,
                              peakMemoryBytes: nil,
                              runtimeOptions: request.runtimeOptions)
    }

    private static func callID() -> String {
        "call_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16).lowercased()
    }

    /// llama-server's error body; a prompt past the slot's context becomes the app's overflow error.
    private static func serverError(status: Int, body: String) -> AppInferenceError {
        let error = ((try? JSONSerialization.jsonObject(with: Data(body.utf8))) as? [String: Any])?["error"]
            as? [String: Any]
        if (error?["type"] as? String) == "exceed_context_size_error" {
            return .contextOverflow(prompt: error?["n_prompt_tokens"] as? Int ?? 0, maxNew: 0,
                                    maxContext: error?["n_ctx"] as? Int ?? 0)
        }
        return .unknown("HTTP \(status): \(String(body.prefix(500)))")
    }

    // MARK: HTTP

    private func urlRequest(_ path: String, _ body: [String: Any]) throws -> URLRequest {
        var request = URLRequest(url: endpoint.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        return request
    }

    private func postJSON(_ path: String, _ body: [String: Any]) async throws -> [String: Any] {
        let (data, response) = try await session.data(for: urlRequest(path, body))
        return try Self.decode(data, response)
    }

    private func getJSON(_ path: String) async throws -> [String: Any] {
        let (data, response) = try await session.data(from: endpoint.appendingPathComponent(path))
        return try Self.decode(data, response)
    }

    private static func decode(_ data: Data, _ response: URLResponse) throws -> [String: Any] {
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else { throw serverError(status: status, body: String(decoding: data, as: UTF8.self)) }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AppInferenceError.unknown("not a JSON object: \(String(decoding: data.prefix(200), as: UTF8.self))")
        }
        return object
    }
}
