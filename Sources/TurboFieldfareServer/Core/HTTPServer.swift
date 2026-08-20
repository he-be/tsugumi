import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import TurboFieldfare

extension ServerRequestError {
    /// ERR-2's mapping, read off the error's own type. Kept next to the HTTP
    /// layer because the error type itself is transport-agnostic.
    var httpStatus: HTTPResponseStatus {
        HTTPResponseStatus(statusCode: type.httpStatusCode)
    }
}

/// What `/props` answers with about this process (EP-4).
///
/// Everything here is known before the model is: the flags say the path and the
/// context, and the template is the repo's own file. That is what lets `/props`
/// be one value handed to the server at startup rather than a question asked of
/// a backend that may not exist yet (LIF-1).
public struct ServerProperties: Equatable, Sendable {
    /// EP-4 `model_path`: the directory this server was started with.
    public let modelPath: String
    /// EP-4's effective `n_ctx` — the value `--ctx-size` rounded down to
    /// (FLAG-2), which a client has no other way to read.
    public let contextLength: Int
    /// EP-4 `chat_template`: the template the server actually renders with,
    /// which is the repo-owned variant of DEV-12 and not the file the
    /// checkpoint ships.
    public let chatTemplate: String

    /// EP-4 `total_slots`. Generation is one slot on this machine, fixed
    /// (DEV-3), so this is a constant and not a flag.
    public static let totalSlots = 1

    /// EP-4 `build_info`. SPEC names the field but not its contents, and this
    /// checkout stamps no build number, so the honest value is the name of the
    /// binary answering.
    public static let buildInfo = "TurboFieldfareServer"

    public init(modelPath: String = "",
                contextLength: Int = 0,
                chatTemplate: String = "") {
        self.modelPath = modelPath
        self.contextLength = contextLength
        self.chatTemplate = chatTemplate
    }
}

public actor TurboFieldfareHTTPServer {
    /// The text-only body ceiling. Kept as the name it always had; a server
    /// configured for images raises its own ceiling from `ServerImagePolicy`.
    public static let maximumBodyBytes = ServerImagePolicy.textBodyBytes

    private let group: MultiThreadedEventLoopGroup
    private let modelID: String
    private let readiness: ServerReadiness
    private let coordinator: ServerCoordinator
    private let heartbeatInterval: TimeAmount
    private let imagePolicy: ServerImagePolicy
    private let defaults: ChatRequestDefaults
    private let properties: ServerProperties
    private let childChannels = ChildChannelRegistry()
    private var channel: Channel?
    private var shutdownTask: Task<Void, any Error>?

    public init(modelID: String,
                queueLimit: Int,
                backend: (any ServerInferenceBackend)?,
                heartbeatInterval: TimeAmount = .seconds(5),
                imagePolicy: ServerImagePolicy = .default,
                defaults: ChatRequestDefaults = ChatRequestDefaults(),
                properties: ServerProperties = ServerProperties(),
                group: MultiThreadedEventLoopGroup = .init(numberOfThreads: 1)) {
        self.group = group
        self.modelID = modelID
        self.readiness = ServerReadiness(backend)
        self.coordinator = ServerCoordinator(queueLimit: queueLimit)
        self.heartbeatInterval = heartbeatInterval
        self.imagePolicy = imagePolicy
        self.defaults = defaults
        self.properties = properties
    }

    /// LIF-2 → LIF-3. The load finished and the endpoints may answer from the
    /// model. Until this is called the server is listening but not ready, which
    /// is the whole point of LIF-1: the client sees a status, never a refused
    /// connection.
    public func modelDidLoad(_ backend: any ServerInferenceBackend) {
        readiness.modelDidLoad(backend)
    }

    /// LIF-3: whether the load has landed.
    public var isReady: Bool { readiness.backend != nil }

    public func start(port: Int) async throws -> Channel {
        let modelID = self.modelID
        let readiness = self.readiness
        let coordinator = self.coordinator
        let heartbeatInterval = self.heartbeatInterval
        let imagePolicy = self.imagePolicy
        let defaults = self.defaults
        let properties = self.properties
        let childChannels = self.childChannels
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 16)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                childChannels.insert(channel)
                return channel.pipeline.configureHTTPServerPipeline(
                    withPipeliningAssistance: true,
                    withErrorHandling: true
                ).flatMap {
                    channel.pipeline.addHandler(ServerHTTPHandler(
                        modelID: modelID,
                        readiness: readiness,
                        coordinator: coordinator,
                        heartbeatInterval: heartbeatInterval,
                        imagePolicy: imagePolicy,
                        defaults: defaults,
                        properties: properties,
                        childChannels: childChannels))
                }
            }
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
        let channel = try await bootstrap.bind(host: "127.0.0.1", port: port).get()
        self.channel = channel
        return channel
    }

    public func shutdown() async throws {
        if let shutdownTask {
            try await shutdownTask.value
            return
        }

        let listeningChannel = channel
        channel = nil
        let childChannels = self.childChannels
        let coordinator = self.coordinator
        let group = self.group
        let task = Task { @Sendable in
            var firstError: (any Error)?
            await coordinator.shutdown()
            if let listeningChannel {
                do {
                    try await listeningChannel.close().get()
                } catch ChannelError.alreadyClosed {
                } catch {
                    firstError = error
                }
            }
            await childChannels.closeAll()
            do {
                try await group.shutdownGracefully()
            } catch {
                if firstError == nil {
                    firstError = error
                }
            }
            if let firstError {
                throw firstError
            }
        }
        shutdownTask = task
        try await task.value
    }

    var queuedRequestCount: Int {
        get async { await coordinator.queuedCount }
    }

    var hasActiveRequest: Bool {
        get async { await coordinator.isActive }
    }

    var acceptedConnectionCount: Int {
        childChannels.count
    }
}

/// LIF-1: the listening socket is open before the model exists, so the handler
/// cannot capture a backend when the pipeline is built — it reads one out of
/// here per request, and finds nothing until the load lands.
private final class ServerReadiness: Sendable {
    private let state: Mutex<(any ServerInferenceBackend)?>

    init(_ backend: (any ServerInferenceBackend)?) {
        state = Mutex(backend)
    }

    var backend: (any ServerInferenceBackend)? {
        state.withLock { $0 }
    }

    func modelDidLoad(_ backend: any ServerInferenceBackend) {
        state.withLock { $0 = backend }
    }
}

private final class ServerHTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let modelID: String
    private let readiness: ServerReadiness
    private let coordinator: ServerCoordinator
    private let heartbeatInterval: TimeAmount
    private let imagePolicy: ServerImagePolicy
    private let defaults: ChatRequestDefaults
    private let properties: ServerProperties
    private let maximumBodyBytes: Int
    private let childChannels: ChildChannelRegistry
    private var head: HTTPRequestHead?
    private var body = ByteBuffer()
    private var oversized = false
    private var activeTask: Task<Void, Never>?

    init(modelID: String,
         readiness: ServerReadiness,
         coordinator: ServerCoordinator,
         heartbeatInterval: TimeAmount,
         imagePolicy: ServerImagePolicy,
         defaults: ChatRequestDefaults,
         properties: ServerProperties,
         childChannels: ChildChannelRegistry) {
        self.modelID = modelID
        self.readiness = readiness
        self.coordinator = coordinator
        self.heartbeatInterval = heartbeatInterval
        self.imagePolicy = imagePolicy
        self.defaults = defaults
        self.properties = properties
        self.maximumBodyBytes = imagePolicy.maximumBodyBytes
        self.childChannels = childChannels
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            self.head = head
            body.clear()
            oversized = false
        case .body(var part):
            if body.readableBytes + part.readableBytes > maximumBodyBytes {
                oversized = true
            } else {
                body.writeBuffer(&part)
            }
        case .end:
            guard let head else { return }
            self.head = nil
            if oversized {
                writeError(context, status: .payloadTooLarge,
                           OpenAIErrorEnvelope(message: "request body is too large",
                                               code: "request_too_large"))
                return
            }
            route(head: head, body: body, context: context)
        }
    }

    func channelInactive(context: ChannelHandlerContext) {
        activeTask?.cancel()
        activeTask = nil
        childChannels.remove(context.channel)
        context.fireChannelInactive()
    }

    private func route(head: HTTPRequestHead,
                       body: ByteBuffer,
                       context: ChannelHandlerContext) {
        let path = head.uri.split(separator: "?", maxSplits: 1,
                                  omittingEmptySubsequences: false).first.map(String.init) ?? head.uri
        // LIF-2: nothing is answered from the model's side until there is one,
        // and that includes the routes that never touch it — the reference
        // implementation refuses the same way, from a middleware ahead of its
        // routing table (`server-http.cpp:255`). A client that gets this knows
        // the server exists and is coming up, which "connection refused" cannot
        // tell it (LIF-1).
        guard readiness.backend != nil else {
            writeError(context, status: .serviceUnavailable, Self.loadingEnvelope)
            return
        }
        switch (head.method, path) {
        // EP-1: one handler, both spellings, no API key.
        case (.GET, "/health"), (.GET, "/v1/health"):
            writeJSON(context, status: .ok, object: ["status": "ok"])
        case (.GET, "/v1/models"):
            let response = OpenAIModelList(
                object: "list",
                data: [.init(id: modelID,
                             object: "model",
                             created: 0,
                             ownedBy: "turbofieldfare")])
            writeCodable(context, status: .ok, response)
        case (.POST, "/v1/chat/completions"):
            guard head.headers.first(name: "content-type")?
                .lowercased().hasPrefix("application/json") == true else {
                writeError(context, status: .unsupportedMediaType,
                           OpenAIErrorEnvelope(message: "content-type must be application/json",
                                               code: "unsupported_media_type"))
                return
            }
            handleCompletion(body: body, context: context)
        case (_, "/health"), (_, "/v1/health"), (_, "/v1/models"),
             (_, "/v1/chat/completions"):
            writeError(context, status: .methodNotAllowed,
                       OpenAIErrorEnvelope(message: "method not allowed",
                                           code: "method_not_allowed"))
        default:
            writeError(context, status: .notFound,
                       OpenAIErrorEnvelope(message: "route not found",
                                           code: "not_found"))
        }
    }

    /// LIF-2, quoted. The one body SPEC writes out in full, because a client
    /// has to be able to match on it.
    private static let loadingEnvelope = OpenAIErrorEnvelope(
        message: "Loading model",
        type: .unavailable,
        param: nil,
        code: "model_loading")

    private func handleCompletion(body: ByteBuffer,
                                  context: ChannelHandlerContext) {
        // LIF-2 has already turned every request away while this is nil, so
        // reaching here without a model is not a state the server has.
        guard let backend = readiness.backend else {
            writeError(context, status: .serviceUnavailable, Self.loadingEnvelope)
            return
        }
        do {
            let bytes = body.getBytes(at: body.readerIndex, length: body.readableBytes) ?? []
            let request = try ChatRequestParser.parse(
                Data(bytes),
                imagePolicy: imagePolicy,
                defaults: defaults)
            let responseID = "chatcmpl-" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
            // R5: the answer carries the name the client asked for. A request
            // that named no model gets this server's own id back.
            let responseModel = request.model.isEmpty ? modelID : request.model
            let created = Int(Date().timeIntervalSince1970)
            let contextBox = SendableContext(context)
            let streamState = StreamState()
            let phaseState = RequestPhaseState()
            let startStream: @Sendable () -> Void = {
                guard request.stream,
                      streamState.start(eventLoop: contextBox.value.eventLoop,
                                        interval: self.heartbeatInterval,
                                        ping: {
                          self.writeHeartbeat(contextBox.value)
                      }) else { return }
                let future = self.beginStream(
                    contextBox.value,
                    self.chunk(id: responseID, created: created, model: responseModel,
                               delta: ["role": "assistant"],
                               finishReason: nil))
                streamState.setStartFuture(future)
            }
            let onQueued: @Sendable () -> Void = {
                phaseState.set("queued")
                ServerLog.queued(id: responseID)
                startStream()
            }
            activeTask = childChannels.startTask {
                defer { streamState.stop() }
                let started = ContinuousClock.now
                ServerLog.accepted(id: responseID,
                                   streaming: request.stream,
                                   thinking: request.enableThinking)
                do {
                    let completion = try await self.coordinator.runPreparing(
                        onQueued: onQueued,
                        prepare: {
                            let prepared = try await backend.prepare(request)
                            phaseState.set("prepared")
                            ServerLog.prepared(id: responseID,
                                               promptTokens: prepared.promptTokenCount)
                            return prepared
                        },
                        operation: { prepared in
                            try Task.checkCancellation()
                            startStream()
                            try await streamState.waitUntilStarted()
                            try Task.checkCancellation()
                            phaseState.set("generating")
                            ServerLog.generating(id: responseID)
                            return try await backend.generate(prepared) { event in
                                guard request.stream else { return }
                                switch event {
                                case .content(let text):
                                    self.writeStreamChunk(
                                        contextBox.value,
                                        self.chunk(id: responseID, created: created,
                                                   model: responseModel,
                                                   delta: ["content": text],
                                                   finishReason: nil))
                                case .reasoning(let text):
                                    self.writeStreamChunk(
                                        contextBox.value,
                                        self.chunk(id: responseID, created: created,
                                                   model: responseModel,
                                                   delta: ["reasoning_content": text],
                                                   finishReason: nil))
                                case .toolCall(let call):
                                    self.writeToolCall(contextBox.value,
                                                       id: responseID,
                                                       created: created,
                                                       model: responseModel,
                                                       toolIndex: streamState.nextToolIndex(),
                                                       call: call)
                                }
                            }
                    })
                    ServerLog.completed(id: responseID,
                                        duration: started.duration(to: .now),
                                        completion: completion)
                    if request.stream {
                        streamState.stop()
                        self.finishStream(contextBox.value,
                                          id: responseID,
                                          created: created,
                                          responseModel: responseModel,
                                          completion: completion,
                                          includeUsage: request.includeUsage)
                    } else {
                        self.writeCompletion(contextBox.value,
                                             id: responseID,
                                             created: created,
                                             responseModel: responseModel,
                                             completion: completion)
                    }
                } catch {
                    streamState.stop()
                    self.handleAsyncError(error,
                                          context: contextBox.value,
                                          id: responseID,
                                          phase: phaseState.value,
                                          stream: streamState.isStarted)
                }
            }
        } catch let error as ServerRequestError {
            writeError(context, status: error.httpStatus, error.envelope)
        } catch {
            writeError(context, status: .badRequest,
                       OpenAIErrorEnvelope(message: "malformed JSON request",
                                           code: "invalid_json"))
        }
    }

    private func writeCompletion(_ context: ChannelHandlerContext,
                                 id: String,
                                 created: Int,
                                 responseModel: String,
                                 completion: ServerCompletion) {
        let encodedContent: Any =
            completion.content.isEmpty && !completion.toolCalls.isEmpty
                ? NSNull()
                : completion.content
        var message: [String: Any] = [
            "role": "assistant",
            "content": encodedContent,
        ]
        // DeepSeek's field, which the OpenAI-compatible clients here read
        // (pi's adapter takes reasoning_content, reasoning, or reasoning_text).
        // Absent rather than empty when the request did not reason, so a
        // client cannot mistake "no thought channel" for "thought nothing".
        if !completion.reasoningContent.isEmpty {
            message["reasoning_content"] = completion.reasoningContent
        }
        if !completion.toolCalls.isEmpty {
            message["tool_calls"] = completion.toolCalls.map(toolCallObject)
        }
        let object: [String: Any] = [
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": responseModel,
            "choices": [[
                "index": 0,
                "message": message,
                "finish_reason": completion.finishReason,
            ]],
            "usage": usageObject(completion.usage),
        ]
        writeJSON(context, status: .ok, object: object)
    }

    private func beginStream(
        _ context: ChannelHandlerContext,
        _ initialChunk: [String: Any]
    ) -> EventLoopFuture<Void> {
        guard let data = try? JSONSerialization.data(withJSONObject: initialChunk) else {
            return context.eventLoop.makeFailedFuture(ServerRequestError.invalid(
                message: "stream response could not be encoded",
                param: nil,
                code: "internal_error"))
        }
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "connection", value: "keep-alive")
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        let contextBox = SendableContext(context)
        let promise = context.eventLoop.makePromise(of: Void.self)
        context.eventLoop.execute {
            contextBox.value.write(self.wrapOutboundOut(.head(head)),
                promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count + 8)
            buffer.writeString("data: ")
            buffer.writeBytes(data)
            buffer.writeString("\n\n")
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))),
                promise: promise)
        }
        return promise.futureResult
    }

    private func writeToolCall(_ context: ChannelHandlerContext,
                               id: String,
                               created: Int,
                               model: String,
                               toolIndex: Int,
                               call: ParsedToolCall) {
        let fragments = utf8Fragments(call.argumentsJSON, maximumBytes: 1024)
        for (index, fragment) in fragments.enumerated() {
            var function: [String: Any] = ["arguments": fragment]
            var tool: [String: Any] = ["index": toolIndex, "function": function]
            if index == 0 {
                function["name"] = call.name
                tool["id"] = call.id
                tool["type"] = "function"
                tool["function"] = function
            }
            writeStreamChunk(
                context,
                chunk(id: id, created: created, model: model,
                      delta: ["tool_calls": [tool]],
                      finishReason: nil))
        }
    }

    private func finishStream(_ context: ChannelHandlerContext,
                              id: String,
                              created: Int,
                              responseModel: String,
                              completion: ServerCompletion,
                              includeUsage: Bool) {
        writeStreamChunk(
            context,
            chunk(id: id, created: created, model: responseModel,
                  delta: [:],
                  finishReason: completion.finishReason))
        if includeUsage {
            writeStreamChunk(context, [
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": responseModel,
                "choices": [],
                "usage": usageObject(completion.usage),
            ])
        }
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            let buffer = contextBox.value.channel.allocator.buffer(string: "data: [DONE]\n\n")
            contextBox.value.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
        }
    }

    private func chunk(id: String,
                       created: Int,
                       model: String,
                       delta: [String: Any],
                       finishReason: String?) -> [String: Any] {
        let encodedReason: Any = finishReason.map { $0 as Any } ?? NSNull()
        return [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [[
                "index": 0,
                "delta": delta,
                "finish_reason": encodedReason,
            ]],
        ]
    }

    private func writeStreamChunk(_ context: ChannelHandlerContext,
                                  _ object: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count + 8)
            buffer.writeString("data: ")
            buffer.writeBytes(data)
            buffer.writeString("\n\n")
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
        }
    }

    private func writeHeartbeat(_ context: ChannelHandlerContext) {
        let buffer = context.channel.allocator.buffer(string: ": ping\n\n")
        context.writeAndFlush(
            wrapOutboundOut(.body(.byteBuffer(buffer))),
            promise: nil)
    }

    private func handleAsyncError(_ error: Error,
                                  context: ChannelHandlerContext,
                                  id: String,
                                  phase: String,
                                  stream: Bool) {
        let envelope: OpenAIErrorEnvelope
        let status: HTTPResponseStatus
        if let requestError = error as? ServerRequestError {
            status = requestError.httpStatus
            envelope = requestError.envelope
        } else {
            status = .internalServerError
            envelope = OpenAIErrorEnvelope(
                message: "generation failed; see TurboFieldfareServer stderr",
                type: .server,
                code: "internal_error")
        }
        if !(error is CancellationError) {
            ServerLog.failed(id: id, phase: phase, status: status.code, error: error)
        }
        if stream {
            finishStreamWithError(context, envelope: envelope)
            return
        }
        writeError(context, status: status, envelope)
    }

    private func finishStreamWithError(_ context: ChannelHandlerContext,
                                       envelope: OpenAIErrorEnvelope) {
        guard let data = try? JSONEncoder().encode(envelope) else { return }
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            guard contextBox.value.channel.isActive else { return }
            var buffer = contextBox.value.channel.allocator.buffer(
                capacity: data.count + 32)
            buffer.writeString("data: ")
            buffer.writeBytes(data)
            buffer.writeString("\n\ndata: [DONE]\n\n")
            contextBox.value.write(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.end(nil)), promise: nil)
        }
    }

    private func writeCodable<T: Encodable>(_ context: ChannelHandlerContext,
                                            status: HTTPResponseStatus,
                                            _ value: T) {
        guard let data = try? JSONEncoder().encode(value) else { return }
        writeData(context, status: status, data: data)
    }

    private func writeError(_ context: ChannelHandlerContext,
                            status: HTTPResponseStatus,
                            _ error: OpenAIErrorEnvelope) {
        writeCodable(context, status: status, error)
    }

    private func writeJSON(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           object: Any) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        writeData(context, status: status, data: data)
    }

    private func writeData(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           data: Data) {
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var headers = HTTPHeaders()
            headers.add(name: "content-type", value: "application/json")
            headers.add(name: "content-length", value: "\(data.count)")
            contextBox.value.write(self.wrapOutboundOut(.head(
                HTTPResponseHead(version: .http1_1, status: status, headers: headers))),
                promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count)
            buffer.writeBytes(data)
            contextBox.value.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
        }
    }

    private func usageObject(_ usage: OpenAIUsage) -> [String: Any] {
        [
            "prompt_tokens": usage.promptTokens,
            "completion_tokens": usage.completionTokens,
            "total_tokens": usage.totalTokens,
            "prompt_tokens_details": [
                "cached_tokens": usage.promptTokensDetails.cachedTokens,
            ],
        ]
    }

    private func toolCallObject(_ call: ParsedToolCall) -> [String: Any] {
        [
            "id": call.id,
            "type": "function",
            "function": [
                "name": call.name,
                "arguments": call.argumentsJSON,
            ],
        ]
    }

    private func utf8Fragments(_ text: String, maximumBytes: Int) -> [String] {
        guard !text.isEmpty else { return [""] }
        var result: [String] = []
        var current = ""
        var bytes = 0
        for character in text {
            let size = String(character).utf8.count
            if bytes + size > maximumBytes, !current.isEmpty {
                result.append(current)
                current = ""
                bytes = 0
            }
            current.append(character)
            bytes += size
        }
        if !current.isEmpty { result.append(current) }
        return result
    }
}

private final class ChildChannelRegistry: Sendable {
    private struct State {
        var channels: [ObjectIdentifier: Channel] = [:]
        var tasks: [UUID: Task<Void, Never>] = [:]
        var shuttingDown = false
    }

    private let state = Mutex(State())

    func insert(_ channel: Channel) {
        let shouldClose = state.withLock {
            guard !$0.shuttingDown else { return true }
            $0.channels[ObjectIdentifier(channel)] = channel
            return false
        }
        if shouldClose {
            channel.close(promise: nil)
        }
    }

    func remove(_ channel: Channel) {
        _ = state.withLock {
            $0.channels.removeValue(forKey: ObjectIdentifier(channel))
        }
    }

    func startTask(
        _ operation: @escaping @Sendable () async -> Void
    ) -> Task<Void, Never> {
        state.withLock { state in
            let id = UUID()
            let task = Task { [self] in
                defer {
                    _ = self.state.withLock {
                        $0.tasks.removeValue(forKey: id)
                    }
                }
                await operation()
            }
            state.tasks[id] = task
            if state.shuttingDown {
                task.cancel()
            }
            return task
        }
    }

    func closeAll() async {
        let channels = state.withLock {
            $0.shuttingDown = true
            return Array($0.channels.values)
        }
        for channel in channels {
            try? await channel.close().get()
        }
        let tasks = state.withLock { Array($0.tasks.values) }
        for task in tasks {
            task.cancel()
        }
        for task in tasks {
            await task.value
        }
    }

    var count: Int {
        state.withLock { $0.channels.count }
    }
}

private final class SendableContext: @unchecked Sendable {
    let value: ChannelHandlerContext

    init(_ value: ChannelHandlerContext) {
        self.value = value
    }
}

private final class RequestPhaseState: Sendable {
    private let state = Mutex("accepted")

    var value: String { state.withLock { $0 } }

    func set(_ value: String) {
        state.withLock { $0 = value }
    }
}

private final class StreamState: @unchecked Sendable {
    private let lock = NSLock()
    private var started = false
    private var stopped = false
    private var heartbeat: RepeatedTask?
    private var startFuture: EventLoopFuture<Void>?
    private var toolIndex = 0

    var isStarted: Bool {
        lock.withLock { started }
    }

    func start(eventLoop: EventLoop,
               interval: TimeAmount,
               ping: @escaping @Sendable () -> Void) -> Bool {
        lock.withLock {
            guard !started else { return false }
            started = true
            stopped = false
            startFuture = nil
            heartbeat = eventLoop.scheduleRepeatedTask(
                initialDelay: interval,
                delay: interval) { [weak self] _ in
                    guard self?.shouldPing == true else { return }
                    ping()
                }
            return true
        }
    }

    func setStartFuture(_ future: EventLoopFuture<Void>) {
        lock.withLock { startFuture = future }
    }

    func waitUntilStarted() async throws {
        let future = lock.withLock { startFuture }
        if let future {
            try await future.get()
        }
    }

    private var shouldPing: Bool {
        lock.withLock { started && !stopped }
    }

    func stop() {
        lock.withLock {
            stopped = true
            heartbeat?.cancel()
            heartbeat = nil
        }
    }

    func nextToolIndex() -> Int {
        lock.withLock {
            defer { toolIndex += 1 }
            return toolIndex
        }
    }
}
