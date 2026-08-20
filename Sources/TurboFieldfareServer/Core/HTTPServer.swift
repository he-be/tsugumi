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

    /// EP-4 `build_info`, which EP-4 also makes RSP-5's `system_fingerprint`.
    /// One value, defined once in `ServerBuildIdentity` and read from there by
    /// both — re-deriving it here would be a second answer to a question SPEC
    /// says has one.
    public static var buildInfo: String { ServerBuildIdentity.fingerprint }

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
    private let apiKeys: [String]
    private let corsPolicy: ServerCORSPolicy
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
                apiKeys: [String] = [],
                corsPolicy: ServerCORSPolicy = .disabled,
                group: MultiThreadedEventLoopGroup = .init(numberOfThreads: 1)) {
        self.group = group
        self.modelID = modelID
        self.readiness = ServerReadiness(backend)
        self.coordinator = ServerCoordinator(queueLimit: queueLimit)
        self.heartbeatInterval = heartbeatInterval
        self.imagePolicy = imagePolicy
        self.defaults = defaults
        self.properties = properties
        self.apiKeys = apiKeys
        self.corsPolicy = corsPolicy
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
        let apiKeys = self.apiKeys
        let corsPolicy = self.corsPolicy
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
                        apiKeys: apiKeys,
                        corsPolicy: corsPolicy,
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
    private let apiKeys: [String]
    private let corsPolicy: ServerCORSPolicy
    private let maximumBodyBytes: Int
    private let childChannels: ChildChannelRegistry
    private var head: HTTPRequestHead?
    /// FLAG-6: the `Origin` of the request being answered, or nil.
    private var requestOrigin: String?
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
         apiKeys: [String],
         corsPolicy: ServerCORSPolicy,
         childChannels: ChildChannelRegistry) {
        self.modelID = modelID
        self.readiness = readiness
        self.coordinator = coordinator
        self.heartbeatInterval = heartbeatInterval
        self.imagePolicy = imagePolicy
        self.defaults = defaults
        self.properties = properties
        self.apiKeys = apiKeys
        self.corsPolicy = corsPolicy
        self.maximumBodyBytes = imagePolicy.maximumBodyBytes
        self.childChannels = childChannels
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            self.head = head
            // FLAG-6: kept for the whole request, because the response head may
            // be written long after this — a completion answers from its own
            // task, and a stream writes its head later still.
            requestOrigin = head.headers.first(name: "origin")
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
                // ERR-2 / DEV-11: 413 has no type in the taxonomy, so an
                // oversized body is the same 400 as any other request this
                // server will not read.
                writeError(context, status: .badRequest,
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
        // FLAG-6 / LIF-6: preflight is answered ahead of the load gate and
        // ahead of the key, and identically for every path. A browser that
        // cannot preflight cannot read the 503 either, so stopping here would
        // keep it from even showing that the model is still loading; and it
        // never puts an `Authorization` on a preflight, so the key cannot
        // stand in front of it either (FLAG-5). Answering the same way for a
        // route that exists, one this server does not adopt and one it has
        // never heard of is what keeps the routing table unreadable from
        // preflight (EP-7).
        if head.method == .OPTIONS, corsPolicy.isEnabled {
            writePreflight(context)
            return
        }
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
        // FLAG-5: after the load gate and before the routing table, which is
        // the order the reference puts its two middlewares in
        // (`server-http.cpp:302`). Ahead of routing so a caller with no key
        // cannot map which paths exist by reading EP-7's 501s off the 404s.
        guard isAuthorized(head: head, path: path) else {
            writeError(context, status: .unauthorized, Self.invalidAPIKeyEnvelope)
            return
        }
        switch (head.method, path) {
        // EP-1: one handler, both spellings, no API key.
        case (.GET, "/health"), (.GET, "/v1/health"):
            writeJSON(context, status: .ok, object: ["status": "ok"])
        // EP-2 and EP-8: the reference implementation serves this under both
        // spellings, so a base URL without the `/v1` works the same way.
        case (.GET, "/v1/models"), (.GET, "/models"):
            let response = OpenAIModelList(
                object: "list",
                data: [.init(id: modelID,
                             object: "model",
                             created: 0,
                             ownedBy: "turbofieldfare")])
            writeCodable(context, status: .ok, response)
        // EP-4: the capability answer. Nothing here needs the model, but it is
        // behind the readiness gate with everything else (LIF-2).
        case (.GET, "/props"):
            writeCodable(context, status: .ok, Self.props(properties))
        // EP-3 and EP-8.
        case (.POST, "/v1/chat/completions"), (.POST, "/chat/completions"):
            guard head.headers.first(name: "content-type")?
                .lowercased().hasPrefix("application/json") == true else {
                writeError(context, status: .unsupportedMediaType,
                           OpenAIErrorEnvelope(message: "content-type must be application/json",
                                               code: "unsupported_media_type"))
                return
            }
            handleCompletion(body: body, context: context)
        // EP-7, ahead of the method check so that `POST /props` is answered as
        // the endpoint this server does not adopt rather than as the wrong verb
        // on the one it does.
        case _ where Self.isUnsupportedEndpoint(method: head.method, path: path):
            writeError(context, status: .notImplemented,
                       OpenAIErrorEnvelope(
                           message: "\(path) is not implemented by this server",
                           type: .notSupported,
                           code: "endpoint_not_supported"))
        case (_, "/health"), (_, "/v1/health"), (_, "/v1/models"), (_, "/models"),
             (_, "/props"), (_, "/v1/chat/completions"), (_, "/chat/completions"):
            writeError(context, status: .methodNotAllowed,
                       OpenAIErrorEnvelope(message: "method not allowed",
                                           code: "method_not_allowed"))
        default:
            // EP-7: only a path this server has never heard of is a 404, and
            // ERR-2 fixes the type that goes with the number.
            writeError(context, status: .notFound,
                       OpenAIErrorEnvelope(message: "route not found",
                                           type: .notFound,
                                           code: "not_found"))
        }
    }

    /// FLAG-6. The origin header pair, on every response this server writes.
    ///
    /// `Access-Control-Allow-Credentials` is never among them: this server's
    /// authentication is a header a client sets on purpose, not a cookie a
    /// browser attaches by itself, so there is no credentialed cross-origin
    /// request to allow.
    private func addCORSHeaders(to headers: inout HTTPHeaders) {
        guard let allowed = corsPolicy.allowOrigin(for: requestOrigin) else { return }
        headers.add(name: "access-control-allow-origin", value: allowed)
        if corsPolicy.variesByOrigin {
            headers.add(name: "vary", value: "Origin")
        }
    }

    /// FLAG-6's preflight answer: the same one for every path, with an empty
    /// body. What it advertises is this server's own verb set — there is no
    /// DELETE — and the three headers a client of ours actually sends.
    private func writePreflight(_ context: ChannelHandlerContext) {
        var headers = HTTPHeaders()
        addCORSHeaders(to: &headers)
        headers.add(name: "access-control-allow-methods", value: "GET, POST, OPTIONS")
        headers.add(name: "access-control-allow-headers",
                    value: "authorization, content-type, x-api-key")
        headers.add(name: "content-length", value: "0")
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            contextBox.value.write(self.wrapOutboundOut(.head(
                HTTPResponseHead(version: .http1_1, status: .ok, headers: headers))),
                promise: nil)
            contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
        }
    }

    /// FLAG-5's answer to a missing or wrong key.
    ///
    /// OpenAI's own shape for this case: 401 carrying `invalid_request_error`
    /// and `code: "invalid_api_key"`, which outranks the reference for `/v1/*`
    /// wire format (SPEC §0). The reference says `authentication_error` with
    /// the number in `code`, and DEV-1 has already ruled the number out.
    /// ERR-2 gives three statuses to this type already (400, 405, 415) and
    /// tells them apart by `code`; this is the fourth.
    private static let invalidAPIKeyEnvelope = OpenAIErrorEnvelope(
        message: "Invalid API Key",
        type: .invalidRequest,
        param: nil,
        code: "invalid_api_key")

    /// FLAG-5: the endpoints a client needs before it can authenticate
    /// anything. EP-1 names the two health spellings; the reference's public
    /// set at the pin adds both spellings of the model list
    /// (`server-http.cpp:196`), which is how a client learns which model it is
    /// talking to. Everything that reaches the model is behind the key.
    private static let unauthenticatedPaths: Set<String> = [
        "/health", "/v1/health", "/models", "/v1/models",
    ]

    private func isAuthorized(head: HTTPRequestHead, path: String) -> Bool {
        // No key configured is no authentication at all: FLAG-5 keeps the
        // 127.0.0.1 bind as the whole defence until an operator asks for more.
        if apiKeys.isEmpty { return true }
        if Self.unauthenticatedPaths.contains(path) { return true }
        guard let presented = Self.presentedKey(head.headers) else { return false }
        return apiKeys.contains(presented)
    }

    /// The three spellings the reference accepts: `Authorization` with or
    /// without the `Bearer ` prefix, and Anthropic's `X-Api-Key` as a fallback.
    private static func presentedKey(_ headers: HTTPHeaders) -> String? {
        let raw = headers.first(name: "authorization")
            ?? headers.first(name: "x-api-key")
        guard let raw else { return nil }
        let prefix = "Bearer "
        let key = raw.hasPrefix(prefix) ? String(raw.dropFirst(prefix.count)) : raw
        return key.isEmpty ? nil : key
    }

    /// SPEC §3's list of known paths this server does not adopt (DEV-7).
    ///
    /// The list is written by path, not by verb: a client that finds one of
    /// these has the right idea of what a llama.cpp-shaped server offers and
    /// the wrong idea of what this one does, whichever method it used.
    private static let unsupportedPaths: Set<String> = [
        "/v1/embeddings", "/embedding", "/reranking", "/rerank", "/infill",
        "/v1/responses", "/v1/messages", "/v1/chat/completions/control",
        "/lora-adapters", "/v1/completions", "/completion",
    ]

    private static func isUnsupportedEndpoint(method: HTTPMethod, path: String) -> Bool {
        if unsupportedPaths.contains(path) { return true }
        // `POST /props` is the reference implementation's write side; the read
        // side is EP-4 and is answered above.
        if path == "/props", method != .GET { return true }
        // `/slots/{id}?action=…`. `GET /slots` itself is EP-6 and simply does
        // not exist yet, so it stays a 404 until it does.
        if path.hasPrefix("/slots/") { return true }
        return false
    }

    /// EP-4's body.
    ///
    /// `default_generation_settings` is read off `ChatRequestSchema` — the same
    /// table the request parser applies — because SPEC §4 makes `/props` the
    /// truth about the defaults, and a second hand-written copy is a second
    /// thing to keep in step.
    private static func props(_ properties: ServerProperties) -> JSONValue {
        var settings: [String: JSONValue] = [:]
        for field in ChatRequestSchema.fields {
            guard let defaultValue = field.defaultValue else { continue }
            settings[field.name] = defaultValue
        }
        settings["n_ctx"] = .integer(Int64(properties.contextLength))
        return .object([
            "default_generation_settings": .object(settings),
            "total_slots": .integer(Int64(ServerProperties.totalSlots)),
            "model_path": .string(properties.modelPath),
            "chat_template": .string(properties.chatTemplate),
            "modalities": .object(["vision": .bool(true)]),
            "build_info": .string(ServerProperties.buildInfo),
        ])
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
            // RSP-3 `timings_per_token`. Allocated only for a request that asked
            // for it: without one the backend takes the ordinary path and no
            // chunk carries running timings. It is read synchronously from
            // inside the generation callback, which is the only place the route
            // can learn anything about a generation it is not awaiting.
            let monitor = request.timingsPerToken ? ServerTimingsMonitor() : nil
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
                               finishReason: nil,
                               // Nil here by construction: the role chunk is
                               // written before the first token is drawn.
                               timings: monitor?.current))
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
                            return try await backend.generate(
                                prepared, monitor: monitor
                            ) { event in
                                guard request.stream else { return }
                                switch event {
                                case .content(let text):
                                    self.writeStreamChunk(
                                        contextBox.value,
                                        self.chunk(id: responseID, created: created,
                                                   model: responseModel,
                                                   delta: ["content": text],
                                                   finishReason: nil,
                                                   timings: monitor?.current))
                                case .reasoning(let text):
                                    self.writeStreamChunk(
                                        contextBox.value,
                                        self.chunk(id: responseID, created: created,
                                                   model: responseModel,
                                                   delta: ["reasoning_content": text],
                                                   finishReason: nil,
                                                   timings: monitor?.current))
                                case .toolCall(let call):
                                    self.writeToolCall(contextBox.value,
                                                       id: responseID,
                                                       created: created,
                                                       model: responseModel,
                                                       toolIndex: streamState.nextToolIndex(),
                                                       call: call,
                                                       timings: monitor?.current)
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
        var object: [String: Any] = [
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
            // RSP-5, in the place OpenAI puts it: alongside `model`, on the
            // completion object itself. EP-4 makes it the same string `/props`
            // answers with as `build_info`.
            "system_fingerprint": ServerProperties.buildInfo,
        ]
        // RSP-3: what this completion cost. Absent only for a backend that
        // measures nothing, which is the stubs and not a loaded model — an
        // invented `timings` would be worse than a missing one, because the
        // client's context arithmetic is built on it.
        if let timings = completion.timings {
            object["timings"] = timings.jsonObject
        }
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
        addCORSHeaders(to: &headers)
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
                               call: ParsedToolCall,
                               timings: ServerTimings? = nil) {
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
                      finishReason: nil,
                      timings: timings))
        }
    }

    private func finishStream(_ context: ChannelHandlerContext,
                              id: String,
                              created: Int,
                              responseModel: String,
                              completion: ServerCompletion,
                              includeUsage: Bool) {
        // RSP-3 puts the completion's timings on the *final* chunk, and the
        // final chunk is the usage one when the request asked for it. The
        // reference writes them onto `deltas.back()` for the same reason
        // (`server-task.cpp` at the pin) rather than onto a named chunk.
        writeStreamChunk(
            context,
            chunk(id: id, created: created, model: responseModel,
                  delta: [:],
                  finishReason: completion.finishReason,
                  timings: includeUsage ? nil : completion.timings))
        if includeUsage {
            var usageChunk: [String: Any] = [
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": responseModel,
                "choices": [],
                "usage": usageObject(completion.usage),
                "system_fingerprint": ServerProperties.buildInfo,
            ]
            if let timings = completion.timings {
                usageChunk["timings"] = timings.jsonObject
            }
            writeStreamChunk(context, usageChunk)
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
                       finishReason: String?,
                       timings: ServerTimings? = nil) -> [String: Any] {
        let encodedReason: Any = finishReason.map { $0 as Any } ?? NSNull()
        var object: [String: Any] = [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [[
                "index": 0,
                "delta": delta,
                "finish_reason": encodedReason,
            ]],
            // RSP-5 rides every chunk, not only the last one: the reference
            // puts it on each `chat.completion.chunk` it builds
            // (`server-task.cpp`'s `add_delta` at the pin), and a client that
            // only ever sees the stream would otherwise never read it.
            "system_fingerprint": ServerProperties.buildInfo,
        ]
        // RSP-3. Nil on every chunk but the last, unless the request asked for
        // `timings_per_token` — in which case this is the monitor's reading at
        // the moment the token that produced this chunk landed.
        if let timings {
            object["timings"] = timings.jsonObject
        }
        return object
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
            self.addCORSHeaders(to: &headers)
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
