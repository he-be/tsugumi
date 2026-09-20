import Foundation

/// What a llama-server model directory's `manifest.json` names (`docs/qwen38-27b/10` §2): the server binary, the GGUF
/// and the arguments the measured runs used. Nothing here is shipped — both paths are this machine's.
///
///     { "arch": { "family": "qwen3_8_dense_llamacpp" },
///       "llama_server": "~/LLM/prism-llamacpp/src-b10709/build/bin/llama-server",
///       "gguf": "Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf",
///       "server_args": ["-ngl", "99", "-fa", "on", "-np", "1", "--jinja", …] }
///
/// `-m`, `-c`, `--host` and `--port` are the client's: the context is the app's setting and the port is a free one.
public struct LlamaServerModelDirectory: Equatable, Sendable {
    public let server: URL
    public let gguf: URL
    public let serverArguments: [String]

    /// Arguments the client sets itself; a manifest that names one would start a server the client cannot reach or
    /// whose context is not the one the app believes it loaded.
    static let reservedArguments: Set<String> = [
        "-m", "--model", "-c", "--ctx-size", "--host", "--port",
    ]

    public init(modelDirectory: URL) throws {
        struct Manifest: Decodable {
            let llama_server: String
            let gguf: String
            let server_args: [String]?
        }
        let manifestURL = modelDirectory.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL) else {
            throw AppInferenceError.modelNotFound(manifestURL.path)
        }
        let manifest: Manifest
        do {
            manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw AppInferenceError.modelLoadFailed("\(manifestURL.path): \(error)")
        }
        server = Self.resolve(manifest.llama_server, in: modelDirectory)
        gguf = Self.resolve(manifest.gguf, in: modelDirectory)
        serverArguments = manifest.server_args ?? []
        if let reserved = serverArguments.first(where: Self.reservedArguments.contains) {
            throw AppInferenceError.modelLoadFailed(
                "\(manifestURL.path): server_args names \(reserved), which the app sets itself")
        }
        guard FileManager.default.isExecutableFile(atPath: server.path) else {
            throw AppInferenceError.modelLoadFailed("no llama-server at \(server.path)")
        }
        guard FileManager.default.fileExists(atPath: gguf.path) else {
            throw AppInferenceError.modelNotFound(gguf.path)
        }
    }

    private static func resolve(_ path: String, in directory: URL) -> URL {
        let expanded = NSString(string: path).expandingTildeInPath
        if expanded.hasPrefix("/") { return URL(fileURLWithPath: expanded).standardizedFileURL }
        return directory.appendingPathComponent(expanded).standardizedFileURL
    }

    func arguments(contextTokens: Int, port: Int) -> [String] {
        ["-m", gguf.path] + serverArguments
            + ["-c", String(contextTokens), "--host", "127.0.0.1", "--port", String(port)]
    }
}

/// A model that runs in a `llama-server` this app starts and stops (`AppModelKind.runsOnLlamaServer`): the load is
/// the server's start, `unload` is its end, and the rounds go through `RemoteInferenceClient` at 127.0.0.1.
///
/// One server at a time. A context change is a restart (`-c` is a start argument). A server the app left behind —
/// the app was killed, so `unload` never ran — is found by the pid file at the next start and stopped, since two
/// copies of a 10 GB model do not fit this machine.
public final class LlamaServerInferenceClient: AppModelLifecycleClient, AppInferenceRuntimeReporting,
                                                 @unchecked Sendable {
    private struct Running {
        let process: Process
        let remote: RemoteInferenceClient
        let modelDirectory: URL
        let contextTokens: Int
    }

    private let lock = NSLock()
    private var running: Running?
    private let stateDirectory: URL
    private let healthTimeoutSeconds: Double

    public var loadedRuntimeOwnBytes: UInt64? { nil }

    /// `stateDirectory` holds the pid file and the server's log (`llama-server.pid`, `llama-server.log`).
    public init(stateDirectory: URL = LlamaServerInferenceClient.defaultStateDirectory,
                healthTimeoutSeconds: Double = 300) {
        self.stateDirectory = stateDirectory
        self.healthTimeoutSeconds = healthTimeoutSeconds
    }

    /// `~/Library/Application Support/Tsugumi/llama-server`.
    public static var defaultStateDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return base.appendingPathComponent("Tsugumi/llama-server", isDirectory: true)
    }

    public var logURL: URL { stateDirectory.appendingPathComponent("llama-server.log") }
    private var pidURL: URL { stateDirectory.appendingPathComponent("llama-server.pid") }

    /// The running server's address and pid, for a check that reads `/slots` or asks whether the process is gone.
    public var endpoint: URL? { current()?.remote.endpoint }
    public var serverProcessIdentifier: Int32? { current()?.process.processIdentifier }

    // MARK: Lifecycle

    public func ensureLoaded(modelDirectory: URL, maxContextTokens: Int, options: AppRuntimeOptions,
                             forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        let directory = modelDirectory.standardizedFileURL
        if let current = current(), current.process.isRunning,
           current.modelDirectory == directory, current.contextTokens == maxContextTokens {
            onState(.ready(modelDirectory: modelDirectory, loadSeconds: 0))
            return
        }
        let start = Date()
        do {
            onState(.loading(.validatingDirectory))
            let model = try LlamaServerModelDirectory(modelDirectory: directory)
            stopServer()
            try FileManager.default.createDirectory(at: stateDirectory, withIntermediateDirectories: true)
            stopLeftoverServer()

            onState(.loading(.preparingRunner))
            let port = try Self.freePort()
            let process = Process()
            process.executableURL = model.server
            process.arguments = model.arguments(contextTokens: maxContextTokens, port: port)
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
            let log = try FileHandle(forWritingTo: logURL)
            process.standardOutput = log
            process.standardError = log
            process.standardInput = FileHandle.nullDevice
            try process.run()
            try? Data("\(process.processIdentifier)\n".utf8).write(to: pidURL)

            let endpoint = URL(string: "http://127.0.0.1:\(port)")!
            let remote = RemoteInferenceClient(endpoint: endpoint, modelID: directory.lastPathComponent,
                                               dialect: .init(kind: .bonsai27b), routing: .direct)
            do {
                try await waitUntilHealthy(endpoint: endpoint, process: process)
                try await remote.prepare()
            } catch {
                Self.stop(process)
                try? FileManager.default.removeItem(at: pidURL)
                throw error
            }
            setRunning(Running(process: process, remote: remote, modelDirectory: directory,
                               contextTokens: maxContextTokens))
            onState(.ready(modelDirectory: modelDirectory, loadSeconds: Date().timeIntervalSince(start)))
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as AppInferenceError {
            onState(.failed(error))
            throw error
        } catch {
            let appError = AppInferenceError.modelLoadFailed("\(error)")
            onState(.failed(appError))
            throw appError
        }
    }

    public func unload() async { stopServer() }

    /// Stops the server now, without awaiting: what the app calls on its way out.
    public func stopServer() {
        lock.lock()
        let stopping = running
        running = nil
        lock.unlock()
        guard let stopping else { return }
        stopping.remote.cancel()
        Self.stop(stopping.process)
        try? FileManager.default.removeItem(at: pidURL)
    }

    public func cancel() { current()?.remote.cancel() }

    public func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        guard let current = current(), current.process.isRunning else {
            return AsyncThrowingStream { continuation in
                continuation.yield(.failed(.modelNotLoaded, partial: nil))
                continuation.finish()
            }
        }
        return current.remote.generate(request)
    }

    private func current() -> Running? {
        lock.lock(); defer { lock.unlock() }
        return running
    }

    private func setRunning(_ started: Running) {
        lock.lock(); defer { lock.unlock() }
        running = started
    }

    // MARK: The server process

    private func waitUntilHealthy(endpoint: URL, process: Process) async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 2
        let session = URLSession(configuration: configuration)
        let health = endpoint.appendingPathComponent("health")
        let deadline = Date().addingTimeInterval(healthTimeoutSeconds)
        while Date() < deadline {
            try Task.checkCancellation()
            guard process.isRunning else {
                throw AppInferenceError.modelLoadFailed(
                    "llama-server exited with \(process.terminationStatus) while loading: \(logTail())")
            }
            // 503 while the weights load, 200 once the slot is up.
            if let (_, response) = try? await session.data(from: health),
               (response as? HTTPURLResponse)?.statusCode == 200 {
                return
            }
            try await Task.sleep(nanoseconds: 500_000_000)
        }
        throw AppInferenceError.modelLoadFailed(
            "llama-server was not healthy after \(Int(healthTimeoutSeconds)) s: \(logTail())")
    }

    private func logTail() -> String {
        guard let data = try? Data(contentsOf: logURL) else { return "no log at \(logURL.path)" }
        let lines = String(decoding: data.suffix(2_000), as: UTF8.self).split(separator: "\n").suffix(4)
        return lines.joined(separator: " / ")
    }

    /// The server the pid file names, if it is still a llama-server: the pid may have been reused since.
    private func stopLeftoverServer() {
        defer { try? FileManager.default.removeItem(at: pidURL) }
        guard let text = try? String(contentsOf: pidURL, encoding: .utf8),
              let pid = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)), pid > 1,
              Self.commandName(of: pid)?.hasSuffix("llama-server") == true else { return }
        Self.stop(pid: pid)
    }

    private static func stop(_ process: Process) {
        guard process.isRunning else { return }
        stop(pid: process.processIdentifier)
        process.waitUntilExit()
    }

    /// SIGTERM to the process and whatever it started, then SIGKILL to what is left after 5 seconds.
    private static func stop(pid: Int32) {
        let family = descendants(of: pid) + [pid]
        for member in family { kill(member, SIGTERM) }
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline, family.contains(where: { kill($0, 0) == 0 && !isZombie($0) }) {
            usleep(100_000)
        }
        for member in family where kill(member, 0) == 0 { kill(member, SIGKILL) }
    }

    private static func descendants(of pid: Int32) -> [Int32] {
        let children = run("/usr/bin/pgrep", ["-P", String(pid)])
            .split(separator: "\n").compactMap { Int32($0) }
        return children.flatMap { descendants(of: $0) } + children
    }

    private static func commandName(of pid: Int32) -> String? {
        let name = run("/bin/ps", ["-p", String(pid), "-o", "comm="])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return name.isEmpty ? nil : name
    }

    private static func isZombie(_ pid: Int32) -> Bool {
        run("/bin/ps", ["-p", String(pid), "-o", "stat="]).hasPrefix("Z")
    }

    private static func run(_ tool: String, _ arguments: [String]) -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: tool)
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        guard (try? process.run()) != nil else { return "" }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return String(decoding: data, as: UTF8.self)
    }

    /// A port nothing listens on now: bound at 0 and released, for the server to take.
    static func freePort() throws -> Int {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw AppInferenceError.modelLoadFailed("no socket for a free port") }
        defer { close(descriptor) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        address.sin_port = 0
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { generic -> Bool in
                bind(descriptor, generic, length) == 0 && getsockname(descriptor, generic, &length) == 0
            }
        }
        guard bound else { throw AppInferenceError.modelLoadFailed("no free port on 127.0.0.1") }
        return Int(UInt16(bigEndian: address.sin_port))
    }
}
