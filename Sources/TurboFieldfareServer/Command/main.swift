import Darwin
import Foundation
import TurboFieldfare
import TurboFieldfareServerCore

let arguments: ServerArguments
let runtimeConfiguration: RuntimeConfiguration
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
    // Resolved here so an unusable flag combination exits with usage instead of
    // failing after the model has started loading.
    runtimeConfiguration = try arguments.resolvedRuntimeConfiguration()
} catch ServerArgumentError.help {
    print(ServerArguments.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerArguments.usage)\n".utf8))
    exit(2)
}

do {
    let signals = ServerTerminationSignals()
    let modelURL = URL(fileURLWithPath: arguments.model).standardizedFileURL
    let server = TurboFieldfareHTTPServer(
        modelID: arguments.modelID,
        queueLimit: arguments.queueLimit,
        backend: nil,
        imagePolicy: arguments.imagePolicy,
        defaults: ChatRequestDefaults(thinking: arguments.thinkingPolicy),
        // EP-4. All of it is known before the model is, which is what lets
        // `/props` be answered by a server that is still loading one.
        properties: ServerProperties(
            modelPath: modelURL.path,
            contextLength: arguments.maxContext,
            chatTemplate: try ServerChatTemplate.jinja()),
        // FLAG-5. Empty unless the operator asked for a key, in which case the
        // 127.0.0.1 bind stays the whole defence exactly as it is today.
        apiKeys: arguments.apiKeys,
        // FLAG-6. Disabled unless the operator named origins, so nothing about
        // a browser's reach changes for anyone who passes neither flag.
        corsPolicy: arguments.corsPolicy)
    // LIF-1: the port opens before the model does, and everything answers 503
    // `unavailable_error` until the load lands (LIF-2). The load takes tens of
    // seconds; a client that connects during it is told the server is coming
    // up, which a refused connection cannot say.
    _ = try await server.start(port: arguments.port)
    print("TurboFieldfareServer listening at http://127.0.0.1:\(arguments.port) loading model=\(arguments.modelID)")
    // stdout はパイプに繋がれると全バッファリングになる (`Scripts/demo/serve.py`
    // は子プロセスの stdout を読む)。この行は「もう繋がる」の合図で、
    // プロセスが終わるまで見えないのでは意味がないので、ここで押し出す。
    fflush(stdout)

    // LIF-5: SIGINT / SIGTERM ends the process, and it has to do so during the
    // load as well — which is only reachable now that the listener is up
    // first. The load is not cancellable, so termination does not wait for it.
    let terminator: Task<Void, Never> = Task {
        _ = await signals.wait()
        // A second signal kills outright, before the shutdown that a client
        // (or a model still loading) can make the operator wait on.
        signals.restoreDefaultDisposition()
        try? await server.shutdown()
        await signals.cancel()
        exit(0)
    }

    let backend = try await ServerModelSession.load(
        modelDirectory: modelURL,
        maxContext: arguments.maxContext,
        runtimeConfiguration: runtimeConfiguration,
        integrityPolicy: arguments.verification,
        draftBlockSize: arguments.draftBlockSize,
        imagePolicy: arguments.imagePolicy)
    // LIF-3: from here `/health` is 200 and the endpoints answer for real.
    await server.modelDidLoad(backend)
    // `expert_io`: どちらの腕で回っているか。既定は mmap (docs/mtp/52 §5a)。
    // `TF_EXPERT_MMAP=0` で pread に戻る。
    let expertIO = MmapExpertMapping.isEnabled ? "mmap" : "pread"
    print("TurboFieldfareServer ready at http://127.0.0.1:\(arguments.port) model=\(arguments.modelID) context=\(arguments.maxContext) slots=\(runtimeConfiguration.expertCacheSlots) expert_io=\(expertIO) mtp=\(arguments.draftBlockSize) thinking=\(arguments.thinkingPolicy.rawValue)")
    fflush(stdout)

    await terminator.value
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
