import Darwin
import Foundation
import TurboFieldfare
import TurboFieldfareServerCore

let arguments: ServerArguments
let runtimeConfiguration: RuntimeConfiguration
/// Which family the install declares decides the tokenizer, the runner and the
/// backend, and that decision has to be made before any of them exists
/// (`Model.declaredFamily` reads `manifest.json` and nothing else). It is made
/// here, above the listener, because `/props` has to answer with this family's
/// chat template while the model is still loading (LIF-2).
let isOrnith: Bool
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
    isOrnith = Model.declaredFamily(
        at: URL(fileURLWithPath: arguments.model).standardizedFileURL) == "qwen3_5_moe"
    // Resolved here so an unusable flag combination exits with usage instead of
    // failing after the model has started loading.
    //
    // GEN-7 on Ornith is not the logits head: that family's head writes no
    // logit anywhere, and a constrained draw is re-folded with a mask instead
    // (`docs/qwen35moe/25-CLI-TOOLS.md` §2). Asking for the logits head would
    // describe a runner this family does not build.
    runtimeConfiguration = try arguments.resolvedRuntimeConfiguration(
        forceLogitsHead: !isOrnith)
    // Same reason the runtime configuration is resolved here: a flag this
    // family cannot honour should exit with usage, not after a listener is up
    // and a tokenizer has been read.
    if isOrnith { try QwenServerSession.validateFlags(draftBlockSize: arguments.draftBlockSize) }
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
        // RSN-1: the on/off axis of `--reasoning-budget`. Only `0` closes the
        // channel; the default `-1` is unlimited, not off.
        defaults: ChatRequestDefaults(thinking: arguments.thinkingPolicy),
        // EP-4. All of it is known before the model is, which is what lets
        // `/props` be answered by a server that is still loading one.
        properties: ServerProperties(
            modelPath: modelURL.path,
            contextLength: arguments.maxContext,
            // Ornith ships the template that wrote every assistant turn the
            // model has seen, so the server renders *that* one and EP-4 says
            // so. Gemma's is the repo-owned variant of DEV-12, because
            // upstream has none.
            chatTemplate: isOrnith
                ? try QwenTokenizer.chatTemplateJinja(forModelDirectory: modelURL)
                : try ServerChatTemplate.jinja(),
            // Phase 9: Ornith's tower is not written, and the backend answers
            // an image with a 400. EP-4 says so before a client sends one.
            supportsVision: !isOrnith),
        // FLAG-5. Empty unless the operator asked for a key, in which case the
        // 127.0.0.1 bind stays the whole defence exactly as it is today.
        apiKeys: arguments.apiKeys,
        // FLAG-6. Disabled unless the operator named origins, so nothing about
        // a browser's reach changes for anyone who passes neither flag.
        corsPolicy: arguments.corsPolicy,
        // EP-6. `--slots` / `--no-slots` and `--metrics`, with the reference's
        // defaults: the slot state is readable, the Prometheus exposition is
        // not until it is asked for.
        slotsEndpointEnabled: arguments.slotsEndpointEnabled,
        metricsEndpointEnabled: arguments.metricsEndpointEnabled)
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

    let backend: any ServerInferenceBackend = if isOrnith {
        // Phase 8. No image policy and no draft block size: this backend
        // refuses both rather than carrying a parameter it cannot honour
        // (`QwenServerSession`).
        try await QwenServerSession.load(
            modelDirectory: modelURL,
            maxContext: arguments.maxContext,
            runtimeConfiguration: runtimeConfiguration,
            integrityPolicy: arguments.verification,
            draftBlockSize: arguments.draftBlockSize,
            reasoningBudget: arguments.reasoningBudget,
            reasoningFormat: arguments.reasoningFormat)
    } else {
        try await ServerModelSession.load(
            modelDirectory: modelURL,
            maxContext: arguments.maxContext,
            runtimeConfiguration: runtimeConfiguration,
            integrityPolicy: arguments.verification,
            draftBlockSize: arguments.draftBlockSize,
            imagePolicy: arguments.imagePolicy,
            // RSN-4 / RSN-3: the budget a request falls back to, and where the
            // thought channel comes back in the response.
            reasoningBudget: arguments.reasoningBudget,
            reasoningFormat: arguments.reasoningFormat)
    }
    // LIF-3: from here `/health` is 200 and the endpoints answer for real.
    await server.modelDidLoad(backend)
    // `expert_io`: どちらの腕で回っているか。既定は mmap (docs/mtp/52 §5a)。
    // `TF_EXPERT_MMAP=0` で pread に戻る。
    let expertIO = MmapExpertMapping.isEnabled ? "mmap" : "pread"
    print("TurboFieldfareServer ready at http://127.0.0.1:\(arguments.port) model=\(arguments.modelID) family=\(isOrnith ? "qwen3_5_moe" : "gemma4") context=\(arguments.maxContext) slots=\(runtimeConfiguration.expertCacheSlots) expert_io=\(expertIO) mtp=\(arguments.draftBlockSize) reasoning_budget=\(arguments.reasoningBudget) reasoning_format=\(arguments.reasoningFormat.rawValue)")
    fflush(stdout)

    await terminator.value
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
