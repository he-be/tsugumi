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
    let backend = try await ServerModelSession.load(
        modelDirectory: modelURL,
        maxContext: arguments.maxContext,
        runtimeConfiguration: runtimeConfiguration,
        integrityPolicy: arguments.verification,
        draftBlockSize: arguments.draftBlockSize,
        imagePolicy: arguments.imagePolicy)
    let server = TurboFieldfareHTTPServer(
        modelID: arguments.modelID,
        queueLimit: arguments.queueLimit,
        backend: backend,
        imagePolicy: arguments.imagePolicy,
        defaults: ChatRequestDefaults(thinking: arguments.thinkingPolicy))
    _ = try await server.start(port: arguments.port)
    // `expert_io`: どちらの腕で回っているか。既定は mmap (docs/mtp/52 §5a)。
    // `TF_EXPERT_MMAP=0` で pread に戻る。
    let expertIO = MmapExpertMapping.isEnabled ? "mmap" : "pread"
    print("TurboFieldfareServer ready at http://127.0.0.1:\(arguments.port) model=\(arguments.modelID) context=\(arguments.maxContext) slots=\(runtimeConfiguration.expertCacheSlots) expert_io=\(expertIO) mtp=\(arguments.draftBlockSize) thinking=\(arguments.thinkingPolicy.rawValue)")
    // stdout はパイプに繋がれると全バッファリングになる (`Scripts/demo/serve.py`
    // は子プロセスの stdout を読む)。この 1 行は「もう受け付けている」の合図で、
    // プロセスが終わるまで見えないのでは意味がないので、ここで押し出す。
    fflush(stdout)

    _ = await signals.wait()
    try await server.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
