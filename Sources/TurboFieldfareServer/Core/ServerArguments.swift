import Foundation
import TurboFieldfare

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let port: Int
    public let modelID: String
    public let maxContext: Int
    public let queueLimit: Int
    public let promptCacheMode: ServerPromptCacheMode
    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let verification: ModelIntegrityPolicy

    public static let usage = """
    usage: TurboFieldfareServer --model <completed .gturbo directory> [options]

      --model <dir>              Required model directory.
      --port <1...65535>         Loopback port (default 8080).
      --model-id <id>            API model identifier (default gemma-4-26b-a4b-it).
      --max-context <tokens>     4096, 8192, 16384, 32768, or 65536 (default 16384).
      --queue-limit <count>      Maximum queued requests (default 4).
      --prompt-cache-mode <off|single-prefix>
                                 Prompt KV reuse mode (default single-prefix).
      --expert-cache-slots <n>   Expert-cache slots: 8, 16, 24, 32, 48, 64, 80, 96, or 112
                                 (default 48). Costs about 100 MB per slot; a value
                                 the device cannot keep resident is rejected at load.
      --expert-cache-policy <s>  Expert-cache policy: lfu or lru (default lfu).
      --prefill on|off           Enable or disable chunked prompt prefill (default on).
                                 Chunked prefill requires 16 or more cache slots.
      --prefill-chunk-tokens <n> Prefill chunk size: 32, 64, 128, 256, 512, 1024, or 2048
                                 (default 2048). Wider chunks reuse each routed expert
                                 across more tokens, so the SSD moves far fewer bytes
                                 per token; they cost KV ring rows and prefill scratch,
                                 and a width the device cannot keep resident is
                                 rejected at load.
      --rdadvise <s>             Read-advice policy: off, default, bounded, or adaptive
                                 (default off).
      --verification <s>         Model integrity: full-sha256 or trusted-install
                                 (default full-sha256). trusted-install checks sizes
                                 against verified-install.json instead of rehashing
                                 every layer file on first touch.
      --help                     Show this help.
    """

    // Mirrors the CLI's runtime flags so both binaries accept the same options
    // with the same validation, instead of the server pinning production
    // defaults. RuntimeConfiguration traps on unsupported values, so every
    // bound is checked here before the initializer runs.
    public func resolvedRuntimeConfiguration(
        forceLogitsHead: Bool = true
    ) throws -> RuntimeConfiguration {
        guard RuntimeConfiguration.allowedExpertCacheSlots.contains(expertCacheSlots) else {
            throw ServerArgumentError.invalid(
                    "--expert-cache-slots must be one of "
                    + RuntimeConfiguration.allowedExpertCacheSlots
                        .map(String.init).joined(separator: ", "))
        }
        guard RuntimeConfiguration.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw ServerArgumentError.invalid(
                    "--prefill-chunk-tokens must be one of "
                    + RuntimeConfiguration.allowedPrefillChunkTokens
                        .map(String.init).joined(separator: ", "))
        }
        guard prefillPolicy == .off
                || expertCacheSlots >= RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
        else {
            throw ServerArgumentError.invalid(
                "--expert-cache-slots \(expertCacheSlots) requires --prefill off")
        }
        return RuntimeConfiguration(
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy,
            rdadvisePolicy: rdadvisePolicy,
            prefillEnabled: prefillPolicy == .chunked,
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: forceLogitsHead)
    }

    public static func parse(_ input: [String]) throws -> ServerArguments {
        var model: String?
        var port = 8080
        var modelID = "gemma-4-26b-a4b-it"
        var maxContext = 16_384
        var queueLimit = 4
        var promptCacheMode: ServerPromptCacheMode = .singlePrefix
        var expertCacheSlots = RuntimeConfiguration.production.expertCacheSlots
        var expertCachePolicy = RuntimeExpertCachePolicy.lfu
        var prefillPolicy = RuntimePrefillPolicy.chunked
        var prefillChunkTokens = RuntimeConfiguration.production.prefillChunkTokens
        var rdadvisePolicy = RDAdvicePolicyMode.off
        var verification = ModelIntegrityPolicy.fullSha256
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            guard index + 1 < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index + 1]
            index += 2
            switch flag {
            case "--model":
                model = value
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelID = value
            case "--max-context":
                guard let parsed = Int(value),
                      [4_096, 8_192, 16_384, 32_768, 65_536].contains(parsed) else {
                    throw ServerArgumentError.invalid("--max-context is not supported")
                }
                maxContext = parsed
            case "--queue-limit":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--queue-limit must be positive")
                }
                queueLimit = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off or single-prefix")
                }
                promptCacheMode = parsed
            case "--expert-cache-slots":
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) else {
                    throw ServerArgumentError.invalid(
                    "--expert-cache-slots must be one of "
                    + RuntimeConfiguration.allowedExpertCacheSlots
                        .map(String.init).joined(separator: ", "))
                }
                expertCacheSlots = parsed
            case "--expert-cache-policy":
                guard let parsed = RuntimeExpertCachePolicy(rawValue: value) else {
                    throw ServerArgumentError.invalid("--expert-cache-policy must be lfu or lru")
                }
                expertCachePolicy = parsed
            case "--prefill":
                switch value {
                case "on": prefillPolicy = .chunked
                case "off": prefillPolicy = .off
                default: throw ServerArgumentError.invalid("--prefill must be on or off")
                }
            case "--prefill-chunk-tokens":
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) else {
                    throw ServerArgumentError.invalid(
                    "--prefill-chunk-tokens must be one of "
                    + RuntimeConfiguration.allowedPrefillChunkTokens
                        .map(String.init).joined(separator: ", "))
                }
                prefillChunkTokens = parsed
            case "--rdadvise":
                guard let parsed = RDAdvicePolicyMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--rdadvise must be off, default, bounded, or adaptive")
                }
                rdadvisePolicy = parsed
            case "--verification":
                guard let parsed = ModelIntegrityPolicy(commandLineName: value) else {
                    throw ServerArgumentError.invalid(
                        "--verification must be full-sha256 or trusted-install")
                }
                verification = parsed
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        guard let model else { throw ServerArgumentError.invalid("--model is required") }
        return ServerArguments(model: model,
                               port: port,
                               modelID: modelID,
                               maxContext: maxContext,
                               queueLimit: queueLimit,
                               promptCacheMode: promptCacheMode,
                               expertCacheSlots: expertCacheSlots,
                               expertCachePolicy: expertCachePolicy,
                               prefillPolicy: prefillPolicy,
                               prefillChunkTokens: prefillChunkTokens,
                               rdadvisePolicy: rdadvisePolicy,
                               verification: verification)
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}
