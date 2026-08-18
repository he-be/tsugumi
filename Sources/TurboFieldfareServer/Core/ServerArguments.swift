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
    public let draftBlockSize: Int
    public let imagePolicy: ServerImagePolicy

    public static let usage = """
    usage: TurboFieldfareServer --model <completed .gturbo directory> [options]

      --model <dir>              Required model directory.
      --port <1...65535>         Loopback port (default 8080).
      --model-id <id>            API model identifier (default gemma-4-26b-a4b-it).
      --max-context <tokens>     4096, 8192, 16384, 32768, 65536, or 131072
                                 (default 16384). Only the full-attention layers
                                 grow with the context, but they grow linearly:
                                 the KV cache measures 1.97 GB at 65536 and
                                 3.31 GB at 131072, which has to come out of the
                                 same Metal working set as the expert cache, so
                                 a long context wants fewer --expert-cache-slots.
                                 A combination the device cannot keep resident is
                                 rejected at load with the arithmetic.
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
      --draft-block-size <n>     MTP speculative block width: 0 (off, the default)
                                 or 2...8. Needs a model installed with the
                                 drafter section and --prefill on. Accepted
                                 tokens are the ones the target itself drew, so
                                 speculation moves the wall clock and not the
                                 text; a request that asks for a repetition
                                 penalty other than 1.0 falls back to plain
                                 decode for that request alone.
      --image-tokens <n>         Soft-token budget per image: 70, 140, or 280
                                 (default 280). An upper bound, not the count — the
                                 count follows the image's aspect ratio. Images arrive
                                 as image_url content parts holding a data: URI;
                                 http(s) URLs are refused, never fetched. Requires a
                                 model installed with a vision tower and --prefill on.
      --max-images <n>           Images accepted per request (default 4).
      --max-image-bytes <n>      Decoded bytes accepted per image (default 8388608).
                                 Raises the request-body ceiling to match.
      --max-image-pixels <n>     Pixels accepted per image (default 50000000), read
                                 from the container header before any decode.
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
        var draftBlockSize = 0
        var imageTokens = ServerImagePolicy.default.maxSoftTokens
        var maxImages = ServerImagePolicy.default.maxImagesPerRequest
        var maxImageBytes = ServerImagePolicy.default.maxImageBytes
        var maxImagePixels = ServerImagePolicy.default.maxImagePixels
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
                      [4_096, 8_192, 16_384, 32_768, 65_536, 131_072].contains(parsed) else {
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
            case "--draft-block-size":
                guard let parsed = Int(value),
                      parsed == 0
                        || (2...SpeculativeBlock.maxTokens).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--draft-block-size must be 0 or 2...\(SpeculativeBlock.maxTokens)")
                }
                draftBlockSize = parsed
            case "--image-tokens":
                guard let parsed = Int(value),
                      VisionPreprocessorConfig.supportedSoftTokens.contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--image-tokens must be one of "
                        + VisionPreprocessorConfig.supportedSoftTokens
                            .map(String.init).joined(separator: ", "))
                }
                imageTokens = parsed
            case "--max-images":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--max-images must be positive")
                }
                maxImages = parsed
            case "--max-image-bytes":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--max-image-bytes must be positive")
                }
                maxImageBytes = parsed
            case "--max-image-pixels":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--max-image-pixels must be positive")
                }
                maxImagePixels = parsed
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        guard let model else { throw ServerArgumentError.invalid("--model is required") }
        // The speculative block runs through the chunked prefill path (D6), so
        // the two flags cannot disagree.
        guard draftBlockSize == 0 || prefillPolicy == .chunked else {
            throw ServerArgumentError.invalid(
                "--draft-block-size \(draftBlockSize) requires --prefill on")
        }
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
                               verification: verification,
                               draftBlockSize: draftBlockSize,
                               imagePolicy: ServerImagePolicy(
                                   maxSoftTokens: imageTokens,
                                   maxImagesPerRequest: maxImages,
                                   maxImageBytes: maxImageBytes,
                                   maxImagePixels: maxImagePixels))
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
