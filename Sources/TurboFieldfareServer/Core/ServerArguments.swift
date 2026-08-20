import Foundation
import TurboFieldfare

/// The process default for the chat template's thought channel.
///
/// A request may override it per call (`chat_template_kwargs.enable_thinking`
/// or `reasoning_effort`); this is what applies when it says nothing.
public enum ServerThinkingPolicy: String, Equatable, Sendable {
    case off
    case on

    public var isEnabled: Bool { self == .on }
}

/// FLAG-6. What `--cors-origins` was set to.
public enum ServerCORSPolicy: Equatable, Sendable {
    /// No flag. No CORS header goes out on any response and `OPTIONS` is not a
    /// preflight — the 127.0.0.1 bind is the whole defence, as it is today.
    /// The reference defaults to `*` with credentials, which would open a
    /// loopback server to any page the browser has open (DEV-20).
    case disabled
    /// `*`. Answered as `*`, which does not depend on the request's `Origin`
    /// and so needs no `Vary`.
    case any
    /// A list. The request's `Origin` is matched against it and only the one
    /// that matched comes back — returning the list verbatim, as the reference
    /// does, is a value no browser accepts (DEV-20).
    case origins([String])

    /// The `Access-Control-Allow-Origin` for a request that presented this
    /// `Origin`, or nil for no header at all. A request that matches nothing is
    /// still answered — CORS is enforced by the browser, and refusing here
    /// would break every non-browser client that happens to send an `Origin`.
    public func allowOrigin(for requestOrigin: String?) -> String? {
        switch self {
        case .disabled:
            nil
        case .any:
            "*"
        case .origins(let allowed):
            requestOrigin.flatMap { allowed.contains($0) ? $0 : nil }
        }
    }

    /// Whether the answer depends on the request's `Origin`. `*` does not, so
    /// it needs no `Vary`.
    public var variesByOrigin: Bool {
        if case .origins = self { return true }
        return false
    }

    /// Whether `OPTIONS` is a preflight at all. Without the flag it is not:
    /// answering a preflight with no `Access-Control-Allow-Origin` tells a
    /// browser nothing it can use, so the verb stays what it is today.
    public var isEnabled: Bool { self != .disabled }
}

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let port: Int
    public let modelID: String
    public let maxContext: Int
    public let queueLimit: Int
    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let verification: ModelIntegrityPolicy
    public let draftBlockSize: Int
    public let imagePolicy: ServerImagePolicy
    /// RSN-1 / FLAG-1. `--reasoning-budget N`: `-1` is unlimited and the
    /// default, `0` closes the thought channel, `N > 0` caps it.
    public let reasoningBudget: Int
    /// RSN-3 / FLAG-1. `--reasoning-format auto|none`.
    public let reasoningFormat: ReasoningFormat

    /// RSN-1: the on/off axis of `--reasoning-budget`, in the shape the request
    /// parser takes its process default in (`ChatRequestDefaults`). Only a
    /// budget of zero closes the channel; every other value, the default `-1`
    /// included, leaves it open for a request to use or to override.
    public var thinkingPolicy: ServerThinkingPolicy {
        reasoningBudget == 0 ? .off : .on
    }
    /// SPEC §3 **EP-6**: `--slots` / `--no-slots`, the reference's spelling and
    /// the reference's default — the monitoring answer is on unless an operator
    /// turns it off.
    public let slotsEndpointEnabled: Bool
    /// EP-6: `--metrics`, off unless it is asked for, which is again what the
    /// reference does. A Prometheus endpoint nobody scrapes is surface with no
    /// reader.
    public let metricsEndpointEnabled: Bool
    /// FLAG-5. Empty means no authentication, which is the default and is what
    /// every existing runbook starts.
    public let apiKeys: [String]
    /// FLAG-6. Disabled means no CORS headers at all, which is the default.
    public let corsPolicy: ServerCORSPolicy

    public static let usage = """
    usage: TurboFieldfareServer --model <completed .gturbo directory> [options]

      --model <dir>              Required model directory.
      --port <1...65535>         Loopback port (default 8080).
      --model-id <id>            API model identifier (default gemma-4-26b-a4b-it).
      -c, --ctx-size <tokens>    Context length (default 16384). Any token count
                                 is accepted and rounded down to a size this
                                 machine was measured at: 4096, 8192, 16384,
                                 32768, 65536, or 131072. /props reports the
                                 value that survived as n_ctx. Only the
                                 full-attention layers grow with the context, but
                                 they grow linearly: the KV cache measures
                                 1.97 GB at 65536 and 3.31 GB at 131072, which
                                 has to come out of the same Metal working set as
                                 the expert cache, so a long context wants fewer
                                 --expert-cache-slots. A combination the device
                                 cannot keep resident is rejected at load with
                                 the arithmetic.
      --queue-limit <count>      Maximum queued requests (default 4).
      --expert-cache-slots <n>   Expert-cache slots (default 32). Any count is
                                 accepted and rounded down to one this machine was
                                 measured at: 8, 16, 24, or 32.
                                 32 is the operating point and the ceiling: on the
                                 default expert path (mmap) a slot is not a private
                                 copy, so more slots do not show up as footprint --
                                 they show up as pages asked to stay resident, and
                                 past this point that costs throughput rather than
                                 buying it. The load-time guard still rejects a
                                 combination this device cannot hold.
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
      --reasoning-budget <n>     Default thinking budget in tokens: -1 for unlimited
                                 (the default), 0 to close the thought channel, or a
                                 positive ceiling. Reasoning is generated text: it
                                 spends the request's completion budget and comes back
                                 as reasoning_content, separate from the answer. When
                                 the budget runs out — this ceiling, or what is left
                                 of the request's max_tokens — the closing tag is
                                 forced into the stream so the model leaves the
                                 thought channel and writes an answer, instead of
                                 returning reasoning with an empty body. A request
                                 overrides the default with reasoning_budget_tokens,
                                 chat_template_kwargs.enable_thinking, or
                                 reasoning_effort. Declaring tools does not close the
                                 channel.
      --reasoning-format <s>     auto (default) returns the thought channel separately
                                 as reasoning_content; none leaves it in the answer as
                                 raw text. A request overrides it with
                                 reasoning_format.
      --max-images <n>           Images accepted per request (default 4).
      --max-image-bytes <n>      Decoded bytes accepted per image (default 8388608).
                                 Raises the request-body ceiling to match.
      --max-image-pixels <n>     Pixels accepted per image (default 50000000), read
                                 from the container header before any decode.
      --api-key <key[,key...]>   Require this bearer token on every endpoint that
                                 reaches the model (default: none, no
                                 authentication). Repeat the flag or separate
                                 keys with commas to accept more than one.
                                 /health, /v1/health, /models and /v1/models stay
                                 open. The server binds 127.0.0.1 either way.
      --slots, --no-slots        Expose GET /slots, the one generation slot's state
                                 (default: exposed). Turned off, the endpoint
                                 answers 501 like any path this server does not
                                 serve.
      --metrics                  Expose GET /metrics, a Prometheus text exposition
                                 of the prompt and generation counters and the
                                 request gauges (default: not exposed).
      --cors-origins <o[,o...]>  Origins allowed to read this server's answers from
                                 a browser, or * for any (default: none, and no
                                 CORS header is sent). A listed origin is echoed
                                 back on its own; the list itself is never sent,
                                 because Access-Control-Allow-Origin takes one
                                 origin or *. Credentials are never allowed.
      --help                     Show this help.
    """

    /// FLAG-1 / FLAG-4. Flags this server used to take and no longer does, each
    /// with the sentence that tells an operator what to write instead.
    private static let retiredFlags = [
        "--thinking": "use --reasoning-budget 0 to close the thought channel and "
            + "--reasoning-budget -1 (the default) to leave it open",
        "--prompt-cache-mode": "prompt reuse is per request now; send "
            + "cache_prompt: false to opt a request out",
        "--max-context": "use -c/--ctx-size <tokens>, the reference "
            + "implementation's spelling; it takes any token count and rounds "
            + "down to a size this machine can hold",
    ]

    /// FLAG-2 / DEV-2. The context lengths this machine's KV arithmetic was
    /// measured at. `-c` takes any token count and rounds it down to one of
    /// these, so a value between the steps — or past the top — is a size, not
    /// an error; the reference implementation refuses no `-c` either. The value
    /// that survives is what `/props` answers with as `n_ctx` (EP-4).
    public static let supportedContextSizes = [4_096, 8_192, 16_384, 32_768, 65_536, 131_072]

    /// The largest supported size no larger than `requested`. Below the
    /// smallest there is nothing to round down to, so the smallest is the
    /// floor: the request was for less context than this runtime is wired for,
    /// and the answer to that is the least it can give, not a refusal.
    static func supportedContextSize(roundingDown requested: Int) -> Int {
        Self.roundedDown(requested, to: supportedContextSizes)
    }

    /// FLAG-2 / DEV-2, the other flag the rule covers. `--expert-cache-slots`
    /// rounds down the same way `-c` does, to the counts this machine's
    /// working-set arithmetic was measured at
    /// (`RuntimeConfiguration.allowedExpertCacheSlots`); 32 is the operating
    /// point and the ceiling, so asking for more is answered with the ceiling.
    static func supportedExpertCacheSlots(roundingDown requested: Int) -> Int {
        Self.roundedDown(requested, to: RuntimeConfiguration.allowedExpertCacheSlots)
    }

    /// The largest value in `supported` (ascending) that is no larger than
    /// `requested`, with the smallest as the floor.
    private static func roundedDown(_ requested: Int, to supported: [Int]) -> Int {
        supported.last { $0 <= requested } ?? supported[0]
    }

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
        var reasoningBudget = -1
        var reasoningFormat = ReasoningFormat.auto
        var apiKeys: [String] = []
        var corsPolicy = ServerCORSPolicy.disabled
        // EP-6, the reference's defaults at the pin.
        var slotsEndpointEnabled = true
        var metricsEndpointEnabled = false
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            // FLAG-4. A retired flag is a usage error like any other unknown
            // one — same exit code, same usage dump — but it is refused by name
            // so the message can say what replaced it. Checked before the
            // "requires a value" guard so `--thinking` alone is answered with
            // the retirement rather than with a demand for a value it will
            // never accept.
            if let replacement = Self.retiredFlags[flag] {
                throw ServerArgumentError.invalid("\(flag) was retired; \(replacement)")
            }
            // EP-6. The three switches take no value, so they are read before
            // the "requires a value" guard — otherwise a trailing `--metrics`
            // would be refused, and `--no-slots --model …` would swallow the
            // next flag as its argument.
            switch flag {
            case "--slots":
                slotsEndpointEnabled = true
                index += 1
                continue
            case "--no-slots":
                slotsEndpointEnabled = false
                index += 1
                continue
            case "--metrics":
                metricsEndpointEnabled = true
                index += 1
                continue
            default:
                break
            }
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
            case "-c", "--ctx-size":
                // FLAG-1: the reference implementation's spelling, both halves
                // of it. FLAG-2: a free token count, rounded down to a size
                // this machine can hold rather than refused for missing the
                // enumeration.
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid(
                        "--ctx-size must be a positive number of tokens")
                }
                maxContext = Self.supportedContextSize(roundingDown: parsed)
            case "--queue-limit":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--queue-limit must be positive")
                }
                queueLimit = parsed
            case "--expert-cache-slots":
                // FLAG-2: a free slot count, rounded down to one this machine
                // can hold, exactly as `-c` is (DEV-2 covers both).
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid(
                        "--expert-cache-slots must be a positive number of slots")
                }
                expertCacheSlots = Self.supportedExpertCacheSlots(roundingDown: parsed)
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
            case "--api-key":
                // FLAG-5. One key or a comma-separated list, the way the
                // reference spells it; repeating the flag adds to the set
                // rather than replacing it, so a rotation can name the old and
                // the new key on one command line.
                let keys = value.split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }
                guard !keys.isEmpty else {
                    throw ServerArgumentError.invalid("--api-key must not be empty")
                }
                apiKeys.append(contentsOf: keys)
            case "--cors-origins":
                // FLAG-6. `*` or a comma-separated list; the list is matched
                // against the request's `Origin` rather than sent as-is.
                if value.trimmingCharacters(in: .whitespaces) == "*" {
                    corsPolicy = .any
                } else {
                    let origins = value.split(separator: ",")
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                        .filter { !$0.isEmpty }
                    guard !origins.isEmpty else {
                        throw ServerArgumentError.invalid(
                            "--cors-origins must be * or a comma-separated list of origins")
                    }
                    corsPolicy = .origins(origins)
                }
            case "--reasoning-budget":
                // RSN-1. The same range as REQ-reasoning-budget, so the flag
                // and the request field refuse the same numbers.
                guard let parsed = Int(value), parsed >= -1 else {
                    throw ServerArgumentError.invalid(
                        "--reasoning-budget must be -1 (unlimited), 0 (no thinking), "
                        + "or a positive token count")
                }
                reasoningBudget = parsed
            case "--reasoning-format":
                guard let parsed = ReasoningFormat(rawValue: value) else {
                    throw ServerArgumentError.invalid("--reasoning-format must be auto or none")
                }
                reasoningFormat = parsed
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
                                   maxImagePixels: maxImagePixels),
                               reasoningBudget: reasoningBudget,
                               reasoningFormat: reasoningFormat,
                               slotsEndpointEnabled: slotsEndpointEnabled,
                               metricsEndpointEnabled: metricsEndpointEnabled,
                               apiKeys: apiKeys,
                               corsPolicy: corsPolicy)
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
