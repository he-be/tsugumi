import TurboFieldfare

public struct Args: Equatable, Sendable {
    public var model: String
    public var prompt: String?
    public var messagesFile: String?
    /// Images attached to the last user turn, in the order given.
    public var images: [String]
    /// Soft-token budget per image. Upstream's `max_soft_tokens`: an upper
    /// bound, not the count — the count follows the aspect ratio.
    public var imageTokens: Int
    public var maxNew: Int
    public var maxContext: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var stops: [String]
    public var quiet: Bool
    public var expertCacheSlots: Int
    public var expertCachePolicy: RuntimeExpertCachePolicy
    public var prefillPolicy: RuntimePrefillPolicy
    public var prefillChunkTokens: Int
    public var rdadvisePolicy: RDAdvicePolicyMode
    public var verification: ModelIntegrityPolicy
    /// Mirrors the chat template's `enable_thinking`. Only affects
    /// `--messages-file`; a raw `--prompt` is passed through verbatim.
    public var thinking: Bool
    /// MTP speculative block width `bs` (`docs/mtp/03-DESIGN.md` D7): 0 disables
    /// speculation, 2...8 runs one bonus token plus `bs - 1` drafted ones.
    public var draftBlockSize: Int
    public var dumpExpertTrace: String?
    /// A JSON file of OpenAI-shaped function declarations. Ornith only
    /// (`docs/qwen35moe/25-CLI-TOOLS.md`); the Gemma arm refuses it rather than
    /// ignoring it.
    public var toolsFile: String?
    /// Which tools the request lets the model reach for (SPEC §6 GEN-1/4).
    public var toolChoice: CLIToolChoice
    /// Whether one turn may carry more than one call.
    public var parallelToolCalls: Bool

    public init(model: String,
                prompt: String? = nil,
                messagesFile: String? = nil,
                images: [String] = [],
                imageTokens: Int = 280,
                maxNew: Int = 1_024,
                maxContext: Int = 4096,
                temperature: Float = 1.0,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stops: [String] = [],
                quiet: Bool = false,
                expertCacheSlots: Int = RuntimeConfiguration.production.expertCacheSlots,
                expertCachePolicy: RuntimeExpertCachePolicy = RuntimeConfiguration.production.expertCachePolicy,
                prefillPolicy: RuntimePrefillPolicy = RuntimeConfiguration.production.prefillPolicy,
                prefillChunkTokens: Int = RuntimeConfiguration.production.prefillChunkTokens,
                rdadvisePolicy: RDAdvicePolicyMode = RuntimeConfiguration.production.rdadvisePolicy,
                verification: ModelIntegrityPolicy = .fullSha256,
                thinking: Bool = false,
                draftBlockSize: Int = 0,
                dumpExpertTrace: String? = nil,
                toolsFile: String? = nil,
                toolChoice: CLIToolChoice = .auto,
                parallelToolCalls: Bool = true) {
        self.model = model
        self.prompt = prompt
        self.messagesFile = messagesFile
        self.images = images
        self.imageTokens = imageTokens
        self.maxNew = maxNew
        self.maxContext = maxContext
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stops = stops
        self.quiet = quiet
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.prefillPolicy = prefillPolicy
        self.prefillChunkTokens = prefillChunkTokens
        self.rdadvisePolicy = rdadvisePolicy
        self.verification = verification
        self.thinking = thinking
        self.draftBlockSize = draftBlockSize
        self.dumpExpertTrace = dumpExpertTrace
        self.toolsFile = toolsFile
        self.toolChoice = toolChoice
        self.parallelToolCalls = parallelToolCalls
    }
}

/// `--tool-choice`, in the four shapes SPEC §6 gives it.
///
/// The server has its own `ChatToolChoice` and this is deliberately not that
/// type: it lives in `TurboFieldfareServerCore`, which links NIO, and the CLI
/// has no business pulling an HTTP stack in to name four cases. What the two
/// share is the grammar builder underneath them.
public enum CLIToolChoice: Equatable, Sendable {
    case auto
    case none
    case required
    case function(String)

    public init(commandLineName: String) {
        switch commandLineName {
        case "auto": self = .auto
        case "none": self = .none
        case "required": self = .required
        default: self = .function(commandLineName)
        }
    }
}

public enum ArgsError: Error, Equatable, CustomStringConvertible {
    case helpRequested
    case unknownFlag(String)
    case missingValue(flag: String)
    case invalidValue(flag: String, value: String)
    case requiredMissing(String)
    case mutuallyExclusive(String, String)
    case modeMissing

    public var description: String {
        switch self {
        case .helpRequested: return "help requested"
        case .unknownFlag(let flag): return "unknown flag: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .invalidValue(let flag, let value): return "invalid value for \(flag): \(value)"
        case .requiredMissing(let flag): return "required flag missing: \(flag)"
        case .mutuallyExclusive(let a, let b): return "\(a) and \(b) are mutually exclusive"
        case .modeMissing: return "one of --prompt or --messages-file is required"
        }
    }
}

extension Args {
    public static let usage = """
    TurboFieldfareCLI — Gemma 4 26B-A4B text generation

    usage: TurboFieldfareCLI --model <dir> (--prompt <string> | --messages-file <path>) [options]

    required:
      --model <dir>             Path to a .gturbo model directory.
      --prompt <string>         Raw-completion prompt.
      --messages-file <path>    JSON chat messages with role and content fields.
                                content may also be a list of parts:
                                [{"type":"text","text":"..."},
                                 {"type":"image","path":"photo.jpg"}]

    options:
      --image <path>             Attach an image to the last user turn
                                 (repeatable, --messages-file only). Requires a
                                 model installed with a vision tower and
                                 --prefill on. Images are appended after that
                                 turn's text, in the order given.
      --image-tokens <n>         Soft-token budget per image: 70, 140, or 280
                                 (default 280). An upper bound, not the count —
                                 the count follows the image's aspect ratio.
      --max-new <int>            Generated-token limit (default 1024).
      --max-context <int>        Context limit in tokens (default 4096).
      --temperature <float>      Sampling temperature (default 1.0, the model-recommended
                                 value; 0 = greedy).
      --top-k <int>              Top-k truncation, 1...256 (default 64; 0 = off).
      --top-p <float>            Nucleus truncation (default 0.95).
      --repetition-penalty <f>   Repetition penalty (default 1.0).
      --seed <uint64>            Deterministic sampling seed (default off).
      --stop <string>            Stop substring (repeatable).
      --quiet                    Suppress the timing footer.
      --expert-cache-slots <n>   Expert-cache slots: 8, 16, 24, or 32 (default 32).
                                 32 is the operating point and the ceiling; a value
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
      --rdadvise <s>             Read-advice policy: off, default, bounded, or adaptive (default off).
      --verification <s>         Model integrity: full-sha256 or trusted-install
                                 (default full-sha256). trusted-install checks sizes
                                 against verified-install.json instead of rehashing
                                 every layer file on first touch.
      --thinking on|off          Chat-template reasoning channel (default off).
                                 off closes the thought channel in the generation
                                 prompt, asking for a direct answer; on leads the
                                 system turn with <|think|> and lets the model
                                 reason first. Reasoning is generated text, so it
                                 spends --max-new. Applies to --messages-file only.
      --draft-block-size <n>     MTP speculative block width: 0 (off, the default)
                                 or 2...8. Needs a model installed with the
                                 drafter section, chunked prefill, and a
                                 repetition penalty of 1.0. Accepted tokens are
                                 the ones the target itself drew, so speculation
                                 moves the wall clock and not the text.
      --dump-expert-trace <path> Write every routed-expert request to a TSV trace.
      --tools <path>             JSON array of function declarations, in the
                                 OpenAI shape ([{"type":"function","function":
                                 {"name":…,"description":…,"parameters":…}}] —
                                 a bare [{"name":…}] list is accepted too).
                                 Ornith (qwen3_5_moe) installs only. The calls
                                 the model writes are printed to stdout as one
                                 JSON object per line.
      --tool-choice <s>          auto (default), none, required, or a declared
                                 function name. auto constrains a call once it
                                 starts; required constrains from the first
                                 generated token, so the turn is a call.
      --parallel-tool-calls on|off
                                 Whether one turn may carry more than one call
                                 (default on).
      --help                     Show this message.
    """

    public func resolvedRuntimeConfiguration(
        forceLogitsHead: Bool) throws -> RuntimeConfiguration {
        guard RuntimeConfiguration.allowedExpertCacheSlots.contains(expertCacheSlots) else {
            throw ArgsError.invalidValue(
                flag: "--expert-cache-slots", value: "\(expertCacheSlots)")
        }
        guard RuntimeConfiguration.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw ArgsError.invalidValue(
                flag: "--prefill-chunk-tokens", value: "\(prefillChunkTokens)")
        }
        let chunkedPrefillSupported = expertCacheSlots >=
            RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
        guard prefillPolicy == .off || chunkedPrefillSupported else {
            throw ArgsError.invalidValue(
                flag: "--expert-cache-slots",
                value: "\(expertCacheSlots) requires --prefill off")
        }
        return RuntimeConfiguration(
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy,
            rdadvisePolicy: rdadvisePolicy,
            prefillEnabled: prefillPolicy == .chunked,
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: forceLogitsHead)
    }

    public static func parse(_ argv: [String]) throws -> Args {
        var model: String?
        var prompt: String?
        var messagesFile: String?
        var images: [String] = []
        var imageTokens = 280
        var maxNew = 1_024
        var maxContext = 4096
        var temperature: Float = 1.0
        var topK: Int? = 64
        var topP: Float? = 0.95
        var repetitionPenalty: Float = 1.0
        var seed: UInt64?
        var stops: [String] = []
        var quiet = false
        let runtimeDefaults = RuntimeConfiguration.production
        var expertCacheSlots = runtimeDefaults.expertCacheSlots
        var expertCachePolicy = runtimeDefaults.expertCachePolicy
        var prefillPolicy = runtimeDefaults.prefillPolicy
        var prefillChunkTokens = runtimeDefaults.prefillChunkTokens
        var rdadvisePolicy = runtimeDefaults.rdadvisePolicy
        var verification = ModelIntegrityPolicy.fullSha256
        var thinking = false
        var draftBlockSize = 0
        var dumpExpertTrace: String?
        var toolsFile: String?
        var toolChoice = CLIToolChoice.auto
        var parallelToolCalls = true

        var index = 0
        while index < argv.count {
            let flag = argv[index]
            switch flag {
            case "--help":
                throw ArgsError.helpRequested
            case "--quiet":
                quiet = true
                index += 1
            case "--model":
                model = try takeValue(argv, &index, flag: flag)
            case "--prompt":
                prompt = try takeValue(argv, &index, flag: flag)
            case "--messages-file":
                messagesFile = try takeValue(argv, &index, flag: flag)
            case "--image":
                images.append(try takeValue(argv, &index, flag: flag))
            case "--image-tokens":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                      VisionPreprocessorConfig.supportedSoftTokens.contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                imageTokens = parsed
            case "--max-new":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxNew = parsed
            case "--max-context":
                let value = try takeValue(argv, &index, flag: flag)
                // 上限は 128K (`RuntimeConfiguration.maximumContextTokens`)。
                // これより上は測っていないし、KV だけで working set を使い切る。
                guard let parsed = Int(value), parsed > 0,
                      parsed <= RuntimeConfiguration.maximumContextTokens else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxContext = parsed
            case "--temperature":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed >= 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                temperature = parsed
            case "--top-k":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (0...256).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topK = parsed == 0 ? nil : parsed
            case "--top-p":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0, parsed <= 1 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topP = parsed
            case "--repetition-penalty":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                repetitionPenalty = parsed
            case "--seed":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = UInt64(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                seed = parsed
            case "--draft-block-size":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                      parsed == 0 || (2...SpeculativeBlock.maxTokens).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                draftBlockSize = parsed
            case "--stop":
                stops.append(try takeValue(argv, &index, flag: flag))
            case "--expert-cache-slots":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                expertCacheSlots = parsed
            case "--expert-cache-policy":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RuntimeExpertCachePolicy(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                expertCachePolicy = parsed
            case "--prefill":
                let value = try takeValue(argv, &index, flag: flag)
                switch value {
                case "on": prefillPolicy = .chunked
                case "off": prefillPolicy = .off
                default: throw ArgsError.invalidValue(flag: flag, value: value)
                }
            case "--prefill-chunk-tokens":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                prefillChunkTokens = parsed
            case "--rdadvise":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RDAdvicePolicyMode(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                rdadvisePolicy = parsed
            case "--verification":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = ModelIntegrityPolicy(commandLineName: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                verification = parsed
            case "--thinking":
                let value = try takeValue(argv, &index, flag: flag)
                switch value {
                case "on": thinking = true
                case "off": thinking = false
                default: throw ArgsError.invalidValue(flag: flag, value: value)
                }
            case "--dump-expert-trace":
                dumpExpertTrace = try takeValue(argv, &index, flag: flag)
            case "--tools":
                toolsFile = try takeValue(argv, &index, flag: flag)
            case "--tool-choice":
                let value = try takeValue(argv, &index, flag: flag)
                guard !value.isEmpty, !value.hasPrefix("--") else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                toolChoice = CLIToolChoice(commandLineName: value)
            case "--parallel-tool-calls":
                let value = try takeValue(argv, &index, flag: flag)
                switch value {
                case "on": parallelToolCalls = true
                case "off": parallelToolCalls = false
                default: throw ArgsError.invalidValue(flag: flag, value: value)
                }
            default:
                throw ArgsError.unknownFlag(flag)
            }
        }

        guard let model else { throw ArgsError.requiredMissing("--model") }
        if prompt != nil && messagesFile != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--messages-file")
        }
        if prompt == nil && messagesFile == nil { throw ArgsError.modeMissing }
        // A raw completion is passed through verbatim, so there is no turn to
        // attach an image to and no place the soft tokens could legitimately go.
        if !images.isEmpty && messagesFile == nil {
            throw ArgsError.mutuallyExclusive("--image", "--prompt")
        }
        if !images.isEmpty && prefillPolicy == .off {
            throw ArgsError.invalidValue(flag: "--prefill", value: "off with --image")
        }
        // The declarations are rendered into the system turn by the chat
        // template, so a raw completion has nowhere to put them — and a
        // `--prompt` run that silently dropped them would look like a model
        // that ignores its tools.
        if toolsFile != nil && messagesFile == nil {
            throw ArgsError.mutuallyExclusive("--tools", "--prompt")
        }
        if toolsFile == nil, toolChoice != .auto {
            throw ArgsError.invalidValue(flag: "--tool-choice",
                                         value: "requires --tools")
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw ArgsError.invalidValue(
                flag: "--top-p",
                value: "\(topP) requires --top-k between 1 and 256")
        }
        let arguments = Args(model: model,
                             prompt: prompt,
                             messagesFile: messagesFile,
                             images: images,
                             imageTokens: imageTokens,
                             maxNew: maxNew,
                             maxContext: maxContext,
                             temperature: temperature,
                             topK: topK,
                             topP: topP,
                             repetitionPenalty: repetitionPenalty,
                             seed: seed,
                             stops: stops,
                             quiet: quiet,
                             expertCacheSlots: expertCacheSlots,
                             expertCachePolicy: expertCachePolicy,
                             prefillPolicy: prefillPolicy,
                             prefillChunkTokens: prefillChunkTokens,
                             rdadvisePolicy: rdadvisePolicy,
                             verification: verification,
                             thinking: thinking,
                             draftBlockSize: draftBlockSize,
                             dumpExpertTrace: dumpExpertTrace,
                             toolsFile: toolsFile,
                             toolChoice: toolChoice,
                             parallelToolCalls: parallelToolCalls)
        if draftBlockSize > 0 {
            guard prefillPolicy == .chunked else {
                throw ArgsError.invalidValue(flag: "--draft-block-size",
                                             value: "\(draftBlockSize) requires --prefill on")
            }
            guard repetitionPenalty == 1.0 else {
                throw ArgsError.invalidValue(
                    flag: "--draft-block-size",
                    value: "\(draftBlockSize) requires --repetition-penalty 1.0")
            }
        }
        _ = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        return arguments
    }

    private static func takeValue(_ argv: [String],
                                  _ index: inout Int,
                                  flag: String) throws -> String {
        guard index + 1 < argv.count else { throw ArgsError.missingValue(flag: flag) }
        let value = argv[index + 1]
        index += 2
        return value
    }
}
