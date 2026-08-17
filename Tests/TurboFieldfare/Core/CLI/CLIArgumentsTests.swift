import Testing
import TurboFieldfare
@testable import TurboFieldfareCLICore

@Suite struct CLIArgumentsTests {
    @Test func defaultsUseProductionGenerationValues() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(arguments.model == "m.gturbo")
        #expect(arguments.prompt == "hi")
        #expect(arguments.messagesFile == nil)
        #expect(arguments.maxNew == 1_024)
        #expect(arguments.maxContext == 4096)
        #expect(arguments.temperature == 1.0)
        #expect(arguments.topK == 64)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1)
        #expect(arguments.seed == nil)
        #expect(arguments.stops.isEmpty)
        #expect(!arguments.quiet)

        let runtime = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        #expect(runtime == RuntimeConfiguration.production)
    }

    @Test func generationOptionsParseAndStopsRepeat() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--max-new", "32", "--max-context", "512",
            "--temperature", "0", "--top-k", "40", "--top-p", "0.95",
            "--repetition-penalty", "1.1", "--seed", "42",
            "--stop", "A", "--stop", "B", "--quiet",
        ])
        #expect(arguments.maxNew == 32)
        #expect(arguments.maxContext == 512)
        #expect(arguments.temperature == 0)
        #expect(arguments.topK == 40)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1.1)
        #expect(arguments.seed == 42)
        #expect(arguments.stops == ["A", "B"])
        #expect(arguments.quiet)
    }

    @Test func topKZeroRequiresTopPToBeDisabled() throws {
        let disabled = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--top-k", "0", "--top-p", "1",
        ])
        #expect(disabled.topK == nil)
        #expect(disabled.topP == 1)

        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--top-k", "0",
            ])
        }
    }

    @Test func topKAboveKernelLimitRejected() {
        #expect(throws: ArgsError.invalidValue(flag: "--top-k", value: "257")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--top-k", "257",
            ])
        }
    }

    @Test func helpListsExactlyThePublicOptions() {
        let expected: Set<String> = [
            "--model", "--prompt", "--messages-file", "--image", "--image-tokens",
            "--max-new", "--max-context",
            "--temperature", "--top-k", "--top-p", "--repetition-penalty",
            "--seed", "--stop", "--quiet", "--expert-cache-slots",
            "--expert-cache-policy", "--prefill", "--prefill-chunk-tokens",
            "--rdadvise", "--verification", "--thinking", "--dump-expert-trace",
            "--help",
        ]
        let words = Args.usage.split { $0.isWhitespace || $0 == "(" || $0 == ")" }
        // Prose mentions flags mid-sentence ("spends --max-new."), so the
        // trailing punctuation is not part of the flag.
        let options = Set(words.map { String($0).trimmingCharacters(in: ["."]) }
            .filter { $0.hasPrefix("--") })
        #expect(options == expected)
    }

    @Test func runtimeOptionsReachTypedConfiguration() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--expert-cache-slots", "24",
            "--expert-cache-policy", "lru",
            "--prefill", "off",
            "--prefill-chunk-tokens", "64",
            "--rdadvise", "adaptive",
        ])

        #expect(arguments.expertCacheSlots == 24)
        #expect(arguments.expertCachePolicy == .lru)
        #expect(arguments.prefillPolicy == .off)
        #expect(arguments.prefillChunkTokens == 64)
        #expect(arguments.rdadvisePolicy == .adaptive)

        let runtime = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: true)
        #expect(runtime.expertCacheSlots == 24)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.prefillPolicy == .off)
        #expect(runtime.prefillChunkTokens == 64)
        #expect(runtime.rdadvisePolicy == .adaptive)
        #expect(runtime.headPath == .logits)
    }

    @Test func everySupportedRuntimeOptionParses() throws {
        for value in RuntimeConfiguration.allowedExpertCacheSlots {
            let prefill = value < RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
                ? "off" : "on"
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--expert-cache-slots", "\(value)",
                "--prefill", prefill,
            ])
            #expect(arguments.expertCacheSlots == value)
        }
        for value in RuntimeConfiguration.allowedPrefillChunkTokens {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--prefill-chunk-tokens", "\(value)",
            ])
            #expect(arguments.prefillChunkTokens == value)
        }
        for value in ["lfu", "lru"] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--expert-cache-policy", value,
            ])
            #expect(arguments.expertCachePolicy.rawValue == value)
        }
        for value in ["off", "default", "bounded", "adaptive"] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--rdadvise", value,
            ])
            #expect(arguments.rdadvisePolicy.rawValue == value)
        }
        #expect(try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--prefill", "on",
        ]).prefillPolicy == .chunked)
        #expect(try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--prefill", "off",
        ]).prefillPolicy == .off)
    }

    @Test func unsupportedRuntimeOptionValuesAreRejected() {
        let invalidValues = [
            ("--expert-cache-slots", "7"),
            ("--expert-cache-policy", "fifo"),
            ("--prefill", "yes"),
            ("--prefill-chunk-tokens", "96"),
            ("--prefill-chunk-tokens", "4096"),
            ("--rdadvise", "automatic"),
        ]
        for (flag, value) in invalidValues {
            #expect(throws: ArgsError.invalidValue(flag: flag, value: value)) {
                _ = try Args.parse([
                    "--model", "m.gturbo", "--prompt", "hi", flag, value,
                ])
            }
        }

        #expect(throws: ArgsError.invalidValue(
            flag: "--expert-cache-slots", value: "8 requires --prefill off")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--expert-cache-slots", "8", "--prefill", "on",
            ])
        }
    }

    @Test func programmaticArgumentsCannotReachRuntimePreconditions() {
        var arguments = Args(model: "m.gturbo", prompt: "hi")
        arguments.expertCacheSlots = 7
        #expect(throws: ArgsError.invalidValue(
            flag: "--expert-cache-slots", value: "7")) {
            _ = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        }

        arguments.expertCacheSlots = RuntimeConfiguration.production.expertCacheSlots
        arguments.prefillChunkTokens = 96
        #expect(throws: ArgsError.invalidValue(
            flag: "--prefill-chunk-tokens", value: "96")) {
            _ = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        }

        arguments.prefillChunkTokens = RuntimeConfiguration.production.prefillChunkTokens
        arguments.expertCacheSlots = 8
        #expect(throws: ArgsError.invalidValue(
            flag: "--expert-cache-slots", value: "8 requires --prefill off")) {
            _ = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        }
    }

    @Test func unsupportedSelectorsAreRejected() {
        for flag in ["--runtime-profile", "--experiment-id", "-h"] {
            #expect(throws: ArgsError.unknownFlag(flag)) {
                _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", flag])
            }
        }
    }

    @Test func modelAndPromptAreRequired() {
        #expect(throws: ArgsError.requiredMissing("--model")) {
            _ = try Args.parse(["--prompt", "hi"])
        }
        #expect(throws: ArgsError.modeMissing) {
            _ = try Args.parse(["--model", "m.gturbo"])
        }
    }

    @Test func messagesFileSelectsChatMode() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--messages-file", "chat.json",
        ])
        #expect(arguments.prompt == nil)
        #expect(arguments.messagesFile == "chat.json")
    }

    // MARK: - Images (PLAN_VISION §4-6)

    @Test func imagesAttachToAMessagesFileAndDefaultToTheFullBudget() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--messages-file", "chat.json",
            "--image", "a.jpg", "--image", "b.png",
        ])
        #expect(arguments.images == ["a.jpg", "b.png"])
        #expect(arguments.imageTokens == 280)
    }

    /// A raw completion is passed through verbatim: there is no turn to attach
    /// an image to, and silently inventing one would change what the model is
    /// asked.
    @Test func imagesRequireAMessagesFile() {
        #expect(throws: ArgsError.mutuallyExclusive("--image", "--prompt")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--image", "a.jpg",
            ])
        }
    }

    /// Scalar replay has nowhere to put a soft token, so an image with prefill
    /// off is rejected at parse time rather than deep inside the runner.
    @Test func imagesRequireChunkedPrefill() {
        #expect(throws: ArgsError.invalidValue(flag: "--prefill", value: "off with --image")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--messages-file", "chat.json",
                "--image", "a.jpg", "--prefill", "off",
            ])
        }
    }

    @Test(arguments: [70, 140, 280])
    func supportedImageTokenBudgetsParse(_ budget: Int) throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--messages-file", "chat.json",
            "--image-tokens", "\(budget)",
        ])
        #expect(arguments.imageTokens == budget)
    }

    /// 560 and 1120 are budgets upstream accepts and this runtime does not; the
    /// rejection has to be explicit rather than a silently clamped 280.
    @Test(arguments: ["0", "100", "560", "1120"])
    func unsupportedImageTokenBudgetsAreRejected(_ value: String) {
        #expect(throws: ArgsError.invalidValue(flag: "--image-tokens", value: value)) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--messages-file", "chat.json",
                "--image-tokens", value,
            ])
        }
    }

    @Test func promptAndMessagesFileAreMutuallyExclusive() {
        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--messages-file")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--messages-file", "chat.json",
            ])
        }
    }
}
