import Hub
import Testing
@testable import TurboFieldfare

/// Structural verification of Ornith's `tokenizer.json` declarations, driven by
/// `Config` literals — no network and no tokenizer fixture required, the same
/// shape as `DecoderConfigTests` on the Gemma side.
///
/// The two families' checks have to reject each other. Neither runtime can
/// decode the other's vocabulary, and both failures are silent: Gemma's
/// pipeline over a byte-level vocabulary prints `ãģĵãĤĵ`, and Ornith's over a
/// metaspace one prints replacement characters. A wrong `TURBO_FIELDFARE_
/// TOKENIZER_DIR` is the way that happens in practice.
@Suite("Ornith decoder configuration")
struct QwenDecoderConfigTests {
    /// What `scratch/ornith-oq4e-g64.gturbo/tokenizer/tokenizer.json` declares
    /// (実測(上流)).
    private static let ornith: Config = [
        "decoder": [
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false,
        ],
        "preTokenizer": [
            "type": "Sequence",
            "pretokenizers": [
                ["type": "Split", "behavior": "Isolated"],
                ["type": "ByteLevel", "add_prefix_space": false],
            ],
        ],
    ]

    private static let gemma: Config = [
        "decoder": [
            "type": "Sequence",
            "decoders": [
                ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
                ["type": "ByteFallback"],
                ["type": "Fuse"],
            ],
        ],
        "preTokenizer": ["type": "Metaspace", "replacement": "▁"],
    ]

    @Test("The pinned Ornith declaration passes both checks")
    func ornithPasses() throws {
        try QwenTokenizer.verifyDecoderConfiguration(Self.ornith)
        try QwenTokenizer.verifyPreTokenizerConfiguration(Self.ornith)
    }

    @Test("A bare ByteLevel pre-tokenizer passes")
    func bareByteLevelPreTokenizerPasses() throws {
        try QwenTokenizer.verifyPreTokenizerConfiguration(
            ["preTokenizer": ["type": "ByteLevel"]])
    }

    @Test("Gemma's decoder declaration is rejected")
    func gemmaDecoderRejected() {
        #expect(throws: QwenTokenizerError.self) {
            try QwenTokenizer.verifyDecoderConfiguration(Self.gemma)
        }
    }

    @Test("A metaspace pre-tokenizer is rejected")
    func metaspacePreTokenizerRejected() {
        #expect(throws: QwenTokenizerError.self) {
            try QwenTokenizer.verifyPreTokenizerConfiguration(Self.gemma)
        }
    }

    @Test("A sequence with no ByteLevel step is rejected")
    func sequenceWithoutByteLevelRejected() {
        #expect(throws: QwenTokenizerError.self) {
            try QwenTokenizer.verifyPreTokenizerConfiguration([
                "preTokenizer": [
                    "type": "Sequence",
                    "pretokenizers": [["type": "Split", "behavior": "Isolated"]],
                ],
            ])
        }
    }

    @Test("A missing declaration is rejected, not defaulted")
    func missingDeclarationRejected() {
        #expect(throws: QwenTokenizerError.self) {
            try QwenTokenizer.verifyDecoderConfiguration(["model": ["type": "BPE"]])
        }
        #expect(throws: QwenTokenizerError.self) {
            try QwenTokenizer.verifyPreTokenizerConfiguration(["model": ["type": "BPE"]])
        }
    }

    @Test("Gemma's check rejects the Ornith declaration")
    func gemmaCheckRejectsOrnith() {
        // The other direction of the same claim: the Gemma loader must not
        // accept this tokenizer either, or a mixed-up `--model` would decode
        // every Ornith token as if its characters were text.
        #expect(throws: GFTokenizerError.self) {
            try GFTokenizer.verifyDecoderConfiguration(Self.ornith)
        }
    }
}
