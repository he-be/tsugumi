import Foundation
import Testing
@testable import TsugumiAppCore

@Suite struct RuriUnigramTokenizerTests {
    /// Six specials, the 256 byte pieces, then the pieces the cases need.
    static let pieces: [(String, Double)] = [
        ("\\uFEFF", -1.0), ("\\uFEFF\\uFEFF", -1.5), ("e", -2), ("▁", -3), ("a", -2), ("ab", -1), ("b", -2),
    ]

    static func tokenizer() throws -> (RuriUnigramTokenizer, (String) -> Int32) {
        var vocab = ["<unk>", "<s>", "</s>", "<pad>", "<sep>", "<mask>"].map { #"["\#($0)", 0.0]"# }
        vocab += (0...255).map { String(format: #"["<0x%02X>", 0.0]"#, $0) }
        vocab += pieces.map { #"["\#($0.0)", \#($0.1)]"# }
        let json = #"{"model": {"type": "Unigram", "vocab": [\#(vocab.joined(separator: ", "))]}}"#
        let tokenizer = try RuriUnigramTokenizer(json: Data(json.utf8))
        func id(_ piece: String) -> Int32 {
            if piece.hasPrefix("<0x") { return 6 + Int32(piece.dropFirst(3).prefix(2), radix: 16)! }
            let escaped = piece.unicodeScalars.map { $0.value == 0xFEFF ? "\\uFEFF" : String($0) }.joined()
            return Int32(6 + 256 + pieces.firstIndex { $0.0 == escaped }!)
        }
        return (tokenizer, id)
    }

    @Test func leadingByteOrderMarkIsAPieceOfItsOwn() throws {
        // JSONSerialization dropped the leading U+FEFF: "﻿" became "" and "﻿﻿" took its place.
        let (tokenizer, id) = try Self.tokenizer()
        #expect(tokenizer.encode("\u{FEFF}") == [id("\u{FEFF}")])
        #expect(tokenizer.encode("\u{FEFF}\u{FEFF}") == [id("\u{FEFF}\u{FEFF}")])
    }

    @Test func bestScoringSplitAndSpacesAsMetaspace() throws {
        let (tokenizer, id) = try Self.tokenizer()
        #expect(tokenizer.encode("ab") == [id("ab")])
        #expect(tokenizer.encode("a b") == [id("a"), id("▁"), id("b")])
    }

    @Test func unknownScalarsFallBackToTheirBytesOneScalarAtATime() throws {
        // A combining mark is its own character to tokenizers, not part of the "e" before it (Swift's Character).
        let (tokenizer, id) = try Self.tokenizer()
        #expect(tokenizer.encode("e\u{0301}") == [id("e"), id("<0xCC>"), id("<0x81>")])
        // The precomposed form is not folded into the decomposed one.
        #expect(tokenizer.encode("\u{00E9}") == [id("<0xC3>"), id("<0xA9>")])
    }
}

/// The converted model itself, where it has been put (docs/qwen38/39). Skipped elsewhere.
@Suite struct RuriSectionEmbedderModelTests {
    static let directory = URL(fileURLWithPath: NSString(string: "~/LLM/ruri-v3-130m-ane-2048").expandingTildeInPath)

    @Test(.enabled(if: FileManager.default.fileExists(atPath: directory.path)))
    func matchesKohagiOnTheReadmeSentence() throws {
        let embedder = try RuriSectionEmbedder(directory: Self.directory)
        #expect(embedder.buckets == [64, 128, 256, 512, 1024, 2048])
        // kohagi's Core ML output for the same sentence, no prefix (docs/qwen38/38 §2-1): 0.00702, -0.03196, …
        let vector = try embedder.vector("瑠璃も玻璃も照らせば光る")
        #expect(vector.count == 512)
        #expect(abs(vector[0] - 0.00702) < 2e-3)
        #expect(abs(vector[1] - -0.03196) < 2e-3)
        #expect(abs(vector.reduce(0) { $0 + $1 * $1 } - 1) < 1e-4)
        // A text past the largest bucket is cut, not refused.
        #expect(try embedder.embed(documents: [String(repeating: "東京駅は鉄道駅である。", count: 400)]).first?.count == 512)
    }
}
