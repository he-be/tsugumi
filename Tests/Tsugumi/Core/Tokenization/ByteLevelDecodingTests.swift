import Foundation
import Testing
@testable import Tsugumi

/// The Ornith decode pipeline, without a tokenizer: the byte alphabet and the
/// streaming rule are pure functions of bytes, and their contract is what makes
/// `QwenDetokenizer` equal to a batch decode.
///
/// The end-to-end agreement with upstream is
/// `TsugumiKernelCheck --qwen-tokenizer`, which needs the installed
/// vocabulary; everything provable without one is proved here.
@Suite("Byte-level decoding")
struct ByteLevelDecodingTests {
    // MARK: - The alphabet

    @Test("The byte alphabet is a bijection over all 256 bytes")
    func alphabetIsBijection() {
        let table = ByteLevelDecoding.byteForScalar
        #expect(table.count == 256)
        #expect(Set(table.values).count == 256)
    }

    @Test("Printable Latin-1 stands for itself, everything else is lifted")
    func alphabetPinnedValues() {
        // Space and newline are the lifted bytes a reader meets first, and the
        // ones a hand-rolled table gets wrong: they are 0x20 and 0x0A, not the
        // scalars they look like.
        #expect(ByteLevelDecoding.bytes(of: "Ġ") == [0x20])
        #expect(ByteLevelDecoding.bytes(of: "Ċ") == [0x0A])
        #expect(ByteLevelDecoding.bytes(of: "A") == [0x41])
        #expect(ByteLevelDecoding.bytes(of: "¡") == [0xA1])
        #expect(ByteLevelDecoding.bytes(of: "ÿ") == [0xFF])
        #expect(ByteLevelDecoding.bytes(of: "ĠA") == [0x20, 0x41])
    }

    @Test("A character outside the alphabet is refused, not guessed")
    func alphabetRefusesForeignCharacters() {
        // swift-transformers force-unwraps this lookup; a foreign token there
        // is a crash. Here it is `nil`, and the detokenizer turns it into
        // replacement characters.
        #expect(ByteLevelDecoding.bytes(of: "日") == nil)
        #expect(ByteLevelDecoding.bytes(of: "▁") == nil)
    }

    // MARK: - Streaming equals batch

    /// Every token of a byte-level vocabulary is *some* byte string, so the
    /// contract is over byte strings: pushing them in any grouping must produce
    /// the concatenation that decoding them all at once produces.
    @Test("Streaming decode equals the batch decode of the same bytes")
    func streamingEqualsBatch() {
        var generator = SplitMix64(seed: 0x51CE_BEEF_5EED_0001)
        for _ in 0..<2_000 {
            let count = Int(generator.next() % 24) + 1
            var bytes: [UInt8] = []
            for _ in 0..<count {
                // Half uniformly random (invalid sequences abound), half drawn
                // from real UTF-8 so complete codepoints occur often.
                if generator.next() % 2 == 0 {
                    bytes.append(UInt8(truncatingIfNeeded: generator.next()))
                } else {
                    let scalars = ["a", " ", "日", "é", "👨", "。", "\n", "ß"]
                    let pick = scalars[Int(generator.next() % UInt64(scalars.count))]
                    bytes.append(contentsOf: Array(pick.utf8))
                }
            }
            let batch = String(decoding: bytes, as: UTF8.self)

            // Push in random-sized groups, the way tokens of different lengths
            // arrive.
            var run = ByteLevelRun()
            var streamed = ""
            var index = 0
            while index < bytes.count {
                let take = min(bytes.count - index, Int(generator.next() % 4) + 1)
                streamed += run.push(token(for: Array(bytes[index..<(index + take)])))
                index += take
            }
            streamed += run.commit()
            #expect(streamed == batch, "bytes \(bytes)")
        }
    }

    @Test("Nothing is emitted while a codepoint is still incomplete")
    func incompleteCodepointIsHeld() {
        var run = ByteLevelRun()
        let bytes = Array("日".utf8) // E6 97 A5
        #expect(run.push(token(for: [bytes[0]])) == "")
        #expect(run.push(token(for: [bytes[1]])) == "")
        #expect(run.push(token(for: [bytes[2]])) == "日")
        #expect(run.commit() == "")
    }

    @Test("A truncated tail is flushed as the lossy conversion sees it")
    func truncatedTailFlushes() {
        var run = ByteLevelRun()
        #expect(run.push(token(for: Array("日".utf8.dropLast()))) == "")
        #expect(run.commit() == "\u{FFFD}")
        #expect(run.commit() == "")
    }

    @Test("An invalid sequence settles immediately instead of waiting")
    func invalidSequenceSettlesLive() {
        // E0 80 can no longer become anything: the first continuation after E0
        // must be A0...BF. The bytes decode now, not at the end of the stream.
        var run = ByteLevelRun()
        #expect(run.push(token(for: [0xE0])) == "")
        let settled = run.push(token(for: [0x80]))
        #expect(settled == String(decoding: [0xE0, 0x80] as [UInt8], as: UTF8.self))
        #expect(run.commit() == "")
    }

    // MARK: - The pending rule itself

    @Test("Pending length counts only sequences that can still complete")
    func pendingSuffixLength() {
        #expect(ByteLevelRun.pendingSuffixLength([]) == 0)
        #expect(ByteLevelRun.pendingSuffixLength([0x41]) == 0)          // ASCII
        #expect(ByteLevelRun.pendingSuffixLength([0xE6]) == 1)          // 3-byte lead
        #expect(ByteLevelRun.pendingSuffixLength([0xE6, 0x97]) == 2)
        #expect(ByteLevelRun.pendingSuffixLength([0xE6, 0x97, 0xA5]) == 0) // complete
        #expect(ByteLevelRun.pendingSuffixLength([0xF0, 0x9F, 0x91]) == 3) // 4-byte lead
        #expect(ByteLevelRun.pendingSuffixLength([0x80]) == 0)          // stray continuation
        #expect(ByteLevelRun.pendingSuffixLength([0xC0]) == 0)          // overlong lead
        #expect(ByteLevelRun.pendingSuffixLength([0xE0, 0x80]) == 0)    // already invalid
        #expect(ByteLevelRun.pendingSuffixLength([0xED, 0xA0]) == 0)    // surrogate half
        #expect(ByteLevelRun.pendingSuffixLength([0xF4, 0x90]) == 0)    // beyond U+10FFFF
        #expect(ByteLevelRun.pendingSuffixLength([0x41, 0xE6, 0x97]) == 2)
    }

    // MARK: - Helpers

    /// The byte-level spelling of a byte string — what a token of those bytes
    /// looks like in the vocabulary.
    private func token(for bytes: [UInt8]) -> String {
        var text = String.UnicodeScalarView()
        for byte in bytes { text.append(Self.scalarForByte[byte]!) }
        return String(text)
    }

    private static let scalarForByte = Dictionary(uniqueKeysWithValues:
        ByteLevelDecoding.byteForScalar.map { ($0.value, $0.key) })
}

/// Deterministic generator: the failure a random byte string finds has to be
/// reproducible from the seed alone.
private struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { self.state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
