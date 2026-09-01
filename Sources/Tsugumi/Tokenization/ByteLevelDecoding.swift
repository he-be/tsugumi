import Foundation

/// The Ornith (Qwen 3.5) detokenization pipeline: GPT-2 byte-level BPE.
///
/// `tokenizer.json` declares `decoder: ByteLevel`, which is a different world
/// from Gemma's `Sequence[Replace("▁" -> " "), ByteFallback, Fuse]`
/// (`GemmaDecoding`):
///
/// - **Every** token is bytes, not just a `<0xXX>` fallback run. Each character
///   of a token stands for one byte through the GPT-2 byte↔unicode table, so a
///   multi-byte codepoint can be split across tokens at any position — and
///   routinely is, for Japanese.
/// - Added tokens (`<|im_end|>`, `<think>`, …) are *not* byte-encoded. The
///   reference decoder emits their content literally and lets the byte run on
///   either side of them continue, which is what `QwenDetokenizer` does.
///
/// Consequently a per-token decode is not text: it is bytes, and the text only
/// appears once a codepoint is complete. `ByteLevelRun` is the streaming form —
/// it holds back the incomplete tail and nothing else, so streaming and batch
/// decode are the same string by construction.
public enum ByteLevelDecoding {
    /// GPT-2's byte↔unicode alphabet, inverted.
    ///
    /// Built by the reference procedure (`bytes_to_unicode` in GPT-2 /
    /// `huggingface/tokenizers`): the printable Latin-1 bytes stand for
    /// themselves, and the remaining 68 bytes are lifted to U+0100… in
    /// increasing byte order. Built here rather than borrowed from
    /// swift-transformers because that table is module-internal.
    public static let byteForScalar: [Unicode.Scalar: UInt8] = {
        var standsForItself = Set<UInt8>()
        for range in [UInt8(0x21)...UInt8(0x7E), UInt8(0xA1)...UInt8(0xAC), UInt8(0xAE)...UInt8(0xFF)] {
            standsForItself.formUnion(range)
        }
        var table: [Unicode.Scalar: UInt8] = [:]
        var lifted: UInt32 = 0
        for byte in UInt8.min...UInt8.max {
            if standsForItself.contains(byte) {
                table[Unicode.Scalar(byte)] = byte
            } else {
                // 256 + n is always a valid scalar (U+0100...U+0143 here).
                table[Unicode.Scalar(256 + lifted)!] = byte
                lifted += 1
            }
        }
        return table
    }()

    /// The bytes a byte-level token stands for, or `nil` if any character of it
    /// is outside the alphabet — which a token from this vocabulary never is,
    /// so the caller treats it as corruption rather than as text.
    public static func bytes(of token: String) -> [UInt8]? {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(token.unicodeScalars.count)
        for scalar in token.unicodeScalars {
            guard let byte = byteForScalar[scalar] else { return nil }
            bytes.append(byte)
        }
        return bytes
    }
}

/// A byte-level run being decoded incrementally.
///
/// `push` appends a token's bytes and returns the text that is now *settled*:
/// every byte whose decoding can no longer be changed by what follows. What it
/// holds back is exactly the trailing partial-but-still-valid UTF-8 sequence —
/// at most 3 bytes. A sequence that has already gone wrong is settled (nothing
/// can rescue it), so an invalid stream decodes live instead of freezing.
///
/// Splitting at settled boundaries is what makes streaming equal batch: Swift's
/// `String(decoding:as:)` replaces each maximal invalid subpart with one
/// U+FFFD, the same rule `huggingface/tokenizers` uses (Rust's lossy UTF-8
/// conversion), and a maximal subpart never straddles a settled boundary.
public struct ByteLevelRun {
    private var bytes: [UInt8] = []

    public init() {}

    /// Text settled by appending `token`'s bytes. A token outside the byte
    /// alphabet contributes one U+FFFD per character it has, after settling
    /// whatever came before it.
    public mutating func push(_ token: String) -> String {
        guard let more = ByteLevelDecoding.bytes(of: token) else {
            return commit() + String(repeating: "\u{FFFD}", count: token.unicodeScalars.count)
        }
        bytes.append(contentsOf: more)
        return settle()
    }

    /// Close the run: an added token follows, or the stream ends. A held-back
    /// tail decodes the way the reference does at end of input — the lossy
    /// conversion's one U+FFFD for the truncated maximal subpart.
    public mutating func commit() -> String {
        guard !bytes.isEmpty else { return "" }
        defer { bytes.removeAll(keepingCapacity: true) }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Emit everything except the trailing sequence that could still complete.
    private mutating func settle() -> String {
        let pending = Self.pendingSuffixLength(bytes)
        guard pending < bytes.count else { return "" }
        let text = String(decoding: bytes[0..<(bytes.count - pending)], as: UTF8.self)
        bytes.removeFirst(bytes.count - pending)
        return text
    }

    /// How many trailing bytes are a valid *incomplete* UTF-8 sequence, i.e.
    /// one that some following byte could still complete. 0 means the whole
    /// buffer is settled — either complete or already invalid.
    public static func pendingSuffixLength(_ bytes: [UInt8]) -> Int {
        // A sequence is at most 4 bytes, so only the last 3 can be pending.
        let window = min(3, bytes.count)
        guard window > 0 else { return 0 }
        var start = bytes.count - 1
        while start > bytes.count - window, bytes[start] & 0xC0 == 0x80 {
            start -= 1
        }
        let lead = bytes[start]
        let expected: Int
        switch lead {
        case 0xC2...0xDF: expected = 2
        case 0xE0...0xEF: expected = 3
        case 0xF0...0xF4: expected = 4
        default: return 0 // ASCII, a stray continuation, or an invalid lead
        }
        let have = bytes.count - start
        guard have < expected else { return 0 }
        // RFC 3629 constrains the first continuation after E0/ED/F0/F4; a byte
        // outside it means the sequence is already invalid, hence settled.
        if have >= 2 {
            let first = bytes[start + 1]
            let allowed: ClosedRange<UInt8>
            switch lead {
            case 0xE0: allowed = 0xA0...0xBF
            case 0xED: allowed = 0x80...0x9F
            case 0xF0: allowed = 0x90...0xBF
            case 0xF4: allowed = 0x80...0x8F
            default: allowed = 0x80...0xBF
            }
            guard allowed.contains(first) else { return 0 }
        }
        return have
    }
}
