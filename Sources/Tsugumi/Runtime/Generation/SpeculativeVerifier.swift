import Metal

/// A producer that can score a whole block of proposed tokens in one pass and
/// then throw away the ones the accept rule rejected.
///
/// Deliberately a second protocol rather than more surface on `LogitProducer`
/// (`docs/mtp/03-DESIGN.md` D3): every existing producer — including the
/// scripted ones the decode tests run on — keeps compiling untouched, and the
/// speculative loop can ask `as? any SpeculativeVerifier` instead of carrying a
/// capability flag.
public protocol SpeculativeVerifier: LogitProducer {
    /// Append `tokens` to the KV cache and leave one row of output per token.
    ///
    /// `logitRows` receives `tokens.count` rows of FP16 logits, row stride
    /// `vocab`, row *i* holding the distribution for the token that follows
    /// `tokens[i]`. When the runner drives the fused greedy head, `greedyTokens`
    /// receives `tokens.count` `UInt32` argmaxes instead and `logitRows` is not
    /// written; passing `greedyTokens` to a logits-head runner is rejected.
    ///
    /// On return the cursor has advanced by `tokens.count`. The caller decides
    /// how many of those rows survive and calls `rewind(to:)` with the rest.
    func verifyBlock(tokens: ArraySlice<Int32>,
                     startPosition: Int,
                     into logitRows: MTLBuffer,
                     greedyTokens: MTLBuffer?) async throws

    /// Drop every KV row at or after `position`.
    func rewind(to position: Int) throws

    /// Widest block `verifyBlock` will accept.
    var maxSpeculativeBlockTokens: Int { get }
}

public enum SpeculativeBlock {
    /// The `--draft-block-size` ceiling (`docs/mtp/03-DESIGN.md` D7). The
    /// optimum moves with the task — 4 on the pinned pair, higher on the most
    /// predictable prompts (`docs/mtp/10-M0-RESULTS.md` §3) — so the cap is not
    /// set at the measured optimum.
    public static let maxTokens = 8

    /// Bytes one `verifyBlock` logits buffer needs. 512 KiB per row at the
    /// pinned vocabulary; 4 MiB at the widest block.
    public static func logitRowsBytes(vocab: Int, blockTokens: Int) -> Int {
        vocab * blockTokens * MemoryLayout<Float16>.stride
    }
}

public enum SpeculativeVerifyError: Error, CustomStringConvertible, Equatable {
    case blockTooWide(String)
    case cursorMismatch(String)
    case headMismatch(String)
    case bufferTooSmall(String)

    public var description: String {
        switch self {
        case .blockTooWide(let reason),
             .cursorMismatch(let reason),
             .headMismatch(let reason),
             .bufferTooSmall(let reason):
            return reason
        }
    }
}
