import Metal

/// What a speculative round needs on top of `verifyBlock` (`docs/mtp/03-DESIGN.md` D6):
/// proposals from the drafter, and the target hidden the drafter conditions on.
///
/// Split from `SpeculativeVerifier` for the same reason that one was split from
/// `LogitProducer`: M4's verify-block check drives verification alone, and it
/// keeps compiling without a drafter installed.
public protocol SpeculativeDrafting: SpeculativeVerifier {
    /// Whether this producer can draft at all — a model installed without
    /// `flags.mtpDraft` cannot, and the caller has to fall back to plain decode
    /// rather than fail the request.
    var isDraftInstalled: Bool { get }

    /// Elements between rows of `speculativeHiddenRows` (the backbone hidden size).
    var speculativeHiddenRowStride: Int { get }

    /// When set, every head emission also writes the **post-norm** final hidden
    /// of each of its head rows here, row *i* at `i * speculativeHiddenRowStride`
    /// FP16 elements in: one row for a prompt prefill (its final row), `k` rows
    /// for a `verifyBlock` of `k` tokens.
    ///
    /// The head applies the final RMSNorm inside its own kernel, so the
    /// post-norm hidden the drafter needs exists nowhere else
    /// (`docs/mtp/02-RUNTIME-FIT.md` N3). Capturing it costs one extra
    /// hidden-wide norm per row and only when this is non-nil.
    var speculativeHiddenRows: MTLBuffer? { get set }

    /// `count` greedy proposals for the positions after `position`.
    ///
    /// `bonusToken` is the token that will occupy `position` and has no K/V row
    /// yet; `hidden` holds the post-norm hidden of the row that *produced* it,
    /// at row `hiddenRow` (14-M3.5 §6 pins this convention).
    func draftProposals(bonusToken: Int32,
                        position: Int,
                        hidden: MTLBuffer,
                        hiddenRow: Int,
                        count: Int) throws -> [Int32]

    /// Drafter wall time and step count so far, for the footer. Diagnostics
    /// only — 22-GOAL-RESET §6 keeps micro numbers out of pass/fail.
    var draftStepCount: Int { get }
    var draftNanos: UInt64 { get }
    var verifyBlockCount: Int { get }
    var verifyBlockNanos: UInt64 { get }
}

public enum SpeculativeDraftError: Error, CustomStringConvertible, Equatable {
    case notInstalled(String)
    case blockSizeUnsupported(String)
    case unsupportedConfig(String)

    public var description: String {
        switch self {
        case .notInstalled(let reason),
             .blockSizeUnsupported(let reason),
             .unsupportedConfig(let reason):
            return reason
        }
    }
}
