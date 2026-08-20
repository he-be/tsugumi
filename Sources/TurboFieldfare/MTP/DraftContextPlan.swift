import Foundation

/// Which shader library the MTP drafter's kernels come from.
///
/// The drafter is quantized on its own terms: the pinned pair is `affine` at
/// group 64 against a target that is `sym` at group 32. Both facts are baked
/// into the compiled library (`MetalContext.setAffineGroupSize`,
/// `setAffineScheme`), so sharing the target's context is sound only when the
/// two agree on *both*. A disagreement on either one needs a context of its
/// own — which is cheap, because contexts share the device and pass buffers
/// freely (`MetalContext.init(sharingDeviceWith:)`).
///
/// Deciding on the group size alone is not enough, and the way it fails is
/// silent: an `affine` drafter decoded by a `sym` library reads `-8 * scale`
/// where the file stores a real zero point, so it keeps producing tokens —
/// just not the right ones. That is the failure mode
/// `docs/mtp/45-W2-SYM-ADOPTION.md` §3a describes.
enum DraftContextPlan: Equatable, Sendable {
    /// The drafter runs on the target's context.
    case shareTarget
    /// The drafter needs its own context, specialized like this.
    case ownContext(groupSize: Int, scheme: Quantization.AffineScheme)

    /// The plan for running `draft` against a target context specialized for
    /// `targetGroupSize` / `targetScheme`.
    static func resolve(draft: DraftConfig,
                        targetGroupSize: Int,
                        targetScheme: Quantization.AffineScheme) -> DraftContextPlan {
        if draft.quantGroupSize == targetGroupSize, draft.quantScheme == targetScheme {
            return .shareTarget
        }
        return .ownContext(groupSize: draft.quantGroupSize, scheme: draft.quantScheme)
    }

    /// The context this plan calls for, built against `target`'s device.
    func context(target: MetalContext) throws -> MetalContext {
        switch self {
        case .shareTarget:
            return target
        case let .ownContext(groupSize, scheme):
            let context = try MetalContext(sharingDeviceWith: target)
            try context.setAffineGroupSize(groupSize)
            // Set rather than assumed: a fresh context defaults to `affine`,
            // which is what the pinned drafter happens to need. Relying on that
            // would make the `sym` direction wrong and silent.
            try context.setAffineScheme(scheme)
            return context
        }
    }
}
