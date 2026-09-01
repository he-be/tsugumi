import Testing
import Foundation
import Metal
@testable import Tsugumi

/// The drafter's shader library has to match the drafter's *file*, not the
/// target's. Since `docs/mtp/45-W2-SYM-ADOPTION.md` the target can be packed
/// `sym` while the pinned drafter stays `affine`, and §3a of that document is
/// the failure this suite guards: a kernel that derives the zero point where
/// the file stores one keeps producing tokens, so the mismatch never announces
/// itself. It has to be decided before the library is compiled.
@Suite struct DraftContextPlanTests {

    /// A drafter geometry that only differs in the two quantization fields the
    /// plan reads. Every other field is the pinned drafter's and is irrelevant
    /// here — `crossCheck(against:)` is what guards those.
    private func draft(groupSize: Int,
                       scheme: Quantization.AffineScheme) -> DraftConfig {
        DraftConfig(hiddenSize: 1024,
                    numLayers: 3,
                    numHeads: 8,
                    numKVHeads: 2,
                    numFullKVHeads: 2,
                    headDim: 128,
                    fullHeadDim: 256,
                    intermediateSize: 4096,
                    backboneHiddenSize: 2560,
                    vocabSize: 262144,
                    slidingWindow: 1024,
                    ropeTheta: 10000,
                    fullRopeTheta: 1000000,
                    partialRotaryFactor: 0.5,
                    rmsNormEps: 1e-6,
                    fullAttentionLayerMask: [0, 0, 1],
                    sharedSlidingKVLayer: 46,
                    sharedFullKVLayer: 47,
                    quantBits: 4,
                    quantGroupSize: groupSize,
                    quantScheme: scheme)
    }

    /// The shipped pair: the target is `sym` at group 32 (45 §4) and the pinned
    /// drafter is `affine` at group 64 (44 §1). Two disagreements, one context.
    @Test func pinnedPair_getsItsOwnAffineContext() {
        let plan = DraftContextPlan.resolve(draft: draft(groupSize: 64, scheme: .affine),
                                            targetGroupSize: 32,
                                            targetScheme: .sym)
        #expect(plan == .ownContext(groupSize: 64, scheme: .affine))
    }

    /// The defect this suite was written for. Matching group sizes are *not*
    /// enough: deciding on the group size alone hands a `sym` library to an
    /// `affine` drafter, which reads `-8 * scale` over a stored bias.
    @Test func sameGroupSizeDifferentScheme_doesNotShareTheTargetContext() {
        let plan = DraftContextPlan.resolve(draft: draft(groupSize: 32, scheme: .affine),
                                            targetGroupSize: 32,
                                            targetScheme: .sym)
        #expect(plan == .ownContext(groupSize: 32, scheme: .affine))
    }

    /// The mirror image: a `sym` drafter against an `affine` target. The
    /// drafter would read a bias array its file does not carry.
    @Test func sameGroupSizeSymDrafterOnAffineTarget_doesNotShareTheTargetContext() {
        let plan = DraftContextPlan.resolve(draft: draft(groupSize: 64, scheme: .sym),
                                            targetGroupSize: 64,
                                            targetScheme: .affine)
        #expect(plan == .ownContext(groupSize: 64, scheme: .sym))
    }

    /// Agreement on both is the only case that may share, and it must: a second
    /// context means a second compiled library for no reason.
    @Test func sameGroupSizeSameScheme_sharesTheTargetContext() {
        for scheme in Quantization.AffineScheme.allCases {
            let plan = DraftContextPlan.resolve(draft: draft(groupSize: 64, scheme: scheme),
                                                targetGroupSize: 64,
                                                targetScheme: scheme)
            #expect(plan == .shareTarget, "scheme=\(scheme.rawValue)")
        }
    }

    /// The plan is only half of it: the context it builds has to actually carry
    /// both specializations. A fresh context defaults to `affine`, so a `sym`
    /// drafter that is never told would compile the wrong library and pass this
    /// suite's decision tests while still being wrong.
    @Test func ownContext_isSpecializedForTheDrafterNotTheTarget() throws {
        let target = try MetalContext()
        try target.setAffineGroupSize(32)
        try target.setAffineScheme(.sym)

        let plan = DraftContextPlan.resolve(draft: draft(groupSize: 64, scheme: .affine),
                                            targetGroupSize: target.affineGroupSize,
                                            targetScheme: target.affineScheme)
        let context = try plan.context(target: target)

        #expect(context !== target)
        #expect(context.affineGroupSize == 64)
        #expect(context.affineScheme == .affine)
        // The target keeps its own specialization.
        #expect(target.affineGroupSize == 32)
        #expect(target.affineScheme == .sym)
    }

    /// Same check for a `sym` drafter, which is the direction a fresh context's
    /// default would hide.
    @Test func ownContext_carriesSymToTheDrafterLibrary() throws {
        let target = try MetalContext()
        try target.setAffineGroupSize(32)
        try target.setAffineScheme(.affine)

        let plan = DraftContextPlan.resolve(draft: draft(groupSize: 64, scheme: .sym),
                                            targetGroupSize: target.affineGroupSize,
                                            targetScheme: target.affineScheme)
        let context = try plan.context(target: target)

        #expect(context.affineGroupSize == 64)
        #expect(context.affineScheme == .sym)
    }

    /// A shared plan hands back the very same context, not a copy of it.
    @Test func shareTarget_returnsTheTargetContextItself() throws {
        let target = try MetalContext()
        try target.setAffineGroupSize(64)
        try target.setAffineScheme(.affine)

        let plan = DraftContextPlan.resolve(draft: draft(groupSize: 64, scheme: .affine),
                                            targetGroupSize: target.affineGroupSize,
                                            targetScheme: target.affineScheme)
        #expect(try plan.context(target: target) === target)
    }

    /// `DraftConfig` is where the plan reads the drafter's scheme from, so the
    /// manifest's `draft.quant.scheme` has to reach it. The packer has written
    /// that field since the drafter was added (`MoEPackJSON.draftSection`).
    @Test func draftConfig_carriesTheSchemeFromTheManifest() {
        for (wire, expected) in [("affine", Quantization.AffineScheme.affine),
                                 ("sym", .sym),
                                 ("SYM", .sym)] {
            let config = DraftConfig(manifest: manifestDraft(scheme: wire))
            #expect(config.quantScheme == expected, "wire=\(wire)")
            #expect(config.quantGroupSize == 64)
            #expect(config.quantBits == 4)
        }
    }

    private func manifestDraft(scheme: String) -> ManifestDraft {
        ManifestDraft(hiddenSize: 1024,
                      numLayers: 3,
                      numHeads: 8,
                      numKVHeads: 2,
                      numFullKVHeads: 2,
                      headDim: 128,
                      fullHeadDim: 256,
                      intermediateSize: 4096,
                      backboneHiddenSize: 2560,
                      vocabSize: 262144,
                      slidingWindow: 1024,
                      ropeTheta: 10000,
                      fullRopeTheta: 1000000,
                      partialRotaryFactor: 0.5,
                      rmsNormEps: 1e-6,
                      hiddenActivation: "gelu_pytorch_tanh",
                      tieWordEmbeddings: true,
                      attentionKEqV: false,
                      fullAttentionLayerMask: [0, 0, 1],
                      sharedSlidingKVLayer: 46,
                      sharedFullKVLayer: 47,
                      quant: ManifestQuantSlot(weightBits: 4,
                                               scheme: scheme,
                                               scaleType: "BF16",
                                               biasType: scheme.lowercased() == "sym" ? "none" : "BF16",
                                               groupSize: 64),
                      weightsPath: "draft/draft_weights.bin",
                      tensorCount: 94,
                      payloadBytes: 236_114_440,
                      sourceRepo: "mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit",
                      sourceRevision: "bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c")
    }
}
