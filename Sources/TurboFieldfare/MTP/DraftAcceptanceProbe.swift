import Foundation
import Metal

/// M3 (`docs/mtp/04-PHASES.md` §2): the acceptance rate of the real drafter,
/// measured **without** a verify pass, KV rewind, or any change to the decode
/// loop.
///
/// At every decode step the target hands over the two things upstream's
/// `SinglePositionMultiTokenCandidateGenerator` feeds the drafter — the last
/// *seen* token and the post-norm final hidden the target computed **for that
/// same position** — and the drafter runs `drafts` steps from them, writing
/// nothing to the KV cache. So a round anchored at position `p` is built from
/// `(token at p, hidden at p, KV [0, p])` and proposes the tokens at
/// `p + 1, p + 2, ...`; the first proposal races the target's own draw from
/// that very hidden, which is what an MTP head is for.
///
/// The proposals are compared against the tokens the target actually produces
/// on the following steps; the length of the matching prefix *is* the
/// acceptance length for that round (**derived** from D5's same-seed
/// sample-and-compare rule: the target's draw is what it is, so a proposal is
/// accepted exactly when it equals that draw).
///
/// One run at `drafts = k` answers every `bs <= k + 1` at once: the drafter is
/// greedy and its proposals do not depend on `bs`, so the `bs` round is the
/// `bs - 1` prefix of this one (**derived**).
///
/// Enabled only by `TF_DRAFT_PROBE=<path>`; when it is unset the decode path
/// pays one optional-nil test per step. The probe adds a full drafter forward
/// per draft step to every decode step, so wall-clock numbers from a probing
/// run say nothing about MTP's speed — that is what M0's cost model and M5's
/// A/B are for. The per-step drafter cost *is* reported (`draft_ms`), because
/// the go/no-go formula needs it.
final class DraftAcceptanceProbe {
    /// One drafting round: the drafter's proposals for the positions after
    /// `position`, plus the bonus token they were conditioned on.
    private struct Round {
        let position: Int
        let bonus: Int32
        let proposals: [Int32]
        /// Target tokens observed so far for `position + 1 ...`.
        var actual: [Int32] = []
        /// Set once a mismatch is seen; the rest of the round cannot be
        /// accepted, so it can be written out without waiting for more tokens.
        var broken = false
    }

    /// How a round at target position `p` (input token `t_p`) is assembled.
    /// Upstream does not read the same way twice — HF slices the hidden of the
    /// token *before* the bonus token, mlx-vlm puts the query one slot past the
    /// cache — and 12-M2's fixtures pin only the arithmetic, not the loop's
    /// convention. So all three readings are measurable and the numbers decide.
    enum Mode: String, Sendable {
        /// EAGLE / HF `SinglePositionMultiTokenCandidateGenerator`: the hidden
        /// that *produced* `t_p`, the query at `t_p`'s own index, and a cache
        /// that stops before it (`t_p` has no K/V of its own yet).
        case prevBonus = "prev-bonus"
        /// As above but with the query on the last cached position.
        case prevLast = "prev-last"
        /// The hidden the target computed *for* `t_p`, the query on `t_p`'s own
        /// cached position (`position == kvLen - 1`, the M2 fixture layout).
        case same = "same"
    }

    /// `TF_DRAFT_PROBE=<path>` turns it on; `TF_DRAFT_PROBE_DRAFTS=<k>` sets
    /// the draft depth (default 3, i.e. bs = 2/3/4 in one run);
    /// `TF_DRAFT_PROBE_MODE` picks the convention above (default `prev-bonus`).
    static func settings(_ env: [String: String] = ProcessInfo.processInfo.environment)
        -> (path: String, drafts: Int, mode: Mode)? {
        guard let path = env["TF_DRAFT_PROBE"], !path.isEmpty else { return nil }
        let drafts = env["TF_DRAFT_PROBE_DRAFTS"].flatMap { Int($0) } ?? 3
        guard drafts >= 1 else { return nil }
        let mode = env["TF_DRAFT_PROBE_MODE"].flatMap(Mode.init(rawValue:)) ?? .prevBonus
        return (path, min(drafts, 16), mode)
    }

    private let drafter: DraftForward
    private let config: DraftConfig
    /// The bonus token's embedding comes from the *target's* table, so the
    /// lookup runs on the target's context (group 32) while the drafter runs on
    /// its own (group 64) — see `MetalContext.init(sharingDeviceWith:)`.
    private let embed: EmbedLookupInt4
    private let embedTable: TensorView
    private let embedQueue: MTLCommandQueue
    private let draftQueue: MTLCommandQueue
    private let drafts: Int
    private let mode: Mode
    private let embedScale: Float

    /// The runner writes this step's post-norm final hidden here; `prevHidden`
    /// keeps the one before it, which is what the EAGLE-shaped modes feed the
    /// drafter. Swapped at the end of every round.
    private(set) var hiddenCaptureBuffer: MTLBuffer
    private var prevHidden: MTLBuffer
    private var prevHiddenPosition = -1

    // Ping-pong so a draft step's `post_projection` output feeds the next step
    // without aliasing the buffer it is reading from.
    private let hiddenA: MTLBuffer
    private let hiddenB: MTLBuffer
    private let embedBuffer: MTLBuffer
    private let tokenBuffer: MTLBuffer

    private let handle: FileHandle
    private var pending: [Round] = []
    private var rounds = 0
    private var acceptedHistogram: [Int]
    private var draftSteps = 0
    private var draftNanos: UInt64 = 0
    private var finished = false

    init(context: MetalContext,
         draftContext: MetalContext,
         weights: DraftWeights,
         embedTable: TensorView,
         backboneHiddenSize: Int,
         path: String,
         drafts: Int,
         mode: Mode) throws {
        self.drafter = try DraftForward(context: draftContext, weights: weights)
        self.config = weights.config
        self.embed = try EmbedLookupInt4(context: context)
        self.embedTable = embedTable
        self.embedQueue = context.queue
        self.draftQueue = draftContext.queue
        self.drafts = drafts
        self.mode = mode
        self.embedScale = Float(backboneHiddenSize).squareRoot()
        self.acceptedHistogram = [Int](repeating: 0, count: drafts + 1)

        let device = context.device
        func buffer(_ count: Int, shared: Bool = false) throws -> MTLBuffer {
            guard let b = device.makeBuffer(
                length: max(count * MemoryLayout<Float16>.size, 4),
                options: shared ? .storageModeShared : .storageModePrivate) else {
                throw MetalError.noDevice
            }
            return b
        }
        self.hiddenCaptureBuffer = try buffer(backboneHiddenSize)
        self.prevHidden = try buffer(backboneHiddenSize)
        self.hiddenA = try buffer(backboneHiddenSize)
        self.hiddenB = try buffer(backboneHiddenSize)
        self.embedBuffer = try buffer(backboneHiddenSize)
        self.tokenBuffer = try buffer(2, shared: true)

        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard FileManager.default.createFile(atPath: path, contents: nil) else {
            throw DraftError.geometryMismatch("TF_DRAFT_PROBE: cannot create \(path)")
        }
        self.handle = try FileHandle(forWritingTo: url)
        handle.write(Data("""
            # turbo-fieldfare draft acceptance probe v1
            # drafts=\(drafts) (covers bs=2..\(drafts + 1)) mode=\(mode.rawValue)
            # pos    = target position whose input token anchored the round
            # bonus  = the target's input token at that position
            # accept = length of the matching prefix (0..\(drafts))
            pos\tbonus\tproposed\tactual\taccept\n
            """.utf8))
    }

    /// Called at the top of every decode step: `token` is the target's input
    /// at `position`, which is also the token every open round is waiting to
    /// check its next proposal against.
    func observe(token: Int32, position: Int) {
        guard !finished else { return }
        record(actual: token, at: position)
    }

    /// Called at the end of the same decode step, once the target has written
    /// its K/V for `position` and the runner has put that position's post-norm
    /// hidden in `hiddenCaptureBuffer`.
    func draft(bonusToken: Int32, position: Int, kv: KVCacheManager) throws {
        guard !finished else { return }
        defer { swapHidden(position: position) }
        // `prev-*` needs the hidden from `position - 1`, which does not exist on
        // the first decode step after a prefill or a continuation.
        let hidden: MTLBuffer
        let ropePosition: Int
        let kvLength: Int
        switch mode {
        case .prevBonus, .prevLast:
            guard prevHiddenPosition == position - 1 else { return }
            hidden = prevHidden
            ropePosition = mode == .prevBonus ? position : position - 1
            kvLength = position
        case .same:
            hidden = hiddenCaptureBuffer
            ropePosition = position
            kvLength = position + 1
        }
        let proposals = try draftRound(bonusToken: bonusToken, hidden: hidden,
                                       ropePosition: ropePosition, kvLength: kvLength,
                                       kv: kv)
        pending.append(Round(position: position, bonus: bonusToken, proposals: proposals))
    }

    private func swapHidden(position: Int) {
        let captured = hiddenCaptureBuffer
        hiddenCaptureBuffer = prevHidden
        prevHidden = captured
        prevHiddenPosition = position
    }

    /// `drafts` greedy drafter steps from `(embed(bonus), hidden)` against the
    /// target's shared KV for `[0, kvLength)`. The RoPE position stays constant
    /// for the whole round — the drafter writes no K/V, so there is nothing new
    /// to attend to (upstream holds `position_ids` constant for the same
    /// reason).
    private func draftRound(bonusToken: Int32, hidden: MTLBuffer,
                            ropePosition: Int, kvLength: Int,
                            kv: KVCacheManager) throws -> [Int32] {
        let slidingLayer = config.sharedSlidingKVLayer
        let fullLayer = config.sharedFullKVLayer
        let slidingK = kv.keyView(layer: slidingLayer)
        let slidingV = kv.valueView(layer: slidingLayer)
        let fullK = kv.keyView(layer: fullLayer)
        let fullV = kv.valueView(layer: fullLayer)
        let ringCapacity = kv.ringCapacity(layer: slidingLayer)

        var proposals: [Int32] = []
        var token = bonusToken
        var input = hidden
        var output = hiddenA
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        for _ in 0..<drafts {
            // Two queues, so the embedding has to land before the drafter's
            // command buffer is even encoded.
            guard let embedCB = embedQueue.makeCommandBuffer() else { throw MetalError.noDevice }
            embed.encode(commandBuffer: embedCB,
                         table: embedTable.buffer, tableOffset: Int(embedTable.offset),
                         scales: embedTable.buffer, scalesOffset: Int(embedTable.scaleOffset),
                         biases: embedTable.buffer, biasesOffset: Int(embedTable.biasOffset),
                         out: embedBuffer,
                         tokenId: UInt32(bitPattern: token),
                         d: UInt32(config.backboneHiddenSize),
                         outScale: embedScale)
            embedCB.commit()
            embedCB.waitUntilCompleted()
            if let error = embedCB.error { throw error }

            guard let cb = draftQueue.makeCommandBuffer() else { throw MetalError.noDevice }
            try drafter.encode(commandBuffer: cb,
                               targetEmbed: embedBuffer,
                               lastHidden: input,
                               slidingK: slidingK.buffer, slidingKOffset: slidingK.offset,
                               slidingV: slidingV.buffer, slidingVOffset: slidingV.offset,
                               fullK: fullK.buffer, fullKOffset: fullK.offset,
                               fullV: fullV.buffer, fullVOffset: fullV.offset,
                               slidingRingCapacity: UInt32(ringCapacity),
                               position: UInt32(ropePosition),
                               kvLength: UInt32(kvLength),
                               outLastHidden: output,
                               outToken: tokenBuffer)
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }

            token = Int32(bitPattern: tokenBuffer.contents().load(as: UInt32.self))
            proposals.append(token)
            input = output
            output = (output === hiddenA) ? hiddenB : hiddenA
        }
        draftNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start
        draftSteps += drafts
        return proposals
    }

    /// Feed the target's actual token to every round still waiting on it, and
    /// write out the rounds that are settled.
    private func record(actual token: Int32, at position: Int) {
        for index in pending.indices where !pending[index].broken {
            let next = pending[index].position + 1 + pending[index].actual.count
            guard next == position else { continue }
            pending[index].actual.append(token)
            if token != pending[index].proposals[pending[index].actual.count - 1] {
                pending[index].broken = true
            }
        }
        // `pending` is in position order, and a round settles at most `drafts`
        // steps after it opens, so draining from the front keeps the log in
        // position order without holding anything back for long.
        while let first = pending.first,
              first.broken || first.actual.count == first.proposals.count {
            write(first)
            pending.removeFirst()
        }
    }

    private func write(_ round: Round) {
        var accepted = 0
        while accepted < round.actual.count,
              round.actual[accepted] == round.proposals[accepted] {
            accepted += 1
        }
        rounds += 1
        acceptedHistogram[accepted] += 1
        let proposed = round.proposals.map(String.init).joined(separator: ",")
        let actual = round.actual.map(String.init).joined(separator: ",")
        handle.write(Data("\(round.position)\t\(round.bonus)\t\(proposed)\t\(actual)\t\(accepted)\n".utf8))
    }

    /// Summary block for the generation that just ended, then a fresh start:
    /// one probing process usually runs several prompts (`bench.sh ja` runs
    /// three), and mixing their rounds into one average would hide the
    /// per-prompt spread the go/no-go call needs.
    ///
    /// Rounds still waiting for target tokens when generation stopped are
    /// dropped: their acceptance is only partially known, and at most `drafts`
    /// of them exist.
    func flushSummary() {
        guard !finished, rounds > 0 || !pending.isEmpty else { return }
        let unresolved = pending.count
        pending.removeAll()
        var lines = """
            # --- summary ---
            # rounds=\(rounds) unresolved_dropped=\(unresolved)
            # accept histogram (0..\(drafts)): \
            \(acceptedHistogram.map(String.init).joined(separator: " "))\n
            """
        if rounds > 0 {
            for bs in 2...(drafts + 1) {
                let cap = bs - 1
                var total = 0
                for (accepted, count) in acceptedHistogram.enumerated() {
                    total += min(accepted, cap) * count
                }
                let mean = Double(total) / Double(rounds)
                let tokensPerRound = 1.0 + mean
                lines += String(format: "# bs=%d mean_accept=%.4f tokens_per_round=%.4f\n",
                                bs, mean, tokensPerRound)
            }
        }
        if draftSteps > 0 {
            lines += String(format: "# draft_ms=%.3f steps=%d\n",
                            Double(draftNanos) / Double(draftSteps) / 1e6, draftSteps)
        }
        handle.write(Data(lines.utf8))
        rounds = 0
        acceptedHistogram = [Int](repeating: 0, count: drafts + 1)
        draftSteps = 0
        draftNanos = 0
        // The next generation restarts the position counter, so the hidden held
        // here belongs to another sequence.
        prevHiddenPosition = -1
    }

    func finish() {
        guard !finished else { return }
        flushSummary()
        finished = true
        try? handle.close()
    }

    deinit { finish() }
}
