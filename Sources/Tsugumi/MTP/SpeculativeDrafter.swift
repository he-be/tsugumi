import Foundation
import Metal

/// The production drafter: `bs - 1` greedy proposals per speculative round.
///
/// Same arithmetic as `DraftAcceptanceProbe.draftRound` — the M3.5 measurement
/// and the M5 loop have to run the drafter identically or the measured
/// acceptance lengths say nothing about the loop — with the probe's bookkeeping
/// (round log, dumps, hidden ping-pong across decode steps) removed. What is
/// left is one function: from `(bonus token, the target hidden that produced
/// it, the target's shared K/V)`, walk `count` steps.
///
/// The input convention is the one 14-M3.5 §6 settled on, and it is llama.cpp's
/// `draft-mtp`: the hidden of the position *before* the bonus token, RoPE at the
/// bonus token's own position, K/V rows `[0, position)`, and the position held
/// constant across every step of the round (the drafter writes no K/V, so there
/// is nothing new for a later step to attend to).
final class SpeculativeDrafter {
    private let drafter: DraftForward
    private let config: DraftConfig
    /// The bonus token's embedding comes from the *target's* table, so the
    /// lookup runs on the target's context (group 32) while the drafter runs on
    /// its own (group 64) — see `MetalContext.init(sharingDeviceWith:)`.
    private let embed: EmbedLookupInt4
    private let embedTable: TensorView
    private let embedQueue: MTLCommandQueue
    private let draftQueue: MTLCommandQueue
    private let embedScale: Float

    // Ping-pong so a step's `post_projection` output feeds the next step
    // without aliasing the buffer it is reading from.
    private let hiddenA: MTLBuffer
    private let hiddenB: MTLBuffer
    private let embedBuffer: MTLBuffer
    private let tokenBuffer: MTLBuffer

    private(set) var steps = 0
    private(set) var nanos: UInt64 = 0

    init(context: MetalContext,
         draftContext: MetalContext,
         weights: DraftWeights,
         embedTable: TensorView,
         backboneHiddenSize: Int) throws {
        self.drafter = try DraftForward(context: draftContext, weights: weights)
        self.config = weights.config
        self.embed = try EmbedLookupInt4(context: context)
        self.embedTable = embedTable
        self.embedQueue = context.queue
        self.draftQueue = draftContext.queue
        self.embedScale = Float(backboneHiddenSize).squareRoot()

        let device = context.device
        func buffer(_ count: Int, shared: Bool = false) throws -> MTLBuffer {
            guard let b = device.makeBuffer(
                length: max(count * MemoryLayout<Float16>.size, 4),
                options: shared ? .storageModeShared : .storageModePrivate) else {
                throw MetalError.noDevice
            }
            return b
        }
        self.hiddenA = try buffer(backboneHiddenSize)
        self.hiddenB = try buffer(backboneHiddenSize)
        self.embedBuffer = try buffer(backboneHiddenSize)
        self.tokenBuffer = try buffer(2, shared: true)
    }

    /// `count` greedy proposals for the positions after `position`.
    ///
    /// - Parameters:
    ///   - bonusToken: the token that will occupy `position`; the target has not
    ///     run a forward on it yet, so it has no K/V row.
    ///   - hidden: the target's post-norm final hidden for the row that
    ///     *produced* `bonusToken`, at `hiddenRow * backboneHidden` elements in.
    ///   - position: absolute position of the bonus token, which is also the
    ///     number of K/V rows the drafter attends to.
    func propose(bonusToken: Int32,
                 position: Int,
                 hidden: MTLBuffer,
                 hiddenRow: Int,
                 count: Int,
                 kv: KVCacheManager) throws -> [Int32] {
        guard count > 0 else { return [] }
        let slidingK = kv.keyView(layer: config.sharedSlidingKVLayer)
        let slidingV = kv.valueView(layer: config.sharedSlidingKVLayer)
        let fullK = kv.keyView(layer: config.sharedFullKVLayer)
        let fullV = kv.valueView(layer: config.sharedFullKVLayer)
        let ringCapacity = kv.ringCapacity(layer: config.sharedSlidingKVLayer)
        let kvLength = max(1, position)

        var proposals: [Int32] = []
        proposals.reserveCapacity(count)
        var token = bonusToken
        var input = hidden
        var inputElementOffset = hiddenRow * config.backboneHiddenSize
        var output = hiddenA
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        for _ in 0..<count {
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
                               lastHiddenOffset: inputElementOffset,
                               slidingK: slidingK.buffer, slidingKOffset: slidingK.offset,
                               slidingV: slidingV.buffer, slidingVOffset: slidingV.offset,
                               fullK: fullK.buffer, fullKOffset: fullK.offset,
                               fullV: fullV.buffer, fullVOffset: fullV.offset,
                               slidingRingCapacity: UInt32(ringCapacity),
                               position: UInt32(position),
                               kvLength: UInt32(kvLength),
                               outLastHidden: output,
                               outToken: tokenBuffer)
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }

            token = Int32(bitPattern: tokenBuffer.contents().load(as: UInt32.self))
            proposals.append(token)
            input = output
            inputElementOffset = 0
            output = (output === hiddenA) ? hiddenB : hiddenA
        }
        nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start
        steps += count
        return proposals
    }
}
