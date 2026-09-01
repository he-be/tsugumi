import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// Tests `KVCacheManager` FP16 shape, growth, separate K/V storage, ring,
/// and reset semantics against the Gemma 4 config.
@Suite struct KVCacheManagerTests {

    private let config = ArchConfig.gemma4_26B_A4B

    private func makeManager(maxContext: Int,
                             fp16RingEnabled: Bool = false,
                             maxPrefillChunkTokens: Int = 128,
                             maxSpeculativeBlockTokens: Int = 0,
                             fp16RingCapacityOverride: Int? = nil) throws -> (MetalContext, KVCacheManager) {
        let ctx = try MetalContext()
        let kv = try KVCacheManager(device: ctx.device,
                                    config: config,
                                    maxContext: maxContext,
                                    fp16RingEnabled: fp16RingEnabled,
                                    slidingWindow: config.slidingWindow,
                                    maxPrefillChunkTokens: maxPrefillChunkTokens,
                                    maxSpeculativeBlockTokens: maxSpeculativeBlockTokens,
                                    fp16RingCapacityOverride: fp16RingCapacityOverride)
        return (ctx, kv)
    }


    @Test func strideAndBufferSizes_matchConfig() throws {
        let (_, kv) = try makeManager(maxContext: 128)

        // SWA: numKVHeads(8) * headDim(256) * 2 = 4096 B/token.
        // Full: numFullKVHeads(2) * fullHeadDim(512) * 2 = 2048 B/token.
        #expect(kv.kRange(layer: 0, start: 0, count: 1).stride == 8 * 256 * 2)
        #expect(kv.kRange(layer: 5, start: 0, count: 1).stride == 2 * 512 * 2)
        #expect(kv.keyBuffer(layer: 0, validTokenCount: 0).length == 128 * 4096)
        #expect(kv.keyBuffer(layer: 5, validTokenCount: 0).length == 128 * 2048)
    }

    @Test func linearGrowth_tracksAdvance() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        #expect(kv.position == 0)
        for n in 1...100 {
            kv.advance()
            #expect(kv.position == n)
        }
    }

    /// Full layers share the raw k_proj output, then diverge: K runs k_norm +
    /// RoPE while V runs no-scale v_norm without RoPE. They therefore require
    /// separate cache slots.
    @Test func fullLayer_separatesKAndVBuffers() throws {
        let (_, kv) = try makeManager(maxContext: 16)
        let k = kv.keyBuffer(layer: 5, validTokenCount: 0)
        let v = kv.valueBuffer(layer: 5, validTokenCount: 0)
        #expect(k !== v, "full-layer K and V must NOT alias")
        let ks = kv.kSlot(layer: 5, position: 3)
        let vs = kv.vSlot(layer: 5, position: 3)
        #expect(ks.buffer !== vs.buffer, "full-layer K/V slots must NOT alias")
        // Offsets are still per-position-strided in both buffers.
        #expect(ks.offset == vs.offset)
    }

    @Test func swaLayer_hasSeparateKVBuffers() throws {
        let (_, kv) = try makeManager(maxContext: 16)
        #expect(kv.keyBuffer(layer: 0, validTokenCount: 0)
                !== kv.valueBuffer(layer: 0, validTokenCount: 0))
    }

    @Test func slotOffsets_areLinear() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        #expect(kv.kSlot(layer: 0, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 0, position: 3).offset == 3 * 4096)
        #expect(kv.vSlot(layer: 5, position: 7).offset == 7 * 2048)
    }

    @Test func fp16Ring_capsSWALayersAndLeavesFullLayersLinear() throws {
        let (_, kv) = try makeManager(maxContext: 4096,
                                      fp16RingEnabled: true)

        #expect(kv.fp16RingEnabled)
        #expect(kv.capacity(layer: 0) == 1152)
        #expect(kv.ringCapacity(layer: 0) == 1152)
        #expect(kv.keyBuffer(layer: 0, validTokenCount: 0).length == 1152 * 4096)
        #expect(kv.capacity(layer: 5) == 4096)
        #expect(kv.ringCapacity(layer: 5) == 0)
        #expect(kv.keyBuffer(layer: 5, validTokenCount: 0).length == 4096 * 2048)
    }

    /// A wide prefill chunk has to fit inside the ring alongside the sliding
    /// window, so the ring grows with it — 2048-token chunks cost 3072 rows per
    /// SWA layer instead of 1152, which is the memory the chunk width buys its
    /// expert-I/O savings with.
    @Test func fp16Ring_widensWithThePrefillChunk() throws {
        let (_, kv) = try makeManager(maxContext: 4096,
                                      fp16RingEnabled: true,
                                      maxPrefillChunkTokens: 2048)

        #expect(kv.capacity(layer: 0) == 3072)
        #expect(kv.ringCapacity(layer: 0) == 3072)
        #expect(kv.keyBuffer(layer: 0, validTokenCount: 0).length == 3072 * 4096)
        // Full-attention layers are linear either way: the chunk width does not
        // touch them.
        #expect(kv.capacity(layer: 5) == 4096)
    }

    @Test func fp16Ring_shortSessionCapsSWAToMaxContext() throws {
        let (_, kv) = try makeManager(maxContext: 256,
                                      fp16RingEnabled: true)

        #expect(kv.fp16RingEnabled)
        #expect(kv.capacity(layer: 0) == 256)
        #expect(kv.ringCapacity(layer: 0) == 256)
        #expect(kv.keyBuffer(layer: 0, validTokenCount: 0).length == 256 * 4096)
        #expect(kv.capacity(layer: 5) == 256)
        #expect(kv.ringCapacity(layer: 5) == 0)
        #expect(kv.keyBuffer(layer: 5, validTokenCount: 0).length == 256 * 2048)
    }

    @Test func fp16Ring_slotOffsetsWrapOnlyForSWALayers() throws {
        let (_, kv) = try makeManager(maxContext: 128,
                                      fp16RingEnabled: true,
                                      fp16RingCapacityOverride: 32)

        #expect(kv.kSlot(layer: 0, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 0, position: 31).offset == 31 * 4096)
        #expect(kv.kSlot(layer: 0, position: 32).offset == 0)
        #expect(kv.vSlot(layer: 0, position: 35).offset == 3 * 4096)

        #expect(kv.kSlot(layer: 5, position: 35).offset == 35 * 2048)
        #expect(kv.vSlot(layer: 5, position: 35).offset == 35 * 2048)
    }

    @Test func fp16Ring_rangesMustNotWrap() throws {
        let (_, kv) = try makeManager(maxContext: 128,
                                      fp16RingEnabled: true,
                                      fp16RingCapacityOverride: 32)

        let k = kv.kRange(layer: 0, start: 28, count: 4)
        #expect(k.offset == 28 * 4096)
        let v = kv.vRange(layer: 0, start: 32, count: 3)
        #expect(v.offset == 0)
    }

    @Test func rangeSlotsHaveLinearOffsets() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        let swaStride = kv.kRange(layer: 0, start: 0, count: 1).stride
        let fullStride = kv.vRange(layer: 5, start: 0, count: 1).stride

        let k = kv.kRange(layer: 0, start: 7, count: 3)
        let v = kv.vRange(layer: 5, start: 11, count: 5)

        #expect(k.offset == 7 * swaStride)
        #expect(k.stride == swaStride)
        #expect(v.offset == 11 * fullStride)
        #expect(v.stride == fullStride)
        #expect(k.buffer === kv.keyBuffer(layer: 0, validTokenCount: 0))
        #expect(v.buffer === kv.valueBuffer(layer: 5, validTokenCount: 0))
    }

    @Test func advanceByCountTracksCursor() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        kv.advance(by: 31)
        #expect(kv.position == 31)
        kv.advance(by: 0)
        #expect(kv.position == 31)
        kv.advance()
        #expect(kv.position == 32)
    }

    /// The speculative-block headroom check is a precondition on the ring, not
    /// a request for more of it: the block writes into rows the prefill chunk
    /// already paid for. The narrowest chunk width the front ends accept (32)
    /// still clears the widest block (8).
    @Test func speculativeBlockHeadroom_doesNotChangeCapacity() throws {
        let (_, plain) = try makeManager(maxContext: 4096,
                                         fp16RingEnabled: true,
                                         maxPrefillChunkTokens: 32)
        let (_, speculative) = try makeManager(maxContext: 4096,
                                               fp16RingEnabled: true,
                                               maxPrefillChunkTokens: 32,
                                               maxSpeculativeBlockTokens: 8)
        #expect(plain.capacity(layer: 0) == 1056)
        #expect(speculative.capacity(layer: 0) == plain.capacity(layer: 0))
        #expect(speculative.capacity(layer: 0) >= config.slidingWindow + 8)
    }

    /// The speculative rewind (`docs/mtp/03-DESIGN.md` D4). Only the cursor
    /// moves; a row rewritten afterwards has to land on the same physical slot
    /// the discarded one used, which is what makes "write k, keep a" safe.
    @Test func rewind_movesOnlyTheCursor() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        kv.advance(by: 40)
        let slotBefore = kv.kSlot(layer: 0, position: 37)

        kv.rewind(to: 37)
        #expect(kv.position == 37)
        #expect(kv.kSlot(layer: 0, position: 37).offset == slotBefore.offset)
        #expect(kv.kSlot(layer: 0, position: 37).buffer === slotBefore.buffer)

        kv.advance(by: 3)
        #expect(kv.position == 40)
    }

    @Test func rewind_toSameOrZeroIsAllowed() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        kv.advance(by: 12)
        kv.rewind(to: 12)
        #expect(kv.position == 12)
        kv.rewind(to: 0)
        #expect(kv.position == 0)
    }

    /// Rewinding across a ring wrap has to keep the slot map intact too: the
    /// map is `position % capacity` with no state of its own, so the rewritten
    /// row reuses the wrapped slot rather than a fresh one.
    @Test func rewind_survivesARingWrap() throws {
        let (_, kv) = try makeManager(maxContext: 128,
                                      fp16RingEnabled: true,
                                      fp16RingCapacityOverride: 32)
        kv.advance(by: 36)
        #expect(kv.kSlot(layer: 0, position: 35).offset == 3 * 4096)
        kv.rewind(to: 34)
        #expect(kv.position == 34)
        #expect(kv.kSlot(layer: 0, position: 35).offset == 3 * 4096)
    }

    @Test func reset_clearsPosition() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        for _ in 0..<100 { kv.advance() }
        #expect(kv.position == 100)
        kv.reset()
        #expect(kv.position == 0)
        // Cursor reusable after reset.
        kv.advance()
        #expect(kv.position == 1)
    }

}
