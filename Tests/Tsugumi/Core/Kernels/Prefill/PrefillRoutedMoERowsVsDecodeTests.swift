import Foundation
import Metal
import Testing
@testable import Tsugumi
import TsugumiValidationSupport

/// The routed MoE rows path against the *decode* routed MoE kernels.
///
/// This is the claim M4.5 rests on: a speculative verify block is decode
/// generalized to k rows, not a second implementation of it
/// (docs/mtp/16-M4.5-PLAN.md §0, §4 a). One row through the rows path has to
/// come out of the arithmetic bit for bit the way `moe_phase1_gate_up_act_u16load`
/// and the GEMV inside `moe_phase2_down_reduce_k8` come out — anything else and
/// the two paths have drifted, and the M4 argmax agreement is being carried by
/// tolerance rather than by construction.
@Suite struct PrefillRoutedMoERowsVsDecodeTests {
    /// Both reductions are exercised: at either affine group size these shapes
    /// span more than one 128-byte vectorized block and leave a scalar tail.
    private static let d = 576
    private static let f = 384
    private static let topK = 8

    @Test func rowsPathMatchesDecodeBitForBit() throws {
        let d = Self.d
        let f = Self.f
        let topK = Self.topK
        let ctx = try MetalContext()

        let pool = PrefillGroupedRoutedMoETests.makeSyntheticExpertPool(
            numExperts: topK, d: d, f: f)
        // Decode binds one buffer per expert at offset 0.
        let expertBuffers: [MTLBuffer] = try (0..<topK).map { expert in
            let start = expert * pool.stride
            let bytes = Array(pool.bytes[start..<(start + pool.stride)])
            guard let buffer = ctx.device.makeBuffer(bytes: bytes,
                                                     length: bytes.count,
                                                     options: .storageModeShared) else {
                throw PrefillGroupedRoutedMoEError.allocationFailed("expert \(expert)")
            }
            return buffer
        }

        var rng = SeedTree(0x4D5).key("rows-vs-decode")
        let x = (0..<d).map { _ in Float16(rng.uniform(-0.8, 0.8)) }

        let decode = try MoE(context: ctx)
        // Weight 1 on slot 0 and 0 elsewhere leaves `y = down(act_0)` in the
        // reduction, so the raw per-slot GEMV result is readable through the
        // decode kernel that never exposes it directly.
        let routingWeights = (0..<topK).map { Float16($0 == 0 ? 1 : 0) }
        guard let xBuffer = Fp16Buffer.make(ctx.device, halves: x),
              let decodeActs = Fp16Buffer.make(ctx.device, count: topK * f),
              let decodeOut = Fp16Buffer.make(ctx.device, count: d),
              let residual = Fp16Buffer.make(
                ctx.device, halves: [Float16](repeating: 0, count: d)),
              let routing = Fp16Buffer.make(ctx.device, halves: routingWeights),
              let decodeArgs = decode.makeRoutedArgumentBuffer(routedBlobs: expertBuffers,
                                                              topK: UInt32(topK)),
              let decodeCB = ctx.queue.makeCommandBuffer() else {
            Issue.record("decode allocation failed")
            return
        }
        decode.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: decodeCB,
            routedArgBuffer: decodeArgs,
            routedBlobs: expertBuffers,
            routedOffsets: pool.offsets,
            x: xBuffer,
            acts: decodeActs,
            d: UInt32(d), f: UInt32(f), topK: UInt32(topK))
        decode.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: decodeCB,
            routedArgBuffer: decodeArgs,
            routedBlobs: expertBuffers,
            routedOffsets: pool.offsets,
            acts: decodeActs,
            routingWeights: routing,
            residual: residual,
            y: decodeOut,
            d: UInt32(d), f: UInt32(f), topK: UInt32(topK))
        decodeCB.commit()
        decodeCB.waitUntilCompleted()
        if let error = decodeCB.error { throw error }

        // One token routed to all eight experts: the rows path sees one row per
        // expert, which is what a k=1 verify block gives it.
        let pairs = (0..<topK).map { rank in
            PrefillTokenExpertPair(token: 0,
                                   expert: UInt32(rank),
                                   rank: UInt32(rank),
                                   weight: Float16(1))
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: 1, topK: topK, numExperts: topK, tileExpertCount: 16)
        try #require(routes.tiles.count == 1)
        let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: 0, routes: routes)
        try #require(expertIDs == Array(0..<topK))

        let grouped = try PrefillGroupedRoutedMoE(context: ctx)
        try #require(grouped.usesExpertRowsPath(maxPairsPerExpert: routes.maxPairsPerExpert))
        let views = expertIDs.map { expert in
            TensorView(buffer: expertBuffers[expert],
                       offset: 0,
                       length: UInt64(pool.stride),
                       scaleOffset: 0, scaleLength: 0,
                       biasOffset: 0, biasLength: 0,
                       shape: (0, UInt32(expert), 0, 0),
                       dtype: 0)
        }
        let binding = try PrefillStreamedTileBinding(expertIDs: expertIDs, views: views)
        let params = PrefillGroupedRoutedMoEStreamedParams(
            pairStart: 0,
            pairCount: UInt32(routes.sortedPairs.count),
            d: UInt32(d),
            routedIntermediate: UInt32(f),
            topK: UInt32(topK),
            hiddenStrideElements: UInt32(d),
            binding: binding,
            offsets: pool.offsets)
        let argumentBuffer = try grouped.makeStreamedArgumentBuffer(device: ctx.device,
                                                                   binding: binding)
        guard let rowsAct = Fp16Buffer.make(ctx.device, count: topK * f),
              let rowsPartials = Fp16Buffer.make(ctx.device, count: topK * d),
              let rowsCB = ctx.queue.makeCommandBuffer() else {
            Issue.record("rows allocation failed")
            return
        }
        let blocks = grouped.encodeStreamedRows(
            commandBuffer: rowsCB,
            hidden: xBuffer,
            sortedPairs: try #require(ctx.device.makeBuffer(
                bytes: routes.sortedPairs,
                length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
                options: .storageModeShared)),
            routePartials: rowsPartials,
            gateUpActScratch: rowsAct,
            argumentBuffer: argumentBuffer,
            binding: binding,
            groups: routes.groups,
            params: params,
            maxRows: routes.sortedPairs.count)
        #expect(blocks == topK)
        rowsCB.commit()
        rowsCB.waitUntilCompleted()
        if let error = rowsCB.error { throw error }

        // Phase 1: gate, up and the non-linearity.
        let decodeActValues = Fp16Buffer.readHalf(decodeActs, count: topK * f)
        let rowsActValues = Fp16Buffer.readHalf(rowsAct, count: topK * f)
        var actMismatches = 0
        for slot in 0..<topK {
            for column in 0..<f where
                decodeActValues[slot * f + column] != rowsActValues[slot * f + column] {
                actMismatches += 1
            }
        }
        #expect(actMismatches == 0, "\(actMismatches) of \(topK * f) activations differ")

        // Phase 2: the down GEMV, read through the slot the routing weight kept.
        let decodeDown = Fp16Buffer.readHalf(decodeOut, count: d)
        let rowsDown = Fp16Buffer.readHalf(rowsPartials, count: topK * d)
        var downMismatches = 0
        for column in 0..<d where decodeDown[column] != rowsDown[column] {
            downMismatches += 1
        }
        #expect(downMismatches == 0, "\(downMismatches) of \(d) down outputs differ")
    }
}
