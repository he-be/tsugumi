import Metal
import Testing
import TurboFieldfareValidationSupport

@testable import TurboFieldfare

extension PrefillGroupedRoutedMoETests {
  @Test func streamedBatchedMatchesReferenceAcrossPartialMicrobatch() throws {
    let d = 64
    let f = 64
    let rows = 3
    let topK = 2
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      [
        Self.pair(token: 0, expert: 2, rank: 0),
        Self.pair(token: 0, expert: 0, rank: 1),
        Self.pair(token: 1, expert: 1, rank: 0),
        Self.pair(token: 1, expert: 2, rank: 1),
        Self.pair(token: 2, expert: 0, rank: 0),
        Self.pair(token: 2, expert: 1, rank: 1),
      ],
      queryCount: rows,
      topK: topK,
      numExperts: 16,
      tileExpertCount: 16)
    let pool = Self.makeSyntheticExpertPool(numExperts: 16, d: d, f: f)
    let hidden = (0..<(rows * d)).map { i in
      Float16(Float((i % 17) - 8) * 0.01)
    }
    let expected = Self.cpuSyntheticRoutePartials(
      routes: routes,
      hidden: hidden,
      hiddenStride: d,
      pool: pool,
      topK: topK,
      d: d,
      f: f)

    let ctx = try MetalContext()
    let grouped = try PrefillGroupedRoutedMoE(context: ctx)
    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let outputBuffer = Fp16Buffer.make(
        ctx.device,
        halves: [Float16](repeating: -77, count: rows * topK * d)),
      let activationScratch = ctx.device.makeBuffer(
        length: 3 * 4 * f * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let downScratch = ctx.device.makeBuffer(
        length: 4 * d * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let commandBuffer = ctx.queue.makeCommandBuffer()
    else {
      Issue.record("allocation failed")
      return
    }

    let expertIDs = Array(0..<16)
    let binding = try PrefillStreamedTileBinding(
      expertIDs: expertIDs,
      views: Self.streamedViewsWithNonzeroOffsets(
        device: ctx.device,
        pool: pool,
        expertIDs: expertIDs))
    let params = PrefillGroupedRoutedMoEStreamedParams(
      pairStart: 0,
      pairCount: UInt32(routes.sortedPairs.count),
      d: UInt32(d),
      routedIntermediate: UInt32(f),
      topK: UInt32(topK),
      hiddenStrideElements: UInt32(d),
      binding: binding,
      offsets: pool.offsets)
    let argumentBuffer = try grouped.makeStreamedArgumentBuffer(
      device: ctx.device,
      binding: binding)
    let microbatches = grouped.encodeStreamedBatched(
      commandBuffer: commandBuffer,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: outputBuffer,
      gateUpActScratch: activationScratch,
      downScratch: downScratch,
      argumentBuffer: argumentBuffer,
      binding: binding,
      params: params,
      pairMicrobatchRows: 4)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error { throw error }

    let actual = Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d)
    let maxAbsoluteError = zip(actual, expected).reduce(Float(0)) {
      max($0, abs(Float($1.0) - Float($1.1)))
    }
    #expect(microbatches == 2)
    #expect(maxAbsoluteError <= 0.0015, "maxAbsoluteError=\(maxAbsoluteError)")
    #expect(binding.views.allSatisfy { $0.offset > 0 })
  }

  /// The tiled path against the same CPU reference, on a route set that makes
  /// the planner do all three things it can do: fill 64-row blocks, leave a
  /// partial tail block, and split one expert across batches.
  @Test func streamedTiledMatchesReferenceAcrossBatchesAndPartialBlocks() throws {
    let d = 64
    let f = 64
    let rows = 64
    let topK = 4
    let batchRows = 64
    let pairs = (0..<(rows * topK)).map { index in
      Self.pair(token: UInt32(index / topK),
                expert: UInt32(index % 3),
                rank: UInt32(index % topK))
    }
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      pairs,
      queryCount: rows,
      topK: topK,
      numExperts: 8,
      tileExpertCount: 16)
    #expect(routes.tiles.count == 1)

    let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: 0, routes: routes)
    let pool = Self.makeSyntheticExpertPool(numExperts: 8, d: d, f: f)
    let hidden = (0..<(rows * d)).map { i in
      Float16(Float((i % 17) - 8) * 0.01)
    }
    let expected = Self.cpuSyntheticRoutePartials(
      routes: routes,
      hidden: hidden,
      hiddenStride: d,
      pool: pool,
      topK: topK,
      d: d,
      f: f)

    let ctx = try MetalContext()
    let grouped = try PrefillGroupedRoutedMoE(context: ctx)
    try #require(grouped.usesExpertGEMMPath(d: d, f: f),
                 "tiled routed path unavailable; nothing under test")
    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let outputBuffer = Fp16Buffer.make(
        ctx.device,
        halves: [Float16](repeating: -77, count: rows * topK * d)),
      let activationScratch = ctx.device.makeBuffer(
        length: 3 * batchRows * f * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let commandBuffer = ctx.queue.makeCommandBuffer()
    else {
      Issue.record("allocation failed")
      return
    }

    let binding = try PrefillStreamedTileBinding(
      expertIDs: expertIDs,
      views: Self.streamedViewsWithNonzeroOffsets(
        device: ctx.device,
        pool: pool,
        expertIDs: expertIDs))
    let params = PrefillGroupedRoutedMoEStreamedParams(
      pairStart: 0,
      pairCount: UInt32(routes.sortedPairs.count),
      d: UInt32(d),
      routedIntermediate: UInt32(f),
      topK: UInt32(topK),
      hiddenStrideElements: UInt32(d),
      binding: binding,
      offsets: pool.offsets)
    let argumentBuffer = try grouped.makeStreamedArgumentBuffer(
      device: ctx.device,
      binding: binding)
    let batches = grouped.encodeStreamedTiled(
      commandBuffer: commandBuffer,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: outputBuffer,
      gateUpActScratch: activationScratch,
      argumentBuffer: argumentBuffer,
      binding: binding,
      groups: routes.groups,
      params: params,
      maxRowsPerBatch: batchRows)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error { throw error }

    let actual = Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d)
    let maxAbsoluteError = zip(actual, expected).reduce(Float(0)) {
      max($0, abs(Float($1.0) - Float($1.1)))
    }
    #expect(batches > 3, "planner produced \(batches) batches; wanted several")
    #expect(maxAbsoluteError <= 0.0015, "maxAbsoluteError=\(maxAbsoluteError)")
    // Every pair must have been written; -77 would mean a skipped route.
    #expect(!actual.contains(Float16(-77)))
  }

}
