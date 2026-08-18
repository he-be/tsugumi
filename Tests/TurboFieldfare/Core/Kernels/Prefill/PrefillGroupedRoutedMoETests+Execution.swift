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

  /// The rows path — the one a speculative verify block takes — against the
  /// same CPU reference, on a route set shaped like a real block: a few tokens,
  /// experts holding between one and `rows` route rows each.
  ///
  /// `d` and `f` are chosen so the reduction covers both halves of the kernel:
  /// at either affine group size 320 is more than one 128-byte vectorized block
  /// and leaves a scalar tail.
  @Test(arguments: [1, 2, 4, 8] as [Int])
  func streamedRowsMatchesReference(rows: Int) throws {
    let d = 320
    let f = 320
    let topK = 2
    let numExperts = 8
    // Token t routes to experts (t % 3) and (t % 2) + 3, so the widest expert
    // holds several rows and the narrowest holds one.
    var pairs: [PrefillTokenExpertPair] = []
    for token in 0..<rows {
      pairs.append(Self.pair(token: UInt32(token), expert: UInt32(token % 3), rank: 0))
      pairs.append(Self.pair(token: UInt32(token), expert: UInt32(token % 2 + 3), rank: 1))
    }
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      pairs,
      queryCount: rows,
      topK: topK,
      numExperts: numExperts,
      tileExpertCount: 16)
    try #require(routes.tiles.count == 1)
    try #require(routes.maxPairsPerExpert <= PrefillGroupedRoutedMoE.rowsMaxPerExpert)

    let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: 0, routes: routes)
    let pool = Self.makeSyntheticExpertPool(numExperts: numExperts, d: d, f: f)
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
    try #require(grouped.usesExpertRowsPath(maxPairsPerExpert: routes.maxPairsPerExpert),
                 "rows routed path unavailable; nothing under test")
    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let outputBuffer = Fp16Buffer.make(
        ctx.device,
        halves: [Float16](repeating: -77, count: rows * topK * d)),
      let activationScratch = ctx.device.makeBuffer(
        length: 3 * routes.sortedPairs.count * f * MemoryLayout<Float16>.stride,
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
    let blocks = grouped.encodeStreamedRows(
      commandBuffer: commandBuffer,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: outputBuffer,
      gateUpActScratch: activationScratch,
      argumentBuffer: argumentBuffer,
      binding: binding,
      groups: routes.groups,
      params: params,
      maxRows: routes.sortedPairs.count)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error { throw error }

    let actual = Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d)
    let maxAbsoluteError = zip(actual, expected).reduce(Float(0)) {
      max($0, abs(Float($1.0) - Float($1.1)))
    }
    #expect(blocks == routes.groups.count)
    #expect(maxAbsoluteError <= 0.0015, "rows=\(rows) maxAbsoluteError=\(maxAbsoluteError)")
    #expect(!actual.contains(Float16(-77)))
  }

  /// Rows and tiled at the production routed shape, both against the same CPU
  /// reference.
  ///
  /// The rows path exists to run a verify block in decode's arithmetic, so it
  /// must not be the looser of the two: the tiled path stages its dequantized
  /// weights as FP16 (one rounding per weight), while the rows path keeps the
  /// affine factoring in FP32 the way the decode GEMV does. If this ever
  /// inverts, the verify block is trading accuracy for speed and the M4
  /// argmax agreement is on borrowed time.
  @Test func streamedRowsIsNoLooserThanTiledAtProductionShape() throws {
    let d = 2816
    let f = 704
    let rows = 4
    let topK = 2
    let numExperts = 4
    var pairs: [PrefillTokenExpertPair] = []
    for token in 0..<rows {
      pairs.append(Self.pair(token: UInt32(token), expert: UInt32(token % 2), rank: 0))
      pairs.append(Self.pair(token: UInt32(token), expert: UInt32(token % 2 + 2), rank: 1))
    }
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      pairs, queryCount: rows, topK: topK, numExperts: numExperts, tileExpertCount: 16)
    try #require(routes.tiles.count == 1)

    let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: 0, routes: routes)
    let pool = Self.makeSyntheticExpertPool(numExperts: numExperts, d: d, f: f)
    let hidden = (0..<(rows * d)).map { i in Float16(Float((i % 23) - 11) * 0.01) }
    let expected = Self.cpuSyntheticRoutePartials(
      routes: routes, hidden: hidden, hiddenStride: d, pool: pool,
      topK: topK, d: d, f: f)

    let ctx = try MetalContext()
    let grouped = try PrefillGroupedRoutedMoE(context: ctx)
    try #require(grouped.usesExpertRowsPath(maxPairsPerExpert: routes.maxPairsPerExpert))
    try #require(grouped.usesExpertGEMMPath(d: d, f: f))

    let binding = try PrefillStreamedTileBinding(
      expertIDs: expertIDs,
      views: Self.streamedViewsWithNonzeroOffsets(
        device: ctx.device, pool: pool, expertIDs: expertIDs))
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
      device: ctx.device, binding: binding)

    func run(_ encode: (MTLCommandBuffer, MTLBuffer, MTLBuffer, MTLBuffer) -> Void) throws
      -> [Float16] {
      guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
        let pairBuffer = ctx.device.makeBuffer(
          bytes: routes.sortedPairs,
          length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
          options: .storageModeShared),
        let outputBuffer = Fp16Buffer.make(
          ctx.device, halves: [Float16](repeating: -77, count: rows * topK * d)),
        let commandBuffer = ctx.queue.makeCommandBuffer()
      else {
        Issue.record("allocation failed")
        return []
      }
      encode(commandBuffer, hiddenBuffer, pairBuffer, outputBuffer)
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      if let error = commandBuffer.error { throw error }
      return Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d)
    }

    var rowsScratchHolder: MTLBuffer?
    func scratchBuffer() throws -> MTLBuffer {
      if let rowsScratchHolder { return rowsScratchHolder }
      guard let buffer = ctx.device.makeBuffer(
              length: 3 * routes.sortedPairs.count * f * MemoryLayout<Float16>.stride,
              options: .storageModePrivate) else {
        throw PrefillGroupedRoutedMoEError.allocationFailed("scratch")
      }
      rowsScratchHolder = buffer
      return buffer
    }

    let rowsScratch = try scratchBuffer()
    let rowsActual = try run { cb, hiddenBuffer, pairBuffer, outputBuffer in
      _ = grouped.encodeStreamedRows(
        commandBuffer: cb, hidden: hiddenBuffer, sortedPairs: pairBuffer,
        routePartials: outputBuffer, gateUpActScratch: rowsScratch,
        argumentBuffer: argumentBuffer, binding: binding,
        groups: routes.groups, params: params,
        maxRows: routes.sortedPairs.count)
    }
    let tiledActual = try run { cb, hiddenBuffer, pairBuffer, outputBuffer in
      _ = grouped.encodeStreamedTiled(
        commandBuffer: cb, hidden: hiddenBuffer, sortedPairs: pairBuffer,
        routePartials: outputBuffer, gateUpActScratch: rowsScratch,
        argumentBuffer: argumentBuffer, binding: binding,
        groups: routes.groups, params: params,
        maxRowsPerBatch: 64)
    }

    func relativeError(_ actual: [Float16]) -> Float {
      var worst: Float = 0
      var norm: Float = 0
      for i in expected.indices {
        norm = max(norm, abs(Float(expected[i])))
        worst = max(worst, abs(Float(actual[i]) - Float(expected[i])))
      }
      return norm > 0 ? worst / norm : .infinity
    }

    let rowsError = relativeError(rowsActual)
    let tiledError = relativeError(tiledActual)
    #expect(rowsError <= tiledError,
            "rows=\(rowsError) tiled=\(tiledError) against the CPU reference")
    #expect(rowsError < 5e-3, "rows relative error \(rowsError)")
  }

  /// Several tiles of one layer, encoded the way the runner encodes them: one
  /// command buffer per tile, committed as it is built, with the next tile
  /// encoded before the previous one has completed.
  ///
  /// Every tile stages its activations in the same scratch region, so if
  /// consecutive command buffers on one queue could overlap, tile i+1's gate
  /// pass would land on top of the rows tile i's down pass is still reading.
  /// The runner relies on them not overlapping; this is that reliance written
  /// down, on the path a verify block takes.
  @Test func streamedRowsSurvivesPipelinedTiles() throws {
    let d = 704
    let f = 704
    let rows = 4
    let topK = 8
    let numExperts = 64
    // Four tokens x top-8, spread so every tile holds several experts and no
    // expert holds more than `rows` pairs: the production verify shape.
    var pairs: [PrefillTokenExpertPair] = []
    for token in 0..<rows {
      for rank in 0..<topK {
        pairs.append(Self.pair(token: UInt32(token),
                               expert: UInt32((token + rank * 3) % 24),
                               rank: UInt32(rank)))
      }
    }
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      pairs, queryCount: rows, topK: topK, numExperts: numExperts, tileExpertCount: 8)
    try #require(routes.tiles.count >= 3, "wanted several tiles, got \(routes.tiles.count)")
    try #require(routes.maxPairsPerExpert <= PrefillGroupedRoutedMoE.rowsMaxPerExpert)

    let pool = Self.makeSyntheticExpertPool(numExperts: numExperts, d: d, f: f)
    let hidden = (0..<(rows * d)).map { i in Float16(Float((i % 23) - 11) * 0.01) }
    let expected = Self.cpuSyntheticRoutePartials(
      routes: routes, hidden: hidden, hiddenStride: d, pool: pool,
      topK: topK, d: d, f: f)

    let ctx = try MetalContext()
    let grouped = try PrefillGroupedRoutedMoE(context: ctx)
    try #require(grouped.usesExpertRowsPath(maxPairsPerExpert: routes.maxPairsPerExpert))

    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let outputBuffer = Fp16Buffer.make(
        ctx.device, halves: [Float16](repeating: -77, count: rows * topK * d)),
      // One scratch for every tile, exactly as the chunk scratch is shared.
      let scratch = ctx.device.makeBuffer(
        length: 3 * routes.sortedPairs.count * f * MemoryLayout<Float16>.stride,
        options: .storageModePrivate)
    else {
      Issue.record("allocation failed")
      return
    }

    var inFlight: [MTLCommandBuffer] = []
    for tileIndex in routes.tiles.indices {
      let tile = routes.tiles[tileIndex]
      let groupStart = Int(tile.groupStart)
      let tileGroups = Array(routes.groups[groupStart..<(groupStart + Int(tile.groupCount))])
      let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex,
                                                              routes: routes)
      let binding = try PrefillStreamedTileBinding(
        expertIDs: expertIDs,
        views: Self.streamedViewsWithNonzeroOffsets(
          device: ctx.device, pool: pool, expertIDs: expertIDs))
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
        device: ctx.device, binding: binding)
      guard let commandBuffer = ctx.queue.makeCommandBuffer() else {
        Issue.record("command buffer allocation failed")
        return
      }
      let blocks = grouped.encodeStreamedRows(
        commandBuffer: commandBuffer,
        hidden: hiddenBuffer,
        sortedPairs: pairBuffer,
        routePartials: outputBuffer,
        gateUpActScratch: scratch,
        argumentBuffer: argumentBuffer,
        binding: binding,
        groups: tileGroups,
        params: params,
        maxRows: routes.sortedPairs.count)
      #expect(blocks == tileGroups.count)
      withExtendedLifetime((binding, argumentBuffer)) {
        commandBuffer.commit()
      }
      inFlight.append(commandBuffer)
      // The runner keeps one tile in flight while it builds the next.
      if inFlight.count > 1 { inFlight.removeFirst().waitUntilCompleted() }
    }
    for commandBuffer in inFlight { commandBuffer.waitUntilCompleted() }

    let actual = Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d)
    var worst: Float = 0
    var norm: Float = 0
    for i in expected.indices {
      norm = max(norm, abs(Float(expected[i])))
      worst = max(worst, abs(Float(actual[i]) - Float(expected[i])))
    }
    #expect(!actual.contains(Float16(-77)))
    #expect(worst / norm < 5e-3,
            "tiles=\(routes.tiles.count) relative error \(worst / norm)")
  }

  /// A tile with an expert wider than the row ceiling is not this path's, and
  /// it must decline rather than compute part of it.
  @Test func streamedRowsDeclinesExpertWiderThanRowCeiling() throws {
    let d = 64
    let f = 64
    let topK = 1
    let wide = PrefillGroupedRoutedMoE.rowsMaxPerExpert + 1
    let pairs = (0..<wide).map { token in
      Self.pair(token: UInt32(token), expert: 0, rank: 0)
    }
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      pairs,
      queryCount: wide,
      topK: topK,
      numExperts: 4,
      tileExpertCount: 16)

    let ctx = try MetalContext()
    let grouped = try PrefillGroupedRoutedMoE(context: ctx)
    #expect(!grouped.usesExpertRowsPath(maxPairsPerExpert: routes.maxPairsPerExpert))

    let pool = Self.makeSyntheticExpertPool(numExperts: 4, d: d, f: f)
    let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: 0, routes: routes)
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
    guard let hiddenBuffer = Fp16Buffer.make(
            ctx.device, halves: [Float16](repeating: 0, count: wide * d)),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let outputBuffer = Fp16Buffer.make(
        ctx.device, halves: [Float16](repeating: -77, count: wide * topK * d)),
      let scratch = ctx.device.makeBuffer(
        length: 3 * wide * f * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let commandBuffer = ctx.queue.makeCommandBuffer()
    else {
      Issue.record("allocation failed")
      return
    }
    let blocks = grouped.encodeStreamedRows(
      commandBuffer: commandBuffer,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: outputBuffer,
      gateUpActScratch: scratch,
      argumentBuffer: argumentBuffer,
      binding: binding,
      groups: routes.groups,
      params: params,
      maxRows: wide)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    #expect(blocks == 0)
    // Nothing encoded means nothing written.
    #expect(Fp16Buffer.readHalf(outputBuffer, count: wide * topK * d)
              .allSatisfy { $0 == Float16(-77) })
  }

  /// A tile whose rows do not fit the activation scratch is declined too — the
  /// staging is `[rows, F]` inside a buffer the chunk sized in advance.
  @Test func streamedRowsDeclinesRowsBeyondScratch() throws {
    let d = 64
    let f = 64
    let topK = 1
    let pairs = (0..<4).map { token in
      Self.pair(token: UInt32(token), expert: UInt32(token % 2), rank: 0)
    }
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      pairs, queryCount: 4, topK: topK, numExperts: 4, tileExpertCount: 16)

    let ctx = try MetalContext()
    let grouped = try PrefillGroupedRoutedMoE(context: ctx)
    let pool = Self.makeSyntheticExpertPool(numExperts: 4, d: d, f: f)
    let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: 0, routes: routes)
    let binding = try PrefillStreamedTileBinding(
      expertIDs: expertIDs,
      views: Self.streamedViewsWithNonzeroOffsets(
        device: ctx.device, pool: pool, expertIDs: expertIDs))
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
      device: ctx.device, binding: binding)
    guard let hiddenBuffer = Fp16Buffer.make(
            ctx.device, halves: [Float16](repeating: 0, count: 4 * d)),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let outputBuffer = Fp16Buffer.make(
        ctx.device, halves: [Float16](repeating: -77, count: 4 * topK * d)),
      let scratch = ctx.device.makeBuffer(
        length: 3 * 4 * f * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let commandBuffer = ctx.queue.makeCommandBuffer()
    else {
      Issue.record("allocation failed")
      return
    }
    let blocks = grouped.encodeStreamedRows(
      commandBuffer: commandBuffer,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: outputBuffer,
      gateUpActScratch: scratch,
      argumentBuffer: argumentBuffer,
      binding: binding,
      groups: routes.groups,
      params: params,
      maxRows: routes.sortedPairs.count - 1)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    #expect(blocks == 0)
  }

}
