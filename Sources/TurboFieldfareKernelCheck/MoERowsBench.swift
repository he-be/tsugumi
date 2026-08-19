import Foundation
import Metal
import TurboFieldfare

// MARK: - Routed MoE rows kernels, off the block timeline
//
// `docs/mtp/28-M8-PROPOSAL.md` §2 fits three whole-pipeline points and gets a
// marginal cost of about 37 us per route row against about 27.5 us for reading
// one expert's weights outright: the row loop is not sharing the weight read the
// way the dense rows kernel does (`docs/mtp/20-M4.8-RESULTS.md` §3, "k=4 costs
// what k=1 costs").
//
// `docs/mtp/27-M7-RESULTS.md` §6 then cut the row loop three ways -- one 32-bit
// weight load instead of two 16-bit, two output rows per SIMD group, the nibble
// unpack hoisted out of the row loop -- and read "unchanged" on all three. But
// that was read off the verify block's GPU busy at 48 slots, which is 47-48 ms
// with expert I/O on the same timeline: a 1 ms move is inside the noise, and
// §7 of the same document shows I/O is the floor there anyway.
//
// This bench takes the two kernels off that timeline. Weights are synthetic and
// already resident, one tile of 16 experts, no cache and no I/O, `iterations`
// encodings inside one command buffer, and -- unlike the block profile -- gate/up
// and down are timed separately. What it reports is the slope in r, which is the
// quantity 28 §2 derived indirectly.
//
// Production routed shapes (`ArchConfig.gemma4_26B_A4B`): D = 2816, F = 704,
// affine group 32. Note the asymmetry that falls out of those numbers: gate/up
// reduces over D = 2816 = 88 groups = 11 whole vectorized blocks, while down
// reduces over F = 704 = 22 groups = 2 whole blocks and a 6-group scalar tail
// that runs on half the SIMD group's lanes. None of 27 §6's three cuts touched
// that tail.

private let moeRowsD = 2816
private let moeRowsF = 704
private let moeRowsTopK = 8
private let moeRowsTileExperts = 16
private let moeRowsMaxPerExpert = 8

/// Byte offsets inside one expert blob, laid out the way the packer does:
/// gate (packed, scales, biases), then up, then down.
private struct MoEBlobLayout {
    var gateWOff = 0, gateSOff = 0, gateBOff = 0
    var upWOff = 0, upSOff = 0, upBOff = 0
    var downWOff = 0, downSOff = 0, downBOff = 0
    var stride = 0

    init(d: Int, f: Int, groupSize: Int) {
        // gate and up are [F, D]; down is [D, F].
        func advance(_ cursor: inout Int, rows: Int, cols: Int) -> (Int, Int, Int) {
            let w = cursor
            cursor += rows * cols / 2
            let s = cursor
            cursor += rows * (cols / groupSize) * 2
            let b = cursor
            cursor += rows * (cols / groupSize) * 2
            return (w, s, b)
        }
        var cursor = 0
        (gateWOff, gateSOff, gateBOff) = advance(&cursor, rows: f, cols: d)
        (upWOff, upSOff, upBOff) = advance(&cursor, rows: f, cols: d)
        (downWOff, downSOff, downBOff) = advance(&cursor, rows: d, cols: f)
        stride = cursor
    }
}

/// `PrefillGroupedRoutedMoEStreamedParamsMSL` (`prefill.metal`), built here so
/// the bench does not depend on the runtime's binding types.
private struct MoERowsParams {
    var pairStart: UInt32 = 0
    var pairCount: UInt32 = 0
    var d: UInt32 = 0
    var f: UInt32 = 0
    var topK: UInt32 = 0
    var hiddenStrideElements: UInt32 = 0
    var liveExpertCount: UInt32 = 0
    var localExperts: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                       UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) =
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    var gateWOff: UInt32 = 0
    var gateSOff: UInt32 = 0
    var gateBOff: UInt32 = 0
    var upWOff: UInt32 = 0
    var upSOff: UInt32 = 0
    var upBOff: UInt32 = 0
    var downWOff: UInt32 = 0
    var downSOff: UInt32 = 0
    var downBOff: UInt32 = 0
}

/// `PrefillRoutedGEMMBlockMSL`.
private struct MoERowsBlock {
    var localSlot: UInt32
    var pairStart: UInt32
    var rowCount: UInt32
    var localRow: UInt32
}

/// `PrefillTokenExpertPairMSL`.
private struct MoERowsPair {
    var token: UInt32
    var expert: UInt32
    var rank: UInt32
    var weightBits: UInt32
}

/// One expert blob of the right size, filled with a cheap pattern.
///
/// Timing does not depend on the nibble values; the scales are kept small and
/// positive so the accumulator stays finite and the GPU never takes a denormal
/// or NaN path that the real weights would not.
private func syntheticExpertBlob(device: MTLDevice, layout: MoEBlobLayout)
    -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: layout.stride,
                                         options: .storageModeShared) else {
        fatalError("expert blob allocation failed (\(layout.stride) bytes)")
    }
    let bytes = buffer.contents().bindMemory(to: UInt8.self, capacity: layout.stride)
    var state: UInt64 = 0x9E3779B97F4A7C15
    for index in 0..<layout.stride {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        bytes[index] = UInt8(truncatingIfNeeded: state >> 33)
    }
    // Overwrite the affine regions with BF16 0.0078125 / 0.001953125.
    let words = buffer.contents().bindMemory(to: UInt16.self,
                                             capacity: layout.stride / 2)
    func fill(_ start: Int, _ end: Int, _ value: UInt16) {
        for word in (start / 2)..<(end / 2) { words[word] = value }
    }
    fill(layout.gateSOff, layout.gateBOff, 0x3C00)
    fill(layout.gateBOff, layout.upWOff, 0x3B00)
    fill(layout.upSOff, layout.upBOff, 0x3C00)
    fill(layout.upBOff, layout.downWOff, 0x3B00)
    fill(layout.downSOff, layout.downBOff, 0x3C00)
    fill(layout.downBOff, layout.stride, 0x3B00)
    return buffer
}

private func fp16Buffer(device: MTLDevice, count: Int, fill: Bool) -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: count * 2,
                                         options: .storageModeShared) else {
        fatalError("fp16 allocation failed (\(count) halves)")
    }
    if fill {
        let halves = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        for index in 0..<count {
            halves[index] = Float16(Float(index % 61) * 0.01 - 0.3)
        }
    }
    return buffer
}

/// Which of the two dispatches to encode. The block profile only ever reports
/// their sum, and 28 §2's fit cannot separate them.
private enum MoERowsStage: String, CaseIterable {
    case gateUp = "gate/up"
    case down = "down"
}

/// One tile's worth of state, reused across the r sweep.
private struct MoERowsFixture {
    let layout: MoEBlobLayout
    let blobs: [MTLBuffer]
    let argumentBuffer: MTLBuffer
    let hidden: MTLBuffer
    let act: MTLBuffer
    let partials: MTLBuffer
    let pairs: MTLBuffer
    let params: MoERowsParams
}

private func makeMoERowsFixture(context: MetalContext, groupSize: Int) throws
    -> MoERowsFixture {
    let layout = MoEBlobLayout(d: moeRowsD, f: moeRowsF, groupSize: groupSize)
    let blobs = (0..<moeRowsTileExperts).map { _ in
        syntheticExpertBlob(device: context.device, layout: layout)
    }

    guard let function = try context.library.makeFunction(
        name: "prefill_moe_rows_gate_up_act") else {
        fatalError("prefill_moe_rows_gate_up_act missing from the library")
    }
    let encoder = function.makeArgumentEncoder(bufferIndex: 9)
    guard let argumentBuffer = context.device.makeBuffer(
        length: encoder.encodedLength, options: .storageModeShared) else {
        fatalError("argument buffer allocation failed")
    }
    encoder.setArgumentBuffer(argumentBuffer, offset: 0)
    for (index, blob) in blobs.enumerated() {
        encoder.setBuffer(blob, offset: 0, index: index)
    }

    // The widest sweep point: 16 experts x 8 rows, one token per row.
    let maxRows = moeRowsTileExperts * moeRowsMaxPerExpert
    var pairs: [MoERowsPair] = []
    pairs.reserveCapacity(maxRows)
    for expert in 0..<moeRowsTileExperts {
        for row in 0..<moeRowsMaxPerExpert {
            pairs.append(MoERowsPair(token: UInt32(row),
                                     expert: UInt32(expert),
                                     rank: UInt32(row),
                                     weightBits: 0))
        }
    }
    guard let pairBuffer = context.device.makeBuffer(
        bytes: pairs,
        length: pairs.count * MemoryLayout<MoERowsPair>.stride,
        options: .storageModeShared) else {
        fatalError("sorted-pair allocation failed")
    }

    var params = MoERowsParams()
    params.d = UInt32(moeRowsD)
    params.f = UInt32(moeRowsF)
    params.topK = UInt32(moeRowsTopK)
    params.hiddenStrideElements = UInt32(moeRowsD)
    params.liveExpertCount = UInt32(moeRowsTileExperts)
    params.gateWOff = UInt32(layout.gateWOff)
    params.gateSOff = UInt32(layout.gateSOff)
    params.gateBOff = UInt32(layout.gateBOff)
    params.upWOff = UInt32(layout.upWOff)
    params.upSOff = UInt32(layout.upSOff)
    params.upBOff = UInt32(layout.upBOff)
    params.downWOff = UInt32(layout.downWOff)
    params.downSOff = UInt32(layout.downSOff)
    params.downBOff = UInt32(layout.downBOff)

    return MoERowsFixture(
        layout: layout,
        blobs: blobs,
        argumentBuffer: argumentBuffer,
        hidden: fp16Buffer(device: context.device,
                           count: moeRowsMaxPerExpert * moeRowsD, fill: true),
        act: fp16Buffer(device: context.device, count: maxRows * moeRowsF, fill: false),
        // down scatters to (token * top_k + rank) * D.
        partials: fp16Buffer(device: context.device,
                             count: moeRowsMaxPerExpert * moeRowsTopK * moeRowsD,
                             fill: false),
        pairs: pairBuffer,
        params: params)
}

/// One dispatch of one stage over the experts `select` keeps.
///
/// `rowCounts` is per expert, so a tile can carry the row mixture production
/// actually sees instead of a uniform `r`. `local_row` is the prefix sum over
/// *all* experts, not just the selected ones: splitting a tile across two
/// dispatches must not move where a block writes, which is exactly the
/// invariant `encodeStreamedRows` relies on (31 §7).
private func encodeMoERows(_ stage: MoERowsStage,
                           pso: MTLComputePipelineState,
                           commandBuffer: MTLCommandBuffer,
                           fixture: MoERowsFixture,
                           rowCounts: [Int],
                           select: (Int) -> Bool = { _ in true }) {
    var blocks: [MoERowsBlock] = []
    blocks.reserveCapacity(rowCounts.count)
    var localRow = 0
    for expert in rowCounts.indices {
        let count = rowCounts[expert]
        if select(count) {
            blocks.append(MoERowsBlock(
                localSlot: UInt32(expert),
                // Pairs are laid out at the widest stride so one buffer serves
                // every sweep point; a block takes the first `count` of its own.
                pairStart: UInt32(expert * moeRowsMaxPerExpert),
                rowCount: UInt32(count),
                localRow: UInt32(localRow)))
        }
        localRow += count
    }
    guard !blocks.isEmpty else { return }
    var params = fixture.params
    params.pairCount = UInt32(localRow)

    guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
    enc.setComputePipelineState(pso)
    if stage == .gateUp {
        enc.setBuffer(fixture.hidden, offset: 0, index: 0)
    }
    enc.setBuffer(fixture.pairs, offset: 0, index: 1)
    enc.setBytes(&blocks, length: blocks.count * MemoryLayout<MoERowsBlock>.stride,
                 index: 2)
    if stage == .down {
        enc.setBuffer(fixture.partials, offset: 0, index: 5)
    }
    enc.setBuffer(fixture.act, offset: 0, index: 7)
    enc.setBuffer(fixture.argumentBuffer, offset: 0, index: 9)
    enc.setBytes(&params, length: MemoryLayout<MoERowsParams>.stride, index: 10)
    for blob in fixture.blobs { enc.useResource(blob, usage: .read) }

    let outputRows = stage == .gateUp ? moeRowsF : moeRowsD
    let tiles = (outputRows + 7) / 8
    enc.dispatchThreadgroups(
        MTLSize(width: tiles, height: blocks.count, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32 * 8, height: 1, depth: 1))
    enc.endEncoding()
}

/// The uniform sweep point: `rowsPerExpert` rows on all sixteen experts.
private func encodeMoERows(_ stage: MoERowsStage,
                           pso: MTLComputePipelineState,
                           commandBuffer: MTLCommandBuffer,
                           fixture: MoERowsFixture,
                           rowsPerExpert: Int) {
    encodeMoERows(stage, pso: pso, commandBuffer: commandBuffer, fixture: fixture,
                  rowCounts: [Int](repeating: rowsPerExpert,
                                   count: moeRowsTileExperts))
}

/// `--moe-rows-bench`: the routed MoE rows kernels at the production shapes.
func runMoERowsBench(groupSize: Int, iterations: Int) throws {
    let context = try makeContext(groupSize: groupSize)
    let fixture = try makeMoERowsFixture(context: context, groupSize: groupSize)

    // Production picks a pipeline specialized to the tile's widest block
    // (`PrefillGroupedRoutedMoE.encodeStreamedRows`), so `cap = r` is the shipped
    // configuration. `cap = 8` is the same kernel with the row arrays sized for
    // eight rows it will not use, which is what the register-pressure half of
    // 28 §2's hypothesis predicts should be slower.
    func pipeline(_ name: String, cap: Int) throws -> MTLComputePipelineState {
        try context.pipeline(name, constants: [
            MetalFunctionConstant(index: 16, value: .uint32(UInt32(cap)))
        ])
    }

    let expertsPerBlock = 558.0  // 27 §6, bs=4, 48 slots
    print("=== routed MoE rows kernels "
            + "(D=\(moeRowsD) F=\(moeRowsF) group \(groupSize), "
            + "\(moeRowsTileExperts) experts/tile, \(iterations) iterations) ===")
    print("us per dispatch of one 16-expert tile; per-expert divides by 16.")
    print("`ms/blk` scales one expert to the \(Int(expertsPerBlock)) a bs=4 verify")
    print("block touches at 48 slots, for comparison with 27 §6's `moe` column.")

    for specialized in [true, false] {
        print("")
        print(specialized ? "  cap = r (shipped)" : "  cap = 8 (unspecialized)")
        var header = "    stage      "
        for r in 1...8 { header += String(format: "     r=%d", r) }
        header += "   | us/row"
        print(header)
        var totals = [Double](repeating: 0, count: 8)
        for stage in MoERowsStage.allCases {
            let name = stage == .gateUp
                ? "prefill_moe_rows_gate_up_act" : "prefill_moe_rows_down"
            var line = String(format: "    %-10@", stage.rawValue as NSString)
            var perRow: [Double] = []
            for r in 1...8 {
                let pso = try pipeline(name, cap: specialized ? r : 8)
                let seconds = try gpuSecondsThrowing(
                    context: context, iterations: iterations,
                    label: "\(stage.rawValue) r=\(r)") { cmd in
                    encodeMoERows(stage, pso: pso, commandBuffer: cmd,
                                  fixture: fixture, rowsPerExpert: r)
                }
                totals[r - 1] += seconds
                perRow.append(seconds)
                line += String(format: " %8.1f", seconds * 1e6)
            }
            // Marginal cost of one route row, per expert: the r=1..8 slope.
            let slope = (perRow[7] - perRow[0]) / 7 / Double(moeRowsTileExperts)
            line += String(format: "   | %6.2f", slope * 1e6)
            print(line)
        }
        var sum = "    both      "
        for r in 1...8 { sum += String(format: " %8.1f", totals[r - 1] * 1e6) }
        let slope = (totals[7] - totals[0]) / 7 / Double(moeRowsTileExperts)
        sum += String(format: "   | %6.2f", slope * 1e6)
        print(sum)
        let atFour = totals[3] / Double(moeRowsTileExperts) * expertsPerBlock
        print(String(format: "    r=4 scaled to %.0f experts: %.1f ms/blk",
                     expertsPerBlock, atFour * 1e3))
    }

    // `encodeStreamedRows` picks one pipeline for the whole tile, specialized to
    // the *widest* block in it. Average rows per expert is 1.72 at bs=4 (27 §6),
    // but the widest of sixteen draws is routinely 4 or 5 -- so the narrow blocks
    // are paying the wide block's register allocation. This grid is what that
    // costs: read down a column to see one expert's `r` rows getting more
    // expensive purely because a tile-mate is wider.
    print("")
    print("  cost of a cap-r mismatch (us per dispatch, r rows on all 16 experts)")
    for stage in MoERowsStage.allCases {
        let name = stage == .gateUp
            ? "prefill_moe_rows_gate_up_act" : "prefill_moe_rows_down"
        print("")
        print("    \(stage.rawValue)")
        var header = "      cap \\ r "
        for r in 1...8 { header += String(format: "     r=%d", r) }
        print(header)
        for cap in 1...8 {
            let pso = try pipeline(name, cap: cap)
            var line = String(format: "      cap=%d   ", cap)
            for r in 1...8 {
                guard r <= cap else { line += "        ."; continue }
                let seconds = try gpuSecondsThrowing(
                    context: context, iterations: iterations,
                    label: "\(stage.rawValue) cap=\(cap) r=\(r)") { cmd in
                    encodeMoERows(stage, pso: pso, commandBuffer: cmd,
                                  fixture: fixture, rowsPerExpert: r)
                }
                line += String(format: " %8.1f", seconds * 1e6)
            }
            print(line)
        }
    }

    // The row mixture a bs=4 verify block actually hands these kernels, run
    // three ways at several tile widths.
    //
    // The width matters more than 31 §5 assumed. A tile *binds* up to sixteen
    // experts, but the live experts a 4-token block puts in one are far fewer:
    // instrumenting `encodeStreamedRows` over a 48-slot run reads 6560 blocks
    // across 1144 tiles -- 5.7 experts per tile -- with row counts
    // 1:4036 2:1257 3:818 4:449 (mean 1.65). Those proportions are what
    // `mixture(width:)` lays down; width 16 reproduces 31 §5's tile.
    func mixture(width: Int) -> [Int] {
        let share = [1: 0.615, 2: 0.192, 3: 0.125, 4: 0.068]
        var rows: [Int] = []
        for r in [4, 3, 2] {
            let n = Int((Double(width) * share[r]!).rounded())
            rows.append(contentsOf: [Int](repeating: r, count: n))
        }
        // Ones fill the rest, so the tile is exactly `width` experts wide.
        rows.append(contentsOf: [Int](repeating: 1, count: max(0, width - rows.count)))
        return rows.sorted()
    }

    /// One scheme: for each stage, the dispatches to encode as (cap, predicate).
    struct MixScheme {
        let name: String
        let buckets: (MoERowsStage, [Int]) -> [(cap: Int, keep: (Int) -> Bool)]
    }
    let schemes: [MixScheme] = [
        MixScheme(name: "current (one dispatch)") { _, mix in
            [(cap: mix.max() ?? 1, keep: { _ in true })]
        },
        MixScheme(name: "`down` split at 2") { stage, mix in
            let widest = mix.max() ?? 1
            guard stage == .down else { return [(cap: widest, keep: { _ in true })] }
            let narrow = mix.filter { $0 <= 2 }
            let wide = mix.filter { $0 >= 3 }
            guard !narrow.isEmpty, !wide.isEmpty else {
                return [(cap: widest, keep: { _ in true })]
            }
            return [(cap: narrow.max()!, keep: { $0 <= 2 }),
                    (cap: wide.max()!, keep: { $0 >= 3 })]
        },
        MixScheme(name: "fully bucketed") { _, mix in
            Set(mix).sorted().map { r in (cap: r, keep: { $0 == r }) }
        }
    ]

    print("")
    print("  production row mixture, by tile width (us per tile)")
    print("    `split` cuts `down` at the §4 cliff (<=2 / >=3); "
            + "width 5.7 is what production runs.")
    var header = "      " + "width  mixture".padding(toLength: 30, withPad: " ",
                                                     startingAt: 0)
    for scheme in schemes {
        header += String(format: " %22@", scheme.name as NSString)
    }
    print(header)
    for width in [4, 6, 8, 16] {
        let mix = mixture(width: width)
        var line = "      "
            + ("\(width)      " + mix.map(String.init).joined())
                .padding(toLength: 30, withPad: " ", startingAt: 0)
        for scheme in schemes {
            var total = 0.0
            for stage in MoERowsStage.allCases {
                let name = stage == .gateUp
                    ? "prefill_moe_rows_gate_up_act" : "prefill_moe_rows_down"
                let buckets = scheme.buckets(stage, mix)
                var psos: [MTLComputePipelineState] = []
                for bucket in buckets { psos.append(try pipeline(name, cap: bucket.cap)) }
                total += try gpuSecondsThrowing(
                    context: context, iterations: iterations,
                    label: "w\(width) \(scheme.name) \(stage.rawValue)") { cmd in
                    for (index, bucket) in buckets.enumerated() {
                        encodeMoERows(stage, pso: psos[index], commandBuffer: cmd,
                                      fixture: fixture, rowCounts: mix,
                                      select: bucket.keep)
                    }
                }
            }
            line += String(format: " %13.1f (%5.1f/e)",
                           total * 1e6, total * 1e6 / Double(width))
        }
        print(line)
    }
}
