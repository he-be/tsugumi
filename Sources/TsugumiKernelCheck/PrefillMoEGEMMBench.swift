import Foundation
import Metal
import Tsugumi
import TsugumiValidationSupport

// MARK: - Prefill routed MoE tiled GEMM, off the prefill timeline
//
// `--moe-gemm-bench` (docs/m6-prefill/28). The level-2 profile puts
// `moe.gateup` at 0.74 s for 4.72 TFLOP (6.4 TFLOP/s), while `qkv`, which runs
// the same 64x64x64 `matmul2d` staging on a dense matrix, reaches 9.9. This
// bench times the routed gate/up dispatch alone so the difference can be
// split: rows per expert (a 64-row block is partly empty unless the expert's
// row count is a multiple of 64), one dispatch per tile (production commits
// each expert tile on its own), and the kernel.
//
// One tile of 16 experts, production shapes (D = 2816, F = 704), symmetric
// int4, weights synthetic and resident. Blocks are cut the way
// `PrefillRoutedGEMMPlanner` cuts them. Each case reports the median GPU time
// of one dispatch in its own command buffer, and of `perBuffer` dispatches in
// one buffer divided by `perBuffer`; the useful rate counts real rows only,
// the padded rate counts every row of every 64-row block.

private let benchD = 2816
private let benchF = 704
private let benchExperts = 16
private let benchTokens = 4096

func runPrefillMoEGEMMBench(groupSize: Int, iterations: Int) throws {
    affineScheme = .sym
    let context = try makeContext(groupSize: groupSize)
    let device = context.device
    var rng = SplitMix64(seed: 0x6d6f65)

    // Blob layout as the packer writes it: gate (W, S, B), up, down.
    var params = MoEGEMMBenchParams()
    var bytes: [UInt8] = []
    func append(rows: Int, cols: Int) -> (UInt32, UInt32, UInt32) {
        let w = UInt32(bytes.count)
        bytes += (0..<(rows * cols / 2)).map { _ in UInt8(truncatingIfNeeded: rng.next()) }
        let s = UInt32(bytes.count)
        for _ in 0..<(rows * cols / groupSize) { bytes += [0x24, 0x3C] }   // bf16 0.01
        let b = UInt32(bytes.count)
        bytes += [UInt8](repeating: 0, count: rows * cols / groupSize * 2)
        return (w, s, b)
    }
    (params.gateWOff, params.gateSOff, params.gateBOff) = append(rows: benchF, cols: benchD)
    (params.upWOff, params.upSOff, params.upBOff) = append(rows: benchF, cols: benchD)
    (params.downWOff, params.downSOff, params.downBOff) = append(rows: benchD, cols: benchF)
    let blobs = (0..<benchExperts).compactMap { _ in
        device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared)
    }
    let hiddenHalves = (0..<(benchTokens * benchD)).map { _ in
        Float16(Float(rng.next() % 2001) / 1000 - 1)
    }
    guard blobs.count == benchExperts,
          let hidden = device.makeBuffer(bytes: hiddenHalves,
                                         length: hiddenHalves.count * 2,
                                         options: .storageModeShared),
          let function = try context.library.makeFunction(name: "prefill_moe_gemm_int4") else {
        fatalError("moe gemm bench fixture failed")
    }
    let argumentEncoder = function.makeArgumentEncoder(bufferIndex: 9)
    guard let argumentBuffer = device.makeBuffer(length: argumentEncoder.encodedLength,
                                                 options: .storageModeShared) else {
        fatalError("moe gemm bench allocation failed")
    }
    argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
    for (index, blob) in blobs.enumerated() {
        argumentEncoder.setBuffer(blob, offset: 0, index: index)
    }

    // (name, tiers): each tier is (kernel, tile rows, tile columns, threads),
    // tallest first. An expert's rows are cut into blocks of the tallest
    // tier; the remainder goes to the shortest tier that holds it.
    typealias Tier = (MTLComputePipelineState, Int, Int, Int)
    var kernels: [(String, [Tier])] = []
    func pso(_ name: String) -> MTLComputePipelineState? { try? context.pipeline(name) }
    if let mpp = pso("prefill_moe_gemm_int4_mpp") {
        kernels.append(("mpp", [(mpp, 64, 64, 128)]))
    }
    if let pf = pso("prefill_moe_gemm_int4_mpp_pf"),
       let m32 = pso("prefill_moe_gemm_int4_mpp_pf_m32") {
        kernels.append(("mpp-pf", [(pf, 64, 64, 128)]))
        kernels.append(("mpp-pf-64/32", [(pf, 64, 64, 128), (m32, 32, 64, 128)]))
        if let m128 = pso("prefill_moe_gemm_int4_mpp_pf_m128s8") {
            kernels.append(("mpp-pf-128s8/64/32",
                            [(m128, 128, 64, 256), (pf, 64, 64, 128), (m32, 32, 64, 128)]))
        }
        if let n128 = pso("prefill_moe_gemm_int4_mpp_pf_n128s8") {
            kernels.append(("mpp-pf-n128s8", [(n128, 64, 128, 256)]))
            if let n128m32 = pso("prefill_moe_gemm_int4_mpp_pf_m32n128s8") {
                kernels.append(("mpp-pf-n128s8-64/32",
                                [(n128, 64, 128, 256), (n128m32, 32, 128, 256)]))
            }
        }
    }
    kernels.append(("simdgroup", [(try context.pipeline("prefill_moe_gemm_int4"), 64, 64, 128)]))

    let perBuffer = 8
    print("moe-gemm-bench g=\(groupSize) sym D=\(benchD) F=\(benchF) experts=\(benchExperts) iterations=\(iterations)")
    print("kernel\trows/expert\trows\tblocks\tone_us\tbatched_us\tuseful_tflops_one\tuseful_tflops_batched\tpadded_tflops_batched")
    for rowsPerExpert in [12, 27, 64, 96, 128, 140, 155, 192, 256] {
        // Pairs: expert e takes rows [e*r, (e+1)*r), tokens drawn without a
        // pattern so the gather is scattered as in production.
        var pairs: [MoEGEMMBenchPair] = []
        for expert in 0..<benchExperts {
            for _ in 0..<rowsPerExpert {
                pairs.append(MoEGEMMBenchPair(token: UInt32(rng.next() % UInt64(benchTokens)),
                                              expert: UInt32(expert), rank: 0, weightBits: 0))
            }
        }
        let rows = pairs.count
        var p = params
        p.pairCount = UInt32(rows)
        p.d = UInt32(benchD)
        p.f = UInt32(benchF)
        p.topK = 8
        p.hiddenStrideElements = UInt32(benchD)
        p.liveExpertCount = UInt32(benchExperts)
        guard let pairBuffer = device.makeBuffer(bytes: pairs,
                                                 length: pairs.count * MemoryLayout<MoEGEMMBenchPair>.stride,
                                                 options: .storageModeShared),
              let act = device.makeBuffer(length: 3 * rows * benchF * 2, options: .storageModePrivate) else {
            fatalError("moe gemm bench allocation failed")
        }
        let flops = 2.0 * 2.0 * Double(rows) * Double(benchF) * Double(benchD)
        for (name, tiers) in kernels {
            var tierBlocks = [[MoEGEMMBenchBlock]](repeating: [], count: tiers.count)
            for expert in 0..<benchExperts {
                let start = expert * rowsPerExpert
                var done = 0
                while done < rowsPerExpert {
                    let left = rowsPerExpert - done
                    var tier = 0
                    if left < tiers[0].1 {
                        tier = tiers.lastIndex { $0.1 >= left } ?? 0
                    }
                    let count = min(tiers[tier].1, left)
                    tierBlocks[tier].append(MoEGEMMBenchBlock(localSlot: UInt32(expert),
                                                              pairStart: UInt32(start + done),
                                                              rowCount: UInt32(count),
                                                              localRow: UInt32(start + done)))
                    done += count
                }
            }
            let blockCount = tierBlocks.reduce(0) { $0 + $1.count }
            let paddedRows = zip(tiers, tierBlocks).reduce(0) { $0 + $1.0.1 * $1.1.count }
            let paddedFlops = 2.0 * 2.0 * Double(paddedRows) * Double(benchF) * Double(benchD)
            func encode(_ cmd: MTLCommandBuffer) {
                for (tier, blocks) in zip(tiers, tierBlocks) where !blocks.isEmpty {
                guard let enc = cmd.makeComputeCommandEncoder() else { return }
                enc.setComputePipelineState(tier.0)
                enc.setBuffer(hidden, offset: 0, index: 0)
                enc.setBuffer(pairBuffer, offset: 0, index: 1)
                var blockCopy = blocks
                enc.setBytes(&blockCopy, length: blockCopy.count * MemoryLayout<MoEGEMMBenchBlock>.stride,
                             index: 2)
                enc.setBuffer(act, offset: 0, index: 5)
                enc.setBuffer(act, offset: 0, index: 7)
                enc.setBuffer(argumentBuffer, offset: 0, index: 9)
                var pv = p
                enc.setBytes(&pv, length: MemoryLayout<MoEGEMMBenchParams>.stride, index: 10)
                var mode: UInt32 = 0
                enc.setBytes(&mode, length: 4, index: 11)
                for blob in blobs { enc.useResource(blob, usage: .read) }
                enc.dispatchThreadgroups(MTLSize(width: (benchF + tier.2 - 1) / tier.2,
                                                 height: blocks.count, depth: 2),
                                         threadsPerThreadgroup: MTLSize(width: tier.3, height: 1, depth: 1))
                enc.endEncoding()
                }
            }
            func time(dispatches: Int) -> Double {
                var samples: [Double] = []
                for iteration in 0..<(iterations + 2) {
                    guard let cmd = context.queue.makeCommandBuffer() else { fatalError() }
                    for _ in 0..<dispatches { encode(cmd) }
                    waitAndCheck(cmd, "moe gemm bench \(name)")
                    if iteration >= 2 { samples.append((cmd.gpuEndTime - cmd.gpuStartTime) / Double(dispatches)) }
                }
                samples.sort()
                return samples[samples.count / 2]
            }
            let one = time(dispatches: 1)
            let batched = time(dispatches: perBuffer)
            print(String(format: "%@\t%d\t%d\t%d\t%.1f\t%.1f\t%.2f\t%.2f\t%.2f",
                         name, rowsPerExpert, rows, blockCount, one * 1e6, batched * 1e6,
                         flops / one / 1e12, flops / batched / 1e12, paddedFlops / batched / 1e12))
        }
    }
}

/// `PrefillGroupedRoutedMoEStreamedParamsMSL`.
private struct MoEGEMMBenchParams {
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
private struct MoEGEMMBenchBlock {
    var localSlot: UInt32
    var pairStart: UInt32
    var rowCount: UInt32
    var localRow: UInt32
}

/// `PrefillTokenExpertPairMSL`.
private struct MoEGEMMBenchPair {
    var token: UInt32
    var expert: UInt32
    var rank: UInt32
    var weightBits: UInt32
}
