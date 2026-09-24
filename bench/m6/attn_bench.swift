// Kernel-only timing of the prefill sliding-window attention (docs/m6-prefill 06 §2 の候補 sgmm の切り分け用)。
// 記録値ではない。本番ループの数字は prefill_bench.sh で取る。
//   swiftc -O bench/m6/attn_bench.swift -o /tmp/attn_bench
//   /tmp/attn_bench Sources/Tsugumi/Metal/Prefill/prefill.metal \
//       attention_prefill_causal_qblock_d256:256:32 attention_prefill_causal_sgmm_d256:128:32 \
//       attention_prefill_causal_sgmm_gqa_d256:256:32:2
// 形は Gemma 4 の SWA 層: 16/8 ヘッド、d256、窓 1024、2478 トークンを 2048 + 430 の 2 チャンク、25 層分を 1 コマンドバッファ。
// TF_ATTN_BENCH_SHAPE=full で全注意の層: 16/2 ヘッド、d512、窓なし、5 層分 (docs/m6-prefill/36)。
//   TF_ATTN_BENCH_SHAPE=full /tmp/attn_bench Sources/Tsugumi/Metal/Prefill/prefill.metal \
//       attention_prefill_causal_qblock_d512:256:16 attention_prefill_full_mpp_d512_r32:128:32 \
//       attention_prefill_full_mpp_d512_r64:256:64 unfused:128:64
// `unfused` は 3 カーネル (scores → softmax → pv、docs/m6-prefill/36・37) を PrefillAttention と同じ帯の幅と頭のまとめ方 (64 MiB) で。
// SWA の形 (既定) でも全注意の形でも回る。
import Metal
import Foundation

let args = CommandLine.arguments
let srcPath = args[1]
let kernels = Array(args.dropFirst(2))
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let src = try! String(contentsOfFile: srcPath, encoding: .utf8)
let opts = MTLCompileOptions()
if #available(macOS 26.0, *) { opts.languageVersion = .version4_0 } else { opts.languageVersion = .version3_2 }
opts.preprocessorMacros = ["MOEPACK_AFFINE_GROUP_SIZE": NSNumber(value: 32), "MOEPACK_AFFINE_SYMMETRIC": NSNumber(value: 1)]
let lib = try! device.makeLibrary(source: src, options: opts)

struct Params { var startPosition: UInt32; var queryCount: UInt32; var headDim: UInt32; var numQHeads: UInt32; var numKVHeads: UInt32; var kvValidCount: UInt32; var slidingWindow: UInt32; var kvStride: UInt32; var qStride: UInt32; var oStride: UInt32; var scale: Float; var hasSpanMask: UInt32 }

let full = ProcessInfo.processInfo.environment["TF_ATTN_BENCH_SHAPE"] == "full"
let H = 16, KVH = full ? 2 : 8, D = full ? 512 : 256, total = 2478, window = full ? 0 : 1024
let layers = full ? 5 : 25
func buf(_ n: Int, seed: UInt32) -> MTLBuffer {
    let b = device.makeBuffer(length: n * 2, options: .storageModeShared)!
    let p = b.contents().bindMemory(to: Float16.self, capacity: n)
    var s = seed
    for i in 0..<n { s = s &* 1664525 &+ 1013904223; p[i] = Float16(Float(Int(s >> 16) % 2000 - 1000) / 3000) }
    return b
}
let q = buf(2048 * H * D, seed: 1), k = buf(total * KVH * D, seed: 2), v = buf(total * KVH * D, seed: 3)
// Each chunk writes its own rows, so the comparison below covers every row of both chunks.
let outs = kernels.map { _ in device.makeBuffer(length: total * H * D * 2, options: .storageModeShared)! }
let dummy = device.makeBuffer(length: 16, options: .storageModeShared)!

for (ki, name) in kernels.enumerated() {
    let parts = name.split(separator: ":")   // fn:threads:queriesPerGroup[:headsPerGroup]
    let fn = String(parts[0]); let threads = Int(parts[1])!; let qpg = Int(parts[2])!
    let hpg = parts.count > 3 ? Int(parts[3])! : 1
    let cv = MTLFunctionConstantValues()
    let unfused = fn == "unfused"
    let pso = try! device.makeComputePipelineState(function: lib.makeFunction(
        name: unfused ? "attention_prefill_unfused_pv_d\(D)" : fn, constantValues: cv))
    let upso = unfused ? ["attention_prefill_unfused_scores_d\(D)", "attention_prefill_unfused_softmax"].map {
        try! device.makeComputePipelineState(function: lib.makeFunction(name: $0, constantValues: cv)) } : []
    let budget = 64 << 20
    let scratch = device.makeBuffer(length: budget, options: .storageModePrivate)!
    struct UParams { var headBase: UInt32; var scoreStride: UInt32; var headRows: UInt32; var ring: UInt32 }
    var times: [Double] = []
    for rep in 0..<6 {
        let cb = queue.makeCommandBuffer()!
        for (start, T) in [(0, 2048), (2048, total - 2048)] {
            for _ in 0..<layers {
                var p = Params(startPosition: UInt32(start), queryCount: UInt32(T), headDim: UInt32(D), numQHeads: UInt32(H), numKVHeads: UInt32(KVH), kvValidCount: UInt32(start + T), slidingWindow: UInt32(window), kvStride: UInt32(KVH * D), qStride: UInt32(H * D), oStride: UInt32(H * D), scale: 1, hasSpanMask: 0)
                let enc = cb.makeComputeCommandEncoder()!
                if unfused {
                    var stride = (start + T + 63) / 64 * 64
                    if window > 0 && window < start + T { stride = min(stride, (window + 127 + 63) / 64 * 64) }
                    let perHead = T * stride * 6 + T * 4
                    var group = H
                    while group > 1 && (group * perHead > budget || H % group != 0) { group -= 1 }
                    let pOff = group * T * stride * 4, sumOff = pOff + group * T * stride * 2
                    for hb in Swift.stride(from: 0, to: H, by: group) {
                        var u = UParams(headBase: UInt32(hb), scoreStride: UInt32(stride), headRows: UInt32(T), ring: 0)
                        let tpt = MTLSize(width: 128, height: 1, depth: 1)
                        enc.setComputePipelineState(upso[0])
                        enc.setBuffer(q, offset: 0, index: 0); enc.setBuffer(k, offset: 0, index: 1)
                        enc.setBuffer(scratch, offset: 0, index: 2)
                        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 4)
                        enc.setBytes(&u, length: MemoryLayout<UParams>.stride, index: 5)
                        enc.dispatchThreadgroups(MTLSize(width: stride / 64, height: (T + 63) / 64, depth: group), threadsPerThreadgroup: tpt)
                        enc.setComputePipelineState(upso[1])
                        enc.setBuffer(scratch, offset: pOff, index: 3); enc.setBuffer(scratch, offset: sumOff, index: 6)
                        enc.dispatchThreadgroups(MTLSize(width: (T + 3) / 4, height: group, depth: 1), threadsPerThreadgroup: tpt)
                        enc.setComputePipelineState(pso)
                        enc.setBuffer(outs[ki], offset: start * H * D * 2, index: 0); enc.setBuffer(v, offset: 0, index: 1)
                        enc.dispatchThreadgroups(MTLSize(width: D / 64, height: (T + 63) / 64, depth: group), threadsPerThreadgroup: tpt)
                    }
                    enc.endEncoding()
                    continue
                }
                enc.setComputePipelineState(pso)
                enc.setBuffer(q, offset: 0, index: 0); enc.setBuffer(k, offset: 0, index: 1); enc.setBuffer(v, offset: 0, index: 2)
                enc.setBuffer(outs[ki], offset: start * H * D * 2, index: 3)
                enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 4)
                enc.setBuffer(dummy, offset: 0, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: (T + qpg - 1) / qpg, height: H / hpg, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
                enc.endEncoding()
            }
        }
        cb.commit(); cb.waitUntilCompleted()
        if rep > 0 { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    times.sort()
    print(String(format: "%@  %d layers x 2 chunks: median %.3f s (min %.3f max %.3f)  regs? maxThreads=%d", fn, layers, times[times.count/2], times[0], times.last!, pso.maxTotalThreadsPerThreadgroup))
}
// compare every output row of both chunks against kernel 0
if kernels.count > 1 {
    let n = total * H * D
    let a = outs[0].contents().bindMemory(to: Float16.self, capacity: n)
    for ki in 1..<kernels.count {
        let b = outs[ki].contents().bindMemory(to: Float16.self, capacity: n)
        var m: Float = 0
        var bad = 0
        for i in 0..<n {
            let d = abs(Float(a[i]) - Float(b[i]))
            if !d.isFinite { bad += 1 } else { m = max(m, d) }
        }
        print("maxdiff vs \(kernels[0]) (all \(total) rows): \(m)  non-finite: \(bad)")
    }
}
