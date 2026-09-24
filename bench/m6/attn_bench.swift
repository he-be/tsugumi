// Kernel-only timing of the prefill sliding-window attention (docs/m6-prefill 06 §2 の候補 sgmm の切り分け用)。
// 記録値ではない。本番ループの数字は prefill_bench.sh で取る。
//   swiftc -O bench/m6/attn_bench.swift -o /tmp/attn_bench
//   /tmp/attn_bench Sources/Tsugumi/Metal/Prefill/prefill.metal \
//       attention_prefill_causal_qblock_d256:256:32 attention_prefill_causal_sgmm_d256:128:32
// 形は Gemma 4 の SWA 層: 16/8 ヘッド、d256、窓 1024、2478 トークンを 2048 + 430 の 2 チャンク、25 層分を 1 コマンドバッファ。
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

let H = 16, KVH = 8, D = 256, total = 2478, window = 1024
func buf(_ n: Int, seed: UInt32) -> MTLBuffer {
    let b = device.makeBuffer(length: n * 2, options: .storageModeShared)!
    let p = b.contents().bindMemory(to: Float16.self, capacity: n)
    var s = seed
    for i in 0..<n { s = s &* 1664525 &+ 1013904223; p[i] = Float16(Float(Int(s >> 16) % 2000 - 1000) / 3000) }
    return b
}
let q = buf(2048 * H * D, seed: 1), k = buf(total * KVH * D, seed: 2), v = buf(total * KVH * D, seed: 3)
let outs = kernels.map { _ in device.makeBuffer(length: 2048 * H * D * 2, options: .storageModeShared)! }
let dummy = device.makeBuffer(length: 16, options: .storageModeShared)!

for (ki, name) in kernels.enumerated() {
    let parts = name.split(separator: ":")   // fn:threads:queriesPerGroup
    let fn = String(parts[0]); let threads = Int(parts[1])!; let qpg = Int(parts[2])!
    let cv = MTLFunctionConstantValues()
    let pso = try! device.makeComputePipelineState(function: lib.makeFunction(name: fn, constantValues: cv))
    var times: [Double] = []
    for rep in 0..<6 {
        let cb = queue.makeCommandBuffer()!
        for (start, T) in [(0, 2048), (2048, total - 2048)] {
            for _ in 0..<25 {
                var p = Params(startPosition: UInt32(start), queryCount: UInt32(T), headDim: UInt32(D), numQHeads: UInt32(H), numKVHeads: UInt32(KVH), kvValidCount: UInt32(start + T), slidingWindow: UInt32(window), kvStride: UInt32(KVH * D), qStride: UInt32(H * D), oStride: UInt32(H * D), scale: 1, hasSpanMask: 0)
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pso)
                enc.setBuffer(q, offset: 0, index: 0); enc.setBuffer(k, offset: 0, index: 1); enc.setBuffer(v, offset: 0, index: 2)
                enc.setBuffer(outs[ki], offset: 0, index: 3)
                enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 4)
                enc.setBuffer(dummy, offset: 0, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: (T + qpg - 1) / qpg, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
                enc.endEncoding()
            }
        }
        cb.commit(); cb.waitUntilCompleted()
        if rep > 0 { times.append(cb.gpuEndTime - cb.gpuStartTime) }
    }
    times.sort()
    print(String(format: "%@  25 layers x 2 chunks: median %.3f s (min %.3f max %.3f)  regs? maxThreads=%d", fn, times[times.count/2], times[0], times.last!, pso.maxTotalThreadsPerThreadgroup))
}
// compare outputs of first chunk against kernel 0
if kernels.count > 1 {
    let n = 2478 - 2048
    let a = outs[0].contents().bindMemory(to: Float16.self, capacity: 2048 * H * D)
    for ki in 1..<kernels.count {
        let b = outs[ki].contents().bindMemory(to: Float16.self, capacity: 2048 * H * D)
        var m: Float = 0
        for i in 0..<(n * H * D) { m = max(m, abs(Float(a[i]) - Float(b[i]))) }
        print("maxdiff vs \(kernels[0]) (second chunk rows): \(m)")
    }
}
