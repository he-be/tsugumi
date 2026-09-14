import Foundation
import Metal
import Tsugumi

// MARK: - `--q38-wy-bench [tokens] [chunks,...] [iterations]`
//
// One GDN layer of the Qwen3.8 runner: the serial `q38_gdn_step` (A) against the chunked WY form
// `Qwen38GDNChunk` (B) on the same inputs (docs/qwen38/07). Inputs are random in the model's shapes:
// q/k through `q38_gdn_qk_norm`, decay/beta through `q38_gdn_gates` with A = -[1, 16] and dt bias as in
// the Qwen3-Next init, and a random nonzero starting state. Prints the GPU time of each arm (median)
// and the relative max difference of the outputs and of the final state. `Q38_WY_SPLIT=1` commits B stage
// by stage and prints each stage's GPU ms (B's own GPU column then reads 0).

func runQ38WYBench(tokens T: Int, chunks: [Int], iterations: Int) throws {
    let context = try MetalContext()
    let device = context.device
    let queue = context.queue
    let lib = try MetalContext.moduleLibrary(device: device, module: "qwen38")
    func pso(_ name: String) throws -> MTLComputePipelineState {
        try device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    }
    let psoNorm = try pso("q38_gdn_qk_norm"), psoGates = try pso("q38_gdn_gates"), psoStep = try pso("q38_gdn_step")
    let hk = 16, hv = 48, d = 128
    let C = (2 * hk + hv) * d

    var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
    func uniform() -> Float {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return Float(seed >> 40) / Float(1 << 24)
    }
    func normal() -> Float { sqrt(-2 * log(max(uniform(), 1e-7))) * cos(2 * .pi * uniform()) }
    func buffer(_ values: [Float]) -> MTLBuffer {
        device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared)!
    }
    func buffer(count: Int) -> MTLBuffer { device.makeBuffer(length: count * 4, options: .storageModeShared)! }

    let conv = buffer((0..<(T * C)).map { _ in normal() })
    let ga = buffer((0..<(T * hv)).map { _ in normal() })
    let gb = buffer((0..<(T * hv)).map { _ in normal() })
    let aW = buffer((0..<hv).map { _ in -(1 + 15 * uniform()) })
    let dtW = buffer((0..<hv).map { _ in log(exp(0.001 + 0.099 * uniform()) - 1) })
    let state0 = (0..<(hv * d * d)).map { _ in 0.3 * normal() }
    var gp = (UInt32(hk), UInt32(hv), UInt32(d), UInt32(T))
    var cV = UInt32(C)
    do {
        let cb = queue.makeCommandBuffer()!
        var enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psoNorm)
        enc.setBuffer(conv, offset: 0, index: 0)
        enc.setBytes(&gp, length: 16, index: 1)
        enc.setBytes(&cV, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: 2 * hk, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 64, depth: 1))
        enc.endEncoding()
        enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psoGates)
        enc.setBuffer(ga, offset: 0, index: 0)
        enc.setBuffer(gb, offset: 0, index: 1)
        enc.setBuffer(aW, offset: 0, index: 2)
        enc.setBuffer(dtW, offset: 0, index: 3)
        enc.setBytes(&gp, length: 16, index: 4)
        enc.dispatchThreads(MTLSize(width: hv, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 64, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    let decay = UnsafeBufferPointer(start: ga.contents().bindMemory(to: Float.self, capacity: T * hv), count: T * hv)
    let sorted = decay.sorted()
    print(String(format: "T=%d  decay min %.3g  p10 %.3g  median %.3g  p90 %.3g  max %.4g",
                 T, sorted[0], sorted[sorted.count / 10], sorted[sorted.count / 2], sorted[sorted.count * 9 / 10], sorted.last!))

    let stateA = buffer(state0), outA = buffer(count: T * hv * d)
    func armA() -> Double {
        memcpy(stateA.contents(), state0, state0.count * 4)
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psoStep)
        enc.setBuffer(conv, offset: 0, index: 0)
        enc.setBuffer(ga, offset: 0, index: 1)
        enc.setBuffer(gb, offset: 0, index: 2)
        enc.setBuffer(stateA, offset: 0, index: 5)
        enc.setBuffer(outA, offset: 0, index: 6)
        enc.setBytes(&gp, length: 16, index: 7)
        enc.setBytes(&cV, length: 4, index: 8)
        var snapN: UInt32 = 0
        enc.setBuffer(stateA, offset: 0, index: 9)
        enc.setBytes(&snapN, length: 4, index: 10)
        enc.setBuffer(stateA, offset: 0, index: 11)
        enc.dispatchThreadgroups(MTLSize(width: d, height: hv, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return (cb.gpuEndTime - cb.gpuStartTime) * 1000
    }
    let timesA = (0..<iterations).map { _ in armA() }.sorted()
    print(String(format: "A serial      GPU %8.1f ms", timesA[iterations / 2]))

    func relMax(_ x: MTLBuffer, _ y: MTLBuffer, _ n: Int) -> (Float, Float) {
        let p = x.contents().bindMemory(to: Float.self, capacity: n), q = y.contents().bindMemory(to: Float.self, capacity: n)
        var diff: Float = 0, scale: Float = 0
        for i in 0..<n {
            diff = max(diff, abs(p[i] - q[i]))
            scale = max(scale, abs(p[i]))
        }
        return (diff / scale, scale)
    }
    let split = ProcessInfo.processInfo.environment["Q38_WY_SPLIT"] == "1"
    for L in chunks {
        let chunk = try Qwen38GDNChunk(device: device, library: lib, keyHeads: hk, valueHeads: hv, headDim: d, chunk: L)
        let stateB = buffer(state0), outB = buffer(count: T * hv * d)
        var gpu: [Double] = [], encode: [Double] = []
        for _ in 0..<iterations {
            memcpy(stateB.contents(), state0, state0.count * 4)
            var cb = queue.makeCommandBuffer()!
            let e0 = CFAbsoluteTimeGetCurrent()
            var stages: [String: Double] = [:], labels: [String] = []
            chunk.encode(cb, conv: conv, a: ga, b: gb, state: stateB, out: outB, T: T, section: split ? { label, c in
                c.commit()
                c.waitUntilCompleted()
                if stages[label] == nil { labels.append(label) }
                stages[label, default: 0] += (c.gpuEndTime - c.gpuStartTime) * 1000
                return queue.makeCommandBuffer()!
            } : nil)
            if split {
                print("  " + labels.map { String(format: "%@ %.1f", $0, stages[$0]!) }.joined(separator: "  "))
                cb = queue.makeCommandBuffer()!
            }
            encode.append((CFAbsoluteTimeGetCurrent() - e0) * 1000)
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            gpu.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
        }
        let (dO, sO) = relMax(outA, outB, T * hv * d)
        let (dS, sS) = relMax(stateA, stateB, hv * d * d)
        print(String(format: "B chunk %4d  GPU %8.1f ms  encode %6.1f ms  out rel %.3g (max %.3g)  state rel %.3g (max %.3g)",
                     L, gpu.sorted()[iterations / 2], encode.sorted()[iterations / 2], dO, sO, dS, sS))
    }
}
