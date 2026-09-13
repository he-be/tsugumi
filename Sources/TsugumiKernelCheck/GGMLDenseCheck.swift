import Foundation
import Metal
import Tsugumi

// MARK: - GGML dense GEMV (`ggml_dense.metal`) against a CPU double reference
//
// Real tensors from the DS4-IQ2 GGUF, read through `GGUFFile` no-copy buffers
// (so the GPU reads the mapped file pages the runner will read). Shapes cover
// every dense form the Qwen3.8-Flash-Next decode step uses: Q8_0 wide and
// narrow, F16 hyper-connection down/up/inject, F32 router, and the 248K-row
// LM head. The reference dequantizes rows in Swift and sums in Double.
//
//     .build/release/TsugumiKernelCheck --ggml-dense ~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/<main>.gguf

private func ggmlRowDouble(_ file: GGUFFile, _ t: GGUFFile.Tensor, row: Int) -> [Double] {
    let n = t.rowWidth
    let p = file.base + t.offset + row * t.bytesPerRow
    var out = [Double](repeating: 0, count: n)
    switch t.type {
    case .f32:
        for i in 0..<n { out[i] = Double(p.loadUnaligned(fromByteOffset: i * 4, as: Float.self)) }
    case .f16:
        for i in 0..<n { out[i] = Double(Float(p.loadUnaligned(fromByteOffset: i * 2, as: Float16.self))) }
    case .q8_0:
        for b in 0..<(n / 32) {
            let d = Double(Float(p.loadUnaligned(fromByteOffset: b * 34, as: Float16.self)))
            for j in 0..<32 {
                out[b * 32 + j] = d * Double(p.loadUnaligned(fromByteOffset: b * 34 + 2 + j, as: Int8.self))
            }
        }
    default:
        preconditionFailure("unsupported \(t.type)")
    }
    return out
}

func runGGMLDenseCheck(ggufPath: String) throws -> Bool {
    let file = try GGUFFile(url: URL(fileURLWithPath: (ggufPath as NSString).expandingTildeInPath))
    let context = try MetalContext()
    let device = context.device
    let gemv = try GGMLDenseGEMV(device: device)
    let cases = [
        "blk.0.attn_qkv.weight",       // Q8_0 10240 x 2560
        "blk.0.ssm_out.weight",        // Q8_0 2560 x 6144
        "blk.3.attn_k.weight",         // Q8_0 512 x 2560
        "blk.0.ffn_down_shexp.weight", // Q8_0 2560 x 640
        "blk.0.hc_attn_down.weight",   // F16 320 x 10240
        "blk.0.hc_attn_up.weight",     // F16 10240 x 320
        "blk.0.hc_attn_inject.weight", // F16 4 x 10240
        "blk.0.ffn_gate_inp.weight",   // F32 512 x 2560
        "blk.0.ssm_alpha.weight",      // F32 48 x 2560
        "output.weight",               // Q8_0 248320 x 2560 (reference checks a row sample)
    ]
    var rng = SystemRandomNumberGenerator()
    var allPass = true
    print("GGML dense GEMV check: \(file.url.lastPathComponent)")
    for name in cases {
        let t = try file.tensor(name)
        let m = t.rowCount, n = t.rowWidth
        let x = (0..<n).map { _ in Float.random(in: -1...1, using: &rng) }
        guard let (wbuf, woff) = file.noCopyBuffer(device: device, tensor: t),
              let xbuf = device.makeBuffer(bytes: x, length: n * 4, options: .storageModeShared),
              let ybuf = device.makeBuffer(length: m * 4, options: .storageModeShared) else {
            throw GGUFFile.Error.format("buffer allocation failed for \(name)")
        }
        let cb = context.queue.makeCommandBuffer()!
        gemv.encode(commandBuffer: cb, type: t.type, weights: wbuf, weightsOffset: woff,
                    x: xbuf, y: ybuf, m: m, n: n)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw error }
        let y = UnsafeBufferPointer(start: ybuf.contents().bindMemory(to: Float.self, capacity: m), count: m)
        // Rows checked: all of them, except a spread sample on the LM head.
        let rows: [Int] = m > 4096 ? stride(from: 0, to: m, by: m / 997).map { $0 } + [m - 1] : Array(0..<m)
        var maxDiff = 0.0, refMax = 0.0
        for r in rows {
            let w = ggmlRowDouble(file, t, row: r)
            var acc = 0.0
            for i in 0..<n { acc += w[i] * Double(x[i]) }
            refMax = max(refMax, abs(acc))
            maxDiff = max(maxDiff, abs(Double(y[r]) - acc))
        }
        let rel = maxDiff / refMax
        let pass = rel < 1e-5 && refMax > 0
        allPass = allPass && pass
        print(String(format: "  %-30@ %@ [%d x %d] rows %d  rel err %.2e  %@",
                     name as NSString, "\(t.type)" as NSString, m, n, rows.count, rel,
                     pass ? "PASS" : "FAIL"))
    }
    print("  \(allPass ? "PASS" : "FAIL")")
    return allPass
}

// MARK: - Q8_0 GEMV geometry bench (`--ggml-dense-bench <gguf> [iterations]`)
//
// GPU time of the three Q8_0 kernels on the real decode-step tensors. Each
// sample is one command buffer carrying 48 dispatches of one tensor (a token's
// worth for that projection); the median of `iterations` samples is reported,
// kernels interleaved A B C C B A. Outputs are compared against the current
// kernel (`ggml_q8_0_gemv`).
func runGGMLDenseBench(ggufPath: String, iterations: Int) throws {
    let file = try GGUFFile(url: URL(fileURLWithPath: (ggufPath as NSString).expandingTildeInPath))
    let context = try MetalContext()
    let device = context.device
    let library = try MetalContext.moduleLibrary(device: device, module: "ggml_dense")
    func pso(_ name: String) throws -> MTLComputePipelineState {
        try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
    }
    let kernels = [("tg64/4rows", try pso("ggml_q8_0_gemv")), ("lane256", try pso("ggml_q8_0_gemv_lane")),
                   ("rows", try pso("ggml_q8_0_gemv_rows"))]
    let cases = ["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
                 "blk.3.attn_q.weight", "blk.3.attn_output.weight", "blk.0.ffn_up_shexp.weight",
                 "blk.0.ffn_down_shexp.weight"]
    let gemv = try GGMLDenseGEMV(device: device)
    print("GGMLDenseGEMV as dispatched by the runner: 48 per sample, median of \(iterations) (GPU ms)")
    for name in ["blk.0.hc_attn_down.weight", "blk.0.hc_attn_up.weight", "blk.0.hc_attn_inject.weight",
                 "blk.0.ffn_gate_inp.weight", "blk.0.ssm_alpha.weight", "blk.0.attn_qkv.weight"] {
        let t = try file.tensor(name)
        let ms = try ggmlDenseEncodeMedian(file: file, context: context, gemv: gemv, name: name, iterations: iterations)
        print(String(format: "  %-28@ %@ [%6d x %5d]  %7.2f", name as NSString, "\(t.type)" as NSString,
                     t.rowCount, t.rowWidth, ms))
    }
    // Float kernels: stride (current) vs chunk, same dispatch, outputs compared.
    print("F16/F32 stride vs chunk: 48 per sample, median of \(iterations) (GPU ms)")
    for name in ["blk.0.hc_attn_down.weight", "blk.0.hc_attn_up.weight", "blk.0.hc_attn_inject.weight",
                 "blk.0.ffn_gate_inp.weight", "blk.0.ssm_alpha.weight"] {
        let t = try file.tensor(name)
        let m = t.rowCount, n = t.rowWidth
        let pair = t.type == .f16 ? [try pso("ggml_f16_gemv"), try pso("ggml_f16_gemv_chunk")]
                                  : [try pso("ggml_f32_gemv"), try pso("ggml_f32_gemv_chunk")]
        var rng = SystemRandomNumberGenerator()
        let x = (0..<n).map { _ in Float.random(in: -1...1, using: &rng) }
        let (wbuf, woff) = file.noCopyBuffer(device: device, tensor: t)!
        let xbuf = device.makeBuffer(bytes: x, length: n * 4, options: .storageModeShared)!
        let ys = pair.map { _ in device.makeBuffer(length: m * 4, options: .storageModeShared)! }
        var times: [[Double]] = [[], []]
        for i in 0...(2 * iterations) {
            for k in (i % 2 == 0 ? [0, 1] : [1, 0]) {
                let cb = context.queue.makeCommandBuffer()!
                for _ in 0..<48 {
                    let enc = cb.makeComputeCommandEncoder()!
                    enc.setComputePipelineState(pair[k])
                    enc.setBuffer(wbuf, offset: woff, index: 0)
                    enc.setBuffer(xbuf, offset: 0, index: 1)
                    enc.setBuffer(ys[k], offset: 0, index: 2)
                    var mv = UInt32(m), nv = UInt32(n)
                    enc.setBytes(&mv, length: 4, index: 3)
                    enc.setBytes(&nv, length: 4, index: 4)
                    enc.dispatchThreadgroups(MTLSize(width: (m + 7) / 8, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                    enc.endEncoding()
                }
                cb.commit()
                cb.waitUntilCompleted()
                if i > 0 { times[k].append((cb.gpuEndTime - cb.gpuStartTime) * 1000) }
            }
        }
        let y0 = UnsafeBufferPointer(start: ys[0].contents().bindMemory(to: Float.self, capacity: m), count: m)
        let y1 = UnsafeBufferPointer(start: ys[1].contents().bindMemory(to: Float.self, capacity: m), count: m)
        var d = 0.0, r = 0.0
        for i in 0..<m { d = max(d, Double(abs(y1[i] - y0[i]))); r = max(r, Double(abs(y0[i]))) }
        let med = times.map { $0.sorted()[$0.count / 2] }
        print(String(format: "  %-28@ %@ [%6d x %5d]  stride %7.2f  chunk %7.2f  (diff %.0e)",
                     name as NSString, "\(t.type)" as NSString, m, n, med[0], med[1], d / r))
    }
    var rng = SystemRandomNumberGenerator()
    print("Q8_0 GEMV geometry bench: 48 dispatches per sample, median of \(iterations) (GPU ms)")
    for name in cases {
        let t = try file.tensor(name)
        precondition(t.type == .q8_0)
        let m = t.rowCount, n = t.rowWidth
        let x = (0..<n).map { _ in Float.random(in: -1...1, using: &rng) }
        let (wbuf, woff) = file.noCopyBuffer(device: device, tensor: t)!
        let xbuf = device.makeBuffer(bytes: x, length: n * 4, options: .storageModeShared)!
        let ys = kernels.map { _ in device.makeBuffer(length: m * 4, options: .storageModeShared)! }
        func sample(_ k: Int) -> Double {
            let (_, p) = kernels[k]
            let cb = context.queue.makeCommandBuffer()!
            for _ in 0..<48 {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(p)
                enc.setBuffer(wbuf, offset: woff, index: 0)
                enc.setBuffer(xbuf, offset: 0, index: 1)
                enc.setBuffer(ys[k], offset: 0, index: 2)
                var mv = UInt32(m), nv = UInt32(n)
                enc.setBytes(&mv, length: 4, index: 3)
                enc.setBytes(&nv, length: 4, index: 4)
                switch k {
                case 0: enc.dispatchThreadgroups(MTLSize(width: (m + 7) / 8, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                case 1: enc.dispatchThreadgroups(MTLSize(width: (m + 7) / 8, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                default: enc.dispatchThreads(MTLSize(width: m, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, 64), height: 1, depth: 1))
                }
                enc.endEncoding()
            }
            cb.commit()
            cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) * 1000
        }
        for k in kernels.indices { _ = sample(k) }  // warm pages and pipelines
        var times = kernels.map { _ in [Double]() }
        for i in 0..<iterations {
            let order = i % 2 == 0 ? Array(kernels.indices) : kernels.indices.reversed()
            for k in order { times[k].append(sample(k)) }
        }
        let y0 = UnsafeBufferPointer(start: ys[0].contents().bindMemory(to: Float.self, capacity: m), count: m)
        var line = String(format: "  %-26@ [%6d x %5d]", name as NSString, m, n)
        for k in kernels.indices {
            let yk = UnsafeBufferPointer(start: ys[k].contents().bindMemory(to: Float.self, capacity: m), count: m)
            var d = 0.0, r = 0.0
            for i in 0..<m { d = max(d, Double(abs(yk[i] - y0[i]))); r = max(r, Double(abs(y0[i]))) }
            let med = times[k].sorted()[times[k].count / 2]
            line += String(format: "  %@ %7.2f (%.0e)", kernels[k].0 as NSString, med, d / r)
        }
        print(line)
    }
}

/// GPU ms of `GGMLDenseGEMV` (the runner's dispatch) on one tensor, 48 dispatches
/// per command buffer, median of `iterations`.
func ggmlDenseEncodeMedian(file: GGUFFile, context: MetalContext, gemv: GGMLDenseGEMV,
                           name: String, iterations: Int) throws -> Double {
    let device = context.device
    let t = try file.tensor(name)
    let m = t.rowCount, n = t.rowWidth
    var rng = SystemRandomNumberGenerator()
    let x = (0..<n).map { _ in Float.random(in: -1...1, using: &rng) }
    let (wbuf, woff) = file.noCopyBuffer(device: device, tensor: t)!
    let xbuf = device.makeBuffer(bytes: x, length: n * 4, options: .storageModeShared)!
    let ybuf = device.makeBuffer(length: m * 4, options: .storageModeShared)!
    var times: [Double] = []
    for i in 0...iterations {
        let cb = context.queue.makeCommandBuffer()!
        for _ in 0..<48 {
            gemv.encode(commandBuffer: cb, type: t.type, weights: wbuf, weightsOffset: woff,
                        x: xbuf, y: ybuf, m: m, n: n)
        }
        cb.commit()
        cb.waitUntilCompleted()
        if i > 0 { times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000) }
    }
    return times.sorted()[times.count / 2]
}
