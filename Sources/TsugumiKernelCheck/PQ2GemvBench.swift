import Foundation
import Metal
import Tsugumi

// MARK: - `--pq2-gemv-bench [--pq2-shapes MxN,...] [--pq2-cols C] [--pq2-repeats R] [--pq2-iterations I]`
//
// The PQ2_0 GEMVs of `ggml_iq.metal` on synthetic PQ2_0 weights, to put next to
// llama.cpp's `test-backend-ops perf -o MUL_MAT` for the same type and shape
// (docs/qwen38-27b/14). The default shape m=4096 / k=14336 is the one
// test-backend-ops uses, so the two numbers are the same work on the same device.
//
// Per shape: the blocks are filled with a fixed generator (2-bit codes 0/1/2, fp16
// scale per 128 weights), y is compared against the same bytes multiplied in
// double on the CPU, and the GPU time is `R` dispatches back to back in one
// command buffer divided by `R` (median of `I` command buffers), which is how
// test-backend-ops averages its runs.

private struct PQ2Random {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }   // [0, 1)
}

/// Synthetic PQ2_0 rows `[m, n]`: 34 B per 128 weights (fp16 d, then 2 bits each).
private func pq2Weights(m: Int, n: Int, seed: UInt64) -> [UInt8] {
    precondition(n % 128 == 0)
    let blocksPerRow = n / 128
    var bytes = [UInt8](repeating: 0, count: m * blocksPerRow * 34)
    var rng = PQ2Random(seed: seed)
    for b in 0..<(m * blocksPerRow) {
        let at = b * 34
        // Scales of a real tensor are around 1e-2; the exact value does not matter here.
        let d = Float16(0.005 + 0.02 * rng.unit())
        withUnsafeBytes(of: d) { bytes[at] = $0[0]; bytes[at + 1] = $0[1] }
        for j in 0..<32 {
            var byte: UInt8 = 0
            for k in 0..<4 {
                let code = UInt8(rng.next() % 3)   // 0 = -1, 1 = 0, 2 = +1
                byte |= code << UInt8(2 * k)
            }
            bytes[at + 2 + j] = byte
        }
    }
    return bytes
}

/// The same bytes, multiplied in double: `y[t][r] = sum_j (q - 1) * d * x[t][j]`.
private func pq2Reference(weights: [UInt8], x: [Float], m: Int, n: Int, tokens: Int) -> [Double] {
    let blocksPerRow = n / 128
    var y = [Double](repeating: 0, count: tokens * m)
    weights.withUnsafeBufferPointer { w in
        for r in 0..<m {
            let row = r * blocksPerRow * 34
            for t in 0..<tokens {
                var acc = 0.0
                for b in 0..<blocksPerRow {
                    let at = row + b * 34
                    let d = Double(Float16(bitPattern: UInt16(w[at]) | (UInt16(w[at + 1]) << 8)))
                    for j in 0..<32 {
                        let byte = w[at + 2 + j]
                        for k in 0..<4 {
                            let q = Double(Int((byte >> UInt8(2 * k)) & 3) - 1)
                            acc += q * d * Double(x[t * n + b * 128 + j * 4 + k])
                        }
                    }
                }
                y[t * m + r] = acc
            }
        }
    }
    return y
}

/// The kernels under test: name in `ggml_iq.metal` and how many rows one threadgroup covers.
private let pq2Variants: [(name: String, rowsPerTG: Int)] = [
    ("ggml_pq2_0_gemv", 8),          // the shared IQ / K form (4 rows per SIMD group)
    ("ggml_pq2_0_gemv_r4", 8),       // x in registers, 32 weights per lane, 4 rows per SIMD group
    ("ggml_pq2_0_gemv_r8w16", 16),   // x in registers, 16 weights per lane, 8 rows per SIMD group
]

func runPQ2GemvBench(shapes: [(m: Int, n: Int)], cols: [Int], repeats: Int, iterations: Int) throws -> Bool {
    let context = try MetalContext()
    let device = context.device
    let library = try MetalContext.moduleLibrary(device: device, module: "ggml_iq")
    let pipelines: [(name: String, rowsPerTG: Int, pso: MTLComputePipelineState)] = try pq2Variants.map {
        guard let function = library.makeFunction(name: $0.name) else {
            throw GGUFFile.Error.format("ggml_iq: \($0.name) missing")
        }
        return ($0.name, $0.rowsPerTG, try device.makeComputePipelineState(function: function))
    }
    setvbuf(stdout, nil, _IOLBF, 0)
    print("PQ2_0 GEMV bench: \(device.name), \(repeats) dispatches per command buffer, median of \(iterations)")
    var allPass = true
    for (m, n) in shapes {
        let bytes = pq2Weights(m: m, n: n, seed: 0x9E3779B97F4A7C15)
        let maxCols = cols.max() ?? 1
        var rng = PQ2Random(seed: 12345)
        var x = [Float](repeating: 0, count: maxCols * n)
        for i in 0..<x.count { x[i] = 2 * rng.unit() - 1 }
        guard let wbuf = bytes.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: bytes.count, options: .storageModeShared) }),
              let xbuf = x.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: x.count * 4, options: .storageModeShared) }),
              let ybuf = device.makeBuffer(length: maxCols * m * 4, options: .storageModeShared) else {
            throw GGUFFile.Error.format("buffer allocation failed for [\(m) x \(n)]")
        }

        func encode(_ cb: MTLCommandBuffer, pso: MTLComputePipelineState, rowsPerTG: Int, tokens: Int, times: Int) {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            enc.setBuffer(wbuf, offset: 0, index: 0)
            enc.setBuffer(xbuf, offset: 0, index: 1)
            enc.setBuffer(ybuf, offset: 0, index: 2)
            var mv = UInt32(m), nv = UInt32(n)
            enc.setBytes(&mv, length: 4, index: 3)
            enc.setBytes(&nv, length: 4, index: 4)
            for _ in 0..<times {
                enc.dispatchThreadgroups(MTLSize(width: (m + rowsPerTG - 1) / rowsPerTG, height: tokens, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            enc.endEncoding()
        }

        let ref = pq2Reference(weights: bytes, x: x, m: m, n: n, tokens: 1)
        let weightBytes = Double(m) * Double(n) / 128 * 34
        print(String(format: "  [%d x %d] %.1f MB of weights", m, n, weightBytes / 1e6))
        for (name, rowsPerTG, pso) in pipelines {
            // Correctness on one column against the same bytes in double.
            memset(ybuf.contents(), 0, maxCols * m * 4)
            let cb = context.queue.makeCommandBuffer()!
            encode(cb, pso: pso, rowsPerTG: rowsPerTG, tokens: 1, times: 1)
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            let y = UnsafeBufferPointer(start: ybuf.contents().bindMemory(to: Float.self, capacity: m), count: m)
            var maxDiff = 0.0, refMax = 0.0
            for i in 0..<m {
                refMax = max(refMax, abs(ref[i]))
                maxDiff = max(maxDiff, abs(Double(y[i]) - ref[i]))
            }
            let rel = refMax > 0 ? maxDiff / refMax : .infinity
            let pass = rel < 1e-5
            allPass = allPass && pass
            var line = String(format: "    %-22@ rel err %.2e %@ ", name as NSString, rel, pass ? "  " : "FAIL")
            for tokens in cols {
                var us: [Double] = []
                for i in 0..<(iterations + 1) {
                    let cb = context.queue.makeCommandBuffer()!
                    encode(cb, pso: pso, rowsPerTG: rowsPerTG, tokens: tokens, times: repeats)
                    cb.commit()
                    cb.waitUntilCompleted()
                    if let error = cb.error { throw error }
                    if i == 0 { continue }   // warm-up command buffer
                    us.append((cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(repeats))
                }
                us.sort()
                let median = us[us.count / 2]
                line += String(format: " %d 列 %7.1f us", tokens, median)
                if tokens == 1 { line += String(format: " (%.0f GB/s)", weightBytes / median / 1e3) }
            }
            print(line)
        }
    }
    print("  \(allPass ? "PASS" : "FAIL")")
    return allPass
}
