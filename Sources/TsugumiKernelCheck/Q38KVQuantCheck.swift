import Foundation
import Metal
import Tsugumi

// MARK: - Q8_0 / F16 KV storage kernels (`--q38-kv-quant-check [dump dir]`, docs/qwen38/22 §3 順 2)
//
// `q38_kv_quantize` (and the same `q38_q8_write` that `q38_attn_prep` uses for K) against ggml
// `quantize_row_q8_0_ref` done on the host: d = max|x| / 127 as half, q = roundf(x * (1 / d)). Byte for byte.
// `q38_kv_dequantize`, `q38_f32_to_f16` and `q38_f16_to_f32` likewise. Rows: Gaussian at several scales, rows with
// outliers, all-zero and tiny rows, and rows built on halfway cases (d = 1, x = n + 0.5). With a dump dir the input
// floats and the kernel's bytes are written for `Scripts/qwen38/reference_forward.py`'s `q8_0_roundtrip`.

func runQwen38KVQuantCheck(dumpDir: String?) throws -> Bool {
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let lib = try MetalContext.moduleLibrary(device: device, module: "qwen38")
    func pso(_ name: String) throws -> MTLComputePipelineState {
        try device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    }
    let quant = try pso("q38_kv_quantize"), dequant = try pso("q38_kv_dequantize")
    let toHalf = try pso("q38_f32_to_f16"), toFloat = try pso("q38_f16_to_f32")
    let rows = 1024, D = 256, block = 34, rowBytes = D / 32 * block
    var x = [Float](repeating: 0, count: rows * D)
    var rng = SystemRandomNumberGenerator()
    func gauss() -> Float {
        let u = Float.random(in: Float.leastNonzeroMagnitude..<1, using: &rng), v = Float.random(in: 0..<1, using: &rng)
        return (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
    }
    for r in 0..<rows {
        let base = r * D
        switch r % 8 {
        case 0: for i in 0..<D { x[base + i] = gauss() }
        case 1: for i in 0..<D { x[base + i] = gauss() * 30 }
        case 2: for i in 0..<D { x[base + i] = gauss() * 1e-3 }
        case 3: for i in 0..<D { x[base + i] = gauss() * (i % 37 == 0 ? 200 : 1) }
        case 4: break                                                           // zeros
        case 5: for i in 0..<D { x[base + i] = Float(Int.random(in: -127...126, using: &rng)) + 0.5 }  // halfway
        case 6:
            for i in 0..<D { x[base + i] = Float(Int.random(in: -126...126, using: &rng)) + 0.5 }
            for k in 0..<(D / 32) { x[base + k * 32] = 127 }                    // d = 1 exactly
        default: for i in 0..<D { x[base + i] = gauss() * 1e-30 }               // subnormal d
        }
    }
    // Host ggml reference.
    var want = [UInt8](repeating: 0, count: rows * rowBytes)
    for r in 0..<rows {
        for k in 0..<(D / 32) {
            let xs = (0..<32).map { x[r * D + k * 32 + $0] }
            let amax = xs.map(abs).max()!
            let d = amax / 127
            let id: Float = d != 0 ? 1 / d : 0
            let o = r * rowBytes + k * block
            let dh = Float16(d).bitPattern
            want[o] = UInt8(dh & 0xFF)
            want[o + 1] = UInt8(dh >> 8)
            for j in 0..<32 { want[o + 2 + j] = UInt8(bitPattern: Int8((xs[j] * id).rounded(.toNearestOrAwayFromZero))) }
        }
    }
    let xBuf = device.makeBuffer(bytes: x, length: x.count * 4, options: .storageModeShared)!
    let row0 = 3
    let cache = device.makeBuffer(length: (rows + row0) * rowBytes, options: .storageModeShared)!
    let back = device.makeBuffer(length: rows * D * 4, options: .storageModeShared)!
    let half = device.makeBuffer(length: rows * D * 2, options: .storageModeShared)!
    let halfBack = device.makeBuffer(length: rows * D * 4, options: .storageModeShared)!
    let cb = queue.makeCommandBuffer()!
    func dispatch(_ p: MTLComputePipelineState, _ n: Int, _ setup: (MTLComputeCommandEncoder) -> Void) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        setup(enc)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(p.maxTotalThreadsPerThreadgroup, n), height: 1, depth: 1))
        enc.endEncoding()
    }
    var dV = UInt32(D), r0 = UInt32(row0)
    dispatch(quant, rows) { enc in
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(cache, offset: 0, index: 1)
        enc.setBytes(&dV, length: 4, index: 2)
        enc.setBytes(&r0, length: 4, index: 3)
    }
    dispatch(dequant, rows * D) { enc in
        enc.setBuffer(cache, offset: row0 * rowBytes, index: 0)
        enc.setBuffer(back, offset: 0, index: 1)
        enc.setBytes(&dV, length: 4, index: 2)
    }
    dispatch(toHalf, rows * D) { enc in
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(half, offset: 0, index: 1)
    }
    dispatch(toFloat, rows * D) { enc in
        enc.setBuffer(half, offset: 0, index: 0)
        enc.setBuffer(halfBack, offset: 0, index: 1)
    }
    cb.commit()
    cb.waitUntilCompleted()
    if let error = cb.error { throw error }

    let got = UnsafeRawBufferPointer(start: cache.contents() + row0 * rowBytes, count: rows * rowBytes)
    var badBlocks = 0, badRowsByKind = [Int](repeating: 0, count: 8)
    for r in 0..<rows {
        var rowBad = false
        for k in 0..<(D / 32) {
            let o = r * rowBytes + k * block
            if !(0..<block).allSatisfy({ got[o + $0] == want[o + $0] }) { badBlocks += 1; rowBad = true }
        }
        if rowBad { badRowsByKind[r % 8] += 1 }
    }
    var badDequant = 0, maxErr: Float = 0, maxScale: Float = 0
    let bp = back.contents().bindMemory(to: Float.self, capacity: rows * D)
    for r in 0..<rows {
        for i in 0..<(D) {
            let o = r * rowBytes + (i / 32) * block
            let d = Float(Float16(bitPattern: UInt16(want[o]) | UInt16(want[o + 1]) << 8))
            let q = Float(Int8(bitPattern: want[o + 2 + i % 32]))
            if bp[r * D + i].bitPattern != (d * q).bitPattern { badDequant += 1 }
            maxErr = max(maxErr, abs(bp[r * D + i] - x[r * D + i]))
            maxScale = max(maxScale, abs(x[r * D + i]))
        }
    }
    let hp = half.contents().bindMemory(to: UInt16.self, capacity: rows * D)
    let hb = halfBack.contents().bindMemory(to: Float.self, capacity: rows * D)
    var badHalf = 0, badHalfBack = 0
    for i in 0..<(rows * D) {
        if hp[i] != Float16(x[i]).bitPattern { badHalf += 1 }
        if hb[i].bitPattern != Float(Float16(bitPattern: hp[i])).bitPattern { badHalfBack += 1 }
    }
    let untouched = (0..<(row0 * rowBytes)).allSatisfy { cache.contents().load(fromByteOffset: $0, as: UInt8.self) == 0 }
    print("Qwen3.8 KV Q8_0 / F16 kernels, \(rows) rows x \(D)")
    print("  q38_kv_quantize vs ggml ref: \(badBlocks) / \(rows * D / 32) blocks differ (by row kind \(badRowsByKind)), rows before row0 untouched \(untouched)")
    print("  q38_kv_dequantize: \(badDequant) elements differ from d * q; max |x' - x| \(maxErr) (max |x| \(maxScale))")
    print("  q38_f32_to_f16: \(badHalf) differ; q38_f16_to_f32: \(badHalfBack) differ")
    if let dir = dumpDir {
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try Data(bytes: x, count: x.count * 4).write(to: URL(fileURLWithPath: dir + "/x.f32"))
        try Data(got).write(to: URL(fileURLWithPath: dir + "/q8.bin"))
        print("  dumped x.f32 and q8.bin to \(dir)")
    }
    let passed = badBlocks == 0 && badDequant == 0 && badHalf == 0 && badHalfBack == 0 && untouched
    print(passed ? "PASS" : "FAIL")
    return passed
}
