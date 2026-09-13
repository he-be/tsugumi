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
