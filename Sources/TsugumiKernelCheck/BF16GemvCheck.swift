import Foundation
import Metal
import Tsugumi

// MARK: - `--bf16-gemv-check <gguf> <bf16 sidecar> [tokens,...]`
//
// docs/qwen38/15 §2 W-4: every tensor of the BF16 sidecar (`Scripts/qwen38/bf16_sidecar.py`) through
// `GGMLDenseGEMV` against the GGUF's F32 tensor of the same name, same seeded x, y compared byte for byte.
// T < 32 runs the lane kernels (chunk form for rows of 1024+), T >= 32 dequantizes and calls sgemm.

func runBF16GemvCheck(ggufPath: String, sidecarPath: String, tokens: [Int]) throws -> Bool {
    let file = try GGUFFile(url: URL(fileURLWithPath: (ggufPath as NSString).expandingTildeInPath))
    let side = try GGUFFile(url: URL(fileURLWithPath: (sidecarPath as NSString).expandingTildeInPath))
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return false }
    let dense = try GGMLDenseGEMV(device: device)
    var state: UInt64 = 0x2545_F491_4F6C_DD1D
    func uniform() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(state >> 40) / Float(1 << 24) * 2 - 1
    }
    var pass = true
    var checked = 0, differing = [Int: Int]()
    for name in side.tensors.keys.sorted() {
        let tb = side.tensors[name]!
        let tf = try file.tensor(name)
        precondition(tf.type == .f32 && tb.type == .bf16 && tf.dims == tb.dims, name)
        let (fb, fo) = file.noCopyBuffer(device: device, tensor: tf)!
        let (bb, bo) = side.noCopyBuffer(device: device, tensor: tb)!
        let m = tf.rowCount, n = tf.rowWidth
        for T in tokens {
            let xs = (0..<(T * n)).map { _ in uniform() }
            let x = xs.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
            let yf = device.makeBuffer(length: T * m * 4, options: .storageModeShared)!
            let yb = device.makeBuffer(length: T * m * 4, options: .storageModeShared)!
            let cb = queue.makeCommandBuffer()!
            dense.encode(commandBuffer: cb, type: .f32, weights: fb, weightsOffset: fo, x: x, y: yf, m: m, n: n, tokens: T)
            dense.encode(commandBuffer: cb, type: .bf16, weights: bb, weightsOffset: bo, x: x, y: yb, m: m, n: n, tokens: T)
            cb.commit(); cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            let d = memcmp(yf.contents(), yb.contents(), T * m * 4) == 0 ? 0 : {
                let a = yf.contents().bindMemory(to: UInt32.self, capacity: T * m)
                let b = yb.contents().bindMemory(to: UInt32.self, capacity: T * m)
                return (0..<(T * m)).reduce(0) { $0 + (a[$1] != b[$1] ? 1 : 0) }
            }()
            if d != 0 { pass = false; print("  \(name) T=\(T): \(d) / \(T * m) floats differ") }
            differing[T, default: 0] += d
            checked += 1
        }
        dense.dropScratch()
    }
    print("BF16 GEMV check: \(side.tensors.count) tensors x T in \(tokens), \(checked) cases; differing floats by T: "
          + tokens.map { "T=\($0) \(differing[$0] ?? 0)" }.joined(separator: ", "))
    print(pass ? "PASS" : "FAIL")
    return pass
}
