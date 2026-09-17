import Foundation
import Metal
import Tsugumi

// MARK: - Qwen3.8-27B dense GEMV (`ggml_iq.metal`) against the gguf-py reference
//
// Fixture from `Scripts/qwen38_27b/dense_kernel_fixture.py`: per case a real
// tensor of the 27B GGUF, x [T, n] and y = x W^T [T, m] summed in float64 over
// gguf-py's dequantization. The kernel reads the tensor through a `GGUFFile`
// no-copy buffer, T tokens in one dispatch, once on the direct kernels and
// once on the dequant + MPS sgemm path (`mpsMinTokens` = 1; the 248K-row
// tensors stay direct). Also prints the GPU time of one token on the direct
// kernel (median of 10), the decode shape (docs/qwen38-27b/01 §3-1, 03).
//
//     .build/release/TsugumiKernelCheck --q27-dense scratch/qwen38_27b/dense-fixture

private struct Q27DenseFixture: Decodable {
    struct Case: Decodable {
        let name: String
        let type: String
        let m: Int
        let n: Int
        let tokens: Int
        let x: String
        let y: String
    }
    let gguf: String
    let cases: [Case]
}

func runQ27DenseCheck(fixtureDir: String) throws -> Bool {
    let dir = URL(fileURLWithPath: (fixtureDir as NSString).expandingTildeInPath)
    let fixture = try JSONDecoder().decode(Q27DenseFixture.self, from: Data(contentsOf: dir.appendingPathComponent("meta.json")))
    let file = try GGUFFile(url: URL(fileURLWithPath: fixture.gguf))
    let context = try MetalContext()
    let device = context.device
    let gemv = try GGMLDenseGEMV(device: device)
    gemv.mpsMaxWeights = 128 << 20
    var allPass = true
    print("Qwen3.8-27B dense GEMV check: \(file.url.lastPathComponent)")
    for c in fixture.cases {
        let t = try file.tensor(c.name)
        precondition(t.rowCount == c.m && t.rowWidth == c.n, "\(c.name): shape mismatch")
        let x = try Data(contentsOf: dir.appendingPathComponent(c.x))
        let ref = try Data(contentsOf: dir.appendingPathComponent(c.y)).withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
        precondition(x.count == c.tokens * c.n * 4 && ref.count == c.tokens * c.m)
        guard let (wbuf, woff) = file.noCopyBuffer(device: device, tensor: t),
              let xbuf = x.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: x.count, options: .storageModeShared) }),
              let ybuf = device.makeBuffer(length: c.tokens * c.m * 4, options: .storageModeShared) else {
            throw GGUFFile.Error.format("buffer allocation failed for \(c.name)")
        }
        func relErr(mpsMinTokens: Int) throws -> Double {
            gemv.mpsMinTokens = mpsMinTokens
            let cb = context.queue.makeCommandBuffer()!
            gemv.encode(commandBuffer: cb, type: t.type, weights: wbuf, weightsOffset: woff,
                        x: xbuf, y: ybuf, m: c.m, n: c.n, tokens: c.tokens)
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            let y = UnsafeBufferPointer(start: ybuf.contents().bindMemory(to: Float.self, capacity: c.tokens * c.m),
                                        count: c.tokens * c.m)
            var maxDiff = 0.0, refMax = 0.0
            for i in 0..<ref.count {
                refMax = max(refMax, abs(ref[i]))
                maxDiff = max(maxDiff, abs(Double(y[i]) - ref[i]))
            }
            return refMax > 0 ? maxDiff / refMax : .infinity
        }
        let rel = try relErr(mpsMinTokens: 0)
        let sgemm = c.m * c.n <= gemv.mpsMaxWeights
        let relSgemm = sgemm ? try relErr(mpsMinTokens: 1) : 0
        gemv.mpsMinTokens = 0
        gemv.dropScratch()

        var times: [Double] = []
        for _ in 0..<10 {
            let cb = context.queue.makeCommandBuffer()!
            gemv.encode(commandBuffer: cb, type: t.type, weights: wbuf, weightsOffset: woff,
                        x: xbuf, y: ybuf, m: c.m, n: c.n, tokens: 1)
            cb.commit()
            cb.waitUntilCompleted()
            times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
        }
        times.sort()

        let pass = rel < 1e-5 && relSgemm < 1e-5
        allPass = allPass && pass
        let sgemmText = sgemm ? String(format: "%.2e", relSgemm) : "   (direct)"
        print(String(format: "  %@ %@ [%d x %d] T=%d  rel err direct %.2e sgemm %@  1 token %.2f ms  %@",
                     c.type as NSString, c.name as NSString, c.m, c.n, c.tokens, rel, sgemmText as NSString,
                     times[times.count / 2], pass ? "PASS" : "FAIL"))
    }
    print("  \(allPass ? "PASS" : "FAIL")")
    return allPass
}
