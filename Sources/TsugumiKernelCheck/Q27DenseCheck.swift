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

// MARK: - Qwen3.8-27B runner against the CPU reference (`--q27-decode <ref log> [--q27-ref-logits F] [--q27-chunk N]`)
//
// `Qwen38DenseRunner` runs the reference's prompt plus its greedy continuation (`Scripts/qwen38_27b/reference_forward.py`
// log) `chunk` tokens at a time with every position's logits kept (chunk 1: the decode step), and compares each
// position's top-1 and, with the reference's `--dump-logits` file, the relative logit error (docs/qwen38-27b/04).

func runQ27DecodeCheck(refLog: String, refLogits: String?, gguf: String, chunk: Int) throws -> Bool {
    let ref = try parseQwen38RefLog(refLog)
    let n = ref.top1.count
    precondition(!ref.prompt.isEmpty && n > 0, "reference log has no prompt or positions")
    var seq = ref.prompt
    while seq.count < n { seq.append(ref.top1[seq.count - 1]) }
    setvbuf(stdout, nil, _IOLBF, 0)
    let runner = try Qwen38DenseRunner(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                       capacity: n + 1, maxBatch: chunk)
    var refLogitsData: Data?
    if let refLogits { refLogitsData = try Data(contentsOf: URL(fileURLWithPath: refLogits)) }
    print("Qwen3.8-27B check: \(n) positions in chunks of \(chunk), KV \(runner.kvQ8 ? "q8_0" : "f32") (\(refLog))")
    var mismatches = 0
    var worstLogit = 0.0
    var start = 0
    while start < n {
        let T = min(chunk, n - start)
        let all = try runner.forward(tokens: Array(seq[start..<(start + T)]), startPos: start, allLogits: true)
        let pr = runner.lastProfile
        for t in 0..<T {
            let pos = start + t
            let logits = UnsafeBufferPointer(rebasing: all[(t * runner.vocab)..<((t + 1) * runner.vocab)])
            var best = 0
            for i in 1..<logits.count where logits[i] > logits[best] { best = i }
            let ok = best == ref.top1[pos]
            if !ok { mismatches += 1 }
            var note = ""
            if let data = refLogitsData {
                data.withUnsafeBytes { raw in
                    let r = raw.bindMemory(to: Float.self)
                    let v = runner.vocab
                    var maxDiff = 0.0, refMax = 0.0
                    for i in 0..<v {
                        refMax = max(refMax, abs(Double(r[pos * v + i])))
                        maxDiff = max(maxDiff, abs(Double(logits[i]) - Double(r[pos * v + i])))
                    }
                    worstLogit = max(worstLogit, maxDiff / refMax)
                    note = String(format: "  logit rel err %.2e", maxDiff / refMax)
                }
            }
            let timing = t == T - 1
                ? String(format: "  %.2fs (embed %.0f layers %.0f head %.0f ms, gpu %.0f ms)", pr.total, pr.embed * 1000,
                         pr.layers * 1000, pr.head * 1000, pr.gpu * 1000)
                : ""
            print(String(format: "  [%3d] input %7d  top1 %7d  ref %7d  %@%@%@", pos, seq[pos], best, ref.top1[pos],
                         ok ? "ok  " : "DIFF", note, timing))
        }
        start += T
    }
    let pass = mismatches == 0 && (refLogitsData == nil || worstLogit < qwen38LogitTolerance())
    print("  top-1 mismatches: \(mismatches)" + (refLogitsData != nil ? String(format: ", worst logit rel err %.2e", worstLogit) : ""))
    print("  \(pass ? "PASS" : "FAIL")")
    return pass
}

// MARK: - Qwen3.8-27B prefill + decode speed (`--q27-bench <token file> [--q27-tokens N] [--q27-chunk C] [--q27-decode-steps D]`)
//
// N token ids (comma-separated file) through `forward` C at a time (last logits only), the prefill scratch dropped,
// then D greedy decode steps. Prints each chunk and each step (docs/qwen38-27b/05). Run it under
// `Scripts/qwen38/guarded.sh` with `GUARD_LOG` for wired / Swapouts.
func runQ27Bench(tokenFile: String, tokens: Int, chunk: Int, decodeSteps: Int, gguf: String) throws -> Bool {
    let text = try String(contentsOfFile: (tokenFile as NSString).expandingTildeInPath, encoding: .utf8)
    let ids = text.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    let n = min(tokens, ids.count)
    setvbuf(stdout, nil, _IOLBF, 0)
    let runner = try Qwen38DenseRunner(gguf: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath),
                                       capacity: n + decodeSteps + 1, maxBatch: chunk)
    print("Qwen3.8-27B bench: \(n) tokens in chunks of \(chunk), \(decodeSteps) decode steps, KV \(runner.kvQ8 ? "q8_0" : "f32"), resident \(runner.denseResident)")
    let tAll = Date()
    var start = 0
    var last = 0
    while start < n {
        let T = min(chunk, n - start)
        let t0 = Date()
        let logits = try runner.forward(tokens: Array(ids[start..<(start + T)]), startPos: start)
        var best = 0
        for i in 1..<logits.count where logits[i] > logits[best] { best = i }
        last = best
        let pr = runner.lastProfile
        print(String(format: "  prefill [%5d..<%5d] %.2fs (gpu %.2fs) %.1f tok/s", start, start + T,
                     Date().timeIntervalSince(t0), pr.gpu, Double(T) / Date().timeIntervalSince(t0)))
        start += T
    }
    let prefill = Date().timeIntervalSince(tAll)
    print(String(format: "  prefill total %.1fs, %.1f tok/s", prefill, Double(n) / prefill))
    runner.dropScratch()
    var times: [Double] = []
    var token = last
    var generated: [Int] = []
    for i in 0..<decodeSteps {
        let t0 = Date()
        let logits = try runner.step(token: token, pos: n + i)
        let dt = Date().timeIntervalSince(t0)
        times.append(dt)
        generated.append(token)
        var best = 0
        for j in 1..<logits.count where logits[j] > logits[best] { best = j }
        token = best
        let pr = runner.lastProfile
        print(String(format: "  decode [%5d] %.3fs (gpu %.3fs)", n + i, dt, pr.gpu))
    }
    if times.count > 1 {
        let steady = times.dropFirst().sorted()
        let median = steady[steady.count / 2]
        print(String(format: "  decode median (steps 2..) %.3fs = %.2f tok/s, first %.3fs", median, 1 / median, times[0]))
    }
    print("  generated ids: \(generated.map(String.init).joined(separator: ","))")
    return true
}
