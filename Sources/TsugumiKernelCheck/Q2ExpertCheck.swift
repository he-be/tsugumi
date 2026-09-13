import Foundation
import Metal
import Tsugumi

// MARK: - Q2 の routed expert カーネル (`docs/investigations/QWEN38_FLASH_NEXT_VERIFY_PLAN.md`)
//
// Qwen3.8-Flash-Next の DS4-IQ2 GGUF は routed expert を IQ2_XXS (gate/up) と
// Q2_K (down、入力 640 → 768 列に pad) で持つ。本ランタイムの MoE カーネルは
// affine int4/int8 しか読めないので、`moe_ggml.metal` に 2 本を足した。
// このチェックが答えるのは 2 つ:
//
//   1. 算式が合っているか。`Scripts/qwen38/expert_kernel_fixture.py` が実物の
//      1 層から切り出した expert 10 個と、gguf-py で逆量子化した float64 の
//      正解に対して、acts と y を突き合わせる。
//   2. Metal で遅くないか。同じ D=2560 / F=640 の affine int4 g64 (Tsugumi の
//      `moe_phase1_gate_up_act_u16load` / `moe_phase2_down_reduce_k8`、合成の重み)
//      と、k=8 で GPU 時間を並べる。どちらも常駐済み・I/O 無し・汎用 PSO
//      (形の function constant 無し) で、差はカーネルと読むバイト数だけ。
//
//     swift run -c release TsugumiKernelCheck --q2-expert scratch/qwen38/expert-fixture-l24
//     swift run -c release TsugumiKernelCheck --q2-expert scratch/qwen38/expert-fixture-l24 --q2-expert-bench 20

private struct Q2Fixture {
    let d: Int
    let f: Int
    let downIn: Int
    let topK: Int
    let gateOff: Int
    let upOff: Int
    let downOff: Int
    let gateRowBytes: Int
    let downRowBytes: Int
    let blobs: [Data]
    let x: [Float16]
    let residual: [Float16]
    let weights: [Float16]
    let refActs: [Float]
    let refY: [Float]
}

private func q2ReadArray<T>(_ url: URL, as: T.Type) throws -> [T] {
    let data = try Data(contentsOf: url)
    return data.withUnsafeBytes { raw in
        Array(raw.bindMemory(to: T.self))
    }
}

private func loadQ2Fixture(_ dir: URL) throws -> Q2Fixture {
    let meta = try JSONSerialization.jsonObject(
        with: Data(contentsOf: dir.appendingPathComponent("meta.json"))) as! [String: Any]
    let d = meta["D"] as! Int
    let f = meta["F"] as! Int
    let downIn = meta["down_in"] as! Int
    let topK = meta["top_k"] as! Int
    let offsets = meta["offsets"] as! [[String: Any]]
    let all = try Data(contentsOf: dir.appendingPathComponent("blobs.bin"))
    let blobs = offsets.map { entry -> Data in
        let base = entry["base"] as! Int
        let stride = entry["stride"] as! Int
        return all.subdata(in: base..<(base + stride))
    }
    let first = offsets[0]
    let gateBytes = meta["gate_bytes"] as! Int
    let downBytes = meta["down_bytes"] as! Int
    return Q2Fixture(
        d: d, f: f, downIn: downIn, topK: topK,
        gateOff: first["gate"] as! Int, upOff: first["up"] as! Int,
        downOff: first["down"] as! Int,
        gateRowBytes: gateBytes / f, downRowBytes: downBytes / d,
        blobs: blobs,
        x: try q2ReadArray(dir.appendingPathComponent("x.f16"), as: Float16.self),
        residual: try q2ReadArray(dir.appendingPathComponent("residual.f16"), as: Float16.self),
        weights: try q2ReadArray(dir.appendingPathComponent("weights.f16"), as: Float16.self),
        refActs: try q2ReadArray(dir.appendingPathComponent("ref_acts.f32"), as: Float.self),
        refY: try q2ReadArray(dir.appendingPathComponent("ref_y.f32"), as: Float.self))
}

/// `GgmlExpertOffsets` (`moe_ggml.metal`).
private struct Q2ExpertOffsetsMSL {
    var gateRowBytes: UInt32
    var downRowBytes: UInt32
}

private let q2ThreadsPerGroup = 64          // 2 SIMD groups x 32 lanes, fixed by the codebook copy
private let q2RowsPerThreadgroup = 8        // 2 groups x 4 rows
private let q2CodebookBytes = 256 * 8 + 128

private func q2Buffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer {
    values.withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: max(raw.count, 4),
                          options: .storageModeShared)!
    }
}

private func q2Read<T>(_ buffer: MTLBuffer, count: Int, as: T.Type) -> [T] {
    Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: T.self, capacity: count),
                              count: count))
}

private final class Q2Kernels {
    let phase1: MTLComputePipelineState
    let phase2: MTLComputePipelineState
    let argEncoder: MTLArgumentEncoder

    init(device: MTLDevice) throws {
        let library = try MetalContext.moduleLibrary(device: device, module: "moe_ggml")
        guard let f1 = library.makeFunction(name: "moe_iq2xxs_phase1_gate_up_act"),
              let f2 = library.makeFunction(name: "moe_q2k_phase2_down_reduce") else {
            throw NSError(domain: "Q2ExpertCheck", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "moe_ggml kernels missing"])
        }
        phase1 = try device.makeComputePipelineState(function: f1)
        phase2 = try device.makeComputePipelineState(function: f2)
        argEncoder = f1.makeArgumentEncoder(bufferIndex: 0)
    }
}

/// One routed MoE layer's worth of buffers for the Q2 kernels.
private struct Q2Layer {
    let blobs: [MTLBuffer]
    let argBuffer: MTLBuffer
    var offsets: Q2ExpertOffsetsMSL
    let partOffsets: MTLBuffer
    let pairSlot: MTLBuffer   // one token: pair k is slot k
    let x: MTLBuffer
    let acts: MTLBuffer
    let weights: MTLBuffer
    let residual: MTLBuffer
    let y: MTLBuffer
    let d: Int
    let f: Int
    let actStride: Int
    let topK: Int
}

private func makeQ2Layer(_ fx: Q2Fixture, kernels: Q2Kernels, device: MTLDevice,
                         topK: Int) -> Q2Layer {
    let blobs = fx.blobs.prefix(topK).map { data in
        data.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count,
                              options: .storageModeShared)!
        }
    }
    let arg = device.makeBuffer(length: kernels.argEncoder.encodedLength,
                                options: .storageModeShared)!
    kernels.argEncoder.setArgumentBuffer(arg, offset: 0)
    // One contiguous blob per slot: gate, up and down all point at it.
    for (i, blob) in blobs.enumerated() {
        kernels.argEncoder.setBuffer(blob, offset: 0, index: i)
        kernels.argEncoder.setBuffer(blob, offset: 0, index: 512 + i)
        kernels.argEncoder.setBuffer(blob, offset: 0, index: 1024 + i)
    }
    let parts = (0..<topK).flatMap { _ in [UInt32(fx.gateOff), UInt32(fx.upOff), UInt32(fx.downOff)] }
    return Q2Layer(
        blobs: Array(blobs), argBuffer: arg,
        offsets: Q2ExpertOffsetsMSL(gateRowBytes: UInt32(fx.gateRowBytes),
                                    downRowBytes: UInt32(fx.downRowBytes)),
        partOffsets: q2Buffer(device, parts),
        pairSlot: q2Buffer(device, (0..<topK).map { UInt32($0) }),
        x: q2Buffer(device, fx.x.map { Float($0) }),
        acts: q2Buffer(device, [Float](repeating: 0, count: topK * fx.downIn)),
        weights: q2Buffer(device, fx.weights.prefix(topK).map { Float($0) }),
        residual: q2Buffer(device, fx.residual.map { Float($0) }),
        y: q2Buffer(device, [Float](repeating: 0, count: fx.d)),
        d: fx.d, f: fx.f, actStride: fx.downIn, topK: topK)
}

private func encodeQ2Phase1(_ layer: Q2Layer, kernels: Q2Kernels, cb: MTLCommandBuffer) {
    var offsets = layer.offsets
    var d = UInt32(layer.d), f = UInt32(layer.f), k = UInt32(layer.topK)
    var stride = UInt32(layer.actStride)
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(kernels.phase1)
    enc.setBuffer(layer.argBuffer, offset: 0, index: 0)
    for blob in layer.blobs { enc.useResource(blob, usage: .read) }
    enc.setBytes(&offsets, length: MemoryLayout<Q2ExpertOffsetsMSL>.stride, index: 1)
    enc.setBuffer(layer.x, offset: 0, index: 2)
    enc.setBuffer(layer.acts, offset: 0, index: 3)
    enc.setBytes(&d, length: 4, index: 4)
    enc.setBytes(&f, length: 4, index: 5)
    enc.setBytes(&k, length: 4, index: 6)
    enc.setBytes(&stride, length: 4, index: 7)
    enc.setBuffer(layer.partOffsets, offset: 0, index: 8)
    var tokens = UInt32(1)
    enc.setBytes(&tokens, length: 4, index: 9)
    enc.setBuffer(layer.pairSlot, offset: 0, index: 10)
    enc.setThreadgroupMemoryLength(q2CodebookBytes, index: 0)
    let rows = layer.topK * layer.f
    enc.dispatchThreadgroups(
        MTLSize(width: (rows + q2RowsPerThreadgroup - 1) / q2RowsPerThreadgroup, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: q2ThreadsPerGroup, height: 1, depth: 1))
    enc.endEncoding()
}

private func encodeQ2Phase2(_ layer: Q2Layer, kernels: Q2Kernels, cb: MTLCommandBuffer) {
    var offsets = layer.offsets
    var d = UInt32(layer.d), k = UInt32(layer.topK)
    var stride = UInt32(layer.actStride)
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(kernels.phase2)
    enc.setBuffer(layer.argBuffer, offset: 0, index: 0)
    for blob in layer.blobs { enc.useResource(blob, usage: .read) }
    enc.setBytes(&offsets, length: MemoryLayout<Q2ExpertOffsetsMSL>.stride, index: 1)
    enc.setBuffer(layer.acts, offset: 0, index: 2)
    enc.setBuffer(layer.weights, offset: 0, index: 3)
    enc.setBuffer(layer.residual, offset: 0, index: 4)
    enc.setBuffer(layer.y, offset: 0, index: 5)
    enc.setBytes(&d, length: 4, index: 6)
    enc.setBytes(&stride, length: 4, index: 7)
    enc.setBytes(&k, length: 4, index: 8)
    enc.setBuffer(layer.partOffsets, offset: 0, index: 9)
    var tokens = UInt32(1)
    enc.setBytes(&tokens, length: 4, index: 10)
    enc.setBuffer(layer.pairSlot, offset: 0, index: 11)
    enc.dispatchThreadgroups(
        MTLSize(width: (layer.d + q2RowsPerThreadgroup - 1) / q2RowsPerThreadgroup, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: q2ThreadsPerGroup, height: 1, depth: 1))
    enc.endEncoding()
}

/// max|a - r| / max|r|, NaN を素通しさせない。
private func q2Relative(_ actual: [Float], _ reference: [Float]) -> Double {
    precondition(actual.count == reference.count, "shape mismatch — harness bug")
    var maxDiff = 0.0, refNorm = 0.0
    for i in actual.indices {
        refNorm = max(refNorm, abs(Double(reference[i])))
        if !actual[i].isFinite { return .infinity }
        maxDiff = max(maxDiff, abs(Double(actual[i]) - Double(reference[i])))
    }
    precondition(refNorm > 1e-4, "reference has no signal — harness bug")
    return maxDiff / refNorm
}

/// 算式の突き合わせ。通れば true。
func runQ2ExpertCheck(fixtureDir: String) throws -> Bool {
    let fx = try loadQ2Fixture(URL(fileURLWithPath: fixtureDir))
    let context = try MetalContext()
    let device = context.device
    let kernels = try Q2Kernels(device: device)
    print("Q2 routed expert check: D=\(fx.d) F=\(fx.f) down_in=\(fx.downIn) k=\(fx.topK) "
          + "gate row \(fx.gateRowBytes) B, down row \(fx.downRowBytes) B, "
          + "expert \(fx.blobs[0].count) B")

    let layer = makeQ2Layer(fx, kernels: kernels, device: device, topK: fx.topK)
    let cb = context.queue.makeCommandBuffer()!
    encodeQ2Phase1(layer, kernels: kernels, cb: cb)
    encodeQ2Phase2(layer, kernels: kernels, cb: cb)
    cb.commit()
    cb.waitUntilCompleted()
    if let error = cb.error { throw error }

    let actsAll = q2Read(layer.acts, count: fx.topK * fx.downIn, as: Float.self)
    var acts: [Float] = []
    var padNonZero = 0
    for slot in 0..<fx.topK {
        let row = actsAll[(slot * fx.downIn)..<((slot + 1) * fx.downIn)]
        acts.append(contentsOf: row.prefix(fx.f).map { Float($0) })
        padNonZero += row.dropFirst(fx.f).filter { $0 != 0 }.count
    }
    let y = q2Read(layer.y, count: fx.d, as: Float.self)
    let actsErr = q2Relative(acts, fx.refActs)
    let yErr = q2Relative(y, fx.refY)
    // 何もしないカーネル (y = residual) がどれだけ外れるか: 閾値がそれより十分小さいことを見る。
    let identityErr = q2Relative(fx.residual.map { Float($0) }, fx.refY)
    let threshold = 1e-5
    let pass = actsErr < threshold && yErr < threshold && padNonZero == 0
    print(String(format: "  acts  rel err %.3e   (threshold %.0e)", actsErr, threshold))
    print(String(format: "  y     rel err %.3e   (y = residual would be %.3e)", yErr, identityErr))
    print("  act pad (cols \(fx.f)..<\(fx.downIn)) nonzero: \(padNonZero)")
    print("  \(pass ? "PASS" : "FAIL")")
    return pass
}

// MARK: - Bench

private struct Int4Layer {
    let blobs: [MTLBuffer]
    let argBuffer: MTLBuffer
    let offsets: MoEExpertOffsets
    let x: MTLBuffer
    let acts: MTLBuffer
    let weights: MTLBuffer
    let residual: MTLBuffer
    let y: MTLBuffer
}

private func makeInt4Layer(moe: MoE, device: MTLDevice, d: Int, f: Int, groupSize: Int,
                           topK: Int) -> (Int4Layer, Int) {
    // gate / up are [F, D], down is [D, F]: packed nibbles, BF16 scales, BF16 biases.
    var cursor = 0
    func advance(rows: Int, cols: Int) -> (UInt32, UInt32, UInt32) {
        let w = cursor; cursor += rows * cols / 2
        let s = cursor; cursor += rows * (cols / groupSize) * 2
        let b = cursor; cursor += rows * (cols / groupSize) * 2
        return (UInt32(w), UInt32(s), UInt32(b))
    }
    let g = advance(rows: f, cols: d)
    let u = advance(rows: f, cols: d)
    let dn = advance(rows: d, cols: f)
    let stride = cursor
    let offsets = MoEExpertOffsets(gateWOff: g.0, gateSOff: g.1, gateBOff: g.2,
                                   upWOff: u.0, upSOff: u.1, upBOff: u.2,
                                   downWOff: dn.0, downSOff: dn.1, downBOff: dn.2)
    var state: UInt64 = 0x9E3779B97F4A7C15
    let blobs = (0..<topK).map { _ -> MTLBuffer in
        let buffer = device.makeBuffer(length: stride, options: .storageModeShared)!
        let bytes = buffer.contents().bindMemory(to: UInt8.self, capacity: stride)
        for i in 0..<stride {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            bytes[i] = UInt8(truncatingIfNeeded: state >> 33)
        }
        let words = buffer.contents().bindMemory(to: UInt16.self, capacity: stride / 2)
        func fill(_ start: UInt32, _ end: UInt32, _ value: UInt16) {
            for w in Int(start / 2)..<Int(end / 2) { words[w] = value }
        }
        fill(g.1, g.2, 0x3C00); fill(g.2, u.0, 0x3B00)
        fill(u.1, u.2, 0x3C00); fill(u.2, dn.0, 0x3B00)
        fill(dn.1, dn.2, 0x3C00); fill(dn.2, UInt32(stride), 0x3B00)
        return buffer
    }
    let x = (0..<d).map { Float16(Float($0 % 61) * 0.01 - 0.3) }
    let layer = Int4Layer(
        blobs: blobs,
        argBuffer: moe.makeRoutedArgumentBuffer(routedBlobs: blobs, topK: UInt32(topK))!,
        offsets: offsets,
        x: q2Buffer(device, x),
        acts: q2Buffer(device, [Float16](repeating: 0, count: topK * f)),
        weights: q2Buffer(device, [Float16](repeating: 0.125, count: topK)),
        residual: q2Buffer(device, [Float16](repeating: 0, count: d)),
        y: q2Buffer(device, [Float16](repeating: 0, count: d)))
    return (layer, stride)
}

private enum Q2BenchStage: String, CaseIterable {
    case gateUp = "gate/up"
    case down = "down"
    case both = "layer"
}

/// `layersPerBuffer` 回を 1 本の command buffer に積み、GPU 時間 / 層の中央値を返す (ms)。
private func q2Time(context: MetalContext, iterations: Int, layersPerBuffer: Int,
                    encode: (MTLCommandBuffer) -> Void) -> (median: Double, p10: Double, p90: Double) {
    var samples: [Double] = []
    for i in 0..<(iterations + 2) {
        let cb = context.queue.makeCommandBuffer()!
        for _ in 0..<layersPerBuffer { encode(cb) }
        cb.commit()
        cb.waitUntilCompleted()
        if i >= 2 {  // 最初の 2 本は暖機
            samples.append((cb.gpuEndTime - cb.gpuStartTime) * 1000 / Double(layersPerBuffer))
        }
    }
    samples.sort()
    func at(_ q: Double) -> Double { samples[min(samples.count - 1, Int(Double(samples.count) * q))] }
    return (at(0.5), at(0.1), at(0.9))
}

func runQ2ExpertBench(fixtureDir: String, iterations: Int) throws {
    let fx = try loadQ2Fixture(URL(fileURLWithPath: fixtureDir))
    let context = try MetalContext()
    try context.setAffineGroupSize(64)
    let device = context.device
    let kernels = try Q2Kernels(device: device)
    let moe = try MoE(context: context, gateActivation: .silu)
    let layersPerBuffer = 48

    let q2k10 = makeQ2Layer(fx, kernels: kernels, device: device, topK: fx.topK)
    let q2k8 = makeQ2Layer(fx, kernels: kernels, device: device, topK: 8)
    let (int4, int4Stride) = makeInt4Layer(moe: moe, device: device, d: fx.d, f: fx.f,
                                           groupSize: 64, topK: 8)
    print("Q2 routed expert bench: \(device.name), D=\(fx.d) F=\(fx.f), "
          + "\(layersPerBuffer) layers per command buffer, \(iterations) buffers per cell (median p10-p90)")
    print("  bytes/expert: IQ2_XXS+Q2_K \(fx.blobs[0].count), affine int4 g64 \(int4Stride)")

    func int4Encode(_ stage: Q2BenchStage, _ cb: MTLCommandBuffer) {
        if stage != .down {
            moe.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: cb, routedArgBuffer: int4.argBuffer, routedBlobs: int4.blobs,
                routedOffsets: int4.offsets, x: int4.x, acts: int4.acts,
                d: UInt32(fx.d), f: UInt32(fx.f), topK: 8)
        }
        if stage != .gateUp {
            moe.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: cb, routedArgBuffer: int4.argBuffer, routedBlobs: int4.blobs,
                routedOffsets: int4.offsets, acts: int4.acts, routingWeights: int4.weights,
                residual: int4.residual, y: int4.y, d: UInt32(fx.d), f: UInt32(fx.f), topK: 8)
        }
    }
    func q2Encode(_ layer: Q2Layer, _ stage: Q2BenchStage, _ cb: MTLCommandBuffer) {
        if stage != .down { encodeQ2Phase1(layer, kernels: kernels, cb: cb) }
        if stage != .gateUp { encodeQ2Phase2(layer, kernels: kernels, cb: cb) }
    }

    // ABBA: int4 k8, q2 k8, q2 k8, int4 k8 — 熱や周波数の漂流を両側に振る。q2 k10 は最後に 1 回。
    for stage in Q2BenchStage.allCases {
        var a: [Double] = [], b: [Double] = []
        let r1 = q2Time(context: context, iterations: iterations, layersPerBuffer: layersPerBuffer) { int4Encode(stage, $0) }
        let r2 = q2Time(context: context, iterations: iterations, layersPerBuffer: layersPerBuffer) { q2Encode(q2k8, stage, $0) }
        let r3 = q2Time(context: context, iterations: iterations, layersPerBuffer: layersPerBuffer) { q2Encode(q2k8, stage, $0) }
        let r4 = q2Time(context: context, iterations: iterations, layersPerBuffer: layersPerBuffer) { int4Encode(stage, $0) }
        let r5 = q2Time(context: context, iterations: iterations, layersPerBuffer: layersPerBuffer) { q2Encode(q2k10, stage, $0) }
        a = [r1.median, r4.median]; b = [r2.median, r3.median]
        let int4Ms = a.reduce(0, +) / 2, q2Ms = b.reduce(0, +) / 2
        print(String(format: "  %-8@ int4 k8 %.3f / %.3f ms   q2 k8 %.3f / %.3f ms   ratio q2/int4 %.2f   q2 k10 %.3f ms (p10 %.3f p90 %.3f)",
                     stage.rawValue as NSString, r1.median, r4.median, r2.median, r3.median,
                     q2Ms / int4Ms, r5.median, r5.p10, r5.p90))
        if stage == .both {
            print(String(format: "  x48 layers: int4 k8 %.1f ms/token, q2 k8 %.1f ms/token, q2 k10 %.1f ms/token (routed experts only, resident)",
                         int4Ms * 48, q2Ms * 48, r5.median * 48))
        }
    }
}
