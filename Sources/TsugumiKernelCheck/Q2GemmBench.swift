import Foundation
import Metal
import MetalPerformanceShaders
import Tsugumi

// MARK: - `--q2-gemm-bench <gguf> <route .bin> [group experts] [iterations]`
//
// One layer's routed experts for a real batch (`Q38_DUMP_ROUTE`, docs/qwen38/06): the runner's
// per-pair IQ2_XXS / Q2_K kernels (A) against gathering the pairs by expert, expanding each
// expert's gate/up/down to float32 once and multiplying with MPS sgemm, then scattering (B).
// Both arms get the same random inputs and residual; the relative difference of y is printed.

func runQ2GemmBench(ggufPath: String, routePath: String, group G: Int, iterations: Int, tokens: Int = .max) throws {
    let file = try GGUFFile(url: URL(fileURLWithPath: (ggufPath as NSString).expandingTildeInPath))
    let context = try MetalContext()
    let device = context.device
    let queue = context.queue
    let lib = try MetalContext.moduleLibrary(device: device, module: "moe_ggml")
    func pso(_ name: String) throws -> MTLComputePipelineState {
        try device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    }
    let psoPhase1 = try pso("moe_iq2xxs_phase1_gate_up_act")
    let psoPhase2 = try pso("moe_q2k_phase2_down_reduce")
    let psoDeqGU = try pso("moe_iq2xxs_dequant_gate_up_f32")
    let psoDeqDown = try pso("moe_q2k_dequant_down_f32")
    let psoGather = try pso("moe_gather_pair_rows")
    let psoSilu = try pso("moe_silu_mul_halves")
    let psoScatter = try pso("moe_scatter_weighted")

    let D = 2560, F = 640, actStride = 768, nExperts = 512, topK = 10
    let il = Int(routePath.split(separator: "-").last!.dropFirst().split(separator: ".").first!)!
    let route = try Data(contentsOf: URL(fileURLWithPath: routePath))
    let T0 = Int(route.withUnsafeBytes { $0.load(as: Int32.self) })
    let P0 = T0 * topK
    let T = min(T0, tokens)   // the first `tokens` tokens of the batch
    let P = T * topK
    var experts = [Int32](repeating: 0, count: P)
    var weights = [Float](repeating: 0, count: P)
    route.withUnsafeBytes { raw in
        experts.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[4..<(4 + 4 * P)])) }
        weights.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[(4 + 4 * P0)..<(4 + 4 * P0 + 4 * P)])) }
    }

    // Slots in first-seen order (as the runner), pairs sorted by slot for B.
    var slotOf = [Int](repeating: -1, count: nExperts)
    var slots: [Int] = []
    var pairSlot = [UInt32](repeating: 0, count: P)
    for p in 0..<P {
        let ex = Int(experts[p])
        if slotOf[ex] < 0 { slotOf[ex] = slots.count; slots.append(ex) }
        pairSlot[p] = UInt32(slotOf[ex])
    }
    let S = slots.count
    var count = [Int](repeating: 0, count: S)
    for p in 0..<P { count[Int(pairSlot[p])] += 1 }
    var start = [Int](repeating: 0, count: S + 1)
    for s in 0..<S { start[s + 1] = start[s] + count[s] }
    var fill = start
    var order = [UInt32](repeating: 0, count: P)   // sorted position -> pair
    var at = [UInt32](repeating: 0, count: P)      // pair -> sorted position
    for p in 0..<P {
        let s = Int(pairSlot[p])
        order[fill[s]] = UInt32(p)
        at[p] = UInt32(fill[s])
        fill[s] += 1
    }
    let sortedCounts = count.sorted()
    print(String(format: "layer %d, T=%d: %d experts, pairs per expert median %d / max %d / min %d, %d under 8",
                 il, T, S, sortedCounts[S / 2], sortedCounts.last!, sortedCounts.first!,
                 count.filter { $0 < 8 }.count))

    let gateT = try file.tensor("blk.\(il).ffn_gate_exps.weight")
    let upT = try file.tensor("blk.\(il).ffn_up_exps.weight")
    let downT = try file.tensor("blk.\(il).ffn_down_exps.weight")
    let gateRow = gateT.bytesPerRow, downRow = downT.bytesPerRow
    file.preadRanges([(gateT.offset, gateT.byteCount), (upT.offset, upT.byteCount), (downT.offset, downT.byteCount)], threads: 4)
    let (gBuf, gOff) = file.noCopyBuffer(device: device, tensor: gateT)!
    let (uBuf, uOff) = file.noCopyBuffer(device: device, tensor: upT)!
    let (dBuf, dOff) = file.noCopyBuffer(device: device, tensor: downT)!

    func shared<T>(_ a: [T]) -> MTLBuffer {
        a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 1), options: .storageModeShared)! }
    }
    func floats(_ n: Int) -> MTLBuffer { device.makeBuffer(length: n * 4, options: .storageModeShared)! }
    var rng = SystemRandomNumberGenerator()
    let x = shared((0..<(T * D)).map { _ in Float.random(in: -1...1, using: &rng) })
    let residual = shared((0..<(T * D)).map { _ in Float.random(in: -0.1...0.1, using: &rng) })
    let wBuf = shared(weights)
    let pairSlotBuf = shared(pairSlot)
    let orderBuf = shared(order)
    let atBuf = shared(at)
    let yA = floats(T * D), yB = floats(T * D)

    // A: the runner's argument buffer of per-slot views (here all into the three tensor views).
    let p1fn = lib.makeFunction(name: "moe_iq2xxs_phase1_gate_up_act")!
    let argEnc = p1fn.makeArgumentEncoder(bufferIndex: 0)
    let arg = device.makeBuffer(length: argEnc.encodedLength, options: .storageModeShared)!
    argEnc.setArgumentBuffer(arg, offset: 0)
    var partOff = [UInt32](repeating: 0, count: 3 * nExperts)
    for (s, ex) in slots.enumerated() {
        argEnc.setBuffer(gBuf, offset: 0, index: s)
        argEnc.setBuffer(uBuf, offset: 0, index: nExperts + s)
        argEnc.setBuffer(dBuf, offset: 0, index: 2 * nExperts + s)
        partOff[3 * s] = UInt32(gOff + ex * gateRow * F)
        partOff[3 * s + 1] = UInt32(uOff + ex * gateRow * F)
        partOff[3 * s + 2] = UInt32(dOff + ex * downRow * D)
    }
    let partOffBuf = shared(partOff)
    let acts = floats(P * actStride)   // pad zeroed by the allocation

    func encodeA(_ cb: MTLCommandBuffer) {
        var offsets = (UInt32(gateRow), UInt32(downRow))
        var dV = UInt32(D), fV = UInt32(F), kV = UInt32(topK), sV = UInt32(actStride), tV = UInt32(T)
        var enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psoPhase1)
        enc.setBuffer(arg, offset: 0, index: 0)
        enc.useResources([gBuf, uBuf], usage: .read)
        enc.setBytes(&offsets, length: 8, index: 1)
        enc.setBuffer(x, offset: 0, index: 2)
        enc.setBuffer(acts, offset: 0, index: 3)
        enc.setBytes(&dV, length: 4, index: 4)
        enc.setBytes(&fV, length: 4, index: 5)
        enc.setBytes(&kV, length: 4, index: 6)
        enc.setBytes(&sV, length: 4, index: 7)
        enc.setBuffer(partOffBuf, offset: 0, index: 8)
        enc.setBytes(&tV, length: 4, index: 9)
        enc.setBuffer(pairSlotBuf, offset: 0, index: 10)
        enc.setThreadgroupMemoryLength(256 * 8 + 128, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: (P * F + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(psoPhase2)
        enc.setBuffer(arg, offset: 0, index: 0)
        enc.useResource(dBuf, usage: .read)
        enc.setBytes(&offsets, length: 8, index: 1)
        enc.setBuffer(acts, offset: 0, index: 2)
        enc.setBuffer(wBuf, offset: 0, index: 3)
        enc.setBuffer(residual, offset: 0, index: 4)
        enc.setBuffer(yA, offset: 0, index: 5)
        enc.setBytes(&dV, length: 4, index: 6)
        enc.setBytes(&sV, length: 4, index: 7)
        enc.setBytes(&kV, length: 4, index: 8)
        enc.setBuffer(partOffBuf, offset: 0, index: 9)
        enc.setBytes(&tV, length: 4, index: 10)
        enc.setBuffer(pairSlotBuf, offset: 0, index: 11)
        enc.dispatchThreadgroups(MTLSize(width: (T * D + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    // B scratch: gathered inputs, gate|up outputs, down outputs, and G experts of float32 weights.
    let xg = floats(P * D), gu = floats(P * 2 * F), dn = floats(P * D)
    let wGU = floats(G * 2 * F * D), wDown = floats(G * D * F)
    var muls: [Int: (MPSMatrixMultiplication, MPSMatrixMultiplication)] = [:]
    for n in Set(count) {
        muls[n] = (MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                           resultRows: n, resultColumns: 2 * F, interiorColumns: D, alpha: 1, beta: 0),
                   MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                           resultRows: n, resultColumns: D, interiorColumns: F, alpha: 1, beta: 0))
    }
    func matrix(_ b: MTLBuffer, _ offset: Int, _ rows: Int, _ cols: Int, rowBytes: Int? = nil) -> MPSMatrix {
        MPSMatrix(buffer: b, offset: offset,
                  descriptor: MPSMatrixDescriptor(rows: rows, columns: cols, rowBytes: rowBytes ?? cols * 4, dataType: .float32))
    }

    /// Encodes B; `split` commits after each stage and returns GPU ms per stage.
    func runB(split: Bool) -> (total: Double, stages: [String: Double], encode: Double) {
        var stages: [String: Double] = [:]
        var encodeSeconds = 0.0
        var cb = queue.makeCommandBuffer()!
        var cbs: [MTLCommandBuffer] = []
        func cut(_ label: String) {
            guard split else { return }
            let t0 = CFAbsoluteTimeGetCurrent()
            cb.commit(); cb.waitUntilCompleted()
            encodeSeconds -= CFAbsoluteTimeGetCurrent() - t0
            stages[label, default: 0] += (cb.gpuEndTime - cb.gpuStartTime) * 1000
            cb = queue.makeCommandBuffer()!
        }
        let tEnc = CFAbsoluteTimeGetCurrent()
        do {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoGather)
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(orderBuf, offset: 0, index: 1)
            enc.setBuffer(xg, offset: 0, index: 2)
            var wV = UInt32(D), kV = UInt32(topK)
            enc.setBytes(&wV, length: 4, index: 3)
            enc.setBytes(&kV, length: 4, index: 4)
            enc.dispatchThreads(MTLSize(width: (D + 31) / 32, height: P, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 64, depth: 1))
            enc.endEncoding()
        }
        cut("gather")
        for (gi, g0) in stride(from: 0, to: S, by: G).enumerated() {
            let n = min(G, S - g0)
            do {
                var firstSlot = UInt32(g0)
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(psoDeqGU)
                enc.setBuffer(arg, offset: 0, index: 0)
                enc.useResources([gBuf, uBuf], usage: .read)
                enc.setBuffer(partOffBuf, offset: 0, index: 1)
                enc.setBuffer(wGU, offset: 0, index: 2)
                var dV = UInt32(D), fV = UInt32(F)
                enc.setBytes(&dV, length: 4, index: 3)
                enc.setBytes(&fV, length: 4, index: 4)
                enc.setBytes(&firstSlot, length: 4, index: 5)
                enc.dispatchThreads(MTLSize(width: D / 32, height: 2 * F, depth: n),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 32, depth: 1))
                enc.endEncoding()
                let enc2 = cb.makeComputeCommandEncoder()!
                enc2.setComputePipelineState(psoDeqDown)
                enc2.setBuffer(arg, offset: 0, index: 0)
                enc2.useResource(dBuf, usage: .read)
                enc2.setBuffer(partOffBuf, offset: 0, index: 1)
                enc2.setBuffer(wDown, offset: 0, index: 2)
                var sV = UInt32(actStride), cV = UInt32(F)
                enc2.setBytes(&dV, length: 4, index: 3)
                enc2.setBytes(&sV, length: 4, index: 4)
                enc2.setBytes(&cV, length: 4, index: 5)
                enc2.setBytes(&firstSlot, length: 4, index: 6)
                enc2.dispatchThreads(MTLSize(width: F / 16, height: D, depth: n),
                                     threadsPerThreadgroup: MTLSize(width: 4, height: 64, depth: 1))
                enc2.endEncoding()
            }
            cut("dequant")
            for s in g0..<(g0 + n) {
                let (m1, _) = muls[count[s]]!
                m1.encode(commandBuffer: cb,
                          leftMatrix: matrix(xg, start[s] * D * 4, count[s], D),
                          rightMatrix: matrix(wGU, (s - g0) * 2 * F * D * 4, 2 * F, D),
                          resultMatrix: matrix(gu, start[s] * 2 * F * 4, count[s], 2 * F))
            }
            cut("gemm gate/up")
            do {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(psoSilu)
                enc.setBuffer(gu, offset: 0, index: 0)
                var fV = UInt32(F), firstV = UInt32(start[g0])
                enc.setBytes(&fV, length: 4, index: 1)
                enc.setBytes(&firstV, length: 4, index: 2)
                enc.dispatchThreads(MTLSize(width: F, height: start[g0 + n] - start[g0], depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 16, depth: 1))
                enc.endEncoding()
            }
            cut("silu")
            for s in g0..<(g0 + n) {
                let (_, m2) = muls[count[s]]!
                m2.encode(commandBuffer: cb,
                          leftMatrix: matrix(gu, start[s] * 2 * F * 4, count[s], F, rowBytes: 2 * F * 4),
                          rightMatrix: matrix(wDown, (s - g0) * D * F * 4, D, F),
                          resultMatrix: matrix(dn, start[s] * D * 4, count[s], D))
            }
            cut("gemm down")
            if !split && gi % 4 == 3 {   // keep one command buffer from holding every group
                cb.commit(); cbs.append(cb); cb = queue.makeCommandBuffer()!
            }
        }
        do {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoScatter)
            enc.setBuffer(dn, offset: 0, index: 0)
            enc.setBuffer(atBuf, offset: 0, index: 1)
            enc.setBuffer(wBuf, offset: 0, index: 2)
            enc.setBuffer(residual, offset: 0, index: 3)
            enc.setBuffer(yB, offset: 0, index: 4)
            var dV = UInt32(D), kV = UInt32(topK)
            enc.setBytes(&dV, length: 4, index: 5)
            enc.setBytes(&kV, length: 4, index: 6)
            enc.dispatchThreads(MTLSize(width: (D + 31) / 32, height: T, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 64, depth: 1))
            enc.endEncoding()
        }
        cut("scatter")
        encodeSeconds += CFAbsoluteTimeGetCurrent() - tEnc
        if split { return (stages.values.reduce(0, +), stages, encodeSeconds * 1000) }
        cb.commit(); cbs.append(cb)
        cb.waitUntilCompleted()
        let total = cbs.reduce(0.0) { $0 + ($1.gpuEndTime - $1.gpuStartTime) * 1000 }
        return (total, [:], encodeSeconds * 1000)
    }

    let med = { (v: [Double]) in v.sorted()[v.count / 2] }
    var aMs: [Double] = [], bMs: [Double] = [], bEnc: [Double] = [], bWall: [Double] = []
    for it in 0..<(iterations + 1) {
        let cb = queue.makeCommandBuffer()!
        encodeA(cb)
        cb.commit(); cb.waitUntilCompleted()
        if it > 0 { aMs.append((cb.gpuEndTime - cb.gpuStartTime) * 1000) }
        let t0 = CFAbsoluteTimeGetCurrent()
        let r = runB(split: false)
        if it > 0 { bMs.append(r.total); bEnc.append(r.encode); bWall.append((CFAbsoluteTimeGetCurrent() - t0) * 1000) }
    }
    let pa = yA.contents().bindMemory(to: Float.self, capacity: T * D)
    let pb = yB.contents().bindMemory(to: Float.self, capacity: T * D)
    let pr = residual.contents().bindMemory(to: Float.self, capacity: T * D)
    var d = 0.0, m = 0.0
    for i in 0..<(T * D) {
        d = max(d, Double(abs(pa[i] - pb[i])))
        m = max(m, Double(abs(pa[i] - pr[i])))
    }
    print(String(format: "  A per-pair kernels: GPU %.1f ms", med(aMs)))
    print(String(format: "  B dequant + sgemm (group %d): GPU %.1f ms, encode %.1f ms, wall %.1f ms", G, med(bMs), med(bEnc), med(bWall)))
    print(String(format: "  max |yA - yB| / max |yA - residual| = %.2e", d / m))
    let split = runB(split: true)
    let parts = split.stages.sorted { $0.value > $1.value }.map { String(format: "%@ %.1f", $0.key, $0.value) }
    print("  B stages (separate buffers, GPU ms): " + parts.joined(separator: ", "))
}
