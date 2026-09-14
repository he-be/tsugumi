import Foundation
import Metal
import MetalPerformanceShaders
import Tsugumi

// MARK: - `--q2-down-sidecar <gguf> <down sidecar> <route .bin> [tokens,...] [--q2-down-save dir | --q2-down-against dir]`
//
// docs/qwen38/15 §2 W-2: the Q2_K down rows with the pad of their last block dropped (252 -> 212 B,
// `Scripts/qwen38/down_sidecar.py`) must give the same bits as the original rows. One layer of a real route
// dump (`Q38_DUMP_ROUTE`), its first T tokens, the same seeded x / residual / pairs through both kernel forms
// the runner uses — per-pair (`moe_q2k_phase2_down_reduce`) and dequantize + sgemm (`moe_q2k_dequant_down_f32`) —
// once with the down views into the GGUF and once into the sidecar. y is compared byte for byte.
// `--q2-down-save` writes the GGUF-layout y of each form, `--q2-down-against` compares both layouts with a
// saved run (the kernels before a change against after it).

func runQ2DownSidecarCheck(ggufPath: String, sidecarPath: String, routePath: String, tokens: [Int],
                           save: String?, against: String?) throws -> Bool {
    let file = try GGUFFile(url: URL(fileURLWithPath: (ggufPath as NSString).expandingTildeInPath))
    let side = try GGUFFile(url: URL(fileURLWithPath: (sidecarPath as NSString).expandingTildeInPath))
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

    let D = 2560, F = 640, actStride = 768, nExperts = 512, topK = 10, G = 16
    let il = Int(routePath.split(separator: "-").last!.dropFirst().split(separator: ".").first!)!
    let route = try Data(contentsOf: URL(fileURLWithPath: routePath))
    let T0 = Int(route.withUnsafeBytes { $0.load(as: Int32.self) })

    let gateT = try file.tensor("blk.\(il).ffn_gate_exps.weight")
    let upT = try file.tensor("blk.\(il).ffn_up_exps.weight")
    let layouts = [("gguf", file, try file.tensor("blk.\(il).ffn_down_exps.weight")),
                   ("sidecar", side, try side.tensor("blk.\(il).ffn_down_exps.weight"))]
    let gateRow = gateT.bytesPerRow
    file.preadRanges([(gateT.offset, gateT.byteCount), (upT.offset, upT.byteCount), (layouts[0].2.offset, layouts[0].2.byteCount)], threads: 4)
    side.preadRanges([(layouts[1].2.offset, layouts[1].2.byteCount)], threads: 4)
    let (gBuf, gOff) = file.noCopyBuffer(device: device, tensor: gateT)!
    let (uBuf, uOff) = file.noCopyBuffer(device: device, tensor: upT)!
    print("layer \(il): down row \(layouts[0].2.bytesPerRow) B (\(layouts[0].2.type)) / \(layouts[1].2.bytesPerRow) B (\(layouts[1].2.type))")

    func shared<T>(_ a: [T]) -> MTLBuffer {
        a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 1), options: .storageModeShared)! }
    }
    func floats(_ n: Int) -> MTLBuffer { device.makeBuffer(length: n * 4, options: .storageModeShared)! }
    func matrix(_ b: MTLBuffer, _ offset: Int, _ rows: Int, _ cols: Int, rowBytes: Int? = nil) -> MPSMatrix {
        MPSMatrix(buffer: b, offset: offset,
                  descriptor: MPSMatrixDescriptor(rows: rows, columns: cols, rowBytes: rowBytes ?? cols * 4, dataType: .float32))
    }

    var pass = true
    for T in tokens.map({ min($0, T0) }) {
        let P = T * topK
        var experts = [Int32](repeating: 0, count: P)
        var weights = [Float](repeating: 0, count: P)
        route.withUnsafeBytes { raw in
            experts.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[4..<(4 + 4 * P)])) }
            weights.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[(4 + 4 * T0 * topK)..<(4 + 4 * T0 * topK + 4 * P)])) }
        }
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
        var order = [UInt32](repeating: 0, count: P)
        var at = [UInt32](repeating: 0, count: P)
        for p in 0..<P {
            let s = Int(pairSlot[p])
            order[fill[s]] = UInt32(p)
            at[p] = UInt32(fill[s])
            fill[s] += 1
        }
        // Seeded, so a saved run and a later one see the same inputs.
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15 &+ UInt64(T)
        func uniform() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / Float(1 << 24) * 2 - 1
        }
        let x = shared((0..<(T * D)).map { _ in uniform() })
        let residual = shared((0..<(T * D)).map { _ in 0.1 * uniform() })
        let wBuf = shared(weights), pairSlotBuf = shared(pairSlot), orderBuf = shared(order), atBuf = shared(at)
        let p1fn = lib.makeFunction(name: "moe_iq2xxs_phase1_gate_up_act")!
        let argEnc = p1fn.makeArgumentEncoder(bufferIndex: 0)

        var results: [(a: Data, b: Data)] = []
        for (_, lf, downT) in layouts {
            let (dBuf, dOff) = lf.noCopyBuffer(device: device, tensor: downT)!
            let downRow = downT.bytesPerRow
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
            var offsets = (UInt32(gateRow), UInt32(downRow))
            var dV = UInt32(D), fV = UInt32(F), kV = UInt32(topK), sV = UInt32(actStride), tV = UInt32(T)
            var rbV = UInt32(downRow)

            // A: per-pair kernels.
            let acts = floats(P * actStride)
            let yA = floats(T * D)
            var cb = queue.makeCommandBuffer()!
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
            cb.commit(); cb.waitUntilCompleted()
            if let error = cb.error { throw error }

            // B: gather, dequantize G experts at a time, sgemm, SiLU, sgemm, scatter (as `routedGemm`).
            let xg = floats(P * D), gu = floats(P * 2 * F), dn = floats(P * D), yB = floats(T * D)
            let wGU = floats(G * 2 * F * D), wDown = floats(G * D * F)
            cb = queue.makeCommandBuffer()!
            enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoGather)
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(orderBuf, offset: 0, index: 1)
            enc.setBuffer(xg, offset: 0, index: 2)
            enc.setBytes(&dV, length: 4, index: 3)
            enc.setBytes(&kV, length: 4, index: 4)
            enc.dispatchThreads(MTLSize(width: (D + 31) / 32, height: P, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 64, depth: 1))
            enc.endEncoding()
            for g0 in stride(from: 0, to: S, by: G) {
                autoreleasepool {
                    let n = min(G, S - g0)
                    var firstSlot = UInt32(g0)
                    var e = cb.makeComputeCommandEncoder()!
                    e.setComputePipelineState(psoDeqGU)
                    e.setBuffer(arg, offset: 0, index: 0)
                    e.useResources([gBuf, uBuf], usage: .read)
                    e.setBuffer(partOffBuf, offset: 0, index: 1)
                    e.setBuffer(wGU, offset: 0, index: 2)
                    e.setBytes(&dV, length: 4, index: 3)
                    e.setBytes(&fV, length: 4, index: 4)
                    e.setBytes(&firstSlot, length: 4, index: 5)
                    e.dispatchThreads(MTLSize(width: D / 32, height: 2 * F, depth: n),
                                      threadsPerThreadgroup: MTLSize(width: 8, height: 32, depth: 1))
                    e.endEncoding()
                    e = cb.makeComputeCommandEncoder()!
                    e.setComputePipelineState(psoDeqDown)
                    e.setBuffer(arg, offset: 0, index: 0)
                    e.useResource(dBuf, usage: .read)
                    e.setBuffer(partOffBuf, offset: 0, index: 1)
                    e.setBuffer(wDown, offset: 0, index: 2)
                    e.setBytes(&dV, length: 4, index: 3)
                    e.setBytes(&sV, length: 4, index: 4)
                    e.setBytes(&fV, length: 4, index: 5)
                    e.setBytes(&firstSlot, length: 4, index: 6)
                    e.setBytes(&rbV, length: 4, index: 7)
                    e.dispatchThreads(MTLSize(width: F / 16, height: D, depth: n),
                                      threadsPerThreadgroup: MTLSize(width: 4, height: 64, depth: 1))
                    e.endEncoding()
                    for s in g0..<(g0 + n) {
                        MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                                resultRows: count[s], resultColumns: 2 * F, interiorColumns: D, alpha: 1, beta: 0)
                            .encode(commandBuffer: cb,
                                    leftMatrix: matrix(xg, start[s] * D * 4, count[s], D),
                                    rightMatrix: matrix(wGU, (s - g0) * 2 * F * D * 4, 2 * F, D),
                                    resultMatrix: matrix(gu, start[s] * 2 * F * 4, count[s], 2 * F))
                    }
                    e = cb.makeComputeCommandEncoder()!
                    e.setComputePipelineState(psoSilu)
                    e.setBuffer(gu, offset: 0, index: 0)
                    var firstV = UInt32(start[g0])
                    e.setBytes(&fV, length: 4, index: 1)
                    e.setBytes(&firstV, length: 4, index: 2)
                    e.dispatchThreads(MTLSize(width: F, height: start[g0 + n] - start[g0], depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 64, height: 16, depth: 1))
                    e.endEncoding()
                    for s in g0..<(g0 + n) {
                        MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                                resultRows: count[s], resultColumns: D, interiorColumns: F, alpha: 1, beta: 0)
                            .encode(commandBuffer: cb,
                                    leftMatrix: matrix(gu, start[s] * 2 * F * 4, count[s], F, rowBytes: 2 * F * 4),
                                    rightMatrix: matrix(wDown, (s - g0) * D * F * 4, D, F),
                                    resultMatrix: matrix(dn, start[s] * D * 4, count[s], D))
                    }
                    if (g0 / G) % 4 == 3 { cb.commit(); cb.waitUntilCompleted(); cb = queue.makeCommandBuffer()! }
                }
            }
            enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoScatter)
            enc.setBuffer(dn, offset: 0, index: 0)
            enc.setBuffer(atBuf, offset: 0, index: 1)
            enc.setBuffer(wBuf, offset: 0, index: 2)
            enc.setBuffer(residual, offset: 0, index: 3)
            enc.setBuffer(yB, offset: 0, index: 4)
            enc.setBytes(&dV, length: 4, index: 5)
            enc.setBytes(&kV, length: 4, index: 6)
            enc.dispatchThreads(MTLSize(width: (D + 31) / 32, height: T, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 64, depth: 1))
            enc.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            results.append((Data(bytes: yA.contents(), count: T * D * 4), Data(bytes: yB.contents(), count: T * D * 4)))
        }

        func differing(_ u: Data, _ v: Data) -> Int {
            u.withUnsafeBytes { a in v.withUnsafeBytes { b in
                let pa = a.bindMemory(to: UInt32.self), pb = b.bindMemory(to: UInt32.self)
                return (0..<pa.count).reduce(0) { $0 + (pa[$1] != pb[$1] ? 1 : 0) }
            } }
        }
        func rel(_ u: Data, _ v: Data) -> Double {
            u.withUnsafeBytes { a in v.withUnsafeBytes { b in
                let pa = a.bindMemory(to: Float.self), pb = b.bindMemory(to: Float.self)
                var d = 0.0, m = 0.0
                for i in 0..<pa.count { d = max(d, Double(abs(pa[i] - pb[i]))); m = max(m, Double(abs(pa[i]))) }
                return d / m
            } }
        }
        let (full, half) = (results[0], results[1])
        let dA = differing(full.a, half.a), dB = differing(full.b, half.b)
        var line = String(format: "  T=%-5d experts %3d  per-pair gguf vs sidecar: %d / %d floats differ   dequant+sgemm: %d / %d   (per-pair vs sgemm max rel %.2e)",
                          T, S, dA, T * D, dB, T * D, rel(full.a, full.b))
        if dA != 0 || dB != 0 { pass = false }
        if let dir = save {
            try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try full.a.write(to: URL(fileURLWithPath: "\(dir)/l\(il)-t\(T)-A.f32"))
            try full.b.write(to: URL(fileURLWithPath: "\(dir)/l\(il)-t\(T)-B.f32"))
        }
        if let dir = against {
            let oldA = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/l\(il)-t\(T)-A.f32"))
            let oldB = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/l\(il)-t\(T)-B.f32"))
            let vs = [differing(oldA, full.a), differing(oldA, half.a), differing(oldB, full.b), differing(oldB, half.b)]
            line += "   vs saved: A gguf \(vs[0]) sidecar \(vs[1]), B gguf \(vs[2]) sidecar \(vs[3])"
            if vs.contains(where: { $0 != 0 }) { pass = false }
        }
        print(line)
    }
    print(pass ? "PASS" : "FAIL")
    return pass
}
