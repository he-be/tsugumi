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
// Arm C (docs/qfn-prefill/07 E4-(b)): B in fp16 -- the experts dequantized to half, the pair rows gathered as half,
// MPS fp16 matrix multiplication (the path that reaches 18.6 TFLOP/s on the M6's tensor units, docs/m6-prefill/03),
// silu and the weighted scatter reading half rows. Its half kernels are compiled here from moe_ggml.metal's own f32
// dequant kernels (the source is read from this checkout), so nothing enters the runtime library.

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
                var sV = UInt32(actStride), cV = UInt32(F), rbV = UInt32(downRow)
                enc2.setBytes(&dV, length: 4, index: 3)
                enc2.setBytes(&sV, length: 4, index: 4)
                enc2.setBytes(&cV, length: 4, index: 5)
                enc2.setBytes(&firstSlot, length: 4, index: 6)
                enc2.setBytes(&rbV, length: 4, index: 7)
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

    // C: half variants of the dequant kernels plus half gather / silu / scatter, compiled next to moe_ggml.metal.
    let ggmlSource = try String(contentsOfFile: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("../Tsugumi/Metal/MoE/moe_ggml.metal").standardized.path, encoding: .utf8)
    func halfVariant(_ name: String) -> String {
        let start = ggmlSource.range(of: "kernel void \(name)(")!.lowerBound
        let end = ggmlSource.range(of: "\n}\n", range: start..<ggmlSource.endIndex)!.upperBound
        return String(ggmlSource[start..<end])
            .replacingOccurrences(of: "\(name)(", with: "\(name.replacingOccurrences(of: "_f32", with: "_f16"))(")
            .replacingOccurrences(of: "device float* out", with: "device half* out")
            .replacingOccurrences(of: "device float* o =", with: "device half* o =")
            .replacingOccurrences(of: "o[8 * l + j] = db * ", with: "o[8 * l + j] = (half)(db * ")
            .replacingOccurrences(of: "? -v : v);", with: "? -v : v));")
            .replacingOccurrences(of: "o[l] = dl * float((q[l] >> shift) & 3) - ml;", with: "o[l] = half(dl * float((q[l] >> shift) & 3) - ml);")
    }
    let halfKernels = halfVariant("moe_iq2xxs_dequant_gate_up_f32") + "\n" + halfVariant("moe_q2k_dequant_down_f32") + """

    kernel void qfn_gather_pair_rows_h(device const float* x [[buffer(0)]], device const uint* src [[buffer(1)]],
                                       device half* out [[buffer(2)]], constant uint& W [[buffer(3)]],
                                       constant uint& top_k [[buffer(4)]], uint2 pos [[thread_position_in_grid]]) {
        device const float* xi = x + (src[pos.y] / top_k) * W + pos.x * 32;
        device half* o = out + ulong(pos.y) * W + pos.x * 32;
        const uint n = min(32u, W - pos.x * 32);
        for (uint c = 0; c < n; ++c) o[c] = half(xi[c]);
    }
    kernel void qfn_silu_mul_halves_h(device half* gu [[buffer(0)]], constant uint& F [[buffer(1)]],
                                      constant uint& first [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
        const ulong base = ulong(first + pos.y) * 2 * F;
        const float g = float(gu[base + pos.x]);
        gu[base + pos.x] = half(g / (1.0f + exp(-g)) * float(gu[base + F + pos.x]));
    }
    kernel void qfn_scatter_weighted_h(device const half* rows [[buffer(0)]], device const uint* at [[buffer(1)]],
                                       device const float* routing_w [[buffer(2)]], device const float* residual [[buffer(3)]],
                                       device float* y [[buffer(4)]], constant uint& D [[buffer(5)]],
                                       constant uint& top_k [[buffer(6)]], uint2 pos [[thread_position_in_grid]]) {
        const uint c0 = pos.x * 32;
        const uint n = min(32u, D - c0);
        float acc[32] = {0.f};
        for (uint k = 0; k < top_k; ++k) {
            const uint pair = pos.y * top_k + k;
            const float w = routing_w[pair];
            device const half* r = rows + ulong(at[pair]) * D + c0;
            for (uint c = 0; c < n; ++c) acc[c] += w * float(r[c]);
        }
        device const float* res = residual + ulong(pos.y) * D + c0;
        device float* o = y + ulong(pos.y) * D + c0;
        for (uint c = 0; c < n; ++c) o[c] = res[c] + acc[c];
    }
    """
    let hlib = try device.makeLibrary(source: ggmlSource + "\n" + halfKernels, options: nil)
    func hpso(_ name: String) throws -> MTLComputePipelineState {
        try device.makeComputePipelineState(function: hlib.makeFunction(name: name)!)
    }
    let psoDeqGUh = try hpso("moe_iq2xxs_dequant_gate_up_f16"), psoDeqDownh = try hpso("moe_q2k_dequant_down_f16")
    let psoGatherh = try hpso("qfn_gather_pair_rows_h"), psoSiluh = try hpso("qfn_silu_mul_halves_h")
    let psoScatterh = try hpso("qfn_scatter_weighted_h")
    func halves(_ n: Int) -> MTLBuffer { device.makeBuffer(length: n * 2, options: .storageModeShared)! }
    let xgh = halves(P * D), guh = halves(P * 2 * F), dnh = halves(P * D)
    let wGUh = halves(G * 2 * F * D), wDownh = halves(G * D * F)
    let yC = floats(T * D)
    // The argument buffer of arm A holds the same per-slot views; the half dequant kernels take it as is.
    let argH = device.makeBuffer(length: argEnc.encodedLength, options: .storageModeShared)!
    let argEncH = hlib.makeFunction(name: "moe_iq2xxs_dequant_gate_up_f16")!.makeArgumentEncoder(bufferIndex: 0)
    argEncH.setArgumentBuffer(argH, offset: 0)
    for s in 0..<slots.count {
        argEncH.setBuffer(gBuf, offset: 0, index: s)
        argEncH.setBuffer(uBuf, offset: 0, index: nExperts + s)
        argEncH.setBuffer(dBuf, offset: 0, index: 2 * nExperts + s)
    }
    var mulsH: [Int: (MPSMatrixMultiplication, MPSMatrixMultiplication)] = [:]
    for n in Set(count) {
        mulsH[n] = (MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                            resultRows: n, resultColumns: 2 * F, interiorColumns: D, alpha: 1, beta: 0),
                    MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                            resultRows: n, resultColumns: D, interiorColumns: F, alpha: 1, beta: 0))
    }
    func hmatrix(_ b: MTLBuffer, _ offset: Int, _ rows: Int, _ cols: Int, rowBytes: Int? = nil) -> MPSMatrix {
        MPSMatrix(buffer: b, offset: offset,
                  descriptor: MPSMatrixDescriptor(rows: rows, columns: cols, rowBytes: rowBytes ?? cols * 2, dataType: .float16))
    }
    /// Arm C, the same shape as runB.
    func runC(split: Bool) -> (total: Double, stages: [String: Double], encode: Double) {
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
            enc.setComputePipelineState(psoGatherh)
            enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(orderBuf, offset: 0, index: 1); enc.setBuffer(xgh, offset: 0, index: 2)
            var wV = UInt32(D), kV = UInt32(topK)
            enc.setBytes(&wV, length: 4, index: 3); enc.setBytes(&kV, length: 4, index: 4)
            enc.dispatchThreads(MTLSize(width: (D + 31) / 32, height: P, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 64, depth: 1))
            enc.endEncoding()
        }
        cut("gather")
        for (gi, g0) in stride(from: 0, to: S, by: G).enumerated() {
            let n = min(G, S - g0)
            do {
                var firstSlot = UInt32(g0), dV = UInt32(D), fV = UInt32(F)
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(psoDeqGUh)
                enc.setBuffer(argH, offset: 0, index: 0); enc.useResources([gBuf, uBuf], usage: .read)
                enc.setBuffer(partOffBuf, offset: 0, index: 1); enc.setBuffer(wGUh, offset: 0, index: 2)
                enc.setBytes(&dV, length: 4, index: 3); enc.setBytes(&fV, length: 4, index: 4); enc.setBytes(&firstSlot, length: 4, index: 5)
                enc.dispatchThreads(MTLSize(width: D / 32, height: 2 * F, depth: n), threadsPerThreadgroup: MTLSize(width: 8, height: 32, depth: 1))
                enc.endEncoding()
                let enc2 = cb.makeComputeCommandEncoder()!
                enc2.setComputePipelineState(psoDeqDownh)
                enc2.setBuffer(argH, offset: 0, index: 0); enc2.useResource(dBuf, usage: .read)
                enc2.setBuffer(partOffBuf, offset: 0, index: 1); enc2.setBuffer(wDownh, offset: 0, index: 2)
                var sV = UInt32(actStride), cV = UInt32(F), rbV = UInt32(downRow)
                enc2.setBytes(&dV, length: 4, index: 3); enc2.setBytes(&sV, length: 4, index: 4); enc2.setBytes(&cV, length: 4, index: 5)
                enc2.setBytes(&firstSlot, length: 4, index: 6); enc2.setBytes(&rbV, length: 4, index: 7)
                enc2.dispatchThreads(MTLSize(width: F / 16, height: D, depth: n), threadsPerThreadgroup: MTLSize(width: 4, height: 64, depth: 1))
                enc2.endEncoding()
            }
            cut("dequant")
            for s in g0..<(g0 + n) {
                mulsH[count[s]]!.0.encode(commandBuffer: cb, leftMatrix: hmatrix(xgh, start[s] * D * 2, count[s], D),
                                          rightMatrix: hmatrix(wGUh, (s - g0) * 2 * F * D * 2, 2 * F, D),
                                          resultMatrix: hmatrix(guh, start[s] * 2 * F * 2, count[s], 2 * F))
            }
            cut("gemm gate/up")
            do {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(psoSiluh)
                enc.setBuffer(guh, offset: 0, index: 0)
                var fV = UInt32(F), firstV = UInt32(start[g0])
                enc.setBytes(&fV, length: 4, index: 1); enc.setBytes(&firstV, length: 4, index: 2)
                enc.dispatchThreads(MTLSize(width: F, height: start[g0 + n] - start[g0], depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 16, depth: 1))
                enc.endEncoding()
            }
            cut("silu")
            for s in g0..<(g0 + n) {
                mulsH[count[s]]!.1.encode(commandBuffer: cb, leftMatrix: hmatrix(guh, start[s] * 2 * F * 2, count[s], F, rowBytes: 2 * F * 2),
                                          rightMatrix: hmatrix(wDownh, (s - g0) * D * F * 2, D, F),
                                          resultMatrix: hmatrix(dnh, start[s] * D * 2, count[s], D))
            }
            cut("gemm down")
            if !split && gi % 4 == 3 { cb.commit(); cbs.append(cb); cb = queue.makeCommandBuffer()! }
        }
        do {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoScatterh)
            enc.setBuffer(dnh, offset: 0, index: 0); enc.setBuffer(atBuf, offset: 0, index: 1); enc.setBuffer(wBuf, offset: 0, index: 2)
            enc.setBuffer(residual, offset: 0, index: 3); enc.setBuffer(yC, offset: 0, index: 4)
            var dV = UInt32(D), kV = UInt32(topK)
            enc.setBytes(&dV, length: 4, index: 5); enc.setBytes(&kV, length: 4, index: 6)
            enc.dispatchThreads(MTLSize(width: (D + 31) / 32, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 64, depth: 1))
            enc.endEncoding()
        }
        cut("scatter")
        encodeSeconds += CFAbsoluteTimeGetCurrent() - tEnc
        if split { return (stages.values.reduce(0, +), stages, encodeSeconds * 1000) }
        cb.commit(); cbs.append(cb)
        cb.waitUntilCompleted()
        return (cbs.reduce(0.0) { $0 + ($1.gpuEndTime - $1.gpuStartTime) * 1000 }, [:], encodeSeconds * 1000)
    }

    let med = { (v: [Double]) in v.sorted()[v.count / 2] }
    var aMs: [Double] = [], bMs: [Double] = [], bEnc: [Double] = [], bWall: [Double] = []
    var cMs: [Double] = [], cWall: [Double] = []
    for it in 0..<(iterations + 1) {
        let cb = queue.makeCommandBuffer()!
        encodeA(cb)
        cb.commit(); cb.waitUntilCompleted()
        if it > 0 { aMs.append((cb.gpuEndTime - cb.gpuStartTime) * 1000) }
        let t0 = CFAbsoluteTimeGetCurrent()
        let r = runB(split: false)
        if it > 0 { bMs.append(r.total); bEnc.append(r.encode); bWall.append((CFAbsoluteTimeGetCurrent() - t0) * 1000) }
        let t1 = CFAbsoluteTimeGetCurrent()
        let rc = runC(split: false)
        if it > 0 { cMs.append(rc.total); cWall.append((CFAbsoluteTimeGetCurrent() - t1) * 1000) }
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
    print(String(format: "  B dequant + sgemm (group %d): GPU %.1f ms, encode %.1f ms, wall %.1f ms (GPU min %.1f max %.1f, n %d)",
                 G, med(bMs), med(bEnc), med(bWall), bMs.min()!, bMs.max()!, bMs.count))
    print(String(format: "  max |yA - yB| / max |yA - residual| = %.2e", d / m))
    let split = runB(split: true)
    let parts = split.stages.sorted { $0.value > $1.value }.map { String(format: "%@ %.1f", $0.key, $0.value) }
    print("  B stages (separate buffers, GPU ms): " + parts.joined(separator: ", "))
    let pc = yC.contents().bindMemory(to: Float.self, capacity: T * D)
    var dc = 0.0
    for i in 0..<(T * D) { dc = max(dc, Double(abs(pa[i] - pc[i]))) }
    print(String(format: "  C fp16 dequant + MPS fp16 GEMM (group %d): GPU %.1f ms (min %.1f max %.1f, n %d), wall %.1f ms",
                 G, med(cMs), cMs.min()!, cMs.max()!, cMs.count, med(cWall)))
    print(String(format: "  max |yA - yC| / max |yA - residual| = %.2e", dc / m))
    let splitC = runC(split: true)
    print("  C stages (separate buffers, GPU ms): " + splitC.stages.sorted { $0.value > $1.value }
        .map { String(format: "%@ %.1f", $0.key, $0.value) }.joined(separator: ", "))
}
