import Foundation
import Metal

/// Qwen3.8-Flash-Next (qwen4exp) decode off the DS4-IQ2 GGUF: the Q2
/// verification runner (`docs/investigations/QWEN38_FLASH_NEXT_VERIFY_PLAN.md`).
///
/// Correctness first. One token at a time, float32 activations, every weight
/// read in its GGUF bytes (Q8_0 / F16 / F32 dense through no-copy views of the
/// mapped file, IQ2_XXS / Q2_K routed experts through per-expert no-copy views of
/// the same mapping, so the page cache is the expert cache). The layer is GPU
/// work except for the host steps the reference makes easy to check: the
/// router's top-10 (logits read back), QSA's block choice above its budget, and
/// PLE (layer 1, 16 hashed rows from the Q4_1 sidecar).
/// It is compared token for token against `Scripts/qwen38/reference_forward.py`.
package final class Qwen38Runner {
    // Shape, from the GGUF (checked in init).
    package let e = 2560
    package let hc = 4
    package let nTrunk: Int
    package let vocab = 248_320
    private let hcRank = 320
    private let H = 24, Hkv = 2, D = 256, nRot = 64
    private let Hk = 16, Hv = 48, Dl = 128, convK = 4
    private let F = 640, nExperts = 512, topK = 10, downIn = 768
    private let fullInterval: Int
    private let pleLayer: Int
    private let eos: Int
    private let pleMult: [UInt64]
    private let pleOffsets: [UInt64]
    private let pleVocab: [UInt64]
    private let pleNgram: Int
    private let plePer: Int
    private let eps: Float

    package let capacity: Int
    private let file: GGUFFile
    private let pleFile: GGUFFile
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let dense: GGMLDenseGEMV

    private let psoRms, psoUnary, psoMix, psoCombine, psoAddScaled, psoSiluMul: MTLComputePipelineState
    private let psoConv, psoQKNorm, psoStep, psoNormGate, psoAttnPrep, psoAttnDecode: MTLComputePipelineState
    private let psoPhase1, psoPhase2: MTLComputePipelineState
    private let psoIdxBlockKey, psoIdxQPrep, psoIdxScore: MTLComputePipelineState

    private var views: [String: (tensor: GGUFFile.Tensor, buffer: MTLBuffer, offset: Int)] = [:]

    // Scratch (float32).
    private let R, xn, lo, gate, mixed, inj, blk, blkShared: MTLBuffer
    private let qkv, conv, z, ga, gb, lo6144: MTLBuffer
    private let qg, q, qgate, ao: MTLBuffer
    private let routerLogits, shG, shU, shY, shGate: MTLBuffer
    private let acts, routeW: MTLBuffer
    private let routedArg: MTLBuffer
    private let routedArgEncoder: MTLArgumentEncoder
    private let partOffsets: MTLBuffer
    /// No-copy views of one expert's gate / up / down rows in the mapped GGUF,
    /// keyed (layer * nExperts + expert) * 3 + part. Created on first use.
    private var expertParts: [Int: (buffer: MTLBuffer, offset: Int)] = [:]
    private let ropeFreq: MTLBuffer
    private let logits: MTLBuffer
    private let zeroOne: MTLBuffer

    // Per-layer state.
    private var linHist: [Int: MTLBuffer] = [:]
    private var linState: [Int: MTLBuffer] = [:]
    private var kCache: [Int: MTLBuffer] = [:]
    private var vCache: [Int: MTLBuffer] = [:]
    private var idxRawKeys: [Int: MTLBuffer] = [:]    // [cap][128]
    private var idxBlockKeys: [Int: MTLBuffer] = [:]  // [cap/4][128], filled as blocks complete
    private let iq: MTLBuffer
    private let idxScores: MTLBuffer
    private let selection: MTLBuffer
    /// QSA budget in tokens (GGUF: 2048). Lowered only by checks, to make the
    /// selection fire on short prompts the CPU reference can afford.
    package var indexerTopK: Int
    private var pleHist: [Float]
    private var plePrev: [Int]

    package private(set) var lastRoutes: [[Int]] = []
    /// Tokens QSA selected per attention layer in the last step (nil = all visible).
    package private(set) var lastSelection: [Int: [Int]] = [:]

    /// Host wall time of the last `step`, by stage (seconds).
    package struct StepProfile {
        package var ple = 0.0
        package var preRouter = 0.0   // encode + GPU wait of the pre-router buffer, all layers
        package var route = 0.0       // host top-10 + expert copy, all layers
        package var routed = 0.0      // encode + GPU wait of the routed buffer, all layers
        package var head = 0.0
        package var total = 0.0
    }
    package private(set) var lastProfile = StepProfile()

    package init(gguf: URL, ple: URL, capacity: Int) throws {
        let file = try GGUFFile(url: gguf)
        let pleFile = try GGUFFile(url: ple)
        self.file = file
        self.pleFile = pleFile
        self.capacity = capacity
        guard try file.value("general.architecture").string == "qwen4exp" else {
            throw GGUFFile.Error.format("not a qwen4exp GGUF")
        }
        let nLayer = try file.value("qwen4exp.block_count").int!
        nTrunk = nLayer - (try file.value("qwen4exp.nextn_predict_layers").int!)
        fullInterval = try file.value("qwen4exp.full_attention_interval").int!
        pleLayer = (try file.value("qwen4exp.ple.layers").ints!)[0]
        eos = try file.value("qwen4exp.ple.eos_token_id").int!
        pleMult = (try file.value("qwen4exp.ple.layer_multipliers").ints!).map { UInt64(bitPattern: Int64($0)) }
        pleOffsets = (try file.value("qwen4exp.ple.head_offsets").ints!).map { UInt64($0) }
        pleVocab = (try file.value("qwen4exp.ple.head_vocab_sizes").ints!).map { UInt64($0) }
        pleNgram = try file.value("qwen4exp.ple.ngram_size").int!
        plePer = try file.value("qwen4exp.ple.heads_per_ngram").int!
        eps = Float(try file.value("qwen4exp.attention.layer_norm_rms_epsilon").double!)
        let shapeOK = try file.value("qwen4exp.embedding_length").int == e
            && file.value("qwen4exp.expert_used_count").int == topK
            && file.value("qwen4exp.expert_count").int == nExperts
            && file.value("qwen4exp.hyper_connection.count").int == hc
            && file.value("qwen4exp.rope.freq_base").double == 10_000_000
        guard shapeOK else { throw GGUFFile.Error.format("unexpected qwen4exp shape") }

        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw GGUFFile.Error.open("no Metal device")
        }
        self.device = device
        self.queue = queue
        dense = try GGMLDenseGEMV(device: device)

        let lib = try MetalContext.moduleLibrary(device: device, module: "qwen38")
        func pso(_ library: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw GGUFFile.Error.format("kernel \(name) missing")
            }
            return try device.makeComputePipelineState(function: fn)
        }
        psoRms = try pso(lib, "q38_grouped_rms")
        psoUnary = try pso(lib, "q38_unary")
        psoMix = try pso(lib, "q38_hc_mix")
        psoCombine = try pso(lib, "q38_hc_combine")
        psoAddScaled = try pso(lib, "q38_add_scaled")
        psoSiluMul = try pso(lib, "q38_silu_mul")
        psoConv = try pso(lib, "q38_gdn_conv")
        psoQKNorm = try pso(lib, "q38_gdn_qk_norm")
        psoStep = try pso(lib, "q38_gdn_step")
        psoNormGate = try pso(lib, "q38_gdn_norm_gate")
        psoAttnPrep = try pso(lib, "q38_attn_prep")
        psoAttnDecode = try pso(lib, "q38_attn_decode")
        psoIdxBlockKey = try pso(lib, "q38_idx_block_key")
        psoIdxQPrep = try pso(lib, "q38_idx_q_prep")
        psoIdxScore = try pso(lib, "q38_idx_score")
        indexerTopK = try file.value("qwen4exp.attention.indexer.top_k").int ?? 2048
        let moeLib = try MetalContext.moduleLibrary(device: device, module: "moe_ggml")
        psoPhase1 = try pso(moeLib, "moe_iq2xxs_phase1_gate_up_act")
        psoPhase2 = try pso(moeLib, "moe_q2k_phase2_down_reduce")

        func buf(_ count: Int) -> MTLBuffer {
            device.makeBuffer(length: max(count, 1) * 4, options: .storageModeShared)!
        }
        R = buf(hc * e); xn = buf(hc * e); lo = buf(hcRank); gate = buf(hc * e)
        mixed = buf(e); inj = buf(hc); blk = buf(e); blkShared = buf(e)
        qkv = buf(2 * Hk * Dl + Hv * Dl); conv = buf(2 * Hk * Dl + Hv * Dl)
        z = buf(Hv * Dl); ga = buf(Hv); gb = buf(Hv); lo6144 = buf(Hv * Dl)
        qg = buf(2 * H * D); q = buf(H * D); qgate = buf(H * D); ao = buf(H * D)
        routerLogits = buf(nExperts); shG = buf(F); shU = buf(F); shY = buf(e); shGate = buf(1)
        acts = buf(topK * downIn); routeW = buf(topK)
        logits = buf(vocab)
        iq = buf(4 * 128)
        idxScores = buf(capacity / 4 + 1)
        selection = device.makeBuffer(length: max(capacity, 1) * 4, options: .storageModeShared)!
        zeroOne = buf(1)
        zeroOne.contents().bindMemory(to: Float.self, capacity: 1)[0] = 1
        let rotDims = 64
        let freq = (0..<(rotDims / 2)).map { i in
            Float(pow(10_000_000.0, -2.0 * Double(i) / Double(rotDims)))
        }
        ropeFreq = device.makeBuffer(bytes: freq, length: freq.count * 4, options: .storageModeShared)!

        guard let p1 = moeLib.makeFunction(name: "moe_iq2xxs_phase1_gate_up_act") else {
            throw GGUFFile.Error.format("phase1 missing")
        }
        routedArgEncoder = p1.makeArgumentEncoder(bufferIndex: 0)
        routedArg = device.makeBuffer(length: routedArgEncoder.encodedLength, options: .storageModeShared)!
        routedArgEncoder.setArgumentBuffer(routedArg, offset: 0)
        partOffsets = device.makeBuffer(length: 3 * 10 * 4, options: .storageModeShared)!

        pleHist = [Float](repeating: 0, count: (4 - 1) * 3 * 4 * 2560)
        plePrev = [Int](repeating: 248_044, count: 2)
    }

    // MARK: - Weights

    private func view(_ name: String) throws -> (tensor: GGUFFile.Tensor, buffer: MTLBuffer, offset: Int) {
        if let v = views[name] { return v }
        let t = try file.tensor(name)
        guard let (b, off) = file.noCopyBuffer(device: device, tensor: t) else {
            throw GGUFFile.Error.format("no-copy view failed for \(name)")
        }
        let v = (t, b, off)
        views[name] = v
        return v
    }

    private func isLinear(_ il: Int) -> Bool { (il + 1) % fullInterval != 0 }

    // MARK: - Encoding helpers

    private func run(_ cb: MTLCommandBuffer, _ pso: MTLComputePipelineState, _ n: MTLSize,
                     _ setup: (MTLComputeCommandEncoder) -> Void) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        setup(enc)
        let w = min(pso.threadExecutionWidth, max(n.width, 1))
        let hgt = min(max(pso.maxTotalThreadsPerThreadgroup / w, 1), max(n.height, 1))
        enc.dispatchThreads(n, threadsPerThreadgroup: MTLSize(width: w, height: hgt, depth: 1))
        enc.endEncoding()
    }

    private func gemv(_ cb: MTLCommandBuffer, _ name: String, x: MTLBuffer, xOffset: Int = 0,
                      y: MTLBuffer, yOffset: Int = 0) throws {
        let v = try view(name)
        dense.encode(commandBuffer: cb, type: v.tensor.type, weights: v.buffer, weightsOffset: v.offset,
                     x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                     m: v.tensor.rowCount, n: v.tensor.rowWidth)
    }

    private func setF32(_ enc: MTLComputeCommandEncoder, _ name: String, index: Int) throws {
        let v = try view(name)
        precondition(v.tensor.type == .f32, "\(name) is not F32")
        enc.setBuffer(v.buffer, offset: v.offset, index: index)
    }

    private func rms(_ cb: MTLCommandBuffer, x: MTLBuffer, gamma: String, out: MTLBuffer,
                     groups: Int, n: Int, wide: Bool) throws {
        let g = try view(gamma)
        run(cb, psoRms, MTLSize(width: groups, height: 1, depth: 1)) { enc in
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(g.buffer, offset: g.offset, index: 1)
            enc.setBuffer(out, offset: 0, index: 2)
            var p = (UInt32(groups), UInt32(n), eps)
            enc.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 3)
            var wideV = UInt32(wide ? 1 : 0)
            enc.setBytes(&wideV, length: 4, index: 4)
        }
    }

    private func unary(_ cb: MTLCommandBuffer, _ x: MTLBuffer, n: Int, op: UInt32, inScale: Float, outScale: Float) {
        run(cb, psoUnary, MTLSize(width: n, height: 1, depth: 1)) { enc in
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(x, offset: 0, index: 1)
            var p = (op, inScale, outScale)
            enc.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 2)
        }
    }

    private func hcMix(_ cb: MTLCommandBuffer, prefix: String, inject: Bool) throws {
        try rms(cb, x: R, gamma: prefix + "_norm.weight", out: xn, groups: hc, n: e, wide: true)
        try gemv(cb, prefix + "_down.weight", x: xn, y: lo)
        unary(cb, lo, n: hcRank, op: 0, inScale: 1 / Float(hc), outScale: 1)
        try gemv(cb, prefix + "_up.weight", x: lo, y: gate)
        run(cb, psoMix, MTLSize(width: e, height: 1, depth: 1)) { enc in
            enc.setBuffer(xn, offset: 0, index: 0)
            enc.setBuffer(gate, offset: 0, index: 1)
            enc.setBuffer(mixed, offset: 0, index: 2)
            var p = (UInt32(hc), UInt32(e))
            enc.setBytes(&p, length: 8, index: 3)
        }
        if inject {
            try gemv(cb, prefix + "_inject.weight", x: xn, y: inj)
            unary(cb, inj, n: hc, op: 1, inScale: 1 / Float(hc), outScale: 2)
        }
    }

    private func combine(_ cb: MTLCommandBuffer, block: MTLBuffer) {
        run(cb, psoCombine, MTLSize(width: hc * e, height: 1, depth: 1)) { enc in
            enc.setBuffer(R, offset: 0, index: 0)
            enc.setBuffer(block, offset: 0, index: 1)
            enc.setBuffer(inj, offset: 0, index: 2)
            var p = (UInt32(hc), UInt32(e))
            enc.setBytes(&p, length: 8, index: 3)
        }
    }

    // MARK: - Blocks

    private func linear(_ cb: MTLCommandBuffer, il: Int) throws {
        let pre = "blk.\(il)."
        let C = 2 * Hk * Dl + Hv * Dl
        try gemv(cb, pre + "attn_qkv.weight", x: mixed, y: qkv)
        try gemv(cb, pre + "attn_gate.weight", x: mixed, y: z)
        try gemv(cb, pre + "ssm_beta.weight", x: mixed, y: gb)
        try gemv(cb, pre + "ssm_alpha.weight", x: mixed, y: ga)
        let hist = linHist[il] ?? device.makeBuffer(length: (convK - 1) * C * 4, options: .storageModeShared)!
        let state = linState[il] ?? device.makeBuffer(length: Hv * Dl * Dl * 4, options: .storageModeShared)!
        linHist[il] = hist
        linState[il] = state
        let cw = try view(pre + "ssm_conv1d.weight")
        run(cb, psoConv, MTLSize(width: C, height: 1, depth: 1)) { enc in
            enc.setBuffer(qkv, offset: 0, index: 0)
            enc.setBuffer(hist, offset: 0, index: 1)
            enc.setBuffer(cw.buffer, offset: cw.offset, index: 2)
            enc.setBuffer(conv, offset: 0, index: 3)
            var c = UInt32(C), k = UInt32(convK)
            enc.setBytes(&c, length: 4, index: 4)
            enc.setBytes(&k, length: 4, index: 5)
        }
        var gp = (UInt32(Hk), UInt32(Hv), UInt32(Dl))
        run(cb, psoQKNorm, MTLSize(width: 2 * Hk, height: 1, depth: 1)) { enc in
            enc.setBuffer(conv, offset: 0, index: 0)
            enc.setBytes(&gp, length: 12, index: 1)
        }
        try run(cb, psoStep, MTLSize(width: Dl, height: Hv, depth: 1)) { enc in
            enc.setBuffer(conv, offset: 0, index: 0)
            enc.setBuffer(ga, offset: 0, index: 1)
            enc.setBuffer(gb, offset: 0, index: 2)
            try? setF32(enc, pre + "ssm_a", index: 3)
            try? setF32(enc, pre + "ssm_dt.bias", index: 4)
            enc.setBuffer(state, offset: 0, index: 5)
            enc.setBuffer(lo6144, offset: 0, index: 6)
            enc.setBytes(&gp, length: 12, index: 7)
        }
        let nw = try view(pre + "ssm_norm.weight")
        run(cb, psoNormGate, MTLSize(width: Hv, height: 1, depth: 1)) { enc in
            enc.setBuffer(lo6144, offset: 0, index: 0)
            enc.setBuffer(z, offset: 0, index: 1)
            enc.setBuffer(nw.buffer, offset: nw.offset, index: 2)
            var d = UInt32(Dl), ep = eps
            enc.setBytes(&d, length: 4, index: 3)
            enc.setBytes(&ep, length: 4, index: 4)
        }
        try gemv(cb, pre + "ssm_out.weight", x: lo6144, y: blk)
    }

    private func attention(_ cb: inout MTLCommandBuffer, il: Int, pos: Int) throws {
        let pre = "blk.\(il)."
        let kc = kCache[il] ?? device.makeBuffer(length: capacity * Hkv * D * 4, options: .storageModeShared)!
        let vc = vCache[il] ?? device.makeBuffer(length: capacity * Hkv * D * 4, options: .storageModeShared)!
        let ik = idxRawKeys[il] ?? device.makeBuffer(length: capacity * 128 * 4, options: .storageModeShared)!
        let bk = idxBlockKeys[il] ?? device.makeBuffer(length: (capacity / 4 + 1) * 128 * 4, options: .storageModeShared)!
        kCache[il] = kc
        vCache[il] = vc
        idxRawKeys[il] = ik
        idxBlockKeys[il] = bk
        try gemv(cb, pre + "attn_q.weight", x: mixed, y: qg)
        try gemv(cb, pre + "attn_k.weight", x: mixed, y: kc, yOffset: pos * Hkv * D * 4)
        try gemv(cb, pre + "attn_v.weight", x: mixed, y: vc, yOffset: pos * Hkv * D * 4)

        // QSA indexer: raw key for this token; a block's key once its 4th token is in.
        try gemv(cb, pre + "indexer.q_proj.weight", x: mixed, y: iq)
        try gemv(cb, pre + "indexer.k_proj.weight", x: mixed, y: ik, yOffset: pos * 128 * 4)
        let gk = try view(pre + "indexer.k_norm.weight")
        let gq = try view(pre + "indexer.q_norm.weight")
        if (pos + 1) % 4 == 0 {
            var bp = (UInt32(H), UInt32(Hkv), UInt32(128), UInt32(nRot), UInt32(pos + 1 - 4), eps)
            run(cb, psoIdxBlockKey, MTLSize(width: 1, height: 1, depth: 1)) { enc in
                enc.setBuffer(ik, offset: 0, index: 0)
                enc.setBuffer(gk.buffer, offset: gk.offset, index: 1)
                enc.setBuffer(ropeFreq, offset: 0, index: 2)
                enc.setBuffer(bk, offset: 0, index: 3)
                enc.setBytes(&bp, length: MemoryLayout.size(ofValue: bp), index: 4)
            }
        }
        let nBlocks = (pos + 1) / 4
        let kBlocks = indexerTopK / 4
        var useSel: UInt32 = 0
        var nSel = UInt32(pos + 1)
        if nBlocks > kBlocks {
            var qp = (UInt32(H), UInt32(Hkv), UInt32(128), UInt32(nRot), UInt32(pos), eps)
            run(cb, psoIdxQPrep, MTLSize(width: 4, height: 1, depth: 1)) { enc in
                enc.setBuffer(iq, offset: 0, index: 0)
                enc.setBuffer(gq.buffer, offset: gq.offset, index: 1)
                enc.setBuffer(ropeFreq, offset: 0, index: 2)
                enc.setBytes(&qp, length: MemoryLayout.size(ofValue: qp), index: 3)
            }
            run(cb, psoIdxScore, MTLSize(width: nBlocks, height: 1, depth: 1)) { enc in
                enc.setBuffer(iq, offset: 0, index: 0)
                enc.setBuffer(bk, offset: 0, index: 1)
                enc.setBuffer(idxScores, offset: 0, index: 2)
                var heads = UInt32(4), d = UInt32(128)
                enc.setBytes(&heads, length: 4, index: 3)
                enc.setBytes(&d, length: 4, index: 4)
            }
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            cb = queue.makeCommandBuffer()!
            // Top kBlocks by score, lower block index first on ties; tokens in block order, then the tail.
            let sc = idxScores.contents().bindMemory(to: Float.self, capacity: nBlocks)
            let order = (0..<nBlocks).sorted { sc[$0] != sc[$1] ? sc[$0] > sc[$1] : $0 < $1 }
            let taken = order.prefix(kBlocks).sorted()
            let sp = selection.contents().bindMemory(to: UInt32.self, capacity: capacity)
            var n = 0
            for b in taken { for t in 0..<4 { sp[n] = UInt32(b * 4 + t); n += 1 } }
            for t in (nBlocks * 4)..<(pos + 1) { sp[n] = UInt32(t); n += 1 }  // the tail may be empty
            nSel = UInt32(n)
            useSel = 1
            lastSelection[il] = Array(UnsafeBufferPointer(start: sp, count: n)).map(Int.init)
        } else {
            lastSelection[il] = nil
        }

        var p = (UInt32(H), UInt32(Hkv), UInt32(D), UInt32(nRot), UInt32(pos), eps)
        let qn = try view(pre + "attn_q_norm.weight")
        let kn = try view(pre + "attn_k_norm.weight")
        run(cb, psoAttnPrep, MTLSize(width: H + Hkv, height: 1, depth: 1)) { enc in
            enc.setBuffer(qg, offset: 0, index: 0)
            enc.setBuffer(kc, offset: 0, index: 1)
            enc.setBuffer(qn.buffer, offset: qn.offset, index: 2)
            enc.setBuffer(kn.buffer, offset: kn.offset, index: 3)
            enc.setBuffer(ropeFreq, offset: 0, index: 4)
            enc.setBuffer(q, offset: 0, index: 5)
            enc.setBuffer(qgate, offset: 0, index: 6)
            enc.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 7)
        }
        run(cb, psoAttnDecode, MTLSize(width: H, height: 1, depth: 1)) { enc in
            enc.setBuffer(q, offset: 0, index: 0)
            enc.setBuffer(kc, offset: 0, index: 1)
            enc.setBuffer(vc, offset: 0, index: 2)
            enc.setBuffer(qgate, offset: 0, index: 3)
            enc.setBuffer(ao, offset: 0, index: 4)
            enc.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 5)
            enc.setBuffer(selection, offset: 0, index: 6)
            enc.setBytes(&nSel, length: 4, index: 7)
            enc.setBytes(&useSel, length: 4, index: 8)
        }
        try gemv(cb, pre + "attn_output.weight", x: ao, y: blk)
    }

    /// Routed + shared experts for the mixed input already in `mixed`; result in `blk`.
    private func moeRouted(il: Int, after cb: MTLCommandBuffer) throws -> MTLCommandBuffer {
        let pre = "blk.\(il)."
        // Host top-10 (ds4 reference: softmax over all, lowest index wins ties, renormalize).
        let lg = routerLogits.contents().bindMemory(to: Float.self, capacity: nExperts)
        var mx = -Double.greatestFiniteMagnitude
        for i in 0..<nExperts { mx = max(mx, Double(lg[i])) }
        var prob = [Double](repeating: 0, count: nExperts)
        var sum = 0.0
        for i in 0..<nExperts { prob[i] = exp(Double(lg[i]) - mx); sum += prob[i] }
        var sel: [Int] = []
        var wsum = 0.0
        for _ in 0..<topK {
            var best = -1
            for i in 0..<nExperts where !sel.contains(i) {
                if best < 0 || prob[i] > prob[best] { best = i }
            }
            sel.append(best)
            wsum += prob[best]
        }
        lastRoutes.append(sel)
        let w = routeW.contents().bindMemory(to: Float.self, capacity: topK)
        for (i, ex) in sel.enumerated() { w[i] = Float(prob[ex] / wsum) }

        let gateT = try file.tensor(pre + "ffn_gate_exps.weight")
        let upT = try file.tensor(pre + "ffn_up_exps.weight")
        let downT = try file.tensor(pre + "ffn_down_exps.weight")
        let gateBytes = gateT.bytesPerRow * F
        let downBytes = downT.bytesPerRow * e
        let po = partOffsets.contents().bindMemory(to: UInt32.self, capacity: 3 * topK)
        var used: [MTLBuffer] = []
        for (slot, ex) in sel.enumerated() {
            let parts: [(GGUFFile.Tensor, Int)] = [(gateT, gateBytes), (upT, gateBytes), (downT, downBytes)]
            for (part, (t, bytes)) in parts.enumerated() {
                let key = (il * nExperts + ex) * 3 + part
                let v: (buffer: MTLBuffer, offset: Int)
                if let cached = expertParts[key] {
                    v = cached
                } else {
                    guard let made = file.noCopyBuffer(device: device, offset: t.offset + ex * bytes, byteCount: bytes) else {
                        throw GGUFFile.Error.format("expert view failed: layer \(il) expert \(ex)")
                    }
                    v = made
                    expertParts[key] = v
                }
                routedArgEncoder.setBuffer(v.buffer, offset: 0, index: part * 16 + slot)
                po[3 * slot + part] = UInt32(v.offset)
                used.append(v.buffer)
            }
        }

        let cb2 = queue.makeCommandBuffer()!
        // Shared expert (already computed into shY / shGate by the pre-router buffer).
        memset(blkShared.contents(), 0, e * 4)
        run(cb2, psoAddScaled, MTLSize(width: e, height: 1, depth: 1)) { enc in
            enc.setBuffer(blkShared, offset: 0, index: 0)
            enc.setBuffer(shY, offset: 0, index: 1)
            enc.setBuffer(shGate, offset: 0, index: 2)
            var op = UInt32(1)
            enc.setBytes(&op, length: 4, index: 3)
        }
        var offsets = (UInt32(gateT.bytesPerRow), UInt32(downT.bytesPerRow))
        var dV = UInt32(e), fV = UInt32(F), kV = UInt32(topK), strideV = UInt32(downIn)
        do {
            let enc = cb2.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoPhase1)
            enc.setBuffer(routedArg, offset: 0, index: 0)
            for (i, b) in used.enumerated() where i % 3 != 2 { enc.useResource(b, usage: .read) }
            enc.setBytes(&offsets, length: 8, index: 1)
            enc.setBuffer(mixed, offset: 0, index: 2)
            enc.setBuffer(acts, offset: 0, index: 3)
            enc.setBytes(&dV, length: 4, index: 4)
            enc.setBytes(&fV, length: 4, index: 5)
            enc.setBytes(&kV, length: 4, index: 6)
            enc.setBytes(&strideV, length: 4, index: 7)
            enc.setBuffer(partOffsets, offset: 0, index: 8)
            enc.setThreadgroupMemoryLength(256 * 8 + 128, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: (topK * F + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding()
        }
        do {
            let enc = cb2.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoPhase2)
            enc.setBuffer(routedArg, offset: 0, index: 0)
            for (i, b) in used.enumerated() where i % 3 == 2 { enc.useResource(b, usage: .read) }
            enc.setBytes(&offsets, length: 8, index: 1)
            enc.setBuffer(acts, offset: 0, index: 2)
            enc.setBuffer(routeW, offset: 0, index: 3)
            enc.setBuffer(blkShared, offset: 0, index: 4)
            enc.setBuffer(blk, offset: 0, index: 5)
            enc.setBytes(&dV, length: 4, index: 6)
            enc.setBytes(&strideV, length: 4, index: 7)
            enc.setBytes(&kV, length: 4, index: 8)
            enc.setBuffer(partOffsets, offset: 0, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: (e + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding()
        }
        return cb2
    }

    // MARK: - PLE (host)

    private func q8MatVec(_ t: GGUFFile.Tensor, _ x: [Float]) -> [Float] {
        let n = t.rowWidth, nb = n / 32
        var out = [Float](repeating: 0, count: t.rowCount)
        let base = file.base + t.offset
        for r in 0..<t.rowCount {
            var acc = 0.0
            let row = base + r * nb * 34
            for b in 0..<nb {
                let d = Double(Float(row.loadUnaligned(fromByteOffset: b * 34, as: Float16.self)))
                var dot: Float = 0
                for j in 0..<32 {
                    dot += Float(row.loadUnaligned(fromByteOffset: b * 34 + 2 + j, as: Int8.self)) * x[b * 32 + j]
                }
                acc += d * Double(dot)
            }
            out[r] = Float(acc)
        }
        return out
    }

    private func f32(_ name: String) throws -> UnsafeBufferPointer<Float> {
        let t = try file.tensor(name)
        precondition(t.type == .f32)
        return UnsafeBufferPointer(start: (file.base + t.offset).assumingMemoryBound(to: Float.self),
                                   count: t.byteCount / 4)
    }

    private static func sigmoid(_ x: Float) -> Float { x >= 0 ? 1 / (1 + exp(-x)) : exp(x) / (1 + exp(x)) }
    private static func silu(_ x: Float) -> Float { x / (1 + exp(-x)) }

    private func groupedRms(_ x: UnsafePointer<Float>, _ g: UnsafeBufferPointer<Float>) -> [Float] {
        var out = [Float](repeating: 0, count: hc * e)
        for s in 0..<hc {
            var ss = 0.0
            for d in 0..<e { ss += Double(x[s * e + d]) * Double(x[s * e + d]) }
            let scale = 1 / sqrt(Float(ss / Double(e)) + eps)
            for d in 0..<e { out[s * e + d] = x[s * e + d] * scale * g[s * e + d] }
        }
        return out
    }

    private func pleBlock(il: Int, token: Int) throws {
        let pre = "blk.\(il)."
        // Hash rows (ds4 `qwen4_ple_step`).
        var ctx = [token]
        var cut = false
        for s in 1..<pleNgram {
            let t = cut ? eos : plePrev[s - 1]
            cut = cut || t == eos
            ctx.append(cut ? eos : t)
        }
        var rows: [Int] = []
        for n in 2...pleNgram {
            var mixedHash = UInt64(ctx[0]) &* pleMult[0]
            for j in 1..<n { mixedHash ^= UInt64(ctx[j]) &* pleMult[j] }
            for g in 0..<plePer {
                let h = (n - 2) * plePer + g
                rows.append(Int(mixedHash % pleVocab[h] + pleOffsets[h]))
            }
        }
        plePrev = [token] + plePrev.dropLast()

        // 16 Q4_1 rows of 160.
        let pt = try pleFile.tensor("ple.weight")
        precondition(pt.type == .q4_1 && pt.rowWidth == 160)
        var emb = [Float](repeating: 0, count: e)
        for (h, r) in rows.enumerated() {
            let p = pleFile.base + pt.offset + r * pt.bytesPerRow
            for b in 0..<5 {
                let d = Float(p.loadUnaligned(fromByteOffset: b * 20, as: Float16.self))
                let m = Float(p.loadUnaligned(fromByteOffset: b * 20 + 2, as: Float16.self))
                for j in 0..<16 {
                    let byte = p.load(fromByteOffset: b * 20 + 4 + j, as: UInt8.self)
                    emb[h * 160 + b * 32 + j] = d * Float(byte & 0x0F) + m
                    emb[h * 160 + b * 32 + 16 + j] = d * Float(byte >> 4) + m
                }
            }
        }
        let key = q8MatVec(try file.tensor(pre + "ple_key.weight"), emb)
        let value = q8MatVec(try file.tensor(pre + "ple_value.weight"), emb)
        let keyn = key.withUnsafeBufferPointer { groupedRms($0.baseAddress!, try! f32(pre + "ple_norm_key.weight")) }
        let rp = R.contents().bindMemory(to: Float.self, capacity: hc * e)
        let query = groupedRms(UnsafePointer(rp), try f32(pre + "ple_norm_query.weight"))
        var gated = [Float](repeating: 0, count: hc * e)
        for s in 0..<hc {
            var dot = 0.0
            for d in 0..<e { dot += Double(keyn[s * e + d]) * Double(query[s * e + d]) }
            var g = Float(dot / sqrt(Double(e)))
            let mag = sqrt(max(abs(g), 1e-6))
            g = Self.sigmoid(g > 0 ? mag : (g < 0 ? -mag : 0))
            for d in 0..<e { gated[s * e + d] = g * value[d] }
        }
        let normed = gated.withUnsafeBufferPointer { groupedRms($0.baseAddress!, try! f32(pre + "ple_norm_conv.weight")) }
        let cw = try f32(pre + "ple_conv1d.weight")  // [hc*e][4]
        let width = hc * e
        let histRows = (4 - 1) * pleNgram
        for c in 0..<width {
            var acc = 0.0
            for k in 0..<4 {
                let back = (4 - 1 - k) * pleNgram
                let xk = back == 0 ? normed[c] : pleHist[(histRows - back) * width + c]
                acc += Double(cw[c * 4 + k]) * Double(xk)
            }
            rp[c] += gated[c] + Self.silu(Float(acc))
        }
        pleHist.removeFirst(width)
        pleHist.append(contentsOf: normed)
    }

    // MARK: - Step

    /// Forward `token` at `pos`; returns the logits.
    package func step(token: Int, pos: Int) throws -> UnsafeBufferPointer<Float> {
        precondition(pos < capacity)
        lastRoutes.removeAll(keepingCapacity: true)
        let emb = try file.tensor("token_embd.weight")
        precondition(emb.type == .bf16)
        let src = file.base + emb.offset + token * e * 2
        let rp = R.contents().bindMemory(to: Float.self, capacity: hc * e)
        for d in 0..<e {
            let v = Float(bitPattern: UInt32(src.loadUnaligned(fromByteOffset: d * 2, as: UInt16.self)) << 16)
            for s in 0..<hc { rp[s * e + d] = v }
        }

        var prof = StepProfile()
        let tStep = CFAbsoluteTimeGetCurrent()
        for il in 0..<nTrunk {
            var t0 = CFAbsoluteTimeGetCurrent()
            if il == pleLayer {
                try pleBlock(il: il, token: token)
                prof.ple += CFAbsoluteTimeGetCurrent() - t0
                t0 = CFAbsoluteTimeGetCurrent()
            }
            let pre = "blk.\(il)."
            var cb = queue.makeCommandBuffer()!
            try hcMix(cb, prefix: pre + "hc_attn", inject: true)
            if isLinear(il) { try linear(cb, il: il) } else { try attention(&cb, il: il, pos: pos) }
            combine(cb, block: blk)
            try hcMix(cb, prefix: pre + "hc_ffn", inject: true)
            try gemv(cb, pre + "ffn_gate_shexp.weight", x: mixed, y: shG)
            try gemv(cb, pre + "ffn_up_shexp.weight", x: mixed, y: shU)
            run(cb, psoSiluMul, MTLSize(width: F, height: 1, depth: 1)) { enc in
                enc.setBuffer(shG, offset: 0, index: 0)
                enc.setBuffer(shU, offset: 0, index: 1)
            }
            try gemv(cb, pre + "ffn_down_shexp.weight", x: shG, y: shY)
            try gemv(cb, pre + "ffn_gate_inp_shexp.weight", x: mixed, y: shGate)
            try gemv(cb, pre + "ffn_gate_inp.weight", x: mixed, y: routerLogits)
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            let t1 = CFAbsoluteTimeGetCurrent()
            prof.preRouter += t1 - t0

            let cb2 = try moeRouted(il: il, after: cb)
            combine(cb2, block: blk)
            let t2 = CFAbsoluteTimeGetCurrent()
            prof.route += t2 - t1
            cb2.commit()
            cb2.waitUntilCompleted()
            if let error = cb2.error { throw error }
            prof.routed += CFAbsoluteTimeGetCurrent() - t2
        }

        let tHead = CFAbsoluteTimeGetCurrent()
        let cb = queue.makeCommandBuffer()!
        try hcMix(cb, prefix: "output_hc", inject: false)
        try gemv(cb, "output.weight", x: mixed, y: logits)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw error }
        prof.head = CFAbsoluteTimeGetCurrent() - tHead
        prof.total = CFAbsoluteTimeGetCurrent() - tStep
        lastProfile = prof
        return UnsafeBufferPointer(start: logits.contents().bindMemory(to: Float.self, capacity: vocab),
                                   count: vocab)
    }
}
