import Foundation
import Metal
import MetalPerformanceShaders
import QuartzCore

/// Qwen3.8-27B (GGUF arch `qwen35`, dense) off the ISTA GSQ-RCO IQ3_S-mtp GGUF (`docs/qwen38-27b/01` .. `04`).
///
/// `Qwen38Runner` with the hyper-connections, PLE, QSA indexer and MoE taken out and `post_attention_norm` plus a
/// dense SwiGLU put in: x += mixer(rms(x, attn_norm)); x += ffn(rms(x, post_attention_norm)). The GDN and gated
/// attention kernels are `qwen38.metal`'s (the GDN output-norm gate specialized to SiLU, function constant 1), the
/// dense rows `ggml_iq.metal`'s mixed K / IQ types through no-copy views of the mapped GGUF. Every query attends
/// its whole prefix (no indexer): the lane passes below `attnMpsMinTokens`, the per-group sgemm from there.
/// `token_embd` (IQ2_S) is never mapped whole: each token's row is dequantized on the GPU from a view of its pages.
/// Float32 activations, compared token for token against `Scripts/qwen38_27b/reference_forward.py`.
package final class Qwen38DenseRunner {
    // Shape, from the GGUF (checked in init).
    package let e = 5120
    package let vocab = 248_320
    package let nTrunk: Int
    private let H = 24, Hkv = 4, D = 256, nRot = 64
    private let Hk = 16, Hv = 48, Dl = 128, convK = 4
    private let F = 17_408
    private let fullInterval: Int
    private let eps: Float

    package let capacity: Int
    package let maxBatch: Int
    /// `Q38_KV_TYPE=q8_0` (default): the K (after RMS + RoPE) and V caches are ggml `block_q8_0` rows, as in
    /// `Qwen38Runner` (docs/qwen38/22). `f32` keeps them float32.
    package let kvQ8: Bool
    private let kvRowBytes: Int
    private let file: GGUFFile
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let dense: GGMLDenseGEMV
    private let qwen38Lib: MTLLibrary

    private let psoRmsScale, psoRmsApply, psoCombine, psoSiluMul: MTLComputePipelineState
    private let psoConv, psoConvHist, psoQKNorm, psoGates, psoStep, psoNormGate: MTLComputePipelineState
    private let psoAttnPrep, psoAttnScore, psoAttnStat, psoAttnWeight, psoAttnMix: MTLComputePipelineState
    private let psoAttnGatherQ, psoAttnGatherKV, psoAttnScatter, psoKVQuantize: MTLComputePipelineState

    private var views: [String: (tensor: GGUFFile.Tensor, buffer: MTLBuffer, offset: Int)] = [:]

    // Scratch (float32), `maxBatch` rows each.
    private let R, xn, blk, ones, rmsScale: MTLBuffer
    private let qkv, conv, z, ga, gb, gdnOut: MTLBuffer
    private let qg, q, qgate, ao: MTLBuffer
    private let ffnGate, ffnUp: MTLBuffer
    private let attnMax, attnSum, nq, useSel, selection: MTLBuffer
    private let kTmp, vTmp: MTLBuffer?
    private var attnScores: MTLBuffer
    private var attnQG, attnOG, attnKG, attnVG: MTLBuffer?
    private var attnMuls: [[Int]: MPSMatrixMultiplication] = [:]
    private var logits: MTLBuffer
    private let ropeFreq: MTLBuffer

    /// Batches of at least this many tokens take the sgemm attention (`Q38_ATTN_MPS_MIN_T`, 0 = never).
    package var attnMpsMinTokens = Int(ProcessInfo.processInfo.environment["Q38_ATTN_MPS_MIN_T"] ?? "") ?? 32
    /// Largest score matrix (floats) one sgemm attention pass may hold: queries go in sub-batches of
    /// `attnScoreFloats / (6 n)` rows (at 32K and T = 512 the whole batch would be 3.7 GB; `Q27_ATTN_SCORE_MB`, default 256).
    package var attnScoreFloats = (Int(ProcessInfo.processInfo.environment["Q27_ATTN_SCORE_MB"] ?? "") ?? 256) << 18
    /// Batches of at least `gdnMinTokens` run the GDN step in chunks of `gdnChunk` (WY form, `Qwen38GDNChunk`).
    package var gdnChunk = Int(ProcessInfo.processInfo.environment["Q38_GDN_CHUNK"] ?? "") ?? 32
    package var gdnMinTokens = Int(ProcessInfo.processInfo.environment["Q38_GDN_MIN_T"] ?? "") ?? 16
    private var gdnChunked: Qwen38GDNChunk?
    /// Keep every weight view in one residency set, requested after each forward that made new views (`Q27_RESIDENT=1`).
    /// Without it a decode token took 3.9 s for 0.36 s of GPU time, with it 0.11 s, but at the default wired limit the
    /// 52-token prefill then swapped 574 MB in 2 s (docs/qwen38-27b/04 §3), so it stays opt-in for now.
    package var denseResident = ProcessInfo.processInfo.environment["Q27_RESIDENT"] == "1"
    private var denseSet: MTLResidencySet?
    private var denseSetCount = 0

    // Per-layer state.
    private var linHist: [Int: MTLBuffer] = [:]
    private var linState: [Int: MTLBuffer] = [:]
    private var kCache: [Int: MTLBuffer] = [:]
    private var vCache: [Int: MTLBuffer] = [:]

    /// Host wall time and GPU time of the last `forward` (seconds).
    package struct StepProfile {
        package var embed = 0.0
        package var layers = 0.0
        package var head = 0.0
        package var total = 0.0
        package var gpu = 0.0
    }
    package private(set) var lastProfile = StepProfile()

    package static var defaultKVType: String { ProcessInfo.processInfo.environment["Q38_KV_TYPE"] ?? "q8_0" }

    package init(gguf: URL, capacity: Int, maxBatch: Int = 1, kvType: String? = nil) throws {
        let kvName = kvType ?? Self.defaultKVType
        guard kvName == "f32" || kvName == "q8_0" else { throw GGUFFile.Error.format("Q38_KV_TYPE \(kvName): f32 or q8_0") }
        kvQ8 = kvName == "q8_0"
        kvRowBytes = kvQ8 ? Hkv * (D / 32) * 34 : Hkv * D * 4
        let file = try GGUFFile(url: gguf)
        self.file = file
        self.capacity = capacity
        self.maxBatch = max(maxBatch, 1)
        guard try file.value("general.architecture").string == "qwen35" else {
            throw GGUFFile.Error.format("not a qwen35 GGUF")
        }
        let nLayer = try file.value("qwen35.block_count").int!
        nTrunk = nLayer - (try file.value("qwen35.nextn_predict_layers").int!)
        fullInterval = try file.value("qwen35.full_attention_interval").int!
        eps = Float(try file.value("qwen35.attention.layer_norm_rms_epsilon").double!)
        let shapeOK = try file.value("qwen35.embedding_length").int == e
            && file.value("qwen35.feed_forward_length").int == F
            && file.value("qwen35.attention.head_count").int == H
            && file.value("qwen35.attention.head_count_kv").int == Hkv
            && file.value("qwen35.attention.key_length").int == D
            && file.value("qwen35.rope.dimension_count").int == nRot
            && file.value("qwen35.ssm.state_size").int == Dl
            && file.value("qwen35.ssm.group_count").int == Hk
            && file.value("qwen35.ssm.time_step_rank").int == Hv
            && file.value("qwen35.ssm.conv_kernel").int == convK
            && file.value("qwen35.rope.freq_base").double == 10_000_000
        guard shapeOK else { throw GGUFFile.Error.format("unexpected qwen35 shape") }

        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw GGUFFile.Error.open("no Metal device")
        }
        self.device = device
        self.queue = queue
        dense = try GGMLDenseGEMV(device: device)
        // The FFN rows (17,408 x 5,120 = 89M weights, 356 MB of float32) take the dequant + sgemm prefill path
        // (`Q27_MPS_MAX_W` in millions, default 96); the LM head stays direct.
        dense.mpsMaxWeights = (Int(ProcessInfo.processInfo.environment["Q27_MPS_MAX_W"] ?? "") ?? 96) * 1_000_000

        let lib = try MetalContext.moduleLibrary(device: device, module: "qwen38")
        qwen38Lib = lib
        func pso(_ name: String, _ constants: MTLFunctionConstantValues? = nil) throws -> MTLComputePipelineState {
            let fn = try constants.map { try lib.makeFunction(name: name, constantValues: $0) } ?? lib.makeFunction(name: name)
            guard let fn else { throw GGUFFile.Error.format("kernel \(name) missing") }
            return try device.makeComputePipelineState(function: fn)
        }
        let kvConstants = MTLFunctionConstantValues()
        var q8Flag = kvQ8
        kvConstants.setConstantValue(&q8Flag, type: .bool, index: 0)
        let siluConstants = MTLFunctionConstantValues()
        var silu = true
        siluConstants.setConstantValue(&silu, type: .bool, index: 1)
        psoRmsScale = try pso("q38_group_rms_scale")
        psoRmsApply = try pso("q38_rms_apply")
        psoCombine = try pso("q38_hc_combine")
        psoSiluMul = try pso("q38_silu_mul")
        psoConv = try pso("q38_gdn_conv")
        psoConvHist = try pso("q38_gdn_conv_hist")
        psoQKNorm = try pso("q38_gdn_qk_norm")
        psoGates = try pso("q38_gdn_gates")
        psoStep = try pso("q38_gdn_step")
        psoNormGate = try pso("q38_gdn_norm_gate", siluConstants)
        psoAttnPrep = try pso("q38_attn_prep", kvConstants)
        psoAttnScore = try pso("q38_attn_score", kvConstants)
        psoAttnStat = try pso("q38_attn_stat")
        psoAttnWeight = try pso("q38_attn_weight")
        psoAttnMix = try pso("q38_attn_mix", kvConstants)
        psoAttnGatherQ = try pso("q38_attn_gather_q")
        psoAttnGatherKV = try pso("q38_attn_gather_kv", kvConstants)
        psoAttnScatter = try pso("q38_attn_scatter_out")
        psoKVQuantize = try pso("q38_kv_quantize")

        let B = self.maxBatch
        func buf(_ count: Int) -> MTLBuffer {
            device.makeBuffer(length: max(count, 1) * 4, options: .storageModeShared)!
        }
        let C = 2 * Hk * Dl + Hv * Dl
        R = buf(B * e); xn = buf(B * e); blk = buf(B * e); rmsScale = buf(B)
        ones = buf(B)
        ones.contents().bindMemory(to: Float.self, capacity: B).update(repeating: 1, count: B)
        qkv = buf(B * C); conv = buf(B * C)
        z = buf(B * Hv * Dl); ga = buf(B * Hv); gb = buf(B * Hv); gdnOut = buf(B * Hv * Dl)
        qg = buf(B * 2 * H * D); q = buf(B * H * D); qgate = buf(B * H * D); ao = buf(B * H * D)
        ffnGate = buf(B * F); ffnUp = buf(B * F)
        attnMax = buf(B * H); attnSum = buf(B * H); nq = buf(B); useSel = buf(B); selection = buf(1)
        kTmp = kvQ8 ? buf(B * Hkv * D) : nil
        vTmp = kvQ8 ? buf(B * Hkv * D) : nil
        attnScores = buf(1)
        logits = buf(vocab)
        let rotDims = nRot
        let freq = (0..<(rotDims / 2)).map { i in Float(pow(10_000_000.0, -2.0 * Double(i) / Double(rotDims))) }
        ropeFreq = device.makeBuffer(bytes: freq, length: freq.count * 4, options: .storageModeShared)!
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

    /// dispatchThreadgroups over `groups`, 32 threads (one SIMD group of lanes) each.
    private func lanes(_ cb: MTLCommandBuffer, _ pso: MTLComputePipelineState, _ groups: MTLSize,
                       _ setup: (MTLComputeCommandEncoder) -> Void) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        setup(enc)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func size(_ w: Int, _ h: Int = 1, _ d: Int = 1) -> MTLSize { MTLSize(width: w, height: h, depth: d) }

    private func gemv(_ cb: MTLCommandBuffer, _ name: String, x: MTLBuffer, xOffset: Int = 0,
                      y: MTLBuffer, yOffset: Int = 0, tokens: Int) throws {
        let v = try view(name)
        dense.encode(commandBuffer: cb, type: v.tensor.type, weights: v.buffer, weightsOffset: v.offset,
                     x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                     m: v.tensor.rowCount, n: v.tensor.rowWidth, tokens: tokens)
    }

    private func setF32(_ enc: MTLComputeCommandEncoder, _ name: String, index: Int) throws {
        let v = try view(name)
        precondition(v.tensor.type == .f32, "\(name) is not F32")
        enc.setBuffer(v.buffer, offset: v.offset, index: index)
    }

    /// RMS norm of `rows` rows of e with the F32 gamma `gamma`.
    private func rms(_ cb: MTLCommandBuffer, x: MTLBuffer, gamma: String, out: MTLBuffer, rows: Int) throws {
        let g = try view(gamma)
        var p = (UInt32(rows), UInt32(e), eps, UInt32(e))
        let len = MemoryLayout.size(ofValue: p)
        lanes(cb, psoRmsScale, size(rows)) { enc in
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(rmsScale, offset: 0, index: 1)
            enc.setBytes(&p, length: len, index: 2)
        }
        run(cb, psoRmsApply, size(rows * e)) { enc in
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(g.buffer, offset: g.offset, index: 1)
            enc.setBuffer(out, offset: 0, index: 2)
            enc.setBytes(&p, length: len, index: 3)
            enc.setBuffer(rmsScale, offset: 0, index: 4)
        }
    }

    /// R += blk (`q38_hc_combine` with one stream and unit injection).
    private func addResidual(_ cb: MTLCommandBuffer, T: Int) {
        run(cb, psoCombine, size(T * e)) { enc in
            enc.setBuffer(R, offset: 0, index: 0)
            enc.setBuffer(blk, offset: 0, index: 1)
            enc.setBuffer(ones, offset: 0, index: 2)
            var p = (UInt32(1), UInt32(e))
            enc.setBytes(&p, length: 8, index: 3)
        }
    }

    // MARK: - Blocks

    /// Gated DeltaNet on `xn`; result in `blk`.
    private func linear(_ cb: MTLCommandBuffer, il: Int, T: Int) throws {
        let pre = "blk.\(il)."
        let C = 2 * Hk * Dl + Hv * Dl
        try gemv(cb, pre + "attn_qkv.weight", x: xn, y: qkv, tokens: T)
        try gemv(cb, pre + "attn_gate.weight", x: xn, y: z, tokens: T)
        try gemv(cb, pre + "ssm_beta.weight", x: xn, y: gb, tokens: T)
        try gemv(cb, pre + "ssm_alpha.weight", x: xn, y: ga, tokens: T)
        let hist = linHist[il] ?? device.makeBuffer(length: (convK - 1) * C * 4, options: .storageModeShared)!
        let state = linState[il] ?? device.makeBuffer(length: Hv * Dl * Dl * 4, options: .storageModeShared)!
        linHist[il] = hist
        linState[il] = state
        let cw = try view(pre + "ssm_conv1d.weight")
        var cp = (UInt32(C), UInt32(convK), UInt32(T))
        let cpLen = MemoryLayout.size(ofValue: cp)
        run(cb, psoConv, size(C, T)) { enc in
            enc.setBuffer(qkv, offset: 0, index: 0)
            enc.setBuffer(hist, offset: 0, index: 1)
            enc.setBuffer(cw.buffer, offset: cw.offset, index: 2)
            enc.setBuffer(conv, offset: 0, index: 3)
            enc.setBytes(&cp, length: cpLen, index: 4)
        }
        run(cb, psoConvHist, size(C)) { enc in
            enc.setBuffer(qkv, offset: 0, index: 0)
            enc.setBuffer(hist, offset: 0, index: 1)
            enc.setBytes(&cp, length: cpLen, index: 2)
        }
        var gp = (UInt32(Hk), UInt32(Hv), UInt32(Dl), UInt32(T))
        let gpLen = MemoryLayout.size(ofValue: gp)
        var cV = UInt32(C)
        run(cb, psoQKNorm, size(2 * Hk, T)) { enc in
            enc.setBuffer(conv, offset: 0, index: 0)
            enc.setBytes(&gp, length: gpLen, index: 1)
            enc.setBytes(&cV, length: 4, index: 2)
        }
        let a = try view(pre + "ssm_a"), dt = try view(pre + "ssm_dt.bias")
        precondition(a.tensor.type == .f32 && dt.tensor.type == .f32)
        run(cb, psoGates, size(Hv, T)) { enc in
            enc.setBuffer(ga, offset: 0, index: 0)
            enc.setBuffer(gb, offset: 0, index: 1)
            enc.setBuffer(a.buffer, offset: a.offset, index: 2)
            enc.setBuffer(dt.buffer, offset: dt.offset, index: 3)
            enc.setBytes(&gp, length: gpLen, index: 4)
        }
        if gdnChunk > 0 && T >= gdnMinTokens {
            if gdnChunked?.chunk != gdnChunk {
                gdnChunked = try Qwen38GDNChunk(device: device, library: qwen38Lib, keyHeads: Hk, valueHeads: Hv,
                                                headDim: Dl, chunk: gdnChunk)
            }
            gdnChunked!.encode(cb, conv: conv, a: ga, b: gb, state: state, out: gdnOut, T: T)
        } else {
            var snapN = UInt32(0)
            lanes(cb, psoStep, size(Dl, Hv)) { enc in
                enc.setBuffer(conv, offset: 0, index: 0)
                enc.setBuffer(ga, offset: 0, index: 1)
                enc.setBuffer(gb, offset: 0, index: 2)
                enc.setBuffer(state, offset: 0, index: 5)
                enc.setBuffer(gdnOut, offset: 0, index: 6)
                enc.setBytes(&gp, length: gpLen, index: 7)
                enc.setBytes(&cV, length: 4, index: 8)
                enc.setBuffer(state, offset: 0, index: 9)
                enc.setBytes(&snapN, length: 4, index: 10)
                enc.setBuffer(state, offset: 0, index: 11)
            }
        }
        let nw = try view(pre + "ssm_norm.weight")
        run(cb, psoNormGate, size(Hv, T)) { enc in
            enc.setBuffer(gdnOut, offset: 0, index: 0)
            enc.setBuffer(z, offset: 0, index: 1)
            enc.setBuffer(nw.buffer, offset: nw.offset, index: 2)
            var ep = eps
            enc.setBytes(&gp, length: gpLen, index: 3)
            enc.setBytes(&ep, length: 4, index: 4)
        }
        try gemv(cb, pre + "ssm_out.weight", x: gdnOut, y: blk, tokens: T)
    }

    /// Gated attention on `xn` at positions `pos0 ..< pos0 + T`, each query over its whole prefix; result in `blk`.
    /// `nq` / `useSel` are set by `forward`.
    private func attention(_ cb: MTLCommandBuffer, il: Int, pos0: Int, T: Int) throws {
        let pre = "blk.\(il)."
        let kc = kCache[il] ?? device.makeBuffer(length: capacity * kvRowBytes, options: .storageModeShared)!
        let vc = vCache[il] ?? device.makeBuffer(length: capacity * kvRowBytes, options: .storageModeShared)!
        kCache[il] = kc
        vCache[il] = vc
        try gemv(cb, pre + "attn_q.weight", x: xn, y: qg, tokens: T)
        if kvQ8 {
            // K is quantized by `q38_attn_prep` after its norm and rope.
            try gemv(cb, pre + "attn_k.weight", x: xn, y: kTmp!, tokens: T)
            try gemv(cb, pre + "attn_v.weight", x: xn, y: vTmp!, tokens: T)
            var dV = UInt32(D), row0 = UInt32(pos0 * Hkv)
            run(cb, psoKVQuantize, size(T * Hkv)) { enc in
                enc.setBuffer(vTmp, offset: 0, index: 0)
                enc.setBuffer(vc, offset: 0, index: 1)
                enc.setBytes(&dV, length: 4, index: 2)
                enc.setBytes(&row0, length: 4, index: 3)
            }
        } else {
            try gemv(cb, pre + "attn_k.weight", x: xn, y: kc, yOffset: pos0 * Hkv * D * 4, tokens: T)
            try gemv(cb, pre + "attn_v.weight", x: xn, y: vc, yOffset: pos0 * Hkv * D * 4, tokens: T)
        }
        var pp = (UInt32(H), UInt32(Hkv), UInt32(D), UInt32(nRot), UInt32(pos0), eps)
        let qn = try view(pre + "attn_q_norm.weight")
        let kn = try view(pre + "attn_k_norm.weight")
        run(cb, psoAttnPrep, size(H + Hkv, T)) { enc in
            enc.setBuffer(qg, offset: 0, index: 0)
            enc.setBuffer(kc, offset: 0, index: 1)
            enc.setBuffer(qn.buffer, offset: qn.offset, index: 2)
            enc.setBuffer(kn.buffer, offset: kn.offset, index: 3)
            enc.setBuffer(ropeFreq, offset: 0, index: 4)
            enc.setBuffer(q, offset: 0, index: 5)
            enc.setBuffer(qgate, offset: 0, index: 6)
            enc.setBytes(&pp, length: MemoryLayout.size(ofValue: pp), index: 7)
            enc.setBuffer(kvQ8 ? kTmp! : kc, offset: 0, index: 8)
        }
        let n = pos0 + T
        if attnMpsMinTokens > 0 && T >= attnMpsMinTokens {
            attentionSgemm(cb, kc: kc, vc: vc, n: n, T: T)
        } else {
            let nCap = n
            if attnScores.length < T * H * nCap * 4 {
                attnScores = device.makeBuffer(length: T * H * nCap * 4, options: .storageModeShared)!
            }
            var ap = (UInt32(H), UInt32(Hkv), UInt32(D), UInt32(nCap), UInt32(T))
            let apLen = MemoryLayout.size(ofValue: ap)
            lanes(cb, psoAttnScore, size(n, H, T)) { enc in
                enc.setBuffer(q, offset: 0, index: 0)
                enc.setBuffer(kc, offset: 0, index: 1)
                enc.setBuffer(attnScores, offset: 0, index: 2)
                enc.setBuffer(selection, offset: 0, index: 3)
                enc.setBytes(&ap, length: apLen, index: 4)
                enc.setBuffer(nq, offset: 0, index: 5)
                enc.setBuffer(useSel, offset: 0, index: 6)
            }
            for op in [UInt32(0), 1] {
                lanes(cb, psoAttnStat, size(H, T)) { enc in
                    enc.setBuffer(attnScores, offset: 0, index: 0)
                    enc.setBuffer(attnMax, offset: 0, index: 1)
                    enc.setBuffer(attnSum, offset: 0, index: 2)
                    enc.setBytes(&ap, length: apLen, index: 3)
                    var o = op
                    enc.setBytes(&o, length: 4, index: 4)
                    enc.setBuffer(nq, offset: 0, index: 5)
                }
            }
            run(cb, psoAttnWeight, size(n, T * H)) { enc in
                enc.setBuffer(attnScores, offset: 0, index: 0)
                enc.setBuffer(attnMax, offset: 0, index: 1)
                enc.setBuffer(attnSum, offset: 0, index: 2)
                enc.setBytes(&ap, length: apLen, index: 3)
                enc.setBuffer(nq, offset: 0, index: 4)
            }
            lanes(cb, psoAttnMix, size(D, H, T)) { enc in
                enc.setBuffer(attnScores, offset: 0, index: 0)
                enc.setBuffer(vc, offset: 0, index: 1)
                enc.setBuffer(qgate, offset: 0, index: 2)
                enc.setBuffer(ao, offset: 0, index: 3)
                enc.setBuffer(selection, offset: 0, index: 4)
                enc.setBytes(&ap, length: apLen, index: 5)
                enc.setBuffer(nq, offset: 0, index: 6)
                enc.setBuffer(useSel, offset: 0, index: 7)
            }
        }
        try gemv(cb, pre + "attn_output.weight", x: ao, y: blk, tokens: T)
    }

    /// `Qwen38Runner.attentionSgemm`: one KV group at a time as two sgemms over the first `n` cache rows, the T
    /// queries in sub-batches whose score matrix fits `attnScoreFloats`; writes `ao`.
    private func attentionSgemm(_ cb: MTLCommandBuffer, kc: MTLBuffer, vc: MTLBuffer, n: Int, T: Int) {
        let G = H / Hkv
        let sub = max(1, min(T, attnScoreFloats / (G * n)))
        func ensure(_ b: inout MTLBuffer?, _ floats: Int) {
            if (b?.length ?? 0) < floats * 4 { b = device.makeBuffer(length: floats * 4, options: .storageModePrivate) }
        }
        ensure(&attnQG, sub * G * D)
        ensure(&attnOG, sub * G * D)
        ensure(&attnKG, n * D)
        ensure(&attnVG, n * D)
        if attnScores.length < sub * G * n * 4 {
            attnScores = device.makeBuffer(length: sub * G * n * 4, options: .storageModeShared)!
        }
        var gpKV = (UInt32(H), UInt32(Hkv), UInt32(D), UInt32(n), UInt32(T))
        let len = MemoryLayout.size(ofValue: gpKV)
        let kvDesc = MPSMatrixDescriptor(rows: n, columns: D, rowBytes: D * 4, dataType: .float32)
        for g in 0..<Hkv {
            var gV = UInt32(g)
            for (cache, out) in [(kc, attnKG), (vc, attnVG)] {
                run(cb, psoAttnGatherKV, size(n * D)) { enc in
                    enc.setBuffer(cache, offset: 0, index: 0)
                    enc.setBuffer(out, offset: 0, index: 1)
                    enc.setBytes(&gpKV, length: len, index: 2)
                    enc.setBytes(&gV, length: 4, index: 3)
                }
            }
            var t0 = 0
            while t0 < T {
                let Ts = min(sub, T - t0)
                var ap = (UInt32(G), UInt32(1), UInt32(D), UInt32(n), UInt32(Ts))   // stat / weight: G rows per token
                var gp = (UInt32(H), UInt32(Hkv), UInt32(D), UInt32(n), UInt32(Ts)) // gather / scatter
                let scoresDesc = MPSMatrixDescriptor(rows: Ts * G, columns: n, rowBytes: n * 4, dataType: .float32)
                let qDesc = MPSMatrixDescriptor(rows: Ts * G, columns: D, rowBytes: D * 4, dataType: .float32)
                let mulQK = attnMuls[[0, Ts, n]] ?? MPSMatrixMultiplication(
                    device: device, transposeLeft: false, transposeRight: true,
                    resultRows: Ts * G, resultColumns: n, interiorColumns: D, alpha: 1 / Double(D).squareRoot(), beta: 0)
                let mulWV = attnMuls[[1, Ts, n]] ?? MPSMatrixMultiplication(
                    device: device, transposeLeft: false, transposeRight: false,
                    resultRows: Ts * G, resultColumns: D, interiorColumns: n, alpha: 1, beta: 0)
                attnMuls[[0, Ts, n]] = mulQK
                attnMuls[[1, Ts, n]] = mulWV
                let rowOff = t0 * H * D * 4
                run(cb, psoAttnGatherQ, size(Ts * G * D)) { enc in
                    enc.setBuffer(q, offset: rowOff, index: 0)
                    enc.setBuffer(attnQG, offset: 0, index: 1)
                    enc.setBytes(&gp, length: len, index: 2)
                    enc.setBytes(&gV, length: 4, index: 3)
                }
                mulQK.encode(commandBuffer: cb, leftMatrix: MPSMatrix(buffer: attnQG!, descriptor: qDesc),
                             rightMatrix: MPSMatrix(buffer: attnKG!, descriptor: kvDesc),
                             resultMatrix: MPSMatrix(buffer: attnScores, descriptor: scoresDesc))
                for op in [UInt32(0), 1] {
                    lanes(cb, psoAttnStat, size(G, Ts)) { enc in
                        enc.setBuffer(attnScores, offset: 0, index: 0)
                        enc.setBuffer(attnMax, offset: 0, index: 1)
                        enc.setBuffer(attnSum, offset: 0, index: 2)
                        enc.setBytes(&ap, length: len, index: 3)
                        var o = op
                        enc.setBytes(&o, length: 4, index: 4)
                        enc.setBuffer(nq, offset: t0 * 4, index: 5)
                    }
                }
                run(cb, psoAttnWeight, size(n, Ts * G)) { enc in
                    enc.setBuffer(attnScores, offset: 0, index: 0)
                    enc.setBuffer(attnMax, offset: 0, index: 1)
                    enc.setBuffer(attnSum, offset: 0, index: 2)
                    enc.setBytes(&ap, length: len, index: 3)
                    enc.setBuffer(nq, offset: t0 * 4, index: 4)
                }
                mulWV.encode(commandBuffer: cb, leftMatrix: MPSMatrix(buffer: attnScores, descriptor: scoresDesc),
                             rightMatrix: MPSMatrix(buffer: attnVG!, descriptor: kvDesc),
                             resultMatrix: MPSMatrix(buffer: attnOG!, descriptor: qDesc))
                run(cb, psoAttnScatter, size(Ts * G * D)) { enc in
                    enc.setBuffer(attnOG, offset: 0, index: 0)
                    enc.setBuffer(qgate, offset: rowOff, index: 1)
                    enc.setBuffer(ao, offset: rowOff, index: 2)
                    enc.setBytes(&gp, length: len, index: 3)
                    enc.setBytes(&gV, length: 4, index: 4)
                }
                t0 += Ts
            }
        }
    }

    /// Drops the grown prefill scratch (dequantized rows, attention scores) before decode.
    package func dropScratch() {
        dense.dropScratch()
        attnQG = nil; attnOG = nil; attnKG = nil; attnVG = nil
        attnScores = device.makeBuffer(length: 4, options: .storageModeShared)!
        attnMuls.removeAll()
    }

    /// Dense SwiGLU on `xn`; result in `blk`.
    private func ffn(_ cb: MTLCommandBuffer, il: Int, T: Int) throws {
        let pre = "blk.\(il)."
        try gemv(cb, pre + "ffn_gate.weight", x: xn, y: ffnGate, tokens: T)
        try gemv(cb, pre + "ffn_up.weight", x: xn, y: ffnUp, tokens: T)
        run(cb, psoSiluMul, size(T * F)) { enc in
            enc.setBuffer(ffnGate, offset: 0, index: 0)
            enc.setBuffer(ffnUp, offset: 0, index: 1)
        }
        try gemv(cb, pre + "ffn_down.weight", x: ffnGate, y: blk, tokens: T)
    }

    // MARK: - Forward

    /// Back to position 0: the GDN state and conv history zeroed. KV rows are by position.
    package func reset() {
        for b in Array(linHist.values) + Array(linState.values) { memset(b.contents(), 0, b.length) }
    }

    /// Token embeddings (IQ2_S rows dequantized on the GPU from views of their pages) into `R`.
    private func embed(_ cb: MTLCommandBuffer, _ tokens: [Int]) throws {
        let emb = try file.tensor("token_embd.weight")
        precondition(emb.rowWidth == e)
        for (t, token) in tokens.enumerated() {
            guard let (b, off) = file.noCopyBuffer(device: device, offset: emb.offset + token * emb.bytesPerRow,
                                                   byteCount: emb.bytesPerRow) else {
                throw GGUFFile.Error.format("no-copy view failed for token \(token)")
            }
            dense.encodeDequant(commandBuffer: cb, type: emb.type, weights: b, weightsOffset: off,
                                out: R, outOffset: t * e * 4, m: 1, n: e)
        }
    }

    /// Forward `token` at `pos`; returns its logits.
    package func step(token: Int, pos: Int) throws -> UnsafeBufferPointer<Float> {
        try forward(tokens: [token], startPos: pos)
    }

    /// Forward `tokens` at positions `startPos ..< startPos + tokens.count` (the cache must hold every earlier
    /// position). Returns the last token's logits, or with `allLogits` every token's (`[T][vocab]`).
    package func forward(tokens: [Int], startPos: Int, allLogits: Bool = false) throws -> UnsafeBufferPointer<Float> {
        try autoreleasepool { try forwardBody(tokens: tokens, startPos: startPos, allLogits: allLogits) }
    }

    private func forwardBody(tokens: [Int], startPos: Int, allLogits: Bool) throws -> UnsafeBufferPointer<Float> {
        let T = tokens.count
        precondition(T >= 1 && T <= maxBatch && startPos + T <= capacity)
        var prof = StepProfile()
        let tStep = CFAbsoluteTimeGetCurrent()
        var buffers: [MTLCommandBuffer] = []
        let nqp = nq.contents().bindMemory(to: UInt32.self, capacity: T)
        let usp = useSel.contents().bindMemory(to: UInt32.self, capacity: T)
        for t in 0..<T {
            nqp[t] = UInt32(startPos + t + 1)
            usp[t] = 0
        }
        let cb0 = queue.makeCommandBuffer()!
        try embed(cb0, tokens)
        cb0.commit()
        buffers.append(cb0)
        prof.embed = CFAbsoluteTimeGetCurrent() - tStep

        let tLayers = CFAbsoluteTimeGetCurrent()
        for il in 0..<nTrunk {
            let pre = "blk.\(il)."
            let cb = queue.makeCommandBuffer()!
            try rms(cb, x: R, gamma: pre + "attn_norm.weight", out: xn, rows: T)
            if isLinear(il) { try linear(cb, il: il, T: T) } else { try attention(cb, il: il, pos0: startPos, T: T) }
            addResidual(cb, T: T)
            try rms(cb, x: R, gamma: pre + "post_attention_norm.weight", out: xn, rows: T)
            try ffn(cb, il: il, T: T)
            addResidual(cb, T: T)
            cb.commit()
            buffers.append(cb)
        }
        prof.layers = CFAbsoluteTimeGetCurrent() - tLayers

        let tHead = CFAbsoluteTimeGetCurrent()
        let rows = allLogits ? T : 1
        if logits.length < rows * vocab * 4 {
            logits = device.makeBuffer(length: rows * vocab * 4, options: .storageModeShared)!
        }
        let cb = queue.makeCommandBuffer()!
        try rms(cb, x: R, gamma: "output_norm.weight", out: xn, rows: T)
        try gemv(cb, "output.weight", x: xn, xOffset: allLogits ? 0 : (T - 1) * e * 4, y: logits, tokens: rows)
        cb.commit()
        buffers.append(cb)
        cb.waitUntilCompleted()
        for b in buffers {
            if let error = b.error { throw error }
            prof.gpu += b.gpuEndTime - b.gpuStartTime
        }
        prof.head = CFAbsoluteTimeGetCurrent() - tHead
        if denseResident && views.count != denseSetCount {
            if denseSet == nil {
                let d = MTLResidencySetDescriptor()
                d.label = "q27-dense"
                denseSet = try device.makeResidencySet(descriptor: d)
                queue.addResidencySet(denseSet!)
            }
            let set = denseSet!
            set.removeAllAllocations()
            var seen = Set<ObjectIdentifier>()
            for v in views.values where seen.insert(ObjectIdentifier(v.buffer)).inserted { set.addAllocation(v.buffer) }
            set.commit()
            set.requestResidency()
            denseSetCount = views.count
        }
        prof.total = CFAbsoluteTimeGetCurrent() - tStep
        lastProfile = prof
        return UnsafeBufferPointer(start: logits.contents().bindMemory(to: Float.self, capacity: rows * vocab),
                                   count: rows * vocab)
    }
}
