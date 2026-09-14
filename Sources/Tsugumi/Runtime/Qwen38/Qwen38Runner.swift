import Foundation
import Metal
import MetalPerformanceShaders
import QuartzCore

/// Qwen3.8-Flash-Next (qwen4exp) off the DS4-IQ2 GGUF: the Q2 verification runner
/// (`docs/qwen38/01-Q2-FIRST-LIGHT.md` .. `04-QSA-GPU.md`).
///
/// Correctness first. `forward` runs T >= 1 tokens at once (decode is T = 1),
/// float32 activations, every weight read in its GGUF bytes (Q8_0 / F16 / F32
/// dense through no-copy views of the mapped file, IQ2_XXS / Q2_K routed experts
/// through per-expert no-copy views of the same mapping, so the page cache is the
/// expert cache). The layer is GPU work except for what the reference makes easy
/// to check on the host: the router's top-10 (logits read back), QSA's block
/// choice above its budget for batches under 32 tokens (larger batches select on
/// the GPU, `attentionSelected`), and PLE's hashed n-gram rows.
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
    private let idxHeads = 4, idxD = 128
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
    /// Largest T one `forward` takes.
    package let maxBatch: Int
    private let file: GGUFFile
    private let pleFile: GGUFFile
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let dense: GGMLDenseGEMV

    private let psoRmsScale, psoRmsApply, psoUnary, psoMix, psoCombine, psoAddScaled, psoSiluMul: MTLComputePipelineState
    private let qwen38Lib: MTLLibrary
    private let psoConv, psoConvHist, psoQKNorm, psoGates, psoStep, psoNormGate, psoAttnPrep: MTLComputePipelineState
    private let psoAttnScore, psoAttnStat, psoAttnWeight, psoAttnMix: MTLComputePipelineState
    private let psoAttnGatherQ, psoAttnGatherKV, psoAttnScatter: MTLComputePipelineState
    private let psoIdxBlockKey, psoIdxQPrep, psoIdxScore: MTLComputePipelineState
    private let psoIdxReluSum, psoIdxTopK, psoIdxUnion, psoAttnMaskSel, psoAttnGatherKVList: MTLComputePipelineState
    private let psoPleGate, psoPleGated, psoPleConvAdd, psoPleHist: MTLComputePipelineState
    private let psoPhase1, psoPhase2: MTLComputePipelineState
    private let psoDeqGateUp, psoDeqDown, psoGatherPairs, psoSiluHalves, psoScatterWeighted: MTLComputePipelineState
    private let psoDeqQ4KGateUp, psoDeqMXFP4Down: MTLComputePipelineState

    private var views: [String: (tensor: GGUFFile.Tensor, buffer: MTLBuffer, offset: Int)] = [:]

    // Scratch (float32), `batchRows` rows each (`allocateBatch`).
    private var R, xn, lo, gate, mixed, inj, blk, blkShared, rmsScale: MTLBuffer!
    private var qkv, conv, z, ga, gb, lo6144: MTLBuffer!
    private var qg, q, qgate, ao, iq: MTLBuffer!
    private var routerLogits, previewLogits, shG, shU, shY, shGate: MTLBuffer!
    private var acts, routeW, pairSlot: MTLBuffer!
    private var pleEmb, pleKey, pleValue, pleKeyN, pleQuery, pleGateBuf, pleGated, pleNormed: MTLBuffer!
    private let pleHist: MTLBuffer
    private var attnMax, attnSum, nq, useSel: MTLBuffer!
    /// Rows the batch scratch is allocated for: `maxBatch` while prefilling, `smallBatchRows` in decode.
    /// Sized from maxBatch they stayed resident through decode (~3 GB at chunk 2048) and a 12K decode
    /// swapped (`docs/qwen38/09`). `Q38_SHRINK_BATCH=0` keeps them at maxBatch.
    private var batchRows = 0
    private let smallBatchRows = 32
    package var shrinkBatch = ProcessInfo.processInfo.environment["Q38_SHRINK_BATCH"] != "0"
    private var attnScores: MTLBuffer
    private var selection: MTLBuffer
    private var nCap = 0
    private var attnQG, attnOG, attnKG, attnVG: MTLBuffer?
    private var idxDots, attnSelScores, selAny, selList: MTLBuffer?
    private var selThr, selCut: MTLBuffer!
    /// Queries per union in `attentionSelected` (`Q38_ATTN_SEL_B`).
    package var attnSelBatch = Int(ProcessInfo.processInfo.environment["Q38_ATTN_SEL_B"] ?? "") ?? 64
    /// Checks only: run the host path too on every selected batch and print the largest `ao` difference
    /// (`Q38_ATTN_SEL_COMPARE=1`).
    package var compareSelected = ProcessInfo.processInfo.environment["Q38_ATTN_SEL_COMPARE"] == "1"
    private var attnMuls: [[Int]: MPSMatrixMultiplication] = [:]
    /// Batches of at least this many tokens take the sgemm attention, `attentionSgemm` below the QSA
    /// budget and `attentionSelected` above it (`Q38_ATTN_MPS_MIN_T`, 0 = never: the host selection
    /// and the five passes, kept as the oracle).
    package var attnMpsMinTokens = Int(ProcessInfo.processInfo.environment["Q38_ATTN_MPS_MIN_T"] ?? "") ?? 32
    private var idxScores: MTLBuffer!
    private var logits: MTLBuffer
    private let routedArg: MTLBuffer
    private let routedArgEncoder: MTLArgumentEncoder
    private let partOffsets: MTLBuffer
    /// No-copy views of one expert's gate / up / down rows in the mapped GGUF,
    /// keyed (layer * nExperts + expert) * 3 + part. Created on first use.
    private var expertParts: [Int: (buffer: MTLBuffer, offset: Int)] = [:]
    private let ropeFreq: MTLBuffer

    // Per-layer state.
    private var linHist: [Int: MTLBuffer] = [:]
    private var linState: [Int: MTLBuffer] = [:]
    private var kCache: [Int: MTLBuffer] = [:]
    private var vCache: [Int: MTLBuffer] = [:]
    private var idxRawKeys: [Int: MTLBuffer] = [:]    // [cap][128]
    private var idxBlockKeys: [Int: MTLBuffer] = [:]  // [cap/4][128], filled as blocks complete
    /// QSA budget in tokens (GGUF: 2048). Lowered only by checks, to make the
    /// selection fire on short prompts the CPU reference can afford.
    package var indexerTopK: Int
    private var plePrev: [Int]

    /// Host wall time of the last `forward`, by stage (seconds).
    package struct StepProfile {
        package var ple = 0.0         // token embeddings + PLE n-gram rows (the rest of PLE is in preRouter)
        package var preRouter = 0.0   // encode + GPU wait of the pre-router buffers, all layers
        package var route = 0.0       // host top-10, expert views and advise, all layers
        package var routed = 0.0      // encode + GPU wait of the routed buffers, all layers
        package var head = 0.0
        package var total = 0.0
        package var preGPU = 0.0      // gpuEndTime - gpuStartTime of the pre-router buffers
        package var routedGPU = 0.0   // same, routed buffers
        /// GPU ms by pre-router section, filled only when `splitPreRouter` is set.
        package var sections: [String: Double] = [:]
        package var missBytes = 0      // selected expert bytes not in the page cache at route time (`countMisses`)
        package var distinctExperts = 0  // distinct (layer, expert) pairs of the batch, summed over layers
        package var newViewBytes = 0     // expert views made by this forward (their first command buffer, docs/qwen38/13)
        package var routeTopK = 0.0, routeViews = 0.0, routeAdvise = 0.0   // parts of `route`
        package var missTime = 0.0      // the `countMisses` mincore calls (inside `routeAdvise`)
        /// Parts of `routed` - `routedGPU` (trunk only), from the buffers' host times: `commit` to `kernelStartTime`,
        /// `kernelStartTime` to `gpuStartTime` (scheduling, making the no-copy expert views resident),
        /// `gpuEndTime` to `waitUntilCompleted` returning.
        package var routedToKernel = 0.0, routedKernelToGPU = 0.0, routedAfterGPU = 0.0
        package var adviseCalls = 0
        /// Router preview (`previewTopN`): distinct experts of layers 1..<48 that the previous layer's preview named,
        /// all of them, and the preview's distinct experts.
        package var previewHit = 0, previewActual = 0, previewNamed = 0
        package var previewAdvise = 0.0
    }
    package private(set) var lastProfile = StepProfile()
    /// Checks only: commit the pre-router work in several buffers so `sections` can
    /// attribute its GPU time (ple, hc_attn, mixer, combine + hc_ffn, shared + router).
    package var splitPreRouter = false
    /// Selected experts are `F_RDADVISE`d before the routed buffer, so the misses read in
    /// parallel instead of faulting one by one inside the buffer. On the 53-token prompt
    /// that took the median decode token from 240-325 ms to 160-170 ms; a residency set of
    /// recent experts on top did not help (`docs/qwen38/02` §3).
    /// `Q38_ADVISE=0` turns it off; `Q38_COUNT_MISS=1` counts non-resident bytes (costs ~15 ms/token).
    package var adviseExperts = ProcessInfo.processInfo.environment["Q38_ADVISE"] != "0"
    /// Batches of at least this many tokens read their selected experts with `pread` on `readThreads`
    /// threads before the routed buffer instead of `F_RDADVISE`: at 8K / chunk 4096 the routed wait fell
    /// from 18 s to ~0 for a 5.6 s read (`docs/qwen38/05-EXPERT-READ.md`). Decode keeps the advise
    /// (`Q38_PREAD_MIN_T`, 0 = never; `Q38_READ_THREADS`).
    package var preadMinTokens = Int(ProcessInfo.processInfo.environment["Q38_PREAD_MIN_T"] ?? "") ?? 32
    package var readThreads = Int(ProcessInfo.processInfo.environment["Q38_READ_THREADS"] ?? "") ?? 4
    package var countMisses = ProcessInfo.processInfo.environment["Q38_COUNT_MISS"] == "1"
    /// Batches of at least this many tokens run the routed experts as gather by expert, float32 dequant of
    /// each selected expert once, MPS sgemm and a weighted scatter (`routedGemm`) instead of the per-pair
    /// kernels: one layer at T = 4096 took 569 -> 243 ms, T = 1024 142 -> 134 ms, T = 512 71 -> 113 ms
    /// (`docs/qwen38/06`). `Q38_GEMM_MIN_T`, 0 = never; `Q38_GEMM_GROUP` experts dequantized per dispatch.
    package var gemmMinTokens = Int(ProcessInfo.processInfo.environment["Q38_GEMM_MIN_T"] ?? "") ?? 1024
    package var gemmGroup = Int(ProcessInfo.processInfo.environment["Q38_GEMM_GROUP"] ?? "") ?? 4
    /// In such a batch, experts with fewer pairs than this skip the dequant + sgemm (a fixed ~0.35 ms each)
    /// and run the per-pair kernels over their gathered rows (`docs/qwen38/08`). `Q38_GEMM_MIN_PAIRS`, 0 = none.
    package var gemmMinPairs = Int(ProcessInfo.processInfo.environment["Q38_GEMM_MIN_PAIRS"] ?? "") ?? 24
    /// Batches of at least `gdnMinTokens` run the GDN step in chunks of `gdnChunk` tokens (WY form,
    /// `Qwen38GDNChunk`) instead of the token-serial `q38_gdn_step`: one layer at T = 4096 took 170 -> 40 ms
    /// with chunk 32 (`docs/qwen38/07`). `Q38_GDN_CHUNK` (0 = never), `Q38_GDN_MIN_T`.
    package var gdnChunk = Int(ProcessInfo.processInfo.environment["Q38_GDN_CHUNK"] ?? "") ?? 32
    package var gdnMinTokens = Int(ProcessInfo.processInfo.environment["Q38_GDN_MIN_T"] ?? "") ?? 16
    private var gdnChunked: Qwen38GDNChunk?
    private var gemmRows, gemmGateUp, gemmWGateUp, gemmWDown, gemmOrder, gemmAt, gemmPosSlot, gemmOnes: MTLBuffer?
    /// Checks only: batches of at least 1024 tokens write `<prefix>-l<layer>.bin` (Int32 T, then T x topK
    /// Int32 experts and T x topK Float32 routing weights), the input of `--q2-gemm-bench` (`Q38_DUMP_ROUTE`).
    package var routeDumpPrefix = ProcessInfo.processInfo.environment["Q38_DUMP_ROUTE"]
    /// Unselected experts an advise run may bridge (`Q38_ADVISE_GAP`, default 0: adjacent only).
    package var adviseGap = Int(ProcessInfo.processInfo.environment["Q38_ADVISE_GAP"] ?? "") ?? 0
    /// From this fraction of a layer's experts selected, advise whole expert tensors (`Q38_ADVISE_WHOLE`, >1 = never).
    package var adviseWholeFraction = Double(ProcessInfo.processInfo.environment["Q38_ADVISE_WHOLE"] ?? "") ?? 2.0
    /// Keep the dense weights (every no-copy tensor view) in a residency set so expert reads
    /// cannot evict them (`Q38_DENSE_RESIDENT=1`). Off by default since `docs/qwen38/09`: with pread expert
    /// reads the prefill and the steady decode run at the same speed without it (only the first decode
    /// token after a prefill pages the dense back in, +1.5 s), and it kept 5.4 GB wired through the prefill.
    package var denseResident = ProcessInfo.processInfo.environment["Q38_DENSE_RESIDENT"] == "1"
    private var denseSet: MTLResidencySet?
    /// Tsugumi's decode shape (Gemma `RealForwardRunner`, Ornith `QwenForwardRunner`): the routed buffer is
    /// committed without a wait and the next layer's pre-router buffer queued behind it, so one wait per layer
    /// (on the router logits) absorbs the previous layer's routed work and the host encodes under its expert
    /// page-in (`Q38_PIPELINE=0`: wait on each routed buffer).
    package var pipeline = ProcessInfo.processInfo.environment["Q38_PIPELINE"] != "0"
    /// The shared expert leaves the pre-router buffer: its own buffer is committed as soon as the host has the
    /// top-10, so it runs on the GPU while the host advises and the expert pages come in (`Q38_SHARED_LATE=0`).
    package var sharedLate = ProcessInfo.processInfo.environment["Q38_SHARED_LATE"] != "0"
    /// Cross-layer router preview (Ornith `TF_QWEN_EXPERT_PREFETCH`, Gemma `mtp/29`): the pre-router buffer of layer L
    /// also runs layer L+1's router on layer L's FFN input, and the host takes each row's top `previewTopN`
    /// (`Q38_PREVIEW_N`, 0 = off). Exact routing is untouched; `previewAdvise` (`Q38_PREVIEW_ADVISE=1`) issues
    /// `F_RDADVISE` for the named experts right after layer L's routed commit, so their pages come in under
    /// layer L's routed work and layer L+1's pre-router. Pipeline, batches under 32 tokens only (`docs/qwen38/12`).
    package var previewTopN = Int(ProcessInfo.processInfo.environment["Q38_PREVIEW_N"] ?? "") ?? 0
    package var previewAdvise = ProcessInfo.processInfo.environment["Q38_PREVIEW_ADVISE"] == "1"
    private var denseSetCount = 0

    // MTP (blk.<nTrunk>, `docs/qwen38/10`).
    /// `forward` copies the trunk's final residual rows (T x hc x e, before the `output_hc` mix) to `hidden`:
    /// the MTP head's input, llama.cpp `t_h_nextn`.
    package var exportHidden = false
    private var hiddenOut: MTLBuffer?
    /// The MTP head's own residual rows after its last forward (the next chained draft's input).
    private var mtpHiddenOut: MTLBuffer?
    private var mtpCat: MTLBuffer?
    package private(set) var lastMTPProfile = StepProfile()

    // Speculative rollback (`docs/qwen38/10` §5-4).
    /// Set before a verify forward of 2 ..< `gdnMinTokens` tokens: it then keeps what `rollbackToFirst` needs
    /// (the GDN state after its first token, written by `q38_gdn_step`; the conv and PLE histories from before the
    /// batch; the first token's rows that enter them). KV, indexer keys and the MTP KV are by position and need nothing.
    package var snapshotFirst = false
    private var snapshotTaken = false
    private var linStateSnap: [Int: MTLBuffer] = [:]
    private var linHistBefore: [Int: MTLBuffer] = [:]
    private var linQkv0: [Int: MTLBuffer] = [:]
    private var pleHistBefore: [Float] = []
    private var plePrevBefore: [Int] = []
    private var firstToken = 0

    package init(gguf: URL, ple: URL, capacity: Int, maxBatch: Int = 1) throws {
        let file = try GGUFFile(url: gguf)
        let pleFile = try GGUFFile(url: ple)
        self.file = file
        self.pleFile = pleFile
        self.capacity = capacity
        self.maxBatch = max(maxBatch, 1)
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
        qwen38Lib = lib
        func pso(_ library: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw GGUFFile.Error.format("kernel \(name) missing")
            }
            return try device.makeComputePipelineState(function: fn)
        }
        psoRmsScale = try pso(lib, "q38_group_rms_scale")
        psoRmsApply = try pso(lib, "q38_rms_apply")
        psoUnary = try pso(lib, "q38_unary")
        psoMix = try pso(lib, "q38_hc_mix")
        psoCombine = try pso(lib, "q38_hc_combine")
        psoAddScaled = try pso(lib, "q38_add_scaled")
        psoSiluMul = try pso(lib, "q38_silu_mul")
        psoConv = try pso(lib, "q38_gdn_conv")
        psoConvHist = try pso(lib, "q38_gdn_conv_hist")
        psoQKNorm = try pso(lib, "q38_gdn_qk_norm")
        psoGates = try pso(lib, "q38_gdn_gates")
        psoStep = try pso(lib, "q38_gdn_step")
        psoNormGate = try pso(lib, "q38_gdn_norm_gate")
        psoAttnPrep = try pso(lib, "q38_attn_prep")
        psoAttnScore = try pso(lib, "q38_attn_score")
        psoAttnStat = try pso(lib, "q38_attn_stat")
        psoAttnWeight = try pso(lib, "q38_attn_weight")
        psoAttnMix = try pso(lib, "q38_attn_mix")
        psoAttnGatherQ = try pso(lib, "q38_attn_gather_q")
        psoAttnGatherKV = try pso(lib, "q38_attn_gather_kv")
        psoAttnScatter = try pso(lib, "q38_attn_scatter_out")
        psoIdxBlockKey = try pso(lib, "q38_idx_block_key")
        psoIdxQPrep = try pso(lib, "q38_idx_q_prep")
        psoIdxScore = try pso(lib, "q38_idx_score")
        psoIdxReluSum = try pso(lib, "q38_idx_relu_sum")
        psoIdxTopK = try pso(lib, "q38_idx_topk")
        psoIdxUnion = try pso(lib, "q38_idx_union")
        psoAttnMaskSel = try pso(lib, "q38_attn_mask_sel")
        psoAttnGatherKVList = try pso(lib, "q38_attn_gather_kv_list")
        psoPleGate = try pso(lib, "q38_ple_gate")
        psoPleGated = try pso(lib, "q38_ple_gated")
        psoPleConvAdd = try pso(lib, "q38_ple_conv_add")
        psoPleHist = try pso(lib, "q38_ple_hist")
        indexerTopK = try file.value("qwen4exp.attention.indexer.top_k").int ?? 2048
        let moeLib = try MetalContext.moduleLibrary(device: device, module: "moe_ggml")
        psoPhase1 = try pso(moeLib, "moe_iq2xxs_phase1_gate_up_act")
        psoPhase2 = try pso(moeLib, "moe_q2k_phase2_down_reduce")
        psoDeqGateUp = try pso(moeLib, "moe_iq2xxs_dequant_gate_up_f32")
        psoDeqDown = try pso(moeLib, "moe_q2k_dequant_down_f32")
        psoGatherPairs = try pso(moeLib, "moe_gather_pair_rows")
        psoSiluHalves = try pso(moeLib, "moe_silu_mul_halves")
        psoScatterWeighted = try pso(moeLib, "moe_scatter_weighted")
        psoDeqQ4KGateUp = try pso(moeLib, "moe_q4k_dequant_gate_up_f32")
        psoDeqMXFP4Down = try pso(moeLib, "moe_mxfp4_dequant_down_f32")

        func buf(_ count: Int) -> MTLBuffer {
            device.makeBuffer(length: max(count, 1) * 4, options: .storageModeShared)!
        }
        pleHist = buf((4 - 1) * pleNgram * hc * e)
        attnScores = buf(1); selection = buf(1)
        logits = buf(vocab)
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
        partOffsets = buf(3 * nExperts)

        plePrev = [Int](repeating: 248_044, count: 2)
        allocateBatch(rows: shrinkBatch ? min(smallBatchRows, self.maxBatch) : self.maxBatch)
    }

    /// (Re)allocates the batch scratch for `rows` rows and drops the grown scratch (attention, gemm, dense).
    /// The contents are per forward; the state (KV, indexer keys, GDN, `pleHist`) is elsewhere.
    private func allocateBatch(rows B: Int) {
        guard B != batchRows else { return }
        batchRows = B
        func buf(_ count: Int) -> MTLBuffer {
            device.makeBuffer(length: max(count, 1) * 4, options: .storageModeShared)!
        }
        let C = 2 * Hk * Dl + Hv * Dl
        R = buf(B * hc * e); xn = buf(B * hc * e); lo = buf(B * hcRank); gate = buf(B * hc * e)
        mixed = buf(B * e); inj = buf(B * hc); blk = buf(B * e); blkShared = buf(B * e); rmsScale = buf(B * hc)
        qkv = buf(B * C); conv = buf(B * C)
        z = buf(B * Hv * Dl); ga = buf(B * Hv); gb = buf(B * Hv); lo6144 = buf(B * Hv * Dl)
        qg = buf(B * 2 * H * D); q = buf(B * H * D); qgate = buf(B * H * D); ao = buf(B * H * D)
        iq = buf(B * idxHeads * idxD)
        routerLogits = buf(B * nExperts); previewLogits = buf(B * nExperts); shG = buf(B * F); shU = buf(B * F); shY = buf(B * e); shGate = buf(B)
        acts = buf(B * topK * downIn); routeW = buf(B * topK); pairSlot = buf(B * topK)
        pleEmb = buf(B * e); pleKey = buf(B * hc * e); pleValue = buf(B * e); pleKeyN = buf(B * hc * e)
        pleQuery = buf(B * hc * e); pleGateBuf = buf(B * hc); pleGated = buf(B * hc * e); pleNormed = buf(B * hc * e)
        attnMax = buf(B * H); attnSum = buf(B * H); nq = buf(B); useSel = buf(B); selThr = buf(B); selCut = buf(B)
        idxScores = buf(B * (capacity / 4 + 1))
        attnScores = buf(1); selection = buf(1)
        attnQG = nil; attnOG = nil; attnKG = nil; attnVG = nil
        idxDots = nil; attnSelScores = nil; selAny = nil; selList = nil
        gemmRows = nil; gemmGateUp = nil; gemmWGateUp = nil; gemmWDown = nil; gemmOrder = nil; gemmAt = nil
        gemmPosSlot = nil; gemmOnes = nil
        if logits.length > vocab * 4 { logits = buf(vocab) }
        dense.dropScratch()
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

    /// dispatchThreads over `n`, thread group sized from the pipeline.
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

    /// Grouped RMS norm over `groups` groups of `n`; gamma is `gammaLen` long and repeats.
    private func rms(_ cb: MTLCommandBuffer, x: MTLBuffer, gamma: String, out: MTLBuffer,
                     groups: Int, n: Int, gammaLen: Int) throws {
        let g = try view(gamma)
        precondition(groups <= rmsScale.length / 4)
        var p = (UInt32(groups), UInt32(n), eps, UInt32(gammaLen))
        let len = MemoryLayout.size(ofValue: p)
        lanes(cb, psoRmsScale, size(groups)) { enc in
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(rmsScale, offset: 0, index: 1)
            enc.setBytes(&p, length: len, index: 2)
        }
        run(cb, psoRmsApply, size(groups * n)) { enc in
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(g.buffer, offset: g.offset, index: 1)
            enc.setBuffer(out, offset: 0, index: 2)
            enc.setBytes(&p, length: len, index: 3)
            enc.setBuffer(rmsScale, offset: 0, index: 4)
        }
    }

    private func unary(_ cb: MTLCommandBuffer, _ x: MTLBuffer, n: Int, op: UInt32, inScale: Float, outScale: Float) {
        run(cb, psoUnary, size(n)) { enc in
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(x, offset: 0, index: 1)
            var p = (op, inScale, outScale)
            enc.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 2)
        }
    }

    private func hcMix(_ cb: MTLCommandBuffer, prefix: String, inject: Bool, T: Int) throws {
        try rms(cb, x: R, gamma: prefix + "_norm.weight", out: xn, groups: T * hc, n: e, gammaLen: hc * e)
        try gemv(cb, prefix + "_down.weight", x: xn, y: lo, tokens: T)
        unary(cb, lo, n: T * hcRank, op: 0, inScale: 1 / Float(hc), outScale: 1)
        try gemv(cb, prefix + "_up.weight", x: lo, y: gate, tokens: T)
        run(cb, psoMix, size(e, T)) { enc in
            enc.setBuffer(xn, offset: 0, index: 0)
            enc.setBuffer(gate, offset: 0, index: 1)
            enc.setBuffer(mixed, offset: 0, index: 2)
            var p = (UInt32(hc), UInt32(e))
            enc.setBytes(&p, length: 8, index: 3)
        }
        if inject {
            try gemv(cb, prefix + "_inject.weight", x: xn, y: inj, tokens: T)
            unary(cb, inj, n: T * hc, op: 1, inScale: 1 / Float(hc), outScale: 2)
        }
    }

    private func combine(_ cb: MTLCommandBuffer, block: MTLBuffer, T: Int) {
        run(cb, psoCombine, size(T * hc * e)) { enc in
            enc.setBuffer(R, offset: 0, index: 0)
            enc.setBuffer(block, offset: 0, index: 1)
            enc.setBuffer(inj, offset: 0, index: 2)
            var p = (UInt32(hc), UInt32(e))
            enc.setBytes(&p, length: 8, index: 3)
        }
    }

    // MARK: - Blocks

    /// `section` (checks only) commits the work so far and returns the buffer to continue in.
    private func linear(_ cb0: MTLCommandBuffer, il: Int, T: Int, section: (String) throws -> MTLCommandBuffer) throws {
        var cb = cb0
        let pre = "blk.\(il)."
        let C = 2 * Hk * Dl + Hv * Dl
        try gemv(cb, pre + "attn_qkv.weight", x: mixed, y: qkv, tokens: T)
        if snapshotTaken {
            let q0 = linQkv0[il] ?? device.makeBuffer(length: C * 4, options: .storageModeShared)!
            linQkv0[il] = q0
            let blit = cb.makeBlitCommandEncoder()!
            blit.copy(from: qkv, sourceOffset: 0, to: q0, destinationOffset: 0, size: C * 4)
            blit.endEncoding()
        }
        try gemv(cb, pre + "attn_gate.weight", x: mixed, y: z, tokens: T)
        try gemv(cb, pre + "ssm_beta.weight", x: mixed, y: gb, tokens: T)
        try gemv(cb, pre + "ssm_alpha.weight", x: mixed, y: ga, tokens: T)
        cb = try section("gdn_in")
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
        try run(cb, psoGates, size(Hv, T)) { enc in
            enc.setBuffer(ga, offset: 0, index: 0)
            enc.setBuffer(gb, offset: 0, index: 1)
            try? setF32(enc, pre + "ssm_a", index: 2)
            try? setF32(enc, pre + "ssm_dt.bias", index: 3)
            enc.setBytes(&gp, length: gpLen, index: 4)
        }
        if gdnChunk > 0 && T >= gdnMinTokens {
            if gdnChunked?.chunk != gdnChunk {
                gdnChunked = try Qwen38GDNChunk(device: device, library: qwen38Lib, keyHeads: Hk, valueHeads: Hv,
                                                headDim: Dl, chunk: gdnChunk)
            }
            gdnChunked!.encode(cb, conv: conv, a: ga, b: gb, state: state, out: lo6144, T: T)
        } else {
            var snapT = UInt32.max
            var snap = state
            if snapshotTaken {
                snap = linStateSnap[il] ?? device.makeBuffer(length: state.length, options: .storageModeShared)!
                linStateSnap[il] = snap
                snapT = 0
            }
            lanes(cb, psoStep, size(Dl, Hv)) { enc in
                enc.setBuffer(conv, offset: 0, index: 0)
                enc.setBuffer(ga, offset: 0, index: 1)
                enc.setBuffer(gb, offset: 0, index: 2)
                enc.setBuffer(state, offset: 0, index: 5)
                enc.setBuffer(lo6144, offset: 0, index: 6)
                enc.setBytes(&gp, length: gpLen, index: 7)
                enc.setBytes(&cV, length: 4, index: 8)
                enc.setBuffer(snap, offset: 0, index: 9)
                enc.setBytes(&snapT, length: 4, index: 10)
            }
        }
        cb = try section("gdn_step")
        let nw = try view(pre + "ssm_norm.weight")
        run(cb, psoNormGate, size(Hv, T)) { enc in
            enc.setBuffer(lo6144, offset: 0, index: 0)
            enc.setBuffer(z, offset: 0, index: 1)
            enc.setBuffer(nw.buffer, offset: nw.offset, index: 2)
            var ep = eps
            enc.setBytes(&gp, length: gpLen, index: 3)
            enc.setBytes(&ep, length: 4, index: 4)
        }
        try gemv(cb, pre + "ssm_out.weight", x: lo6144, y: blk, tokens: T)
    }

    /// Row stride `nCap` of the host-path scores and selection for queries that see up to `n` tokens.
    /// Set per call (the stride is read only within the call): the dense MTP layer sees the whole cache.
    private func ensureAttnScratch(_ n: Int) { nCap = n }

    /// `attnScores` of at least `floats` floats and `selection` of `selectionRows` rows of `nCap`, grown on
    /// demand: sized from maxBatch they held 400 MB at chunk 2048 (1.6 GB for a decode step after chunk
    /// 4096) though only the host lanes (T < 32) and `attentionSgemm` (T x 12 x n) read them (`docs/qwen38/09`).
    private func ensureAttnScores(floats: Int, selectionRows: Int = 0) {
        if attnScores.length < floats * 4 {
            attnScores = device.makeBuffer(length: floats * 4, options: .storageModeShared)!
        }
        if selection.length < selectionRows * nCap * 4 {
            selection = device.makeBuffer(length: selectionRows * nCap * 4, options: .storageModeShared)!
        }
    }

    /// `dense` (the MTP layer, as llama.cpp `graph_mtp`): no indexer, every query attends its whole prefix.
    private func attention(_ cb: inout MTLCommandBuffer, il: Int, pos0: Int, T: Int, dense: Bool = false) throws {
        let pre = "blk.\(il)."
        let kc = kCache[il] ?? device.makeBuffer(length: capacity * Hkv * D * 4, options: .storageModeShared)!
        let vc = vCache[il] ?? device.makeBuffer(length: capacity * Hkv * D * 4, options: .storageModeShared)!
        kCache[il] = kc
        vCache[il] = vc
        try gemv(cb, pre + "attn_q.weight", x: mixed, y: qg, tokens: T)
        try gemv(cb, pre + "attn_k.weight", x: mixed, y: kc, yOffset: pos0 * Hkv * D * 4, tokens: T)
        try gemv(cb, pre + "attn_v.weight", x: mixed, y: vc, yOffset: pos0 * Hkv * D * 4, tokens: T)
        if dense {
            try attentionHost(&cb, il: il, pos0: pos0, T: T, kc: kc, vc: vc, bk: kc, kBlocks: capacity)
            try gemv(cb, pre + "attn_output.weight", x: ao, y: blk, tokens: T)
            return
        }
        let ik = idxRawKeys[il] ?? device.makeBuffer(length: capacity * idxD * 4, options: .storageModeShared)!
        let bk = idxBlockKeys[il] ?? device.makeBuffer(length: (capacity / 4 + 1) * idxD * 4, options: .storageModeShared)!
        idxRawKeys[il] = ik
        idxBlockKeys[il] = bk

        // QSA indexer: raw keys for the batch; a block's key once its 4th token is in.
        try gemv(cb, pre + "indexer.q_proj.weight", x: mixed, y: iq, tokens: T)
        try gemv(cb, pre + "indexer.k_proj.weight", x: mixed, y: ik, yOffset: pos0 * idxD * 4, tokens: T)
        let gk = try view(pre + "indexer.k_norm.weight")
        let firstBlock = (max(pos0 - 3, 0) + 3) / 4       // first block whose 4th token is in the batch
        let endBlock = (pos0 + T) / 4
        if endBlock > firstBlock {
            var bp = (UInt32(idxHeads), UInt32(Hkv), UInt32(idxD), UInt32(nRot), UInt32(firstBlock), eps)
            run(cb, psoIdxBlockKey, size(endBlock - firstBlock)) { enc in
                enc.setBuffer(ik, offset: 0, index: 0)
                enc.setBuffer(gk.buffer, offset: gk.offset, index: 1)
                enc.setBuffer(ropeFreq, offset: 0, index: 2)
                enc.setBuffer(bk, offset: 0, index: 3)
                enc.setBytes(&bp, length: MemoryLayout.size(ofValue: bp), index: 4)
            }
        }
        let kBlocks = indexerTopK / 4
        let lastBlocks = (pos0 + T) / 4
        if attnMpsMinTokens > 0 && T >= attnMpsMinTokens && lastBlocks > kBlocks {
            if compareSelected {
                // Both paths on the same input: the host one first, the batch's k rows and indexer queries
                // restored in between (both are roped in place).
                cb.commit()
                cb.waitUntilCompleted()
                cb = queue.makeCommandBuffer()!
                let rowBytes = Hkv * D * 4
                let kRows = Data(bytes: kc.contents() + pos0 * rowBytes, count: T * rowBytes)
                let iqRows = Data(bytes: iq.contents(), count: T * idxHeads * idxD * 4)
                try attentionHost(&cb, il: il, pos0: pos0, T: T, kc: kc, vc: vc, bk: bk, kBlocks: kBlocks)
                cb.commit()
                cb.waitUntilCompleted()
                cb = queue.makeCommandBuffer()!
                let hostOut = Data(bytes: ao.contents(), count: T * H * D * 4)
                kRows.withUnsafeBytes { (kc.contents() + pos0 * rowBytes).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
                iqRows.withUnsafeBytes { iq.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
                try attentionSelected(&cb, il: il, pos0: pos0, T: T, kc: kc, vc: vc, bk: bk, kBlocks: kBlocks)
                cb.commit()
                cb.waitUntilCompleted()
                cb = queue.makeCommandBuffer()!
                let o = ao.contents().bindMemory(to: Float.self, capacity: T * H * D)
                var maxDiff: Float = 0, maxAbs: Float = 0
                hostOut.withUnsafeBytes { raw in
                    let h = raw.bindMemory(to: Float.self)
                    for i in 0..<(T * H * D) { maxDiff = max(maxDiff, abs(o[i] - h[i])); maxAbs = max(maxAbs, abs(h[i])) }
                }
                FileHandle.standardError.write(String(format: "attn compare il %d pos %d T %d: max |diff| %.3e, max |host| %.3e, rel %.2e\n",
                                                      il, pos0, T, maxDiff, maxAbs, maxDiff / maxAbs).data(using: .utf8)!)
            } else {
                try attentionSelected(&cb, il: il, pos0: pos0, T: T, kc: kc, vc: vc, bk: bk, kBlocks: kBlocks)
            }
        } else {
            try attentionHost(&cb, il: il, pos0: pos0, T: T, kc: kc, vc: vc, bk: bk, kBlocks: kBlocks)
        }
        try gemv(cb, pre + "attn_output.weight", x: ao, y: blk, tokens: T)
    }

    /// The host-selection attention (and `attentionSgemm` for large batches below the budget). Writes `ao`.
    private func attentionHost(_ cb: inout MTLCommandBuffer, il: Int, pos0: Int, T: Int,
                               kc: MTLBuffer, vc: MTLBuffer, bk: MTLBuffer, kBlocks: Int) throws {
        let lastBlocks = (pos0 + T) / 4
        let gq = try view("blk.\(il).indexer.q_norm.weight")
        ensureAttnScratch(min(capacity, kBlocks * 4 + 3))
        let nqp = nq.contents().bindMemory(to: UInt32.self, capacity: T)
        let usp = useSel.contents().bindMemory(to: UInt32.self, capacity: T)
        if lastBlocks > kBlocks {
            var qp = (UInt32(idxHeads), UInt32(Hkv), UInt32(idxD), UInt32(nRot), UInt32(pos0), eps)
            run(cb, psoIdxQPrep, size(idxHeads, T)) { enc in
                enc.setBuffer(iq, offset: 0, index: 0)
                enc.setBuffer(gq.buffer, offset: gq.offset, index: 1)
                enc.setBuffer(ropeFreq, offset: 0, index: 2)
                enc.setBytes(&qp, length: MemoryLayout.size(ofValue: qp), index: 3)
            }
            run(cb, psoIdxScore, size(lastBlocks, T)) { enc in
                enc.setBuffer(iq, offset: 0, index: 0)
                enc.setBuffer(bk, offset: 0, index: 1)
                enc.setBuffer(idxScores, offset: 0, index: 2)
                var heads = UInt32(idxHeads), d = UInt32(idxD), nb = UInt32(lastBlocks)
                enc.setBytes(&heads, length: 4, index: 3)
                enc.setBytes(&d, length: 4, index: 4)
                enc.setBytes(&nb, length: 4, index: 5)
            }
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            cb = queue.makeCommandBuffer()!
        }
        // Per query: top kBlocks by score (lower block index first on ties), tokens in
        // block order, then the tail of the incomplete block; below the budget, all of them.
        let sc = idxScores.contents().bindMemory(to: Float.self, capacity: T * max(lastBlocks, 1))
        if lastBlocks > kBlocks { ensureAttnScores(floats: 0, selectionRows: T) }
        let sp = selection.contents().bindMemory(to: UInt32.self, capacity: T * nCap)
        var nMax = 0
        for t in 0..<T {
            let pos = pos0 + t
            let nBlocks = (pos + 1) / 4
            if nBlocks > kBlocks {
                let row = sc + t * lastBlocks
                let order = (0..<nBlocks).sorted { row[$0] != row[$1] ? row[$0] > row[$1] : $0 < $1 }
                let taken = order.prefix(kBlocks).sorted()
                var n = 0
                let base = t * nCap
                for b in taken { for j in 0..<4 { sp[base + n] = UInt32(b * 4 + j); n += 1 } }
                for j in (nBlocks * 4)..<(pos + 1) { sp[base + n] = UInt32(j); n += 1 }  // the tail may be empty
                nqp[t] = UInt32(n)
                usp[t] = 1
            } else {
                nqp[t] = UInt32(pos + 1)
                usp[t] = 0
            }
            nMax = max(nMax, Int(nqp[t]))
        }

        try attnPrep(cb, il: il, pos0: pos0, T: T, kc: kc)
        var ap = (UInt32(H), UInt32(Hkv), UInt32(D), UInt32(nCap), UInt32(T))
        let apLen = MemoryLayout.size(ofValue: ap)
        if attnMpsMinTokens > 0 && T >= attnMpsMinTokens && lastBlocks <= kBlocks {
            try attentionSgemm(cb, kc: kc, vc: vc, n: nMax, T: T)
            return
        }
        ensureAttnScores(floats: T * H * nCap, selectionRows: T)
        lanes(cb, psoAttnScore, size(nMax, H, T)) { enc in
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
        run(cb, psoAttnWeight, size(nMax, T * H)) { enc in
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

    /// q[t][h] and the batch's k-cache rows normed and roped, gates split off.
    private func attnPrep(_ cb: MTLCommandBuffer, il: Int, pos0: Int, T: Int, kc: MTLBuffer) throws {
        let pre = "blk.\(il)."
        var p = (UInt32(H), UInt32(Hkv), UInt32(D), UInt32(nRot), UInt32(pos0), eps)
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
            enc.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 7)
        }
    }

    private func sgemm(_ key: [Int], rows: Int, cols: Int, interior: Int, transposeRight: Bool,
                       alpha: Double) -> MPSMatrixMultiplication {
        if let m = attnMuls[key] { return m }
        let m = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: transposeRight,
                                        resultRows: rows, resultColumns: cols, interiorColumns: interior,
                                        alpha: alpha, beta: 0)
        attnMuls[key] = m
        return m
    }

    private func ensureBuffer(_ b: inout MTLBuffer?, bytes: Int, shared: Bool = false) {
        if (b?.length ?? 0) < bytes {
            b = device.makeBuffer(length: max(bytes, 4), options: shared ? .storageModeShared : .storageModePrivate)
        }
    }

    /// Attention for a batch above the QSA budget with no host sort (block keys through the batch in `bk`):
    /// indexer scores as sgemm, the top `kBlocks` per query by `q38_idx_topk`, then for each sub-batch of
    /// `attnSelBatch` queries the sgemm attention over the union of their tokens, the tokens a query does
    /// not attend masked out before the softmax. Same selection rule as the host path. Writes `ao`.
    private func attentionSelected(_ cb: inout MTLCommandBuffer, il: Int, pos0: Int, T: Int,
                                   kc: MTLBuffer, vc: MTLBuffer, bk: MTLBuffer, kBlocks: Int) throws {
        let pre = "blk.\(il)."
        let nb = (pos0 + T) / 4
        let gq = try view(pre + "indexer.q_norm.weight")
        var qp = (UInt32(idxHeads), UInt32(Hkv), UInt32(idxD), UInt32(nRot), UInt32(pos0), eps)
        run(cb, psoIdxQPrep, size(idxHeads, T)) { enc in
            enc.setBuffer(iq, offset: 0, index: 0)
            enc.setBuffer(gq.buffer, offset: gq.offset, index: 1)
            enc.setBuffer(ropeFreq, offset: 0, index: 2)
            enc.setBytes(&qp, length: MemoryLayout.size(ofValue: qp), index: 3)
        }
        // Indexer scores, in query chunks that keep the per-head dots within 64 MB.
        let rowsPer = max(1, min(T, (16 << 20) / (idxHeads * nb)))
        ensureBuffer(&idxDots, bytes: rowsPer * idxHeads * nb * 4)
        let keys = MPSMatrix(buffer: bk, descriptor: MPSMatrixDescriptor(rows: nb, columns: idxD, rowBytes: idxD * 4, dataType: .float32))
        var r0 = 0
        while r0 < T {
            let R = min(rowsPer, T - r0)
            let mul = sgemm([2, R * idxHeads, nb], rows: R * idxHeads, cols: nb, interior: idxD, transposeRight: true, alpha: 1)
            mul.encode(commandBuffer: cb,
                       leftMatrix: MPSMatrix(buffer: iq, offset: r0 * idxHeads * idxD * 4,
                                             descriptor: MPSMatrixDescriptor(rows: R * idxHeads, columns: idxD, rowBytes: idxD * 4, dataType: .float32)),
                       rightMatrix: keys,
                       resultMatrix: MPSMatrix(buffer: idxDots!, descriptor: MPSMatrixDescriptor(rows: R * idxHeads, columns: nb, rowBytes: nb * 4, dataType: .float32)))
            run(cb, psoIdxReluSum, size(nb, R)) { enc in
                enc.setBuffer(idxDots, offset: 0, index: 0)
                enc.setBuffer(idxScores, offset: r0 * nb * 4, index: 1)
                var heads = UInt32(idxHeads), w = UInt32(nb)
                enc.setBytes(&heads, length: 4, index: 2)
                enc.setBytes(&w, length: 4, index: 3)
            }
            r0 += R
        }
        var tp = (UInt32(nb), UInt32(kBlocks), UInt32(pos0), UInt32(T))
        lanes(cb, psoIdxTopK, size(T)) { enc in
            enc.setBuffer(idxScores, offset: 0, index: 0)
            enc.setBuffer(selThr, offset: 0, index: 1)
            enc.setBuffer(selCut, offset: 0, index: 2)
            enc.setBytes(&tp, length: MemoryLayout.size(ofValue: tp), index: 3)
        }
        let SB = attnSelBatch
        let S = (T + SB - 1) / SB
        ensureBuffer(&selAny, bytes: S * nb, shared: true)
        for s in 0..<S {
            let r0 = s * SB, rows = min(SB, T - r0)
            var up = (UInt32(nb), UInt32(kBlocks), UInt32(pos0 + r0), UInt32(rows))
            run(cb, psoIdxUnion, size((pos0 + r0 + rows) / 4)) { enc in
                enc.setBuffer(idxScores, offset: r0 * nb * 4, index: 0)
                enc.setBuffer(selThr, offset: r0 * 4, index: 1)
                enc.setBuffer(selCut, offset: r0 * 4, index: 2)
                enc.setBuffer(selAny, offset: s * nb, index: 3)
                enc.setBytes(&up, length: MemoryLayout.size(ofValue: up), index: 4)
            }
        }
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw error }
        cb = queue.makeCommandBuffer()!

        // Each sub-batch's token list: the union's blocks before the first query's incomplete block, then
        // every token from there to the last query (the mask sorts them out), padded to 256 with ~0.
        let any = selAny!.contents().assumingMemoryBound(to: UInt8.self)
        var lists: [UInt32] = []
        var spans: [(offset: Int, count: Int)] = []
        for s in 0..<S {
            let r0 = s * SB, rows = min(SB, T - r0)
            let offset = lists.count
            let tail = (pos0 + r0 + 1) / 4
            for b in 0..<tail where any[s * nb + b] != 0 {
                for j in 0..<4 { lists.append(UInt32(b * 4 + j)) }
            }
            for j in (tail * 4)..<(pos0 + r0 + rows) { lists.append(UInt32(j)) }
            let count = (lists.count - offset + 255) / 256 * 256
            lists.append(contentsOf: repeatElement(UInt32.max, count: offset + count - lists.count))
            spans.append((offset, count))
        }
        ensureBuffer(&selList, bytes: lists.count * 4, shared: true)
        lists.withUnsafeBytes { selList!.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }

        try attnPrep(cb, il: il, pos0: pos0, T: T, kc: kc)
        let G = H / Hkv
        let uMax = spans.map(\.count).max()!
        ensureBuffer(&attnQG, bytes: SB * G * D * 4)
        ensureBuffer(&attnOG, bytes: SB * G * D * 4)
        ensureBuffer(&attnKG, bytes: uMax * D * 4)
        ensureBuffer(&attnVG, bytes: uMax * D * 4)
        ensureBuffer(&attnSelScores, bytes: SB * G * uMax * 4)
        let nqp = nq.contents().bindMemory(to: UInt32.self, capacity: T)
        for s in 0..<S {
            let r0 = s * SB, rows = min(SB, T - r0)
            let (lo, U) = spans[s]
            for t in r0..<(r0 + rows) { nqp[t] = UInt32(U) }
            var gp = (UInt32(H), UInt32(Hkv), UInt32(D), UInt32(U), UInt32(rows))  // gather / scatter
            var ap = (UInt32(G), UInt32(1), UInt32(D), UInt32(U), UInt32(rows))    // stat / weight: G rows per query
            var mp = (UInt32(G), UInt32(U), UInt32(nb), UInt32(pos0 + r0))
            let len = MemoryLayout.size(ofValue: ap)
            let qDesc = MPSMatrixDescriptor(rows: rows * G, columns: D, rowBytes: D * 4, dataType: .float32)
            let kvDesc = MPSMatrixDescriptor(rows: U, columns: D, rowBytes: D * 4, dataType: .float32)
            let scoresDesc = MPSMatrixDescriptor(rows: rows * G, columns: U, rowBytes: U * 4, dataType: .float32)
            let mulQK = sgemm([3, rows * G, U], rows: rows * G, cols: U, interior: D, transposeRight: true,
                              alpha: 1 / Double(D).squareRoot())
            let mulWV = sgemm([4, rows * G, U], rows: rows * G, cols: D, interior: U, transposeRight: false, alpha: 1)
            for g in 0..<Hkv {
                var gV = UInt32(g)
                run(cb, psoAttnGatherQ, size(rows * G * D)) { enc in
                    enc.setBuffer(q, offset: r0 * H * D * 4, index: 0)
                    enc.setBuffer(attnQG, offset: 0, index: 1)
                    enc.setBytes(&gp, length: len, index: 2)
                    enc.setBytes(&gV, length: 4, index: 3)
                }
                for (cache, out) in [(kc, attnKG), (vc, attnVG)] {
                    run(cb, psoAttnGatherKVList, size(U * D)) { enc in
                        enc.setBuffer(cache, offset: 0, index: 0)
                        enc.setBuffer(out, offset: 0, index: 1)
                        enc.setBuffer(selList, offset: lo * 4, index: 2)
                        enc.setBytes(&gp, length: len, index: 3)
                        enc.setBytes(&gV, length: 4, index: 4)
                    }
                }
                mulQK.encode(commandBuffer: cb, leftMatrix: MPSMatrix(buffer: attnQG!, descriptor: qDesc),
                             rightMatrix: MPSMatrix(buffer: attnKG!, descriptor: kvDesc),
                             resultMatrix: MPSMatrix(buffer: attnSelScores!, descriptor: scoresDesc))
                run(cb, psoAttnMaskSel, size(U, rows * G)) { enc in
                    enc.setBuffer(attnSelScores, offset: 0, index: 0)
                    enc.setBuffer(selList, offset: lo * 4, index: 1)
                    enc.setBuffer(idxScores, offset: r0 * nb * 4, index: 2)
                    enc.setBuffer(selThr, offset: r0 * 4, index: 3)
                    enc.setBuffer(selCut, offset: r0 * 4, index: 4)
                    enc.setBytes(&mp, length: MemoryLayout.size(ofValue: mp), index: 5)
                }
                for op in [UInt32(0), 1] {
                    lanes(cb, psoAttnStat, size(G, rows)) { enc in
                        enc.setBuffer(attnSelScores, offset: 0, index: 0)
                        enc.setBuffer(attnMax, offset: 0, index: 1)
                        enc.setBuffer(attnSum, offset: 0, index: 2)
                        enc.setBytes(&ap, length: len, index: 3)
                        var o = op
                        enc.setBytes(&o, length: 4, index: 4)
                        enc.setBuffer(nq, offset: r0 * 4, index: 5)
                    }
                }
                run(cb, psoAttnWeight, size(U, rows * G)) { enc in
                    enc.setBuffer(attnSelScores, offset: 0, index: 0)
                    enc.setBuffer(attnMax, offset: 0, index: 1)
                    enc.setBuffer(attnSum, offset: 0, index: 2)
                    enc.setBytes(&ap, length: len, index: 3)
                    enc.setBuffer(nq, offset: r0 * 4, index: 4)
                }
                mulWV.encode(commandBuffer: cb, leftMatrix: MPSMatrix(buffer: attnSelScores!, descriptor: scoresDesc),
                             rightMatrix: MPSMatrix(buffer: attnVG!, descriptor: kvDesc),
                             resultMatrix: MPSMatrix(buffer: attnOG!, descriptor: qDesc))
                run(cb, psoAttnScatter, size(rows * G * D)) { enc in
                    enc.setBuffer(attnOG, offset: 0, index: 0)
                    enc.setBuffer(qgate, offset: r0 * H * D * 4, index: 1)
                    enc.setBuffer(ao, offset: r0 * H * D * 4, index: 2)
                    enc.setBytes(&gp, length: len, index: 3)
                    enc.setBytes(&gV, length: 4, index: 4)
                }
            }
        }
    }

    /// Attention for queries that all see the causal prefix (`nq[t]` of the first `n` tokens),
    /// one KV group at a time as two sgemms; writes `ao`.
    private func attentionSgemm(_ cb: MTLCommandBuffer, kc: MTLBuffer, vc: MTLBuffer, n: Int, T: Int) throws {
        let G = H / Hkv
        func ensure(_ b: inout MTLBuffer?, _ floats: Int) {
            if (b?.length ?? 0) < floats * 4 { b = device.makeBuffer(length: floats * 4, options: .storageModePrivate) }
        }
        ensure(&attnQG, maxBatch * G * D)
        ensure(&attnOG, maxBatch * G * D)
        ensure(&attnKG, n * D)
        ensure(&attnVG, n * D)
        ensureAttnScores(floats: T * G * n)   // row width n here
        var ap = (UInt32(G), UInt32(1), UInt32(D), UInt32(n), UInt32(T))   // stat / weight: G rows per token
        var gp = (UInt32(H), UInt32(Hkv), UInt32(D), UInt32(n), UInt32(T)) // gather / scatter
        let len = MemoryLayout.size(ofValue: ap)
        let scoresDesc = MPSMatrixDescriptor(rows: T * G, columns: n, rowBytes: n * 4, dataType: .float32)
        let qDesc = MPSMatrixDescriptor(rows: T * G, columns: D, rowBytes: D * 4, dataType: .float32)
        let kvDesc = MPSMatrixDescriptor(rows: n, columns: D, rowBytes: D * 4, dataType: .float32)
        let mulQK = attnMuls[[0, T, n]] ?? MPSMatrixMultiplication(
            device: device, transposeLeft: false, transposeRight: true,
            resultRows: T * G, resultColumns: n, interiorColumns: D, alpha: 1 / Double(D).squareRoot(), beta: 0)
        let mulWV = attnMuls[[1, T, n]] ?? MPSMatrixMultiplication(
            device: device, transposeLeft: false, transposeRight: false,
            resultRows: T * G, resultColumns: D, interiorColumns: n, alpha: 1, beta: 0)
        attnMuls[[0, T, n]] = mulQK
        attnMuls[[1, T, n]] = mulWV
        for g in 0..<Hkv {
            var gV = UInt32(g)
            run(cb, psoAttnGatherQ, size(T * G * D)) { enc in
                enc.setBuffer(q, offset: 0, index: 0)
                enc.setBuffer(attnQG, offset: 0, index: 1)
                enc.setBytes(&gp, length: len, index: 2)
                enc.setBytes(&gV, length: 4, index: 3)
            }
            for (cache, out) in [(kc, attnKG), (vc, attnVG)] {
                run(cb, psoAttnGatherKV, size(n * D)) { enc in
                    enc.setBuffer(cache, offset: 0, index: 0)
                    enc.setBuffer(out, offset: 0, index: 1)
                    enc.setBytes(&gp, length: len, index: 2)
                    enc.setBytes(&gV, length: 4, index: 3)
                }
            }
            mulQK.encode(commandBuffer: cb, leftMatrix: MPSMatrix(buffer: attnQG!, descriptor: qDesc),
                         rightMatrix: MPSMatrix(buffer: attnKG!, descriptor: kvDesc),
                         resultMatrix: MPSMatrix(buffer: attnScores, descriptor: scoresDesc))
            for op in [UInt32(0), 1] {
                lanes(cb, psoAttnStat, size(G, T)) { enc in
                    enc.setBuffer(attnScores, offset: 0, index: 0)
                    enc.setBuffer(attnMax, offset: 0, index: 1)
                    enc.setBuffer(attnSum, offset: 0, index: 2)
                    enc.setBytes(&ap, length: len, index: 3)
                    var o = op
                    enc.setBytes(&o, length: 4, index: 4)
                    enc.setBuffer(nq, offset: 0, index: 5)
                }
            }
            run(cb, psoAttnWeight, size(n, T * G)) { enc in
                enc.setBuffer(attnScores, offset: 0, index: 0)
                enc.setBuffer(attnMax, offset: 0, index: 1)
                enc.setBuffer(attnSum, offset: 0, index: 2)
                enc.setBytes(&ap, length: len, index: 3)
                enc.setBuffer(nq, offset: 0, index: 4)
            }
            mulWV.encode(commandBuffer: cb, leftMatrix: MPSMatrix(buffer: attnScores, descriptor: scoresDesc),
                         rightMatrix: MPSMatrix(buffer: attnVG!, descriptor: kvDesc),
                         resultMatrix: MPSMatrix(buffer: attnOG!, descriptor: qDesc))
            run(cb, psoAttnScatter, size(T * G * D)) { enc in
                enc.setBuffer(attnOG, offset: 0, index: 0)
                enc.setBuffer(qgate, offset: 0, index: 1)
                enc.setBuffer(ao, offset: 0, index: 2)
                enc.setBytes(&gp, length: len, index: 3)
                enc.setBytes(&gV, length: 4, index: 4)
            }
        }
    }

    /// Routed + shared experts for the T mixed inputs already in `mixed`; result in `blk`.
    /// `sharedCommitted`: the shared expert is not in the pre-router buffer; `commitShared` queues it (into
    /// `blkShared`) right after the top-10, ahead of the views and the advise.
    private func moeRouted(il: Int, T: Int, prof: inout StepProfile, sharedCommitted: Bool = false,
                           previewed: Set<Int>? = nil) throws -> MTLCommandBuffer {
        let pre = "blk.\(il)."
        let tStart = CFAbsoluteTimeGetCurrent()
        // Host top-10 per token (ds4 reference: softmax over all, lowest index wins ties, renormalize).
        let lg = routerLogits.contents().bindMemory(to: Float.self, capacity: T * nExperts)
        let w = routeW.contents().bindMemory(to: Float.self, capacity: T * topK)
        let ps = pairSlot.contents().bindMemory(to: UInt32.self, capacity: T * topK)
        var slotOf = [Int](repeating: -1, count: nExperts)
        var slots: [Int] = []
        var prob = [Double](repeating: 0, count: nExperts)
        var chosenP = [Double](repeating: 0, count: topK)
        for t in 0..<T {
            let row = lg + t * nExperts
            var mx = -Double.greatestFiniteMagnitude
            for i in 0..<nExperts { mx = max(mx, Double(row[i])) }
            for i in 0..<nExperts { prob[i] = exp(Double(row[i]) - mx) }
            var sel = [Int](repeating: 0, count: topK)
            var wsum = 0.0
            for k in 0..<topK {
                var best = -1
                var bestP = -1.0
                for i in 0..<nExperts where prob[i] > bestP {   // taken experts are marked -1
                    best = i
                    bestP = prob[i]
                }
                sel[k] = best
                wsum += bestP
                chosenP[k] = bestP
                prob[best] = -1
            }
            for (k, ex) in sel.enumerated() {
                if slotOf[ex] < 0 { slotOf[ex] = slots.count; slots.append(ex) }
                ps[t * topK + k] = UInt32(slotOf[ex])
                w[t * topK + k] = Float(chosenP[k] / wsum)
            }
        }
        // The MTP layer's Q4_K / MXFP4 experts have only the dequant + sgemm route.
        let mtpLayer = il >= nTrunk
        let gemm = mtpLayer || (gemmMinTokens > 0 && T >= gemmMinTokens)
        var gemmBigSlots = slots.count
        if gemm && !mtpLayer && gemmMinPairs > 0 {
            // Experts with at least `gemmMinPairs` pairs take the first slots (so `routedGemm` dequantizes
            // contiguous slot groups), the rest the tail; relative order kept.
            var count = [Int](repeating: 0, count: slots.count)
            for p in 0..<(T * topK) { count[Int(ps[p])] += 1 }
            let big = slots.indices.filter { count[$0] >= gemmMinPairs }
            let small = slots.indices.filter { count[$0] < gemmMinPairs }
            var renum = [UInt32](repeating: 0, count: slots.count)
            for (i, s) in (big + small).enumerated() { renum[s] = UInt32(i) }
            slots = (big + small).map { slots[$0] }
            for p in 0..<(T * topK) { ps[p] = renum[Int(ps[p])] }
            gemmBigSlots = big.count
        }
        prof.distinctExperts += slots.count
        if let prefix = routeDumpPrefix, T >= 1024 {
            var data = Data()
            withUnsafeBytes(of: Int32(T)) { data.append(contentsOf: $0) }
            let experts = (0..<(T * topK)).map { Int32(slots[Int(ps[$0])]) }
            experts.withUnsafeBytes { data.append(contentsOf: $0) }
            data.append(UnsafeBufferPointer(start: w, count: T * topK))
            try data.write(to: URL(fileURLWithPath: "\(prefix)-l\(il).bin"))
        }
        if sharedCommitted { try commitShared(pre: pre, T: T) }
        if let named = previewed {
            prof.previewActual += slots.count
            prof.previewNamed += named.count
            prof.previewHit += slots.reduce(0) { $0 + (named.contains($1) ? 1 : 0) }
        }
        let tTopK = CFAbsoluteTimeGetCurrent()
        prof.routeTopK += tTopK - tStart

        let gateT = try file.tensor(pre + "ffn_gate_exps.weight")
        let upT = try file.tensor(pre + "ffn_up_exps.weight")
        let downT = try file.tensor(pre + "ffn_down_exps.weight")
        let gateBytes = gateT.bytesPerRow * F
        let downBytes = downT.bytesPerRow * e
        let po = partOffsets.contents().bindMemory(to: UInt32.self, capacity: 3 * nExperts)
        var used: [[MTLBuffer]] = [[], [], []]
        let parts: [(GGUFFile.Tensor, Int)] = [(gateT, gateBytes), (upT, gateBytes), (downT, downBytes)]
        for (slot, ex) in slots.enumerated() {
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
                    prof.newViewBytes += made.buffer.length
                }
                routedArgEncoder.setBuffer(v.buffer, offset: 0, index: part * nExperts + slot)
                po[3 * slot + part] = UInt32(v.offset)
                used[part].append(v.buffer)
            }
        }
        let tViews = CFAbsoluteTimeGetCurrent()
        // One advise per run of experts no more than `adviseGap` apart in the tensor: a
        // 128-token batch picks ~40 % of a layer's experts, and 28K separate calls cost 1.5 s.
        let sorted = slots.sorted()
        if adviseExperts && Double(slots.count) >= adviseWholeFraction * Double(nExperts) {
            // Most of the layer is needed: one read-ahead per tensor.
            for (t, _) in parts { file.adviseRead(offset: t.offset, byteCount: t.byteCount) }
            prof.adviseCalls += parts.count
        } else {
        var runs: [(offset: Int, byteCount: Int)] = []
        let preadThisBatch = adviseExperts && preadMinTokens > 0 && T >= preadMinTokens
        for (t, bytes) in parts {
            var i = 0
            while i < sorted.count {
                var j = i
                while j + 1 < sorted.count && sorted[j + 1] - sorted[j] <= adviseGap + 1 { j += 1 }
                let fileOffset = t.offset + sorted[i] * bytes
                let length = (sorted[j] - sorted[i] + 1) * bytes
                if countMisses {
                    let tm = CFAbsoluteTimeGetCurrent()
                    for ex in sorted[i]...sorted[j] where slotOf[ex] >= 0 {
                        prof.missBytes += file.nonResidentBytes(offset: t.offset + ex * bytes, byteCount: bytes)
                    }
                    prof.missTime += CFAbsoluteTimeGetCurrent() - tm
                }
                if adviseExperts && !preadThisBatch { file.adviseRead(offset: fileOffset, byteCount: length) }
                runs.append((fileOffset, length))
                prof.adviseCalls += 1
                i = j + 1
            }
        }
        if preadThisBatch { file.preadRanges(runs, threads: readThreads) }
        }
        let tAdvise = CFAbsoluteTimeGetCurrent()
        prof.routeViews += tViews - tTopK
        prof.routeAdvise += tAdvise - tViews

        let cb2 = queue.makeCommandBuffer()!
        // Shared expert (already computed into shY / shGate by the pre-router buffer).
        if !sharedCommitted { encodeSharedSum(cb2, T: T) }
        if gemm {
            routedGemm(cb2, T: T, slots: slots.count, bigSlots: gemmBigSlots,
                       rowBytes: (UInt32(gateT.bytesPerRow), UInt32(downT.bytesPerRow)), pairSlot: ps, used: used,
                       dequant: mtpLayer ? (psoDeqQ4KGateUp, psoDeqMXFP4Down) : (psoDeqGateUp, psoDeqDown),
                       downStride: downT.rowWidth)
            return cb2
        }
        var offsets = (UInt32(gateT.bytesPerRow), UInt32(downT.bytesPerRow))
        var dV = UInt32(e), fV = UInt32(F), kV = UInt32(topK), strideV = UInt32(downIn), tV = UInt32(T)
        do {
            let enc = cb2.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoPhase1)
            enc.setBuffer(routedArg, offset: 0, index: 0)
            enc.useResources(used[0] + used[1], usage: .read)
            enc.setBytes(&offsets, length: 8, index: 1)
            enc.setBuffer(mixed, offset: 0, index: 2)
            enc.setBuffer(acts, offset: 0, index: 3)
            enc.setBytes(&dV, length: 4, index: 4)
            enc.setBytes(&fV, length: 4, index: 5)
            enc.setBytes(&kV, length: 4, index: 6)
            enc.setBytes(&strideV, length: 4, index: 7)
            enc.setBuffer(partOffsets, offset: 0, index: 8)
            enc.setBytes(&tV, length: 4, index: 9)
            enc.setBuffer(pairSlot, offset: 0, index: 10)
            enc.setThreadgroupMemoryLength(256 * 8 + 128, index: 0)
            enc.dispatchThreadgroups(size((T * topK * F + 7) / 8), threadsPerThreadgroup: size(64))
            enc.endEncoding()
        }
        do {
            let enc = cb2.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoPhase2)
            enc.setBuffer(routedArg, offset: 0, index: 0)
            enc.useResources(used[2], usage: .read)
            enc.setBytes(&offsets, length: 8, index: 1)
            enc.setBuffer(acts, offset: 0, index: 2)
            enc.setBuffer(routeW, offset: 0, index: 3)
            enc.setBuffer(blkShared, offset: 0, index: 4)
            enc.setBuffer(blk, offset: 0, index: 5)
            enc.setBytes(&dV, length: 4, index: 6)
            enc.setBytes(&strideV, length: 4, index: 7)
            enc.setBytes(&kV, length: 4, index: 8)
            enc.setBuffer(partOffsets, offset: 0, index: 9)
            enc.setBytes(&tV, length: 4, index: 10)
            enc.setBuffer(pairSlot, offset: 0, index: 11)
            enc.dispatchThreadgroups(size((T * e + 7) / 8), threadsPerThreadgroup: size(64))
            enc.endEncoding()
        }
        return cb2
    }

    /// Routed experts for T >= `gemmMinTokens` into `blk` (residual `blkShared`): the pairs' rows gathered
    /// by expert into `gemmRows`, then for every `gemmGroup` experts the IQ2_XXS gate/up and Q2_K down rows
    /// dequantized to float32, [n x 2F] = X W_gu^T by sgemm into `gemmGateUp` (sized to the largest group),
    /// SiLU(gate) * up in place, [n x D] = A W_down^T
    /// by sgemm written back over the same expert's `gemmRows` (already consumed), and the weighted scatter.
    /// Reads the experts through the same per-slot views as the per-pair kernels (`routedArg`, `partOffsets`,
    /// `used` by part), so only the selected experts' pages are made resident.
    /// Slots `bigSlots..<S` (experts with few pairs, laid out last) skip the dequant: the per-pair kernels
    /// run over their rows with top_k 1 and write the unweighted output back in place.
    private func routedGemm(_ cb: MTLCommandBuffer, T: Int, slots S: Int, bigSlots SB: Int, rowBytes: (UInt32, UInt32),
                            pairSlot ps: UnsafeMutablePointer<UInt32>, used: [[MTLBuffer]],
                            dequant: (gateUp: MTLComputePipelineState, down: MTLComputePipelineState), downStride: Int) {
        let P = T * topK, G = max(gemmGroup, 1)
        var count = [Int](repeating: 0, count: S)
        for p in 0..<P { count[Int(ps[p])] += 1 }
        var start = [Int](repeating: 0, count: S + 1)
        for s in 0..<S { start[s + 1] = start[s] + count[s] }
        let pairs = batchRows * topK
        ensureBuffer(&gemmOrder, bytes: pairs * 4, shared: true)
        ensureBuffer(&gemmAt, bytes: pairs * 4, shared: true)
        ensureBuffer(&gemmRows, bytes: pairs * e * 4)
        var groupPairs = 0
        for g0 in stride(from: 0, to: SB, by: G) { groupPairs = max(groupPairs, start[min(g0 + G, SB)] - start[g0]) }
        ensureBuffer(&gemmGateUp, bytes: max(groupPairs, 1) * 2 * F * 4)
        ensureBuffer(&gemmWGateUp, bytes: G * 2 * F * e * 4)
        ensureBuffer(&gemmWDown, bytes: G * e * F * 4)
        let order = gemmOrder!.contents().bindMemory(to: UInt32.self, capacity: P)
        let at = gemmAt!.contents().bindMemory(to: UInt32.self, capacity: P)
        var fill = start
        for p in 0..<P {
            let s = Int(ps[p])
            order[fill[s]] = UInt32(p)
            at[p] = UInt32(fill[s])
            fill[s] += 1
        }
        let rows = gemmRows!, gu = gemmGateUp!, wGU = gemmWGateUp!, wDown = gemmWDown!
        var dV = UInt32(e), fV = UInt32(F), kV = UInt32(topK), strideV = UInt32(downStride)
        func matrix(_ b: MTLBuffer, _ offset: Int, _ r: Int, _ c: Int, rowBytes: Int? = nil) -> MPSMatrix {
            MPSMatrix(buffer: b, offset: offset,
                      descriptor: MPSMatrixDescriptor(rows: r, columns: c, rowBytes: rowBytes ?? c * 4, dataType: .float32))
        }

        run(cb, psoGatherPairs, size((e + 31) / 32, P)) { enc in
            enc.setBuffer(mixed, offset: 0, index: 0)
            enc.setBuffer(gemmOrder, offset: 0, index: 1)
            enc.setBuffer(rows, offset: 0, index: 2)
            enc.setBytes(&dV, length: 4, index: 3)
            enc.setBytes(&kV, length: 4, index: 4)
        }
        for g0 in stride(from: 0, to: SB, by: G) {
            // The per-expert MPSMatrix objects are autoreleased; without a pool they pile up for the whole
            // forward (+1.3 GB footprint at 8K, docs/qwen38/06 §3).
            autoreleasepool {
                let n = min(G, SB - g0)
                var firstSlot = UInt32(g0)
                var enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(dequant.gateUp)
                enc.setBuffer(routedArg, offset: 0, index: 0)
                enc.useResources(Array(used[0][g0..<(g0 + n)] + used[1][g0..<(g0 + n)]), usage: .read)
                enc.setBuffer(partOffsets, offset: 0, index: 1)
                enc.setBuffer(wGU, offset: 0, index: 2)
                enc.setBytes(&dV, length: 4, index: 3)
                enc.setBytes(&fV, length: 4, index: 4)
                enc.setBytes(&firstSlot, length: 4, index: 5)
                enc.dispatchThreads(size(e / 32, 2 * F, n), threadsPerThreadgroup: size(8, 32))
                enc.endEncoding()
                enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(dequant.down)
                enc.setBuffer(routedArg, offset: 0, index: 0)
                enc.useResources(Array(used[2][g0..<(g0 + n)]), usage: .read)
                enc.setBuffer(partOffsets, offset: 0, index: 1)
                enc.setBuffer(wDown, offset: 0, index: 2)
                enc.setBytes(&dV, length: 4, index: 3)
                enc.setBytes(&strideV, length: 4, index: 4)
                enc.setBytes(&fV, length: 4, index: 5)
                enc.setBytes(&firstSlot, length: 4, index: 6)
                enc.dispatchThreads(size(F / 16, e, n), threadsPerThreadgroup: size(4, 64))
                enc.endEncoding()
                for s in g0..<(g0 + n) {
                    sgemm([10, count[s]], rows: count[s], cols: 2 * F, interior: e, transposeRight: true, alpha: 1)
                        .encode(commandBuffer: cb,
                                leftMatrix: matrix(rows, start[s] * e * 4, count[s], e),
                                rightMatrix: matrix(wGU, (s - g0) * 2 * F * e * 4, 2 * F, e),
                                resultMatrix: matrix(gu, (start[s] - start[g0]) * 2 * F * 4, count[s], 2 * F))
                }
                run(cb, psoSiluHalves, size(F, start[g0 + n] - start[g0])) { enc in
                    var first = UInt32(0)
                    enc.setBuffer(gu, offset: 0, index: 0)
                    enc.setBytes(&fV, length: 4, index: 1)
                    enc.setBytes(&first, length: 4, index: 2)
                }
                for s in g0..<(g0 + n) {
                    sgemm([11, count[s]], rows: count[s], cols: e, interior: F, transposeRight: true, alpha: 1)
                        .encode(commandBuffer: cb,
                                leftMatrix: matrix(gu, (start[s] - start[g0]) * 2 * F * 4, count[s], F, rowBytes: 2 * F * 4),
                                rightMatrix: matrix(wDown, (s - g0) * e * F * 4, e, F),
                                resultMatrix: matrix(rows, start[s] * e * 4, count[s], e))
                }
            }
        }
        let p0 = start[SB], nSmall = P - p0
        if nSmall > 0 {
            ensureBuffer(&gemmPosSlot, bytes: pairs * 4, shared: true)
            if gemmOnes == nil {
                gemmOnes = device.makeBuffer(length: pairs * 4, options: .storageModeShared)
                let one = gemmOnes!.contents().bindMemory(to: Float.self, capacity: pairs)
                for i in 0..<pairs { one[i] = 1 }
            }
            let posSlot = gemmPosSlot!.contents().bindMemory(to: UInt32.self, capacity: nSmall)
            for i in 0..<nSmall { posSlot[i] = ps[Int(order[p0 + i])] }
            var offsets = rowBytes
            var oneV = UInt32(1), nV = UInt32(nSmall)
            var enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoPhase1)
            enc.setBuffer(routedArg, offset: 0, index: 0)
            enc.useResources(Array(used[0][SB..<S] + used[1][SB..<S]), usage: .read)
            enc.setBytes(&offsets, length: 8, index: 1)
            enc.setBuffer(rows, offset: p0 * e * 4, index: 2)
            enc.setBuffer(acts, offset: 0, index: 3)
            enc.setBytes(&dV, length: 4, index: 4)
            enc.setBytes(&fV, length: 4, index: 5)
            enc.setBytes(&oneV, length: 4, index: 6)
            enc.setBytes(&strideV, length: 4, index: 7)
            enc.setBuffer(partOffsets, offset: 0, index: 8)
            enc.setBytes(&nV, length: 4, index: 9)
            enc.setBuffer(gemmPosSlot, offset: 0, index: 10)
            enc.setThreadgroupMemoryLength(256 * 8 + 128, index: 0)
            enc.dispatchThreadgroups(size((nSmall * F + 7) / 8), threadsPerThreadgroup: size(64))
            enc.endEncoding()
            // phase 2 adds its residual: the region (already read) is cleared and used as both.
            let blit = cb.makeBlitCommandEncoder()!
            blit.fill(buffer: rows, range: (p0 * e * 4)..<(P * e * 4), value: 0)
            blit.endEncoding()
            enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(psoPhase2)
            enc.setBuffer(routedArg, offset: 0, index: 0)
            enc.useResources(Array(used[2][SB..<S]), usage: .read)
            enc.setBytes(&offsets, length: 8, index: 1)
            enc.setBuffer(acts, offset: 0, index: 2)
            enc.setBuffer(gemmOnes, offset: 0, index: 3)
            enc.setBuffer(rows, offset: p0 * e * 4, index: 4)
            enc.setBuffer(rows, offset: p0 * e * 4, index: 5)
            enc.setBytes(&dV, length: 4, index: 6)
            enc.setBytes(&strideV, length: 4, index: 7)
            enc.setBytes(&oneV, length: 4, index: 8)
            enc.setBuffer(partOffsets, offset: 0, index: 9)
            enc.setBytes(&nV, length: 4, index: 10)
            enc.setBuffer(gemmPosSlot, offset: 0, index: 11)
            enc.dispatchThreadgroups(size((nSmall * e + 7) / 8), threadsPerThreadgroup: size(64))
            enc.endEncoding()
        }
        run(cb, psoScatterWeighted, size((e + 31) / 32, T)) { enc in
            enc.setBuffer(rows, offset: 0, index: 0)
            enc.setBuffer(gemmAt, offset: 0, index: 1)
            enc.setBuffer(routeW, offset: 0, index: 2)
            enc.setBuffer(blkShared, offset: 0, index: 3)
            enc.setBuffer(blk, offset: 0, index: 4)
            enc.setBytes(&dV, length: 4, index: 5)
            enc.setBytes(&kV, length: 4, index: 6)
        }
    }

    // MARK: - Memory


    /// Bytes held by the runner, by kind (checks only, `Q38_MEM_LOG`): Metal's allocated total, the fixed
    /// per-batch buffers, the grown scratch, caches and the no-copy views (dense in the residency set, experts).
    package func memoryReport() -> [(String, Int)] {
        func len(_ b: MTLBuffer?) -> Int { b?.length ?? 0 }
        func lens(_ bs: [MTLBuffer?]) -> Int { bs.reduce(0) { $0 + len($1) } }
        var seen = Set<ObjectIdentifier>()
        var denseViews = 0
        for v in views.values where seen.insert(ObjectIdentifier(v.buffer)).inserted { denseViews += v.buffer.length }
        seen.removeAll()
        var experts = 0
        for v in expertParts.values where seen.insert(ObjectIdentifier(v.buffer)).inserted { experts += v.buffer.length }
        let rows: [(String, Int)] = [
            ("fixed per-batch", lens([R, xn, lo, gate, mixed, inj, blk, blkShared, rmsScale, qkv, conv, z, ga, gb, lo6144,
                                      qg, q, qgate, ao, iq, routerLogits, shG, shU, shY, shGate, acts, routeW, pairSlot,
                                      pleEmb, pleKey, pleValue, pleKeyN, pleQuery, pleGateBuf, pleGated, pleNormed, pleHist,
                                      attnMax, attnSum, nq, useSel, selThr, selCut, idxScores, partOffsets])),
            ("attnScores+selection", lens([attnScores, selection])),
            ("attn QG/OG/KG/VG", lens([attnQG, attnOG, attnKG, attnVG])),
            ("attnSelScores", len(attnSelScores)),
            ("idxDots+selAny+selList", lens([idxDots, selAny, selList])),
            ("gemm", lens([gemmRows, gemmGateUp, gemmWGateUp, gemmWDown, gemmOrder, gemmAt, gemmPosSlot, gemmOnes])),
            ("hidden + mtp", lens([hiddenOut, mtpHiddenOut, mtpCat])),
            ("dense scratch", dense.scratchBytes),
            ("logits", len(logits)),
            ("KV + indexer keys", lens(Array(kCache.values) + Array(vCache.values) + Array(idxRawKeys.values) + Array(idxBlockKeys.values))),
            ("GDN hist + state", lens(Array(linHist.values) + Array(linState.values))),
            ("dense views (resident)", denseViews),
            ("expert views (\(expertParts.count))", experts),
        ]
        let allocated = device.currentAllocatedSize
        let other = allocated - rows.reduce(0) { $0 + $1.1 }
        return [("metal allocated", allocated), ("other (MPS etc.)", other)] + rows
    }

    // MARK: - PLE

    /// Host half: hashed n-gram rows (ds4 `qwen4_ple_step`), 16 Q4_1 rows of 160 into `pleEmb[t]`.
    private func pleRows(tokens: [Int]) throws {
        let pt = try pleFile.tensor("ple.weight")
        precondition(pt.type == .q4_1 && pt.rowWidth == 160)
        let emb = pleEmb.contents().bindMemory(to: Float.self, capacity: tokens.count * e)
        for (ti, token) in tokens.enumerated() {
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
            let out = emb + ti * e
            for (h, r) in rows.enumerated() {
                let p = pleFile.base + pt.offset + r * pt.bytesPerRow
                for b in 0..<5 {
                    let d = Float(p.loadUnaligned(fromByteOffset: b * 20, as: Float16.self))
                    let m = Float(p.loadUnaligned(fromByteOffset: b * 20 + 2, as: Float16.self))
                    for j in 0..<16 {
                        let byte = p.load(fromByteOffset: b * 20 + 4 + j, as: UInt8.self)
                        out[h * 160 + b * 32 + j] = d * Float(byte & 0x0F) + m
                        out[h * 160 + b * 32 + 16 + j] = d * Float(byte >> 4) + m
                    }
                }
            }
        }
    }

    /// GPU half: gated key/value against the residual, then the dilated conv, added to R.
    private func pleBlock(_ cb: MTLCommandBuffer, il: Int, T: Int) throws {
        let pre = "blk.\(il)."
        let W = hc * e
        try gemv(cb, pre + "ple_key.weight", x: pleEmb, y: pleKey, tokens: T)
        try gemv(cb, pre + "ple_value.weight", x: pleEmb, y: pleValue, tokens: T)
        try rms(cb, x: pleKey, gamma: pre + "ple_norm_key.weight", out: pleKeyN, groups: T * hc, n: e, gammaLen: W)
        try rms(cb, x: R, gamma: pre + "ple_norm_query.weight", out: pleQuery, groups: T * hc, n: e, gammaLen: W)
        lanes(cb, psoPleGate, size(T * hc)) { enc in
            enc.setBuffer(pleKeyN, offset: 0, index: 0)
            enc.setBuffer(pleQuery, offset: 0, index: 1)
            enc.setBuffer(pleGateBuf, offset: 0, index: 2)
            var eV = UInt32(e)
            enc.setBytes(&eV, length: 4, index: 3)
        }
        run(cb, psoPleGated, size(T * W)) { enc in
            enc.setBuffer(pleGateBuf, offset: 0, index: 0)
            enc.setBuffer(pleValue, offset: 0, index: 1)
            enc.setBuffer(pleGated, offset: 0, index: 2)
            var p = (UInt32(hc), UInt32(e))
            enc.setBytes(&p, length: 8, index: 3)
        }
        try rms(cb, x: pleGated, gamma: pre + "ple_norm_conv.weight", out: pleNormed, groups: T * hc, n: e, gammaLen: W)
        let cw = try view(pre + "ple_conv1d.weight")
        var cp = (UInt32(W), UInt32(pleNgram), UInt32((4 - 1) * pleNgram), UInt32(T))
        let cpLen = MemoryLayout.size(ofValue: cp)
        run(cb, psoPleConvAdd, size(W, T)) { enc in
            enc.setBuffer(R, offset: 0, index: 0)
            enc.setBuffer(pleGated, offset: 0, index: 1)
            enc.setBuffer(pleNormed, offset: 0, index: 2)
            enc.setBuffer(pleHist, offset: 0, index: 3)
            enc.setBuffer(cw.buffer, offset: cw.offset, index: 4)
            enc.setBytes(&cp, length: cpLen, index: 5)
        }
        run(cb, psoPleHist, size(W)) { enc in
            enc.setBuffer(pleNormed, offset: 0, index: 0)
            enc.setBuffer(pleHist, offset: 0, index: 1)
            enc.setBytes(&cp, length: cpLen, index: 2)
        }
    }

    // MARK: - Forward

    /// After a `snapshotFirst` forward, sets the recurrent state to what it was after that batch's first token only
    /// (the rest of the batch was rejected): the GDN state from the step kernel's copy (buffers swapped), the conv
    /// and PLE histories shifted by one from their copies with the first token's rows, and `plePrev`. Call it before
    /// the next trunk forward (it reads that forward's first PLE row).
    package func rollbackToFirst() {
        precondition(snapshotTaken, "rollbackToFirst without a snapshotFirst forward")
        let C = 2 * Hk * Dl + Hv * Dl
        for il in 0..<nTrunk where isLinear(il) {
            let snap = linStateSnap[il]!
            linStateSnap[il] = linState[il]
            linState[il] = snap
            let hist = linHist[il]!.contents(), before = linHistBefore[il]!.contents()
            hist.copyMemory(from: before + C * 4, byteCount: (convK - 2) * C * 4)
            (hist + (convK - 2) * C * 4).copyMemory(from: linQkv0[il]!.contents(), byteCount: C * 4)
        }
        let W = hc * e
        let rowsH = pleHist.length / 4 / W
        let ph = pleHist.contents().bindMemory(to: Float.self, capacity: rowsH * W)
        pleHistBefore.withUnsafeBufferPointer { ph.update(from: $0.baseAddress! + W, count: (rowsH - 1) * W) }
        (ph + (rowsH - 1) * W).update(from: pleNormed.contents().bindMemory(to: Float.self, capacity: W), count: W)
        plePrev = [firstToken] + plePrevBefore.dropLast()
        snapshotTaken = false
    }

    /// Shared expert into `shY` / `shGate` and the router logits, from `mixed`.
    private func sharedAndRouter(_ cb: MTLCommandBuffer, pre: String, T: Int) throws {
        try shared(cb, pre: pre, T: T)
        try gemv(cb, pre + "ffn_gate_inp.weight", x: mixed, y: routerLogits, tokens: T)
    }

    /// `blkShared` = sigmoid(`shGate`) * `shY`.
    private func encodeSharedSum(_ cb: MTLCommandBuffer, T: Int) {
        run(cb, psoAddScaled, size(T * e)) { enc in
            enc.setBuffer(blkShared, offset: 0, index: 0)
            enc.setBuffer(shY, offset: 0, index: 1)
            enc.setBuffer(shGate, offset: 0, index: 2)
            var op = UInt32(2), n = UInt32(e)
            enc.setBytes(&op, length: 4, index: 3)
            enc.setBytes(&n, length: 4, index: 4)
        }
    }

    /// The shared expert in its own buffer, committed without a wait (`sharedLate`).
    private func commitShared(pre: String, T: Int) throws {
        let cb = queue.makeCommandBuffer()!
        try shared(cb, pre: pre, T: T)
        encodeSharedSum(cb, T: T)
        cb.commit()
    }

    /// Each row's top `n` of `previewLogits` (lowest index first on ties), as one set.
    private func previewExperts(T: Int, n: Int) -> Set<Int> {
        let lg = previewLogits.contents().bindMemory(to: Float.self, capacity: T * nExperts)
        var named = Set<Int>()
        for t in 0..<T {
            let row = lg + t * nExperts
            var top: [(Float, Int)] = []
            for i in 0..<nExperts {
                let v = row[i]
                if top.count == n, v <= top[n - 1].0 { continue }
                var at = top.count
                while at > 0 && top[at - 1].0 < v { at -= 1 }
                top.insert((v, i), at: at)
                if top.count > n { top.removeLast() }
            }
            for (_, i) in top { named.insert(i) }
        }
        return named
    }

    /// `F_RDADVISE` for the experts `named` of layer `il` (adjacent runs merged, as in `moeRouted`).
    private func advisePreview(il: Int, named: Set<Int>) throws {
        let pre = "blk.\(il)."
        let parts = [try file.tensor(pre + "ffn_gate_exps.weight"), try file.tensor(pre + "ffn_up_exps.weight"),
                     try file.tensor(pre + "ffn_down_exps.weight")]
        let bytes = [parts[0].bytesPerRow * F, parts[1].bytesPerRow * F, parts[2].bytesPerRow * e]
        let sorted = named.sorted()
        for (k, t) in parts.enumerated() {
            var i = 0
            while i < sorted.count {
                var j = i
                while j + 1 < sorted.count && sorted[j + 1] == sorted[j] + 1 { j += 1 }
                file.adviseRead(offset: t.offset + sorted[i] * bytes[k], byteCount: (sorted[j] - sorted[i] + 1) * bytes[k])
                i = j + 1
            }
        }
    }

    private func shared(_ cb: MTLCommandBuffer, pre: String, T: Int) throws {
        try gemv(cb, pre + "ffn_gate_shexp.weight", x: mixed, y: shG, tokens: T)
        try gemv(cb, pre + "ffn_up_shexp.weight", x: mixed, y: shU, tokens: T)
        run(cb, psoSiluMul, size(T * F)) { enc in
            enc.setBuffer(shG, offset: 0, index: 0)
            enc.setBuffer(shU, offset: 0, index: 1)
        }
        try gemv(cb, pre + "ffn_down_shexp.weight", x: shG, y: shY, tokens: T)
        try gemv(cb, pre + "ffn_gate_inp_shexp.weight", x: mixed, y: shGate, tokens: T)
    }

    private func resizeBatch(T: Int) {
        if T > batchRows {
            allocateBatch(rows: maxBatch)
        } else if shrinkBatch && T <= smallBatchRows && batchRows > smallBatchRows {
            allocateBatch(rows: smallBatchRows)
        }
    }

    /// Token embeddings (BF16, host) of `tokens`, `copies` times each, into `out` (rows of e).
    private func embed(_ tokens: [Int], copies: Int, into out: MTLBuffer) throws {
        let emb = try file.tensor("token_embd.weight")
        precondition(emb.type == .bf16)
        let rp = out.contents().bindMemory(to: Float.self, capacity: tokens.count * copies * e)
        for (t, token) in tokens.enumerated() {
            let src = file.base + emb.offset + token * e * 2
            for d in 0..<e {
                let v = Float(bitPattern: UInt32(src.loadUnaligned(fromByteOffset: d * 2, as: UInt16.self)) << 16)
                for s in 0..<copies { rp[(t * copies + s) * e + d] = v }
            }
        }
    }

    /// The trunk's final residual row `row` of the last `forward` with `exportHidden` (hc x e floats).
    package func hidden(row: Int) -> UnsafePointer<Float> {
        UnsafePointer(hiddenOut!.contents().bindMemory(to: Float.self, capacity: (row + 1) * hc * e) + row * hc * e)
    }

    /// The MTP head's residual row `row` of its last forward (hc x e floats): a chained draft's `hidden`.
    package func mtpHidden(row: Int) -> UnsafePointer<Float> {
        UnsafePointer(mtpHiddenOut!.contents().bindMemory(to: Float.self, capacity: (row + 1) * hc * e) + row * hc * e)
    }

    /// The MTP draft head (blk.<nTrunk>) on `tokens` at `startPos ..< startPos + T`, token t paired with the residual
    /// row `hidden[t]` (T x hc x e floats) of the position before it, as llama.cpp `graph_mtp` (qwen4exp.cpp
    /// `a9e9c3c`, 489-660): embedding -> enorm, repeated per stream; h -> per-stream RMS -> hnorm; per stream
    /// [e, h] -> eh_proj; then one layer (hc_attn mix, dense gated attention with its own KV, hc_ffn mix, shared +
    /// routed experts at Q4_K / MXFP4), its residual kept (`mtpHidden`), the `nextn.hc_head` mix and `output.weight`.
    /// Returns the last row's logits, or every row's with `allLogits`. Valid until the next forward of either kind.
    package func mtpForward(tokens: [Int], startPos: Int, hidden: UnsafePointer<Float>,
                            allLogits: Bool = false) throws -> UnsafeBufferPointer<Float> {
        try autoreleasepool { try mtpForwardBody(tokens: tokens, startPos: startPos, hidden: hidden, allLogits: allLogits) }
    }

    private func mtpForwardBody(tokens: [Int], startPos: Int, hidden: UnsafePointer<Float>,
                                allLogits: Bool) throws -> UnsafeBufferPointer<Float> {
        let T = tokens.count
        precondition(T >= 1 && T <= maxBatch && startPos + T <= capacity)
        var prof = StepProfile()
        let tStep = CFAbsoluteTimeGetCurrent()
        resizeBatch(T: T)
        let il = nTrunk, pre = "blk.\(nTrunk)."
        try embed(tokens, copies: 1, into: blk)
        R.contents().copyMemory(from: hidden, byteCount: T * hc * e * 4)
        var cb = queue.makeCommandBuffer()!
        try rms(cb, x: blk, gamma: pre + "nextn.enorm.weight", out: mixed, groups: T, n: e, gammaLen: e)
        try rms(cb, x: R, gamma: pre + "nextn.hnorm.weight", out: xn, groups: T * hc, n: e, gammaLen: hc * e)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw error }
        prof.preGPU += cb.gpuEndTime - cb.gpuStartTime
        // Per stream, not pooled (qwen4exp.cpp): row (t, s) = [enorm(x_t), hnorm(h_t)[s]].
        ensureBuffer(&mtpCat, bytes: T * hc * 2 * e * 4, shared: true)
        let cat = mtpCat!.contents().bindMemory(to: Float.self, capacity: T * hc * 2 * e)
        let en = mixed.contents().bindMemory(to: Float.self, capacity: T * e)
        let hn = xn.contents().bindMemory(to: Float.self, capacity: T * hc * e)
        for t in 0..<T {
            for s in 0..<hc {
                let o = cat + (t * hc + s) * 2 * e
                o.update(from: en + t * e, count: e)
                (o + e).update(from: hn + (t * hc + s) * e, count: e)
            }
        }
        cb = queue.makeCommandBuffer()!
        try gemv(cb, pre + "nextn.eh_proj.weight", x: mtpCat!, y: R, tokens: T * hc)
        try hcMix(cb, prefix: pre + "hc_attn", inject: true, T: T)
        try attention(&cb, il: il, pos0: startPos, T: T, dense: true)
        combine(cb, block: blk, T: T)
        try hcMix(cb, prefix: pre + "hc_ffn", inject: true, T: T)
        try sharedAndRouter(cb, pre: pre, T: T)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw error }
        let t1 = CFAbsoluteTimeGetCurrent()
        prof.ple = 0
        prof.preRouter = t1 - tStep
        prof.preGPU += cb.gpuEndTime - cb.gpuStartTime

        let cb2 = try moeRouted(il: il, T: T, prof: &prof)
        combine(cb2, block: blk, T: T)
        let t2 = CFAbsoluteTimeGetCurrent()
        prof.route = t2 - t1
        cb2.commit()
        cb2.waitUntilCompleted()
        if let error = cb2.error { throw error }
        prof.routed = CFAbsoluteTimeGetCurrent() - t2
        prof.routedGPU = cb2.gpuEndTime - cb2.gpuStartTime

        let tHead = CFAbsoluteTimeGetCurrent()
        ensureBuffer(&mtpHiddenOut, bytes: T * hc * e * 4, shared: true)
        mtpHiddenOut!.contents().copyMemory(from: R.contents(), byteCount: T * hc * e * 4)
        let rows = allLogits ? T : 1
        if !allLogits && T > 1 {
            memmove(R.contents(), R.contents() + (T - 1) * hc * e * 4, hc * e * 4)
        }
        if logits.length < rows * vocab * 4 {
            logits = device.makeBuffer(length: rows * vocab * 4, options: .storageModeShared)!
        }
        let cb3 = queue.makeCommandBuffer()!
        try hcMix(cb3, prefix: pre + "nextn.hc_head", inject: false, T: rows)
        try gemv(cb3, "output.weight", x: mixed, y: logits, tokens: rows)
        cb3.commit()
        cb3.waitUntilCompleted()
        if let error = cb3.error { throw error }
        prof.head = CFAbsoluteTimeGetCurrent() - tHead
        prof.total = CFAbsoluteTimeGetCurrent() - tStep
        lastMTPProfile = prof
        return UnsafeBufferPointer(start: logits.contents().bindMemory(to: Float.self, capacity: rows * vocab),
                                   count: rows * vocab)
    }

    /// Forward `token` at `pos`; returns its logits.
    package func step(token: Int, pos: Int) throws -> UnsafeBufferPointer<Float> {
        try forward(tokens: [token], startPos: pos)
    }

    /// Forward `tokens` at positions `startPos ..< startPos + tokens.count` (the cache must hold
    /// every earlier position). Returns the last token's logits, or with `allLogits` every
    /// token's (`[T][vocab]`).
    package func forward(tokens: [Int], startPos: Int, allLogits: Bool = false) throws -> UnsafeBufferPointer<Float> {
        // Command buffers and encoders are autoreleased; without a pool (the CLI) they, and the batch scratch
        // they reference, outlive the forward (+1.2 GB after `allocateBatch` shrank it, docs/qwen38/09).
        try autoreleasepool { try forwardBody(tokens: tokens, startPos: startPos, allLogits: allLogits) }
    }

    private func forwardBody(tokens: [Int], startPos: Int, allLogits: Bool) throws -> UnsafeBufferPointer<Float> {
        let T = tokens.count
        precondition(T >= 1 && T <= maxBatch && startPos + T <= capacity)
        var prof = StepProfile()
        let tStep = CFAbsoluteTimeGetCurrent()
        resizeBatch(T: T)
        snapshotTaken = false
        if snapshotFirst {
            precondition(T >= 2 && (gdnChunk == 0 || T < gdnMinTokens), "snapshotFirst needs the serial GDN step")
            let C = 2 * Hk * Dl + Hv * Dl
            for il in 0..<nTrunk where isLinear(il) {
                let before = linHistBefore[il] ?? device.makeBuffer(length: (convK - 1) * C * 4, options: .storageModeShared)!
                linHistBefore[il] = before
                if let hist = linHist[il] {
                    before.contents().copyMemory(from: hist.contents(), byteCount: hist.length)
                } else {
                    memset(before.contents(), 0, before.length)
                }
            }
            pleHistBefore = Array(UnsafeBufferPointer(start: pleHist.contents().bindMemory(to: Float.self, capacity: pleHist.length / 4),
                                                      count: pleHist.length / 4))
            plePrevBefore = plePrev
            firstToken = tokens[0]
            snapshotTaken = true
        }
        try embed(tokens, copies: hc, into: R)
        try pleRows(tokens: tokens)
        prof.ple = CFAbsoluteTimeGetCurrent() - tStep

        // The last routed buffer and its host commit time; with `pipeline` a later wait has completed it.
        var pendingRouted: (cb: MTLCommandBuffer, commit: CFTimeInterval)?
        func settleRouted(waited: Bool) throws {
            guard let p = pendingRouted else { return }
            pendingRouted = nil
            if !waited { p.cb.waitUntilCompleted() }
            let hostDone = CACurrentMediaTime()
            if let error = p.cb.error { throw error }
            prof.routedGPU += p.cb.gpuEndTime - p.cb.gpuStartTime
            prof.routedToKernel += p.cb.kernelStartTime - p.commit
            prof.routedKernelToGPU += p.cb.gpuStartTime - p.cb.kernelStartTime
            if !waited { prof.routedAfterGPU += hostDone - p.cb.gpuEndTime }
        }
        let late = pipeline && sharedLate
        let preview = pipeline && previewTopN > 0 && T < 32   // decode and verify batches only
        var previewed: Set<Int>?     // layer il's experts named by layer il-1's preview

        for il in 0..<nTrunk {
            let t0 = CFAbsoluteTimeGetCurrent()
            let pre = "blk.\(il)."
            var cb = queue.makeCommandBuffer()!
            @discardableResult
            func section(_ label: String) throws -> MTLCommandBuffer {
                guard splitPreRouter else { return cb }
                cb.commit()
                cb.waitUntilCompleted()
                if let error = cb.error { throw error }
                prof.sections[label, default: 0] += (cb.gpuEndTime - cb.gpuStartTime) * 1000
                prof.preGPU += cb.gpuEndTime - cb.gpuStartTime
                cb = queue.makeCommandBuffer()!
                return cb
            }
            if il == pleLayer {
                try pleBlock(cb, il: il, T: T)
                try section("ple")
            }
            try hcMix(cb, prefix: pre + "hc_attn", inject: true, T: T)
            try section("hc_attn")
            if isLinear(il) { try linear(cb, il: il, T: T, section: section) } else { try attention(&cb, il: il, pos0: startPos, T: T) }
            try section(isLinear(il) ? "gdn" : "attn")
            combine(cb, block: blk, T: T)
            try hcMix(cb, prefix: pre + "hc_ffn", inject: true, T: T)
            try section("hc_ffn")
            if late {
                try gemv(cb, pre + "ffn_gate_inp.weight", x: mixed, y: routerLogits, tokens: T)
            } else {
                try sharedAndRouter(cb, pre: pre, T: T)
            }
            if preview && il + 1 < nTrunk {
                try gemv(cb, "blk.\(il + 1).ffn_gate_inp.weight", x: mixed, y: previewLogits, tokens: T)
            }
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            try settleRouted(waited: true)
            let t1 = CFAbsoluteTimeGetCurrent()
            prof.preRouter += t1 - t0
            prof.preGPU += cb.gpuEndTime - cb.gpuStartTime
            if splitPreRouter { prof.sections["shexp+router", default: 0] += (cb.gpuEndTime - cb.gpuStartTime) * 1000 }

            let cb2 = try moeRouted(il: il, T: T, prof: &prof, sharedCommitted: late, previewed: previewed)
            combine(cb2, block: blk, T: T)
            let t2 = CFAbsoluteTimeGetCurrent()
            prof.route += t2 - t1
            pendingRouted = (cb2, CACurrentMediaTime())
            cb2.commit()
            // With `pipeline` the next layer's wait (or the one below) completes it: `routed` is then only the
            // commit, and its GPU and page-in time land in the next `preRouter`.
            if !pipeline { try settleRouted(waited: false) }
            prof.routed += CFAbsoluteTimeGetCurrent() - t2
            if preview && il + 1 < nTrunk {
                let tp = CFAbsoluteTimeGetCurrent()
                let named = previewExperts(T: T, n: previewTopN)
                if previewAdvise { try advisePreview(il: il + 1, named: named) }
                previewed = named
                prof.previewAdvise += CFAbsoluteTimeGetCurrent() - tp
            }
        }
        if let p = pendingRouted {
            // The host reads `R` below.
            let t3 = CFAbsoluteTimeGetCurrent()
            p.cb.waitUntilCompleted()
            try settleRouted(waited: true)
            prof.routed += CFAbsoluteTimeGetCurrent() - t3
        }

        let tHead = CFAbsoluteTimeGetCurrent()
        if exportHidden {
            ensureBuffer(&hiddenOut, bytes: T * hc * e * 4, shared: true)
            hiddenOut!.contents().copyMemory(from: R.contents(), byteCount: T * hc * e * 4)
        }
        var rows = T
        if !allLogits && T > 1 {
            // Only the last token's logits: move its residual rows to the front.
            memmove(R.contents(), R.contents() + (T - 1) * hc * e * 4, hc * e * 4)
            rows = 1
        }
        if logits.length < rows * vocab * 4 {
            logits = device.makeBuffer(length: rows * vocab * 4, options: .storageModeShared)!
        }
        let cb = queue.makeCommandBuffer()!
        try hcMix(cb, prefix: "output_hc", inject: false, T: rows)
        try gemv(cb, "output.weight", x: mixed, y: logits, tokens: rows)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw error }
        prof.head = CFAbsoluteTimeGetCurrent() - tHead
        if denseResident && views.count != denseSetCount {
            if denseSet == nil {
                let d = MTLResidencySetDescriptor()
                d.label = "q38-dense"
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
