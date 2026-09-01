import Foundation
import Metal

/// Swift wrappers for the Qwen3.5-MoE kernels in `Metal/Qwen/qwen.metal`
/// (`docs/qwen35moe/03-DESIGN.md` §2). `qwen_delta_rule` lives on its own in
/// `GatedDeltaNet`; everything else around it is here.
///
/// The module compiles separately from the shared runtime library, so the
/// Gemma 4 path pays nothing for it at startup — the same arrangement
/// `tensorops` and `gdn` already use.
///
/// One linear-attention layer runs these in order:
///
///     in_proj_qkv ──▶ qwen_delta_qkv_prepare ──▶ qwen_delta_rule
///     in_proj_a/b ──▶ qwen_delta_gates      ──┘        │
///     in_proj_z ───────────────────────────▶ qwen_delta_norm_gate ──▶ out_proj
///
/// and one full-attention layer runs `qwen_qkv_epilogue` before the attention
/// kernels and `qwen_attn_output_gate` after them.
public final class QwenKernels {

    /// `linear_conv_kernel_dim`. The causal window lives in registers, so a
    /// different kernel width needs a different shader, not another argument.
    public static let convKernel = 4
    /// Tokens one `qwen_delta_qkv_prepare` threadgroup covers. The causal
    /// window is 4, so tokens are parallel work; running the whole chunk in one
    /// threadgroup leaves only 64 of them and the kernel measures at a quarter
    /// of the machine's bandwidth. Phase 4 sweeps this
    /// (`docs/qwen35moe/17-PHASE2-KERNELS.md` §4).
    public static let tokensPerGroup = 32
    public static let rmsEps: Float = 1e-6
    /// `use_qk_l2norm_in_kernel` adds this to the *sum* of squares, not the
    /// mean (`Scripts/qwen35/reference_forward.py` `l2_norm`).
    public static let l2Eps: Float = 1e-6

    private struct DeltaQKVParams {
        var seqLen: UInt32
        var numKHeads: UInt32
        var numVHeads: UInt32
        var headDim: UInt32
        var convKernel: UInt32
        var tokensPerGroup: UInt32
        var l2Eps: Float
        var qScale: Float
    }

    private struct DeltaGateParams {
        var seqLen: UInt32
        var numVHeads: UInt32
    }

    private struct DeltaNormGateParams {
        var seqLen: UInt32
        var numVHeads: UInt32
        var headDim: UInt32
        var eps: Float
    }

    private struct QKVEpilogueParams {
        var seqLen: UInt32
        var numQHeads: UInt32
        var numKVHeads: UInt32
        var headDim: UInt32
        var rotaryDim: UInt32
        var position: UInt32
        var theta: Float
        var eps: Float
    }

    private struct AttnGateParams {
        var seqLen: UInt32
        var numQHeads: UInt32
        var headDim: UInt32
    }

    private enum Function: String, CaseIterable {
        case deltaQKVPrepare = "qwen_delta_qkv_prepare"
        case deltaGates      = "qwen_delta_gates"
        case deltaNormGate   = "qwen_delta_norm_gate"
        case qkvEpilogue     = "qwen_qkv_epilogue"
        case attnOutputGate  = "qwen_attn_output_gate"
        case moeSharedGate   = "qwen_moe_shared_gate"
        case moeSharedGateLogit = "qwen_moe_shared_gate_logit"
        case siluMul         = "qwen_silu_mul"
        case residualAdd     = "qwen_residual_add"
        case moeSharedGateLogitBlock = "qwen_moe_shared_gate_logit_block"
        case queryCompact    = "qwen_query_compact"
        case bf16GEMV        = "qwen_bf16_gemv_f16"
    }

    private var pipelines: [Function: MTLComputePipelineState] = [:]

    public init(context: MetalContext) throws {
        // Safe math for this module only: the decay gate is a double
        // exponential and its result multiplies the recurrent state on every
        // token, so `exp` accuracy is not cosmetic here. `gdn` keeps the
        // default, so the 125.7 ms measured for `qwen_delta_rule` still stands.
        let library = try MetalContext.moduleLibrary(device: context.device,
                                                     module: "qwen",
                                                     safeMath: true)
        for function in Function.allCases {
            guard let handle = library.makeFunction(name: function.rawValue) else {
                throw MetalError.missingFunction(function.rawValue)
            }
            pipelines[function] = try context.device.makeComputePipelineState(function: handle)
        }
    }

    private func encoder(_ commandBuffer: MTLCommandBuffer,
                         _ function: Function) -> (MTLComputeCommandEncoder,
                                                   MTLComputePipelineState)? {
        guard let pipeline = pipelines[function],
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
        encoder.setComputePipelineState(pipeline)
        return (encoder, pipeline)
    }

    /// Causal depthwise `conv1d` + SiLU + the q/k `l2norm` that feeds
    /// `qwen_delta_rule`, in one dispatch.
    ///
    ///   `qkv`        FP16 `[T, (2*Hk + Hv) * headDim]` — `in_proj_qkv` output
    ///   `convWeight` BF16 `[C, 4]` — MLX axis order, already squeezed
    ///   `stateIn`    FP16 `[3, C]` — oldest row first
    ///   `stateOut`   FP16 `[3, C]` — the last three tokens of this chunk
    ///   `q`, `k`     FP16 `[T, Hk, headDim]`
    ///   `v`          FP16 `[T, Hv, headDim]`
    ///
    /// One threadgroup owns one head for `tokensPerGroup` tokens; the causal
    /// window rides in registers. Tokens are split across threadgroups because
    /// a window of 4 is not a sequential dependency — see the shader header.
    ///
    /// **`stateOut` may alias `stateIn` only when the dispatch has a single
    /// token block** (`seqLen <= tokensPerGroup`, which decode always is): with
    /// more blocks, the group that writes the state and the group that reads it
    /// are different threadgroups with no ordering between them. The call
    /// refuses an unsafe alias rather than racing.
    @discardableResult
    public func encodeDeltaQKVPrepare(commandBuffer: MTLCommandBuffer,
                                      qkv: MTLBuffer, qkvOffset: Int = 0,
                                      convWeight: MTLBuffer, convWeightOffset: Int = 0,
                                      stateIn: MTLBuffer, stateInOffset: Int = 0,
                                      stateOut: MTLBuffer, stateOutOffset: Int = 0,
                                      q: MTLBuffer, qOffset: Int = 0,
                                      k: MTLBuffer, kOffset: Int = 0,
                                      v: MTLBuffer, vOffset: Int = 0,
                                      seqLen: Int,
                                      numKHeads: Int,
                                      numVHeads: Int,
                                      headDim: Int,
                                      tokensPerGroup: Int = QwenKernels.tokensPerGroup) -> Bool {
        let aliased = stateIn === stateOut && stateInOffset == stateOutOffset
        guard seqLen > 0, numKHeads > 0, numVHeads > 0,
              headDim > 0, headDim <= 1024,
              tokensPerGroup > 0,
              !(aliased && seqLen > tokensPerGroup),
              let (encoder, _) = encoder(commandBuffer, .deltaQKVPrepare) else { return false }
        encoder.setBuffer(qkv, offset: qkvOffset, index: 0)
        encoder.setBuffer(convWeight, offset: convWeightOffset, index: 1)
        encoder.setBuffer(stateIn, offset: stateInOffset, index: 2)
        encoder.setBuffer(stateOut, offset: stateOutOffset, index: 3)
        encoder.setBuffer(q, offset: qOffset, index: 4)
        encoder.setBuffer(k, offset: kOffset, index: 5)
        encoder.setBuffer(v, offset: vOffset, index: 6)
        var params = DeltaQKVParams(seqLen: UInt32(seqLen),
                                    numKHeads: UInt32(numKHeads),
                                    numVHeads: UInt32(numVHeads),
                                    headDim: UInt32(headDim),
                                    convKernel: UInt32(Self.convKernel),
                                    tokensPerGroup: UInt32(tokensPerGroup),
                                    l2Eps: Self.l2Eps,
                                    qScale: 1.0 / Float(Double(headDim).squareRoot()))
        encoder.setBytes(&params, length: MemoryLayout<DeltaQKVParams>.stride, index: 7)
        // One threadgroup per (head, token block): q and k contribute
        // `numKHeads` heads each, v the rest.
        let blocks = (seqLen + tokensPerGroup - 1) / tokensPerGroup
        encoder.dispatchThreadgroups(
            MTLSize(width: 2 * numKHeads + numVHeads, height: blocks, depth: 1),
            threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }

    /// `g = exp(-exp(A_log) * softplus(a + dt_bias))` and `beta = sigmoid(b)`,
    /// both FP32 because `g` sits just under 1 where FP16's step would bite.
    @discardableResult
    public func encodeDeltaGates(commandBuffer: MTLCommandBuffer,
                                 a: MTLBuffer, aOffset: Int = 0,
                                 b: MTLBuffer, bOffset: Int = 0,
                                 aLog: MTLBuffer, aLogOffset: Int = 0,
                                 dtBias: MTLBuffer, dtBiasOffset: Int = 0,
                                 g: MTLBuffer, gOffset: Int = 0,
                                 beta: MTLBuffer, betaOffset: Int = 0,
                                 seqLen: Int,
                                 numVHeads: Int) -> Bool {
        guard seqLen > 0, numVHeads > 0,
              let (encoder, pipeline) = encoder(commandBuffer, .deltaGates) else { return false }
        encoder.setBuffer(a, offset: aOffset, index: 0)
        encoder.setBuffer(b, offset: bOffset, index: 1)
        encoder.setBuffer(aLog, offset: aLogOffset, index: 2)
        encoder.setBuffer(dtBias, offset: dtBiasOffset, index: 3)
        encoder.setBuffer(g, offset: gOffset, index: 4)
        encoder.setBuffer(beta, offset: betaOffset, index: 5)
        var params = DeltaGateParams(seqLen: UInt32(seqLen), numVHeads: UInt32(numVHeads))
        encoder.setBytes(&params, length: MemoryLayout<DeltaGateParams>.stride, index: 6)
        dispatch1D(encoder, pipeline, count: seqLen * numVHeads)
        encoder.endEncoding()
        return true
    }

    /// `RMSNormGated`: RMS over `headDim` **without** the `1 + w` convention,
    /// then `* silu(z)`. Weight is one `[headDim]` vector shared by all heads.
    @discardableResult
    public func encodeDeltaNormGate(commandBuffer: MTLCommandBuffer,
                                    o: MTLBuffer, oOffset: Int = 0,
                                    z: MTLBuffer, zOffset: Int = 0,
                                    weight: MTLBuffer, weightOffset: Int = 0,
                                    out: MTLBuffer, outOffset: Int = 0,
                                    seqLen: Int,
                                    numVHeads: Int,
                                    headDim: Int,
                                    eps: Float = QwenKernels.rmsEps) -> Bool {
        guard seqLen > 0, numVHeads > 0, headDim > 0, headDim <= 1024,
              let (encoder, _) = encoder(commandBuffer, .deltaNormGate) else { return false }
        encoder.setBuffer(o, offset: oOffset, index: 0)
        encoder.setBuffer(z, offset: zOffset, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var params = DeltaNormGateParams(seqLen: UInt32(seqLen),
                                         numVHeads: UInt32(numVHeads),
                                         headDim: UInt32(headDim),
                                         eps: eps)
        encoder.setBytes(&params, length: MemoryLayout<DeltaNormGateParams>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: numVHeads, height: seqLen, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(headDim, 128), height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }

    /// `q_norm` / `k_norm` plus Qwen's partial RoPE, in place.
    ///
    /// `q` is the doubled-width `q_proj` output `[T, NQ, 2*headDim]`: the first
    /// `headDim` per head is q, the second is the output gate this kernel does
    /// not touch. `rotaryDim` dimensions are rotated with pairs `(i, rd/2 + i)`
    /// and a `theta` denominator of `rotaryDim` — neither matches the Gemma
    /// kernels, which is why this one exists (`docs/qwen35moe/03-DESIGN.md` §2-2).
    @discardableResult
    public func encodeQKVEpilogue(commandBuffer: MTLCommandBuffer,
                                  q: MTLBuffer, qOffset: Int = 0,
                                  k: MTLBuffer, kOffset: Int = 0,
                                  qWeight: MTLBuffer, qWeightOffset: Int = 0,
                                  kWeight: MTLBuffer, kWeightOffset: Int = 0,
                                  seqLen: Int,
                                  numQHeads: Int,
                                  numKVHeads: Int,
                                  headDim: Int,
                                  rotaryDim: Int,
                                  position: Int,
                                  theta: Float,
                                  eps: Float = QwenKernels.rmsEps) -> Bool {
        guard seqLen > 0, numQHeads > 0, numKVHeads > 0,
              headDim > 0, headDim <= 256,
              rotaryDim > 0, rotaryDim <= headDim, rotaryDim % 2 == 0,
              position >= 0,
              let (encoder, _) = encoder(commandBuffer, .qkvEpilogue) else { return false }
        encoder.setBuffer(q, offset: qOffset, index: 0)
        encoder.setBuffer(k, offset: kOffset, index: 1)
        encoder.setBuffer(qWeight, offset: qWeightOffset, index: 2)
        encoder.setBuffer(kWeight, offset: kWeightOffset, index: 3)
        var params = QKVEpilogueParams(seqLen: UInt32(seqLen),
                                       numQHeads: UInt32(numQHeads),
                                       numKVHeads: UInt32(numKVHeads),
                                       headDim: UInt32(headDim),
                                       rotaryDim: UInt32(rotaryDim),
                                       position: UInt32(position),
                                       theta: theta,
                                       eps: eps)
        encoder.setBytes(&params, length: MemoryLayout<QKVEpilogueParams>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: numQHeads + numKVHeads, height: seqLen, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }

    /// `o *= sigmoid(gate)`, reading the gate out of the doubled-width `q_proj`
    /// output the epilogue left alone.
    @discardableResult
    public func encodeAttnOutputGate(commandBuffer: MTLCommandBuffer,
                                     o: MTLBuffer, oOffset: Int = 0,
                                     qGate: MTLBuffer, qGateOffset: Int = 0,
                                     seqLen: Int,
                                     numQHeads: Int,
                                     headDim: Int) -> Bool {
        guard seqLen > 0, numQHeads > 0, headDim > 0,
              let (encoder, pipeline) = encoder(commandBuffer, .attnOutputGate) else { return false }
        encoder.setBuffer(o, offset: oOffset, index: 0)
        encoder.setBuffer(qGate, offset: qGateOffset, index: 1)
        var params = AttnGateParams(seqLen: UInt32(seqLen),
                                    numQHeads: UInt32(numQHeads),
                                    headDim: UInt32(headDim))
        encoder.setBytes(&params, length: MemoryLayout<AttnGateParams>.stride, index: 2)
        dispatch1D(encoder, pipeline, count: seqLen * numQHeads * headDim)
        encoder.endEncoding()
        return true
    }

    /// `y *= sigmoid(shared_expert_gate . x)` — the scalar gate Gemma has no
    /// counterpart for (`docs/qwen35moe/01-MODEL.md` §3-4).
    @discardableResult
    public func encodeMoESharedGate(commandBuffer: MTLCommandBuffer,
                                    y: MTLBuffer, yOffset: Int = 0,
                                    x: MTLBuffer, xOffset: Int = 0,
                                    weight: MTLBuffer, weightOffset: Int = 0,
                                    hiddenSize: Int) -> Bool {
        guard hiddenSize > 0,
              let (encoder, _) = encoder(commandBuffer, .moeSharedGate) else { return false }
        encoder.setBuffer(y, offset: yOffset, index: 0)
        encoder.setBuffer(x, offset: xOffset, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)
        var d = UInt32(hiddenSize)
        encoder.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }

    /// `y *= sigmoid(logit)` — the same gate as `encodeMoESharedGate` with the
    /// dot product already taken by a dequantizing GEMV. This is the arm the
    /// production checkpoint needs: its `shared_expert_gate.weight` is 8-bit
    /// affine, not the BF16 the fused kernel reads
    /// (`docs/qwen35moe/18-MIXED-BITS.md` §3).
    @discardableResult
    public func encodeMoESharedGateLogit(commandBuffer: MTLCommandBuffer,
                                         y: MTLBuffer, yOffset: Int = 0,
                                         logit: MTLBuffer, logitOffset: Int = 0,
                                         hiddenSize: Int) -> Bool {
        guard hiddenSize > 0,
              let (encoder, pipeline) = encoder(commandBuffer, .moeSharedGateLogit) else {
            return false
        }
        encoder.setBuffer(y, offset: yOffset, index: 0)
        encoder.setBuffer(logit, offset: logitOffset, index: 1)
        var d = UInt32(hiddenSize)
        encoder.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 2)
        dispatch1D(encoder, pipeline, count: hiddenSize)
        encoder.endEncoding()
        return true
    }

    /// `hidden += y`, the plain residual join this family uses twice a layer.
    @discardableResult
    public func encodeResidualAdd(commandBuffer: MTLCommandBuffer,
                                  hidden: MTLBuffer, hiddenOffset: Int = 0,
                                  y: MTLBuffer, yOffset: Int = 0,
                                  count: Int) -> Bool {
        guard count > 0,
              let (encoder, pipeline) = encoder(commandBuffer, .residualAdd) else { return false }
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 0)
        encoder.setBuffer(y, offset: yOffset, index: 1)
        var n = UInt32(count)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 2)
        dispatch1D(encoder, pipeline, count: count)
        encoder.endEncoding()
        return true
    }

    /// `out = silu(gate) * up`.
    @discardableResult
    public func encodeSiluMul(commandBuffer: MTLCommandBuffer,
                              gate: MTLBuffer, gateOffset: Int = 0,
                              up: MTLBuffer, upOffset: Int = 0,
                              out: MTLBuffer, outOffset: Int = 0,
                              count: Int) -> Bool {
        guard count > 0,
              let (encoder, pipeline) = encoder(commandBuffer, .siluMul) else { return false }
        encoder.setBuffer(gate, offset: gateOffset, index: 0)
        encoder.setBuffer(up, offset: upOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        var n = UInt32(count)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 3)
        dispatch1D(encoder, pipeline, count: count)
        encoder.endEncoding()
        return true
    }

    /// `y[t, :] *= sigmoid(logit[t])` — `encodeMoESharedGateLogit` for a whole
    /// prefill chunk, with the dot products already taken by a GEMM of `N == 1`.
    @discardableResult
    public func encodeMoESharedGateLogitBlock(commandBuffer: MTLCommandBuffer,
                                              y: MTLBuffer, yOffset: Int = 0,
                                              logit: MTLBuffer, logitOffset: Int = 0,
                                              hiddenSize: Int,
                                              seqLen: Int) -> Bool {
        guard hiddenSize > 0, seqLen > 0,
              let (encoder, pipeline) = encoder(commandBuffer, .moeSharedGateLogitBlock) else {
            return false
        }
        encoder.setBuffer(y, offset: yOffset, index: 0)
        encoder.setBuffer(logit, offset: logitOffset, index: 1)
        var d = UInt32(hiddenSize)
        var t = UInt32(seqLen)
        encoder.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&t, length: MemoryLayout<UInt32>.stride, index: 3)
        dispatch2D(encoder, pipeline, width: hiddenSize, height: seqLen)
        encoder.endEncoding()
        return true
    }

    /// Gathers the query halves out of the doubled-width `q_proj` output,
    /// `[T, NQ, 2*HD]` to `[T, NQ, HD]`, for the attention kernels' stride.
    ///
    /// Decode does the same thing with 16 blits, one per head
    /// (`docs/qwen35moe/20-PHASE3-DECODE.md` §3); at T = 2048 that would be
    /// 32,768 of them. The output gate still travels in the wide buffer —
    /// `encodeAttnOutputGate` reads it there after attention.
    @discardableResult
    public func encodeQueryCompact(commandBuffer: MTLCommandBuffer,
                                   wide: MTLBuffer, wideOffset: Int = 0,
                                   out: MTLBuffer, outOffset: Int = 0,
                                   seqLen: Int,
                                   numQHeads: Int,
                                   headDim: Int) -> Bool {
        guard seqLen > 0, numQHeads > 0, headDim > 0,
              let (encoder, pipeline) = encoder(commandBuffer, .queryCompact) else { return false }
        encoder.setBuffer(wide, offset: wideOffset, index: 0)
        encoder.setBuffer(out, offset: outOffset, index: 1)
        var hd = UInt32(headDim)
        var nq = UInt32(numQHeads)
        var t = UInt32(seqLen)
        encoder.setBytes(&hd, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&nq, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&t, length: MemoryLayout<UInt32>.stride, index: 4)
        dispatch2D(encoder, pipeline, width: numQHeads * headDim, height: seqLen)
        encoder.endEncoding()
        return true
    }

    private func dispatch2D(_ encoder: MTLComputeCommandEncoder,
                            _ pipeline: MTLComputePipelineState,
                            width: Int, height: Int) {
        let w = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    private func dispatch1D(_ encoder: MTLComputeCommandEncoder,
                            _ pipeline: MTLComputePipelineState,
                            count: Int) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreadgroups(
            MTLSize(width: (count + width - 1) / width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    /// `y = W x` with a BF16 weight matrix, FP16 in and out.
    ///
    /// One tensor in this repository needs it: the MTP head's `mtp.fc`
    /// (2048x4096), which the donor ships as BF16 and the graft copied through
    /// unquantized (`docs/qwen35moe/30-MTP-HEAD-GRAFT.md` §3-1). The router's
    /// BF16 GEMV cannot stand in — it writes FP32 logits for a top-k select,
    /// and what follows `fc` is an RMSNorm that reads FP16.
    ///
    /// One output row per SIMD group; the 16 MB of weights is read once.
    @discardableResult
    public func encodeBF16GEMV(commandBuffer: MTLCommandBuffer,
                               weights: MTLBuffer, weightsOffset: Int = 0,
                               x: MTLBuffer, xOffset: Int = 0,
                               y: MTLBuffer, yOffset: Int = 0,
                               m: Int, k: Int) -> Bool {
        guard m > 0, k > 0, k % 32 == 0,
              let (encoder, _) = encoder(commandBuffer, .bf16GEMV) else { return false }
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(x, offset: xOffset, index: 1)
        encoder.setBuffer(y, offset: yOffset, index: 2)
        var rows = UInt32(m)
        var cols = UInt32(k)
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&cols, length: MemoryLayout<UInt32>.stride, index: 4)
        let simdGroups = 8
        encoder.dispatchThreadgroups(
            MTLSize(width: (m + simdGroups - 1) / simdGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * simdGroups, height: 1, depth: 1))
        encoder.endEncoding()
        return true
    }
}
