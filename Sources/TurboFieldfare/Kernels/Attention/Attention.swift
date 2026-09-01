import Foundation
import Metal


struct AttentionSplitGeometry: Sendable, Equatable {
    let effectiveLength: Int
    let numChunks: Int
    let chunkLength: Int
    let partialThreadgroups: Int
    let useSWAGroupedPartial: Bool
}


/// Dispatch shape of the block-rows split-KV path (`Attention.encodeRows`).
struct AttentionRowsGeometry: Sendable, Equatable {
    /// Union range the chunks are cut over: `[kvStart, startPosition + rows)`.
    let kvStart: UInt32
    let effectiveLength: Int
    let numChunks: Int
    let chunkLength: Int
    /// Pass-1 threadgroups (`numQHeads * numChunks`) and their width.
    let partialThreadgroups: Int
    let partialThreadsPerGroup: Int
}

/// Swift wrapper for sliding-window and full-causal decode attention.
///
/// The kernels assume a single decoded query token (`M_q = 1`) and a
/// contiguous KV cache of length `seqLen`. The MPP-tensor-core prefill path
/// (`M_q > 1`) is separate.
///
/// Buffer contracts (FP16 throughout):
///   - `q`   : `[numQHeads, headDim]`
///   - `k`   : `[seqLen, numKVHeads, headDim]`
///   - `v`   : same shape as `k`. Full-layer K and V must remain distinct after
///             their separate per-head normalization and RoPE paths.
///   - `out` : `[numQHeads, headDim]`
final class Attention {
    private let ctx: MetalContext
    private let psoPartial: MTLComputePipelineState
    private let psoGQAPartial: MTLComputePipelineState
    private let psoCombine: MTLComputePipelineState
    private let psoPartialSWA: MTLComputePipelineState
    private let psoPartialFull: MTLComputePipelineState
    private let psoGQAPartialSWA: MTLComputePipelineState
    private let psoGQAPartialSWAChunks16: MTLComputePipelineState
    private let psoPartialFullChunks16: MTLComputePipelineState
    private let psoCombineSWA: MTLComputePipelineState
    private let psoCombineFull: MTLComputePipelineState
    private let psoCombineSWAChunks16: MTLComputePipelineState
    private let psoCombineFullChunks16: MTLComputePipelineState

    /// Mirrors `kAttnThreads` in `attention.metal`. The kernel was authored
    /// with a hardcoded 256-thread group so its threadgroup-memory scratch
    /// (q_smem[512] + reduce[8] + bcast) sizes are correct.
    static let threadsPerGroup: Int = 256

    /// Project ceilings for the split-KV partial scratch. `kAttnMaxHeadDim` in
    /// attention.metal is 512; the model has 16 Q heads; `maxChunks` bounds the
    /// split factor (and therefore the scratch size: 16·64·512 FP32 ≈ 2 MB).
    static let maxQHeads = 16
    static let maxHeadDim = 512
    static let maxChunks = 64
    /// Full attention uses 16 base chunks by default.
    private static let defaultFullChunks = 16
    private static let defaultGQASWAChunks = 8

    // Partial state written by pass 1, read by pass 2. One shared allocation:
    // attention runs once per layer, serially, and pass 2 hazard-tracks pass 1
    // within the same command buffer — no race (mirrors MoE.routerLogits).
    private let mPartial: MTLBuffer
    private let dPartial: MTLBuffer
    private let oPartial: MTLBuffer

    private struct RowsScratch {
        let m: MTLBuffer
        let d: MTLBuffer
        let o: MTLBuffer
    }
    private struct RowsPipelineKey: Hashable {
        let headDim: UInt32
        let numQHeads: UInt32
        let numKVHeads: UInt32
        let numChunks: UInt32
        let ringCapacity: UInt32
    }
    // Both are first-use caches for the block-rows path; `Attention` is used
    // from the one thread that encodes a layer, like the partial scratch above.
    private var rowsScratchStorage: RowsScratch?
    private var rowsPipelineCache: [RowsPipelineKey: MTLComputePipelineState] = [:]

    init(context: MetalContext) throws {
        self.ctx = context
        self.psoPartial = try context.pipeline("attention_decode_partial")
        self.psoGQAPartial = try context.pipeline("attention_decode_gqa_swa_partial")
        self.psoCombine = try context.pipeline("attention_decode_combine")
        self.psoPartialSWA = try Self.specializedPipeline(context,
                                                          "attention_decode_partial",
                                                          headDim: 256,
                                                          numQHeads: 16,
                                                          numKVHeads: 8)
        self.psoPartialFull = try Self.specializedPipeline(context,
                                                           "attention_decode_partial",
                                                           headDim: 512,
                                                           numQHeads: 16,
                                                           numKVHeads: 2)
        self.psoGQAPartialSWA = try Self.specializedPipeline(context,
                                                             "attention_decode_gqa_swa_partial",
                                                             headDim: 256,
                                                             numQHeads: 16,
                                                             numKVHeads: 8)
        self.psoGQAPartialSWAChunks16 = try Self.specializedPipeline(context,
                                                                     "attention_decode_gqa_swa_partial",
                                                                     headDim: 256,
                                                                     numQHeads: 16,
                                                                     numKVHeads: 8,
                                                                     numChunks: 16)
        self.psoPartialFullChunks16 = try Self.specializedPipeline(context,
                                                                   "attention_decode_partial",
                                                                   headDim: 512,
                                                                   numQHeads: 16,
                                                                   numKVHeads: 2,
                                                                   numChunks: 16)
        self.psoCombineSWA = try Self.specializedPipeline(context,
                                                          "attention_decode_combine",
                                                          headDim: 256,
                                                          numQHeads: 16,
                                                          numKVHeads: 8)
        self.psoCombineFull = try Self.specializedPipeline(context,
                                                           "attention_decode_combine",
                                                           headDim: 512,
                                                           numQHeads: 16,
                                                           numKVHeads: 2)
        self.psoCombineSWAChunks16 = try Self.specializedPipeline(context,
                                                                  "attention_decode_combine",
                                                                  headDim: 256,
                                                                  numQHeads: 16,
                                                                  numKVHeads: 8,
                                                                  numChunks: 16)
        self.psoCombineFullChunks16 = try Self.specializedPipeline(context,
                                                                   "attention_decode_combine",
                                                                   headDim: 512,
                                                                   numQHeads: 16,
                                                                   numKVHeads: 2,
                                                                   numChunks: 16)
        let md = Self.maxQHeads * Self.maxChunks
        guard let m = context.device.makeBuffer(length: md * MemoryLayout<Float>.size,
                                                options: .storageModeShared),
	              let d = context.device.makeBuffer(length: md * MemoryLayout<Float>.size,
	                                                options: .storageModeShared),
	              let o = context.device.makeBuffer(length: md * Self.maxHeadDim * MemoryLayout<Float>.size,
	                                                options: .storageModeShared) else {
            throw MetalError.missingFunction("attention split-KV scratch")
        }
        self.mPartial = m; self.dPartial = d; self.oPartial = o
    }

    /// Number of K/V chunks for a range of `effLen` positions — the split
    /// factor used by the production split path.
    static func chunkCount(effLen: Int, preferGQASWA: Bool = false) -> Int {
        let eff = max(1, effLen)
        let defaultChunks = preferGQASWA ? defaultGQASWAChunks : defaultFullChunks
        return max(1, min(defaultChunks, min(maxChunks, eff)))
    }

    static func splitGeometry(numQHeads: UInt32,
                                     numKVHeads: UInt32,
                                     seqLen: UInt32,
                                     kvStart: UInt32,
                                     preferGQASWA: Bool) -> AttentionSplitGeometry {
        let qPerKV = Int(numQHeads / numKVHeads)
        let useSWAGQAPartial = preferGQASWA && qPerKV <= 2
        let effectiveLength = Int(seqLen) - Int(kvStart)
        let baseChunks = Self.chunkCount(effLen: effectiveLength,
                                         preferGQASWA: useSWAGQAPartial)
        let numChunks = useSWAGQAPartial
            ? max(baseChunks, min(Self.maxChunks, baseChunks * qPerKV))
            : baseChunks
        let chunkLength = (max(1, effectiveLength) + numChunks - 1) / numChunks
        let partialHeadGroups = useSWAGQAPartial ? Int(numKVHeads) : Int(numQHeads)
        return AttentionSplitGeometry(effectiveLength: effectiveLength,
                                      numChunks: numChunks,
                                      chunkLength: chunkLength,
                                      partialThreadgroups: partialHeadGroups * numChunks,
                                      useSWAGroupedPartial: useSWAGQAPartial)
    }


    /// Sliding-window attention. `window` caps the K/V positions to the most
    /// recent `window` entries (`[max(0, seqLen-window), seqLen)`).
    /// `scale` defaults to `rsqrt(head_dim)` for generic callers;
    /// Gemma 4 callers pass 1.0 because the model's configured attention scale
    /// is 1.0.
    func encodeSWA(commandBuffer: MTLCommandBuffer,
                          q: MTLBuffer, qOffset: Int = 0,
                          k: MTLBuffer, kOffset: Int = 0,
                          v: MTLBuffer, vOffset: Int = 0,
                          out: MTLBuffer, outOffset: Int = 0,
                          headDim: UInt32,
                          numQHeads: UInt32,
                          numKVHeads: UInt32,
                          seqLen: UInt32,
                          window: UInt32,
                          scale: Float? = nil,
                          ringCapacity: UInt32 = 0) {
        precondition(numQHeads % numKVHeads == 0,
                     "numQHeads must be a multiple of numKVHeads for GQA")
        precondition(headDim <= 512,
                     "head_dim must be <= 512 (kernel scratch is sized for the full-attn case)")
        let sc = scale ?? Self.defaultScale(headDim: headDim)
        let kvStart = seqLen > window ? seqLen - window : 0

        encodeSplit(commandBuffer: commandBuffer,
                    q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                    v: v, vOffset: vOffset, out: out, outOffset: outOffset,
                    headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
                    seqLen: seqLen, kvStart: kvStart, scale: sc,
                    preferGQASWA: true,
                    ringCapacity: ringCapacity)
    }

    /// Full attention. Gemma 4 reuses the raw K projection as raw V input, but
    /// separate normalization and RoPE make the cache streams distinct here.
    /// `scale` mirrors `encodeSWA`.
    func encodeFull(commandBuffer: MTLCommandBuffer,
                           q: MTLBuffer, qOffset: Int = 0,
                           k: MTLBuffer, kOffset: Int = 0,
                           v: MTLBuffer, vOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           headDim: UInt32,
                           numQHeads: UInt32,
                           numKVHeads: UInt32,
                           seqLen: UInt32,
                           scale: Float? = nil) {
        precondition(numQHeads % numKVHeads == 0,
                     "numQHeads must be a multiple of numKVHeads for GQA")
        precondition(headDim <= 512,
                     "head_dim must be <= 512 (kernel scratch is sized for the full-attn case)")
        precondition(seqLen > 0, "full attention requires at least one KV position")
        let sc = scale ?? Self.defaultScale(headDim: headDim)


        encodeSplit(commandBuffer: commandBuffer,
                    q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                    v: v, vOffset: vOffset, out: out, outOffset: outOffset,
                    headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
                    seqLen: seqLen, kvStart: 0, scale: sc,
                    preferGQASWA: false)
    }


    // MARK: - Speculative block rows (docs/mtp/25-M5.6-RESULTS.md)

    /// Widest block the rows path takes. One SIMD group per row and a 256-thread
    /// ceiling put the bound at 8, which is `SpeculativeVerifier.maxTokens`.
    static let maxRows = 8
    /// Split factor ceiling for the rows path. `chunkCount` never returns more
    /// than 16 today; pinning it bounds the partial scratch at
    /// 8 rows x 16 heads x 16 chunks x 512 floats = 4 MB.
    static let rowsMaxChunks = 16

    /// Chunking for a block of `rows` queries at `startPosition`. `window == 0`
    /// means full attention. The chunks are cut over the *union* of the rows'
    /// ranges — row 0 has the earliest window start, the last row the latest
    /// history end — and the kernel clamps each row to its own range inside.
    static func rowsGeometry(rows: Int,
                             startPosition: Int,
                             window: UInt32,
                             numQHeads: UInt32) -> AttentionRowsGeometry {
        let firstSeqLen = UInt32(startPosition + 1)
        let lastSeqLen = UInt32(startPosition + rows)
        let kvStart = (window != 0 && firstSeqLen > window) ? firstSeqLen - window : 0
        let effectiveLength = Int(lastSeqLen - kvStart)
        let numChunks = max(1, min(rowsMaxChunks, effectiveLength))
        let chunkLength = (effectiveLength + numChunks - 1) / numChunks
        return AttentionRowsGeometry(kvStart: kvStart,
                                     effectiveLength: effectiveLength,
                                     numChunks: numChunks,
                                     chunkLength: chunkLength,
                                     partialThreadgroups: Int(numQHeads) * numChunks,
                                     partialThreadsPerGroup: rows * 32)
    }

    /// Attention for a speculative block's `rows` query rows in one dispatch
    /// pair. Row `r` attends to `[kvStart_r, startPosition + r + 1)`, which is
    /// exactly what `encodeSWA`/`encodeFull` compute one row at a time — this
    /// walks the KV range once for the whole block instead of `rows` times
    /// (docs/mtp/24-M5.5-RESULTS.md §7-1).
    ///
    /// `q` and `out` are `[rows, numQHeads, headDim]`; the offsets point at row 0.
    /// `window == 0` selects full attention (no lower bound).
    func encodeRows(commandBuffer: MTLCommandBuffer,
                    q: MTLBuffer, qOffset: Int = 0,
                    k: MTLBuffer, kOffset: Int = 0,
                    v: MTLBuffer, vOffset: Int = 0,
                    out: MTLBuffer, outOffset: Int = 0,
                    headDim: UInt32,
                    numQHeads: UInt32,
                    numKVHeads: UInt32,
                    rows: Int,
                    startPosition: Int,
                    window: UInt32,
                    scale: Float,
                    ringCapacity: UInt32 = 0) {
        precondition(rows >= 1 && rows <= Self.maxRows,
                     "rows \(rows) outside 1...\(Self.maxRows)")
        precondition(numQHeads % numKVHeads == 0,
                     "numQHeads must be a multiple of numKVHeads for GQA")
        precondition(headDim % 32 == 0 && headDim <= UInt32(Self.maxHeadDim),
                     "head_dim \(headDim) must be a multiple of 32 and <= \(Self.maxHeadDim)")
        precondition(Int(numQHeads) <= Self.maxQHeads,
                     "numQHeads \(numQHeads) exceeds split-KV scratch (max \(Self.maxQHeads))")
        precondition(startPosition >= 0, "startPosition must be non-negative")
        precondition(ringCapacity == 0 || window != 0,
                     "FP16 KV ring is only valid for SWA attention")

        let geometry = Self.rowsGeometry(rows: rows,
                                         startPosition: startPosition,
                                         window: window,
                                         numQHeads: numQHeads)
        let scratch = rowsScratch()
        let partialPSO = rowsPartialPipeline(headDim: headDim,
                                             numQHeads: numQHeads,
                                             numKVHeads: numKVHeads,
                                             numChunks: geometry.numChunks,
                                             ringCapacity: ringCapacity)

        guard let p1 = commandBuffer.makeComputeCommandEncoder() else { return }
        p1.setComputePipelineState(partialPSO)
        p1.setBuffer(q, offset: qOffset, index: 0)
        p1.setBuffer(k, offset: kOffset, index: 1)
        p1.setBuffer(v, offset: vOffset, index: 2)
        p1.setBuffer(scratch.m, offset: 0, index: 3)
        p1.setBuffer(scratch.d, offset: 0, index: 4)
        p1.setBuffer(scratch.o, offset: 0, index: 5)
        var hd = headDim, nq = numQHeads, nkv = numKVHeads
        var sl = UInt32(startPosition + 1), ks = geometry.kvStart
        var cl = UInt32(geometry.chunkLength), nc = UInt32(geometry.numChunks)
        var sc = scale, win = window
        p1.setBytes(&hd,  length: MemoryLayout<UInt32>.size, index: 6)
        p1.setBytes(&nq,  length: MemoryLayout<UInt32>.size, index: 7)
        p1.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 8)
        p1.setBytes(&sl,  length: MemoryLayout<UInt32>.size, index: 9)
        p1.setBytes(&ks,  length: MemoryLayout<UInt32>.size, index: 10)
        p1.setBytes(&cl,  length: MemoryLayout<UInt32>.size, index: 11)
        p1.setBytes(&nc,  length: MemoryLayout<UInt32>.size, index: 12)
        p1.setBytes(&sc,  length: MemoryLayout<Float>.size,  index: 13)
        p1.setBytes(&win, length: MemoryLayout<UInt32>.size, index: 14)
        p1.dispatchThreadgroups(
            MTLSize(width: geometry.partialThreadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: geometry.partialThreadsPerGroup,
                                           height: 1, depth: 1))
        p1.endEncoding()

        // The partial scratch is the decode layout with a leading row axis, so
        // the decode combine merges it unchanged at grid width rows * heads.
        guard let p2 = commandBuffer.makeComputeCommandEncoder() else { return }
        let combinePSO = combinePipeline(headDim: headDim,
                                         numQHeads: numQHeads,
                                         numKVHeads: numKVHeads,
                                         numChunks: geometry.numChunks)
        p2.setComputePipelineState(combinePSO)
        p2.setBuffer(scratch.m, offset: 0, index: 0)
        p2.setBuffer(scratch.d, offset: 0, index: 1)
        p2.setBuffer(scratch.o, offset: 0, index: 2)
        p2.setBuffer(out, offset: outOffset, index: 3)
        var hd2 = headDim, nc2 = UInt32(geometry.numChunks)
        p2.setBytes(&hd2, length: MemoryLayout<UInt32>.size, index: 4)
        p2.setBytes(&nc2, length: MemoryLayout<UInt32>.size, index: 5)
        let combineTGWidth = min(Self.threadsPerGroup,
                                 Int(combinePSO.maxTotalThreadsPerThreadgroup))
        p2.dispatchThreadgroups(
            MTLSize(width: rows * Int(numQHeads), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: combineTGWidth, height: 1, depth: 1))
        p2.endEncoding()
    }

    /// Partial scratch for the rows path, allocated on first use so a run that
    /// never verifies a block (MTP off) does not pay the 4 MB.
    private func rowsScratch() -> RowsScratch {
        if let existing = rowsScratchStorage { return existing }
        let entries = Self.maxRows * Self.maxQHeads * Self.rowsMaxChunks
        guard let m = ctx.device.makeBuffer(length: entries * MemoryLayout<Float>.size,
                                            options: .storageModeShared),
              let d = ctx.device.makeBuffer(length: entries * MemoryLayout<Float>.size,
                                            options: .storageModeShared),
              let o = ctx.device.makeBuffer(length: entries * Self.maxHeadDim * MemoryLayout<Float>.size,
                                            options: .storageModeShared) else {
            preconditionFailure("attention rows split-KV scratch allocation failed")
        }
        let scratch = RowsScratch(m: m, d: d, o: o)
        rowsScratchStorage = scratch
        return scratch
    }

    private func rowsPartialPipeline(headDim: UInt32,
                                     numQHeads: UInt32,
                                     numKVHeads: UInt32,
                                     numChunks: Int,
                                     ringCapacity: UInt32) -> MTLComputePipelineState {
        let key = RowsPipelineKey(headDim: headDim,
                                  numQHeads: numQHeads,
                                  numKVHeads: numKVHeads,
                                  numChunks: UInt32(numChunks),
                                  ringCapacity: ringCapacity)
        if let cached = rowsPipelineCache[key] { return cached }
        do {
            let pso = try Self.specializedPipeline(ctx,
                                                   "attention_rows_partial",
                                                   headDim: headDim,
                                                   numQHeads: numQHeads,
                                                   numKVHeads: numKVHeads,
                                                   numChunks: UInt32(numChunks),
                                                   ringCapacity: ringCapacity == 0 ? nil : ringCapacity)
            rowsPipelineCache[key] = pso
            return pso
        } catch {
            preconditionFailure("failed to build rows attention pipeline: \(error)")
        }
    }

    /// Two-pass split-KV (Flash-Decoding) dispatch shared by SWA and full
    /// attention — they differ only by `kvStart`. Pass 1 fans the head's
    /// `[kvStart, seqLen)` range across `chunkCount` threadgroups per head;
    /// pass 2 merges the partials. Both encoders go on the same command buffer
    /// so pass 2 hazard-tracks the partial scratch written by pass 1.
    private func encodeSplit(commandBuffer: MTLCommandBuffer,
                             q: MTLBuffer, qOffset: Int,
                             k: MTLBuffer, kOffset: Int,
                             v: MTLBuffer, vOffset: Int,
                             out: MTLBuffer, outOffset: Int,
                             headDim: UInt32, numQHeads: UInt32, numKVHeads: UInt32,
                             seqLen: UInt32, kvStart: UInt32, scale: Float,
                             preferGQASWA: Bool,
                             ringCapacity: UInt32 = 0) {
        precondition(Int(numQHeads) <= Self.maxQHeads,
                     "numQHeads \(numQHeads) exceeds split-KV scratch (max \(Self.maxQHeads))")
        precondition(Int(headDim) <= Self.maxHeadDim,
                     "head_dim \(headDim) exceeds split-KV scratch (max \(Self.maxHeadDim))")
        precondition(ringCapacity == 0 || preferGQASWA,
                     "FP16 KV ring is only valid for SWA attention")
        let geometry = Self.splitGeometry(numQHeads: numQHeads,
                                          numKVHeads: numKVHeads,
                                          seqLen: seqLen,
                                          kvStart: kvStart,
                                          preferGQASWA: preferGQASWA)
        let useSWAGQAPartial = geometry.useSWAGroupedPartial
        let nChunks = geometry.numChunks
        let chunkLen = geometry.chunkLength
        let partialPSO = partialPipeline(headDim: headDim,
                                         numQHeads: numQHeads,
                                         numKVHeads: numKVHeads,
                                         numChunks: nChunks,
                                         useGQAPartial: useSWAGQAPartial,
                                         ringCapacity: ringCapacity)
        let tgWidth = min(Self.threadsPerGroup, Int(partialPSO.maxTotalThreadsPerThreadgroup))

        guard let p1 = commandBuffer.makeComputeCommandEncoder() else { return }
        p1.setComputePipelineState(partialPSO)
        p1.setBuffer(q, offset: qOffset, index: 0)
        p1.setBuffer(k, offset: kOffset, index: 1)
        p1.setBuffer(v, offset: vOffset, index: 2)
        p1.setBuffer(mPartial, offset: 0, index: 3)
        p1.setBuffer(dPartial, offset: 0, index: 4)
        p1.setBuffer(oPartial, offset: 0, index: 5)
        var hd = headDim, nq = numQHeads, nkv = numKVHeads, sl = seqLen, ks = kvStart
        var cl = UInt32(chunkLen), nc = UInt32(nChunks), sc = scale
        p1.setBytes(&hd,  length: MemoryLayout<UInt32>.size, index: 6)
        p1.setBytes(&nq,  length: MemoryLayout<UInt32>.size, index: 7)
        p1.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 8)
        p1.setBytes(&sl,  length: MemoryLayout<UInt32>.size, index: 9)
        p1.setBytes(&ks,  length: MemoryLayout<UInt32>.size, index: 10)
        p1.setBytes(&cl,  length: MemoryLayout<UInt32>.size, index: 11)
        p1.setBytes(&nc,  length: MemoryLayout<UInt32>.size, index: 12)
        p1.setBytes(&sc,  length: MemoryLayout<Float>.size,  index: 13)
        let partialGroups = geometry.partialThreadgroups
        p1.dispatchThreadgroups(MTLSize(width: partialGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        p1.endEncoding()

        guard let p2 = commandBuffer.makeComputeCommandEncoder() else { return }
        let combinePSO = combinePipeline(headDim: headDim,
                                         numQHeads: numQHeads,
                                         numKVHeads: numKVHeads,
                                         numChunks: nChunks)
        p2.setComputePipelineState(combinePSO)
        p2.setBuffer(mPartial, offset: 0, index: 0)
        p2.setBuffer(dPartial, offset: 0, index: 1)
        p2.setBuffer(oPartial, offset: 0, index: 2)
        p2.setBuffer(out, offset: outOffset, index: 3)
        var hd2 = headDim, nc2 = UInt32(nChunks)
        p2.setBytes(&hd2, length: MemoryLayout<UInt32>.size, index: 4)
        p2.setBytes(&nc2, length: MemoryLayout<UInt32>.size, index: 5)
        let combineTGWidth = min(Self.threadsPerGroup,
                                 Int(combinePSO.maxTotalThreadsPerThreadgroup))
        p2.dispatchThreadgroups(MTLSize(width: Int(numQHeads), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: combineTGWidth, height: 1, depth: 1))
        p2.endEncoding()
    }

    /// `1 / sqrt(head_dim)` — the classic transformer scaling. Used as the
    /// default for non-Gemma callers (and for the existing tests that pre-date
    /// the runtime scale arg).
    static func defaultScale(headDim: UInt32) -> Float {
        Float(1.0) / Float(headDim).squareRoot()
    }

    private static func specializedPipeline(_ context: MetalContext,
                                            _ name: String,
                                            headDim: UInt32,
                                            numQHeads: UInt32,
                                            numKVHeads: UInt32,
                                            numChunks: UInt32? = nil,
                                            ringCapacity: UInt32? = nil) throws -> MTLComputePipelineState {
        var constants = [
            MetalFunctionConstant(index: 60, value: .uint32(headDim)),
            MetalFunctionConstant(index: 61, value: .uint32(numQHeads)),
            MetalFunctionConstant(index: 62, value: .uint32(numKVHeads)),
            MetalFunctionConstant(index: 63, value: .bool(true)),
        ]
        if let numChunks {
            constants.append(MetalFunctionConstant(index: 65, value: .uint32(numChunks)))
        }
        if let ringCapacity {
            constants.append(MetalFunctionConstant(index: 69, value: .uint32(ringCapacity)))
        }
        return try context.pipeline(name, constants: constants)
    }

    private func partialPipeline(headDim: UInt32,
                                 numQHeads: UInt32,
                                 numKVHeads: UInt32,
                                 numChunks: Int,
                                 useGQAPartial: Bool,
                                 ringCapacity: UInt32 = 0) -> MTLComputePipelineState {
        if ringCapacity > 0 {
            let name = useGQAPartial ? "attention_decode_gqa_swa_partial" : "attention_decode_partial"
            let specializedChunks = numChunks == 16 ? Optional(UInt32(numChunks)) : nil
            do {
                return try Self.specializedPipeline(ctx,
                                                    name,
                                                    headDim: headDim,
                                                    numQHeads: numQHeads,
                                                    numKVHeads: numKVHeads,
                                                    numChunks: specializedChunks,
                                                    ringCapacity: ringCapacity)
            } catch {
                preconditionFailure("failed to build FP16 KV ring attention pipeline: \(error)")
            }
        }
        if useGQAPartial && headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            if numChunks == 16 {
                return psoGQAPartialSWAChunks16
            }
            return psoGQAPartialSWA
        }
        if !useGQAPartial && headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            return psoPartialSWA
        }
        if !useGQAPartial && headDim == 512 && numQHeads == 16 && numKVHeads == 2 {
            if numChunks == 16 {
                return psoPartialFullChunks16
            }
            return psoPartialFull
        }
        return useGQAPartial ? psoGQAPartial : psoPartial
    }

    private func combinePipeline(headDim: UInt32,
                                 numQHeads: UInt32,
                                 numKVHeads: UInt32,
                                 numChunks: Int) -> MTLComputePipelineState {
        if headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            return numChunks == 16 ? psoCombineSWAChunks16 : psoCombineSWA
        }
        if headDim == 512 && numQHeads == 16 && numKVHeads == 2 {
            return numChunks == 16 ? psoCombineFullChunks16 : psoCombineFull
        }
        return psoCombine
    }
}
