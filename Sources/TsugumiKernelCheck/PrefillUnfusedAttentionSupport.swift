import Foundation
import Metal
import Tsugumi

/// `PrefillUnfusedParams` in prefill.metal.
private struct UnfusedParams { var headBase: UInt32; var scoreStride: UInt32; var headRows: UInt32; var ring: UInt32 }

/// The unfused prefill attention (`attention_prefill_unfused_scores_d<D>` →
/// `_softmax` → `_pv_d<D>`, docs/m6-prefill/36・37) over query-head groups of
/// `group`, dispatched the way `PrefillAttention.encodeUnfused` does: each
/// 64-row block keeps `scoreStride` keys from its band start. `params` is the
/// check's copy of `PrefillAttentionParams`.
func encodeUnfusedPrefillAttention<Params>(
    context: MetalContext, commandBuffer: MTLCommandBuffer,
    q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, out: MTLBuffer,
    params: Params, headDim: Int, heads: Int, rows: Int, kvValidCount: Int,
    window: Int, ring: Int, group: Int) throws {
    let scores = try context.pipeline("attention_prefill_unfused_scores_d\(headDim)")
    let softmax = try context.pipeline("attention_prefill_unfused_softmax")
    let pv = try context.pipeline("attention_prefill_unfused_pv_d\(headDim)")
    var stride = (kvValidCount + 63) / 64 * 64
    if window > 0 && window < kvValidCount {
        stride = min(stride, (window + 127 + 63) / 64 * 64)
    }
    let sBytes = group * rows * stride * 4, pBytes = group * rows * stride * 2
    guard let scratch = context.device.makeBuffer(length: sBytes + pBytes + group * rows * 4,
                                                  options: .storageModePrivate),
          let enc = commandBuffer.makeComputeCommandEncoder() else {
        fatalError("unfused prefill attention allocation failed")
    }
    var p = params
    let rowTiles = (rows + 63) / 64
    let threads = MTLSize(width: 128, height: 1, depth: 1)
    for headBase in Swift.stride(from: 0, to: heads, by: group) {
        var u = UnfusedParams(headBase: UInt32(headBase), scoreStride: UInt32(stride),
                              headRows: UInt32(rows), ring: UInt32(ring))
        enc.setComputePipelineState(scores)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(k, offset: 0, index: 1)
        enc.setBuffer(scratch, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 4)
        enc.setBytes(&u, length: MemoryLayout<UnfusedParams>.stride, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: stride / 64, height: rowTiles, depth: group),
                                 threadsPerThreadgroup: threads)
        enc.setComputePipelineState(softmax)
        enc.setBuffer(scratch, offset: sBytes, index: 3)
        enc.setBuffer(scratch, offset: sBytes + pBytes, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: (rows + 3) / 4, height: group, depth: 1),
                                 threadsPerThreadgroup: threads)
        enc.setComputePipelineState(pv)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.setBuffer(v, offset: 0, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: headDim / 64, height: rowTiles, depth: group),
                                 threadsPerThreadgroup: threads)
    }
    enc.endEncoding()
}
