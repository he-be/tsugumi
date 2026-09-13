import Foundation
import Metal
import MetalPerformanceShaders

/// Gated DeltaNet step for a T-token batch in chunks of `chunk` tokens (WY form, `q38_wy_*` in qwen38.metal):
/// the same math as `q38_gdn_step`, but the state is read and written once per chunk instead of once per
/// token. Inside a chunk the tokens are coupled through the inverse of an L x L unit lower-triangular matrix and sgemm,
/// all batched over the value heads (docs/qwen38/07).
///
/// Inputs are the runner's buffers after `q38_gdn_qk_norm` / `q38_gdn_gates`: `conv` rows of
/// (q[hk], k[hk], v[hv]) x d, decay `a` and `b` (beta) per (t, hv), state [hv][d][d] (updated in place),
/// output [t][hv][d].
package final class Qwen38GDNChunk {
    package let chunk: Int
    private let device: MTLDevice
    private let hk: Int, hv: Int, d: Int
    private let psoGather, psoTri, psoRhs, psoOut, psoKw, psoDecay, psoTinv: MTLComputePipelineState
    private var muls: [[Int]: MPSMatrixMultiplication] = [:]
    private let kq, k, gram, A, X, M, gam, w, XY, R, U, MU: MTLBuffer

    package init(device: MTLDevice, library: MTLLibrary, keyHeads: Int, valueHeads: Int, headDim: Int, chunk: Int) throws {
        self.device = device
        self.chunk = chunk
        hk = keyHeads
        hv = valueHeads
        d = headDim
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else { throw GGUFFile.Error.format("kernel \(name) missing") }
            return try device.makeComputePipelineState(function: fn)
        }
        psoGather = try pso("q38_wy_gather")
        psoTri = try pso("q38_wy_tri")
        psoRhs = try pso("q38_wy_rhs")
        psoOut = try pso("q38_wy_out")
        psoKw = try pso("q38_wy_kw")
        psoDecay = try pso("q38_wy_decay_state")
        psoTinv = try pso("q38_wy_tinv")
        precondition(chunk <= 512, "q38_wy_tinv keeps a column of at most 512")
        let L = chunk
        func buf(_ count: Int) -> MTLBuffer { device.makeBuffer(length: count * 4, options: .storageModeShared)! }
        kq = buf(hv * 2 * L * d); k = buf(hv * L * d); gram = buf(hv * 2 * L * L)
        A = buf(hv * L * L); X = buf(hv * L * L); M = buf(hv * L * L); gam = buf(hv * L); w = buf(hv * L)
        XY = buf(hv * 2 * L * d); R = buf(hv * L * d); U = buf(hv * L * d); MU = buf(hv * L * d)
    }

    private func mul(_ key: [Int], rows: Int, cols: Int, interior: Int, transposeLeft: Bool = false,
                     transposeRight: Bool = false, beta: Double = 0) -> MPSMatrixMultiplication {
        if let m = muls[key] { return m }
        let m = MPSMatrixMultiplication(device: device, transposeLeft: transposeLeft, transposeRight: transposeRight,
                                        resultRows: rows, resultColumns: cols, interiorColumns: interior,
                                        alpha: 1, beta: beta)
        m.batchStart = 0
        m.batchSize = hv
        muls[key] = m
        return m
    }

    private func matrix(_ b: MTLBuffer, rows: Int, cols: Int) -> MPSMatrix {
        MPSMatrix(buffer: b, descriptor: MPSMatrixDescriptor(rows: rows, columns: cols, matrices: hv,
                                                             rowBytes: cols * 4, matrixBytes: rows * cols * 4, dataType: .float32))
    }

    /// `section` (benches only) commits the work so far under a label and returns the buffer to continue in.
    package func encode(_ cb0: MTLCommandBuffer, conv: MTLBuffer, a: MTLBuffer, b: MTLBuffer,
                        state: MTLBuffer, out: MTLBuffer, T: Int,
                        section: ((String, MTLCommandBuffer) -> MTLCommandBuffer)? = nil) {
        var cb = cb0
        func mark(_ label: String) { if let section { cb = section(label, cb) } }
        let C = (2 * hk + hv) * d
        for t0 in stride(from: 0, to: T, by: chunk) {
            autoreleasepool {
                let L = min(chunk, T - t0)
                var p = (UInt32(hk), UInt32(hv), UInt32(d), UInt32(L), UInt32(t0), UInt32(C))
                let pLen = MemoryLayout.size(ofValue: p)
                let grid = MTLSize(width: d, height: L, depth: hv)
                let tg = MTLSize(width: 16, height: 16, depth: 1)

                var enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(psoGather)
                enc.setBuffer(conv, offset: 0, index: 0)
                enc.setBuffer(kq, offset: 0, index: 1)
                enc.setBuffer(k, offset: 0, index: 2)
                enc.setBytes(&p, length: pLen, index: 3)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
                enc.endEncoding()
                mark("gather")

                let kqM = matrix(kq, rows: 2 * L, cols: d)
                let S = matrix(state, rows: d, cols: d)
                mul([0, L], rows: 2 * L, cols: L, interior: d, transposeRight: true)
                    .encode(commandBuffer: cb, leftMatrix: kqM, rightMatrix: matrix(k, rows: L, cols: d),
                            resultMatrix: matrix(gram, rows: 2 * L, cols: L))
                mul([1, L], rows: 2 * L, cols: d, interior: d, transposeRight: true)
                    .encode(commandBuffer: cb, leftMatrix: kqM, rightMatrix: S, resultMatrix: matrix(XY, rows: 2 * L, cols: d))
                mark("gram+XY")

                enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(psoTri)
                enc.setBuffer(a, offset: 0, index: 0)
                enc.setBuffer(b, offset: 0, index: 1)
                enc.setBuffer(gram, offset: 0, index: 2)
                enc.setBuffer(A, offset: 0, index: 3)
                enc.setBuffer(M, offset: 0, index: 4)
                enc.setBuffer(gam, offset: 0, index: 5)
                enc.setBuffer(w, offset: 0, index: 6)
                enc.setBytes(&p, length: pLen, index: 7)
                enc.dispatchThreads(MTLSize(width: L, height: hv, depth: 1), threadsPerThreadgroup: tg)
                enc.setComputePipelineState(psoRhs)
                enc.setBuffer(conv, offset: 0, index: 0)
                enc.setBuffer(b, offset: 0, index: 1)
                enc.setBuffer(gam, offset: 0, index: 2)
                enc.setBuffer(XY, offset: 0, index: 3)
                enc.setBuffer(R, offset: 0, index: 4)
                enc.setBytes(&p, length: pLen, index: 5)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
                enc.setComputePipelineState(psoTinv)
                enc.setBuffer(A, offset: 0, index: 0)
                enc.setBuffer(X, offset: 0, index: 1)
                enc.setBytes(&p, length: pLen, index: 2)
                enc.dispatchThreads(MTLSize(width: L, height: hv, depth: 1), threadsPerThreadgroup: tg)
                enc.endEncoding()
                mark("tri+rhs+tinv")

                mul([4, L], rows: L, cols: d, interior: L)
                    .encode(commandBuffer: cb, leftMatrix: matrix(X, rows: L, cols: L), rightMatrix: matrix(R, rows: L, cols: d),
                            resultMatrix: matrix(U, rows: L, cols: d))
                mark("U")
                let UM = matrix(U, rows: L, cols: d)
                mul([2, L], rows: L, cols: d, interior: L)
                    .encode(commandBuffer: cb, leftMatrix: matrix(M, rows: L, cols: L), rightMatrix: UM,
                            resultMatrix: matrix(MU, rows: L, cols: d))
                mark("MU")

                enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(psoOut)
                enc.setBuffer(MU, offset: 0, index: 0)
                enc.setBuffer(XY, offset: 0, index: 1)
                enc.setBuffer(gam, offset: 0, index: 2)
                enc.setBuffer(out, offset: 0, index: 3)
                enc.setBytes(&p, length: pLen, index: 4)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
                enc.setComputePipelineState(psoKw)
                enc.setBuffer(k, offset: 0, index: 0)
                enc.setBuffer(w, offset: 0, index: 1)
                enc.setBytes(&p, length: pLen, index: 2)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
                enc.setComputePipelineState(psoDecay)
                enc.setBuffer(state, offset: 0, index: 0)
                enc.setBuffer(gam, offset: 0, index: 1)
                enc.setBytes(&p, length: pLen, index: 2)
                enc.dispatchThreads(MTLSize(width: d, height: d, depth: hv), threadsPerThreadgroup: tg)
                enc.endEncoding()
                mark("out+kw+decay")

                mul([3, L], rows: d, cols: d, interior: L, transposeLeft: true, beta: 1)
                    .encode(commandBuffer: cb, leftMatrix: UM, rightMatrix: matrix(k, rows: L, cols: d), resultMatrix: S)
                mark("state")
            }
        }
    }
}
