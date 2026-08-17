import Foundation
import TurboFieldfare

/// Float32 CPU references for the Gemma 4 vision tower stages, written from
/// `transformers` v5.6.2 `models/gemma4/modeling_gemma4.py` rather than from
/// the Metal kernels, so that a kernel and its reference cannot share a
/// misreading of the upstream algorithm.
///
/// Every entry point takes a `Variant` that names a deliberate deviation. They
/// exist so a check can prove it would fail: a comparison that has never been
/// seen to fail is not evidence (`PLAN_VISION.md` §6-3).
public enum VisionTowerRef {
    /// `Y[t, n] = sum_k W[n, k] * X[t, k]`, weights row-major `[N, K]`.
    public static func matmul(weights: [Float], x: [Float],
                              t: Int, n: Int, k: Int) -> [Float] {
        precondition(weights.count == n * k && x.count == t * k, "shape mismatch")
        var out = [Float](repeating: 0, count: t * n)
        for row in 0..<t {
            for col in 0..<n {
                var acc: Float = 0
                for i in 0..<k {
                    acc += weights[col * k + i] * x[row * k + i]
                }
                out[row * n + col] = acc
            }
        }
        return out
    }

    /// `2 * (pixels - 0.5)` -> projection -> `+ table[0, x] + table[1, y]`.
    public static func patchEmbed(pixels: [Float], projection: [Float],
                                  positionTable: [Float],
                                  patchCount: Int, patchDim: Int, hidden: Int,
                                  patchesWide: Int, tableLength: Int) -> [Float] {
        let scaled = pixels.map { 2 * ($0 - 0.5) }
        var out = matmul(weights: projection, x: scaled,
                         t: patchCount, n: hidden, k: patchDim)
        for patch in 0..<patchCount {
            let x = patch % patchesWide
            let y = patch / patchesWide
            for d in 0..<hidden {
                out[patch * hidden + d] += positionTable[x * hidden + d]
                    + positionTable[(tableLength + y) * hidden + d]
            }
        }
        return out
    }

    public static func rmsNorm(_ row: [Float], weight: [Float]?, eps: Float) -> [Float] {
        var meanSquare: Float = 0
        for value in row { meanSquare += value * value }
        meanSquare = meanSquare / Float(row.count) + eps
        let inv = pow(meanSquare, -0.5)
        return row.enumerated().map { index, value in
            value * inv * (weight?[index] ?? 1)
        }
    }

    /// What a check is allowed to break on purpose.
    public enum Variant: String, Sendable {
        /// The upstream algorithm.
        case upstream
        /// 2D RoPE with the x and y positions exchanged. Every patch off the
        /// grid diagonal moves.
        case ropeAxesSwapped
        /// Pooling cells emitted column-major instead of row-major.
        case poolColumnMajor
        /// The normalized branch output replaces the residual stream instead of
        /// being added to it — what a wrong buffer binding on the fused
        /// norm-and-add would produce.
        case residualDropped
    }

    /// `hidden[t] += rmsnorm(x[t]) * weight`, the tower layer's residual joins.
    public static func normResidualAdd(hidden: [Float], x: [Float], weight: [Float],
                                       t: Int, d: Int, eps: Float,
                                       variant: Variant = .upstream) -> [Float] {
        precondition(hidden.count == t * d && x.count == t * d && weight.count == d,
                     "shape mismatch")
        var out = hidden
        for row in 0..<t {
            let range = (row * d)..<((row + 1) * d)
            let normed = rmsNorm(Array(x[range]), weight: weight, eps: eps)
            for i in 0..<d {
                out[row * d + i] = variant == .residualDropped
                    ? normed[i]
                    : hidden[row * d + i] + normed[i]
            }
        }
        return out
    }

    /// Per-head Q/K RMSNorm + 2D RoPE, and V's scale-less RMSNorm.
    ///
    /// The head dimension splits into a first half rotated by the patch's x
    /// position and a second half rotated by its y, both using
    /// `inv_freq[j] = theta^(-2j / halfDim)` over `j = 0 ..< halfDim/2`, and
    /// each half is a NeoX rotation of the pairs `(i, i + halfDim/2)`.
    public static func qkNormRoPE2D(q: [Float], k: [Float], v: [Float],
                                    qWeight: [Float], kWeight: [Float],
                                    patchCount: Int, headDim: Int, numHeads: Int,
                                    patchesWide: Int, theta: Float, eps: Float,
                                    variant: Variant = .upstream)
        -> (q: [Float], k: [Float], v: [Float]) {
        let stride = numHeads * headDim
        let halfDim = headDim / 2
        let pairs = halfDim / 2

        func rope(_ row: [Float], x: Float, y: Float) -> [Float] {
            var out = row
            let first = variant == .ropeAxesSwapped ? y : x
            let second = variant == .ropeAxesSwapped ? x : y
            for dim in 0..<2 {
                let position = dim == 0 ? first : second
                for j in 0..<pairs {
                    let i0 = dim * halfDim + j
                    let i1 = i0 + pairs
                    let freq = pow(theta, -Float(2 * j) / Float(halfDim))
                    let angle = position * freq
                    let c = cos(angle)
                    let s = sin(angle)
                    out[i0] = row[i0] * c - row[i1] * s
                    out[i1] = row[i1] * c + row[i0] * s
                }
            }
            return out
        }

        var outQ = q
        var outK = k
        var outV = v
        for patch in 0..<patchCount {
            let x = Float(patch % patchesWide)
            let y = Float(patch / patchesWide)
            for head in 0..<numHeads {
                let base = patch * stride + head * headDim
                let range = base..<(base + headDim)
                let normedQ = rmsNorm(Array(q[range]), weight: qWeight, eps: eps)
                let normedK = rmsNorm(Array(k[range]), weight: kWeight, eps: eps)
                let normedV = rmsNorm(Array(v[range]), weight: nil, eps: eps)
                outQ.replaceSubrange(range, with: rope(normedQ, x: x, y: y))
                outK.replaceSubrange(range, with: rope(normedK, x: x, y: y))
                outV.replaceSubrange(range, with: normedV)
            }
        }
        return (outQ, outK, outV)
    }

    /// Non-causal attention over all patches, softmax in float32.
    public static func attentionFull(q: [Float], k: [Float], v: [Float],
                                     patchCount: Int, headDim: Int, numHeads: Int,
                                     scale: Float) -> [Float] {
        let stride = numHeads * headDim
        var out = [Float](repeating: 0, count: patchCount * stride)
        for head in 0..<numHeads {
            for query in 0..<patchCount {
                let qBase = query * stride + head * headDim
                var scores = [Float](repeating: 0, count: patchCount)
                var maxScore = -Float.infinity
                for key in 0..<patchCount {
                    let kBase = key * stride + head * headDim
                    var acc: Float = 0
                    for d in 0..<headDim {
                        acc += q[qBase + d] * k[kBase + d]
                    }
                    scores[key] = acc * scale
                    maxScore = max(maxScore, scores[key])
                }
                var total: Float = 0
                for key in 0..<patchCount {
                    scores[key] = exp(scores[key] - maxScore)
                    total += scores[key]
                }
                for key in 0..<patchCount {
                    let weight = scores[key] / total
                    let vBase = key * stride + head * headDim
                    for d in 0..<headDim {
                        out[qBase + d] += weight * v[vBase + d]
                    }
                }
            }
        }
        return out
    }

    public static func geluTanh(_ x: Float) -> Float {
        let inner = Float(0.7978845608028654) * (x + Float(0.044715) * x * x * x)
        return 0.5 * x * (1 + tanh(min(max(inner, -20), 20)))
    }

    public static func geluMultiply(gate: [Float], up: [Float]) -> [Float] {
        precondition(gate.count == up.count, "shape mismatch")
        return zip(gate, up).map { geluTanh($0) * $1 }
    }

    /// `k x k` average pool over the patch grid, then `* sqrt(hidden)`, then
    /// `(x - stdBias) * stdScale`.
    public static func poolStandardize(hidden: [Float], d: Int,
                                       patchesWide: Int, patchesHigh: Int,
                                       kernelSize: Int,
                                       stdScale: [Float], stdBias: [Float],
                                       standardize: Bool,
                                       variant: Variant = .upstream) -> [Float] {
        let pooledW = patchesWide / kernelSize
        let pooledH = patchesHigh / kernelSize
        let rootHidden = Float(d).squareRoot()
        var out = [Float](repeating: 0, count: pooledW * pooledH * d)
        for cy in 0..<pooledH {
            for cx in 0..<pooledW {
                let cell = variant == .poolColumnMajor
                    ? cx * pooledH + cy
                    : cy * pooledW + cx
                for dim in 0..<d {
                    var acc: Float = 0
                    for dy in 0..<kernelSize {
                        for dx in 0..<kernelSize {
                            let patch = (cy * kernelSize + dy) * patchesWide
                                + (cx * kernelSize + dx)
                            acc += hidden[patch * d + dim]
                        }
                    }
                    var value = acc / Float(kernelSize * kernelSize) * rootHidden
                    if standardize {
                        value = (value - stdBias[dim]) * stdScale[dim]
                    }
                    out[cell * d + dim] = value
                }
            }
        }
        return out
    }
}

/// BF16 test data: the value the GPU will read and the float it rounds to, kept
/// together so a reference never compares against un-rounded weights.
public struct BF16Tensor {
    public let bits: [UInt16]
    public let values: [Float]

    public init(_ values: [Float]) {
        let bits = values.map { Quantization.bf16Bits($0) }
        self.bits = bits
        self.values = bits.map { Quantization.bf16ToFloat($0) }
    }
}
