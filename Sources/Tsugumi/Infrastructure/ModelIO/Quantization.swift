import Foundation
import MoEPackFormat

public enum Quantization {

    /// Legacy/default affine group size. The *authoritative* value for a loaded
    /// model is `Manifest.quant.embedding.groupSize`; this constant is the
    /// fallback used by fixtures and by callers that predate group-32 support.
    /// See `MetalContext.affineGroupSize` for the shader-side counterpart.
    public static let groupSize: Int = 64

    /// Group sizes the runtime can compile shaders for. The vectorized INT4
    /// GEMV blocks read 128 bytes per SIMD group, so a group must be a whole
    /// number of lanes wide (`groupSize / 8` lanes) and must divide 128 bytes
    /// evenly — 32 and 64 both satisfy this.
    public static let supportedGroupSizes: Set<Int> = [32, 64]

    /// How a 4-bit group encodes its zero point.
    ///
    /// Like the group size, this is a whole-model property baked into the
    /// shader library (`MetalContext.affineScheme`): one process serves one
    /// model, and the MTP drafter -- which is packed `affine` at group 64 while
    /// the target is `sym` at group 32 -- runs on its own context.
    public enum AffineScheme: String, Sendable, CaseIterable {
        /// `w = q * scale + bias`, with a BF16 bias stored per group.
        case affine
        /// `w = scale * (q - 8)`. The bias array is not stored.
        ///
        /// A lattice-aligned QAT checkpoint satisfies `bias == -8 * scale` as a
        /// BF16 *bit pattern* in every group -- verified over all 788,175,872
        /// groups of the shipped checkpoint, against 34-38% of groups failing
        /// on an ordinarily quantized one (`docs/mtp/44-W1-WEIGHT-DIET.md` §1).
        /// Dropping the array takes the routed expert from 5.000 to 4.500 bits
        /// per weight and is exact, not an approximation.
        case sym

        /// Whether a group's zero point is stored alongside its scale.
        public var storesBias: Bool { self == .affine }
    }

    /// `-8 * scale` as a BF16 bit pattern, exactly.
    ///
    /// `-8` is a power of two, so the product is a pure exponent shift and BF16
    /// carries it without rounding. This is what makes the identity checkable
    /// on bits rather than within a tolerance -- and what lets the shader
    /// reconstruct the bias with a single multiply.
    /// Shared with the packer through `MoEPackFormat` so the runtime and
    /// the repacker cannot drift on the one identity `sym` rests on.
    @inline(__always)
    public static func symmetricBiasBits(_ scaleBits: UInt16) -> UInt16 {
        MoEPackAffineV1.symmetricBiasBits(scaleBits)
    }

    /// Whether every group in a scale/bias pair satisfies `bias == -8 * scale`.
    /// Both arrays carry BF16 bit patterns and must be the same length.
    public static func isSymmetric(scaleBits: UnsafeRawBufferPointer,
                                   biasBits: UnsafeRawBufferPointer) -> Bool {
        MoEPackAffineV1.isSymmetric(scaleBits: scaleBits, biasBits: biasBits)
    }

    // MARK: - BF16 helpers
    //
    // BF16 = top 16 bits of FP32 (8-bit exponent, 7-bit mantissa). Stored on
    // disk and on the GPU as the native `bfloat` type; in Swift we carry the
    // raw bits as `UInt16` and convert via this pair. Round-half-to-even on
    // encode matches the IEEE-754 default. No NaN/Inf special-case: fixtures
    // are bounded and finite, and the .moepack importer copies BF16 bytes
    // through unchanged so we never call the encoder on weight data.

    @inline(__always)
    public static func bf16Bits(_ x: Float) -> UInt16 {
        let bits = x.bitPattern
        let lsb  = (bits >> 16) & 1
        let roundingBias: UInt32 = 0x7FFF &+ lsb
        let rounded = (bits &+ roundingBias) >> 16
        return UInt16(truncatingIfNeeded: rounded)
    }

    @inline(__always)
    public static func bf16ToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    // MARK: - INT4 affine

    /// MLX `affine` 4-bit row. Packed unsigned nibbles, BF16 scale + bias per
    /// group of 64. `scales` and `biases` carry BF16 bit patterns as `UInt16`
    /// so the same buffer can be uploaded to a Metal `device const bfloat*`.
    public struct Int4AffineRow {
        public let packed: [UInt8]   // N / 2 bytes; low nibble = even index, high = odd
        public let scales: [UInt16]  // N / 64 BF16 bits
        public let biases: [UInt16]  // N / 64 BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    /// Affine 4-bit quantize: `q ∈ [0..15]`, `w ≈ q * scale + bias`.
    /// Scale and bias are computed from per-group min/max, then rounded to BF16.
    /// Test-fixture only — the runtime importer never calls this.
    public static func quantizeInt4Affine(_ row: [Float],
                                          groupSize: Int = Quantization.groupSize)
        -> Int4AffineRow {
        precondition(row.count % groupSize == 0,
                     "row length \(row.count) is not a multiple of \(groupSize)")

        let nGroups = row.count / groupSize
        var packed = [UInt8](repeating: 0, count: row.count / 2)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            // Constant group: scale=1, bias=value preserves exact reconstruction.
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / 15.0
                biasF  = wmin
            }
            // Round through BF16 first, then quantize against the rounded
            // values so the runtime decode (which reads BF16) reproduces the
            // same q the fixture stored.
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias  = bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                var q = Int(((w - bias) * invScale).rounded())
                q = max(0, min(15, q))
                let nibble = UInt8(q) & 0x0F
                let byteIdx = g * (groupSize / 2) + (k / 2)
                if (k & 1) == 0 {
                    packed[byteIdx] = (packed[byteIdx] & 0xF0) | nibble
                } else {
                    packed[byteIdx] = (packed[byteIdx] & 0x0F) | (nibble << 4)
                }
            }
        }
        return Int4AffineRow(packed: packed, scales: scales, biases: biases)
    }

    /// Symmetric 4-bit quantize: `q ∈ [0..15]`, `w ≈ scale * (q - 8)`.
    ///
    /// Produces the row a lattice-aligned QAT checkpoint carries -- the bias is
    /// `-8 * scale` in every group, so `dequantizeInt4Affine` reads it back
    /// unchanged and the `sym` shader library must reproduce the same values
    /// without ever loading the bias array. Test-fixture only.
    public static func quantizeInt4Symmetric(_ row: [Float],
                                             groupSize: Int = Quantization.groupSize)
        -> Int4AffineRow {
        precondition(row.count % groupSize == 0,
                     "row length \(row.count) is not a multiple of \(groupSize)")

        let nGroups = row.count / groupSize
        var packed = [UInt8](repeating: 0, count: row.count / 2)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            // The lattice is asymmetric: codes run -8...+7 around zero, so the
            // scale has to clear whichever end is further out in *its* units.
            let scaleF = max(max(-wmin / 8, wmax / 7), Float.leastNormalMagnitude)
            let sBits = bf16Bits(scaleF)
            scales[g] = sBits
            biases[g] = symmetricBiasBits(sBits)
            let scale = bf16ToFloat(sBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                var q = Int((w * invScale).rounded()) + 8
                q = max(0, min(15, q))
                let nibble = UInt8(q) & 0x0F
                let byteIdx = g * (groupSize / 2) + (k / 2)
                if (k & 1) == 0 {
                    packed[byteIdx] = (packed[byteIdx] & 0xF0) | nibble
                } else {
                    packed[byteIdx] = (packed[byteIdx] & 0x0F) | (nibble << 4)
                }
            }
        }
        return Int4AffineRow(packed: packed, scales: scales, biases: biases)
    }

    public static func dequantizeInt4Affine(_ r: Int4AffineRow, n: Int,
                                            groupSize: Int = Quantization.groupSize)
        -> [Float] {
        precondition(n == r.packed.count * 2)
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / groupSize
        for g in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[g])
            let bias  = bf16ToFloat(r.biases[g])
            for k in 0..<groupSize {
                let byteIdx = g * (groupSize / 2) + (k / 2)
                let b = r.packed[byteIdx]
                let nibble: Int = (k & 1) == 0 ? Int(b & 0x0F) : Int(b >> 4)
                out[g * groupSize + k] = Float(nibble) * scale + bias
            }
        }
        return out
    }

    // MARK: - INT8 affine

    public struct Int8AffineRow {
        public let packed: [UInt8]   // N unsigned bytes
        public let scales: [UInt16]  // N / 64 BF16 bits
        public let biases: [UInt16]  // N / 64 BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    public static func quantizeInt8Affine(_ row: [Float],
                                          groupSize: Int = Quantization.groupSize)
        -> Int8AffineRow {
        precondition(row.count % groupSize == 0,
                     "row length \(row.count) is not a multiple of \(groupSize)")

        let nGroups = row.count / groupSize
        var packed = [UInt8](repeating: 0, count: row.count)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / 255.0
                biasF  = wmin
            }
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias  = bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                var q = Int(((w - bias) * invScale).rounded())
                q = max(0, min(255, q))
                packed[g * groupSize + k] = UInt8(q)
            }
        }
        return Int8AffineRow(packed: packed, scales: scales, biases: biases)
    }

    public static func dequantizeInt8Affine(_ r: Int8AffineRow, n: Int,
                                            groupSize: Int = Quantization.groupSize)
        -> [Float] {
        precondition(n == r.packed.count)
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / groupSize
        for g in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[g])
            let bias  = bf16ToFloat(r.biases[g])
            for k in 0..<groupSize {
                out[g * groupSize + k] = Float(r.packed[g * groupSize + k]) * scale + bias
            }
        }
        return out
    }
}
