import Foundation

/// The one arithmetic fact the `sym` scheme rests on, shared by the runtime and
/// the packer because both have to agree on it exactly.
///
/// A 4-bit affine group stores `w = q * scale + bias`. When the checkpoint's
/// weights sit on a symmetric int4 lattice — `w = scale * (c - 8)` with
/// `c ∈ [0, 15]` — the bias is `-8 * scale` and carries no information.
/// `-8` is a power of two, so the product is a pure exponent shift: BF16
/// represents it without rounding, and the identity holds as a **bit pattern**.
/// That is what makes dropping the array exact rather than an approximation,
/// and what lets it be verified rather than assumed
/// (`docs/mtp/44-W1-WEIGHT-DIET.md` §1).
package enum MoEPackAffineV1 {

    /// `-8 * scale` in BF16, from BF16 bits. Flips the sign and adds 3 to the
    /// exponent; only an infinity or a NaN can round-trip differently, and a
    /// scale is neither.
    @inline(__always)
    package static func symmetricBiasBits(_ scaleBits: UInt16) -> UInt16 {
        let value = Float(bitPattern: UInt32(scaleBits) << 16) * -8
        let bits = value.bitPattern
        let rounding = UInt32(0x7FFF) &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: (bits &+ rounding) >> 16)
    }

    /// Whether every group in a BF16 scale/bias pair satisfies the identity.
    /// Both buffers carry BF16 bit patterns and must be the same length.
    package static func isSymmetric(scaleBits: UnsafeRawBufferPointer,
                                    biasBits: UnsafeRawBufferPointer) -> Bool {
        guard scaleBits.count == biasBits.count, scaleBits.count % 2 == 0 else {
            return false
        }
        let scales = scaleBits.bindMemory(to: UInt16.self)
        let biases = biasBits.bindMemory(to: UInt16.self)
        for index in scales.indices where symmetricBiasBits(scales[index]) != biases[index] {
            return false
        }
        return true
    }
}
