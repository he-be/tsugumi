import Foundation
import MoEPackFormat

/// Decides whether a source checkpoint can be packed `sym` — that is, without
/// its bias arrays.
///
/// A lattice-aligned QAT checkpoint (Google's `qat-q4_0-unquantized` family,
/// restored onto its int4 grid) satisfies `bias == -8 * scale` in every group.
/// `-8` is a power of two, so the product is a pure exponent shift and BF16
/// carries it without rounding: the identity is checkable on **bit patterns**,
/// not within a tolerance, and dropping the array is exact rather than an
/// approximation. `docs/mtp/44-W1-WEIGHT-DIET.md` §1 measures it holding over
/// all 788,175,872 groups of the shipped checkpoint, against 34-38% of groups
/// failing on an ordinarily quantized one.
///
/// The probe is all-or-nothing on purpose. A checkpoint that satisfies the
/// identity everywhere loses nothing; one that fails it anywhere would need a
/// per-tensor scheme, and the shader library has one scheme for the whole
/// model. So a single failing group sends the whole install back to `affine`.
enum SymmetricProbe {

    struct Result: Sendable {
        let symmetric: Bool
        /// Groups examined, and how many broke the identity. `checked` is the
        /// number the decision actually rests on.
        let checked: UInt64
        let violations: UInt64
        /// Tensor the first violation was found in, for the install log.
        let firstViolation: String?

        var summary: String {
            if symmetric {
                return "symmetric: \(checked) groups verified, bias == -8 * scale in all of them"
            }
            if checked == 0 { return "affine: no readable scale/bias pair to probe" }
            return "affine: \(violations) of \(checked) groups break bias == -8 * scale"
                + (firstViolation.map { " (first in \($0))" } ?? "")
        }
    }

    /// Reads every `.scales`/`.biases` pair belonging to a 4-bit tensor the plan
    /// will lay out and checks the identity.
    ///
    /// `readBytes` is supplied only when the source shards are local files: a
    /// streaming install would have to fetch the bias ranges before it can plan
    /// the file that omits them, so it stays `affine` (§7 of 44).
    static func probe(bases: [String],
                      registry: [String: SourceTensor],
                      readBytes: (SourceTensor) throws -> Data) -> Result {
        var checked: UInt64 = 0
        var violations: UInt64 = 0
        var firstViolation: String?

        for base in bases {
            guard let scales = registry[base + ".scales"],
                  let biases = registry[base + ".biases"],
                  scales.dtype == .bf16, biases.dtype == .bf16,
                  scales.sizeBytes == biases.sizeBytes else {
                // No affine companions, or a shape the packer would reject
                // anyway: nothing to decide here, and `plan` reports it.
                continue
            }
            guard let scaleData = try? readBytes(scales),
                  let biasData = try? readBytes(biases),
                  scaleData.count == biasData.count else {
                return Result(symmetric: false, checked: checked,
                              violations: violations, firstViolation: base)
            }
            let bad = scaleData.withUnsafeBytes { s in
                biasData.withUnsafeBytes { b in
                    MoEPackAffineV1.isSymmetric(scaleBits: s, biasBits: b) ? 0 : 1
                }
            }
            checked += UInt64(scaleData.count / 2)
            if bad != 0 {
                violations += 1
                if firstViolation == nil { firstViolation = base }
                // One counterexample settles it; reading the rest of a 14 GB
                // checkpoint to count them would cost minutes and change
                // nothing.
                return Result(symmetric: false, checked: checked,
                              violations: violations, firstViolation: firstViolation)
            }
        }
        return Result(symmetric: checked > 0, checked: checked,
                      violations: 0, firstViolation: nil)
    }
}
