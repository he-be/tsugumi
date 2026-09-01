import Foundation

package enum MoEPackFormatV1 {
    package static let magic = "MOEPACK"
    /// What the same field said while the project was called TurboFieldfare.
    /// Every already-installed model and every byte published on the Hub
    /// carries it, and the published SHA-256 pins are taken over those bytes,
    /// so the reader accepts it. Nothing writes it any more.
    package static let legacyMagic = "GTURBO"
    package static let versionMajor = 1
    package static let versionMinor = 0
    /// Minor version stamped on a manifest that carries a `vision` section.
    /// Rejection of such a model by an older runtime is the `visionTower` flag's
    /// job (`decode` accepts any `minor >= 0`); the bump is descriptive.
    package static let versionMinorVision = 1
    /// Minor version stamped on a manifest that carries a `draft` section. Same
    /// shape as the vision bump: descriptive, with `mtpDraft` doing the
    /// rejecting.
    package static let versionMinorDraft = 2
    /// Minor version stamped on a manifest whose `arch` carries the
    /// `linearAttention` section. Same shape as the two bumps above:
    /// descriptive, with the `linearAttention` flag doing the rejecting.
    package static let versionMinorLinearAttention = 3
    package static let alignmentBytes: UInt64 = 16_384
    package static let residentHeaderBytes = 24
    package static let residentEntryBytes = 72
    package static let residentIndexMaxBytes: UInt64 = 16 * 1024 * 1024
    /// Bound on `packed_experts/layout.json`. It grows with layers × experts:
    /// Gemma 4's 30 × 128 experts come to 8.5 MB, Qwen3.5-MoE's 40 × 256 to
    /// 22.5 MB. 64 MB leaves room for a model of that shape again without
    /// letting an untrusted file be unbounded. Writer, verifier and runtime
    /// reader all take this one number so they cannot drift apart.
    package static let packedExpertsLayoutMaxBytes: UInt64 = 64 * 1024 * 1024

    package static let knownFlags: Set<String> = [
        "streamingPresent", "turboQuantKV", "aneSharedExpert", "visionTower",
        "mtpDraft", "linearAttention",
    ]

    /// Where a vision-capable install keeps the tower weights. A separate file
    /// from `model_weights.bin` so a text-only run never pays for its bytes.
    package static let visionWeightsPath = "vision/vision_weights.bin"

    /// Where an MTP-capable install keeps the drafter weights. Separate from
    /// `model_weights.bin` for the same reason the tower is: a run that does not
    /// speculate never reads, hashes or pages in its bytes.
    package static let draftWeightsPath = "draft/draft_weights.bin"

    package enum DType: UInt8, Sendable {
        case u32 = 0
        case bf16 = 1
        case fp16 = 2
        case fp32 = 3
    }
}

package enum MoEPackFormatError: Error, Equatable, CustomStringConvertible, Sendable {
    case invalid(field: String, reason: String)
    case overflow(field: String)
    case truncated(field: String)

    package var description: String {
        switch self {
        case let .invalid(field, reason): "\(field): \(reason)"
        case let .overflow(field): "\(field): arithmetic overflow"
        case let .truncated(field): "\(field): truncated"
        }
    }
}

@inline(__always)
package func moepackCheckedAdd(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
    let (value, overflow) = lhs.addingReportingOverflow(rhs)
    guard !overflow else { throw MoEPackFormatError.overflow(field: field) }
    return value
}

@inline(__always)
package func moepackCheckedMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
    let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
    guard !overflow else { throw MoEPackFormatError.overflow(field: field) }
    return value
}

package enum MoEPackPathValidator {
    package static func appleFilesystemKey(_ path: String) -> String {
        path.precomposedStringWithCanonicalMapping
            .lowercased(with: Locale(identifier: "en_US_POSIX"))
    }

    package static func validateRelativePath(_ path: String, field: String) throws {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\0") else {
            throw MoEPackFormatError.invalid(field: field, reason: "unsafe relative path")
        }
        let components = path.components(separatedBy: "/")
        guard components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw MoEPackFormatError.invalid(field: field, reason: "non-canonical path")
        }
        let normalized = NSString.path(withComponents: components)
        guard normalized == path else {
            throw MoEPackFormatError.invalid(field: field, reason: "non-normalized path")
        }
    }

    package static func validateBasename(_ name: String, field: String) throws {
        try validateRelativePath(name, field: field)
        guard !name.contains("/") else {
            throw MoEPackFormatError.invalid(field: field, reason: "expected basename")
        }
    }
}
