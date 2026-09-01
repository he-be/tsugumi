import Foundation
import Tsugumi

public enum AppExpertCachePolicy: String, CaseIterable, Sendable, Identifiable {
    case lfu
    case lru

    public var id: String { rawValue }
    public var label: String { rawValue.uppercased() }
}

public enum AppRDAdvicePolicy: String, CaseIterable, Sendable, Identifiable {
    case off
    case `default`
    case bounded
    case adaptive

    public var id: String { rawValue }
    public var label: String { rawValue.capitalized }

    var runtimeValue: RDAdvicePolicyMode {
        switch self {
        case .off: return .off
        case .default: return .default
        case .bounded: return .bounded
        case .adaptive: return .adaptive
        }
    }
}

public enum AppModelVerification: String, CaseIterable, Sendable, Identifiable {
    case fullSha256 = "full-sha256"
    case trustedInstall = "trusted-install"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .fullSha256: return "Full SHA-256"
        case .trustedInstall: return "Trust verified install"
        }
    }

    var runtimeValue: ModelIntegrityPolicy {
        switch self {
        case .fullSha256: return .fullSha256
        case .trustedInstall: return .sizeCheckTrustedReceipt
        }
    }
}

public struct AppRuntimeOptions: Equatable, Sendable {
    public static let allowedSlotCounts = RuntimeConfiguration.allowedExpertCacheSlots
    public static let allowedPrefillChunkTokens = RuntimeConfiguration.allowedPrefillChunkTokens

    public var expertCacheSlots: Int
    public var expertCachePolicy: AppExpertCachePolicy
    public var prefillEnabled: Bool
    public var prefillChunkTokens: Int
    public var rdadvisePolicy: AppRDAdvicePolicy
    public var modelVerification: AppModelVerification
    /// MTP speculative decoding. On by default — the adopted operating point
    /// for both families (`docs/SERVER_RUNBOOK.md`). The block width is the
    /// model kind's own (`AppModelKind.draftBlockSize`); this only says
    /// whether the speculative loop runs at all. Requires prefill: the MTP
    /// path goes through chunked prefill, so `prefillEnabled == false` turns
    /// this off at load.
    public var mtpEnabled: Bool

    public init(expertCacheSlots: Int = 32,
                expertCachePolicy: AppExpertCachePolicy = .lfu,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 2048,
                rdadvisePolicy: AppRDAdvicePolicy = .off,
                modelVerification: AppModelVerification = .fullSha256,
                mtpEnabled: Bool = true) {
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.prefillEnabled = prefillEnabled
        self.prefillChunkTokens = prefillChunkTokens
        self.rdadvisePolicy = rdadvisePolicy
        self.modelVerification = modelVerification
        self.mtpEnabled = mtpEnabled
    }

    public func validate() throws {
        guard Self.allowedSlotCounts.contains(expertCacheSlots) else {
            throw AppInferenceError.invalidRequest(
                "expert cache slots must be one of \(Self.allowedSlotCounts)")
        }
        guard Self.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw AppInferenceError.invalidRequest(
                "prefill chunk size must be one of \(Self.allowedPrefillChunkTokens)")
        }
    }

    public var prefillConfig: PrefillRuntimeConfig {
        prefillEnabled ? .production(chunkTokens: prefillChunkTokens) : .off
    }

    /// Whether the speculative loop actually runs: the toggle, gated by the
    /// prefill path it rides on.
    public var effectiveMTPEnabled: Bool { mtpEnabled && prefillEnabled }

    public var resultSummary: String {
        let prefill = prefillEnabled ? "prefill \(prefillChunkTokens)" : "prefill off"
        let verification = modelVerification == .fullSha256 ? "full SHA-256" : "trusted receipt"
        let mtp = effectiveMTPEnabled ? "MTP on" : "MTP off"
        return "Cache \(expertCacheSlots) \(expertCachePolicy.label), \(prefill), \(mtp), FP16 KV, RDADVISE \(rdadvisePolicy.label.lowercased()), \(verification)"
    }

    /// Slot labels compare peak memory against the default (32 slots) using
    /// the pinned QAT checkpoint's expert stride: 30 layers x 3,719,168 B
    /// = about 0.11 GB per slot.
    public static func slotsLabel(for slots: Int) -> String {
        switch slots {
        case 8: "8, -4.46 GB"
        case 16: "16, -3.57 GB"
        case 24: "24, -2.68 GB"
        case 32: "32, -1.79 GB"
        case 48: "48, Default"
        case 64: "64, +1.79 GB"
        case 80: "80, +3.57 GB"
        case 96: "96, +5.36 GB"
        case 112: "112, +7.14 GB"
        default: "\(slots)"
        }
    }

    public func resolvedRuntimeConfiguration(forceLogitsHead: Bool) throws -> RuntimeConfiguration {
        try validate()
        return RuntimeConfiguration(
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy == .lru ? .lru : .lfu,
            rdadvisePolicy: rdadvisePolicy.runtimeValue,
            prefillEnabled: prefillEnabled,
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: forceLogitsHead)
    }
}

public struct AppLoadedRuntimeKey: Equatable, Sendable {
    public var modelDirectory: URL
    public var maxContextTokens: Int
    public var expertCacheSlots: Int
    public var expertCachePolicy: AppExpertCachePolicy
    public var rdadvisePolicy: AppRDAdvicePolicy
    public var modelVerification: AppModelVerification
    public var forceLogitsHead: Bool
    /// MTP changes what load allocates (Gemma's speculative scratch, Ornith's
    /// head sidecar), so flipping it means a reload.
    public var mtpEnabled: Bool
    /// Since the two-model rework the family sessions bind their prefill
    /// configuration at load, so these became load-time choices too.
    public var prefillEnabled: Bool
    public var prefillChunkTokens: Int

    public init(modelDirectory: URL,
                maxContextTokens: Int,
                options: AppRuntimeOptions,
                forceLogitsHead: Bool = false) {
        self.modelDirectory = modelDirectory.standardizedFileURL
        self.maxContextTokens = maxContextTokens
        self.expertCacheSlots = options.expertCacheSlots
        self.expertCachePolicy = options.expertCachePolicy
        self.rdadvisePolicy = options.rdadvisePolicy
        self.modelVerification = options.modelVerification
        self.forceLogitsHead = forceLogitsHead
        self.mtpEnabled = options.effectiveMTPEnabled
        self.prefillEnabled = options.prefillEnabled
        self.prefillChunkTokens = options.prefillChunkTokens
    }
}
