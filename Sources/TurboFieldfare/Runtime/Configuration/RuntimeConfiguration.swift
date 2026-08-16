public enum RuntimeHeadPath: String, Codable, Sendable {
    case fusedRows = "fused-rows"
    case logits
}

public enum RuntimePrefillPolicy: String, Codable, Sendable {
    case off
    case chunked
}

public enum RuntimePrefillAttentionPath: String, Codable, Sendable {
    case causalTiled = "causal-tiled"
    case fullTensorOps2DPreferred = "full-tensorops-2d-preferred"
    case fullTensorOps2DValidityV2 = "full-tensorops-2d-validity-v2"
}

public enum RuntimeExpertCachePolicy: String, Codable, Sendable {
    case lfu
    case lru
}

public struct RuntimeConfiguration: Sendable, Equatable {
    /// Slot counts the front ends will accept. Everything above 32 is specific
    /// to machines with enough unified memory to hold it: the cache costs
    /// `numLayers * slots * expertStride` (about 100 MB per slot for
    /// gemma-4-26b-a4b), and `ExpertCacheBudget` rejects a configuration that
    /// would push past the Metal device's recommended working set.
    public static let allowedExpertCacheSlots = [8, 16, 24, 32, 48, 64, 80, 96, 112]
    public static let allowedPrefillChunkTokens = [32, 64, 128]
    public static let minimumExpertCacheSlotsForChunkedPrefill = 16

    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    public let headPath: RuntimeHeadPath

    /// 64 slots is where measured decode throughput stops improving on M3 Pro.
    /// Routing is heavily skewed, so 50% residency already gives a 99.2% decode
    /// hit rate; the remaining expert I/O overlaps the shared-MLP GPU work, which
    /// is why 96 slots buys another 0.6 points of hit rate but no tokens per
    /// second. Going the other way costs real throughput (48 slots is about 4.5%
    /// slower). Machines that cannot hold the cache are rejected by
    /// `ExpertCacheBudget` at runner construction with an actionable error rather
    /// than silently swapping.
    public init(expertCacheSlots: Int = 64,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                rdadvisePolicy: RDAdvicePolicyMode = .off,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                prefillAttentionPath: RuntimePrefillAttentionPath = .fullTensorOps2DPreferred,
                forceLogitsHead: Bool = false) {
        precondition(Self.allowedExpertCacheSlots.contains(expertCacheSlots),
                     "unsupported expert-cache slot count")
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.headPath = forceLogitsHead ? .logits : .fusedRows
    }

    public static var production: RuntimeConfiguration {
        RuntimeConfiguration()
    }

    public var fp16RingEnabled: Bool { true }
    public var rdadviseEnabled: Bool { rdadvisePolicy != .off }
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
    public var modelExpertCachePolicy: ExpertCachePolicy {
        expertCachePolicy == .lru ? .lru : .lfu
    }
}
