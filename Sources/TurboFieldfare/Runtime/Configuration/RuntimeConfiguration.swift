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
    /// Chunk widths the front ends will accept. The width sets how many unique
    /// routed experts one layer touches per chunk, and therefore how many expert
    /// bytes the SSD has to move per token: 128 tokens costs about 62 MB/token,
    /// 2048 tokens about 6 MB/token (`docs/investigations/PREFILL_THROUGHPUT.md`
    /// §2). Wide chunks trade that I/O for KV ring rows
    /// (`slidingWindow + chunkTokens`) and prefill scratch, both of which
    /// `ExpertCacheBudget` accounts for at the configured width — not at this
    /// list's maximum.
    public static let allowedPrefillChunkTokens = [32, 64, 128, 256, 512, 1024, 2048]
    public static let minimumExpertCacheSlotsForChunkedPrefill = 16

    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    public let headPath: RuntimeHeadPath

    /// Expert I/O overlaps the shared-MLP GPU work, so slots buy throughput only
    /// while the misses they remove were still poking out from behind that work.
    /// Past that point they buy hit rate and nothing else: on the group-32 QAT
    /// checkpoint, 96 slots reach a 97.7% decode hit rate against 64 slots'
    /// 90.6% — better than the group-64 baseline manages — at identical tokens
    /// per second.
    ///
    /// 48 is where that ceiling is reached with the least memory. The QAT
    /// checkpoint measures within noise of 64 slots there (+1.2% to +1.8%, three
    /// interleaved runs) while its peak footprint drops 1.8 GB, and 32 is the
    /// cliff (-6% to -7%, hit rate 72-77%). It has more GPU work per token than
    /// the group-64 lineage, so it hides more I/O and tolerates a smaller cache.
    ///
    /// This costs the group-64 baseline about 4.5%, which was measured when 64
    /// was the default and is not re-measured here: 64 is that checkpoint's knee,
    /// where its 99.2% decode hit rate stops improving. Pass
    /// `--expert-cache-slots 64` to get it back.
    ///
    /// Machines that cannot hold the cache are rejected by `ExpertCacheBudget` at
    /// runner construction with an actionable error rather than silently
    /// swapping.
    public init(expertCacheSlots: Int = 48,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                rdadvisePolicy: RDAdvicePolicyMode = .off,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 2048,
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
