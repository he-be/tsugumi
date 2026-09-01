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
    /// The query-blocked kernel wherever a specialisation exists, which is both
    /// head dimensions the pinned model uses. Measured against the TensorOps
    /// path on the full layers in `docs/investigations/PREFILL_THROUGHPUT.md`
    /// §7-6.
    case causalQBlock = "causal-qblock"
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
    ///
    /// **その 100 MB は私有スロットの話である。**既定の mmap 経路
    /// (`docs/mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md` §8) はスロットを 1 本も確保せず、
    /// 同じバイトは層ファイルのページを residency set が常駐要求する形になる。
    /// 実測の peak は運用点 (32) で 4.5 GB → 1.3 GB に落ちた。
    /// **ガードは今も上の式で数えている** — 常駐要求ぶんを working set に
    /// 数えるべきかは**未検証**なので、算術は動かしていない (40 §2a)。
    /// **上限は運用点の 32 である** (ユーザー確定 2026-08-20)。48 以上は
    /// `Scripts/demo/serve.py` も CLI もサーバーも使わない設定で、mmap の腕では
    /// スロットが footprint を動かさないぶん**踏み込んでも気付きにくい** —
    /// 64 スロットは peak 1.27 GB のまま tok/s が半分になり (52 §9)、112 は
    /// 常駐要求 11.3 GB で機械をスワップさせる。**受け付けないのが一番安い。**
    public static let allowedExpertCacheSlots = [8, 16, 24, 32]
    /// 前面が受け付けるコンテキスト長の上限 (ユーザー確定 2026-08-20)。
    /// 128K は 32 スロットの mmap の腕で通る (52 §9、peak 3.88 GB)。
    public static let maximumContextTokens = 131_072
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
    /// Past that point they buy hit rate and nothing else.
    ///
    /// **既定は 32 = 運用点である** (`Scripts/demo/serve.py`、ユーザー確定
    /// 2026-08-20)。以前の既定 48 と「64 で取り戻せる」という註は、私有スロットが
    /// メモリと速度を交換していた頃のもので、**上限が 32 になったので落とした**
    /// (根拠は `allowedExpertCacheSlots` の註)。
    ///
    /// Machines that cannot hold the cache are rejected by `ExpertCacheBudget` at
    /// runner construction with an actionable error rather than silently
    /// swapping.
    public init(expertCacheSlots: Int = 32,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                rdadvisePolicy: RDAdvicePolicyMode = .off,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 2048,
                prefillAttentionPath: RuntimePrefillAttentionPath = .causalQBlock,
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
