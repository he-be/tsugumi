import Foundation
import Metal

/// What a runtime configuration will actually ask the unified memory system for.
///
/// The routed-expert cache is anonymous memory wrapped as `MTLBuffer`s, so
/// overshooting does not fail an allocation — it pushes the OS into compression
/// and swap, which is slower than the SSD reads the cache exists to avoid. The
/// guard is therefore up front, at configuration time, rather than at the point
/// of allocation.
public struct ExpertCacheBudget: Sendable, Equatable {
    /// `model_weights.bin` resident section, mapped once.
    public let residentBytes: UInt64
    /// `vision/vision_weights.bin`, when the model carries a tower. Counted
    /// even for a run that never passes an image: the mapping is lazy but the
    /// guard is a one-time startup decision, and a configuration that only fits
    /// while the tower is untouched is not a configuration that fits
    /// (`PLAN_VISION.md` §3-1). Zero for a text-only model.
    public let visionResidentBytes: UInt64
    /// `numLayers * slotCount * expertStride`.
    public let expertCacheBytes: UInt64
    /// K and V buffers as `KVCacheManager` allocates them.
    public let kvCacheBytes: UInt64
    /// Chunked-prefill scratch as `PrefillChunkScratchBuffers.allocate` sizes
    /// it. Negligible at the default 128-token chunk (about 17 MB) and no longer
    /// negligible at 2048 (about 270 MB, most of it `routePartials`), so it is
    /// counted rather than assumed away.
    public let prefillScratchBytes: UInt64
    /// What Metal reports this device can keep resident comfortably.
    public let recommendedWorkingSetBytes: UInt64
    public let slotCount: Int

    public var totalBytes: UInt64 {
        residentBytes &+ visionResidentBytes &+ expertCacheBytes
            &+ kvCacheBytes &+ prefillScratchBytes
    }

    public var fitsRecommendedWorkingSet: Bool {
        recommendedWorkingSetBytes == 0 || totalBytes <= recommendedWorkingSetBytes
    }

    public var summary: String {
        func gb(_ bytes: UInt64) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }
        let vision = visionResidentBytes == 0 ? "" : " + vision \(gb(visionResidentBytes))"
        return """
            resident \(gb(residentBytes))\(vision) + experts \(gb(expertCacheBytes)) \
            (\(slotCount) slots) + kv \(gb(kvCacheBytes)) \
            + prefill scratch \(gb(prefillScratchBytes)) = \(gb(totalBytes)); \
            device recommends at most \(gb(recommendedWorkingSetBytes))
            """
    }
}

extension Model {
    /// Bytes the KV cache will allocate for `maxContext`, mirroring
    /// `KVCacheManager.init` — sliding-window layers are capped at the ring
    /// capacity, so long contexts only cost the full-attention layers.
    public func kvCacheByteEstimate(maxContext: Int,
                                    fp16RingEnabled: Bool = true,
                                    maxPrefillChunkTokens: Int =
                                        PrefillRuntimeConfig.defaultChunked.chunkTokens)
        -> UInt64 {
        let fp16Size = 2
        let swaStride = config.numKVHeads * config.headDim * fp16Size
        let fullStride = config.numFullKVHeads * config.fullHeadDim * fp16Size
        let swaCapacity = min(maxContext,
                              max(1, config.slidingWindow + maxPrefillChunkTokens))
        var total: UInt64 = 0
        for layer in 0..<config.numLayers {
            let isFull = config.fullAttentionLayerMask[layer] != 0
            let stride = isFull ? fullStride : swaStride
            let capacity = fp16RingEnabled && !isFull ? swaCapacity : maxContext
            // One K buffer and one V buffer per layer.
            total &+= UInt64(capacity * stride) &* 2
        }
        return total
    }

    /// `prefillConfig` must be the one the runner will actually use: a wide
    /// chunk grows the sliding-window ring to `slidingWindow + chunkTokens` rows
    /// — at chunk 2048 that is 2.7x the default ring — and grows the prefill
    /// scratch with it. Sizing the guard from the widest *allowed* chunk instead
    /// would reject configurations that fit.
    public func expertCacheBudget(
        slotCount: Int,
        maxContext: Int,
        prefillConfig: PrefillRuntimeConfig = .defaultChunked
    ) -> ExpertCacheBudget {
        let scratchBytes: UInt64 = prefillConfig.enabled
            ? UInt64(PrefillChunkScratchLayout(config: config,
                                               runtime: prefillConfig).totalPersistentBytes)
            : 0
        return ExpertCacheBudget(
            residentBytes: residentIndex.header.residentSize,
            visionResidentBytes: manifest.vision?.payloadBytes ?? 0,
            expertCacheBytes: UInt64(packedExpertsLayout.numLayers)
                &* UInt64(slotCount)
                &* packedExpertsLayout.expertStride,
            kvCacheBytes: kvCacheByteEstimate(
                maxContext: maxContext,
                maxPrefillChunkTokens: prefillConfig.chunkTokens),
            prefillScratchBytes: scratchBytes,
            recommendedWorkingSetBytes: UInt64(device.recommendedMaxWorkingSetSize),
            slotCount: slotCount)
    }
}

public enum ExpertCacheBudgetError: Error, CustomStringConvertible {
    case exceedsRecommendedWorkingSet(ExpertCacheBudget)

    public var description: String {
        switch self {
        case .exceedsRecommendedWorkingSet(let budget):
            return """
                expert cache configuration exceeds this device's recommended Metal \
                working set — \(budget.summary). Lower --expert-cache-slots or \
                --max-context.
                """
        }
    }
}
