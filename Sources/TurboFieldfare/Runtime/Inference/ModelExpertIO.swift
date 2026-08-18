import Darwin
import Foundation
import Metal
import TurboFieldfareFormat

public struct RoutedExpertFetchPlan: Sendable {
    public let layer: Int
    public let cachePlan: ExpertCachePlan

    public var experts: [Int] { cachePlan.experts }
    public var misses: [Int] { cachePlan.misses }
    public var hits: Int { cachePlan.hits }
    public var assignedSlots: [Int] { cachePlan.assignedSlots }

    public init(layer: Int, cachePlan: ExpertCachePlan) {
        self.layer = layer
        self.cachePlan = cachePlan
    }
}

extension Model {
    /// Empty every open layer's expert cache. Measurement only: it lets a probe
    /// start two phases from the same cold cache (`PreadExpertStreamer.resetCache`).
    /// Layers that were never opened have nothing cached, so they are skipped.
    public func resetExpertCaches() {
        streamersQueue.sync {
            for streamer in streamersBox.streamers {
                streamer?.resetCache()
            }
        }
    }

    public func routedExpertOffsets(layer: Int) -> MoEExpertOffsets {
        let expert = packedExpertsLayout.expert(layer: layer, expert: 0)
        func offset(_ role: String) -> UInt32 {
            guard let tensor = expert.subTensors[role],
                  let offset = UInt32(exactly: tensor.offset) else {
                preconditionFailure("invalid routed expert metadata for role \(role)")
            }
            return offset
        }
        return MoEExpertOffsets(
            gateWOff: offset("gate"),
            gateSOff: offset("gate_scales"),
            gateBOff: offset("gate_biases"),
            upWOff: offset("up"),
            upSOff: offset("up_scales"),
            upBOff: offset("up_biases"),
            downWOff: offset("down"),
            downSOff: offset("down_scales"),
            downBOff: offset("down_biases"))
    }

    public func routedExpertPhysicalOffsets(layer: Int) -> [UInt64] {
        packedExpertsLayout.layers[layer].experts.map(\.offset)
    }

    public func adviseRoutedExperts(layer: Int,
                                    experts: [Int]) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return streamer.adviseExpertMisses(experts: experts)
    }

    public func routedExpertAdviceByteEstimate(layer: Int,
                                               missCount: Int) throws -> UInt64 {
        guard missCount > 0 else { return 0 }
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return UInt64(missCount) * streamer.layout.expertStride
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = []) throws -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        return RoutedExpertFetchPlan(
            layer: layer,
            cachePlan: streamer.planExpertsCached(experts: experts, avoidingSlots: validSlots))
    }

    public func planRoutedExpertsIfPossible(layer: Int,
                                            experts: [Int],
                                            avoidingSlots: Set<Int> = []) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        guard let cachePlan = streamer.planExpertsCachedIfPossible(
            experts: experts,
            avoidingSlots: validSlots)
        else {
            return nil
        }
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan)
    }

    public func routedExpertCacheSlotCount(layer _: Int) -> Int? {
        guard case .pread(let slotCount) = streamingMode else { return nil }
        return slotCount
    }

    public func routedExpertBuffers(for plan: RoutedExpertFetchPlan) throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return Self.makeExpertViews(
            streamer.expertCachePlanBuffers(plan.cachePlan),
            layer: plan.layer,
            experts: plan.experts)
    }

    public func adviseRoutedExperts(plan: RoutedExpertFetchPlan) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return streamer.adviseExpertCachePlanMisses(plan.cachePlan)
    }

    /// The same fetch for a plan that has nothing to read.
    ///
    /// `fetchRoutedExperts` hops to the I/O queue even when every expert is
    /// already resident, and a speculative verify block issues one fetch per
    /// tile per layer — about a hundred hops a block, on a path whose expert
    /// cache hits better than 90% of the time
    /// (docs/mtp/16-M4.5-PLAN.md §4 d). Returns nil when the plan has misses,
    /// which is the caller's signal to take the asynchronous path.
    public func fetchResidentRoutedExperts(plan: RoutedExpertFetchPlan) throws
        -> [TensorView]? {
        guard plan.misses.isEmpty else { return nil }
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let buffers = try streamer.executeExpertCachePlan(plan.cachePlan)
        let views = Self.makeExpertViews(buffers,
                                         layer: plan.layer,
                                         experts: plan.experts)
        telemetry.recordFetch(layer: plan.layer,
                              experts: plan.experts,
                              hits: plan.hits,
                              misses: 0,
                              nanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
        return views
    }

    /// Start a routed-expert read and return before it finishes.
    ///
    /// `fetchRoutedExperts` starts the read and waits for it in one step, so
    /// the GPU has nothing queued while the bytes land — and a verify block
    /// spends about as long reading experts as it spends computing with them
    /// (docs/mtp/18-M4.6-RESULTS.md §2). A block knows every tile's experts as
    /// soon as the routing is grouped, so it can issue the next tile's read
    /// before encoding this one and let the two overlap.
    ///
    /// The plan is made by the caller and therefore already owns its slots; the
    /// caller is responsible for keeping the next plan off the slots a read in
    /// flight is writing into (`avoidingSlots`).
    public func startRoutedExpertFetch(plan: RoutedExpertFetchPlan) throws
        -> RoutedExpertFetchHandle {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        let handle = RoutedExpertFetchHandle()
        let layer = plan.layer
        let experts = plan.experts
        let hits = plan.hits
        let misses = plan.misses.count
        let cachePlan = plan.cachePlan
        let telemetry = self.telemetry
        // Nothing to read: the slots are already right, and hopping to a queue
        // to discover that costs more than the work.
        if misses == 0 {
            let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let buffers = try streamer.executeExpertCachePlan(cachePlan)
            telemetry.recordFetch(layer: layer, experts: experts, hits: hits, misses: 0,
                                  nanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
            handle.complete(.success(Self.makeExpertViews(buffers,
                                                          layer: layer,
                                                          experts: experts)))
            return handle
        }
        DispatchQueue.global(qos: .userInitiated).async {
            let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            do {
                let buffers = try streamer.executeExpertCachePlan(cachePlan)
                telemetry.recordFetch(
                    layer: layer, experts: experts, hits: hits, misses: misses,
                    nanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
                handle.complete(.success(Self.makeExpertViews(buffers,
                                                              layer: layer,
                                                              experts: experts)))
            } catch {
                handle.complete(.failure(error))
            }
        }
        return handle
    }

    public func fetchRoutedExperts(plan: RoutedExpertFetchPlan) async throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let views: [TensorView] = try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.executeExpertCachePlan(plan.cachePlan)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: plan.layer,
                        experts: plan.experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        telemetry.recordFetch(layer: plan.layer,
                              experts: plan.experts,
                              hits: plan.hits,
                              misses: plan.misses.count,
                              nanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
        return views
    }

    public func fetchRoutedExperts(layer: Int, experts: [Int]) async throws -> [TensorView] {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // `loadExpertsCached` plans and executes in one step, so the hit count
        // is only observable from inside the streamer.
        nonisolated(unsafe) var observedHits = 0
        let views: [TensorView] = try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let plan = streamer.planExpertsCached(experts: experts)
                    observedHits = plan.hits
                    let buffers = try streamer.executeExpertCachePlan(plan)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: layer,
                        experts: experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        telemetry.recordFetch(layer: layer,
                              experts: experts,
                              hits: observedHits,
                              misses: experts.count - observedHits,
                              nanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
        return views
    }

    private static func makeExpertViews(
        _ buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
        layer: Int,
        experts: [Int]
    ) -> [TensorView] {
        buffers.enumerated().map { index, entry in
            TensorView(
                buffer: entry.buffer,
                offset: entry.offset,
                length: entry.size,
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (UInt32(layer), UInt32(experts[index]), 0, 0),
                dtype: GTurboFormatV1.DType.u32.rawValue)
        }
    }
}

public enum RoutedExpertFetchError: Error, Equatable, CustomStringConvertible {
    case completedWithoutResult

    public var description: String {
        "routed expert fetch handle was signalled without a result"
    }
}

/// A routed-expert read that has been issued but not waited for.
///
/// Deliberately not an `async` task: the caller is a synchronous encode loop
/// that wants the read running against the GPU work it is about to submit, and
/// a semaphore says that in one line. `wait()` is called exactly once.
public final class RoutedExpertFetchHandle: @unchecked Sendable {
    private let semaphore = DispatchSemaphore(value: 0)
    private var result: Result<[TensorView], Error>?

    fileprivate init() {}

    fileprivate func complete(_ result: Result<[TensorView], Error>) {
        self.result = result
        semaphore.signal()
    }

    public func wait() throws -> [TensorView] {
        semaphore.wait()
        guard let result else {
            throw RoutedExpertFetchError.completedWithoutResult
        }
        return try result.get()
    }
}
