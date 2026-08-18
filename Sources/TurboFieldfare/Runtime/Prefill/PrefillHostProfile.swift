import Foundation

/// Where a chunk's *wall clock* goes on the host, next to `PrefillGPUProfile`'s
/// account of where its GPU time goes.
///
/// The two together are what say whether a stage is expensive or merely slow to
/// reach. 17-M4.5-RESULTS §4 measured a verify block whose GPU spans summed to
/// 70 ms inside a 108 ms block; the GPU profile cannot say what the host was
/// doing for the other 38 ms, because by construction it only sees command
/// buffers. This one covers the whole call, so `total` tracks the block's wall
/// clock and every millisecond lands in exactly one stage.
///
/// Off unless `TF_PREFILL_HOST_PROFILE=1`, and the summary goes to stderr next
/// to the GPU line.
struct PrefillHostProfile {
    enum Stage: String, CaseIterable {
        /// Encoding the layer's pre-MoE half (norm, Q/K/V, RoPE, KV, attention,
        /// o, post-attention, router) plus the shared expert.
        case encodeFront = "enc.front"
        /// Blocked on that half's command buffer.
        case waitFront = "wait.front"
        /// Blocked on the previous layer's routed buffer, which is what makes
        /// its expert slots reusable.
        case drain
        /// Reading the routing back and turning it into sorted, grouped tiles.
        case route
        /// Blocked on a routed-expert read this layer never asked for: the
        /// previous layer guessed it and started it early
        /// (docs/mtp/29-M8-B-PROBE.md §6). What is left here is the part of the
        /// guess that did not fit in the layer ahead of it.
        case waitPrefetch = "wait.prefetch"
        /// Blocked on the shared expert, which was submitted before the
        /// readback.
        case waitShared = "wait.shared"
        /// Building the per-layer metadata and per-tile argument buffers.
        case meta
        /// Placing each tile's experts in the cache, including the preads a
        /// miss costs.
        case fetch
        /// Encoding the routed tiles.
        case encodeMoE = "enc.moe"
        /// Blocked on a tile, mid-layer, because the layer's expert union does
        /// not fit the cache.
        case waitTile = "wait.tile"
        /// The routed reduce and the layer tail.
        case tail
        /// The final norm and the LM head.
        case head
    }

    private static let level = ProcessInfo.processInfo.environment["TF_PREFILL_HOST_PROFILE"]
    static let isEnabled = level == "1"

    private var seconds: [Stage: Double] = [:]
    /// Wall clock of the whole call, so the stages can be checked against it.
    private var callSeconds: Double = 0
    /// What the fetch stages actually moved, so the wait can be read as a rate.
    private var lookaheadTiles = 0
    private var tiles = 0
    private var experts = 0
    private var misses = 0
    private var ioSeconds: Double = 0
    /// Every command buffer's GPU interval, so the call's wall clock can be
    /// split into "the GPU was running something" and "the queue was empty".
    ///
    /// `wait.front` says the host was blocked; it does not say on what. The
    /// front buffer sits behind the previous layer's MoE and shared expert on
    /// the same queue, so a wait can be almost entirely GPU work already in
    /// flight — which is a floor, not an overhead. Only the idle part is what
    /// removing a host round trip could buy (docs/mtp/24-M6-RESULTS.md §1).
    private var gpuQueue = GPUQueueOccupancy()

    /// A timestamp to hand back to `add`. Cheap enough to take unconditionally,
    /// but taken only when the profile is on.
    static func mark() -> UInt64 {
        isEnabled ? DispatchTime.now().uptimeNanoseconds : 0
    }

    mutating func add(_ stage: Stage, since start: UInt64) {
        guard Self.isEnabled, start != 0 else { return }
        let now = DispatchTime.now().uptimeNanoseconds
        guard now > start else { return }
        seconds[stage, default: 0] += Double(now - start) / 1e9
    }

    /// One command buffer's GPU span, taken from the buffer itself once it has
    /// completed.
    mutating func recordGPUInterval(start: Double, end: Double) {
        guard Self.isEnabled else { return }
        gpuQueue.record(start: start, end: end)
    }

    mutating func recordTile(lookahead: Bool) {
        guard Self.isEnabled else { return }
        tiles += 1
        if lookahead { lookaheadTiles += 1 }
    }

    mutating func recordExperts(experts: Int, misses: Int, ioNanos: UInt64) {
        guard Self.isEnabled else { return }
        self.experts += experts
        self.misses += misses
        ioSeconds += Double(ioNanos) / 1e9
    }

    mutating func addCall(since start: UInt64) {
        guard Self.isEnabled, start != 0 else { return }
        let now = DispatchTime.now().uptimeNanoseconds
        guard now > start else { return }
        callSeconds += Double(now - start) / 1e9
    }

    var totalSeconds: Double { seconds.values.reduce(0, +) }

    /// One line in the shape of the GPU one. `other` is the part of the call
    /// that no stage claimed — submission latency and whatever else sits
    /// between the probes.
    var summary: String {
        let total = totalSeconds
        let parts = Stage.allCases.compactMap { stage -> String? in
            guard let value = seconds[stage], value > 0 else { return nil }
            let share = callSeconds > 0 ? value / callSeconds * 100 : 0
            return String(format: "%@=%.3fs(%.0f%%)", stage.rawValue, value, share)
        }
        let other = callSeconds - total
        let bytes = Double(misses) * 3_719_168 / 1_073_741_824
        var line = "[prefill host " + parts.joined(separator: " ")
            + String(format: " other=%.3fs(%.0f%%) call=%.3fs", other,
                     callSeconds > 0 ? other / callSeconds * 100 : 0, callSeconds)
            + String(format: " | tiles=%d ahead=%d experts=%d miss=%d io=%.3fs %.1fGB/s]",
                     tiles, lookaheadTiles, experts, misses, ioSeconds,
                     ioSeconds > 0 ? bytes / ioSeconds : 0)
        if let gpuLine = gpuQueue.summary(label: "prefill gpuq") {
            line += "\n" + gpuLine
        }
        return line
    }

    mutating func reset() {
        seconds.removeAll(keepingCapacity: true)
        callSeconds = 0
        tiles = 0
        lookaheadTiles = 0
        experts = 0
        misses = 0
        ioSeconds = 0
        gpuQueue.reset()
    }
}
