import Foundation
import Metal

/// Where chunked prefill's GPU time goes, per stage.
///
/// Prefill overlaps expert I/O with GPU work, so a wall-clock split cannot say
/// which of the two a stage is waiting on. Every command buffer reports its own
/// `gpuStartTime`/`gpuEndTime`, so summing those per stage gives the GPU-side
/// cost directly, and comparing the total against the prefill wall clock says
/// how much of the run is GPU-bound at all.
///
/// Off unless `TF_PREFILL_GPU_PROFILE=1`, and the summary goes to stderr next to
/// the existing prefill footer (`bench.sh pp` picks it up).
struct PrefillGPUProfile {
    enum Stage: String, CaseIterable {
        /// Embedding, Q/K/V projections, RoPE, KV writes, attention, router.
        case attention = "attn"
        /// The shared expert MLP.
        case shared
        /// Routed experts, including the tile stream's waits.
        case moe
        /// Routed reduce plus the layer tail.
        case tail
        /// Final norm and the LM head, on the last chunk only.
        case head
    }

    static let isEnabled = ProcessInfo.processInfo.environment["TF_PREFILL_GPU_PROFILE"] == "1"

    private var seconds: [Stage: Double] = [:]

    mutating func record(_ stage: Stage, _ commandBuffer: MTLCommandBuffer) {
        guard Self.isEnabled else { return }
        let span = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        guard span > 0 else { return }
        seconds[stage, default: 0] += span
    }

    var totalSeconds: Double { seconds.values.reduce(0, +) }

    /// One line, formatted like the other prefill footers.
    var summary: String {
        let total = totalSeconds
        let parts = Stage.allCases.compactMap { stage -> String? in
            guard let value = seconds[stage], value > 0 else { return nil }
            let share = total > 0 ? value / total * 100 : 0
            return String(format: "%@=%.2fs(%.0f%%)", stage.rawValue, value, share)
        }
        return "[prefill gpu " + parts.joined(separator: " ")
            + String(format: " total=%.2fs]", total)
    }

    mutating func reset() { seconds.removeAll(keepingCapacity: true) }
}
