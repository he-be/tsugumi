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
/// Off unless `TF_PREFILL_GPU_PROFILE` is set, and the summary goes to stderr
/// next to the existing prefill footer (`bench.sh pp` picks it up).
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

    /// Sub-stages of `.attention`, resolved only in detail mode.
    ///
    /// `.attention` is one command buffer per layer, so its 5-way split needs
    /// the layer's work cut into one command buffer per group. That costs a
    /// submission per group (the GPU idles between them), which inflates the
    /// prefill wall clock — so detail mode is for attribution, not for timing
    /// the run. The per-group GPU spans themselves stay comparable, and the
    /// detail total is printed next to `attn` so the inflation is visible.
    enum Detail: String, CaseIterable {
        /// Token embedding lookup (first layer's buffer only).
        case embed
        /// Input RMS norm.
        case norm
        /// Q/K/V projections.
        case qkvProjection = "qkv"
        /// RoPE plus the per-head Q/K norms.
        case rope
        /// KV cache writes.
        case kvCopy = "kvcopy"
        /// Attention proper, sliding-window layers (headDim 256).
        case attentionSWA = "attn.swa"
        /// Attention proper, full layers (headDim 512).
        case attentionFull = "attn.full"
        /// The output projection.
        case outputProjection = "oproj"
        /// Post-attention norms plus the router.
        case post
    }

    private static let level = ProcessInfo.processInfo.environment["TF_PREFILL_GPU_PROFILE"]

    static let isEnabled = level == "1" || level == "2"
    /// `TF_PREFILL_GPU_PROFILE=2`: also split `.attention` into `Detail` groups.
    static let isDetailed = level == "2"

    private var seconds: [Stage: Double] = [:]
    private var detailSeconds: [Detail: Double] = [:]

    mutating func record(_ stage: Stage,
                         _ commandBuffer: MTLCommandBuffer,
                         detail: Detail? = nil) {
        guard Self.isEnabled else { return }
        let span = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        guard span > 0 else { return }
        seconds[stage, default: 0] += span
        if Self.isDetailed, let detail { detailSeconds[detail, default: 0] += span }
    }

    var totalSeconds: Double { seconds.values.reduce(0, +) }

    /// One line, formatted like the other prefill footers, plus a second line
    /// with the `.attention` split when detail mode is on. `label` names the
    /// path, so plain decode can be accounted with the same buckets as a block
    /// and the two read side by side.
    func summary(label: String = "prefill gpu") -> String {
        let total = totalSeconds
        let parts = Stage.allCases.compactMap { stage -> String? in
            guard let value = seconds[stage], value > 0 else { return nil }
            let share = total > 0 ? value / total * 100 : 0
            return String(format: "%@=%.3fs(%.0f%%)", stage.rawValue, value, share)
        }
        var line = "[" + label + " " + parts.joined(separator: " ")
            + String(format: " total=%.3fs]", total)
        guard Self.isDetailed, !detailSeconds.isEmpty else { return line }
        let attentionTotal = detailSeconds.values.reduce(0, +)
        let detailParts = Detail.allCases.compactMap { detail -> String? in
            guard let value = detailSeconds[detail], value > 0 else { return nil }
            let share = attentionTotal > 0 ? value / attentionTotal * 100 : 0
            return String(format: "%@=%.3fs(%.0f%%)", detail.rawValue, value, share)
        }
        line += "\n[" + label + " attn " + detailParts.joined(separator: " ")
            + String(format: " total=%.3fs]", attentionTotal)
        return line
    }

    mutating func reset() {
        seconds.removeAll(keepingCapacity: true)
        detailSeconds.removeAll(keepingCapacity: true)
    }
}
