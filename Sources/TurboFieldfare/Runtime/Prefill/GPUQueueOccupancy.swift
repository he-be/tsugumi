import Foundation
import Metal

/// How much of a stretch of wall clock the GPU queue was actually running
/// something, taken from the command buffers' own GPU timestamps.
///
/// The host profile says where the *host* went (`wait.front` and friends); the
/// GPU profile says what each stage cost on the GPU. Neither answers the
/// question a "stop waiting for the GPU" optimization has to answer first: how
/// much of the wait is the queue being empty. A host blocked on a command
/// buffer that sits behind the previous layer's MoE is not paying a round trip
/// — it is waiting for work that has to happen either way.
///
/// Intervals are merged, not summed, so buffers the driver overlaps are counted
/// once, and the gaps between them are the recoverable part.
struct GPUQueueOccupancy {
    private var intervals: [(start: Double, end: Double)] = []

    var isEmpty: Bool { intervals.isEmpty }
    var count: Int { intervals.count }

    mutating func record(_ commandBuffer: MTLCommandBuffer) {
        record(start: commandBuffer.gpuStartTime, end: commandBuffer.gpuEndTime)
    }

    mutating func record(start: Double, end: Double) {
        guard start > 0, end > start else { return }
        intervals.append((start, end))
    }

    /// (busy, idle, span) in seconds over the union of the recorded intervals.
    var occupancy: (busy: Double, idle: Double, span: Double)? {
        guard !intervals.isEmpty else { return nil }
        let sorted = intervals.sorted { $0.start < $1.start }
        var busy: Double = 0
        var mergedStart = sorted[0].start
        var mergedEnd = sorted[0].end
        for interval in sorted.dropFirst() {
            if interval.start > mergedEnd {
                busy += mergedEnd - mergedStart
                mergedStart = interval.start
                mergedEnd = interval.end
            } else {
                mergedEnd = max(mergedEnd, interval.end)
            }
        }
        busy += mergedEnd - mergedStart
        let span = mergedEnd - sorted[0].start
        return (busy, span - busy, span)
    }

    /// One line in the shape of the other profile footers.
    func summary(label: String, extra: String = "") -> String? {
        guard let gpu = occupancy else { return nil }
        return String(format: "[%@ busy=%.3fs(%.0f%%) idle=%.3fs(%.0f%%) span=%.3fs buffers=%d%@]",
                      label,
                      gpu.busy, gpu.span > 0 ? gpu.busy / gpu.span * 100 : 0,
                      gpu.idle, gpu.span > 0 ? gpu.idle / gpu.span * 100 : 0,
                      gpu.span, intervals.count, extra)
    }

    mutating func reset() {
        intervals.removeAll(keepingCapacity: true)
    }
}
