import Darwin
import Foundation

/// Which part of a generation a routed-expert fetch belongs to. The runner
/// stamps the phase before each forward pass so that one counter set can serve
/// both the chunked prefill and the decode loop without the fetch path needing
/// to know which one is running.
public enum ExpertPhase: Int, Sendable, CaseIterable {
    case prefill = 0
    case decode = 1

    public var name: String {
        switch self {
        case .prefill: return "prefill"
        case .decode: return "decode"
        }
    }
}

/// Routed-expert accounting for one phase. `experts` counts requested expert
/// slots (top-k per token for decode, unique experts per tile for prefill), so
/// `hits + misses == experts`.
public struct ExpertPhaseCounters: Sendable, Equatable {
    public var fetches: Int = 0
    public var experts: Int = 0
    public var hits: Int = 0
    public var misses: Int = 0
    public var fetchNanos: UInt64 = 0

    public init() {}

    public var hitRate: Double {
        experts > 0 ? Double(hits) / Double(experts) : 0
    }
}

public struct ExpertTelemetrySnapshot: Sendable, Equatable {
    public var layersOpened: Int = 0
    /// Wall time inside `openLayerLocked`, including verification.
    public var layerOpenNanos: UInt64 = 0
    /// Wall time spent hashing layer files under `.fullSha256`.
    public var layerVerifyNanos: UInt64 = 0
    public var layerVerifiedBytes: UInt64 = 0
    public var prefill = ExpertPhaseCounters()
    public var decode = ExpertPhaseCounters()
    public var traceRecords: Int = 0

    public init() {}

    public var layerVerifySeconds: Double { Double(layerVerifyNanos) / 1e9 }
    public var layerOpenSeconds: Double { Double(layerOpenNanos) / 1e9 }
}

/// Process-wide instrumentation for the routed-expert path.
///
/// One instance lives on `Model` and is shared by every copy of that struct, so
/// the runner, the prefill kernels and the CLI footer all see the same
/// counters. Everything is guarded by one lock; the call rate is 30 fetches per
/// token, which is far below where the lock could matter.
public final class ExpertTelemetry: @unchecked Sendable {
    private let lock = NSLock()
    private var counters = ExpertTelemetrySnapshot()
    private var phase: ExpertPhase = .prefill
    private var step = 0
    private var trace: ExpertTraceWriter?

    public init() {}

    // MARK: - Phase stamping

    public func beginPhase(_ phase: ExpertPhase, step: Int) {
        lock.lock()
        self.phase = phase
        self.step = step
        lock.unlock()
    }

    public var currentPhase: ExpertPhase {
        lock.lock()
        defer { lock.unlock() }
        return phase
    }

    // MARK: - Recording

    public func recordLayerOpen(totalNanos: UInt64, verifyNanos: UInt64, verifiedBytes: UInt64) {
        lock.lock()
        counters.layersOpened += 1
        counters.layerOpenNanos &+= totalNanos
        counters.layerVerifyNanos &+= verifyNanos
        counters.layerVerifiedBytes &+= verifiedBytes
        lock.unlock()
    }

    public func recordFetch(layer: Int,
                            experts: [Int],
                            hits: Int,
                            misses: Int,
                            nanos: UInt64) {
        lock.lock()
        let phase = self.phase
        let step = self.step
        let bucket: WritableKeyPath<ExpertTelemetrySnapshot, ExpertPhaseCounters> =
            phase == .prefill ? \.prefill : \.decode
        counters[keyPath: bucket].fetches += 1
        counters[keyPath: bucket].experts += experts.count
        counters[keyPath: bucket].hits += hits
        counters[keyPath: bucket].misses += misses
        counters[keyPath: bucket].fetchNanos &+= nanos
        if let trace {
            trace.append(phase: phase,
                         step: step,
                         layer: layer,
                         experts: experts,
                         hits: hits,
                         misses: misses)
            counters.traceRecords += 1
        }
        lock.unlock()
    }

    public func snapshot() -> ExpertTelemetrySnapshot {
        lock.lock()
        defer { lock.unlock() }
        return counters
    }

    public func reset() {
        lock.lock()
        counters = ExpertTelemetrySnapshot()
        lock.unlock()
    }

    // MARK: - Trace

    /// Start recording every routed-expert request to `path`. One trace is
    /// enough to replay the request stream offline against any slot count and
    /// eviction policy, which is why the per-run hit-rate sweep does not need a
    /// rebuild per configuration.
    public func startTrace(path: String, header: [String: String] = [:]) throws {
        let writer = try ExpertTraceWriter(path: path, header: header)
        lock.lock()
        trace = writer
        lock.unlock()
    }

    public func finishTrace() {
        lock.lock()
        let writer = trace
        trace = nil
        lock.unlock()
        writer?.close()
    }

    public var isTracing: Bool {
        lock.lock()
        defer { lock.unlock() }
        return trace != nil
    }
}

/// Append-only TSV writer for the expert trace. Buffered so the hot path costs
/// a string interpolation and a memcpy rather than a syscall.
final class ExpertTraceWriter {
    private let handle: FileHandle
    private var buffer = Data()
    private static let flushThreshold = 1 << 20

    init(path: String, header: [String: String]) throws {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        guard FileManager.default.createFile(atPath: path, contents: nil) else {
            throw ExpertTraceError.cannotCreate(path: path)
        }
        handle = try FileHandle(forWritingTo: url)
        var preamble = "# tsugumi expert trace v1\n"
        for key in header.keys.sorted() {
            preamble += "# \(key)=\(header[key]!)\n"
        }
        preamble += "phase\tstep\tlayer\thits\tmisses\texperts\n"
        buffer.append(Data(preamble.utf8))
    }

    func append(phase: ExpertPhase,
                step: Int,
                layer: Int,
                experts: [Int],
                hits: Int,
                misses: Int) {
        var line = "\(phase.name)\t\(step)\t\(layer)\t\(hits)\t\(misses)\t"
        for (index, expert) in experts.enumerated() {
            if index > 0 { line += "," }
            line += "\(expert)"
        }
        line += "\n"
        buffer.append(Data(line.utf8))
        if buffer.count >= Self.flushThreshold { flush() }
    }

    func close() {
        flush()
        try? handle.close()
    }

    private func flush() {
        guard !buffer.isEmpty else { return }
        try? handle.write(contentsOf: buffer)
        buffer.removeAll(keepingCapacity: true)
    }
}

public enum ExpertTraceError: Error, CustomStringConvertible {
    case cannotCreate(path: String)

    public var description: String {
        switch self {
        case .cannotCreate(let path): return "cannot create expert trace at \(path)"
        }
    }
}

/// Peak memory as the OS accounts it. `phys_footprint` is the number Activity
/// Monitor shows and the one the 12 GB budget is written against;
/// `resident_size_max` is the high-water mark of resident pages.
public struct ProcessMemoryFootprint: Sendable, Equatable {
    public let physFootprintBytes: UInt64
    public let peakPhysFootprintBytes: UInt64
    public let peakResidentBytes: UInt64

    public static func current() -> ProcessMemoryFootprint {
        var vmInfo = task_vm_info_data_t()
        var vmCount = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let vmResult = withUnsafeMutablePointer(to: &vmInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &vmCount)
            }
        }

        var basicInfo = mach_task_basic_info_data_t()
        var basicCount = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<natural_t>.size)
        let basicResult = withUnsafeMutablePointer(to: &basicInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(basicCount)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &basicCount)
            }
        }

        return ProcessMemoryFootprint(
            physFootprintBytes: vmResult == KERN_SUCCESS ? UInt64(vmInfo.phys_footprint) : 0,
            peakPhysFootprintBytes: vmResult == KERN_SUCCESS
                ? UInt64(vmInfo.ledger_phys_footprint_peak)
                : 0,
            peakResidentBytes: basicResult == KERN_SUCCESS
                ? UInt64(basicInfo.resident_size_max)
                : 0)
    }
}
