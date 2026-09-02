import Darwin
import Darwin.Mach
import Foundation

/// The machine's memory, split the way Activity Monitor splits it. The
/// model borrows the page cache for its expert weights (`MmapExpertMapping`),
/// so what decides decode speed is not "free" — after one answer the cache
/// is full of weights and `free` reads near zero at the fastest point — but
/// how much of physical memory is *not* held by apps, the kernel and the
/// compressor. That remainder is what the cache can hold.
public struct AppHostMemory: Equatable, Sendable {
    public var physicalBytes: UInt64
    /// Anonymous pages minus purgeable: what apps (this one included) own.
    public var appBytes: UInt64
    public var wiredBytes: UInt64
    public var compressedBytes: UInt64
    /// File-backed pages plus purgeable: the page cache, weights included.
    public var cachedFileBytes: UInt64

    public init(physicalBytes: UInt64, appBytes: UInt64, wiredBytes: UInt64,
                compressedBytes: UInt64, cachedFileBytes: UInt64) {
        self.physicalBytes = physicalBytes
        self.appBytes = appBytes
        self.wiredBytes = wiredBytes
        self.compressedBytes = compressedBytes
        self.cachedFileBytes = cachedFileBytes
    }

    /// Activity Monitor's "Memory Used".
    public var usedBytes: UInt64 { appBytes &+ wiredBytes &+ compressedBytes }

    /// What is left for the page cache: physical memory nobody holds.
    public var borrowableBytes: UInt64 {
        physicalBytes > usedBytes ? physicalBytes - usedBytes : 0
    }

    /// `host_statistics64` page counts, as `vm_stat` prints them.
    public init(pageSize: UInt64, physicalBytes: UInt64,
                anonymousPages: UInt64, purgeablePages: UInt64,
                wiredPages: UInt64, compressorPages: UInt64,
                fileBackedPages: UInt64) {
        let app = anonymousPages > purgeablePages ? anonymousPages - purgeablePages : 0
        self.init(physicalBytes: physicalBytes,
                  appBytes: app * pageSize,
                  wiredBytes: wiredPages * pageSize,
                  compressedBytes: compressorPages * pageSize,
                  cachedFileBytes: (fileBackedPages + purgeablePages) * pageSize)
    }
}

/// Reads `AppHostMemory` from the kernel. Injectable for tests.
public final class AppHostMemorySampler: Sendable {
    private let read: @Sendable () -> AppHostMemory?

    public init() {
        self.read = { Self.readHostMemory() }
    }

    public init(read: @escaping @Sendable () -> AppHostMemory?) {
        self.read = read
    }

    public func sample() -> AppHostMemory? { read() }

    private static func readHostMemory() -> AppHostMemory? {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return AppHostMemory(
            pageSize: UInt64(sysconf(_SC_PAGESIZE)),
            physicalBytes: ProcessInfo.processInfo.physicalMemory,
            anonymousPages: UInt64(stats.internal_page_count),
            purgeablePages: UInt64(stats.purgeable_count),
            wiredPages: UInt64(stats.wire_count),
            compressorPages: UInt64(stats.compressor_page_count),
            fileBackedPages: UInt64(stats.external_page_count))
    }
}

/// The gauge: how much of the weights the model streams can live in memory
/// right now. 1.0 means the whole model fits and decode runs as fast as this
/// Mac goes; lower means a share of every token's experts comes from the
/// SSD. Nothing breaks below 1.0, it only gets slower.
public struct AppMachineHeadroom: Equatable, Sendable {
    public let host: AppHostMemory
    /// The bytes the model streams (its expert files): what the cache would
    /// hold if it could.
    public let wantedBytes: UInt64

    public init(host: AppHostMemory, wantedBytes: UInt64) {
        self.host = host
        self.wantedBytes = wantedBytes
    }

    public var borrowableBytes: UInt64 { host.borrowableBytes }

    /// 0...1: the borrowable share of what the model wants.
    public var level: Double {
        guard wantedBytes > 0 else { return 1 }
        return min(1, Double(borrowableBytes) / Double(wantedBytes))
    }

    /// The bytes of the model that would not fit, 0 when everything does.
    public var shortfallBytes: UInt64 {
        wantedBytes > borrowableBytes ? wantedBytes - borrowableBytes : 0
    }

    public enum Band: Equatable, Sendable {
        /// The model fits, give or take the last few hundred megabytes.
        case full
        /// Part of it fits. The hot experts stay wired either way, so a
        /// narrow task hardly slows; a broad one loses some (measured
        /// −16% at 0.23 on an 18 GB Mac, docs/MAC_APP.md §4e).
        case partial
        /// Below a quarter: on the 18 GB Mac this is where macOS began
        /// swapping the other apps rather than give up more page cache.
        /// The model keeps going; the rest of the machine pays.
        case tight
    }

    public static let fullThreshold = 0.9
    public static let tightThreshold = 0.25

    public var band: Band {
        if level >= Self.fullThreshold { return .full }
        if level >= Self.tightThreshold { return .partial }
        return .tight
    }

    /// The streamed bytes of a model directory: the expert files under
    /// `packed_experts`. Nil when the directory has none (not installed,
    /// or a layout this build does not know), so the caller can fall back.
    public static func streamedWeightBytes(modelDirectory: URL) -> UInt64? {
        let experts = modelDirectory.appendingPathComponent("packed_experts", isDirectory: true)
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: experts.path),
              !names.isEmpty else { return nil }
        var total: UInt64 = 0
        for name in names {
            let path = experts.appendingPathComponent(name).path
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
                  let size = attributes[.size] as? UInt64 else { continue }
            total += size
        }
        return total > 0 ? total : nil
    }
}
