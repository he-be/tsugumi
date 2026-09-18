import Darwin
import Foundation
import IOKit
import TsugumiAppCore

/// What one local gather round read and swapped (docs/qwen38/42 §3-2), as differences of three counters taken at the
/// round's start and end:
///   `disk_read_bytes`         this process's `ri_diskio_bytesread` (`AppDiskReadSampler`, the app's own figure)
///   `system_disk_read_bytes`  every `IOBlockStorageDriver`'s `Bytes (Read)`, the counter `iostat` reads, all
///                             processes and disks
///   `swapouts`                `vm_statistics64.swapouts` (pages)
final class GatherRoundMeter {
    private let process = AppDiskReadSampler.bytesRead()
    private let system = GatherRoundMeter.systemDiskReadBytes()
    private let swapouts = GatherRoundMeter.swapouts()

    func finish() -> [String: Any] {
        var row: [String: Any] = [:]
        if let a = process, let b = AppDiskReadSampler.bytesRead() { row["disk_read_bytes"] = b &- a }
        if let a = system, let b = Self.systemDiskReadBytes() { row["system_disk_read_bytes"] = b &- a }
        if let a = swapouts, let b = Self.swapouts() { row["swapouts"] = b &- a }
        return row
    }

    static func systemDiskReadBytes() -> UInt64? {
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOBlockStorageDriver"),
                                           &iterator) == KERN_SUCCESS else { return nil }
        defer { IOObjectRelease(iterator) }
        var total: UInt64 = 0
        var found = false
        while case let service = IOIteratorNext(iterator), service != 0 {
            defer { IOObjectRelease(service) }
            guard let stats = IORegistryEntryCreateCFProperty(service, "Statistics" as CFString, kCFAllocatorDefault, 0)?
                .takeRetainedValue() as? [String: Any],
                  let read = stats["Bytes (Read)"] as? NSNumber else { continue }
            total &+= read.uint64Value
            found = true
        }
        return found ? total : nil
    }

    static func swapouts() -> UInt64? {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let rc = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        return rc == KERN_SUCCESS ? stats.swapouts : nil
    }
}
