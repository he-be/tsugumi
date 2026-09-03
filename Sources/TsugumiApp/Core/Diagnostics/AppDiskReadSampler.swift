import Darwin
import Foundation

/// What a round really read from the SSD, in bytes, split at the first
/// token: the prefill's share and the decode's share. Measured, not
/// modelled — `proc_pid_rusage` charges every page the kernel fetched from
/// disk for this process, page-cache misses of the mmapped weights included.
/// Logged per round (`AppTurnMetricsRecord`) so the speedometer's claim,
/// "RAM crowded by other apps means the weights come from the SSD", can be
/// checked against ordinary use instead of a lab hog (docs/MAC_APP.md §4e).
/// Not shown in the HUD: the dial is about the cause, this is the effect.
public struct AppDiskReadDiagnostics: Equatable, Sendable {
    public var prefillBytes: UInt64
    public var decodeBytes: UInt64

    public init(prefillBytes: UInt64, decodeBytes: UInt64) {
        self.prefillBytes = prefillBytes
        self.decodeBytes = decodeBytes
    }
}

/// The process's cumulative disk reads (`ri_diskio_bytesread`). Differences
/// between two samples are what a phase cost; the sampler itself keeps no
/// state.
public enum AppDiskReadSampler {
    public static func bytesRead(pid: pid_t = getpid()) -> UInt64? {
        var info = rusage_info_v4()
        // `rusage_info_t *` is spelled as a pointer to a pointer, but the
        // kernel writes the struct at the address it is handed — callers pass
        // `(rusage_info_t *)&info`. A pointer to a pointer here overruns the
        // stack.
        let rc = withUnsafeMutablePointer(to: &info) { pointer -> Int32 in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(pid, RUSAGE_INFO_V4, $0)
            }
        }
        guard rc == 0 else { return nil }
        return info.ri_diskio_bytesread
    }

    /// The three readings a round takes — at its start, at its first token,
    /// at its end — turned into the two shares. Nil when any reading failed.
    public static func diagnostics(start: UInt64?, firstToken: UInt64?, end: UInt64?)
        -> AppDiskReadDiagnostics? {
        guard let start, let end else { return nil }
        // A round that never produced a token (cancelled in prefill, or an
        // error) has only a prefill share.
        let split = firstToken ?? end
        return AppDiskReadDiagnostics(prefillBytes: split >= start ? split - start : 0,
                                      decodeBytes: end >= split ? end - split : 0)
    }
}
