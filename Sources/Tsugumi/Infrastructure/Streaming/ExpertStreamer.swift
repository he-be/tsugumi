import Foundation

enum StreamerError: Error, CustomStringConvertible {
    case openFailed(path: String, errno: Int32)
    case sizeMismatch(expected: UInt64, actual: UInt64)
    case offsetOutOfRange(UInt64)
    case bufferWrapFailed
    case preadFailed(errno: Int32)
    case allocFailed(errno: Int32)
    case slotOutOfRange(Int)
    case invalidIOSplitConfiguration(String)

    public var description: String {
        switch self {
        case .openFailed(let path, let error):
            return "open(\(path)) failed: errno \(error)"
        case .sizeMismatch(let expected, let actual):
            return "file size mismatch: expected \(expected), got \(actual)"
        case .offsetOutOfRange(let offset):
            return "offset \(offset) is outside the streamed range"
        case .bufferWrapFailed:
            return "failed to wrap expert cache memory in an MTLBuffer"
        case .preadFailed(let error):
            return "pread failed: errno \(error)"
        case .allocFailed(let error):
            return "posix_memalign failed: errno \(error)"
        case .slotOutOfRange(let slot):
            return "expert cache slot \(slot) is out of range"
        case .invalidIOSplitConfiguration(let detail):
            return "invalid expert I/O split configuration: \(detail)"
        }
    }
}

/// Byte layout of one routed-expert layer file.
public struct StreamLayout: Sendable {
    public let path: String
    public let streamOffset: UInt64
    public let streamSize: UInt64
    public let expertsPerLayer: Int
    public let expertStride: UInt64
    public let expertOffsets: [UInt64]?

    public init(path: String,
                streamOffset: UInt64,
                streamSize: UInt64,
                expertsPerLayer: Int,
                expertStride: UInt64,
                expertOffsets: [UInt64]? = nil) {
        self.path = path
        self.streamOffset = streamOffset
        self.streamSize = streamSize
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.expertOffsets = expertOffsets
    }

    @inline(__always)
    public func expertOffset(layer: Int, expert: Int) -> UInt64 {
        if layer == 0, let expertOffsets, expert >= 0, expert < expertOffsets.count {
            return expertOffsets[expert]
        }
        let perLayer = UInt64(expertsPerLayer) * expertStride
        return UInt64(layer) * perLayer + UInt64(expert) * expertStride
    }
}

/// A copy of one layer file on a second SSD, and which experts to read from it
/// (docs/m6-prefill/44). The two drives are read at the same time, so their
/// bandwidths add; each expert is always read from the same side.
public struct ExpertMirror: Sendable {
    /// Borrowed; the streamer duplicates it.
    public let fileDescriptor: Int32
    public let onMirror: [Bool]

    public init(fileDescriptor: Int32, onMirror: [Bool]) {
        self.fileDescriptor = fileDescriptor
        self.onMirror = onMirror
    }

    /// `share` of `count` experts, spread evenly over the index range:
    /// expert `e` is on the mirror when `floor((e + 1) * share)` steps past
    /// `floor(e * share)`. 0 puts none there, 1 puts all.
    public static func evenAssignment(count: Int, share: Double) -> [Bool] {
        let clamped = min(1, max(0, share))
        return (0..<count).map { e in
            Int((Double(e + 1) * clamped).rounded(.down)) != Int((Double(e) * clamped).rounded(.down))
        }
    }
}
