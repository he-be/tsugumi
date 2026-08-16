import Darwin
import Foundation

/// The byte-moving half of a range copy, shared by every `SourceByteProvider`.
/// Where the bytes come from differs (ranged HTTP into a temporary file, or a
/// staged shard read in place); what happens to them afterwards — the tiled
/// pread/pwrite and the digest that proves a range landed — does not.
enum RangeCopyIO {
    static func copyBytes(sourceFD: Int32,
                          sourcePath: String,
                          destinationFD: Int32,
                          destinationPath: String,
                          sourceOffset: UInt64,
                          destinationOffset: UInt64,
                          size: UInt64,
                          scratch: UnsafeMutableRawBufferPointer,
                          audit: RepackAudit) throws {
        var remaining = size
        var source = sourceOffset
        var destination = destinationOffset
        while remaining > 0 {
            try Task.checkCancellation()
            let count = min(Int(remaining), scratch.count)
            try Posix.preadAll(
                fd: sourceFD,
                path: sourcePath,
                buf: scratch.baseAddress!,
                count: count,
                offset: source)
            try Posix.pwriteAll(
                fd: destinationFD,
                path: destinationPath,
                buf: scratch.baseAddress!,
                count: count,
                offset: destination)
            audit.recordTile(bytes: count)
            audit.recordRead(bytes: count)
            audit.recordWrite(bytes: count)
            remaining -= UInt64(count)
            source += UInt64(count)
            destination += UInt64(count)
        }
    }

    /// Digest of everything a coalesced copy wrote, read back from the output
    /// files. The resume checkpoint stores it so a completed range can be
    /// trusted without re-fetching its source bytes.
    static func destinationDigest(
        _ copy: CoalescedRangeCopy,
        partialDirectory: String,
        scratch suppliedScratch: UnsafeMutableRawBufferPointer? = nil
    ) throws -> String {
        let scratch = suppliedScratch ?? UnsafeMutableRawBufferPointer.allocate(
            byteCount: WriterCore.tileBytes,
            alignment: 16_384)
        defer {
            if suppliedScratch == nil { scratch.deallocate() }
        }

        var digest = DestinationDigest(copy: copy)
        for destination in copy.destinations {
            digest.append(try RangeCopyPlanner.normalizedRelativePath(
                destination.destinationPath,
                root: partialDirectory))
            digest.append(destination.destinationOffset)
            digest.append(destination.sourceOffset - copy.sourceOffset)
            digest.append(destination.size)

            let descriptor = try Posix.openReadNoFollow(destination.destinationPath)
            defer { close(descriptor) }
            var remaining = destination.size
            var offset = destination.destinationOffset
            while remaining > 0 {
                let count = min(Int(remaining), scratch.count)
                try Posix.preadAll(
                    fd: descriptor,
                    path: destination.destinationPath,
                    buf: scratch.baseAddress!,
                    count: count,
                    offset: offset)
                digest.append(UnsafeRawBufferPointer(
                    start: scratch.baseAddress,
                    count: count))
                remaining -= UInt64(count)
                offset += UInt64(count)
            }
        }
        return digest.finalize()
    }
}

private struct DestinationDigest {
    private var stream = Sha256Stream()

    init(copy: CoalescedRangeCopy) {
        append("TurboFieldfare.RemoteRangeDestination.v1")
        append(copy.id)
        append(UInt64(copy.destinations.count))
    }

    mutating func append(_ value: UInt64) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { stream.update($0) }
    }

    mutating func append(_ value: String) {
        let data = Data(value.utf8)
        append(UInt64(data.count))
        data.withUnsafeBytes { stream.update($0) }
    }

    mutating func append(_ bytes: UnsafeRawBufferPointer) {
        stream.update(bytes)
    }

    func finalize() -> String {
        stream.finalizeHexString()
    }
}
