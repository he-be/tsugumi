import Foundation
import CommonCrypto
import Darwin

public enum Sha256Verifier {

    /// Compute the lowercase-hex SHA-256 of the entire file at `fileURL` by
    /// streaming through a fixed-size scratch read. Does not allocate the
    /// whole file.
    public static func hashFile(at fileURL: URL,
                                chunkBytes: Int = 1 << 20) throws -> String {
        let fd = open(fileURL.path, O_RDONLY)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(fileURL.path))", errno: errno)
        }
        defer { close(fd) }

        return try hashFile(fileDescriptor: fd, displayName: fileURL.path,
                            chunkBytes: chunkBytes)
    }

    package static func hashFile(fileDescriptor fd: Int32,
                                 displayName: String,
                                 chunkBytes: Int = 1 << 20) throws -> String {
        guard chunkBytes > 0 else {
            throw ModelError.indexCorrupt(detail: "SHA-256 chunk size must be positive")
        }
        var ctx = CC_SHA256_CTX()
        CC_SHA256_Init(&ctx)
        var buf = [UInt8](repeating: 0, count: chunkBytes)
        while true {
            let got: Int = buf.withUnsafeMutableBytes { raw -> Int in
                return read(fd, raw.baseAddress!, chunkBytes)
            }
            if got == 0 { break }
            if got < 0, errno == EINTR { continue }
            if got < 0 {
                throw ModelError.posixFailed(call: "read(\(displayName))", errno: errno)
            }
            buf.withUnsafeBytes { raw in
                _ = CC_SHA256_Update(&ctx, raw.baseAddress!, CC_LONG(got))
            }
        }
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        digest.withUnsafeMutableBytes { raw in
            _ = CC_SHA256_Final(raw.baseAddress!.assumingMemoryBound(to: UInt8.self), &ctx)
        }
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// SHA-256 of the bytes the segments name, in order. A layer file whose
    /// experts are split between the model directory and a mirror copy
    /// (docs/m6-prefill/44) is hashed as the bytes the runtime will actually
    /// read, so it matches the manifest's whole-file digest exactly when those
    /// bytes are right.
    ///
    /// The segments jump between two files, so neither gets the kernel's
    /// sequential read-ahead that a plain `read` loop does. Instead the
    /// segments are cut into pieces of at most `pieceBytes`, and a window of
    /// `windowPieces` is read with every piece in flight at once — both drives
    /// busy — while the previous window is being hashed.
    package static func hashSegments(_ segments: [(fd: Int32, offset: UInt64, count: UInt64)],
                                     displayName: String,
                                     pieceBytes: Int = 4 << 20,
                                     windowPieces: Int = 16) throws -> String {
        guard pieceBytes > 0, windowPieces > 0 else {
            throw ModelError.indexCorrupt(detail: "SHA-256 piece and window must be positive")
        }
        var pieces: [(fd: Int32, offset: UInt64, count: Int)] = []
        for segment in segments {
            var done: UInt64 = 0
            while done < segment.count {
                let count = Int(min(UInt64(pieceBytes), segment.count - done))
                pieces.append((segment.fd, segment.offset + done, count))
                done += UInt64(count)
            }
        }
        let windows = stride(from: 0, to: pieces.count, by: windowPieces).map {
            Array(pieces[$0..<min($0 + windowPieces, pieces.count)])
        }
        let capacity = windowPieces * pieceBytes
        let buffers = [UnsafeMutableRawPointer.allocate(byteCount: max(1, capacity), alignment: 16384),
                       UnsafeMutableRawPointer.allocate(byteCount: max(1, capacity), alignment: 16384)]
        defer { buffers.forEach { $0.deallocate() } }

        final class ReadError: @unchecked Sendable {
            let lock = NSLock()
            var error: Error?
            func record(_ e: Error) { lock.lock(); if error == nil { error = e }; lock.unlock() }
        }
        func readWindow(_ window: [(fd: Int32, offset: UInt64, count: Int)],
                        into base: UnsafeMutableRawPointer, errors: ReadError) {
            var starts: [Int] = []
            var at = 0
            for piece in window { starts.append(at); at += piece.count }
            nonisolated(unsafe) let base = base
            DispatchQueue.concurrentPerform(iterations: window.count) { index in
                let piece = window[index]
                var filled = 0
                while filled < piece.count {
                    let got = pread(piece.fd, base.advanced(by: starts[index] + filled),
                                    piece.count - filled, off_t(piece.offset) + off_t(filled))
                    if got < 0, errno == EINTR { continue }
                    if got < 0 {
                        errors.record(ModelError.posixFailed(call: "pread(\(displayName))", errno: errno))
                        return
                    }
                    if got == 0 {
                        errors.record(ModelError.tensorSizeMismatch(
                            name: displayName, expected: piece.offset + UInt64(piece.count),
                            actual: piece.offset + UInt64(filled)))
                        return
                    }
                    filled += got
                }
            }
        }

        var ctx = CC_SHA256_CTX()
        CC_SHA256_Init(&ctx)
        let errors = ReadError()
        let reader = DispatchQueue(label: "tsugumi.sha256.segments")
        if let first = windows.first { readWindow(first, into: buffers[0], errors: errors) }
        for (index, window) in windows.enumerated() {
            if let error = errors.error { throw error }
            let group = DispatchGroup()
            if index + 1 < windows.count {
                let next = windows[index + 1]
                nonisolated(unsafe) let target = buffers[(index + 1) % 2]
                reader.async(group: group) { readWindow(next, into: target, errors: errors) }
            }
            let bytes = window.reduce(0) { $0 + $1.count }
            _ = CC_SHA256_Update(&ctx, buffers[index % 2], CC_LONG(bytes))
            group.wait()
        }
        if let error = errors.error { throw error }
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        digest.withUnsafeMutableBytes { raw in
            _ = CC_SHA256_Final(raw.baseAddress!.assumingMemoryBound(to: UInt8.self), &ctx)
        }
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    public static func hashData(_ data: Data) -> String {
        var ctx = CC_SHA256_CTX()
        CC_SHA256_Init(&ctx)
        data.withUnsafeBytes { raw in
            if let base = raw.baseAddress, raw.count > 0 {
                _ = CC_SHA256_Update(&ctx, base, CC_LONG(raw.count))
            }
        }
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        digest.withUnsafeMutableBytes { raw in
            _ = CC_SHA256_Final(raw.baseAddress!.assumingMemoryBound(to: UInt8.self), &ctx)
        }
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Throw `ModelError.checksumMismatch(file)` if the on-disk file's
    /// SHA-256 does not match `expectedHex`. Hex comparison is
    /// case-insensitive on the expected side (writer outputs lowercase).
    public static func verifyFile(at fileURL: URL,
                                  named name: String,
                                  expectedHex: String) throws {
        let actual = try hashFile(at: fileURL)
        if actual.lowercased() != expectedHex.lowercased() {
            throw ModelError.checksumMismatch(file: name)
        }
    }

    package static func verifyFile(fileDescriptor fd: Int32,
                                   named name: String,
                                   expectedHex: String) throws {
        let actual = try hashFile(fileDescriptor: fd, displayName: name)
        if actual.lowercased() != expectedHex.lowercased() {
            throw ModelError.checksumMismatch(file: name)
        }
    }
}
