import Foundation
import Metal

/// Read-only GGUF v3 container: metadata, tensor directory, and a whole-file
/// `mmap` that tensors are addressed into.
///
/// Written for the Qwen3.8-Flash-Next Q2 verification
/// (`docs/investigations/QWEN38_FLASH_NEXT_VERIFY_PLAN.md`), which runs straight off
/// the DS4-IQ2 GGUF rather than repacking 41 GiB into a `.moepack`. Only the
/// GGML types that checkpoint and its PLE / down sidecars use have byte sizes here (the down sidecar keeps
/// its 212 B Q2_K rows as I8 bytes, docs/qwen38/15 §2 W), plus the mixed K / IQ types of the
/// Qwen3.8-27B GSQ-RCO GGUF (docs/qwen38-27b/01 §3-1).
public final class GGUFFile: @unchecked Sendable {
    public enum GGMLType: UInt32, Sendable {
        case f32 = 0
        case f16 = 1
        case q4_0 = 2
        case q4_1 = 3
        case q8_0 = 8
        case q2_K = 10
        case q4_K = 12
        case q6_K = 14
        case iq2_xxs = 16
        case iq2_xs = 17
        case iq3_xxs = 18
        case iq3_s = 21
        case iq2_s = 22
        case iq4_xs = 23
        case i8 = 24
        case i64 = 27
        case iq1_m = 29
        case bf16 = 30
        case mxfp4 = 39

        /// (bytes per block, elements per block)
        public var blockLayout: (bytes: Int, elements: Int) {
            switch self {
            case .f32: return (4, 1)
            case .i8: return (1, 1)
            case .f16, .bf16: return (2, 1)
            case .i64: return (8, 1)
            case .q4_0: return (18, 32)
            case .q4_1: return (20, 32)
            case .q8_0: return (34, 32)
            case .mxfp4: return (17, 32)
            case .q2_K: return (84, 256)
            case .q4_K: return (144, 256)
            case .q6_K: return (210, 256)
            case .iq2_xxs: return (66, 256)
            case .iq2_xs: return (74, 256)
            case .iq3_xxs: return (98, 256)
            case .iq3_s: return (110, 256)
            case .iq2_s: return (82, 256)
            case .iq4_xs: return (136, 256)
            case .iq1_m: return (56, 256)
            }
        }
    }

    public enum Value: Sendable {
        case int(Int64)
        case uint(UInt64)
        case float(Double)
        case bool(Bool)
        case string(String)
        case array([Value])

        public var int: Int? {
            switch self {
            case .int(let v): return Int(v)
            case .uint(let v): return Int(v)
            default: return nil
            }
        }
        public var double: Double? {
            if case .float(let v) = self { return v }
            return int.map(Double.init)
        }
        public var string: String? {
            if case .string(let v) = self { return v }
            return nil
        }
        public var ints: [Int]? {
            if case .array(let a) = self { return a.compactMap(\.int) }
            return nil
        }
    }

    public struct Tensor: Sendable {
        public let name: String
        /// GGML order: dims[0] is the row width.
        public let dims: [Int]
        public let type: GGMLType
        /// Absolute file offset of the first byte.
        public let offset: Int
        public let byteCount: Int

        public var rowWidth: Int { dims[0] }
        public var rowCount: Int { dims.dropFirst().reduce(1, *) }
        public var bytesPerRow: Int {
            let layout = type.blockLayout
            return rowWidth / layout.elements * layout.bytes
        }
    }

    public enum Error: Swift.Error, CustomStringConvertible {
        case open(String)
        case format(String)
        case missingTensor(String)
        case missingKey(String)

        public var description: String {
            switch self {
            case .open(let s): return "GGUF open failed: \(s)"
            case .format(let s): return "GGUF format error: \(s)"
            case .missingTensor(let s): return "GGUF tensor missing: \(s)"
            case .missingKey(let s): return "GGUF key missing: \(s)"
            }
        }
    }

    public let url: URL
    public let metadata: [String: Value]
    public let tensors: [String: Tensor]
    public let fileSize: Int
    /// Whole-file read-only mapping.
    public let base: UnsafeRawPointer
    private let fd: Int32

    public init(url: URL) throws {
        self.url = url
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw Error.open("\(url.path): errno \(errno)") }
        var st = stat()
        guard fstat(fd, &st) == 0 else { close(fd); throw Error.open("fstat errno \(errno)") }
        let size = Int(st.st_size)
        guard let map = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0), map != MAP_FAILED else {
            close(fd)
            throw Error.open("mmap errno \(errno)")
        }
        let base = UnsafeRawPointer(map)

        var cursor = 0
        func need(_ n: Int) throws {
            guard cursor + n <= size else { throw Error.format("truncated at \(cursor)") }
        }
        func read<T>(_: T.Type) throws -> T {
            try need(MemoryLayout<T>.size)
            let v = base.loadUnaligned(fromByteOffset: cursor, as: T.self)
            cursor += MemoryLayout<T>.size
            return v
        }
        func readString() throws -> String {
            let n = Int(try read(UInt64.self))
            try need(n)
            let s = String(decoding: UnsafeRawBufferPointer(start: base + cursor, count: n), as: UTF8.self)
            cursor += n
            return s
        }
        func readValue(_ type: UInt32) throws -> Value {
            switch type {
            case 0: return .uint(UInt64(try read(UInt8.self)))
            case 1: return .int(Int64(try read(Int8.self)))
            case 2: return .uint(UInt64(try read(UInt16.self)))
            case 3: return .int(Int64(try read(Int16.self)))
            case 4: return .uint(UInt64(try read(UInt32.self)))
            case 5: return .int(Int64(try read(Int32.self)))
            case 6: return .float(Double(try read(Float.self)))
            case 7: return .bool(try read(UInt8.self) != 0)
            case 8: return .string(try readString())
            case 9:
                let elem = try read(UInt32.self)
                let n = Int(try read(UInt64.self))
                var out: [Value] = []
                out.reserveCapacity(min(n, 1 << 20))
                for _ in 0..<n { out.append(try readValue(elem)) }
                return .array(out)
            case 10: return .uint(try read(UInt64.self))
            case 11: return .int(try read(Int64.self))
            case 12: return .float(try read(Double.self))
            default: throw Error.format("value type \(type)")
            }
        }

        guard try read(UInt32.self) == 0x4655_4747 else { throw Error.format("bad magic") }
        let version = try read(UInt32.self)
        guard version == 3 else { throw Error.format("version \(version)") }
        let tensorCount = Int(try read(UInt64.self))
        let kvCount = Int(try read(UInt64.self))
        var meta: [String: Value] = [:]
        for _ in 0..<kvCount {
            let key = try readString()
            let type = try read(UInt32.self)
            // The tokenizer arrays are large and unused here; walk them without keeping them.
            let value = try readValue(type)
            if !key.hasPrefix("tokenizer.ggml.") { meta[key] = value }
        }
        struct Info { let name: String; let dims: [Int]; let type: UInt32; let offset: Int }
        var infos: [Info] = []
        for _ in 0..<tensorCount {
            let name = try readString()
            let nd = Int(try read(UInt32.self))
            var dims: [Int] = []
            for _ in 0..<nd { dims.append(Int(try read(UInt64.self))) }
            let type = try read(UInt32.self)
            let off = Int(try read(UInt64.self))
            infos.append(Info(name: name, dims: dims, type: type, offset: off))
        }
        let alignment = meta["general.alignment"]?.int ?? 32
        let dataStart = (cursor + alignment - 1) / alignment * alignment
        var table: [String: Tensor] = [:]
        for info in infos {
            guard let type = GGMLType(rawValue: info.type) else {
                throw Error.format("\(info.name): unsupported GGML type \(info.type)")
            }
            let layout = type.blockLayout
            let elements = info.dims.reduce(1, *)
            let bytes = elements / layout.elements * layout.bytes
            let t = Tensor(name: info.name, dims: info.dims, type: type,
                           offset: dataStart + info.offset, byteCount: bytes)
            guard t.offset + bytes <= size else { throw Error.format("\(info.name) past EOF") }
            table[info.name] = t
        }
        self.fd = fd
        self.fileSize = size
        self.base = base
        self.metadata = meta
        self.tensors = table
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: base), fileSize)
        close(fd)
    }

    public func tensor(_ name: String) throws -> Tensor {
        guard let t = tensors[name] else { throw Error.missingTensor(name) }
        return t
    }

    public func value(_ key: String) throws -> Value {
        guard let v = metadata[key] else { throw Error.missingKey(key) }
        return v
    }

    /// A page-aligned, no-copy `MTLBuffer` over `byteCount` bytes at file
    /// `offset`. Returns the buffer and the offset of `offset` inside it.
    /// The pages stay file-backed: nothing is copied or wired until Metal
    /// touches them.
    public func noCopyBuffer(device: MTLDevice, offset: Int, byteCount: Int)
        -> (buffer: MTLBuffer, offset: Int)? {
        let page = Int(getpagesize())
        let start = offset / page * page
        let end = min((offset + byteCount + page - 1) / page * page,
                      (fileSize + page - 1) / page * page)
        guard let buffer = device.makeBuffer(
            bytesNoCopy: UnsafeMutableRawPointer(mutating: base + start),
            length: end - start, options: .storageModeShared, deallocator: nil) else {
            return nil
        }
        return (buffer, offset - start)
    }

    /// Bytes of `offset ..< offset + byteCount` (whole pages) that are not in the page cache.
    public func nonResidentBytes(offset: Int, byteCount: Int) -> Int {
        let page = Int(getpagesize())
        let start = offset / page * page
        let end = min((offset + byteCount + page - 1) / page * page, (fileSize + page - 1) / page * page)
        let pages = (end - start) / page
        var vec = [CChar](repeating: 0, count: pages)
        guard mincore(base + start, end - start, &vec) == 0 else { return 0 }
        return vec.reduce(0) { $0 + (($1 & CChar(MINCORE_INCORE)) == 0 ? page : 0) }
    }

    /// `F_RDADVISE` for a file range (asynchronous read-ahead into the page cache).
    public func adviseRead(offset: Int, byteCount: Int) {
        var ra = radvisory(ra_offset: off_t(offset), ra_count: Int32(clamping: byteCount))
        _ = fcntl(fd, F_RDADVISE, &ra)
    }

    /// Reads file ranges with `pread` on `threads` threads in `blockBytes` pieces and throws the bytes
    /// away: the point is the page cache the mapping shares. Cold expert ranges of the Qwen3.8 GGUF read
    /// this way at 6.2-6.6 GB/s on the M3 Pro, against 0.7 GB/s faulting them through the mapping and
    /// 1.0 GB/s after `F_RDADVISE` (`docs/qwen38/05-EXPERT-READ.md`).
    public func preadRanges(_ ranges: [(offset: Int, byteCount: Int)], threads: Int, blockBytes: Int = 16 << 20) {
        GGUFFile.preadRanges(ranges.map { (self, $0.offset, $0.byteCount) }, threads: threads, blockBytes: blockBytes)
    }

    /// `preadRanges` over ranges of several files (a GGUF and its sidecars) on one set of threads.
    public static func preadRanges(_ ranges: [(file: GGUFFile, offset: Int, byteCount: Int)], threads: Int,
                                   blockBytes: Int = 16 << 20) {
        var blocks: [(Int32, Int, Int)] = []
        for r in ranges {
            var o = r.offset
            let end = min(r.offset + r.byteCount, r.file.fileSize)
            while o < end {
                blocks.append((r.file.fd, o, min(blockBytes, end - o)))
                o += blockBytes
            }
        }
        guard !blocks.isEmpty else { return }
        let n = max(1, min(threads, blocks.count))
        DispatchQueue.concurrentPerform(iterations: n) { lane in
            let buffer = UnsafeMutableRawPointer.allocate(byteCount: blockBytes, alignment: 16384)
            defer { buffer.deallocate() }
            var i = lane
            while i < blocks.count {
                var (fd, o, left) = blocks[i]
                while left > 0 {
                    let r = pread(fd, buffer, left, off_t(o))
                    if r <= 0 { break }
                    o += r
                    left -= r
                }
                i += n
            }
        }
    }

    public func noCopyBuffer(device: MTLDevice, tensor: Tensor) -> (buffer: MTLBuffer, offset: Int)? {
        noCopyBuffer(device: device, offset: tensor.offset, byteCount: tensor.byteCount)
    }
}
