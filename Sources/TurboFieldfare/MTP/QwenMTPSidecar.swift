import Darwin
import Foundation
import Metal

/// The grafted MTP head, as a file the runtime can map
/// (`docs/qwen35moe/36-MTP-DECODE.md` §1).
///
/// The head is **not in the `.gturbo`**: `RepackPlanner.classify` sends
/// `language_model.mtp.` to `.excludedDraft`, so the pack has never carried a
/// byte of it (`30-MTP-HEAD-GRAFT.md` §6-6). Rebuilding the pack to add 503 MB
/// would re-copy 21 GB that does not change, so the head arrives as a sidecar
/// instead — two files and an index, written by
/// `Scripts/qwen35/build_mtp_sidecar.py` from the grafted, baked checkpoint.
///
/// Two mappings, for two different reasons:
///
///   `mtp_core.bin`     50 MB — the nine plain BF16 tensors and the eight
///                      8-bit projections. One `MTLBuffer` over the whole
///                      file; every tensor is a byte offset into it, exactly
///                      as `Model.resident` hands out views into
///                      `model_weights.bin`.
///   `mtp_experts.bin`  453 MB — 256 expert blobs laid out **byte-identically
///                      to the body's `packed_experts`** (gate w/s/b, up
///                      w/s/b, down w/s/b, padded to the page). That is what
///                      lets `MoE.encodeRoutedPersistent…` take them with no
///                      change: one `bytesNoCopy` buffer per expert, the same
///                      shape `MmapExpertMapping` uses for the body.
///
/// **All 256 go into one residency set, committed once.** The body's set is
/// re-synced per layer per token because only 32 of 256 experts are in slots;
/// here every expert is mapped for the whole run, so the set is built at load
/// and never edited — the 0.4 ms `commit()` that costs the body a third of
/// decode (`27-PHASE6-THROUGHPUT.md` §9) is paid once.
public final class QwenMTPSidecar {

    public enum SidecarError: Error, CustomStringConvertible {
        case missing(String)
        case malformed(String)
        case geometry(String)

        public var description: String {
            switch self {
            case .missing(let path): return "MTP sidecar: \(path) is missing"
            case .malformed(let detail): return "MTP sidecar: \(detail)"
            case .geometry(let detail): return "MTP sidecar geometry: \(detail)"
            }
        }
    }

    /// One tensor's place in `mtp_core.bin`, in the shape the GEMV kernels bind.
    public struct CoreTensor: Sendable {
        public let offset: Int
        public let size: Int
        public let scaleOffset: Int
        public let biasOffset: Int
        public let bits: Int
        public let rows: UInt32
        public let cols: UInt32
    }

    struct Arch: Sendable {
        let hiddenSize: Int
        let numHeads: Int
        let numKVHeads: Int
        let headDim: Int
        let numExperts: Int
        let topK: Int
        let moeIntermediateSize: Int
        let sharedIntermediateSize: Int
        let ropeTheta: Float
        let partialRotaryFactor: Double
        let rmsNormEps: Float
    }

    let arch: Arch
    let coreBuffer: MTLBuffer
    /// One buffer per expert, indexed by expert number.
    let expertBuffers: [MTLBuffer]
    let expertOffsets: MoEExpertOffsets
    let expertStride: Int
    let residencySet: any MTLResidencySet
    private let tensors: [String: CoreTensor]

    private let coreBase: UnsafeMutableRawPointer
    private let coreLength: Int
    private let expertBase: UnsafeMutableRawPointer
    private let expertLength: Int

    /// Where the sidecar lives when nothing says otherwise. Chosen the same way
    /// the sample images are (`TF_QWEN_MTP_HEAD` overrides): the head is a
    /// 503 MB artifact built from a checkpoint, so it belongs next to the other
    /// checkpoints rather than in the repository.
    public static let defaultDirectory: String =
        ProcessInfo.processInfo.environment["TF_QWEN_MTP_HEAD"]
            ?? NSString(string: "~/LLM/ornith-mtp-head").expandingTildeInPath

    public init(directory: String, device: MTLDevice) throws {
        let root = URL(fileURLWithPath: NSString(string: directory).expandingTildeInPath)
        let indexPath = root.appendingPathComponent("mtp_head.json")
        guard let data = try? Data(contentsOf: indexPath) else {
            throw SidecarError.missing(indexPath.path)
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let format = json["format"] as? String,
              format == "turbo-fieldfare-mtp-head-v1" else {
            throw SidecarError.malformed("mtp_head.json is not a v1 index")
        }
        guard let archJSON = json["arch"] as? [String: Any],
              let coreJSON = json["core"] as? [String: Any],
              let expertsJSON = json["experts"] as? [String: Any] else {
            throw SidecarError.malformed("mtp_head.json is missing a section")
        }

        func int(_ dict: [String: Any], _ key: String) throws -> Int {
            guard let value = dict[key] as? NSNumber else {
                throw SidecarError.malformed("\(key) is missing")
            }
            return value.intValue
        }
        func double(_ dict: [String: Any], _ key: String) throws -> Double {
            guard let value = dict[key] as? NSNumber else {
                throw SidecarError.malformed("\(key) is missing")
            }
            return value.doubleValue
        }

        self.arch = Arch(hiddenSize: try int(archJSON, "hiddenSize"),
                         numHeads: try int(archJSON, "numHeads"),
                         numKVHeads: try int(archJSON, "numKVHeads"),
                         headDim: try int(archJSON, "headDim"),
                         numExperts: try int(archJSON, "numExperts"),
                         topK: try int(archJSON, "topK"),
                         moeIntermediateSize: try int(archJSON, "moeIntermediateSize"),
                         sharedIntermediateSize: try int(archJSON, "sharedIntermediateSize"),
                         ropeTheta: Float(try double(archJSON, "ropeTheta")),
                         partialRotaryFactor: try double(archJSON, "partialRotaryFactor"),
                         rmsNormEps: Float(try double(archJSON, "rmsNormEps")))

        // --- core -----------------------------------------------------------
        let corePath = root.appendingPathComponent(coreJSON["file"] as? String ?? "mtp_core.bin")
        let coreSize = try int(coreJSON, "size")
        (self.coreBase, self.coreLength) = try Self.map(path: corePath, expected: coreSize)
        guard let coreBuffer = device.makeBuffer(bytesNoCopy: coreBase,
                                                 length: coreLength,
                                                 options: .storageModeShared,
                                                 deallocator: nil) else {
            throw SidecarError.malformed("could not wrap mtp_core.bin")
        }
        coreBuffer.label = "mtp.core"
        self.coreBuffer = coreBuffer

        guard let entries = coreJSON["tensors"] as? [String: [String: Any]] else {
            throw SidecarError.malformed("core.tensors is missing")
        }
        var table: [String: CoreTensor] = [:]
        for (name, entry) in entries {
            let shape = (entry["shape"] as? [NSNumber])?.map(\.intValue) ?? []
            let rows = shape.count >= 1 ? shape[0] : 0
            let cols = shape.count >= 2 ? shape[1] : 1
            table[name] = CoreTensor(offset: try int(entry, "offset"),
                                     size: try int(entry, "size"),
                                     scaleOffset: try int(entry, "scaleOffset"),
                                     biasOffset: try int(entry, "biasOffset"),
                                     bits: try int(entry, "bits"),
                                     rows: UInt32(rows),
                                     cols: UInt32(cols))
        }
        self.tensors = table

        // --- experts ---------------------------------------------------------
        let expertsPath = root.appendingPathComponent(
            expertsJSON["file"] as? String ?? "mtp_experts.bin")
        let stride = try int(expertsJSON, "stride")
        let count = try int(expertsJSON, "count")
        self.expertStride = stride
        (self.expertBase, self.expertLength) =
            try Self.map(path: expertsPath, expected: stride * count)
        let pageSize = Int(getpagesize())
        guard stride % pageSize == 0 else {
            throw SidecarError.geometry(
                "expertStride \(stride) is not a multiple of the page size")
        }
        var buffers: [MTLBuffer] = []
        buffers.reserveCapacity(count)
        for expert in 0..<count {
            guard let buffer = device.makeBuffer(
                      bytesNoCopy: expertBase.advanced(by: expert * stride),
                      length: stride,
                      options: .storageModeShared,
                      deallocator: nil) else {
                throw SidecarError.malformed("could not wrap MTP expert \(expert)")
            }
            buffer.label = "mtp.expert.\(expert)"
            buffers.append(buffer)
        }
        self.expertBuffers = buffers

        guard let sub = expertsJSON["subTensors"] as? [String: [String: Any]],
              let gate = sub["gate"], let up = sub["up"], let down = sub["down"] else {
            throw SidecarError.malformed("experts.subTensors is missing a role")
        }
        self.expertOffsets = MoEExpertOffsets(
            gateWOff: UInt32(try int(gate, "weightOffset")),
            gateSOff: UInt32(try int(gate, "scaleOffset")),
            gateBOff: UInt32(try int(gate, "biasOffset")),
            upWOff: UInt32(try int(up, "weightOffset")),
            upSOff: UInt32(try int(up, "scaleOffset")),
            upBOff: UInt32(try int(up, "biasOffset")),
            downWOff: UInt32(try int(down, "weightOffset")),
            downSOff: UInt32(try int(down, "scaleOffset")),
            downBOff: UInt32(try int(down, "biasOffset")))

        // Every expert, once. Nothing is ever removed, so `commit()` is paid at
        // load rather than per layer per token.
        let descriptor = MTLResidencySetDescriptor()
        descriptor.label = "mtp-experts"
        descriptor.initialCapacity = count
        let set = try device.makeResidencySet(descriptor: descriptor)
        for buffer in buffers { set.addAllocation(buffer) }
        set.commit()
        set.requestResidency()
        self.residencySet = set
    }

    /// Refuse a sidecar whose experts are laid out differently from the body's.
    ///
    /// The MoE kernels take **one** `MoEExpertOffsets` for all eight blobs, so
    /// a sidecar that put `up` before `gate` would read the right bytes with
    /// the wrong meaning and still produce fluent text. The body's layout is
    /// the one `Model.routedExpertOffsets` reports, and it is checked here
    /// rather than trusted because both sides are generated.
    public func checkExpertLayout(matches body: MoEExpertOffsets) throws {
        guard expertOffsets == body else {
            throw SidecarError.geometry(
                "expert blob layout differs from the body's packed_experts")
        }
    }

    public func tensor(_ name: String) throws -> CoreTensor {
        guard let entry = tensors[name] else {
            throw SidecarError.missing("core tensor \(name)")
        }
        return entry
    }

    /// The sidecar's own view of a tensor, in the shape the shared GEMV
    /// helpers take.
    func view(_ name: String) throws -> TensorView {
        let entry = try tensor(name)
        return TensorView(buffer: coreBuffer,
                          offset: UInt64(entry.offset),
                          length: UInt64(entry.size),
                          scaleOffset: UInt64(entry.scaleOffset),
                          scaleLength: 0,
                          biasOffset: UInt64(entry.biasOffset),
                          biasLength: 0,
                          shape: (entry.rows, entry.cols, 0, 0),
                          dtype: entry.bits == 16 ? 1 : 0)
    }

    public var mappedBytes: Int { coreLength + expertLength }

    private static func map(path: URL, expected: Int) throws
        -> (UnsafeMutableRawPointer, Int) {
        let fd = open(path.path, O_RDONLY)
        guard fd >= 0 else { throw SidecarError.missing(path.path) }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0 else {
            throw SidecarError.malformed("cannot stat \(path.lastPathComponent)")
        }
        guard Int(info.st_size) == expected else {
            throw SidecarError.malformed(
                "\(path.lastPathComponent) is \(info.st_size) B, the index says \(expected)")
        }
        let pageSize = Int(getpagesize())
        let length = ((expected + pageSize - 1) / pageSize) * pageSize
        // `MAP_SHARED`, like the body's expert mapping: `MAP_PRIVATE` turns the
        // pages into private anonymous memory the moment they are touched,
        // which is the 3.22 GB `docs/mtp/48` went to the trouble of removing.
        let raw = mmap(nil, length, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0)
        guard let raw, raw != MAP_FAILED else {
            throw SidecarError.missing(path.path)
        }
        // The head runs on every drafted token, so every page is wanted. Asking
        // once at load is what keeps the first draft from paying 453 MB of
        // faults inside a command buffer.
        _ = posix_madvise(raw, length, POSIX_MADV_WILLNEED)
        return (raw, length)
    }
}

extension MoEExpertOffsets: Equatable {
    public static func == (a: MoEExpertOffsets, b: MoEExpertOffsets) -> Bool {
        a.gateWOff == b.gateWOff && a.gateSOff == b.gateSOff && a.gateBOff == b.gateBOff
            && a.upWOff == b.upWOff && a.upSOff == b.upSOff && a.upBOff == b.upBOff
            && a.downWOff == b.downWOff && a.downSOff == b.downSOff
            && a.downBOff == b.downBOff
    }
}
