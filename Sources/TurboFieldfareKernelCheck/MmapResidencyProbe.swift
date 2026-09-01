import Foundation
import Metal
import Darwin

// MARK: - `--mmap-residency-probe`
//
// `docs/mtp/47-D-MMAP-RESIDENCY-PROPOSAL.md` §5 puts the whole D branch on one
// unverified claim: that Metal wires a `bytesNoCopy` buffer at the *buffer's*
// granularity and not at the granularity of the mapping it sits in. If a
// command buffer that names eight experts wires 8 x 3.36 MB the branch lives;
// if it wires the 420 MB layer file those eight experts were mapped from, thirty
// layers come to 12.6 GB and the branch is dead on a 18 GiB machine.
//
// P-1 answers that and nothing else. The kernels, the shape and the group size
// are 43/44's (`bpw_rows_gate_up` / `bpw_rows_down` at D=2816 F=704 group 32),
// and the *bytes are the shipped checkpoint's*: `BpwLayout(.sym)` reproduces the
// production expert blob offsets exactly (gate 0, gate_scales 991232, up
// 1115136, up_scales 2106368, down 2230272, down_scales 3221504), so the probe
// can point the kernel straight at `packed_experts/layer_00.bin`. Only the
// *provenance of the buffer* is swept. The inference path is untouched.

/// `hw.pagesize`. Read once here rather than through `vm_kernel_page_size`,
/// which the SDK exposes as a mutable global.
let mmapProbePageSize = Int(sysconf(_SC_PAGESIZE))

private let mmapProbeExpertsPerStep = 8   // top-k per token, 47 §3
private let mmapProbeRowsPerExpert = 2    // production averages 1.72 (27 §6)

// MARK: - VM instruments

/// System-wide page counters. `wired` is the number P-1 is about; it is
/// system-wide, so the probe also measures its drift across an idle interval
/// and prints that as the noise floor next to the deltas.
struct MmapHostVM {
    var wired = 0, free = 0, active = 0, inactive = 0, external = 0, compressed = 0
    var pageSize = mmapProbePageSize

    static func sample() -> MmapHostVM {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return MmapHostVM() }
        var out = MmapHostVM()
        out.wired = Int(stats.wire_count)
        out.free = Int(stats.free_count)
        out.active = Int(stats.active_count)
        out.inactive = Int(stats.inactive_count)
        out.external = Int(stats.external_page_count)
        out.compressed = Int(stats.compressor_page_count)
        return out
    }
}

/// This task's own footprint, to separate "the kernel wired someone else's
/// pages" from "this process grew".
struct MmapTaskVM {
    var footprint = 0, resident = 0, internalBytes = 0, external = 0, compressed = 0

    static func sample() -> MmapTaskVM {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<natural_t>.stride)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return MmapTaskVM() }
        var out = MmapTaskVM()
        out.footprint = Int(info.phys_footprint)
        out.resident = Int(info.resident_size)
        out.internalBytes = Int(info.internal)
        out.external = Int(info.external)
        out.compressed = Int(info.compressed)
        return out
    }
}

/// Per-mapping counters, summed over every VM region the mapping has been split
/// into. Unlike `MmapHostVM` these carry no system noise at all, which is what
/// makes them the tie-breaker when the wired delta is close to the floor.
struct MmapRegionVM {
    var regions = 0, residentPages = 0, dirtyPages = 0, swappedPages = 0
    var userWired = 0, shareModes: Set<Int> = []

    static func sample(base: UnsafeRawPointer, length: Int) -> MmapRegionVM {
        var out = MmapRegionVM()
        let end = UInt64(UInt(bitPattern: base)) + UInt64(length)
        var address = mach_vm_address_t(UInt(bitPattern: base))
        while address < end {
            var size = mach_vm_size_t(0)
            var depth = natural_t(1024)
            var info = vm_region_submap_info_data_64_t()
            var count = mach_msg_type_number_t(
                MemoryLayout<vm_region_submap_info_data_64_t>.stride
                    / MemoryLayout<natural_t>.stride)
            let kr = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: Int32.self, capacity: Int(count)) {
                    mach_vm_region_recurse(mach_task_self_, &address, &size, &depth,
                                           $0, &count)
                }
            }
            guard kr == KERN_SUCCESS, address < end, size > 0 else { break }
            out.regions += 1
            out.residentPages += Int(info.pages_resident)
            out.dirtyPages += Int(info.pages_dirtied)
            out.swappedPages += Int(info.pages_swapped_out)
            out.userWired += Int(info.user_wired_count)
            out.shareModes.insert(Int(info.share_mode))
            address += size
        }
        return out
    }
}

/// `mincore` over the mapping. `MINCORE_ANONYMOUS` is the one that matters
/// besides residency: a `MAP_PRIVATE` page that turned anonymous is a broken
/// copy-on-write, i.e. Metal made its own copy of the weight and D's whole
/// premise ("clean file-backed pages the OS can just drop", 47 §3) is gone.
struct MmapCoreVM {
    var incore = 0, anonymous = 0, modified = 0, pagedOut = 0, copied = 0

    static func sample(base: UnsafeMutableRawPointer, length: Int) -> MmapCoreVM {
        let pages = length / mmapProbePageSize
        var vec = [Int8](repeating: 0, count: pages)
        var out = MmapCoreVM()
        guard mincore(base, length, &vec) == 0 else { return out }
        for byte in vec {
            let value = Int(UInt8(bitPattern: byte))
            if value & 0x01 != 0 { out.incore += 1 }
            if value & 0x04 != 0 { out.modified += 1 }
            if value & 0x20 != 0 { out.pagedOut += 1 }
            if value & 0x40 != 0 { out.copied += 1 }
            if value & 0x80 != 0 { out.anonymous += 1 }
        }
        return out
    }
}

// MARK: - The mapping under test

/// How the layer file is mapped. D wants the pages clean and file-backed, so
/// `privateRead` (no write permission at all, so copy-on-write can never break)
/// is the shape the proposal assumes. The other two are here because Metal may
/// simply refuse a read-only or a shared pointer, and that refusal is itself the
/// answer to whether the shape is available.
enum MmapProbeMapping: String, CaseIterable {
    case privateRead = "MAP_PRIVATE/r"
    case privateWrite = "MAP_PRIVATE/rw"
    case sharedRead = "MAP_SHARED/r"

    var prot: Int32 {
        self == .privateWrite ? PROT_READ | PROT_WRITE : PROT_READ
    }
    var flags: Int32 {
        self == .sharedRead ? MAP_SHARED | MAP_FILE : MAP_PRIVATE | MAP_FILE
    }
}

/// How the `MTLBuffer`s are cut out of that mapping -- the actual P-1 contrast.
enum MmapProbeArm: String, CaseIterable {
    /// One `bytesNoCopy` buffer per expert, over the whole layer (128 of them).
    /// Thirty layers of this is the 3,840 buffers 47 §5(a) asks about.
    case perExpert = "per-expert (128 buffers)"
    /// A single `bytesNoCopy` buffer spanning the layer file, experts addressed
    /// by offset. This is the shape 47 §5(a) predicts wires 420 MB.
    case perLayer = "per-layer (1 buffer)"
}

private final class MmapProbeMapped {
    let fd: Int32
    let base: UnsafeMutableRawPointer
    let length: Int

    init?(path: String, mapping: MmapProbeMapping) {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { return nil }
        var st = stat()
        guard fstat(fd, &st) == 0 else { close(fd); return nil }
        let length = Int(st.st_size)
        guard let base = mmap(nil, length, mapping.prot, mapping.flags, fd, 0),
              base != MAP_FAILED else { close(fd); return nil }
        self.fd = fd
        self.base = base
        self.length = length
    }

    /// Drop whatever this mapping holds so the next arm starts from the same
    /// place. `MADV_DONTNEED` unmaps the pages from *this* task; the file's
    /// pages stay in the unified buffer cache, which is the warm condition P-2
    /// wants and P-3 will have to defeat separately.
    func drop() { madvise(base, length, MADV_DONTNEED) }


    /// Explicit teardown. Every `MTLBuffer` cut out of this mapping has to be
    /// released before the mapping is, and Swift does not promise a release
    /// order for two locals in the same scope, so the driver calls this itself.
    func release() {
        guard !released else { return }
        released = true
        munmap(base, length)
        close(fd)
    }

    private var released = false

    deinit { release() }
}

/// Evict a file from the unified buffer cache, without `sudo`.
///
/// 47 §6 assumed the cold condition needed `purge` (root) and fell back to
/// `madvise`, but `madvise` only drops *this* task's page table entries -- a
/// fresh mapping still finds every page in the cache. `msync` with
/// `MS_INVALIDATE` does drop them: measured here, a fresh mapping afterwards
/// reads 0 of 26,240 pages resident and a serial touch runs at 0.73 GB/s
/// instead of out of cache.
///
/// It only works while *no other mapping of the file is open*: the first cut of
/// this probe invalidated through a temporary view while the arm's own mapping
/// was already up, and the kernel declined for every trial after the first.
/// So the drop happens before the mapping under test is made, not inside it.
/// `msync` can return success and still leave pages resident, so the drop is
/// verified and retried rather than trusted. After a `MAP_SHARED` arm the
/// driver holds its reference to the file's pages a little past the release of
/// the `MTLBuffer`, and one attempt lands while they are still pinned.
@discardableResult
func mmapProbeColdDrop(path: String, attempts: Int = 40) -> Bool {
    let fd = open(path, O_RDONLY)
    guard fd >= 0 else { return false }
    defer { close(fd) }
    var st = stat()
    guard fstat(fd, &st) == 0 else { return false }
    let length = Int(st.st_size)
    let pages = length / mmapProbePageSize
    var vec = [Int8](repeating: 0, count: pages)

    for attempt in 0..<attempts {
        guard let view = mmap(nil, length, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0),
              view != MAP_FAILED else { return false }
        _ = msync(view, length, MS_INVALIDATE)
        var resident = 0
        if mincore(view, length, &vec) == 0 {
            for byte in vec where UInt8(bitPattern: byte) & 0x01 != 0 { resident += 1 }
        }
        munmap(view, length)
        if resident == 0 { return true }
        if attempt + 1 < attempts { usleep(25_000) }
    }
    return false
}

/// Wiring is held for as long as a command buffer is in flight and released
/// when it retires, so a sample taken after `waitUntilCompleted` sees nothing.
/// The first pass of this probe made exactly that mistake and read a flat zero
/// on every arm. The fix is to give the GPU enough work to stay busy for a few
/// hundred milliseconds and poll while it runs.
struct MmapInFlight {
    var peakWired = 0        // pages, system-wide, above the pre-commit sample
    var peakUserWired = 0    // wire *levels* on this mapping's regions
    var peakRegionResident = 0
    var samples = 0
    var seconds = 0.0
}

// MARK: - Report helpers

private func mmapPagesMB(_ pages: Int) -> String {
    String(format: "%8.1f MB", Double(pages) * Double(mmapProbePageSize) / 1e6)
}

private func mmapSigned(_ pages: Int) -> String {
    String(format: "%+7d pg %+9.1f MB", pages,
           Double(pages) * Double(mmapProbePageSize) / 1e6)
}

/// The control: today's path. Eight experts read with `pread` into private
/// `MTLBuffer` slots, issued in parallel exactly the way
/// `PreadExpertStreamer.executeExpertCachePlan` issues a layer's misses
/// (`DispatchQueue.concurrentPerform` over the miss list, `:345`). This is here
/// so the cold and warm numbers above are compared against something measured
/// in the same harness on the same machine in the same minute -- 40 §4-11 is
/// explicit that carrying 33 §1's bandwidths across conditions turns them into
/// a lie.
final class MmapPreadControl {
    let fd: Int32
    let slots: [MTLBuffer]
    let stride: Int

    init?(device: MTLDevice, path: String, stride: Int, count: Int) {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { return nil }
        var slots: [MTLBuffer] = []
        for _ in 0..<count {
            guard let buffer = device.makeBuffer(length: stride, options: .storageModeShared)
            else { close(fd); return nil }
            slots.append(buffer)
        }
        self.fd = fd
        self.slots = slots
        self.stride = stride
    }

    /// Seconds to get `ranks` in front of the GPU. Returns the wall time of the
    /// parallel issue, which is what the `io` bucket in 41 §4 counts.
    func fetch(ranks: [Int]) -> Double {
        let fd = self.fd
        let stride = self.stride
        // Each iteration fills its own slot, and the buffers outlive the loop.
        nonisolated(unsafe) let slots = self.slots
        let started = Date()
        DispatchQueue.concurrentPerform(iterations: ranks.count) { index in
            let destination = slots[index].contents()
            var filled = 0
            while filled < stride {
                let got = pread(fd, destination.advanced(by: filled), stride - filled,
                                off_t(ranks[index] * stride) + off_t(filled))
                if got <= 0 { break }
                filled += got
            }
        }
        return Date().timeIntervalSince(started)
    }

    deinit { close(fd) }
}

/// Resident pages within just the ranges under test. The first cut of P-4
/// polled `mincore` over the whole 420 MB mapping every 200 us and charged the
/// cost of doing so to `F_RDADVISE`; over 26,240 pages that is not free.
func mmapProbeResident(base: UnsafeMutableRawPointer,
                       ranges: [(offset: Int, length: Int)]) -> Int {
    var total = 0
    var vec = [Int8](repeating: 0, count: (ranges.first?.length ?? 0) / mmapProbePageSize)
    for range in ranges {
        guard mincore(base.advanced(by: range.offset), range.length, &vec) == 0 else { continue }
        for byte in vec where UInt8(bitPattern: byte) & 0x01 != 0 { total += 1 }
    }
    return total
}

/// P-4 (47 §6/§7): the mitigation the proposal already had in the repository --
/// `F_RDADVISE` (`Infrastructure/Streaming/RDAdvice.swift:34`) issued over the
/// eight expert ranges so the pages are in the cache before the command buffer
/// asks for them. Reimplemented here rather than linked so the probe stays out
/// of the inference module, but it is the same `fcntl` call with the same
/// `radvisory`.
func mmapProbeReadAdvise(fd: Int32, ranges: [(offset: Int, length: Int)]) -> Bool {
    var ok = true
    for range in ranges {
        var advice = radvisory(ra_offset: off_t(range.offset),
                               ra_count: Int32(range.length))
        if fcntl(fd, F_RDADVISE, &advice) != 0 { ok = false }
    }
    return ok
}

// MARK: - Driver

/// `gateOnly` runs the last section and nothing else.
///
/// It exists because the survey pollutes the gate. Every `MAP_PRIVATE` trial
/// turns 430 MB anonymous (§2 of the write-up), thirty of them run before the
/// gate section does, and across a session of full runs the ratio walked
/// 1.16 -> 1.31 while the `pread` control stayed flat -- the compressor and
/// swap were filling up underneath. The deciding number has to be taken in a
/// process that has not done that.
func runMmapResidencyProbe(groupSize: Int, modelPath: String,
                           repeats: Int, trials: Int, gateOnly: Bool = false) throws {
    let layersDir = (modelPath as NSString).appendingPathComponent("packed_experts")
    let layoutPath = (layersDir as NSString).appendingPathComponent("layout.json")
    guard let layoutData = FileManager.default.contents(atPath: layoutPath),
          let layout = try JSONSerialization.jsonObject(with: layoutData) as? [String: Any],
          let expertStride = layout["expertStride"] as? Int,
          let expertsPerLayer = layout["expertsPerLayer"] as? Int
    else { throw NSError(domain: "mmap-probe", code: 1, userInfo: [
        NSLocalizedDescriptionKey: "cannot read \(layoutPath)"]) }

    let layerPath = (layersDir as NSString).appendingPathComponent("layer_00.bin")
    guard let layerAttrs = try? FileManager.default.attributesOfItem(atPath: layerPath),
          let layerSize = (layerAttrs[.size] as? NSNumber)?.intValue
    else { throw NSError(domain: "mmap-probe", code: 2, userInfo: [
        NSLocalizedDescriptionKey: "cannot stat \(layerPath)"]) }

    let pageSize = mmapProbePageSize
    let d = 2816, f = 704

    print("=== P-1: does Metal wire a `bytesNoCopy` buffer, or the mapping it "
            + "came from? ===")
    print("47 §6/§7. Kernels, shape and group size are 43/44's; the bytes are the")
    print("shipped checkpoint's. Inference path untouched.")
    print("")
    print("  model               \(modelPath)")
    print("  layer file          \(((layerPath as NSString).lastPathComponent)) "
            + "\(layerSize) B = \(layerSize / pageSize) pages")
    print("  hw.pagesize         \(pageSize)")
    print("  expertStride        \(expertStride) B = "
            + "\(expertStride / pageSize) pages, remainder \(expertStride % pageSize)")
    print("  experts / layer     \(expertsPerLayer)")
    print("  step footprint      \(mmapProbeExpertsPerStep) experts = "
            + String(format: "%.2f MB", Double(mmapProbeExpertsPerStep * expertStride) / 1e6))
    print("  PASS threshold      wired delta at commit ~= "
            + String(format: "%.0f MB", Double(mmapProbeExpertsPerStep * expertStride) / 1e6)
            + ", FAIL ~= "
            + String(format: "%.0f MB", Double(layerSize) / 1e6))
    guard expertStride % pageSize == 0, layerSize % pageSize == 0 else {
        throw NSError(domain: "mmap-probe", code: 3, userInfo: [
            NSLocalizedDescriptionKey: "stride/size not a page multiple; "
                + "`bytesNoCopy` cannot be used without a relayout"])
    }

    // The kernel side is built once and shared by every arm, so nothing in the
    // per-arm deltas is Metal warming up: pipelines, the shared scratch buffers
    // and the driver's own allocations are all in place before the first sample.
    let context = try makeContext(groupSize: groupSize)
    let probe = try BpwProbe(context: context, groupSize: groupSize, d: d, f: f)
    let gateUpPSO = try probe.pipeline(.gateUp, cap: mmapProbeRowsPerExpert, format: .sym)
    let downPSO = try probe.pipeline(.down, cap: mmapProbeRowsPerExpert, format: .sym)
    let rowCounts = [Int](repeating: mmapProbeRowsPerExpert, count: mmapProbeExpertsPerStep)
        + [Int](repeating: 0, count: bpwTileExperts - mmapProbeExpertsPerStep)

    // The eight experts a step touches are not adjacent in the file. Spreading
    // them over the layer is what makes the two arms differ: contiguous ranks
    // would let a per-layer buffer look small by accident.
    let ranks = (0..<mmapProbeExpertsPerStep).map { $0 * (expertsPerLayer / mmapProbeExpertsPerStep) }
    print("  experts used        ranks \(ranks.map(String.init).joined(separator: ","))")
    print("")

    let adviseRanges = ranks.map { (offset: $0 * expertStride, length: expertStride) }
    let wantResident = mmapProbeExpertsPerStep * expertStride / pageSize

    if !gateOnly {
    // ---- noise floor ------------------------------------------------------
    //
    // `wire_count` is system-wide. Measure how far it walks on its own over the
    // same interval an arm takes, so the deltas below can be read against it.
    let noiseStart = MmapHostVM.sample()
    let noiseDeadline = Date().addingTimeInterval(1.0)
    var noiseWorst = 0
    while Date() < noiseDeadline {
        noiseWorst = max(noiseWorst, abs(MmapHostVM.sample().wired - noiseStart.wired))
    }
    print("  system wired noise over 1 s of idle: \(noiseWorst) pages "
            + String(format: "(%.1f MB)", Double(noiseWorst) * Double(pageSize) / 1e6))
    print("")

    var reference: [Float]?

    // `repeats` is sized so the GPU stays busy long enough to be caught in the
    // act. One pass reads 26.9 MB, so at the 135 GB/s floor 43 §2 measured it is
    // a fifth of a millisecond; the default lands near a quarter second.

    for mapping in MmapProbeMapping.allCases {
        print("--- \(mapping.rawValue) "
                + String(repeating: "-", count: max(0, 60 - mapping.rawValue.count)))

        for arm in MmapProbeArm.allCases {
            var flights: [MmapInFlight] = []
            var colds: [(base: Int, faulted: Int, wall: Double, ok: Bool)] = []
            var materialised: [(anon: Int, dirty: Int, incore: Int, regions: Int)] = []
            var walls: [Double] = []
            var note: String?
            var isReferenceArm = false
            var createdBuffers = 0
            var namedBuffers = 0
            var shareModes: Set<Int> = []

            for _ in 0..<trials {
                // A fresh mapping per trial. Anything the previous trial did to
                // the *mapping* (a broken copy-on-write, a fault) is gone; what
                // it did to the *file's* place in the unified buffer cache is
                // not, which is why `incore` is only readable on the very first
                // pass and the anonymous count is what carries the finding.
                let coldOK = mmapProbeColdDrop(path: layerPath)
                guard let mapped = MmapProbeMapped(path: layerPath, mapping: mapping) else {
                    note = "mmap failed: \(String(cString: strerror(errno)))"
                    break
                }
                let base = mapped.base
                let core0 = MmapCoreVM.sample(base: base, length: mapped.length)
                let region0 = MmapRegionVM.sample(base: base, length: mapped.length)

                var owned: [MTLBuffer] = []
                var slots: [(buffer: MTLBuffer, offset: Int)] = []
                var resident: [MTLBuffer] = []
                switch arm {
                case .perExpert:
                    for rank in 0..<expertsPerLayer {
                        guard let buffer = context.device.makeBuffer(
                            bytesNoCopy: base.advanced(by: rank * expertStride),
                            length: expertStride, options: .storageModeShared,
                            deallocator: nil) else {
                            note = "makeBuffer(bytesNoCopy:) returned nil at rank \(rank)"
                            break
                        }
                        owned.append(buffer)
                    }
                    if owned.count == expertsPerLayer {
                        resident = ranks.map { owned[$0] }
                        slots = (0..<bpwTileExperts).map {
                            (owned[ranks[min($0, ranks.count - 1)]], 0)
                        }
                    }
                case .perLayer:
                    guard let buffer = context.device.makeBuffer(
                        bytesNoCopy: base, length: mapped.length,
                        options: .storageModeShared, deallocator: nil) else {
                        note = "makeBuffer(bytesNoCopy:) returned nil for the layer"
                        break
                    }
                    owned = [buffer]
                    resident = [buffer]
                    slots = (0..<bpwTileExperts).map {
                        (buffer, ranks[min($0, ranks.count - 1)] * expertStride)
                    }
                }
                if slots.isEmpty { break }
                createdBuffers = owned.count
                namedBuffers = resident.count

                var fixture: BpwFixture? = probe.makeFixture(format: .sym, slots: slots,
                                                             blobs: resident)
                probe.fillHidden()

                // ---- the cold pass: the granularity question, directly ----
                //
                // With the file evicted, one command buffer naming eight
                // experts is run and the pages it made resident are counted.
                // Eight experts is 1,640 pages; the whole layer is 26,240. This
                // is the reading that does not depend on the wired counter
                // moving, and it is the only one that works for `MAP_SHARED`,
                // where nothing ever turns anonymous.
                let cold0 = MmapCoreVM.sample(base: base, length: mapped.length)
                guard let warm = context.queue.makeCommandBuffer() else { fatalError("cmd") }
                probe.encode(.gateUp, pso: gateUpPSO, commandBuffer: warm,
                             fixture: fixture!, rowCounts: rowCounts)
                probe.encode(.down, pso: downPSO, commandBuffer: warm,
                             fixture: fixture!, rowCounts: rowCounts)
                let coldStart = Date()
                waitAndCheck(warm, "mmap probe cold pass \(mapping.rawValue) \(arm.rawValue)")
                let coldWall = Date().timeIntervalSince(coldStart)
                let cold1 = MmapCoreVM.sample(base: base, length: mapped.length)
                colds.append((base: cold0.incore, faulted: cold1.incore - cold0.incore,
                              wall: coldWall, ok: coldOK))

                // ---- the measured flight --------------------------------
                guard let cmd = context.queue.makeCommandBuffer() else { fatalError("cmd") }
                for _ in 0..<repeats {
                    probe.encode(.gateUp, pso: gateUpPSO, commandBuffer: cmd,
                                 fixture: fixture!, rowCounts: rowCounts)
                    probe.encode(.down, pso: downPSO, commandBuffer: cmd,
                                 fixture: fixture!, rowCounts: rowCounts)
                }
                let hostBase = MmapHostVM.sample()
                let regionBase = MmapRegionVM.sample(base: base, length: mapped.length)
                let done = MmapCompletionFlag()
                cmd.addCompletedHandler { _ in done.set() }
                let started = Date()
                cmd.commit()
                var flight = MmapInFlight()
                while !done.value {
                    let host = MmapHostVM.sample()
                    let region = MmapRegionVM.sample(base: base, length: mapped.length)
                    flight.peakWired = max(flight.peakWired, host.wired - hostBase.wired)
                    flight.peakUserWired = max(flight.peakUserWired,
                                               region.userWired - regionBase.userWired)
                    flight.peakRegionResident = max(flight.peakRegionResident,
                                                    region.residentPages - regionBase.residentPages)
                    flight.samples += 1
                }
                cmd.waitUntilCompleted()
                flight.seconds = Date().timeIntervalSince(started)
                if let error = cmd.error {
                    fatalError("mmap probe flight failed — \(error)")
                }
                flights.append(flight)
                walls.append(flight.seconds / Double(repeats))

                let core3 = MmapCoreVM.sample(base: base, length: mapped.length)
                let region3 = MmapRegionVM.sample(base: base, length: mapped.length)
                shareModes.formUnion(region3.shareModes)
                materialised.append((core3.anonymous - core0.anonymous,
                                     region3.dirtyPages - region0.dirtyPages,
                                     core3.incore,
                                     region3.regions))

                let out = probe.readAct(rows: mmapProbeExpertsPerStep * mmapProbeRowsPerExpert)
                if let ref = reference {
                    if ref.count != out.count || !zip(ref, out).allSatisfy({ $0 == $1 }) {
                        note = "*** output DIFFERS from the first arm ***"
                    }
                } else if out.allSatisfy({ $0.isFinite }) {
                    reference = out
                    isReferenceArm = true
                } else {
                    note = "*** output is not finite ***"
                }

                // Every buffer has to go before the mapping under it does.
                fixture = nil
                owned = []
                resident = []
                slots = []
                mapped.release()
            }

            print("  \(arm.rawValue)")
            print("    output            "
                    + (note ?? (isReferenceArm ? "gate/up reference for the other arms"
                                : "gate/up bit-identical to the reference arm")))
            guard !flights.isEmpty else { print(""); continue }
            print("    buffers           created \(createdBuffers), "
                    + "named by the command buffer \(namedBuffers)")
            if !colds.isEmpty {
                let expected = mmapProbeExpertsPerStep * expertStride / pageSize
                print("    COLD pass         faulted "
                        + colds.map { String($0.faulted) }.joined(separator: " / ")
                        + " pg of \(layerSize / pageSize)   "
                        + "(8 experts = \(expected) pg, whole layer = \(layerSize / pageSize) pg)")
                print("                      = "
                        + colds.map { mmapPagesMB($0.faulted).trimmingCharacters(in: .whitespaces) }
                            .joined(separator: " / ")
                        + "   from a baseline of "
                        + colds.map { String($0.base) }.joined(separator: "/") + " pg resident"
                        + (colds.allSatisfy(\.ok) ? "" : "   (MS_INVALIDATE FAILED)"))
                print("                      wall "
                        + colds.map { String(format: "%.1f ms", $0.wall * 1e3) }
                            .joined(separator: " / "))
            }
            let wired = flights.map(\.peakWired)
            print("    in-flight wired   peak over base, \(flights.count) trials: "
                    + wired.map { String(format: "%+d", $0) }.joined(separator: " / ")
                    + " pg   (max " + mmapPagesMB(wired.max() ?? 0) + ")")
            print("    in-flight region  resident "
                    + flights.map { String(format: "%+d", $0.peakRegionResident) }
                        .joined(separator: " / ")
                    + " pg   user_wired "
                    + flights.map { String($0.peakUserWired) }.joined(separator: " / ")
                    + "   samples " + flights.map { String($0.samples) }.joined(separator: "/"))
            print("    materialised      anonymous "
                    + materialised.map { String($0.anon) }.joined(separator: " / ")
                    + " pg   dirty "
                    + materialised.map { String($0.dirty) }.joined(separator: " / ")
                    + " pg   of \(layerSize / pageSize)")
            print("                      = "
                    + materialised.map { mmapPagesMB($0.anon).trimmingCharacters(in: .whitespaces) }
                        .joined(separator: " / ")
                    + " turned anonymous (private, not droppable)")
            print("    regions/share     \(materialised.map { String($0.regions) }.joined(separator: "/")), "
                    + "share_mode \(shareModes.sorted())")
            print("    per-pass wall     "
                    + walls.map { String(format: "%.3f ms", $0 * 1e3) }.joined(separator: " / ")
                    + String(format: "   (%.1f GB/s over %d experts)",
                             Double(mmapProbeExpertsPerStep * expertStride)
                                / (walls.min() ?? 1) / 1e9, mmapProbeExpertsPerStep))
        }
        print("")
    }

    // ---- the control: today's parallel `pread` into private slots ---------
    print("--- pread into private slots (today's path) "
            + String(repeating: "-", count: 18))
    if let control = MmapPreadControl(device: context.device, path: layerPath,
                                      stride: expertStride,
                                      count: mmapProbeExpertsPerStep) {
        var coldSeconds: [Double] = []
        var warmSeconds: [Double] = []
        for _ in 0..<trials {
            let dropped = mmapProbeColdDrop(path: layerPath)
            let cold = control.fetch(ranks: ranks)
            coldSeconds.append(cold)
            if !dropped { print("    note              cold drop not verified") }
            var best = Double.infinity
            for _ in 0..<5 { best = min(best, control.fetch(ranks: ranks)) }
            warmSeconds.append(best)
        }
        let bytes = Double(mmapProbeExpertsPerStep * expertStride)
        print("    COLD fetch        "
                + coldSeconds.map { String(format: "%.1f ms", $0 * 1e3) }.joined(separator: " / ")
                + "   = "
                + coldSeconds.map { String(format: "%.2f GB/s", bytes / $0 / 1e9) }
                    .joined(separator: " / "))
        print("    WARM fetch        "
                + warmSeconds.map { String(format: "%.2f ms", $0 * 1e3) }.joined(separator: " / ")
                + "   = "
                + warmSeconds.map { String(format: "%.2f GB/s", bytes / $0 / 1e9) }
                    .joined(separator: " / "))
        print("    per expert        "
                + String(format: "cold %.3f ms, warm %.3f ms",
                         (coldSeconds.min() ?? 0) * 1e3 / Double(mmapProbeExpertsPerStep),
                         (warmSeconds.min() ?? 0) * 1e3 / Double(mmapProbeExpertsPerStep)))
        print("    private bytes     "
                + String(format: "%.1f MB of slots, held for the whole run", bytes / 1e6))
    } else {
        print("    control could not be built")
    }
    print("")

    // ---- P-4: does `F_RDADVISE` recover the cold side? --------------------
    //
    // Only run for the shape P-1 and P-2 left standing: `MAP_SHARED`, one
    // buffer per expert. Two readings, because they answer different questions.
    // `issue+run` is what happens if the advice goes out immediately before the
    // command buffer -- no lead time, so it only helps if the readahead
    // overlaps the fault stream. `wait+run` gives the advice the lead time a
    // real `ExpertPrefetch` would (it is issued a layer ahead, 47 §10) and
    // charges the wait, which is the number to compare against the cold `pread`.
    print("--- P-4: F_RDADVISE ahead of the mmap path (MAP_SHARED, per-expert) "
            + String(repeating: "-", count: 0))
    for mode in ["issue+run", "advise, wait for residency, then run"] {
        var totals: [Double] = []
        var waits: [Double] = []
        var runs: [Double] = []
        var landed: [Int] = []
        for _ in 0..<trials {
            mmapProbeColdDrop(path: layerPath)
            guard let mapped = MmapProbeMapped(path: layerPath, mapping: .sharedRead)
            else { break }
            var owned: [MTLBuffer] = []
            for rank in 0..<expertsPerLayer {
                guard let buffer = context.device.makeBuffer(
                    bytesNoCopy: mapped.base.advanced(by: rank * expertStride),
                    length: expertStride, options: .storageModeShared,
                    deallocator: nil) else { break }
                owned.append(buffer)
            }
            guard owned.count == expertsPerLayer else { mapped.release(); break }
            let resident = ranks.map { owned[$0] }
            let slots = (0..<bpwTileExperts).map {
                (buffer: owned[ranks[min($0, ranks.count - 1)]], offset: 0)
            }
            var fixture: BpwFixture? = probe.makeFixture(format: .sym, slots: slots,
                                                         blobs: resident)
            probe.fillHidden()

            let started = Date()
            _ = mmapProbeReadAdvise(fd: mapped.fd, ranges: adviseRanges)
            var waited = 0.0
            if mode != "issue+run" {
                let deadline = Date().addingTimeInterval(2.0)
                while Date() < deadline {
                    if mmapProbeResident(base: mapped.base, ranges: adviseRanges)
                        >= wantResident { break }
                }
                waited = Date().timeIntervalSince(started)
            }
            let runStart = Date()
            guard let cmd = context.queue.makeCommandBuffer() else { fatalError("cmd") }
            probe.encode(.gateUp, pso: gateUpPSO, commandBuffer: cmd,
                         fixture: fixture!, rowCounts: rowCounts)
            probe.encode(.down, pso: downPSO, commandBuffer: cmd,
                         fixture: fixture!, rowCounts: rowCounts)
            waitAndCheck(cmd, "mmap probe P-4")
            let run = Date().timeIntervalSince(runStart)
            totals.append(Date().timeIntervalSince(started))
            waits.append(waited)
            runs.append(run)
            landed.append(mmapProbeResident(base: mapped.base, ranges: adviseRanges))

            fixture = nil
            owned = []
            mapped.release()
        }
        guard !totals.isEmpty else { print("    \(mode): could not run"); continue }
        let bytes = Double(mmapProbeExpertsPerStep * expertStride)
        print("  \(mode)")
        print("    total             "
                + totals.map { String(format: "%.1f ms", $0 * 1e3) }.joined(separator: " / ")
                + "   = "
                + totals.map { String(format: "%.2f GB/s", bytes / $0 / 1e9) }
                    .joined(separator: " / "))
        if mode != "issue+run" {
            print("    of which wait     "
                    + waits.map { String(format: "%.1f ms", $0 * 1e3) }.joined(separator: " / "))
        }
        print("    of which run      "
                + runs.map { String(format: "%.1f ms", $0 * 1e3) }.joined(separator: " / "))
        print("    resident after    "
                + landed.map { String($0) }.joined(separator: " / ")
                + " pg (asked for \(wantResident))")
    }
    print("")

    }  // !gateOnly

    // ---- the gate: cold mmap+advise against cold pread, ABBA --------------
    //
    // P-3's threshold (47 §7) is a ratio between two cold numbers, and the P-4
    // trials above walked upward within each arm (5.2 -> 6.5 -> 7.0 ms), which
    // is exactly the ordering artifact 41 §7-2 got caught by and 40 §5-6 asks
    // to settle with ABBA. So the two are interleaved A B B A and the ratio is
    // taken between medians of the same number of samples in the same window.
    print("--- the gate: cold cost of 8 experts, ABBA interleaved "
            + String(repeating: "-", count: 6))
    var preadCold: [Double] = []
    var mmapCold: [Double] = []

    func coldPreadOnce() -> Double? {
        mmapProbeColdDrop(path: layerPath)
        guard let control = MmapPreadControl(device: context.device, path: layerPath,
                                             stride: expertStride,
                                             count: mmapProbeExpertsPerStep)
        else { return nil }
        return control.fetch(ranks: ranks)
    }

    func coldMmapOnce(advise: Bool) -> Double? {
        mmapProbeColdDrop(path: layerPath)
        guard let mapped = MmapProbeMapped(path: layerPath, mapping: .sharedRead)
        else { return nil }
        defer { mapped.release() }
        var owned: [MTLBuffer] = []
        for rank in 0..<expertsPerLayer {
            guard let buffer = context.device.makeBuffer(
                bytesNoCopy: mapped.base.advanced(by: rank * expertStride),
                length: expertStride, options: .storageModeShared,
                deallocator: nil) else { return nil }
            owned.append(buffer)
        }
        let resident = ranks.map { owned[$0] }
        let slots = (0..<bpwTileExperts).map {
            (buffer: owned[ranks[min($0, ranks.count - 1)]], offset: 0)
        }
        var fixture: BpwFixture? = probe.makeFixture(format: .sym, slots: slots,
                                                     blobs: resident)
        probe.fillHidden()
        let started = Date()
        if advise { _ = mmapProbeReadAdvise(fd: mapped.fd, ranges: adviseRanges) }
        guard let cmd = context.queue.makeCommandBuffer() else { fatalError("cmd") }
        probe.encode(.gateUp, pso: gateUpPSO, commandBuffer: cmd,
                     fixture: fixture!, rowCounts: rowCounts)
        probe.encode(.down, pso: downPSO, commandBuffer: cmd,
                     fixture: fixture!, rowCounts: rowCounts)
        waitAndCheck(cmd, "mmap probe gate")
        // The kernel itself is 0.25 ms warm (measured above); charging it to the
        // mmap side and not to `pread` is the conservative direction.
        let elapsed = Date().timeIntervalSince(started)
        fixture = nil
        owned = []
        return elapsed
    }

    for _ in 0..<max(3, trials) {
        if let value = coldPreadOnce() { preadCold.append(value) }
        if let value = coldMmapOnce(advise: true) { mmapCold.append(value) }
        if let value = coldMmapOnce(advise: true) { mmapCold.append(value) }
        if let value = coldPreadOnce() { preadCold.append(value) }
    }

    func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        return sorted.count % 2 == 1 ? sorted[sorted.count / 2]
            : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
    }
    let bytes = Double(mmapProbeExpertsPerStep * expertStride)
    func line(_ label: String, _ values: [Double]) {
        print("    \(label)  n=\(values.count)  "
                + String(format: "median %.2f ms (%.2f GB/s)  min %.2f  max %.2f",
                         median(values) * 1e3, bytes / median(values) / 1e9,
                         (values.min() ?? 0) * 1e3, (values.max() ?? 0) * 1e3))
        print("        " + values.map { String(format: "%.2f", $0 * 1e3) }
                .joined(separator: " "))
    }
    line("pread, parallel, into private slots", preadCold)
    line("mmap + F_RDADVISE, bytesNoCopy     ", mmapCold)
    if !preadCold.isEmpty && !mmapCold.isEmpty {
        // Two statistics, because they do not agree and the gate sits between
        // them. The pooled one divides the two medians; the paired one takes a
        // ratio inside each ABBA round and then the median of those, which
        // keeps between-round drift out but is the harsher of the two.
        let pooled = median(mmapCold) / median(preadCold)
        var paired: [Double] = []
        let rounds = min(preadCold.count, mmapCold.count) / 2
        for round in 0..<rounds {
            let a = median(Array(preadCold[(2 * round)..<(2 * round + 2)]))
            let b = median(Array(mmapCold[(2 * round)..<(2 * round + 2)]))
            paired.append(b / a)
        }
        // The mmap side is charged with the kernel it ran and the pread side is
        // not, so this is the same comparison with that removed.
        let kernel = 0.00025
        let corrected = (median(mmapCold) - kernel) / median(preadCold)
        print(String(format: "    ratio mmap/pread   pooled medians %.3fx   "
                        + "paired per round %.3fx (n=%d, %.2f..%.2f)",
                     pooled, median(paired), paired.count,
                     paired.min() ?? 0, paired.max() ?? 0))
        print(String(format: "                       with the 0.25 ms kernel "
                        + "deducted from the mmap side: %.3fx", corrected))
        print("    47 §7 gate is 1.20x. "
                + (max(pooled, median(paired)) <= 1.20 ? "PASS."
                   : (min(pooled, corrected) > 1.20 ? "FAIL."
                      : "ON THE GATE -- the statistics straddle it."))) 
    }
    print("")
}

/// A box the completion handler can set from Metal's thread while the driver
/// polls it. `nonisolated(unsafe)` is not needed: the flag is a class with a
/// lock, and the probe is a single-threaded command-line pass.
final class MmapCompletionFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    func set() { lock.lock(); flag = true; lock.unlock() }
    var value: Bool { lock.lock(); defer { lock.unlock() }; return flag }
}
