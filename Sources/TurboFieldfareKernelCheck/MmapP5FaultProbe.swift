import Foundation
import Metal
import Darwin

// MARK: - `--mmap-p5-probe`
//
// `docs/mtp/48-D-P1-P4-MMAP-RESIDENCY.md` §11 found the cold cost of the mmap
// path is not disk. With `F_RDADVISE` issued and all 1,640 pages verified in
// the page cache by `mincore`, the command buffer that names eight experts
// still takes 4.25 ms; the kernel itself is 0.25 ms, so 4.00 ms (2.44 us/page)
// is fault and mapping work happening *inside* the command buffer. §12 shows
// that cost, carried to production's touch rate, lands in the same order as
// today's `io` bucket (14.21 ms/tok), which is what D was supposed to remove.
//
// P-5 (§13) asks one question: can that 4.00 ms be moved out of the command
// buffer? Four arms, plus one that exists only to separate a confound:
//
//   A   today's shape: `useResource` on the eight buffers, nothing else
//   B   `MTLResidencySet` committed and `requestResidency()`d off-thread
//       before encoding. This is macOS 15's replacement for `useResource`, so
//       B drops `useResource` -- that is the pair, not two changes
//   B*  the residency set *and* `useResource`, so "the set helped" can be told
//       apart from "dropping the per-encode `useResource` loop helped"
//   C   fault the pages into this task from the CPU first, one byte per page,
//       spread over `concurrentPerform`
//   D   B + C, both issued at once and waited on together
//
// Two numbers per arm, because they are not interchangeable (§13): the time
// inside the command buffer, and the wall clock of the preparation -- the part
// that would have to hide behind the previous layer's GPU work.
//
// The control is §11's 4.25 ms, reproduced here as arm A under the same
// protocol. The inference path stays untouched; this is a `KernelCheck` flag.

/// What a P-5 arm does differently. `usesUseResource` is what `BpwProbe.encode`
/// keys off (it calls `useResource` on the fixture's `blobs` list and nothing
/// else), so an empty blob list is how an arm opts out.
enum MmapP5Arm: String, CaseIterable {
    case a  = "A   useResource only"
    case b  = "B   residency set"
    case bu = "B*  residency set + useResource"
    case c  = "C   CPU pre-touch"
    case d  = "D   B + C"

    var usesResidencySet: Bool { self != .a && self != .c }
    var usesUseResource: Bool { self == .a || self == .bu || self == .c }
    var cpuPreTouch: Bool { self == .c || self == .d }
}

/// One trial's stopwatch. Everything is wall clock except `gpu`, which is the
/// GPU's own timestamps, so `inBand - gpu` is the driver's share.
private struct MmapP5Trial {
    var prep = 0.0        // arm-specific preparation, wall
    var prepCPU = 0.0     // of which CPU pre-touch
    var prepSet = 0.0     // of which commit + requestResidency
    var encode = 0.0      // makeCommandBuffer + two encoders
    var inBand = 0.0      // commit -> completed
    var gpu = 0.0         // gpuEndTime - gpuStartTime
    var warmEncode = 0.0  // the same command buffer, a second time
    var warmInBand = 0.0
    var warmGPU = 0.0
    var buffers = 0.0     // building the 128 bytesNoCopy buffers
    var advise = 0.0      // F_RDADVISE + wait for 1,640/1,640 (not charged)
    var setBytes = 0      // MTLResidencySet.allocatedSize
    var residentBefore = 0
    var residentAfter = 0
    var ok = true         // output matched arm A bit for bit
    /// The number §13 compares against 4.25 ms: everything between "the eight
    /// experts' pages are in the cache" and "the GPU is done", which is what
    /// §11's `run` column measured.
    var inBandTotal: Double { encode + inBand }
    /// The same work with every page already faulted into this task and the
    /// GPU. 48 §3 measured the kernel at 0.25 ms by amortising a thousand
    /// passes inside one command buffer; a single command buffer carries its
    /// own fixed cost on top, so the honest floor for "in-band with no faults
    /// left" is this -- measured in the same trial, on the same mapping.
    var warmTotal: Double { warmEncode + warmInBand }
    /// What the first touch cost, over that floor. This is the quantity 48 §13
    /// is asking the arms to move out of the command buffer.
    var premium: Double { inBandTotal - warmTotal }
}

private func mmapP5Median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    return sorted.count % 2 == 1 ? sorted[sorted.count / 2]
        : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
}

private func mmapP5Quartiles(_ values: [Double]) -> (lo: Double, hi: Double) {
    guard values.count >= 4 else { return (values.min() ?? 0, values.max() ?? 0) }
    let sorted = values.sorted()
    let half = sorted.count / 2
    return (mmapP5Median(Array(sorted[0..<half])),
            mmapP5Median(Array(sorted[(sorted.count - half)...])))
}

/// Read one byte from every page of every range, in parallel. The stores land
/// in `sink`, which outlives the closure, so the reads cannot be optimised out.
private func mmapP5PreTouch(base: UnsafeMutableRawPointer,
                            ranges: [(offset: Int, length: Int)],
                            sink: UnsafeMutablePointer<UInt64>) {
    let pageSize = mmapProbePageSize
    DispatchQueue.concurrentPerform(iterations: ranges.count) { index in
        let range = ranges[index]
        var acc: UInt64 = 0
        var offset = 0
        while offset < range.length {
            acc &+= UInt64(base.load(fromByteOffset: range.offset + offset, as: UInt8.self))
            offset += pageSize
        }
        sink[index] = acc
    }
}

/// How the arms are interleaved. Neither order is free of 48 §5a's position
/// effect, and they fail differently, which is the point of having both.
enum MmapP5Order: String {
    /// `A B B* C D D C B* B A`. Every arm appears twice per round at mirrored
    /// positions, so a drift that runs one way across a round cancels -- but
    /// each arm still has the same two neighbours every time.
    case palindrome
    /// The same five arms twice per round, rotated one step each round, so
    /// across five rounds every arm has had every other arm in front of it.
    /// This is the one that answers "is the ranking an artifact of who ran
    /// before whom".
    case rotate

    func round(_ index: Int, only: MmapP5Arm? = nil) -> [MmapP5Arm] {
        // `only` exists for one question: arm A shares a process with arms that
        // create `MTLResidencySet`s, and if that leaves the driver warm in some
        // process-global way then A is no longer today's shape and the whole
        // comparison is upside down. Running one arm alone answers it.
        if let arm = only { return [arm, arm] }
        let arms = MmapP5Arm.allCases
        switch self {
        case .palindrome:
            return arms + arms.reversed()
        case .rotate:
            let shift = index % arms.count
            let rotated = Array(arms[shift...] + arms[..<shift])
            return rotated + rotated
        }
    }
}

func runMmapP5FaultProbe(groupSize: Int, modelPath: String, rounds: Int,
                         order orderMode: MmapP5Order, pollute: Int,
                         only: MmapP5Arm?, fixtureEarly: Bool,
                         experts: Int) throws {
    let layersDir = (modelPath as NSString).appendingPathComponent("packed_experts")
    let layoutPath = (layersDir as NSString).appendingPathComponent("layout.json")
    guard let layoutData = FileManager.default.contents(atPath: layoutPath),
          let layout = try JSONSerialization.jsonObject(with: layoutData) as? [String: Any],
          let expertStride = layout["expertStride"] as? Int,
          let expertsPerLayer = layout["expertsPerLayer"] as? Int
    else { throw NSError(domain: "mmap-p5", code: 1, userInfo: [
        NSLocalizedDescriptionKey: "cannot read \(layoutPath)"]) }

    let layerPath = (layersDir as NSString).appendingPathComponent("layer_00.bin")
    guard let attrs = try? FileManager.default.attributesOfItem(atPath: layerPath),
          let layerSize = (attrs[.size] as? NSNumber)?.intValue
    else { throw NSError(domain: "mmap-p5", code: 2, userInfo: [
        NSLocalizedDescriptionKey: "cannot stat \(layerPath)"]) }

    let pageSize = mmapProbePageSize
    let d = 2816, f = 704
    let rowsPerExpert = 2
    let ranks = (0..<experts).map { $0 * (expertsPerLayer / experts) }
    let adviseRanges = ranks.map { (offset: $0 * expertStride, length: expertStride) }
    let wantResident = experts * expertStride / pageSize
    let bytes = Double(experts * expertStride)

    print("=== P-5: can the fault cost be moved out of the command buffer? ===")
    print("48 §13. Control is 48 §11's 4.25 ms: 1,640 pages verified in the page")
    print("cache by `mincore`, then one command buffer naming eight experts.")
    print("Inference path untouched; MAP_SHARED + PROT_READ throughout (48 §2).")
    print("")
    print("  model               \(modelPath)")
    print("  layer file          \(layerSize) B = \(layerSize / pageSize) pages")
    print("  experts / rows      \(experts) x \(rowsPerExpert) rows, ranks "
            + ranks.map(String.init).joined(separator: ","))
    print("  NOTE                48 §11 and P-5 both divide a per-command-buffer")
    print("                      cost by the page count and call it us/page. That")
    print("                      is linear by assumption, not by measurement, so")
    print("                      sweep --mmap-p5-experts to get the slope.")
    print("  step footprint      \(wantResident) pages = "
            + String(format: "%.2f MB", bytes / 1e6))
    if let arm = only { print("  ARMS RESTRICTED TO  \(arm.rawValue)") }
    // The one protocol difference left between arm A and 48 §11's control,
    // which is 2.5x more expensive on what is otherwise the same steps: §11's
    // P-4 loop builds the argument buffer *before* it issues `F_RDADVISE`, so
    // `MTLArgumentEncoder.setBuffer` writes each expert's GPU address while the
    // pages behind it are still evicted. P-5 builds it after the wait. If that
    // ordering is worth milliseconds it is worth knowing, because production
    // has the same choice: `ExpertPrefetch` runs a layer ahead of encoding.
    print("  argument buffer     built \(fixtureEarly ? "BEFORE the advise (48 §11's order)" : "after residency (P-5's order)")")
    print("  interleave          \(orderMode.rawValue), \(rounds) rounds, "
            + "n=\(2 * rounds) per arm")
    print("  PASS (48 §13)       in-band <= 1.0 ms AND prep <= 1.5 ms")
    print("  FAIL (48 §13)       no arm gets in-band below 3.0 ms")
    print("")

    let context = try makeContext(groupSize: groupSize)
    let probe = try BpwProbe(context: context, groupSize: groupSize, d: d, f: f)
    let gateUpPSO = try probe.pipeline(.gateUp, cap: rowsPerExpert, format: .sym)
    let downPSO = try probe.pipeline(.down, cap: rowsPerExpert, format: .sym)
    let rowCounts = [Int](repeating: rowsPerExpert, count: experts)
        + [Int](repeating: 0, count: bpwTileExperts - experts)

    var results: [MmapP5Arm: [MmapP5Trial]] = [:]
    var reference: [Float]?
    var mismatches = 0
    let sink = UnsafeMutablePointer<UInt64>.allocate(capacity: experts)
    sink.initialize(repeating: 0, count: experts)
    defer { sink.deallocate() }

    // ---- optional: reproduce the state 48 §11 was measured in ----------
    //
    // 48 §11's control is 4.25 ms and arm A here is not, on the same protocol.
    // The one thing that differs is process history: §11 was taken in the P-1
    // to P-4 sweep, whose `MAP_PRIVATE` arms turn 430 MB anonymous per trial
    // (§2) and which 48 §5 already caught drifting the gate ratio 1.16 -> 1.31
    // for that reason. So rather than assert the explanation, `--mmap-p5-pollute`
    // does it: N `MAP_PRIVATE` layer-spanning trials, exactly the arm that
    // dirties the memory, then the same five arms measured after them.
    func hostLine(_ label: String) {
        let host = MmapHostVM.sample()
        print(String(format: "  %@  free %.2f GB   compressed %.2f GB   "
                        + "wired %.2f GB", (label as NSString),
                     Double(host.free) * Double(pageSize) / 1e9,
                     Double(host.compressed) * Double(pageSize) / 1e9,
                     Double(host.wired) * Double(pageSize) / 1e9))
    }
    hostLine("host before        ")

    if pollute > 0 {
        print("")
        print("  polluting: \(pollute) x MAP_PRIVATE layer-spanning trials "
                + "(430 MB anonymous each, 48 §2)")
        var dirtied = 0.0
        for _ in 0..<pollute {
            mmapProbeColdDrop(path: layerPath)
            let fd = open(layerPath, O_RDONLY)
            guard fd >= 0 else { break }
            guard let raw = mmap(nil, layerSize, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_FILE, fd, 0),
                  raw != MAP_FAILED else { close(fd); break }
            guard let whole = context.device.makeBuffer(
                bytesNoCopy: raw, length: layerSize,
                options: .storageModeShared, deallocator: nil) else {
                munmap(raw, layerSize); close(fd); break
            }
            var fixture: BpwFixture? = probe.makeFixture(
                format: .sym,
                slots: (0..<bpwTileExperts).map {
                    (buffer: whole, offset: ranks[min($0, ranks.count - 1)] * expertStride)
                },
                blobs: [whole])
            probe.fillHidden()
            guard let cmd = context.queue.makeCommandBuffer() else { fatalError("cmd") }
            probe.encode(.gateUp, pso: gateUpPSO, commandBuffer: cmd,
                         fixture: fixture!, rowCounts: rowCounts)
            probe.encode(.down, pso: downPSO, commandBuffer: cmd,
                         fixture: fixture!, rowCounts: rowCounts)
            waitAndCheck(cmd, "p5 pollution")
            // The whole-layer arm only faults what the kernel reads, so the
            // pages are walked from the CPU to get the full 430 MB dirty --
            // the same amount the P-1 per-layer arm reached over its 1,200
            // repeats.
            var acc: UInt64 = 0
            var offset = 0
            while offset < layerSize {
                acc &+= UInt64(raw.load(fromByteOffset: offset, as: UInt8.self))
                offset += pageSize
            }
            dirtied += Double(acc == .max ? 0 : layerSize)
            fixture = nil
            munmap(raw, layerSize)
            close(fd)
        }
        print(String(format: "  dirtied %.1f GB of anonymous memory", dirtied / 1e9))
        hostLine("host after pollute ")
    }
    print("")

    // 48 §5a measured a position effect (0.58--1.03 ms) as large as the
    // difference being looked for, so the arms are interleaved rather than run
    // in blocks, and the interleave itself is a swept parameter.
    for round in 0..<rounds {
        for arm in orderMode.round(round, only: only) {
            var trial = MmapP5Trial()

            // Same cold protocol as §11: evict, map fresh, then advise and wait
            // until every one of the 1,640 pages is in the cache. The wait is
            // reported but not charged -- P-5 is about what happens after it.
            mmapProbeColdDrop(path: layerPath)
            let fd = open(layerPath, O_RDONLY)
            guard fd >= 0 else { break }
            guard let raw = mmap(nil, layerSize, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0),
                  raw != MAP_FAILED else { close(fd); break }
            let base = raw

            let bufferStart = Date()
            var owned: [MTLBuffer] = []
            owned.reserveCapacity(expertsPerLayer)
            for rank in 0..<expertsPerLayer {
                guard let buffer = context.device.makeBuffer(
                    bytesNoCopy: base.advanced(by: rank * expertStride),
                    length: expertStride, options: .storageModeShared,
                    deallocator: nil) else { break }
                owned.append(buffer)
            }
            trial.buffers = Date().timeIntervalSince(bufferStart)
            guard owned.count == expertsPerLayer else {
                owned = []; munmap(base, layerSize); close(fd); break
            }
            let named = ranks.map { owned[$0] }

            let slots = (0..<bpwTileExperts).map {
                (buffer: owned[ranks[min($0, ranks.count - 1)]], offset: 0)
            }
            var fixture: BpwFixture?
            if fixtureEarly {
                fixture = probe.makeFixture(format: .sym, slots: slots,
                                            blobs: arm.usesUseResource ? named : [])
                probe.fillHidden()
            }

            let adviseStart = Date()
            _ = mmapProbeReadAdvise(fd: fd, ranges: adviseRanges)
            let adviseDeadline = Date().addingTimeInterval(2.0)
            while Date() < adviseDeadline {
                if mmapProbeResident(base: base, ranges: adviseRanges) >= wantResident { break }
            }
            trial.advise = Date().timeIntervalSince(adviseStart)
            trial.residentBefore = mmapProbeResident(base: base, ranges: adviseRanges)

            // ---- arm-specific preparation ------------------------------
            //
            // Both halves are dispatched before either is waited on, so arm D
            // measures the prep the way production would run it (one layer
            // ahead, everything at once) rather than the sum of two serials.
            var residencySet: (any MTLResidencySet)?
            let prepStart = Date()
            var setSeconds = 0.0
            var touchSeconds = 0.0
            let setDone = DispatchSemaphore(value: 0)
            let touchDone = DispatchSemaphore(value: 0)

            if arm.usesResidencySet {
                let descriptor = MTLResidencySetDescriptor()
                descriptor.label = "p5-\(arm.rawValue.prefix(2))"
                descriptor.initialCapacity = experts
                residencySet = try? context.device.makeResidencySet(descriptor: descriptor)
                if let set = residencySet {
                    DispatchQueue.global(qos: .userInitiated).async {
                        let started = Date()
                        for buffer in named { set.addAllocation(buffer) }
                        set.commit()
                        set.requestResidency()
                        setSeconds = Date().timeIntervalSince(started)
                        setDone.signal()
                    }
                } else {
                    setDone.signal()
                }
            } else {
                setDone.signal()
            }

            if arm.cpuPreTouch {
                DispatchQueue.global(qos: .userInitiated).async {
                    let started = Date()
                    mmapP5PreTouch(base: base, ranges: adviseRanges, sink: sink)
                    touchSeconds = Date().timeIntervalSince(started)
                    touchDone.signal()
                }
            } else {
                touchDone.signal()
            }

            setDone.wait()
            touchDone.wait()
            trial.prep = Date().timeIntervalSince(prepStart)
            trial.prepSet = setSeconds
            trial.prepCPU = touchSeconds
            trial.setBytes = Int(residencySet?.allocatedSize ?? 0)

            // ---- the in-band measurement -------------------------------
            //
            // `blobs` is the list `BpwProbe.encode` calls `useResource` on, so
            // an arm that hands the residency set the job passes none.
            if !fixtureEarly {
                fixture = probe.makeFixture(format: .sym, slots: slots,
                                            blobs: arm.usesUseResource ? named : [])
                probe.fillHidden()
            }

            let encodeStart = Date()
            guard let cmd = context.queue.makeCommandBuffer() else { fatalError("cmd") }
            if let set = residencySet { cmd.useResidencySet(set) }
            probe.encode(.gateUp, pso: gateUpPSO, commandBuffer: cmd,
                         fixture: fixture!, rowCounts: rowCounts)
            probe.encode(.down, pso: downPSO, commandBuffer: cmd,
                         fixture: fixture!, rowCounts: rowCounts)
            trial.encode = Date().timeIntervalSince(encodeStart)

            let commitStart = Date()
            cmd.commit()
            cmd.waitUntilCompleted()
            trial.inBand = Date().timeIntervalSince(commitStart)
            if let error = cmd.error { fatalError("P-5 \(arm.rawValue) failed — \(error)") }
            trial.gpu = cmd.gpuEndTime - cmd.gpuStartTime
            trial.residentAfter = mmapProbeResident(base: base, ranges: adviseRanges)

            // The same command buffer again, on the same mapping. Nothing is
            // cold any more -- this is 48 §3's P-2 condition reached from
            // inside this trial, and it is the floor the first pass is charged
            // against.
            let warmEncodeStart = Date()
            guard let warm = context.queue.makeCommandBuffer() else { fatalError("cmd") }
            if let set = residencySet { warm.useResidencySet(set) }
            probe.encode(.gateUp, pso: gateUpPSO, commandBuffer: warm,
                         fixture: fixture!, rowCounts: rowCounts)
            probe.encode(.down, pso: downPSO, commandBuffer: warm,
                         fixture: fixture!, rowCounts: rowCounts)
            trial.warmEncode = Date().timeIntervalSince(warmEncodeStart)
            let warmCommit = Date()
            warm.commit()
            warm.waitUntilCompleted()
            trial.warmInBand = Date().timeIntervalSince(warmCommit)
            if let error = warm.error { fatalError("P-5 warm \(arm.rawValue) — \(error)") }
            trial.warmGPU = warm.gpuEndTime - warm.gpuStartTime

            // Bit-equality against arm A. An arm that skips `useResource` and
            // leans on the residency set has to produce the same weights, or
            // the number it produced is not the number being asked for.
            let out = probe.readAct(rows: experts * rowsPerExpert)
            if let ref = reference {
                trial.ok = ref.count == out.count && zip(ref, out).allSatisfy { $0 == $1 }
                if !trial.ok { mismatches += 1 }
            } else if out.allSatisfy({ $0.isFinite }) {
                reference = out
            }

            fixture = nil
            residencySet?.endResidency()
            residencySet = nil
            owned = []
            munmap(base, layerSize)
            close(fd)

            results[arm, default: []].append(trial)
        }
        if (round + 1) % 2 == 0 {
            print("  ... round \(round + 1)/\(rounds)")
        }
    }

    // ---- report ---------------------------------------------------------
    //
    // The labels carry a `s` where the write-up carries a section mark: these
    // columns are padded by character count and the report is read in a
    // terminal, so a multi-byte glyph in a padded field breaks the table.
    func pad(_ text: String, _ width: Int) -> String {
        text.count >= width ? text
            : text + String(repeating: " ", count: width - text.count)
    }

    print("")
    print("--- in-band: encode + commit + completion, per 8 experts "
            + String(repeating: "-", count: 12))
    print("    the SAME command buffer run a second time on the SAME mapping is")
    print("    the floor: no page is cold, so `premium` is what the first touch")
    print("    cost -- the quantity 48 s13 asks the arms to move out.")
    print("")
    print("    " + pad("arm", 32) + "  n   1st pass         IQR         "
            + " GPU     warm    premium")
    var bestPremium = Double.infinity
    var bestArm = MmapP5Arm.a
    for arm in MmapP5Arm.allCases {
        guard let trials = results[arm], !trials.isEmpty else { continue }
        let first = mmapP5Median(trials.map(\.inBandTotal))
        let quartiles = mmapP5Quartiles(trials.map(\.inBandTotal))
        let gpu = mmapP5Median(trials.map(\.gpu))
        let warm = mmapP5Median(trials.map(\.warmTotal))
        let premium = mmapP5Median(trials.map(\.premium))
        if premium < bestPremium { bestPremium = premium; bestArm = arm }
        print("    " + pad(arm.rawValue, 32)
                + String(format: " %2d  %6.2f ms  %5.2f..%-5.2f  %5.2f  %6.2f  %+8.2f ms",
                         trials.count, first * 1e3,
                         quartiles.lo * 1e3, quartiles.hi * 1e3,
                         gpu * 1e3, warm * 1e3, premium * 1e3))
    }
    print("    (48 s11's control, measured in its own process: 4.25 ms in-band)")
    print("")

    print("--- preparation: what has to hide behind the previous layer "
            + String(repeating: "-", count: 9))
    print("    " + pad("arm", 32) + "  median    of which set  of which touch"
            + "  128 buffers")
    for arm in MmapP5Arm.allCases {
        guard let trials = results[arm], !trials.isEmpty else { continue }
        print("    " + pad(arm.rawValue, 32)
                + String(format: " %6.2f ms  %9.2f ms  %11.2f ms  %8.2f ms",
                         mmapP5Median(trials.map(\.prep)) * 1e3,
                         mmapP5Median(trials.map(\.prepSet)) * 1e3,
                         mmapP5Median(trials.map(\.prepCPU)) * 1e3,
                         mmapP5Median(trials.map(\.buffers)) * 1e3))
    }
    print("")

    print("--- residency and correctness " + String(repeating: "-", count: 39))
    for arm in MmapP5Arm.allCases {
        guard let trials = results[arm], !trials.isEmpty else { continue }
        let setBytes = trials.map(\.setBytes).max() ?? 0
        let bad = trials.filter { !$0.ok }.count
        print("    " + pad(arm.rawValue, 32)
                + String(format: " resident %d/%d before, %d after   set %5.1f MB   ",
                         trials.last?.residentBefore ?? 0, wantResident,
                         trials.last?.residentAfter ?? 0, Double(setBytes) / 1e6)
                + (bad == 0 ? "output bit-identical" : "*** \(bad) MISMATCH ***"))
    }
    print("    advise+wait, not charged to any arm: median "
            + String(format: "%.2f ms", mmapP5Median(
                MmapP5Arm.allCases.flatMap { results[$0] ?? [] }.map(\.advise)) * 1e3)
            + "   (48 s11 measured 2.70 ms)")
    print("")

    hostLine("host after         ")
    print("")

    print("--- raw, in trial order " + String(repeating: "-", count: 45))
    for arm in MmapP5Arm.allCases {
        guard let trials = results[arm], !trials.isEmpty else { continue }
        print("  \(arm.rawValue)")
        print("    1st pass " + trials.map { String(format: "%.2f", $0.inBandTotal * 1e3) }
                .joined(separator: " "))
        print("    warm     " + trials.map { String(format: "%.2f", $0.warmTotal * 1e3) }
                .joined(separator: " "))
        print("    prep     " + trials.map { String(format: "%.2f", $0.prep * 1e3) }
                .joined(separator: " "))
    }
    print("")

    let bestPrep = mmapP5Median((results[bestArm] ?? []).map(\.prep))
    let bestInBand = mmapP5Median((results[bestArm] ?? []).map(\.inBandTotal))
    print("--- 48 s13 verdict " + String(repeating: "-", count: 49))
    print("    best arm (by premium): \(bestArm.rawValue)")
    print(String(format: "    in-band %.2f ms, of which premium %.2f ms; prep %.2f ms",
                 bestInBand * 1e3, bestPremium * 1e3, bestPrep * 1e3))
    if mismatches > 0 {
        print("    *** \(mismatches) trials produced different output; the timings "
                + "above are not comparable ***")
    }
    if bestInBand <= 0.0010 && bestPrep <= 0.0015 {
        print("    PASS — the fault cost leaves the command buffer and the prep "
                + "fits a one-layer lead. Next is P-6.")
    } else if bestInBand >= 0.0030 {
        print("    FAIL — no arm gets in-band below 3 ms. D is not an I/O card on "
                + "this machine (48 §13); close the branch and return to 40-HANDOFF (a).")
    } else {
        print("    BETWEEN — in-band moved but not to 1 ms. 48 §13 leaves this to "
                + "the user with the numbers on the table.")
    }
    print("")
}
