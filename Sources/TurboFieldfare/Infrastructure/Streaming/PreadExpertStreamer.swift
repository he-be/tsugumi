import Darwin
import Foundation
import Metal

public struct ExpertIOAdviceResult: Sendable, Equatable {
    public let requested: Int
    public let failed: Int
    public let calls: Int
    public let bytes: UInt64
    public let skipped: Int
    public let maxCallNanos: UInt64

    public init(requested: Int,
                failed: Int,
                calls: Int? = nil,
                bytes: UInt64 = 0,
                skipped: Int = 0,
                maxCallNanos: UInt64 = 0) {
        self.requested = requested
        self.failed = failed
        self.calls = calls ?? requested
        self.bytes = bytes
        self.skipped = skipped
        self.maxCallNanos = maxCallNanos
    }

    public static func skipped(requested: Int, bytes: UInt64 = 0) -> ExpertIOAdviceResult {
        ExpertIOAdviceResult(requested: requested,
                             failed: 0,
                             calls: 0,
                             bytes: bytes,
                             skipped: requested)
    }

}

public struct ExpertCachePlan: Sendable, Equatable {
    public let experts: [Int]
    public let assignedSlots: [Int]
    public let misses: [Int]
    public let hits: Int

    public init(experts: [Int], assignedSlots: [Int], misses: [Int], hits: Int) {
        self.experts = experts
        self.assignedSlots = assignedSlots
        self.misses = misses
        self.hits = hits
    }
}

public enum ExpertCachePolicy: String, Sendable {
    case lru
    case lfu
}

/// `pread`-based routed-expert streamer with a fixed per-layer slot cache.
public final class PreadExpertStreamer: @unchecked Sendable {
    public static let scratchAlignment = 2 * 1024 * 1024
    public static var cachePolicyDefault: ExpertCachePolicy { .lfu }

    public let layout: StreamLayout
    public let slotCount: Int
    public let cachePolicy: ExpertCachePolicy

    private let fd: Int32
    private let slotPointers: [UnsafeMutableRawPointer]
    private let slotBuffers: [MTLBuffer]
    /// D の試作 (`TF_EXPERT_MMAP=1`)。非 nil のときスロットは 1 本も無く、
    /// エキスパートはファイルのマッピングから直接 GPU に渡る
    /// (`MmapExpertMapping`、docs/mtp/49 §9)。プランと LFU の帳簿はそのままで、
    /// 変わるのは**ミスが何をするか**だけ — `pread` でコピーする代わりに
    /// residency set を更新する。
    let mmap: MmapExpertMapping?

    private var nextSlot = 0
    private let cursorLock = NSLock()

    private var slotExpert: [Int]
    private var slotLastUse: [Int]
    private var expertUseCount: [Int]
    /// How many slots currently hold each expert. Zero is a definite miss, so
    /// the common case costs one array read instead of a scan over every slot.
    /// Usually zero or one; it can exceed one when a plan had to re-read an
    /// expert whose slot was pinned by `avoidingSlots`.
    private var expertResidency: [Int]
    /// One slot known to hold each expert. A hint only — it is validated before
    /// use, and a stale hint just falls back to the scan the old code always did.
    private var expertSlotHint: [Int]
    private var useClock = 0
    private let cacheLock = NSLock()

    public convenience init(layout: StreamLayout,
                            device: MTLDevice,
                            slotCount: Int,
                            cachePolicy: ExpertCachePolicy = .lfu) throws {
        try self.init(layout: layout,
                      device: device,
                      slotCount: slotCount,
                      cachePolicy: cachePolicy,
                      fileDescriptor: nil)
    }

    package init(layout: StreamLayout,
                 device: MTLDevice,
                 slotCount: Int,
                 cachePolicy: ExpertCachePolicy = .lfu,
                 fileDescriptor: Int32?,
                 useMmap: Bool = MmapExpertMapping.isEnabled) throws {
        precondition(slotCount > 0, "slotCount must be positive")
        self.layout = layout
        self.slotCount = slotCount
        self.cachePolicy = cachePolicy
        let pageSize = Int(getpagesize())

        let openedFD = fileDescriptor.map { fcntl($0, F_DUPFD_CLOEXEC, 0) }
            ?? open(layout.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
        guard openedFD >= 0 else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        // `docs/mtp/46-W3-CODE-ENTROPY.md` §5/§6: this fd deliberately does not
        // set `F_NOCACHE`, so an expert read is served from the unified buffer
        // cache whenever the page is still there -- which is why 46 §5a measures
        // the effective bandwidth moving 3.87 -> 9.09 GB/s with generation
        // length alone while the bytes stay fixed. How much of today's `io` is
        // that cache rather than the SSD decides the whole prize of the D branch
        // (49 §10): `mmap` removes the copy out of the cache, it does not remove
        // the SSD. Setting this makes the reads bypass the cache, so the two
        // arms bracket the answer. Instrument only -- never a product setting.
        if ProcessInfo.processInfo.environment["TF_EXPERT_NOCACHE"] == "1" {
            _ = fcntl(openedFD, F_NOCACHE, 1)
        }
        self.fd = openedFD
        var closeFDOnFailure = true
        defer { if closeFDOnFailure { close(openedFD) } }

        var fileStats = stat()
        guard fstat(openedFD, &fileStats) == 0,
              (fileStats.st_mode & S_IFMT) == S_IFREG,
              fileStats.st_size >= 0 else {
            throw StreamerError.openFailed(
                path: layout.path, errno: errno == 0 ? EINVAL : errno)
        }
        let (required, requiredOverflow) = layout.streamOffset
            .addingReportingOverflow(layout.streamSize)
        guard !requiredOverflow, UInt64(fileStats.st_size) >= required else {
            throw StreamerError.sizeMismatch(
                expected: requiredOverflow ? UInt64.max : required,
                actual: UInt64(fileStats.st_size))
        }
        guard layout.expertStride > 0,
              layout.expertStride <= UInt64(Int.max - (pageSize - 1)) else {
            throw StreamerError.invalidIOSplitConfiguration(
                "expertStride \(layout.expertStride) is not addressable")
        }

        let allocationSize = ((Int(layout.expertStride) + pageSize - 1) / pageSize) * pageSize
        var pointers: [UnsafeMutableRawPointer] = []
        var buffers: [MTLBuffer] = []
        pointers.reserveCapacity(slotCount)
        buffers.reserveCapacity(slotCount)

        // D の試作: スロットを 1 個も持たない。私有 anonymous の 3.22 GB
        // (32 スロット x 30 層) が消えるのが D の残る根拠である (50 §4)。
        self.mmap = useMmap
            ? try MmapExpertMapping(layout: layout, device: device,
                                    fileDescriptor: openedFD)
            : nil

        func unwind() {
            for index in buffers.count..<pointers.count {
                free(pointers[index])
            }
        }

        for _ in 0..<(useMmap ? 0 : slotCount) {
            var raw: UnsafeMutableRawPointer?
            let result = posix_memalign(&raw, Self.scratchAlignment, allocationSize)
            guard result == 0, let pointer = raw else {
                unwind()
                throw StreamerError.allocFailed(errno: result)
            }
            pointers.append(pointer)
            nonisolated(unsafe) let capturedPointer = pointer
            guard let buffer = device.makeBuffer(
                bytesNoCopy: pointer,
                length: allocationSize,
                options: .storageModeShared,
                deallocator: { _, _ in free(capturedPointer) })
            else {
                unwind()
                throw StreamerError.bufferWrapFailed
            }
            buffers.append(buffer)
        }

        self.slotPointers = pointers
        self.slotBuffers = buffers
        self.slotExpert = [Int](repeating: -1, count: slotCount)
        self.slotLastUse = [Int](repeating: 0, count: slotCount)
        self.expertUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        self.expertResidency = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        self.expertSlotHint = [Int](repeating: -1, count: max(1, layout.expertsPerLayer))
        closeFDOnFailure = false
    }

    deinit {
        close(fd)
    }

    public func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        cursorLock.lock()
        let slot = nextSlot
        nextSlot = (nextSlot + 1) % slotCount
        cursorLock.unlock()
        return try loadExpert(layer: layer, expert: expert, slot: slot)
    }

    public func loadExpert(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard slot >= 0 && slot < slotCount else {
            throw StreamerError.slotOutOfRange(slot)
        }
        guard expert >= 0, expert < layout.expertsPerLayer else {
            throw StreamerError.offsetOutOfRange(UInt64(max(0, expert)))
        }
        let regionOffset = layout.expertOffset(layer: layer, expert: expert)
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        // D の試作ではバイトは既に GPU から見えるアドレスに居る。コピーは無い。
        if let mmap {
            return (mmap.expertBuffers[expert], 0, layout.expertStride)
        }
        try readFull(
            into: slotPointers[slot],
            fileOffset: layout.streamOffset + regionOffset,
            count: Int(layout.expertStride))
        return (slotBuffers[slot], 0, layout.expertStride)
    }

    public func loadExpertsCached(experts: [Int]) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlan(planExpertsCached(experts: experts))
    }

    public func planExpertsCached(experts: [Int],
                                  avoidingSlots: Set<Int> = []) -> ExpertCachePlan {
        guard let plan = makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots) else {
            preconditionFailure("expert cache cannot place requested misses")
        }
        return plan
    }

    public func planExpertsCachedIfPossible(experts: [Int],
                                            avoidingSlots: Set<Int> = []) -> ExpertCachePlan? {
        makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots)
    }

    /// A plan for experts nothing has asked for yet.
    ///
    /// Running layer `L + 1`'s router weights on layer `L`'s hidden state names
    /// two thirds of the experts layer `L + 1` is about to miss, a layer before
    /// it asks for them (docs/mtp/29-M8-B-PROBE.md §2). A guess is planned by
    /// different rules than a request, because a wrong guess must not cost more
    /// than a right one saves (§6):
    ///
    /// - **The use count does not move.** A request counts every expert it
    ///   places, and under LFU that count is what keeps a slot. A guess that
    ///   counted would leave an expert nothing ever used protected in the cache.
    /// - **Only slots the last plan did not touch may be evicted.** A guess that
    ///   throws out what this layer used in its previous round pays for its hit
    ///   with a miss somewhere else. `nil` when no such slot is free — giving
    ///   the prefetch up is always allowed, since the layer will fetch what it
    ///   needs when it gets there.
    public func planSpeculativeExperts(experts: [Int],
                                       avoidingSlots: Set<Int> = []) -> ExpertCachePlan? {
        guard !experts.isEmpty, experts.count <= slotCount else { return nil }
        return makeExpertCachePlan(experts: experts,
                                   avoidingSlots: avoidingSlots,
                                   speculative: true)
    }

    private func makeExpertCachePlan(experts: [Int],
                                     avoidingSlots rawAvoidingSlots: Set<Int>,
                                     speculative: Bool = false) -> ExpertCachePlan? {
        precondition(experts.count <= slotCount,
                     "expert cache needs at least \(experts.count) slots")
        let avoidingSlots = Set(rawAvoidingSlots.filter { $0 >= 0 && $0 < slotCount })

        cacheLock.lock()
        defer { cacheLock.unlock() }

        // A speculative plan does not advance the clock: what "the last round"
        // means has to stay the round the layer actually ran.
        let clock = speculative ? useClock : useClock + 1
        var assignedSlots = [Int](repeating: -1, count: experts.count)
        var reserved = [Bool](repeating: false, count: slotCount)

        for index in experts.indices {
            let expert = experts[index]
            guard expert >= 0, expert < expertResidency.count,
                  expertResidency[expert] > 0 else { continue }
            let hint = expertSlotHint[expert]
            if hint >= 0, !reserved[hint], slotExpert[hint] == expert {
                assignedSlots[index] = hint
                reserved[hint] = true
                continue
            }
            // The hint is taken by an earlier duplicate or has gone stale.
            // Another slot may still hold this expert, so fall back to a scan.
            for slot in 0..<slotCount
                where !reserved[slot] && slotExpert[slot] == expert {
                assignedSlots[index] = slot
                reserved[slot] = true
                break
            }
        }
        for slot in avoidingSlots where !reserved[slot] {
            reserved[slot] = true
        }
        if speculative {
            // Everything the most recent plan touched — and everything an
            // earlier guess of this round already claimed, which is stamped
            // with the same clock below — is off limits.
            for slot in 0..<slotCount
            where !reserved[slot] && slotLastUse[slot] >= useClock {
                reserved[slot] = true
            }
        }

        let misses = experts.indices.filter { assignedSlots[$0] == -1 }
        // Only the `misses.count` cheapest victims are ever used, so pick them
        // directly. Sorting every slot dominated planning once the cache grew
        // past a few dozen slots, and it ran even when nothing had to be evicted.
        let victims = misses.isEmpty
            ? []
            : cheapestEvictableSlots(count: misses.count, reserved: reserved)
        guard misses.count <= victims.count else { return nil }

        useClock = clock
        // A guess leaves the cache's ranking exactly as it found it: no use
        // count moves, and the slots it merely read from keep the recency the
        // layer's own last plan gave them. Only the slots it takes are stamped,
        // which is what keeps a second guess of the same round off them.
        if !speculative {
            for expert in experts where expert >= 0 && expert < expertUseCount.count {
                expertUseCount[expert] &+= 1
            }
            for slot in assignedSlots where slot >= 0 {
                slotLastUse[slot] = clock
            }
        }
        for (offset, index) in misses.enumerated() {
            let slot = victims[offset]
            assignedSlots[index] = slot
            reserved[slot] = true
            releaseSlotLocked(slot)
            slotLastUse[slot] = clock
        }

        return ExpertCachePlan(
            experts: experts,
            assignedSlots: assignedSlots,
            misses: misses,
            hits: experts.count - misses.count)
    }

    public func executeExpertCachePlan(_ plan: ExpertCachePlan) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.experts.count <= slotCount,
                     "expert cache plan exceeds slot count")
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")

        // D の試作: 読むものが無い。ミスがすることは「residency set に足す」
        // だけで、それは帳簿を更新したあとに 1 回だけ出す (49 §9)。
        if let mmap {
            mmap.syncResidency {
                cacheLock.lock()
                defer { cacheLock.unlock() }
                for index in plan.misses {
                    let slot = plan.assignedSlots[index]
                    let expert = plan.experts[index]
                    releaseSlotLocked(slot)
                    slotExpert[slot] = expert
                    if expert >= 0, expert < expertResidency.count {
                        expertResidency[expert] += 1
                        expertSlotHint[expert] = slot
                    }
                }
                return Set(slotExpert.filter { $0 >= 0 })
            }
            return expertCachePlanBuffers(plan)
        }

        let errorLock = NSLock()
        nonisolated(unsafe) var firstError: Error?
        DispatchQueue.concurrentPerform(iterations: plan.misses.count) { missOffset in
            let index = plan.misses[missOffset]
            do {
                _ = try self.loadExpert(
                    layer: 0,
                    expert: plan.experts[index],
                    slot: plan.assignedSlots[index])
            } catch {
                errorLock.lock()
                if firstError == nil { firstError = error }
                errorLock.unlock()
            }
        }
        if let firstError { throw firstError }

        cacheLock.lock()
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            let expert = plan.experts[index]
            releaseSlotLocked(slot)
            slotExpert[slot] = expert
            if expert >= 0, expert < expertResidency.count {
                expertResidency[expert] += 1
                expertSlotHint[expert] = slot
            }
        }
        cacheLock.unlock()

        return expertCachePlanBuffers(plan)
    }

    public func expertCachePlanBuffers(_ plan: ExpertCachePlan)
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")
        if let mmap {
            return plan.experts.map { expert in
                (mmap.expertBuffers[expert], UInt64(0), layout.expertStride)
            }
        }
        return plan.assignedSlots.map { slot in
            (slotBuffers[slot], UInt64(0), layout.expertStride)
        }
    }

    public func adviseExpertCachePlanMisses(_ plan: ExpertCachePlan) -> ExpertIOAdviceResult {
        let experts = plan.misses.map { plan.experts[$0] }
        return adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExperts(experts: [Int]) -> ExpertIOAdviceResult {
        adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    /// Drop every cached expert, leaving the cache as it was at open time.
    ///
    /// Measurement only (`--verify-cold`, docs/mtp/21-GOAL-CONDITION-RESULTS.md): the
    /// warm cost probe replays a stream decode has already fetched, so it
    /// reports the case where the block has no expert I/O to fold. Starting
    /// each timed phase from the same empty cache measures the other end.
    /// The file's pages stay in the OS page cache — this cools the slot cache,
    /// not the SSD.
    public func resetCache() {
        cacheLock.lock()
        for slot in slotExpert.indices {
            slotExpert[slot] = -1
            slotLastUse[slot] = 0
        }
        for expert in expertUseCount.indices {
            expertUseCount[expert] = 0
            expertResidency[expert] = 0
            expertSlotHint[expert] = -1
        }
        useClock = 0
        cacheLock.unlock()
        cursorLock.lock()
        nextSlot = 0
        cursorLock.unlock()
        mmap?.dropResidency()
    }

    /// Which of `experts` a plan made now would find in a slot.
    ///
    /// A read-only view of the same residency table `makeExpertCachePlan` uses,
    /// so a caller can order its work by what is already there without taking
    /// slots (docs/mtp/27-M7-RESULTS.md §4).
    public func residentExperts(_ experts: [Int]) -> [Bool] {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return experts.map { expert in
            expert >= 0 && expert < expertResidency.count && expertResidency[expert] > 0
        }
    }

    public func adviseExpertMisses(experts: [Int]) -> ExpertIOAdviceResult {
        cacheLock.lock()
        let misses = experts.filter { expert in
            guard expert >= 0, expert < expertResidency.count else { return true }
            return expertResidency[expert] == 0
        }
        cacheLock.unlock()
        return adviseRanges(expertAdviceRanges(experts: misses), requested: misses.count)
    }

    static func coalescedAdjacentAdviceRanges(_ ranges: [(offset: UInt64, count: UInt64)])
        -> [(offset: UInt64, count: UInt64)] {
        let sorted = ranges.filter { $0.count > 0 }.sorted {
            $0.offset == $1.offset ? $0.count < $1.count : $0.offset < $1.offset
        }
        var result: [(offset: UInt64, count: UInt64)] = []
        for range in sorted {
            guard var last = result.popLast() else {
                result.append(range)
                continue
            }
            let lastEnd = last.offset &+ last.count
            let rangeEnd = range.offset &+ range.count
            if range.offset <= lastEnd {
                last.count = max(lastEnd, rangeEnd) - last.offset
                result.append(last)
            } else {
                result.append(last)
                result.append(range)
            }
        }
        return result
    }

    /// The `count` slots that `shouldEvictSlot` ranks first, in that order.
    /// Equivalent to sorting every unreserved slot and taking the prefix, but it
    /// keeps only `count` candidates (topK, so 8 during decode) in flight.
    /// Returns fewer than `count` when there are not enough unreserved slots.
    private func cheapestEvictableSlots(count: Int, reserved: [Bool]) -> [Int] {
        var picked: [Int] = []
        picked.reserveCapacity(count)
        for slot in 0..<slotCount where !reserved[slot] {
            if picked.count < count {
                picked.append(slot)
                var index = picked.count - 1
                while index > 0, shouldEvictSlot(slot, before: picked[index - 1]) {
                    picked[index] = picked[index - 1]
                    index -= 1
                }
                picked[index] = slot
            } else if shouldEvictSlot(slot, before: picked[count - 1]) {
                var index = count - 1
                while index > 0, shouldEvictSlot(slot, before: picked[index - 1]) {
                    picked[index] = picked[index - 1]
                    index -= 1
                }
                picked[index] = slot
            }
        }
        return picked
    }

    /// Mark `slot` as holding nothing, keeping the residency index in step.
    /// Callers must already hold `cacheLock`.
    private func releaseSlotLocked(_ slot: Int) {
        let evicted = slotExpert[slot]
        if evicted >= 0, evicted < expertResidency.count {
            expertResidency[evicted] -= 1
            if expertSlotHint[evicted] == slot { expertSlotHint[evicted] = -1 }
        }
        slotExpert[slot] = -1
    }

    private func shouldEvictSlot(_ lhs: Int, before rhs: Int) -> Bool {
        if cachePolicy == .lru {
            return slotLastUse[lhs] < slotLastUse[rhs]
        }
        let lhsExpert = slotExpert[lhs]
        let rhsExpert = slotExpert[rhs]
        if lhsExpert < 0 || rhsExpert < 0 {
            return lhsExpert < rhsExpert
        }
        let lhsCount = lhsExpert < expertUseCount.count ? expertUseCount[lhsExpert] : 0
        let rhsCount = rhsExpert < expertUseCount.count ? expertUseCount[rhsExpert] : 0
        if lhsCount != rhsCount { return lhsCount < rhsCount }
        return slotLastUse[lhs] < slotLastUse[rhs]
    }

    private func expertAdviceRanges(experts: [Int]) -> [(offset: UInt64, count: UInt64)] {
        experts.compactMap { expert in
            let regionOffset = layout.expertOffset(layer: 0, expert: expert)
            guard regionOffset + layout.expertStride <= layout.streamSize else { return nil }
            return (layout.streamOffset + regionOffset, layout.expertStride)
        }
    }

    private func adviseRanges(_ ranges: [(offset: UInt64, count: UInt64)],
                              requested: Int) -> ExpertIOAdviceResult {
        let coalesced = Self.coalescedAdjacentAdviceRanges(ranges)
        var failed = 0
        var bytes: UInt64 = 0
        var maxCallNanos: UInt64 = 0
        for range in coalesced {
            let result = RDAdvice.call(fd: fd, offset: range.offset, byteCount: range.count)
            if !result.succeeded { failed += 1 }
            bytes &+= result.requestedBytes
            maxCallNanos = max(maxCallNanos, result.elapsedNanos)
        }
        return ExpertIOAdviceResult(
            requested: requested,
            failed: failed,
            calls: coalesced.count,
            bytes: bytes,
            maxCallNanos: maxCallNanos)
    }

    private func readFull(into destination: UnsafeMutableRawPointer,
                          fileOffset: UInt64,
                          count: Int) throws {
        var filled = 0
        while filled < count {
            let readCount = pread(
                fd,
                destination.advanced(by: filled),
                count - filled,
                off_t(fileOffset) + off_t(filled))
            if readCount < 0 {
                throw StreamerError.preadFailed(errno: errno)
            }
            if readCount == 0 {
                throw StreamerError.sizeMismatch(expected: UInt64(count), actual: UInt64(filled))
            }
            filled += readCount
        }
    }
}
