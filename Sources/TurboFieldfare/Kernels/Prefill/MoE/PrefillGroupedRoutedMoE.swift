import Foundation
import Metal
import Darwin

enum PrefillGroupedRoutedMoEBufferIndex {
    static let hidden = 0
    static let sortedPairs = 1
    static let blocks = 2
    static let routePartials = 5
    static let gateUpActScratch = 7
    static let downScratch = 8
    static let expertArgumentState = 9
    static let params = 10
    static let mode = 11
}

/// One 64-row slice of one expert's route pairs, as the tiled GEMM sees it.
/// Matches `PrefillRoutedGEMMBlockMSL`.
struct PrefillRoutedGEMMBlock: Equatable, Sendable {
    var localSlot: UInt32
    var pairStart: UInt32
    var rowCount: UInt32
    var localRow: UInt32
}

/// One dispatch's worth of blocks. `rows` is the batch's row count, which is
/// both the stride between the gate/up/act thirds of the scratch and the row
/// capacity the scratch must have.
struct PrefillRoutedGEMMBatch: Equatable, Sendable {
    var rows: Int
    var blocks: [PrefillRoutedGEMMBlock]
}

/// Cuts a tile's expert groups into batches of blocks for the tiled GEMM.
///
/// Two rules, both about keeping the 64-row weight reuse intact:
/// batches start and end on group boundaries whenever a group fits, so a
/// batch boundary never splits an expert into two partial blocks; and a group
/// larger than the row budget is cut on a multiple of 64 so only its own tail
/// block is partial.
enum PrefillRoutedGEMMPlanner {
    static let tileRows = 64

    static func plan(groups: [PrefillMoEGroup], maxRowsPerBatch: Int) -> [PrefillRoutedGEMMBatch] {
        let budget = max(tileRows, (maxRowsPerBatch / tileRows) * tileRows)
        var batches: [PrefillRoutedGEMMBatch] = []
        var blocks: [PrefillRoutedGEMMBlock] = []
        var rows = 0

        func flush() {
            guard rows > 0 else { return }
            batches.append(PrefillRoutedGEMMBatch(rows: rows, blocks: blocks))
            blocks.removeAll(keepingCapacity: true)
            rows = 0
        }

        for (slot, group) in groups.enumerated() {
            var taken = 0
            let count = Int(group.pairCount)
            while taken < count {
                if rows == budget { flush() }
                // A group that fits in an empty batch is never split.
                if rows > 0, rows + (count - taken) > budget { flush() }
                let take = min(count - taken, budget - rows)
                var done = 0
                while done < take {
                    let rowCount = min(tileRows, take - done)
                    blocks.append(PrefillRoutedGEMMBlock(
                        localSlot: UInt32(slot),
                        pairStart: group.pairStart + UInt32(taken + done),
                        rowCount: UInt32(rowCount),
                        localRow: UInt32(rows + done)))
                    done += rowCount
                }
                rows += take
                taken += take
            }
        }
        flush()
        return batches
    }
}

struct PrefillGroupedRoutedMoEStreamedMetadataBuffers {
    let sortedPairs: MTLBuffer
}

struct PrefillStreamedTileArgumentBuffer {
    let buffer: MTLBuffer
}

public struct PrefillStreamedTileFetchResult {
    public let expertIDs: [Int]
    public let binding: PrefillStreamedTileBinding
    public let usedPlannedFetch: Bool
    public let plannedHits: Int
    public let plannedMissIndices: [Int]
    public let plannedAssignedSlots: [Int]
    public let plannedMissSlots: [Int]

    public init(expertIDs: [Int],
                binding: PrefillStreamedTileBinding,
                usedPlannedFetch: Bool,
                plannedHits: Int,
                plannedMissIndices: [Int],
                plannedAssignedSlots: [Int],
                plannedMissSlots: [Int]) {
        self.expertIDs = expertIDs
        self.binding = binding
        self.usedPlannedFetch = usedPlannedFetch
        self.plannedHits = plannedHits
        self.plannedMissIndices = plannedMissIndices
        self.plannedAssignedSlots = plannedAssignedSlots
        self.plannedMissSlots = plannedMissSlots
    }
}

enum PrefillStreamedTileLifetimeError: Error, Equatable, CustomStringConvertible {
    case duplicateSlots(tileIndex: Int, slots: [Int])
    case slotReuseBeforeCompletion(tileIndex: Int, conflictingTileIndex: Int, slots: [Int])
    case completeWithoutInFlightTile(tileIndex: Int)

    public var description: String {
        switch self {
        case .duplicateSlots(let tileIndex, let slots):
            return "prefill streamed tile \(tileIndex) has duplicate planned slots \(slots)"
        case .slotReuseBeforeCompletion(let tileIndex, let conflictingTileIndex, let slots):
            return "prefill streamed tile \(tileIndex) would reuse planned slots \(slots) while tile \(conflictingTileIndex) is in flight"
        case .completeWithoutInFlightTile(let tileIndex):
            return "prefill streamed tile \(tileIndex) completed without a matching in-flight tile"
        }
    }
}

struct PrefillStreamedTileSlotLifetime: Sendable, Equatable {
    private var inFlightSlotsByTile: [Int: Set<Int>] = [:]

    init() {}

    mutating func begin(tileIndex: Int, plannedSlots: [Int]) throws {
        let slots = try normalizedSlots(tileIndex: tileIndex, plannedSlots: plannedSlots)
        for (otherTile, otherSlots) in inFlightSlotsByTile {
            let overlap = slots.intersection(otherSlots)
            if !overlap.isEmpty {
                throw PrefillStreamedTileLifetimeError.slotReuseBeforeCompletion(
                    tileIndex: tileIndex,
                    conflictingTileIndex: otherTile,
                    slots: overlap.sorted())
            }
        }
        inFlightSlotsByTile[tileIndex] = slots
    }

    mutating func complete(tileIndex: Int) throws {
        guard inFlightSlotsByTile.removeValue(forKey: tileIndex) != nil else {
            throw PrefillStreamedTileLifetimeError.completeWithoutInFlightTile(tileIndex: tileIndex)
        }
    }

    private func normalizedSlots(tileIndex: Int, plannedSlots: [Int]) throws -> Set<Int> {
        var slots = Set<Int>()
        for slot in plannedSlots {
            guard slots.insert(slot).inserted else {
                throw PrefillStreamedTileLifetimeError.duplicateSlots(
                    tileIndex: tileIndex,
                    slots: plannedSlots.sorted())
            }
        }
        return slots
    }
}

struct PrefillGroupedRoutedMoEStreamedParams: Equatable, Sendable {
    var pairStart: UInt32
    var pairCount: UInt32
    var d: UInt32
    var routedIntermediate: UInt32
    var topK: UInt32
    var hiddenStrideElements: UInt32
    var liveExpertCount: UInt32
    var localExpert0: UInt32
    var localExpert1: UInt32
    var localExpert2: UInt32
    var localExpert3: UInt32
    var localExpert4: UInt32
    var localExpert5: UInt32
    var localExpert6: UInt32
    var localExpert7: UInt32
    var localExpert8: UInt32
    var localExpert9: UInt32
    var localExpert10: UInt32
    var localExpert11: UInt32
    var localExpert12: UInt32
    var localExpert13: UInt32
    var localExpert14: UInt32
    var localExpert15: UInt32
    var gateWOff: UInt32
    var gateSOff: UInt32
    var gateBOff: UInt32
    var upWOff: UInt32
    var upSOff: UInt32
    var upBOff: UInt32
    var downWOff: UInt32
    var downSOff: UInt32
    var downBOff: UInt32

    init(pairStart: UInt32,
                pairCount: UInt32,
                d: UInt32,
                routedIntermediate: UInt32,
                topK: UInt32,
                hiddenStrideElements: UInt32,
                binding: PrefillStreamedTileBinding,
                offsets: MoEExpertOffsets) {
        var ids = Array(repeating: UInt32.max, count: 16)
        for (index, expert) in binding.expertIDs.enumerated() {
            ids[index] = UInt32(expert)
        }
        self.pairStart = pairStart
        self.pairCount = pairCount
        self.d = d
        self.routedIntermediate = routedIntermediate
        self.topK = topK
        self.hiddenStrideElements = hiddenStrideElements
        self.liveExpertCount = UInt32(binding.expertIDs.count)
        self.localExpert0 = ids[0]
        self.localExpert1 = ids[1]
        self.localExpert2 = ids[2]
        self.localExpert3 = ids[3]
        self.localExpert4 = ids[4]
        self.localExpert5 = ids[5]
        self.localExpert6 = ids[6]
        self.localExpert7 = ids[7]
        self.localExpert8 = ids[8]
        self.localExpert9 = ids[9]
        self.localExpert10 = ids[10]
        self.localExpert11 = ids[11]
        self.localExpert12 = ids[12]
        self.localExpert13 = ids[13]
        self.localExpert14 = ids[14]
        self.localExpert15 = ids[15]
        self.gateWOff = offsets.gateWOff
        self.gateSOff = offsets.gateSOff
        self.gateBOff = offsets.gateBOff
        self.upWOff = offsets.upWOff
        self.upSOff = offsets.upSOff
        self.upBOff = offsets.upBOff
        self.downWOff = offsets.downWOff
        self.downSOff = offsets.downSOff
        self.downBOff = offsets.downBOff
    }
}

public struct PrefillStreamedTileBinding: Sendable, Equatable {
    public let expertIDs: [Int]
    public let views: [TensorView]

    public init(expertIDs: [Int], views: [TensorView]) throws {
        guard !expertIDs.isEmpty else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding("tile binding must include at least one expert")
        }
        guard expertIDs.count <= 16 else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile binding has \(expertIDs.count) experts; maximum is 16")
        }
        guard expertIDs.count == views.count else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "expertIDs.count \(expertIDs.count) != views.count \(views.count)")
        }
        var seen = Set<Int>()
        for expert in expertIDs {
            guard expert >= 0 else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "expert id \(expert) must be non-negative")
            }
            guard seen.insert(expert).inserted else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "duplicate expert id \(expert) in tile binding")
            }
        }
        self.expertIDs = expertIDs
        self.views = views
    }

    public func localSlot(for expert: UInt32) -> Int? {
        expertIDs.firstIndex(of: Int(expert))
    }

    public static func expertIDs(forTile tileIndex: Int,
                                 routes: PrefillMoEGroupedRoutes) throws -> [Int] {
        guard routes.tiles.indices.contains(tileIndex) else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile index \(tileIndex) is out of range")
        }
        let tile = routes.tiles[tileIndex]
        let groupStart = Int(tile.groupStart)
        let groupCount = Int(tile.groupCount)
        guard groupCount > 0, groupCount <= 16 else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile has \(groupCount) live experts; expected 1...16")
        }
        guard groupStart >= 0, groupStart + groupCount <= routes.groups.count else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile group range \(groupStart)..<\(groupStart + groupCount) exceeds \(routes.groups.count)")
        }
        return routes.groups[groupStart..<(groupStart + groupCount)].map { Int($0.expert) }
    }

    public static func fetchBindingForTile(model: Model,
                                           layer: Int,
                                           tileIndex: Int,
                                           routes: PrefillMoEGroupedRoutes,
                                           plannedFetch: RoutedExpertFetchPlan? = nil,
                                           avoidingSlots: Set<Int> = []) async throws
        -> PrefillStreamedTileFetchResult {
        let expertIDs = try expertIDs(forTile: tileIndex, routes: routes)
        let plan = try plannedFetch ?? model.planRoutedExperts(layer: layer,
                                                               experts: expertIDs,
                                                               avoidingSlots: avoidingSlots)
        let views: [TensorView]
        let usedPlannedFetch: Bool
        let plannedHits: Int
        let plannedMissIndices: [Int]
        let plannedAssignedSlots: [Int]
        let plannedMissSlots: [Int]
        if let plan {
            guard plan.layer == layer, plan.experts == expertIDs else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "preplanned fetch does not match tile \(tileIndex)")
            }
            views = try await model.fetchRoutedExperts(plan: plan)
            usedPlannedFetch = true
            plannedHits = plan.hits
            plannedMissIndices = plan.misses
            plannedAssignedSlots = plan.assignedSlots
            plannedMissSlots = plan.misses.map { plan.assignedSlots[$0] }
        } else {
            views = try await model.fetchRoutedExperts(layer: layer, experts: expertIDs)
            usedPlannedFetch = false
            plannedHits = 0
            plannedMissIndices = []
            plannedAssignedSlots = []
            plannedMissSlots = []
        }
        let binding = try PrefillStreamedTileBinding(expertIDs: expertIDs, views: views)
        return PrefillStreamedTileFetchResult(expertIDs: expertIDs,
                                             binding: binding,
                                             usedPlannedFetch: usedPlannedFetch,
                                             plannedHits: plannedHits,
                                             plannedMissIndices: plannedMissIndices,
                                             plannedAssignedSlots: plannedAssignedSlots,
                                             plannedMissSlots: plannedMissSlots)
    }

    public func validateCoversPairs(_ pairs: [PrefillTokenExpertPair],
                                    pairStart: Int,
                                    pairCount: Int) throws {
        guard pairStart >= 0, pairCount >= 0, pairStart + pairCount <= pairs.count else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "pair range \(pairStart)..<\(pairStart + pairCount) exceeds \(pairs.count)")
        }
        for pair in pairs[pairStart..<(pairStart + pairCount)] {
            guard localSlot(for: pair.expert) != nil else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "route expert \(pair.expert) is not bound in tile")
            }
        }
    }

    public static func == (lhs: PrefillStreamedTileBinding,
                           rhs: PrefillStreamedTileBinding) -> Bool {
        guard lhs.expertIDs == rhs.expertIDs, lhs.views.count == rhs.views.count else {
            return false
        }
        for index in lhs.views.indices {
            let l = lhs.views[index]
            let r = rhs.views[index]
            guard l.buffer === r.buffer,
                  l.offset == r.offset,
                  l.length == r.length,
                  l.scaleOffset == r.scaleOffset,
                  l.scaleLength == r.scaleLength,
                  l.biasOffset == r.biasOffset,
                  l.biasLength == r.biasLength,
                  l.shape == r.shape,
                  l.dtype == r.dtype else {
                return false
            }
        }
        return true
    }
}

enum PrefillGroupedRoutedMoEError: Error, Equatable, CustomStringConvertible {
    case invalidStreamedTileBinding(String)
    case allocationFailed(String)

    public var description: String {
        switch self {
        case .invalidStreamedTileBinding(let reason):
            return "invalid streamed tile binding: \(reason)"
        case .allocationFailed(let label):
            return "failed to allocate \(label)"
        }
    }
}

final class PrefillGroupedRoutedMoE {
    /// Which kernel served a tile, so an A/B measurement can assert the path
    /// it meant to time actually ran.
    enum Path: String, Sendable {
        /// One thread per (pair, row), scalar K reduction, weights re-read
        /// once per 8 pairs.
        case perPairGEMV = "per-pair-gemv"
        /// 64x64 output tile per threadgroup, 8x8 `simdgroup_matrix` products,
        /// weights re-read once per 64 pairs of the same expert.
        case expertGEMM = "expert-gemm"
    }

    static let gemmTileM = 64
    static let gemmTileN = 64
    static let gemmTileK = 32
    static let gemmThreadsPerGroup = 128

    /// `TF_PREFILL_MOE=scalar` forces the per-pair GEMV kernels, so the two
    /// paths can be measured against each other without a rebuild. It is also
    /// the way back to FP32 weight arithmetic: the tiled path stages its
    /// dequantized weights as FP16, one rounding per weight (see
    /// `PrefillInt4QMM`). Anything else, including unset, takes the tiled path.
    private static let forcedPath = ProcessInfo.processInfo.environment["TF_PREFILL_MOE"]

    private let affineGroupSize: Int
    private let batchedPhase1PSO: MTLComputePipelineState
    private let batchedDownPSO: MTLComputePipelineState
    private let gemmPSO: MTLComputePipelineState?
    private let geluMulPSO: MTLComputePipelineState?
    private let streamedArgEncoder: MTLArgumentEncoder

    func makeStreamedArgumentBuffer(device: MTLDevice,
                                           binding: PrefillStreamedTileBinding) throws -> PrefillStreamedTileArgumentBuffer {
        guard let buffer = device.makeBuffer(length: streamedArgEncoder.encodedLength,
                                             options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("prefill streamed expert argument buffer")
        }
        buffer.label = "prefill.groupedMoe.streamedArgumentBuffer"

        streamedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for index in binding.views.indices {
            let view = binding.views[index]
            streamedArgEncoder.setBuffer(view.buffer, offset: Int(view.offset), index: index)
        }

        return PrefillStreamedTileArgumentBuffer(buffer: buffer)
    }

    init(context: MetalContext) throws {
        self.affineGroupSize = context.affineGroupSize
        self.batchedPhase1PSO = try context.pipeline("prefill_grouped_routed_moe_batched_phase1")
        self.batchedDownPSO = try context.pipeline("prefill_grouped_routed_moe_batched_down")
        if Self.forcedPath == "scalar" {
            self.gemmPSO = nil
            self.geluMulPSO = nil
        } else {
            // The tile shape fixes the threadgroup at 128 threads; a build
            // where register pressure caps it lower cannot run this kernel.
            let candidate = try? context.pipeline("prefill_moe_gemm_int4")
            let usable = (candidate?.maxTotalThreadsPerThreadgroup ?? 0) >= Self.gemmThreadsPerGroup
            self.gemmPSO = usable ? candidate : nil
            self.geluMulPSO = usable ? try? context.pipeline("prefill_moe_gate_up_gelu_mul") : nil
        }
        guard let streamedFn = try context.library.makeFunction(name: "prefill_grouped_routed_moe_batched_phase1") else {
            throw MetalError.missingFunction("prefill_grouped_routed_moe_batched_phase1")
        }
        self.streamedArgEncoder = streamedFn.makeArgumentEncoder(
            bufferIndex: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
    }

    /// The tiled kernel walks K in 32-element steps and reads one scale/bias
    /// pair per 8 weights, so both reductions (`D` for gate/up, `F` for down)
    /// have to be aligned to the tile and to the affine group.
    func usesExpertGEMMPath(d: Int, f: Int) -> Bool {
        guard gemmPSO != nil, geluMulPSO != nil else { return false }
        for k in [d, f] where k % Self.gemmTileK != 0 || k % affineGroupSize != 0 {
            return false
        }
        return true
    }

    func makeStreamedMetadataBuffers(
        device: MTLDevice,
        routes: PrefillMoEGroupedRoutes
    ) throws -> PrefillGroupedRoutedMoEStreamedMetadataBuffers {
        let bytes = routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride
        guard let sortedPairs = routes.sortedPairs.withUnsafeBufferPointer({ ptr in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: bytes,
                              options: .storageModeShared)
        }) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("prefill sorted route pairs")
        }
        return PrefillGroupedRoutedMoEStreamedMetadataBuffers(sortedPairs: sortedPairs)
    }

    @discardableResult
    func encodeStreamedBatched(commandBuffer: MTLCommandBuffer,
                                      hidden: MTLBuffer,
                                      hiddenOffset: Int = 0,
                                      sortedPairs: MTLBuffer,
                                      sortedPairsOffset: Int = 0,
                                      routePartials: MTLBuffer,
                                      routePartialsOffset: Int = 0,
                                      gateUpActScratch: MTLBuffer,
                                      gateUpActScratchOffset: Int = 0,
                                      downScratch: MTLBuffer,
                                      downScratchOffset: Int = 0,
                                      argumentBuffer: PrefillStreamedTileArgumentBuffer,
                                      binding: PrefillStreamedTileBinding,
                                      params: PrefillGroupedRoutedMoEStreamedParams,
                                      pairMicrobatchRows: Int = 32) -> Int {
        guard params.pairCount > 0,
              params.liveExpertCount == UInt32(binding.views.count),
              pairMicrobatchRows > 0 else { return 0 }
        var consumed: UInt32 = 0
        var microbatchCount = 0
        while consumed < params.pairCount {
            var p = params
            p.pairStart = params.pairStart + consumed
            p.pairCount = min(UInt32(pairMicrobatchRows), params.pairCount - consumed)

            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(batchedPhase1PSO)
                enc.setBuffer(hidden, offset: hiddenOffset, index: PrefillGroupedRoutedMoEBufferIndex.hidden)
                enc.setBuffer(sortedPairs, offset: sortedPairsOffset, index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
                enc.setBuffer(gateUpActScratch, offset: gateUpActScratchOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.gateUpActScratch)
                enc.setBuffer(argumentBuffer.buffer, offset: 0,
                              index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
                enc.setBytes(&p,
                             length: MemoryLayout<PrefillGroupedRoutedMoEStreamedParams>.stride,
                             index: PrefillGroupedRoutedMoEBufferIndex.params)
                for view in binding.views {
                    enc.useResource(view.buffer, usage: .read)
                }
                enc.dispatchThreads(MTLSize(width: Int(p.routedIntermediate),
                                            height: Int(p.pairCount),
                                            depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                enc.endEncoding()
            }

            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(batchedDownPSO)
                enc.setBuffer(sortedPairs, offset: sortedPairsOffset, index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
                enc.setBuffer(routePartials, offset: routePartialsOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.routePartials)
                enc.setBuffer(gateUpActScratch, offset: gateUpActScratchOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.gateUpActScratch)
                enc.setBuffer(downScratch, offset: downScratchOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.downScratch)
                enc.setBuffer(argumentBuffer.buffer, offset: 0,
                              index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
                enc.setBytes(&p,
                             length: MemoryLayout<PrefillGroupedRoutedMoEStreamedParams>.stride,
                             index: PrefillGroupedRoutedMoEBufferIndex.params)
                for view in binding.views {
                    enc.useResource(view.buffer, usage: .read)
                }
                enc.dispatchThreads(MTLSize(width: Int(p.d),
                                            height: Int(p.pairCount),
                                            depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                enc.endEncoding()
            }

            consumed += p.pairCount
            microbatchCount += 1
        }
        return microbatchCount
    }

    /// Runs a tile as one GEMM per expert-row-block instead of one GEMV per
    /// (pair, row).
    ///
    /// `groups` are the tile's expert groups in binding order — group `i` is
    /// `binding.expertIDs[i]` — which is what makes the expert lookup a plain
    /// index instead of the per-thread search the per-pair kernels do.
    ///
    /// Three dispatches per batch: gate and up over the gathered activations
    /// (one dispatch, `z` picks the projection), the non-linearity, then down
    /// scattered into `routePartials`. `routePartials` is written for every
    /// pair exactly once, so the token-major reduction downstream is unchanged.
    @discardableResult
    func encodeStreamedTiled(commandBuffer: MTLCommandBuffer,
                             hidden: MTLBuffer,
                             hiddenOffset: Int = 0,
                             sortedPairs: MTLBuffer,
                             sortedPairsOffset: Int = 0,
                             routePartials: MTLBuffer,
                             routePartialsOffset: Int = 0,
                             gateUpActScratch: MTLBuffer,
                             gateUpActScratchOffset: Int = 0,
                             argumentBuffer: PrefillStreamedTileArgumentBuffer,
                             binding: PrefillStreamedTileBinding,
                             groups: [PrefillMoEGroup],
                             params: PrefillGroupedRoutedMoEStreamedParams,
                             maxRowsPerBatch: Int) -> Int {
        guard let gemmPSO, let geluMulPSO,
              params.liveExpertCount == UInt32(binding.views.count),
              groups.count == binding.views.count,
              maxRowsPerBatch > 0 else { return 0 }
        // The block list travels in a `setBytes` argument (4 KB), which caps
        // the rows a batch may describe well above any useful batch size.
        let batches = PrefillRoutedGEMMPlanner.plan(groups: groups,
                                                    maxRowsPerBatch: min(maxRowsPerBatch, 4096))
        let nTilesGateUp = (Int(params.routedIntermediate) + Self.gemmTileN - 1) / Self.gemmTileN
        let nTilesDown = (Int(params.d) + Self.gemmTileN - 1) / Self.gemmTileN
        let threadgroup = MTLSize(width: Self.gemmThreadsPerGroup, height: 1, depth: 1)

        for batch in batches {
            var p = params
            // The kernels read `pair_count` as the batch's row count: it is the
            // stride between the gate, up and act thirds of the scratch.
            p.pairStart = 0
            p.pairCount = UInt32(batch.rows)
            var blocks = batch.blocks
            let blockBytes = blocks.count * MemoryLayout<PrefillRoutedGEMMBlock>.stride

            func encodeGEMM(mode: UInt32, nTiles: Int, depth: Int) {
                guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
                enc.setComputePipelineState(gemmPSO)
                enc.setBuffer(hidden, offset: hiddenOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.hidden)
                enc.setBuffer(sortedPairs, offset: sortedPairsOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
                enc.setBytes(&blocks, length: blockBytes,
                             index: PrefillGroupedRoutedMoEBufferIndex.blocks)
                enc.setBuffer(routePartials, offset: routePartialsOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.routePartials)
                enc.setBuffer(gateUpActScratch, offset: gateUpActScratchOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.gateUpActScratch)
                enc.setBuffer(argumentBuffer.buffer, offset: 0,
                              index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
                enc.setBytes(&p,
                             length: MemoryLayout<PrefillGroupedRoutedMoEStreamedParams>.stride,
                             index: PrefillGroupedRoutedMoEBufferIndex.params)
                var modeVar = mode
                enc.setBytes(&modeVar, length: MemoryLayout<UInt32>.size,
                             index: PrefillGroupedRoutedMoEBufferIndex.mode)
                for view in binding.views {
                    enc.useResource(view.buffer, usage: .read)
                }
                enc.dispatchThreadgroups(MTLSize(width: nTiles,
                                                 height: blocks.count,
                                                 depth: depth),
                                         threadsPerThreadgroup: threadgroup)
                enc.endEncoding()
            }

            encodeGEMM(mode: 0, nTiles: nTilesGateUp, depth: 2)

            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(geluMulPSO)
                enc.setBuffer(gateUpActScratch, offset: gateUpActScratchOffset, index: 0)
                var rowsVar = UInt32(batch.rows)
                var fVar = params.routedIntermediate
                enc.setBytes(&rowsVar, length: MemoryLayout<UInt32>.size, index: 1)
                enc.setBytes(&fVar, length: MemoryLayout<UInt32>.size, index: 2)
                let elements = batch.rows * Int(params.routedIntermediate)
                let width = min(geluMulPSO.maxTotalThreadsPerThreadgroup, 256)
                enc.dispatchThreads(MTLSize(width: elements, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
                enc.endEncoding()
            }

            encodeGEMM(mode: 1, nTiles: nTilesDown, depth: 1)
        }
        return batches.count
    }
}
