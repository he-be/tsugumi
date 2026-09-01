import Darwin
import Foundation
import Metal
import Testing

@testable import Tsugumi

extension PreadExpertStreamerTests {
  /// A block plans a layer's whole expert union in one call and then hands each
  /// tile the slice that is its own (docs/mtp/27-M7-RESULTS.md §3). The slices
  /// have to add back up to the plan: same experts, same slots, and every miss
  /// in exactly one slice with an index that points at its own expert.
  @Test func planSlicesPartitionTheLayerPlan() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 6)

    _ = try streamer.loadExpertsCached(experts: [1, 3])  // two of the four resident
    let experts = [1, 3, 0, 2]
    let plan = RoutedExpertFetchPlan(
      layer: 0, cachePlan: streamer.planExpertsCached(experts: experts))
    #expect(plan.hits == 2)
    #expect(plan.misses.count == 2)

    let slices = [0..<2, 2..<3, 3..<4].map { plan.slice($0) }
    #expect(slices.flatMap(\.experts) == experts)
    #expect(slices.flatMap(\.assignedSlots) == plan.assignedSlots)
    #expect(slices.map(\.hits).reduce(0, +) == plan.hits)
    #expect(slices.map { $0.misses.count }.reduce(0, +) == plan.misses.count)
    for slice in slices {
      for index in slice.misses {
        #expect(index >= 0 && index < slice.experts.count)
      }
      #expect(Set(slice.misses).count == slice.misses.count)
    }
    // The first slice is the resident pair, so it reads nothing.
    #expect(slices[0].misses.isEmpty)

    // Executing the slices separately loads the same bytes the whole plan would.
    for slice in slices {
      let buffers = try streamer.executeExpertCachePlan(slice.cachePlan)
      for (index, buffer) in buffers.enumerated() {
        let got = Self.bytes(of: buffer.buffer, offset: 0, count: Self.expertStride)
        #expect(got.allSatisfy { $0 == Self.tagByte(slice.experts[index]) })
      }
    }
  }

  @Test func cachedBatchWithoutExecutorLoadsTaggedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    let results = try streamer.loadExpertsCached(experts: [3, 1, 2])
    for (index, result) in results.enumerated() {
      let expert = [3, 1, 2][index]
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
    }
  }

  @Test func adviseExpertsDoesNotChangeLoadedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)
    let experts = [0, 2, 3]

    let advice = streamer.adviseExperts(experts: experts)
    #expect(advice.requested == experts.count)
    #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
      #expect(advice.failed == 0)
    #else
      #expect(advice.failed == experts.count)
    #endif

    let results = try streamer.loadExpertsCached(experts: experts)
    for (index, result) in results.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[index]) })
    }
  }

  @Test func adviseExpertMissesSkipsResidentSlots() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let advice = streamer.adviseExpertMisses(experts: [0, 1, 2])

    #expect(advice.requested == 2)
    #expect(advice.calls == 1)
    #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
      #expect(advice.failed == 0)
    #else
      #expect(advice.failed == 1)
    #endif
  }

  @Test func plannedCacheLoadExecutesSameMisses() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let experts = [0, 1, 2]
    let plan = streamer.planExpertsCached(experts: experts)

    #expect(plan.hits == 1)
    #expect(plan.misses.map { experts[$0] } == [1, 2])

    let results = try streamer.executeExpertCachePlan(plan)
    for (index, result) in results.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[index]) })
    }
  }

  @Test func plannedCacheBuffersExposeReservedSlotsBeforeExecute() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let experts = [0, 1, 2]
    let plan = streamer.planExpertsCached(experts: experts)
    let reserved = streamer.expertCachePlanBuffers(plan)

    let hitBytes = Self.bytes(of: reserved[0].buffer, offset: 0, count: Self.expertStride)
    #expect(hitBytes.allSatisfy { $0 == Self.tagByte(0) })

    let executed = try streamer.executeExpertCachePlan(plan)
    for i in 0..<experts.count {
      #expect(reserved[i].buffer === executed[i].buffer)
      let got = Self.bytes(of: executed[i].buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[i]) })
    }
  }

  @Test func plannedCacheAvoidsInFlightSlotsForHitsAndMisses() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    let warmed = try streamer.loadExpertsCached(experts: [0, 1])
    let plan = streamer.planExpertsCached(
      experts: [0, 2],
      avoidingSlots: [0, 1])

    #expect(plan.assignedSlots == [0, 2])
    #expect(plan.hits == 1)
    #expect(plan.misses == [1])

    let executed = try streamer.executeExpertCachePlan(plan)
    for (index, expert) in plan.experts.enumerated() {
      let got = Self.bytes(of: executed[index].buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
    }

    let avoidedBytes = Self.bytes(of: warmed[0].buffer, offset: 0, count: Self.expertStride)
    #expect(avoidedBytes.allSatisfy { $0 == Self.tagByte(0) })
  }

  @Test func plannedCacheReturnsNilWhenMissesCannotAvoidInFlightSlots() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0, 1])
    let plan = streamer.planExpertsCachedIfPossible(
      experts: [0, 2, 3, 4],
      avoidingSlots: [0, 1])

    #expect(plan == nil)
  }

  /// A guess may not take a slot the layer used in its most recent plan
  /// (docs/mtp/29-M8-B-PROBE.md §6): a prefetch that evicts the round's own
  /// working set pays for its hit with a miss somewhere else. With every slot
  /// stamped by the last plan there is nowhere to put a guess, and giving it up
  /// is the answer — the layer will read what it needs when it gets there.
  @Test func speculativePlanDeclinesRatherThanEvictTheLastRound() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)

    _ = try streamer.loadExpertsCached(experts: [0, 1])

    #expect(streamer.planSpeculativeExperts(experts: [2]) == nil)
    // Nothing moved: the cache is exactly what the last plan left.
    #expect(streamer.residentExperts([0, 1, 2]) == [true, true, false])
    // A guess bigger than the cache, or no guess at all, is not a plan either.
    #expect(streamer.planSpeculativeExperts(experts: []) == nil)
    #expect(streamer.planSpeculativeExperts(experts: [0, 1, 2]) == nil)
  }

  /// The other half: a slot the last plan did not touch is fair game, and once
  /// the guess has been executed the layer's own plan sees a plain hit — which
  /// is the whole point, since the read happened a layer early.
  @Test func speculativePlanTakesAStaleSlotAndLandsAsAHit() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 3)

    // Two rounds of the same pair leaves the third slot untouched by the last.
    _ = try streamer.loadExpertsCached(experts: [0, 1])
    _ = try streamer.loadExpertsCached(experts: [0, 1])

    let guess = try #require(streamer.planSpeculativeExperts(experts: [2]))
    #expect(guess.misses == [0])
    let buffers = try streamer.executeExpertCachePlan(guess)
    let got = Self.bytes(of: buffers[0].buffer, offset: 0, count: Self.expertStride)
    #expect(got.allSatisfy { $0 == Self.tagByte(2) })

    let plan = streamer.planExpertsCached(experts: [0, 1, 2])
    #expect(plan.hits == 3)
    #expect(plan.misses.isEmpty)
  }

  /// A guess does not earn the expert it read any protection: the use count is
  /// what keeps a slot under LFU, and counting a guess would leave an expert
  /// nothing ever used holding a slot against experts that are being used.
  @Test func speculativePlanDoesNotCountTowardsEviction() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)

    // Expert 0 is asked for twice; the second round leaves the other slot stale.
    _ = try streamer.loadExpertsCached(experts: [0])
    _ = try streamer.loadExpertsCached(experts: [0])
    let guess = try #require(streamer.planSpeculativeExperts(experts: [1]))
    _ = try streamer.executeExpertCachePlan(guess)
    #expect(streamer.residentExperts([0, 1]) == [true, true])

    // One victim is needed. The guessed expert has never been used, so it is
    // the one that goes — not the expert two plans asked for.
    _ = try streamer.loadExpertsCached(experts: [2])
    #expect(streamer.residentExperts([0, 1, 2]) == [true, false, true])
  }

}
