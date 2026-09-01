import Darwin
import Foundation
import Metal
import Testing

@testable import TurboFieldfare

/// The mmap arm (`docs/mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md` §8) — the path CLI
/// and server take by default since 2026-08-20.
///
/// The tests above build streamers with `useMmap: false`, which is the arm the
/// product no longer uses. What is specific here is that **a miss copies
/// nothing**: the buffer a plan hands back is a `bytesNoCopy` window onto the
/// layer file, so the byte checks below are also checking the offset arithmetic
/// that slices one mapping into per-expert buffers.
extension PreadExpertStreamerTests {

  static func makeMmapStreamer(path: String, slotCount: Int) throws -> PreadExpertStreamer {
    try PreadExpertStreamer(
      layout: makeLayout(path: path),
      device: try MetalContext().device,
      slotCount: slotCount,
      fileDescriptor: nil,
      useMmap: true)
  }

  @Test func mmapArmServesTaggedBytesWithoutCopying() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    // Two slots for four experts: the second plan has to evict, which on this
    // arm means updating the residency set rather than overwriting a slot.
    let streamer = try Self.makeMmapStreamer(path: url.path, slotCount: 2)

    var seen: [Int: MTLBuffer] = [:]
    for experts in [[0, 1], [2, 3], [0, 1]] {
      let plan = streamer.planExpertsCached(experts: experts)
      let buffers = try streamer.executeExpertCachePlan(plan)
      #expect(buffers.count == experts.count)
      for (index, expert) in experts.enumerated() {
        let got = Self.bytes(of: buffers[index].buffer, offset: 0, count: Self.expertStride)
        #expect(got.allSatisfy { $0 == Self.tagByte(expert) },
                "expert \(expert) is not the window this plan returned")
        #expect(buffers[index].offset == 0)
        #expect(buffers[index].size == UInt64(Self.expertStride))
        // One buffer per expert for the life of the mapping. A slot cache that
        // copied would hand back whichever slot won this round instead.
        if let first = seen[expert] {
          #expect(first === buffers[index].buffer)
        } else {
          seen[expert] = buffers[index].buffer
        }
      }
    }
    #expect(seen.count == 4)
  }

  /// The slot bookkeeping is the same on both arms — only what a miss *does*
  /// changes (52 §8). A plan that repeats a resident expert still hits.
  @Test func mmapArmKeepsSlotBookkeeping() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try Self.makeMmapStreamer(path: url.path, slotCount: 4)

    let cold = streamer.planExpertsCached(experts: [1, 3])
    #expect(cold.hits == 0)
    #expect(cold.misses.count == 2)
    _ = try streamer.executeExpertCachePlan(cold)

    let warm = streamer.planExpertsCached(experts: [1, 3])
    #expect(warm.hits == 2)
    #expect(warm.misses.isEmpty)

    let mixed = streamer.planExpertsCached(experts: [1, 0])
    #expect(mixed.hits == 1)
    #expect(mixed.misses == [1])
  }

  /// 48 §1: the per-expert `bytesNoCopy` window is only correct while the
  /// stride lands on page boundaries (the shipping stride is 3,358,720 = 205
  /// pages). A format that does not is refused rather than silently handing a
  /// buffer that laps into the next expert.
  @Test func mmapArmRefusesStrideOffThePageBoundary() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let unaligned = StreamLayout(
      path: url.path,
      streamOffset: Self.streamOffset,
      streamSize: UInt64(Self.numExperts * (Self.pageSize + 1)),
      expertsPerLayer: Self.numExperts,
      expertStride: UInt64(Self.pageSize + 1))

    #expect(throws: StreamerError.self) {
      _ = try PreadExpertStreamer(layout: unaligned, device: device, slotCount: 2,
                                  fileDescriptor: nil, useMmap: true)
    }
    // The same layout is fine on the pread arm: it copies, so it never slices
    // the mapping. This is what makes `TF_EXPERT_MMAP=0` a complete way out.
    _ = try PreadExpertStreamer(layout: unaligned, device: device, slotCount: 2,
                                fileDescriptor: nil, useMmap: false)
  }
}
