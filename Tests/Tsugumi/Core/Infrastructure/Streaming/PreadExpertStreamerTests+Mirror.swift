import Darwin
import Foundation
import Metal
import Testing

@testable import Tsugumi

/// A copy of the layer file on a second SSD (`ExpertMirror`,
/// docs/m6-prefill/44). The mirror here is tagged differently from the
/// original, so a readback says which file an expert came from: only the
/// experts the assignment puts on the mirror may come from it, on both arms.
extension PreadExpertStreamerTests {

  static func mirrorTagByte(_ expert: Int) -> UInt8 { UInt8(0xB0 + expert) }

  static func writeMirrorLayer() throws -> URL {
    let total = Int(streamOffset) + Int(streamSize)
    var bytes = [UInt8](repeating: 0, count: total)
    for e in 0..<numExperts {
      let start = Int(streamOffset) + e * expertStride
      for i in 0..<expertStride { bytes[start + i] = mirrorTagByte(e) }
    }
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("pread-streamer-mirror-\(UUID().uuidString).bin")
    try Data(bytes).write(to: url)
    return url
  }

  @Test(arguments: [false, true])
  func mirrorServesOnlyItsAssignedExperts(useMmap: Bool) throws {
    let url = try Self.writeSyntheticLayer()
    let mirrorURL = try Self.writeMirrorLayer()
    defer {
      try? FileManager.default.removeItem(at: url)
      try? FileManager.default.removeItem(at: mirrorURL)
    }
    let mirrorFD = open(mirrorURL.path, O_RDONLY | O_CLOEXEC)
    #expect(mirrorFD >= 0)
    defer { close(mirrorFD) }
    let onMirror = [false, true, false, true]
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path),
      device: try MetalContext().device,
      slotCount: Self.numExperts,
      fileDescriptor: nil,
      useMmap: useMmap,
      mirror: ExpertMirror(fileDescriptor: mirrorFD, onMirror: onMirror))

    let experts = [0, 1, 2, 3]
    let plan = streamer.planExpertsCached(experts: experts)
    let buffers = try streamer.executeExpertCachePlan(plan)
    for (index, expert) in experts.enumerated() {
      let got = Self.bytes(of: buffers[index].buffer, offset: buffers[index].offset,
                           count: Self.expertStride)
      let want = onMirror[expert] ? Self.mirrorTagByte(expert) : Self.tagByte(expert)
      #expect(got.allSatisfy { $0 == want }, "expert \(expert) came from the wrong file")
    }
    // Advice goes to both files; every range is still requested once.
    let advice = streamer.adviseExperts(experts: experts)
    #expect(advice.failed == 0)
    #expect(advice.bytes == UInt64(experts.count * Self.expertStride))
  }

  @Test func mirrorAssignmentMustCoverTheLayer() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let fd = open(url.path, O_RDONLY | O_CLOEXEC)
    defer { close(fd) }
    #expect(throws: (any Error).self) {
      _ = try PreadExpertStreamer(
        layout: Self.makeLayout(path: url.path),
        device: try MetalContext().device,
        slotCount: 2,
        fileDescriptor: nil,
        mirror: ExpertMirror(fileDescriptor: fd, onMirror: [true]))
    }
  }

  @Test func evenAssignmentSpreadsTheShare() {
    #expect(ExpertMirror.evenAssignment(count: 8, share: 0).allSatisfy { !$0 })
    #expect(ExpertMirror.evenAssignment(count: 8, share: 1).allSatisfy { $0 })
    #expect(ExpertMirror.evenAssignment(count: 4, share: 0.5) == [false, true, false, true])
    let assigned = ExpertMirror.evenAssignment(count: 128, share: 0.32)
    #expect(assigned.filter { $0 }.count == 40)
    // No run of mirror experts longer than one: neither drive gets a block.
    #expect(!zip(assigned, assigned.dropFirst()).contains { $0 && $1 })
  }
}
