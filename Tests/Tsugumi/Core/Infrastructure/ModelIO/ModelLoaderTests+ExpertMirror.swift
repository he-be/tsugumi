import Foundation
import Metal
import Testing

@testable import Tsugumi

/// `expert-mirror.json` (docs/m6-prefill/44). The copy is damaged on purpose
/// so a test can tell which file a byte came from: with share 0.5 the odd
/// experts are read from the mirror and the even ones from the model directory.
extension ModelLoaderTests {

  static let toyExpertStride: UInt64 = 16384

  /// Copy the toy's layer files to a mirror directory and register it.
  static func writeExpertMirror(for dir: URL, share: Double = 0.5,
                                manifestSha256: String? = nil) throws -> URL {
    let mirror = FileManager.default.temporaryDirectory
      .appendingPathComponent("expert-mirror-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: mirror, withIntermediateDirectories: true)
    let packed = dir.appendingPathComponent("packed_experts")
    for name in try FileManager.default.contentsOfDirectory(atPath: packed.path)
    where name.hasPrefix("layer_") {
      try FileManager.default.copyItem(at: packed.appendingPathComponent(name),
                                       to: mirror.appendingPathComponent(name))
    }
    let sha = try manifestSha256
      ?? Sha256Verifier.hashFile(at: dir.appendingPathComponent("manifest.json"))
    let receipt = ExpertMirrorReceipt(directory: mirror.path, share: share, manifestSha256: sha)
    try JSONEncoder().encode(receipt)
      .write(to: dir.appendingPathComponent(ExpertMirrorReceipt.fileName))
    return mirror
  }

  static func firstByte(_ view: TensorView) -> UInt8 {
    view.buffer.contents().load(fromByteOffset: Int(view.offset), as: UInt8.self)
  }

  @Test func mirrorIsReadForItsExpertsOnly() throws {
    let dir = try Self.writeToySynthetic()
    let mirror = try Self.writeExpertMirror(for: dir)
    defer {
      try? FileManager.default.removeItem(at: dir)
      try? FileManager.default.removeItem(at: mirror)
    }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let original = dir.appendingPathComponent("packed_experts/layer_00.bin")
    let copy = mirror.appendingPathComponent("layer_00.bin")
    let originalBytes = try Data(contentsOf: original)
    // Damage expert 1 in the copy and expert 0 in the original: expert 1 is
    // read from the copy, expert 0 from the original.
    try Self.flipByte(in: copy, at: 1 * Self.toyExpertStride)
    try Self.flipByte(in: original, at: 0)

    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy(),
                               integrityPolicy: .sizeCheckTrustedReceipt)
    #expect(model.expertMirror?.directory == mirror.path)
    #expect(Self.firstByte(try model.routedExpert(layer: 0, expert: 1))
            == originalBytes[Int(Self.toyExpertStride)] ^ 0xFF)
    #expect(Self.firstByte(try model.routedExpert(layer: 0, expert: 0))
            == originalBytes[0] ^ 0xFF)
  }

  /// Under full-sha256 the hash covers the bytes each expert is read from, so
  /// damage the runtime would read fails the load and damage it would not
  /// read (the copy's even experts) does not.
  @Test func fullSha256HashesTheBytesTheMirrorServes() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    do {
      let dir = try Self.writeToySynthetic()
      let mirror = try Self.writeExpertMirror(for: dir)
      defer {
        try? FileManager.default.removeItem(at: dir)
        try? FileManager.default.removeItem(at: mirror)
      }
      try Self.flipByte(in: mirror.appendingPathComponent("layer_00.bin"),
                        at: 1 * Self.toyExpertStride + 7)
      let model = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
      #expect {
        _ = try model.routedExpert(layer: 0, expert: 1)
      } throws: { error in
        if case ModelError.checksumMismatch = error { return true }
        return false
      }
    }
    do {
      let dir = try Self.writeToySynthetic()
      let mirror = try Self.writeExpertMirror(for: dir)
      defer {
        try? FileManager.default.removeItem(at: dir)
        try? FileManager.default.removeItem(at: mirror)
      }
      try Self.flipByte(in: mirror.appendingPathComponent("layer_00.bin"), at: 7)
      let model = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
      _ = try model.routedExpert(layer: 0, expert: 0)
      _ = try model.routedExpert(layer: 0, expert: 1)
    }
  }

  @Test func mirrorForAnotherManifestOrAMissingVolumeIsSkipped() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }

    let stale = try Self.writeExpertMirror(for: dir, manifestSha256: String(repeating: "0", count: 64))
    defer { try? FileManager.default.removeItem(at: stale) }
    #expect(try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
      .expertMirror == nil)

    let gone = try Self.writeExpertMirror(for: dir)
    try FileManager.default.removeItem(at: gone)
    let model = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
    #expect(model.expertMirror == nil)
    _ = try model.routedExpert(layer: 0, expert: 1)
  }

  @Test func malformedMirrorReceiptFailsTheLoad() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Data("{\"directory\":\"relative\",\"share\":2,\"manifestSha256\":\"x\"}".utf8)
      .write(to: dir.appendingPathComponent(ExpertMirrorReceipt.fileName))
    #expect(throws: ModelError.self) {
      _ = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
    }
  }

  @Test func segmentHashEqualsTheWholeFileHash() throws {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("segments-\(UUID().uuidString).bin")
    defer { try? FileManager.default.removeItem(at: url) }
    let bytes = (0..<100_000).map { UInt8(truncatingIfNeeded: $0 &* 31) }
    try Data(bytes).write(to: url)
    let fd = open(url.path, O_RDONLY)
    defer { close(fd) }
    let whole = try Sha256Verifier.hashFile(at: url)
    let split = try Sha256Verifier.hashSegments(
      [(fd, 0, 12_345), (fd, 12_345, 50_000), (fd, 62_345, 37_655)],
      displayName: "segments", pieceBytes: 4096, windowPieces: 3)
    #expect(split == whole)
  }
}
