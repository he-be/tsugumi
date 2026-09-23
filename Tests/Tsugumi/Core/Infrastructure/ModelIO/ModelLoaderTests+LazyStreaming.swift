import Foundation
import Metal
import Testing

@testable import Tsugumi

extension ModelLoaderTests {
  @Test func touchingOneLayerOpensExactlyOneStreamer() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device,
      expecting: .gemma4Toy())
    #expect(model.openLayerFileCount() == 0)
    _ = try model.routedExpert(layer: 0, expert: 3)
    #expect(model.openLayerFileCount() == 1)
    // Touch layer 0 again — no new open.
    _ = try model.routedExpert(layer: 0, expert: 5)
    #expect(model.openLayerFileCount() == 1)
    // Touch layer 1 — second open.
    _ = try model.routedExpert(layer: 1, expert: 0)
    #expect(model.openLayerFileCount() == 2)
  }

  @Test func routedExpertBytesRoundTrip() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device,
      expecting: .gemma4Toy())
    let view = try model.routedExpert(layer: 1, expert: 4)
    let bufContents = view.buffer.contents()
    let b0 = bufContents.load(fromByteOffset: Int(view.offset), as: UInt8.self)
    let b1 = bufContents.load(fromByteOffset: Int(view.offset) + 1, as: UInt8.self)
    let b2 = bufContents.load(fromByteOffset: Int(view.offset) + 2, as: UInt8.self)
    let b3 = bufContents.load(fromByteOffset: Int(view.offset) + 3, as: UInt8.self)
    #expect(b0 == 1)  // layer 1
    #expect(b1 == 4)  // expert 4
    #expect(b2 == 0xC1)
    #expect(b3 == 0xC2)
  }

  @Test func tamperedLayerFileFailsOnFirstTouch() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device,
      expecting: .gemma4Toy())

    // Flip one byte inside layer_01.bin AFTER the manifest is written.
    let url = dir.appendingPathComponent("packed_experts/layer_01.bin")
    var data = try Data(contentsOf: url)
    data[100] ^= 0xFF
    try data.write(to: url)

    // Layer 0 still loads.
    _ = try model.routedExpert(layer: 0, expert: 0)
    // Layer 1 first touch fails with checksumMismatch.
    #expect {
      _ = try model.routedExpert(layer: 1, expert: 0)
    } throws: { error in
      if case ModelError.checksumMismatch(let f) = error {
        return f == "packed_experts/layer_01.bin"
      }
      return false
    }
  }

  /// A layer hashed ahead of its open (`beginVerifyingRoutedExpertLayerAhead`)
  /// opens from the descriptor that was hashed: same bytes, one open.
  @Test func verifiedAheadLayerOpensWithTheSameBytes() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device,
      expecting: .gemma4Toy())
    model.beginVerifyingRoutedExpertLayerAhead(layer: 1)
    // Asking twice is a no-op.
    model.beginVerifyingRoutedExpertLayerAhead(layer: 1)
    #expect(model.openLayerFileCount() == 0)
    let view = try model.routedExpert(layer: 1, expert: 4)
    #expect(model.openLayerFileCount() == 1)
    let bytes = view.buffer.contents().advanced(by: Int(view.offset))
    #expect(bytes.load(fromByteOffset: 0, as: UInt8.self) == 1)  // layer 1
    #expect(bytes.load(fromByteOffset: 1, as: UInt8.self) == 4)  // expert 4
    #expect(bytes.load(fromByteOffset: 2, as: UInt8.self) == 0xC1)
    #expect(model.telemetry.snapshot().layerVerifiedBytes > 0)
  }

  /// A verify-ahead that fails leaves nothing behind: the open repeats the
  /// check and throws the same error the plain open does.
  @Test func tamperedLayerFileVerifiedAheadStillFails() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device,
      expecting: .gemma4Toy())
    let url = dir.appendingPathComponent("packed_experts/layer_01.bin")
    var data = try Data(contentsOf: url)
    data[100] ^= 0xFF
    try data.write(to: url)

    model.beginVerifyingRoutedExpertLayerAhead(layer: 1)
    #expect {
      _ = try model.routedExpert(layer: 1, expert: 0)
    } throws: { error in
      if case ModelError.checksumMismatch(let f) = error {
        return f == "packed_experts/layer_01.bin"
      }
      return false
    }
  }
}
