import Foundation
import Metal
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareFormat

/// `docs/qwen35moe/04-PHASES.md` 次の一手 #11: the runtime accepts an install
/// whose attention weights are not all the same width, and still rejects the
/// ways that install could be wrong.
@Suite struct QwenRuntimeSchemaTests {

    private func loadToy(_ options: QwenToyInstall.Options = .init()) throws -> Model {
        let directory = try QwenToyInstall.write(options: options)
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try Model.load(directoryURL: directory,
                              device: device,
                              expecting: QwenToyInstall.arch)
    }

    @Test func opensAnInstallWhoseAttentionMixesWidths() throws {
        let model = try loadToy()
        // One role, two widths, decided per layer — the shape of the real
        // `oQ4e-g64` (13-PHASE1-REPACK §4-2) rather than a uniform stand-in.
        #expect(model.residentWeightBits(
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight") == 4)
        #expect(model.residentWeightBits(
            "language_model.model.layers.1.linear_attn.in_proj_qkv.weight") == 8)
        #expect(model.residentWeightBits(
            "language_model.model.layers.3.self_attn.q_proj.weight") == 8)
        #expect(model.residentWeightBits(
            "language_model.model.layers.3.self_attn.k_proj.weight") == 4)
        // The slot says 8 and the index says otherwise for most of them, so a
        // reader that trusted the slot alone would size five tensors wrong.
        #expect(model.manifest.quant?.attention.weightBits == 8)
    }

    @Test func rejectsATensorWiderThanItsSlotDeclares() throws {
        // Same bytes on disk, a manifest that promises 4-bit attention. The
        // slot is a bound, not decoration: an 8-bit tensor under it is an error.
        #expect(throws: ModelError.self) {
            try loadToy(.init(attentionSlotBits: 4))
        }
    }

    @Test func rejectsAGemmaManifestThatWidensTheEmbedding() throws {
        // The width gate is per family. Gemma 4's embedding is read by an
        // INT4-only lookup kernel, so 8 bits there must stay a load failure
        // even though the same number is legal for Qwen.
        let directory = try ModelLoaderTests.writeToySynthetic()
        let url = directory.appendingPathComponent("manifest.json")
        var root = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
            as! [String: Any]
        var quant = root["quant"] as! [String: Any]
        var embedding = quant["embedding"] as! [String: Any]
        embedding["weightBits"] = 8
        quant["embedding"] = embedding
        root["quant"] = quant
        let widened = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
        #expect(throws: ModelError.self) {
            try ManifestReader.decode(data: widened, expecting: .gemma4Toy())
        }
        // The same slot value on the Qwen toy is what its checkpoint really is.
        let qwenDirectory = try QwenToyInstall.write()
        let qwenManifest = try Data(
            contentsOf: qwenDirectory.appendingPathComponent("manifest.json"))
        let decoded = try ManifestReader.decode(data: qwenManifest,
                                                expecting: QwenToyInstall.arch)
        #expect(decoded.quant?.embedding.weightBits == 8)
        #expect(decoded.arch.family == "qwen3_5_moe")
    }

    @Test func rejectsAMissingRecurrentTensor() throws {
        let model = try loadToy()
        for name in ["language_model.model.layers.0.linear_attn.conv1d.weight",
                     "language_model.model.layers.0.linear_attn.A_log",
                     "language_model.model.layers.0.linear_attn.in_proj_a.weight",
                     "language_model.lm_head.weight"] {
            var entries = model.residentIndex.entries
            #expect(entries.removeValue(forKey: name) != nil)
            #expect(throws: ModelError.self) {
                try Model.validateRuntimeSchema(
                    residentIndex: ResidentIndex(header: model.residentIndex.header,
                                                 entries: entries),
                    layout: model.packedExpertsLayout,
                    manifest: model.manifest,
                    config: QwenToyInstall.arch)
            }
        }
    }

    @Test func rejectsAConvolutionKernelWithItsAxesSwapped() throws {
        // `[channels, taps, 1]` and `[taps, channels, 1]` hold the same number
        // of bytes, so only the shape separates them — and reading the taps
        // along the wrong axis is exactly the failure the kernel-side negative
        // control scores at 0.83 relative error (17-PHASE2-KERNELS §2).
        let model = try loadToy()
        let name = "language_model.model.layers.0.linear_attn.conv1d.weight"
        let entry = try #require(model.residentIndex.entries[name])
        var entries = model.residentIndex.entries
        entries[name] = ResidentIndexEntry(
            name: entry.name, dtype: entry.dtype, fileOffset: entry.fileOffset,
            sizeBytes: entry.sizeBytes,
            shape: (entry.shape.1, entry.shape.0, entry.shape.2, entry.shape.3),
            scaleOffset: entry.scaleOffset, scaleSize: entry.scaleSize,
            biasOffset: entry.biasOffset, biasSize: entry.biasSize)
        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: ResidentIndex(header: model.residentIndex.header,
                                             entries: entries),
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: QwenToyInstall.arch)
        }
    }

    @Test func rejectsAQueryProjectionWithoutItsOutputGate() throws {
        // `q_proj` is twice the query width; the second half is the gate. A
        // single-width one would bind and run, with every head's gate read out
        // of the next tensor.
        let model = try loadToy()
        let name = "language_model.model.layers.3.self_attn.q_proj.weight"
        let entry = try #require(model.residentIndex.entries[name])
        var entries = model.residentIndex.entries
        entries[name] = ResidentIndexEntry(
            name: entry.name, dtype: entry.dtype, fileOffset: entry.fileOffset,
            sizeBytes: entry.sizeBytes / 2,
            shape: (entry.shape.0 / 2, entry.shape.1, 0, 0),
            scaleOffset: entry.scaleOffset, scaleSize: entry.scaleSize / 2,
            biasOffset: entry.biasOffset, biasSize: entry.biasSize / 2)
        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: ResidentIndex(header: model.residentIndex.header,
                                             entries: entries),
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: QwenToyInstall.arch)
        }
    }

    @Test func rejectsAPackedByteCountThatIsNeitherWidth() throws {
        // Between the two widths there is nothing. A count that matches neither
        // is a corrupt index, not a third quantization.
        let model = try loadToy()
        let name = "language_model.model.layers.1.linear_attn.in_proj_qkv.weight"
        let entry = try #require(model.residentIndex.entries[name])
        var entries = model.residentIndex.entries
        entries[name] = ResidentIndexEntry(
            name: entry.name, dtype: entry.dtype, fileOffset: entry.fileOffset,
            sizeBytes: entry.sizeBytes * 3 / 4, shape: entry.shape,
            scaleOffset: entry.scaleOffset, scaleSize: entry.scaleSize,
            biasOffset: entry.biasOffset, biasSize: entry.biasSize)
        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: ResidentIndex(header: model.residentIndex.header,
                                             entries: entries),
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: QwenToyInstall.arch)
        }
    }

    @Test func rejectsALinearLayerCountThatDisagreesWithTheMask() throws {
        // `layerCount` is what a reader budgets recurrent state from without
        // walking `layerKinds`.
        let directory = try QwenToyInstall.write()
        let url = directory.appendingPathComponent("manifest.json")
        var root = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
            as! [String: Any]
        var arch = root["arch"] as! [String: Any]
        var linear = arch["linearAttention"] as! [String: Any]
        linear["layerCount"] = 2
        arch["linearAttention"] = linear
        root["arch"] = arch
        let data = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
        // The codec catches this one before the runtime schema does — the count
        // is checked against `layerKinds` at decode.
        #expect(throws: ModelError.self) {
            try ManifestReader.decode(data: data, expecting: QwenToyInstall.arch)
        }
    }
}
