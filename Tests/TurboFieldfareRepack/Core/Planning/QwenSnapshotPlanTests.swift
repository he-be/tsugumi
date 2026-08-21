import Foundation
import Testing
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

/// Planning a Qwen3.5-MoE checkpoint (`docs/qwen35moe/03-DESIGN.md` §1-2): the
/// routed experts are spelled `mlp.switch_mlp`, the MTP head ships inside the
/// text checkpoint with a `layers.0` of its own, and most layers hold a
/// recurrent state instead of a KV cache.
@Suite
struct QwenSnapshotPlanTests {

    /// Builds the snapshot and loads it the way `--source-snapshot` does. The
    /// snapshot is synthetic, so it identifies as unpinned.
    private func makeSnapshot(_ label: String) throws -> (String, LocalSnapshot) {
        let dir = NSTemporaryDirectory() + "qwen-\(label)-\(UInt32.random(in: 0...UInt32.max))"
        _ = try SyntheticQwenSnapshot.build(at: dir)
        return (dir, try LocalSnapshotLoader.load(directory: dir, requireKnownSource: false))
    }

    @Test func configParsesAsALinearAttentionFamily() throws {
        let (dir, snapshot) = try makeSnapshot("arch")
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let arch = snapshot.arch
        #expect(arch.family == ArchInfo.qwen35MoeFamily)
        #expect(!arch.isGemma4)
        #expect(arch.numLayers == 4)
        #expect(arch.layerKinds == ["linear_attention", "linear_attention",
                                    "linear_attention", "full_attention"])
        #expect(arch.fullAttentionLayerMask == [0, 0, 0, 1])
        // No layer slides, so there is no window and no softcap to name.
        #expect(arch.slidingWindow == 0)
        #expect(arch.finalLogitSoftcap == 0)
        #expect(arch.topKExperts == 2)
        // The shared expert is this family's dense FFN.
        #expect(arch.intermediateSize == 128)
        let linear = try #require(arch.linearAttention)
        #expect(linear.layerCount == 3)
        #expect(linear.numKeyHeads == 2)
        #expect(linear.numValueHeads == 4)
        #expect(linear.convKernelDim == 4)
    }

    @Test func gemmaConfigStillParsesAsGemma() throws {
        let dir = NSTemporaryDirectory() + "gemma-arch-\(UInt32.random(in: 0...UInt32.max))"
        defer { try? FileManager.default.removeItem(atPath: dir) }
        _ = try SyntheticSnapshot.build(at: dir)
        let arch = try ArchInfo.load(
            configPath: (dir as NSString).appendingPathComponent("config.json"))
        #expect(arch.isGemma4)
        #expect(arch.linearAttention == nil)
        #expect(arch.slidingWindow == 128)
    }

    @Test func switchMLPIsARoutedExpertAndMTPIsNot() {
        let body = "language_model.model.layers.2.mlp.switch_mlp.gate_proj.weight"
        #expect(RepackPlanner.classify(body, numLayers: 4)
            == .routedExpert(role: "gate", layer: 2))
        let gemma = "language_model.model.layers.1.experts.switch_glu.down_proj.weight"
        #expect(RepackPlanner.classify(gemma, numLayers: 4)
            == .routedExpert(role: "down", layer: 1))
        // The drafter's `layers.0` is not the body's `layers.0`.
        let draft = "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight"
        #expect(RepackPlanner.classify(draft, numLayers: 4) == .excludedDraft)
        #expect(RepackPlanner.classify("language_model.mtp.fc.weight", numLayers: 4)
            == .excludedDraft)
        #expect(RepackPlanner.classify("vision_tower.patch_embed.proj.weight", numLayers: 4)
            == .excludedMultimodal)
    }

    @Test func planSplitsExpertsFromResidentAndLeavesTheDrafterOut() throws {
        let (dir, snapshot) = try makeSnapshot("plan")
        let output = NSTemporaryDirectory() + "qwen-out-\(UInt32.random(in: 0...UInt32.max))"
        defer {
            try? FileManager.default.removeItem(atPath: dir)
            try? FileManager.default.removeItem(atPath: output)
        }
        let arch = snapshot.arch
        let plan = try RepackPlanner.plan(meta: snapshot.metadata, arch: arch,
                                          shardHeaders: snapshot.shardHeaders,
                                          outputDir: output)
        #expect(plan.layers.count == 4)
        for layer in plan.layers {
            #expect(layer.expertsPerLayer == arch.numExperts)
            // gate/up/down × weights/scales/biases.
            #expect(layer.subTensors.count == 9)
            #expect(layer.expertStride % GTurboFormatV1.alignmentBytes == 0)
        }
        let residentNames = Set(plan.resident.entries.map(\.name))
        #expect(residentNames.contains("language_model.model.layers.0.linear_attn.conv1d.weight"))
        #expect(residentNames.contains("language_model.model.layers.3.self_attn.q_norm.weight"))
        #expect(residentNames.contains("language_model.model.layers.0.mlp.gate.weight"))
        #expect(residentNames.contains(
            "language_model.model.layers.0.mlp.shared_expert.down_proj.weight"))
        // Nothing routed, drafted or seen leaks into the text resident file.
        #expect(!residentNames.contains(where: { $0.contains(".mlp.switch_mlp.") }))
        #expect(!residentNames.contains(where: { $0.hasPrefix("language_model.mtp.") }))
        #expect(!residentNames.contains(where: { $0.hasPrefix("vision_tower.") }))
        #expect(!plan.excludedInlineDraftTensorNames.isEmpty)
        #expect(plan.excludedInlineDraftTensorNames
            .allSatisfy { $0.hasPrefix("language_model.mtp.") })
        #expect(plan.excludedMultimodalTensorNames == ["vision_tower.patch_embed.proj.weight"])
    }

    @Test func manifestCarriesTheFamilyAndItsFlag() throws {
        let (dir, snapshot) = try makeSnapshot("manifest")
        let output = NSTemporaryDirectory() + "qwen-man-\(UInt32.random(in: 0...UInt32.max))"
        defer {
            try? FileManager.default.removeItem(atPath: dir)
            try? FileManager.default.removeItem(atPath: output)
        }
        let arch = snapshot.arch
        let plan = try RepackPlanner.plan(meta: snapshot.metadata, arch: arch,
                                          shardHeaders: snapshot.shardHeaders,
                                          outputDir: output)
        let zeroSHA = String(repeating: "0", count: 64)
        let stride = plan.layers[0].expertStride
        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "fixture/qwen",
            sourceSnapshotHash: "sha256:" + snapshot.metadata.indexSha256Hex,
            files: [("model_weights.bin", GTurboJSON.FileEntry(size: 16_384, sha256: zeroSHA))],
            expertsPerLayer: arch.numExperts,
            numLayers: arch.numLayers,
            expertStride: stride,
            bitWidths: GTurboJSON.QuantBitWidths(embedding: 8, attention: 4, router: 16,
                                                 sharedExpert: 8, routedExpert: 4))
        let manifest = try GTurboManifestCodec.decode(data)
        #expect(manifest.arch.family == ArchInfo.qwen35MoeFamily)
        #expect(manifest.arch.layerKinds == arch.layerKinds)
        #expect(manifest.arch.linearAttention?.layerCount == 3)
        #expect(manifest.flags["linearAttention"] == true)
        #expect(manifest.versionMinor >= GTurboFormatV1.versionMinorLinearAttention)
        // The unquantized router is what the reader sees as "BF16, no affine".
        #expect(manifest.quant?.router.weightBits == 16)
        #expect(manifest.quant?.router.scheme == "bf16")
    }
}
