import Foundation

/// Synthesises a tiny MLX-quantized snapshot shaped like Qwen3.5-MoE
/// (`docs/qwen35moe/01-MODEL.md`): most layers hold a recurrent state, a few
/// attend, the routed experts are written as `mlp.switch_mlp`, and an MTP head
/// plus a vision tower ship inside the same checkpoint. Nothing here has to be
/// numerically meaningful — the planner reads shapes, names and dtypes.
enum SyntheticQwenSnapshot {

    struct Snapshot {
        let shardPath: String
        let directory: String
    }

    struct Arch {
        let hidden = 128
        let moeIntermediate = 64
        // A multiple of the group size, as every quantized inner dimension is.
        let sharedExpertIntermediate = 128
        let numHeads = 2
        let numKVHeads = 2
        let headDim = 32
        let vocab = 512
        let numLayers = 4
        let numExperts = 2
        let topK = 2
        let groupSize = 64
        let linearNumKeyHeads = 2
        let linearNumValueHeads = 4
        let linearKeyHeadDim = 16
        let linearValueHeadDim = 16
        let convKernelDim = 4
        /// One layer in four attends, as upstream does at 40 layers.
        let layerTypes = ["linear_attention", "linear_attention",
                          "linear_attention", "full_attention"]
    }

    static func build(at dir: String, seed: UInt64 = 0x5157_454E_3335_0001) throws -> Snapshot {
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)

        let arch = Arch()
        var rng = SplitMix64(seed: seed)
        var tensors: [(String, String, [Int], [UInt8])] = []

        // Embedding and head are 8-bit in both published conversions.
        appendQuantized(name: "language_model.model.embed_tokens",
                        outerShape: [arch.vocab], innerLogical: arch.hidden,
                        bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
        appendQuantized(name: "language_model.lm_head",
                        outerShape: [arch.vocab], innerLogical: arch.hidden,
                        bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
        appendBF16(name: "language_model.model.norm.weight", shape: [arch.hidden],
                   into: &tensors, rng: &rng)

        for layer in 0..<arch.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            appendBlock(prefix: prefix,
                        kind: arch.layerTypes[layer],
                        arch: arch,
                        into: &tensors,
                        rng: &rng)
        }

        // The MTP head ships inside the text checkpoint and has a `layers.0` of
        // its own — the collision the planner has to split off.
        appendQuantized(name: "language_model.mtp.fc",
                        outerShape: [arch.hidden], innerLogical: 2 * arch.hidden,
                        bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
        appendBF16(name: "language_model.mtp.norm.weight", shape: [arch.hidden],
                   into: &tensors, rng: &rng)
        appendBlock(prefix: "language_model.mtp.layers.0",
                    kind: "full_attention",
                    arch: arch,
                    into: &tensors,
                    rng: &rng)

        // Vision tower, excluded from a text install by prefix.
        appendBF16(name: "vision_tower.patch_embed.proj.weight",
                   shape: [arch.hidden, 4], into: &tensors, rng: &rng)

        let shardName = "model-00001-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)

        var quant: [String: Any] = ["bits": 4, "group_size": arch.groupSize, "mode": "affine"]
        for (name, dtype, _, _) in tensors where dtype == "U32" {
            let base = String(name.dropLast(".weight".count))
            if base.hasSuffix("embed_tokens") || base.hasSuffix("lm_head")
                || base.contains("shared_expert") || base.contains("in_proj_a")
                || base.contains("in_proj_b") {
                quant[base] = ["bits": 8, "group_size": arch.groupSize, "mode": "affine"]
            }
        }
        let textConfig: [String: Any] = [
            "hidden_size": arch.hidden,
            "shared_expert_intermediate_size": arch.sharedExpertIntermediate,
            "moe_intermediate_size": arch.moeIntermediate,
            "num_attention_heads": arch.numHeads,
            "num_key_value_heads": arch.numKVHeads,
            "head_dim": arch.headDim,
            "vocab_size": arch.vocab,
            "num_hidden_layers": arch.numLayers,
            "num_experts": arch.numExperts,
            "num_experts_per_tok": arch.topK,
            "linear_num_key_heads": arch.linearNumKeyHeads,
            "linear_num_value_heads": arch.linearNumValueHeads,
            "linear_key_head_dim": arch.linearKeyHeadDim,
            "linear_value_head_dim": arch.linearValueHeadDim,
            "linear_conv_kernel_dim": arch.convKernelDim,
            "partial_rotary_factor": 0.25,
            "rope_parameters": ["rope_theta": 10_000_000.0,
                                "partial_rotary_factor": 0.25,
                                "type": "default"],
            "layer_types": arch.layerTypes,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "model_type": "qwen3_5_moe_text",
        ]
        let config: [String: Any] = [
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "quantization": quant,
            "text_config": textConfig,
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("config.json")))

        var weightMap: [String: String] = [:]
        for (name, _, _, _) in tensors { weightMap[name] = shardName }
        let index: [String: Any] = ["metadata": ["format": "mlx"], "weight_map": weightMap]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath:
                (dir as NSString).appendingPathComponent("model.safetensors.index.json")))
        return Snapshot(shardPath: shardPath, directory: dir)
    }

    /// One transformer block: whichever attention shape the layer has, plus the
    /// MoE block every layer carries.
    private static func appendBlock(prefix: String,
                                    kind: String,
                                    arch: Arch,
                                    into tensors: inout [(String, String, [Int], [UInt8])],
                                    rng: inout SplitMix64) {
        if kind == "full_attention" {
            let qOut = arch.numHeads * arch.headDim
            let kvOut = arch.numKVHeads * arch.headDim
            for (role, out, inner) in [("q_proj", qOut, arch.hidden),
                                       ("k_proj", kvOut, arch.hidden),
                                       ("v_proj", kvOut, arch.hidden),
                                       ("o_proj", arch.hidden, qOut)] {
                appendQuantized(name: prefix + ".self_attn." + role,
                                outerShape: [out], innerLogical: inner,
                                bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            }
            appendBF16(name: prefix + ".self_attn.q_norm.weight", shape: [arch.headDim],
                       into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".self_attn.k_norm.weight", shape: [arch.headDim],
                       into: &tensors, rng: &rng)
        } else {
            let keyWidth = arch.linearNumKeyHeads * arch.linearKeyHeadDim
            let valueWidth = arch.linearNumValueHeads * arch.linearValueHeadDim
            appendQuantized(name: prefix + ".linear_attn.in_proj_qkv",
                            outerShape: [2 * keyWidth + valueWidth], innerLogical: arch.hidden,
                            bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantized(name: prefix + ".linear_attn.in_proj_z",
                            outerShape: [valueWidth], innerLogical: arch.hidden,
                            bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            for role in ["in_proj_a", "in_proj_b"] {
                appendQuantized(name: prefix + ".linear_attn." + role,
                                outerShape: [arch.linearNumValueHeads], innerLogical: arch.hidden,
                                bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            }
            appendQuantized(name: prefix + ".linear_attn.out_proj",
                            outerShape: [arch.hidden], innerLogical: valueWidth,
                            bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            // The MLX conversion carries the kernel as [channels, taps, 1].
            appendBF16(name: prefix + ".linear_attn.conv1d.weight",
                       shape: [2 * keyWidth + valueWidth, arch.convKernelDim, 1],
                       into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".linear_attn.A_log",
                       shape: [arch.linearNumValueHeads], into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".linear_attn.dt_bias",
                       shape: [arch.linearNumValueHeads], into: &tensors, rng: &rng)
            appendBF16(name: prefix + ".linear_attn.norm.weight",
                       shape: [arch.linearValueHeadDim], into: &tensors, rng: &rng)
        }

        appendBF16(name: prefix + ".input_layernorm.weight", shape: [arch.hidden],
                   into: &tensors, rng: &rng)
        appendBF16(name: prefix + ".post_attention_layernorm.weight", shape: [arch.hidden],
                   into: &tensors, rng: &rng)
        // The oQ conversion leaves the router unquantized.
        appendBF16(name: prefix + ".mlp.gate.weight", shape: [arch.numExperts, arch.hidden],
                   into: &tensors, rng: &rng)
        appendQuantized(name: prefix + ".mlp.shared_expert_gate",
                        outerShape: [1], innerLogical: arch.hidden,
                        bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
        for (role, out, inner) in [("gate_proj", arch.sharedExpertIntermediate, arch.hidden),
                                   ("up_proj", arch.sharedExpertIntermediate, arch.hidden),
                                   ("down_proj", arch.hidden, arch.sharedExpertIntermediate)] {
            appendQuantized(name: prefix + ".mlp.shared_expert." + role,
                            outerShape: [out], innerLogical: inner,
                            bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
        }
        // Routed experts: the same three roles Gemma writes as
        // `experts.switch_glu`, stacked over the expert axis.
        for (role, out, inner) in [("gate_proj", arch.moeIntermediate, arch.hidden),
                                   ("up_proj", arch.moeIntermediate, arch.hidden),
                                   ("down_proj", arch.hidden, arch.moeIntermediate)] {
            appendQuantized(name: prefix + ".mlp.switch_mlp." + role,
                            outerShape: [arch.numExperts, out], innerLogical: inner,
                            bits: 4, groupSize: arch.groupSize, into: &tensors, rng: &rng)
        }
    }

    // MARK: - Tensor builders

    private static func appendQuantized(name: String,
                                        outerShape: [Int],
                                        innerLogical: Int,
                                        bits: Int,
                                        groupSize: Int,
                                        into tensors: inout [(String, String, [Int], [UInt8])],
                                        rng: inout SplitMix64) {
        precondition(innerLogical % groupSize == 0)
        let factor = 32 / bits
        precondition(innerLogical % factor == 0)
        let shape = outerShape + [innerLogical / factor]
        tensors.append((name + ".weight", "U32", shape,
                        randomBytes(shape.reduce(1, *) * 4, rng: &rng)))
        let companionShape = outerShape + [innerLogical / groupSize]
        let companionBytes = companionShape.reduce(1, *) * 2
        tensors.append((name + ".scales", "BF16", companionShape,
                        randomBytes(companionBytes, rng: &rng)))
        tensors.append((name + ".biases", "BF16", companionShape,
                        randomBytes(companionBytes, rng: &rng)))
    }

    private static func appendBF16(name: String, shape: [Int],
                                   into tensors: inout [(String, String, [Int], [UInt8])],
                                   rng: inout SplitMix64) {
        tensors.append((name, "BF16", shape, randomBytes(shape.reduce(1, *) * 2, rng: &rng)))
    }

    private static func randomBytes(_ count: Int, rng: inout SplitMix64) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: count)
        for i in 0..<count { bytes[i] = UInt8(rng.next() & 0xFF) }
        return bytes
    }

    private static func writeShard(path: String,
                                   tensors: [(String, String, [Int], [UInt8])]) throws {
        var offset: UInt64 = 0
        var header: [String: Any] = ["__metadata__": ["format": "mlx"]]
        for (name, dtype, shape, bytes) in tensors {
            let end = offset + UInt64(bytes.count)
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [offset, end]]
            offset = end
        }
        var padded = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while padded.count % 8 != 0 { padded.append(0x20) }
        var out = Data()
        withUnsafeBytes(of: UInt64(padded.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(padded)
        for (_, _, _, bytes) in tensors { out.append(contentsOf: bytes) }
        try out.write(to: URL(fileURLWithPath: path))
    }
}
