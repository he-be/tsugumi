import Foundation
import CryptoKit
@testable import TurboFieldfareRepackCore

/// A miniature stand-in for the MLX conversion of Google's QAT assistant: the
/// full drafter inventory at toy dimensions, quantized the way MLX quantizes
/// (U32 packed weights with BF16 scales and biases), with the geometry the
/// synthetic text snapshot uses so the two can legitimately be paired.
enum SyntheticDraftRepo {

    struct Repo {
        let repoID: String
        let revision: String
        /// Served by `FakeHFURLProtocol.repoFiles[repoID]`.
        let files: [String: Data]
        let pin: DraftSourcePin
        /// Drafter payload as it must appear in `draft/draft_weights.bin`, keyed
        /// by the name the index entry gets (upstream prefix stripped) and by
        /// component (`.weight` / `.scales` / `.biases`).
        let expectedTensorBytes: [String: Data]
    }

    /// Mirrors `SyntheticSnapshot.Arch` wherever the drafter is required to:
    /// head dims, K/V head counts, window, RoPE, vocabulary and the backbone
    /// width are the target's, restated.
    static let config = DraftSourceConfig(
        hiddenSize: 32,
        numLayers: 2,
        numHeads: 4,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 32,
        fullHeadDim: 64,
        intermediateSize: 64,
        backboneHiddenSize: 128,
        vocabSize: 512,
        slidingWindow: 128,
        ropeTheta: 10000.0,
        fullRopeTheta: 1000000.0,
        partialRotaryFactor: 0.25,
        rmsNormEps: 1e-6,
        hiddenActivation: "gelu_pytorch_tanh",
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: [0, 1],
        quantBits: 4,
        quantGroupSize: 16,
        quantMode: "affine")

    static func build(repoID: String = "mlx-community/synthetic-draft",
                      revision: String = FakeHFURLProtocol.commit,
                      config: DraftSourceConfig = SyntheticDraftRepo.config) throws -> Repo {
        var rng = SplitMix64(seed: 0xD2AF_7E51_9C0B_3A64)
        var tensors: [(name: String, dtype: String, shape: [Int], bytes: [UInt8])] = []
        var expected: [String: Data] = [:]
        var payloadBytes: UInt64 = 0

        func emit(_ name: String, _ dtype: String, _ shape: [Int]) {
            let width = dtype == "U32" ? 4 : 2
            let count = shape.reduce(1, *) * width
            var bytes = [UInt8](repeating: 0, count: count)
            for i in 0..<count { bytes[i] = UInt8(rng.next() & 0xFF) }
            tensors.append((name: "model." + name, dtype: dtype, shape: shape,
                            bytes: bytes))
            expected[name] = Data(bytes)
            payloadBytes += UInt64(count)
        }

        for entry in DraftRepackPlanner.expectedInventory(config: config) {
            let shape = entry.logicalShape.map { Int($0) }
            guard entry.quantized else {
                emit(entry.name, "BF16", shape)
                continue
            }
            let base = String(entry.name.dropLast(".weight".count))
            var packed = shape
            packed[packed.count - 1] /= (32 / config.quantBits)
            var companion = shape
            companion[companion.count - 1] /= config.quantGroupSize
            emit(entry.name, "U32", packed)
            emit(base + ".scales", "BF16", companion)
            emit(base + ".biases", "BF16", companion)
        }

        let provenanceNames = ["norm.weight", "layers.0.input_layernorm.weight"]
        let provenance = provenanceNames.map { name in
            DraftSourcePin.ProvenanceTensor(
                repoName: "model." + name,
                sha256: hex(SHA256.hash(data: expected[name]!)))
        }

        let shardName = "model.safetensors"
        let shardData = encodeShard(tensors)
        var weightMap: [String: String] = [:]
        for tensor in tensors { weightMap[tensor.name] = shardName }
        let indexData = try JSONSerialization.data(
            withJSONObject: ["metadata": ["total_size": payloadBytes],
                             "weight_map": weightMap],
            options: [.sortedKeys])
        let configData = try JSONSerialization.data(
            withJSONObject: configJSON(config), options: [.sortedKeys])

        let pin = DraftSourcePin(
            repoID: repoID,
            revision: revision,
            indexSha256Hex: hex(SHA256.hash(data: indexData)),
            displayName: "synthetic MTP drafter",
            strippedNamePrefix: "model.",
            expectedTensorCount: DraftRepackPlanner.expectedInventory(config: config).count,
            expectedPayloadBytes: payloadBytes,
            provenanceTensors: provenance,
            config: config)

        return Repo(repoID: repoID,
                    revision: revision,
                    files: [
                        "model.safetensors.index.json": indexData,
                        "config.json": configData,
                        shardName: shardData,
                    ],
                    pin: pin,
                    expectedTensorBytes: expected)
    }

    static func configJSON(_ config: DraftSourceConfig) -> [String: Any] {
        [
            "architectures": ["Gemma4AssistantForCausalLM"],
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": config.backboneHiddenSize,
            "use_ordered_embeddings": false,
            "tie_word_embeddings": config.tieWordEmbeddings,
            "quantization": [
                "bits": config.quantBits,
                "group_size": config.quantGroupSize,
                "mode": config.quantMode,
            ],
            "text_config": [
                "hidden_size": config.hiddenSize,
                "num_hidden_layers": config.numLayers,
                "num_kv_shared_layers": config.numLayers,
                "num_attention_heads": config.numHeads,
                "num_key_value_heads": config.numKVHeads,
                "num_global_key_value_heads": config.numFullKVHeads,
                "head_dim": config.headDim,
                "global_head_dim": config.fullHeadDim,
                "intermediate_size": config.intermediateSize,
                "vocab_size": config.vocabSize,
                "sliding_window": config.slidingWindow,
                "rms_norm_eps": config.rmsNormEps,
                "hidden_activation": config.hiddenActivation,
                "tie_word_embeddings": config.tieWordEmbeddings,
                "attention_k_eq_v": config.attentionKEqV,
                "enable_moe_block": false,
                "final_logit_softcapping": NSNull(),
                "layer_types": config.fullAttentionLayerMask.map {
                    $0 == 1 ? "full_attention" : "sliding_attention"
                },
                "rope_parameters": [
                    "sliding_attention": ["rope_theta": config.ropeTheta,
                                          "rope_type": "default"],
                    "full_attention": ["rope_theta": config.fullRopeTheta,
                                       "rope_type": "proportional",
                                       "partial_rotary_factor": config.partialRotaryFactor],
                ],
            ],
        ]
    }

    /// Removes one tensor from an encoded shard by rewriting its header. The
    /// payload keeps its bytes, so only the inventory changes.
    static func shardDropping(tensor name: String, from shard: Data) throws -> Data {
        let headerSize = shard.prefix(8).withUnsafeBytes { raw -> UInt64 in
            var value: UInt64 = 0
            for i in 0..<8 { value |= UInt64(raw[i]) << UInt64(i * 8) }
            return value
        }
        var header = try JSONSerialization.jsonObject(
            with: shard.subdata(in: 8..<(8 + Int(headerSize)))) as! [String: Any]
        header.removeValue(forKey: name)
        var headerData = try JSONSerialization.data(withJSONObject: header,
                                                    options: [.sortedKeys])
        // Keep the original header length so the payload offsets still resolve.
        guard headerData.count <= Int(headerSize) else { fatalError("header grew") }
        while headerData.count < Int(headerSize) { headerData.append(0x20) }
        var out = shard.prefix(8)
        out.append(headerData)
        out.append(shard.suffix(from: 8 + Int(headerSize)))
        return out
    }

    // MARK: - safetensors helpers

    private static func encodeShard(
        _ tensors: [(name: String, dtype: String, shape: [Int], bytes: [UInt8])]
    ) -> Data {
        var offset = 0
        var header: [String: Any] = ["__metadata__": ["format": "mlx"]]
        for tensor in tensors {
            header[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, offset + tensor.bytes.count],
            ]
            offset += tensor.bytes.count
        }
        var headerData = try! JSONSerialization.data(withJSONObject: header,
                                                     options: [.sortedKeys])
        while headerData.count % 8 != 0 { headerData.append(0x20) }
        var out = Data()
        var lengthLE = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &lengthLE) { out.append(contentsOf: $0) }
        out.append(headerData)
        for tensor in tensors { out.append(contentsOf: tensor.bytes) }
        return out
    }

    private static func hex(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}
