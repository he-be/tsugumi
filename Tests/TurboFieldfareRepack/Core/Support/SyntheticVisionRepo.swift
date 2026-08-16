import Foundation
import CryptoKit
@testable import TurboFieldfareRepackCore

/// A miniature stand-in for Google's unquantized QAT repository: the full tower
/// inventory at toy dimensions, plus the two unquantized text tensors the
/// installer uses to prove both halves come from the same checkpoint.
enum SyntheticVisionRepo {

    struct Repo {
        let repoID: String
        let revision: String
        /// Served by `FakeHFURLProtocol.repoFiles[repoID]`.
        let files: [String: Data]
        let pin: VisionSourcePin
        /// Tower payload as it must appear in `vision/vision_weights.bin`,
        /// keyed by the name the index entry gets (upstream prefix stripped).
        let expectedTensorBytes: [String: Data]
    }

    static let config = VisionSourceConfig(
        hiddenSize: 64,
        numLayers: 2,
        numHeads: 4,
        numKVHeads: 4,
        headDim: 16,
        intermediateSize: 96,
        patchSize: 4,
        poolingKernelSize: 3,
        positionEmbeddingSize: 32,
        ropeTheta: 100.0,
        rmsNormEps: 1e-6,
        hiddenActivation: "gelu_pytorch_tanh",
        standardize: true,
        maxSoftTokens: 8,
        imageTokenID: 258880,
        boiTokenID: 255999,
        eoiTokenID: 258882)

    /// - Parameters:
    ///   - textSnapshotDir: the synthetic text snapshot; its BF16 norm tensors
    ///     are copied verbatim so the parity check passes.
    static func build(textSnapshotDir: String,
                      textHiddenSize: Int = 128,
                      repoID: String = "google/synthetic-vision",
                      revision: String = FakeHFURLProtocol.commit) throws -> Repo {
        let parityNames = [
            "language_model.model.norm.weight",
            "language_model.model.layers.0.input_layernorm.weight",
        ]
        let shardPath = (textSnapshotDir as NSString)
            .appendingPathComponent("model-00001-of-00001.safetensors")
        let textTensors = try readTensors(shardPath: shardPath, names: Set(parityNames))

        var rng = SplitMix64(seed: 0x5EED_1A2B_3C4D_5E6F)
        var tensors: [(name: String, dtype: String, shape: [Int], bytes: [UInt8])] = []
        var expected: [String: Data] = [:]
        var payloadBytes: UInt64 = 0

        let inventory = VisionRepackPlanner.expectedInventory(
            config: config, textHiddenSize: textHiddenSize)
        for entry in inventory {
            let shape = entry.shape.map { Int($0) }
            let count = shape.reduce(1, *) * 2
            var bytes = [UInt8](repeating: 0, count: count)
            for i in 0..<count { bytes[i] = UInt8(rng.next() & 0xFF) }
            tensors.append((name: "model." + entry.name, dtype: "BF16",
                            shape: shape, bytes: bytes))
            expected[entry.name] = Data(bytes)
            payloadBytes += UInt64(count)
        }

        var parity: [VisionSourcePin.ParityTensor] = []
        for name in parityNames {
            guard let tensor = textTensors[name] else {
                fatalError("synthetic snapshot has no \(name)")
            }
            tensors.append((name: "model." + name, dtype: "BF16",
                            shape: tensor.shape, bytes: [UInt8](tensor.bytes)))
            parity.append(VisionSourcePin.ParityTensor(
                textName: name,
                visionRepoName: "model." + name,
                sha256: hex(SHA256.hash(data: tensor.bytes))))
        }

        let shardName = "model-00001-of-00001.safetensors"
        let shardData = encodeShard(tensors)
        var weightMap: [String: String] = [:]
        for tensor in tensors { weightMap[tensor.name] = shardName }
        let indexData = try JSONSerialization.data(
            withJSONObject: ["metadata": ["format": "pt"], "weight_map": weightMap],
            options: [.sortedKeys])
        let configData = try JSONSerialization.data(
            withJSONObject: configJSON(), options: [.sortedKeys])

        let pin = VisionSourcePin(
            repoID: repoID,
            revision: revision,
            indexSha256Hex: hex(SHA256.hash(data: indexData)),
            displayName: "synthetic vision tower",
            tensorPrefixes: ["model.vision_tower.", "model.embed_vision."],
            strippedNamePrefix: "model.",
            expectedTensorCount: inventory.count,
            expectedPayloadBytes: payloadBytes,
            parityTensors: parity,
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

    static func configJSON() -> [String: Any] {
        [
            "model_type": "gemma4",
            "image_token_id": config.imageTokenID,
            "boi_token_id": config.boiTokenID,
            "eoi_token_id": config.eoiTokenID,
            "vision_soft_tokens_per_image": config.maxSoftTokens,
            "vision_config": [
                "hidden_size": config.hiddenSize,
                "num_hidden_layers": config.numLayers,
                "num_attention_heads": config.numHeads,
                "num_key_value_heads": config.numKVHeads,
                "head_dim": config.headDim,
                "intermediate_size": config.intermediateSize,
                "patch_size": config.patchSize,
                "pooling_kernel_size": config.poolingKernelSize,
                "position_embedding_size": config.positionEmbeddingSize,
                "rope_parameters": ["rope_theta": config.ropeTheta,
                                    "rope_type": "default"],
                "rms_norm_eps": config.rmsNormEps,
                "hidden_activation": config.hiddenActivation,
                "standardize": config.standardize,
                "use_clipped_linears": false,
                "default_output_length": config.maxSoftTokens,
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

    private struct RawTensor {
        let shape: [Int]
        let bytes: Data
    }

    private static func readTensors(shardPath: String,
                                    names: Set<String>) throws -> [String: RawTensor] {
        let data = try Data(contentsOf: URL(fileURLWithPath: shardPath))
        let headerSize = data.prefix(8).withUnsafeBytes { raw -> UInt64 in
            var value: UInt64 = 0
            for i in 0..<8 { value |= UInt64(raw[i]) << UInt64(i * 8) }
            return value
        }
        let header = try JSONSerialization.jsonObject(
            with: data.subdata(in: 8..<(8 + Int(headerSize)))) as! [String: Any]
        let base = 8 + Int(headerSize)
        var out: [String: RawTensor] = [:]
        for name in names {
            guard let entry = header[name] as? [String: Any],
                  let offsets = entry["data_offsets"] as? [Any],
                  let shape = entry["shape"] as? [Any] else { continue }
            let begin = (offsets[0] as! NSNumber).intValue
            let end = (offsets[1] as! NSNumber).intValue
            out[name] = RawTensor(shape: shape.map { ($0 as! NSNumber).intValue },
                                  bytes: data.subdata(in: (base + begin)..<(base + end)))
        }
        return out
    }

    private static func encodeShard(
        _ tensors: [(name: String, dtype: String, shape: [Int], bytes: [UInt8])]
    ) -> Data {
        var offset = 0
        var header: [String: Any] = ["__metadata__": ["format": "pt"]]
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
