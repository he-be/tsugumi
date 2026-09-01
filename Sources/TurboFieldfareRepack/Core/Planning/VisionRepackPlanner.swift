import Foundation
import TurboFieldfareFormat

/// Plans the vision tower's resident file. The tower is BF16 throughout, has no
/// quantization companions and no per-expert streaming, so its plan is the
/// simplest possible use of the v1 resident index: a name-ordered list of
/// tensors laid out back to back after the index page.
enum VisionRepackPlanner {

    /// Full expected inventory, derived from the pinned config rather than
    /// listed by hand. A source whose tensor set differs — extra, missing or
    /// reshaped — fails the install instead of producing a tower the runtime
    /// would read past the end of.
    static func expectedInventory(config: VisionSourceConfig,
                                  textHiddenSize: Int) -> [(name: String, shape: [UInt64])] {
        func dims(_ values: Int...) -> [UInt64] { values.map(UInt64.init) }
        var out: [(String, [UInt64])] = []
        let hidden = config.hiddenSize
        out.append(("vision_tower.patch_embedder.input_proj.weight",
                    dims(hidden, config.patchSize * config.patchSize * 3)))
        out.append(("vision_tower.patch_embedder.position_embedding_table",
                    dims(2, config.positionEmbeddingSize, hidden)))
        for layer in 0..<config.numLayers {
            let prefix = "vision_tower.encoder.layers.\(layer)."
            let qOut = config.numHeads * config.headDim
            let kvOut = config.numKVHeads * config.headDim
            out.append((prefix + "self_attn.q_proj.linear.weight", dims(qOut, hidden)))
            out.append((prefix + "self_attn.k_proj.linear.weight", dims(kvOut, hidden)))
            out.append((prefix + "self_attn.v_proj.linear.weight", dims(kvOut, hidden)))
            out.append((prefix + "self_attn.o_proj.linear.weight", dims(hidden, qOut)))
            out.append((prefix + "self_attn.q_norm.weight", dims(config.headDim)))
            out.append((prefix + "self_attn.k_norm.weight", dims(config.headDim)))
            out.append((prefix + "mlp.gate_proj.linear.weight",
                        dims(config.intermediateSize, hidden)))
            out.append((prefix + "mlp.up_proj.linear.weight",
                        dims(config.intermediateSize, hidden)))
            out.append((prefix + "mlp.down_proj.linear.weight",
                        dims(hidden, config.intermediateSize)))
            for norm in ["input_layernorm", "post_attention_layernorm",
                         "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                out.append((prefix + norm + ".weight", dims(hidden)))
            }
        }
        out.append(("vision_tower.std_scale", dims(hidden)))
        out.append(("vision_tower.std_bias", dims(hidden)))
        out.append(("embed_vision.embedding_projection.weight",
                    dims(textHiddenSize, hidden)))
        return out.map { (name: $0.0, shape: $0.1) }
    }

    /// Selects the tower tensors out of the source shard headers, checks them
    /// against the expected inventory, and lays out the resident file.
    static func plan(path: String,
                     pin: VisionSourcePin,
                     textHiddenSize: Int,
                     shardHeaders: [Safetensors.Header]) throws -> VisionFilePlan {
        var registry: [String: SourceTensor] = [:]
        for header in shardHeaders {
            for tensor in header.tensors where
                pin.tensorPrefixes.contains(where: { tensor.name.hasPrefix($0) }) {
                let stripped = tensor.name.hasPrefix(pin.strippedNamePrefix)
                    ? String(tensor.name.dropFirst(pin.strippedNamePrefix.count))
                    : tensor.name
                guard registry.updateValue(tensor, forKey: stripped) == nil else {
                    throw RepackError.configurationInvalid(
                        detail: "duplicate vision tensor \(stripped)")
                }
            }
        }

        let inventory = expectedInventory(config: pin.config,
                                          textHiddenSize: textHiddenSize)
        guard inventory.count == pin.expectedTensorCount else {
            throw RepackError.configurationInvalid(detail: """
                vision inventory derived from the pinned config has \
                \(inventory.count) tensors, expected \(pin.expectedTensorCount)
                """)
        }
        guard registry.count == inventory.count else {
            let expectedNames = Set(inventory.map(\.name))
            let extra = registry.keys.filter { !expectedNames.contains($0) }.sorted()
            let missing = expectedNames.subtracting(registry.keys).sorted()
            throw RepackError.configurationInvalid(detail: """
                vision source has \(registry.count) tower tensors, expected \
                \(inventory.count); unexpected: \(extra.prefix(4).joined(separator: ", ")); \
                missing: \(missing.prefix(4).joined(separator: ", "))
                """)
        }

        var entries: [ResidentEntry] = []
        entries.reserveCapacity(inventory.count)
        let names = inventory.map(\.name)
        let (stringTable, offsets, indexSize) =
            RepackPlanner.residentIndexLayout(names: names)
        var cursor = indexSize
        var payloadBytes: UInt64 = 0

        for (name, shape) in inventory {
            guard let tensor = registry[name] else {
                throw RepackError.missingTensor(name: name)
            }
            guard tensor.dtype == .bf16 else {
                throw RepackError.dtypeMismatch(
                    name: name, detail: "expected BF16 vision weights, got \(tensor.dtype)")
            }
            guard tensor.shape == shape else {
                throw RepackError.shapeMismatch(
                    name: name, detail: "expected \(shape), got \(tensor.shape)")
            }
            guard shape.count <= 4, shape.allSatisfy({ $0 <= UInt64(UInt32.max) }) else {
                throw RepackError.shapeMismatch(
                    name: name, detail: "shape \(shape) is not representable in a v1 index entry")
            }
            entries.append(ResidentEntry(
                name: name,
                dtype: GTurboFormatV1.DType.bf16.rawValue,
                logicalShape4: padTo4(shape),
                fileOffset: cursor,
                sizeBytes: tensor.sizeBytes,
                scaleOffset: 0, scaleSize: 0,
                biasOffset: 0, biasSize: 0,
                quantSpec: nil,
                sourceWeight: tensor, sourceScales: nil, sourceBiases: nil))
            cursor += tensor.sizeBytes
            payloadBytes += tensor.sizeBytes
        }

        guard payloadBytes == pin.expectedPayloadBytes else {
            throw RepackError.configurationInvalid(detail: """
                vision payload is \(payloadBytes) bytes, expected \
                \(pin.expectedPayloadBytes)
                """)
        }

        let resident = ResidentFilePlan(path: path,
                                        entries: entries,
                                        stringTable: stringTable,
                                        stringTableOffsets: offsets,
                                        indexSize: indexSize,
                                        residentSize: cursor - indexSize)
        return VisionFilePlan(resident: resident,
                              source: pin,
                              payloadBytes: payloadBytes)
    }

    private static func padTo4(_ shape: [UInt64]) -> [UInt32] {
        var out = shape.prefix(4).map { UInt32($0) }
        while out.count < 4 { out.append(0) }
        return out
    }
}
