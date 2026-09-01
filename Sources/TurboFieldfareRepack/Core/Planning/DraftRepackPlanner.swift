import Foundation
import TurboFieldfareFormat

/// Plans the MTP drafter's resident file. The drafter is a small int4 model with
/// affine scales and biases, so its plan is the quantized cousin of the vision
/// tower's: a fixed, config-derived list of tensors, each followed by its
/// companions, laid out back to back after the index page.
enum DraftRepackPlanner {

    /// One planned tensor. `logicalShape` is the shape after dequantization;
    /// for a plain BF16 tensor it is the shape as stored.
    struct InventoryEntry {
        let name: String
        let logicalShape: [UInt64]
        let quantized: Bool
    }

    /// Full expected inventory, derived from the pinned config rather than
    /// listed by hand. A source whose tensor set differs — extra, missing or
    /// reshaped — fails the install instead of producing a drafter the runtime
    /// would read past the end of.
    static func expectedInventory(config: DraftSourceConfig) -> [InventoryEntry] {
        func dims(_ values: Int...) -> [UInt64] { values.map(UInt64.init) }
        func quant(_ name: String, _ shape: [UInt64]) -> InventoryEntry {
            InventoryEntry(name: name, logicalShape: shape, quantized: true)
        }
        func plain(_ name: String, _ shape: [UInt64]) -> InventoryEntry {
            InventoryEntry(name: name, logicalShape: shape, quantized: false)
        }
        let hidden = config.hiddenSize
        var out: [InventoryEntry] = []
        out.append(quant("embed_tokens.weight", dims(config.vocabSize, hidden)))
        // Input is `concat(embed(token) * embed_scale, target_hidden)`, so twice
        // the backbone width.
        out.append(quant("pre_projection.weight",
                         dims(hidden, 2 * config.backboneHiddenSize)))
        for layer in 0..<config.numLayers {
            let prefix = "layers.\(layer)."
            let headDim = config.headDim(forLayer: layer)
            let qOut = config.numHeads * headDim
            out.append(quant(prefix + "self_attn.q_proj.weight", dims(qOut, hidden)))
            out.append(plain(prefix + "self_attn.q_norm.weight", dims(headDim)))
            out.append(quant(prefix + "self_attn.o_proj.weight", dims(hidden, qOut)))
            out.append(quant(prefix + "mlp.gate_proj.weight",
                             dims(config.intermediateSize, hidden)))
            out.append(quant(prefix + "mlp.up_proj.weight",
                             dims(config.intermediateSize, hidden)))
            out.append(quant(prefix + "mlp.down_proj.weight",
                             dims(hidden, config.intermediateSize)))
            for norm in ["input_layernorm", "post_attention_layernorm",
                         "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                out.append(plain(prefix + norm + ".weight", dims(hidden)))
            }
            out.append(plain(prefix + "layer_scalar", dims(1)))
        }
        out.append(plain("norm.weight", dims(hidden)))
        out.append(quant("post_projection.weight",
                         dims(config.backboneHiddenSize, hidden)))
        return out
    }

    /// Selects the drafter tensors out of the source shard headers, checks them
    /// against the expected inventory, and lays out the resident file.
    static func plan(path: String,
                     pin: DraftSourcePin,
                     shardHeaders: [Safetensors.Header]) throws -> DraftFilePlan {
        let config = pin.config
        var registry: [String: SourceTensor] = [:]
        for header in shardHeaders {
            for tensor in header.tensors {
                let stripped = tensor.name.hasPrefix(pin.strippedNamePrefix)
                    ? String(tensor.name.dropFirst(pin.strippedNamePrefix.count))
                    : tensor.name
                guard registry.updateValue(tensor, forKey: stripped) == nil else {
                    throw RepackError.configurationInvalid(
                        detail: "duplicate drafter tensor \(stripped)")
                }
            }
        }

        let inventory = expectedInventory(config: config)
        guard inventory.count == pin.expectedTensorCount else {
            throw RepackError.configurationInvalid(detail: """
                drafter inventory derived from the pinned config has \
                \(inventory.count) tensors, expected \(pin.expectedTensorCount)
                """)
        }
        // A quantized tensor arrives as three source tensors; the count below is
        // what the repository must hold, and any leftover means this is not the
        // checkpoint the inventory describes.
        let expectedSourceNames = Set(inventory.flatMap { entry -> [String] in
            guard entry.quantized else { return [entry.name] }
            let base = String(entry.name.dropLast(".weight".count))
            return [entry.name, base + ".scales", base + ".biases"]
        })
        guard registry.count == expectedSourceNames.count else {
            let extra = registry.keys.filter { !expectedSourceNames.contains($0) }.sorted()
            let missing = expectedSourceNames.subtracting(registry.keys).sorted()
            throw RepackError.configurationInvalid(detail: """
                drafter source has \(registry.count) tensors, expected \
                \(expectedSourceNames.count); unexpected: \
                \(extra.prefix(4).joined(separator: ", ")); missing: \
                \(missing.prefix(4).joined(separator: ", "))
                """)
        }

        let names = inventory.map(\.name)
        let (stringTable, offsets, indexSize) =
            RepackPlanner.residentIndexLayout(names: names)
        var entries: [ResidentEntry] = []
        entries.reserveCapacity(inventory.count)
        var cursor = indexSize
        var payloadBytes: UInt64 = 0

        for item in inventory {
            guard let weight = registry[item.name] else {
                throw RepackError.missingTensor(name: item.name)
            }
            guard item.logicalShape.count <= 4,
                  item.logicalShape.allSatisfy({ $0 <= UInt64(UInt32.max) }) else {
                throw RepackError.shapeMismatch(
                    name: item.name,
                    detail: "shape \(item.logicalShape) is not representable in a v1 index entry")
            }
            if item.quantized {
                let base = String(item.name.dropLast(".weight".count))
                guard let scales = registry[base + ".scales"] else {
                    throw RepackError.missingScalesCompanion(name: item.name)
                }
                guard let biases = registry[base + ".biases"] else {
                    throw RepackError.missingBiasesCompanion(name: item.name)
                }
                guard weight.dtype == .u32 else {
                    throw RepackError.dtypeMismatch(
                        name: item.name,
                        detail: "expected U32 packed weights, got \(weight.dtype)")
                }
                guard scales.dtype == .bf16, biases.dtype == .bf16 else {
                    throw RepackError.dtypeMismatch(
                        name: item.name,
                        detail: "expected BF16 scales/biases, got \(scales.dtype)/\(biases.dtype)")
                }
                let packed = try packedShape(item.logicalShape,
                                             bits: config.quantBits,
                                             name: item.name)
                guard weight.shape == packed else {
                    throw RepackError.shapeMismatch(
                        name: item.name, detail: "expected \(packed), got \(weight.shape)")
                }
                let companion = try companionShape(item.logicalShape,
                                                   groupSize: config.quantGroupSize,
                                                   name: item.name)
                guard scales.shape == companion, biases.shape == companion else {
                    throw RepackError.shapeMismatch(
                        name: item.name,
                        detail: "expected \(companion) scales/biases, got "
                            + "\(scales.shape)/\(biases.shape)")
                }
                let weightOffset = cursor
                let scaleOffset = weightOffset + weight.sizeBytes
                let biasOffset = scaleOffset + scales.sizeBytes
                cursor = biasOffset + biases.sizeBytes
                payloadBytes += weight.sizeBytes + scales.sizeBytes + biases.sizeBytes
                entries.append(ResidentEntry(
                    name: item.name,
                    dtype: GTurboFormatV1.DType.u32.rawValue,
                    logicalShape4: padTo4(item.logicalShape),
                    fileOffset: weightOffset, sizeBytes: weight.sizeBytes,
                    scaleOffset: scaleOffset, scaleSize: scales.sizeBytes,
                    biasOffset: biasOffset, biasSize: biases.sizeBytes,
                    quantSpec: QuantSpec(bits: config.quantBits),
                    sourceWeight: weight, sourceScales: scales, sourceBiases: biases))
            } else {
                guard weight.dtype == .bf16 else {
                    throw RepackError.dtypeMismatch(
                        name: item.name,
                        detail: "expected BF16 drafter weights, got \(weight.dtype)")
                }
                guard weight.shape == item.logicalShape else {
                    throw RepackError.shapeMismatch(
                        name: item.name,
                        detail: "expected \(item.logicalShape), got \(weight.shape)")
                }
                entries.append(ResidentEntry(
                    name: item.name,
                    dtype: GTurboFormatV1.DType.bf16.rawValue,
                    logicalShape4: padTo4(item.logicalShape),
                    fileOffset: cursor, sizeBytes: weight.sizeBytes,
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: weight, sourceScales: nil, sourceBiases: nil))
                cursor += weight.sizeBytes
                payloadBytes += weight.sizeBytes
            }
        }

        guard payloadBytes == pin.expectedPayloadBytes else {
            throw RepackError.configurationInvalid(detail: """
                drafter payload is \(payloadBytes) bytes, expected \
                \(pin.expectedPayloadBytes)
                """)
        }

        let resident = ResidentFilePlan(path: path,
                                        entries: entries,
                                        stringTable: stringTable,
                                        stringTableOffsets: offsets,
                                        indexSize: indexSize,
                                        residentSize: cursor - indexSize)
        return DraftFilePlan(resident: resident,
                             source: pin,
                             payloadBytes: payloadBytes)
    }

    /// `[rows, cols]` as MLX stores it once `32 / bits` values share a `UInt32`.
    private static func packedShape(_ logical: [UInt64],
                                    bits: Int,
                                    name: String) throws -> [UInt64] {
        guard bits > 0, bits <= 32, 32 % bits == 0 else {
            throw RepackError.configurationInvalid(
                detail: "unsupported drafter weight bits \(bits)")
        }
        let factor = UInt64(32 / bits)
        guard let last = logical.last, last % factor == 0 else {
            throw RepackError.shapeMismatch(
                name: name, detail: "\(logical) does not pack into \(bits)-bit words")
        }
        var out = logical
        out[out.count - 1] = last / factor
        return out
    }

    /// `[rows, cols / group]`: one scale and one bias per quantization group.
    private static func companionShape(_ logical: [UInt64],
                                       groupSize: Int,
                                       name: String) throws -> [UInt64] {
        guard groupSize > 0 else {
            throw RepackError.configurationInvalid(
                detail: "unsupported drafter group size \(groupSize)")
        }
        guard let last = logical.last, last % UInt64(groupSize) == 0 else {
            throw RepackError.shapeMismatch(
                name: name, detail: "\(logical) is not a multiple of group \(groupSize)")
        }
        var out = logical
        out[out.count - 1] = last / UInt64(groupSize)
        return out
    }

    private static func padTo4(_ shape: [UInt64]) -> [UInt32] {
        var out = shape.prefix(4).map { UInt32($0) }
        while out.count < 4 { out.append(0) }
        return out
    }
}
