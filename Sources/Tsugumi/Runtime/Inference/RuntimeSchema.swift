import Foundation
import MoEPackFormat

/// Shape-and-quantization checks against the resident index, shared by every
/// family this runtime carries.
///
/// The manifest's `quant` slots describe what is **model-wide** — the affine
/// scheme, the group size, the companion dtypes — and bound the weight width.
/// The width itself is read **per tensor, from the index**: `oQ4e-g64` hands
/// out 4 and 8 bit inside one slot, layer by layer, because its imatrix pass
/// spent bits where they paid (`docs/qwen35moe/13-PHASE1-REPACK.md` §4-2).
/// A slot cannot say that, and the index does not have to: `sizeBytes` against
/// the expected element count names the width with no ambiguity, since the two
/// candidate widths differ by exactly a factor of two.
struct ResidentSchemaChecker {
    let index: ResidentIndex

    /// The two widths any affine weight in this runtime may have.
    static let supportedWeightBits = [4, 8]

    func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw ModelError.indexCorrupt(detail: "\(field) byte count overflows UInt64")
        }
        return value
    }

    func checkedIntMultiply(_ lhs: Int, _ rhs: Int, field: String) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw ModelError.indexCorrupt(detail: "\(field) dimension overflows Int")
        }
        return value
    }

    func dimensions(_ rows: Int, _ columns: Int, field: String) throws -> (UInt32, UInt32) {
        guard let r = UInt32(exactly: rows), let c = UInt32(exactly: columns),
              r > 0, c > 0 else {
            throw ModelError.indexCorrupt(detail: "\(field) has invalid dimensions")
        }
        return (r, c)
    }

    /// Unquantized BF16 tensor of rank 1...4. No affine companions, so the
    /// scale and bias regions must be absent rather than merely unused.
    func requireBF16(_ name: String, dims: [Int]) throws {
        guard let entry = index.entries[name] else {
            throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
        }
        guard (1...4).contains(dims.count) else {
            throw ModelError.indexCorrupt(detail: "\(name) has invalid rank \(dims.count)")
        }
        var shape: [UInt32] = []
        var elements: UInt64 = 1
        for dim in dims {
            guard let extent = UInt32(exactly: dim), extent > 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) has invalid dimensions")
            }
            shape.append(extent)
            elements = try checkedMultiply(elements, UInt64(extent), field: name)
        }
        while shape.count < 4 { shape.append(0) }
        let expectedBytes = try checkedMultiply(
            elements, UInt64(MemoryLayout<UInt16>.size), field: name)
        guard entry.dtype == MoEPackFormatV1.DType.bf16.rawValue,
              entry.shape.0 == shape[0], entry.shape.1 == shape[1],
              entry.shape.2 == shape[2], entry.shape.3 == shape[3],
              entry.sizeBytes == expectedBytes,
              entry.scaleOffset == 0, entry.scaleSize == 0,
              entry.biasOffset == 0, entry.biasSize == 0,
              entry.fileOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
            throw ModelError.indexCorrupt(detail: "\(name) does not match the required BF16 schema")
        }
    }

    func requireBF16(_ name: String, count: Int) throws {
        try requireBF16(name, dims: [count])
    }

    /// Unquantized BF16 matrix (a QAT `router.proj.weight`, or Qwen's
    /// `mlp.gate.weight`).
    func requireBF16Matrix(_ name: String, rows: Int, columns: Int) throws {
        try requireBF16(name, dims: [rows, columns])
    }

    /// Byte counts an affine weight of `weightBits` must have at this shape.
    /// The companions do not depend on the width — one scale and one bias per
    /// group of `groupSize` columns, whatever the packing is.
    func affineSizes(rows: Int,
                     columns: Int,
                     weightBits: Int,
                     groupSize: Int,
                     field: String) throws -> (shape: (UInt32, UInt32), weight: UInt64, aux: UInt64) {
        let shape = try dimensions(rows, columns, field: field)
        guard Self.supportedWeightBits.contains(weightBits),
              groupSize > 0,
              columns % groupSize == 0 else {
            throw ModelError.indexCorrupt(detail: "\(field) has unsupported affine quantization")
        }
        let elements = try checkedMultiply(UInt64(rows), UInt64(columns), field: field)
        let bitCount = try checkedMultiply(elements, UInt64(weightBits), field: field)
        guard bitCount % 8 == 0 else {
            throw ModelError.indexCorrupt(detail: "\(field) packed byte count is fractional")
        }
        let groups = UInt64(columns / groupSize)
        let auxElements = try checkedMultiply(UInt64(shape.0), groups, field: field)
        let auxBytes = try checkedMultiply(
            auxElements, UInt64(MemoryLayout<UInt16>.size), field: field)
        return (shape, bitCount / 8, auxBytes)
    }

    func affineSizes(rows: Int,
                     columns: Int,
                     slot: ManifestQuantSlot,
                     field: String) throws -> (shape: (UInt32, UInt32), weight: UInt64, aux: UInt64) {
        try affineSizes(rows: rows, columns: columns,
                        weightBits: slot.weightBits, groupSize: slot.groupSize,
                        field: field)
    }

    /// One affine weight plus its companions.
    ///
    /// `widthVariesByTensor` is the whole mixed-width story. When it is false —
    /// every Gemma tensor, and every Qwen slot whose checkpoint is uniform — the
    /// width is pinned to the slot exactly as it always was. When it is true the
    /// width is read from `sizeBytes` and the slot is only an **upper bound**,
    /// which is the honest reading of a slot that has to summarize 40 layers in
    /// one number. Returns the width that was found, so a caller binding this
    /// tensor to a kernel can ask once and dispatch on the answer.
    @discardableResult
    func requireAffine(_ name: String,
                       rows: Int,
                       columns: Int,
                       slot: ManifestQuantSlot,
                       widthVariesByTensor: Bool = false) throws -> Int {
        guard let entry = index.entries[name] else {
            throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
        }
        let weightBits: Int
        if widthVariesByTensor {
            weightBits = try derivedWeightBits(entry, rows: rows, columns: columns, field: name)
            guard weightBits <= slot.weightBits else {
                throw ModelError.indexCorrupt(
                    detail: "\(name) is \(weightBits)-bit, wider than the "
                        + "\(slot.weightBits)-bit bound its manifest slot declares")
            }
        } else {
            weightBits = slot.weightBits
        }
        let expected = try affineSizes(rows: rows, columns: columns,
                                       weightBits: weightBits, groupSize: slot.groupSize,
                                       field: name)
        let primaryAlignment: UInt64 = weightBits == 4
            ? UInt64(MemoryLayout<UInt16>.alignment)
            : 1
        guard entry.dtype == MoEPackFormatV1.DType.u32.rawValue,
              entry.shape.0 == expected.shape.0,
              entry.shape.1 == expected.shape.1,
              entry.shape.2 == 0, entry.shape.3 == 0,
              entry.sizeBytes == expected.weight,
              entry.scaleSize == expected.aux,
              entry.biasSize == (slot.storesBias ? expected.aux : 0),
              entry.fileOffset % primaryAlignment == 0,
              entry.scaleOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0,
              entry.biasOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
            throw ModelError.indexCorrupt(
                detail: "\(name) affine metadata mismatch: dtype=\(entry.dtype), shape=[\(entry.shape.0),\(entry.shape.1),\(entry.shape.2),\(entry.shape.3)], bytes=\(entry.sizeBytes), scales=\(entry.scaleSize), biases=\(entry.biasSize), expected shape=[\(expected.shape.0),\(expected.shape.1),0,0], bytes=\(expected.weight), aux=\(expected.aux)")
        }
        return weightBits
    }

    /// Width implied by the packed byte count at a known shape. 4 and 8 bit
    /// differ by a factor of two, so at most one of them can match, and a count
    /// matching neither is a corrupt index rather than a third width.
    func derivedWeightBits(_ entry: ResidentIndexEntry,
                           rows: Int,
                           columns: Int,
                           field: String) throws -> Int {
        _ = try dimensions(rows, columns, field: field)
        let elements = try checkedMultiply(UInt64(rows), UInt64(columns), field: field)
        for bits in Self.supportedWeightBits.sorted(by: >)
        where entry.sizeBytes == elements * UInt64(bits) / 8 {
            return bits
        }
        throw ModelError.indexCorrupt(
            detail: "\(field) holds \(entry.sizeBytes) packed bytes, which is no "
                + "supported width at \(rows)x\(columns)")
    }
}

extension Model {

    /// Gate every tensor the executable runtime will dereference: the resident
    /// index and the packed-expert layout have to describe the model the
    /// compiled kernels expect, or the first forward pass reads plausible
    /// garbage instead of failing.
    static func validateRuntimeSchema(residentIndex: ResidentIndex,
                                      layout: PackedExpertsLayout,
                                      manifest: Manifest,
                                      config: ArchConfig) throws {
        guard let quant = manifest.quant else {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required by the executable runtime schema")
        }
        let checker = ResidentSchemaChecker(index: residentIndex)
        if manifest.arch.isGemma4 {
            try validateGemma4ResidentSchema(checker: checker, quant: quant, config: config)
        } else if manifest.arch.family == "qwen3_5_moe" {
            try validateQwenResidentSchema(checker: checker, quant: quant,
                                           arch: manifest.arch, config: config)
        } else {
            throw ModelError.indexCorrupt(
                detail: "unsupported model family \(manifest.arch.family ?? "nil")")
        }
        // The routed experts are the one part both families spell the same way:
        // three roles per expert, stacked in the packed-expert files.
        try validateRoutedExpertSchema(checker: checker, layout: layout,
                                       quant: quant, config: config)
    }

    private static func validateGemma4ResidentSchema(checker: ResidentSchemaChecker,
                                                     quant: ManifestQuant,
                                                     config: ArchConfig) throws {
        try checker.requireAffine(
            "language_model.model.embed_tokens.weight",
            rows: config.vocabSize,
            columns: config.hiddenSize,
            slot: quant.embedding)
        try checker.requireBF16("language_model.model.norm.weight", count: config.hiddenSize)

        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            let isFull = config.fullAttentionLayerMask[layer] != 0
            let headDimension = isFull ? config.fullHeadDim : config.headDim
            let kvHeads = isFull ? config.numFullKVHeads : config.numKVHeads
            let queryDimension = try checker.checkedIntMultiply(
                config.numHeads, headDimension, field: "layer \(layer) query")
            let kvDimension = try checker.checkedIntMultiply(
                kvHeads, headDimension, field: "layer \(layer) key/value")

            for name in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "pre_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm_1.weight",
                "post_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm.weight",
                "router.scale",
            ] {
                try checker.requireBF16("\(prefix).\(name)", count: config.hiddenSize)
            }
            try checker.requireBF16("\(prefix).self_attn.q_norm.weight", count: headDimension)
            try checker.requireBF16("\(prefix).self_attn.k_norm.weight", count: headDimension)
            try checker.requireBF16("\(prefix).router.per_expert_scale", count: config.numExperts)
            try checker.requireBF16("\(prefix).layer_scalar", count: 1)

            try checker.requireAffine("\(prefix).self_attn.q_proj.weight",
                                      rows: queryDimension, columns: config.hiddenSize,
                                      slot: quant.attention)
            try checker.requireAffine("\(prefix).self_attn.k_proj.weight",
                                      rows: kvDimension, columns: config.hiddenSize,
                                      slot: quant.attention)
            if !isFull {
                try checker.requireAffine("\(prefix).self_attn.v_proj.weight",
                                          rows: kvDimension, columns: config.hiddenSize,
                                          slot: quant.attention)
            }
            try checker.requireAffine("\(prefix).self_attn.o_proj.weight",
                                      rows: config.hiddenSize, columns: queryDimension,
                                      slot: quant.attention)
            try checker.requireAffine("\(prefix).mlp.gate_proj.weight",
                                      rows: config.intermediateSize, columns: config.hiddenSize,
                                      slot: quant.sharedExpert)
            try checker.requireAffine("\(prefix).mlp.up_proj.weight",
                                      rows: config.intermediateSize, columns: config.hiddenSize,
                                      slot: quant.sharedExpert)
            try checker.requireAffine("\(prefix).mlp.down_proj.weight",
                                      rows: config.hiddenSize, columns: config.intermediateSize,
                                      slot: quant.sharedExpert)
            if quant.router.weightBits == 16 {
                try checker.requireBF16Matrix("\(prefix).router.proj.weight",
                                              rows: config.numExperts, columns: config.hiddenSize)
            } else {
                try checker.requireAffine("\(prefix).router.proj.weight",
                                          rows: config.numExperts, columns: config.hiddenSize,
                                          slot: quant.router)
            }
        }
    }

    /// Qwen3.5-MoE (`docs/qwen35moe/01-MODEL.md`). Three differences from Gemma
    /// carry most of this function:
    ///
    /// - 30 layers in 40 hold no K/V at all. They project into a recurrent state
    ///   through `linear_attn.*`, whose geometry the manifest states separately
    ///   because `ArchConfig`'s head fields describe the other 10.
    /// - `q_proj` is **twice** the query width: the second half is the output
    ///   gate (`docs/qwen35moe/01-MODEL.md` §3-2), not more heads.
    /// - the word embedding is not tied, so `lm_head` is its own tensor — and an
    ///   8-bit one on the production checkpoint
    ///   (`docs/qwen35moe/17-PHASE2-KERNELS.md` §5).
    private static func validateQwenResidentSchema(checker: ResidentSchemaChecker,
                                                   quant: ManifestQuant,
                                                   arch: ManifestArch,
                                                   config: ArchConfig) throws {
        guard let linear = arch.linearAttention else {
            throw ModelError.indexCorrupt(
                detail: "qwen3_5_moe manifest has no arch.linearAttention section")
        }
        try checker.requireAffine("language_model.model.embed_tokens.weight",
                                  rows: config.vocabSize, columns: config.hiddenSize,
                                  slot: quant.embedding)
        if !config.tieWordEmbeddings {
            try checker.requireAffine("language_model.lm_head.weight",
                                      rows: config.vocabSize, columns: config.hiddenSize,
                                      slot: quant.embedding)
        }
        try checker.requireBF16("language_model.model.norm.weight", count: config.hiddenSize)

        let keyWidth = try checker.checkedIntMultiply(
            linear.numKeyHeads, linear.keyHeadDim, field: "linear key")
        let valueWidth = try checker.checkedIntMultiply(
            linear.numValueHeads, linear.valueHeadDim, field: "linear value")
        let qkvWidth = 2 * keyWidth + valueWidth
        var linearLayers = 0

        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            try checker.requireBF16("\(prefix).input_layernorm.weight", count: config.hiddenSize)
            try checker.requireBF16("\(prefix).post_attention_layernorm.weight",
                                    count: config.hiddenSize)

            if config.fullAttentionLayerMask[layer] != 0 {
                let queryDimension = try checker.checkedIntMultiply(
                    config.numHeads, config.fullHeadDim, field: "layer \(layer) query")
                let kvDimension = try checker.checkedIntMultiply(
                    config.numFullKVHeads, config.fullHeadDim, field: "layer \(layer) key/value")
                // 2x: `q_proj` emits the queries and the output gate together.
                try checker.requireAffine("\(prefix).self_attn.q_proj.weight",
                                          rows: 2 * queryDimension, columns: config.hiddenSize,
                                          slot: quant.attention, widthVariesByTensor: true)
                for role in ["k_proj", "v_proj"] {
                    try checker.requireAffine("\(prefix).self_attn.\(role).weight",
                                              rows: kvDimension, columns: config.hiddenSize,
                                              slot: quant.attention, widthVariesByTensor: true)
                }
                try checker.requireAffine("\(prefix).self_attn.o_proj.weight",
                                          rows: config.hiddenSize, columns: queryDimension,
                                          slot: quant.attention, widthVariesByTensor: true)
                try checker.requireBF16("\(prefix).self_attn.q_norm.weight",
                                        count: config.fullHeadDim)
                try checker.requireBF16("\(prefix).self_attn.k_norm.weight",
                                        count: config.fullHeadDim)
            } else {
                linearLayers += 1
                let linearPrefix = "\(prefix).linear_attn"
                try checker.requireAffine("\(linearPrefix).in_proj_qkv.weight",
                                          rows: qkvWidth, columns: config.hiddenSize,
                                          slot: quant.attention, widthVariesByTensor: true)
                try checker.requireAffine("\(linearPrefix).in_proj_z.weight",
                                          rows: valueWidth, columns: config.hiddenSize,
                                          slot: quant.attention, widthVariesByTensor: true)
                for role in ["in_proj_a", "in_proj_b"] {
                    try checker.requireAffine("\(linearPrefix).\(role).weight",
                                              rows: linear.numValueHeads, columns: config.hiddenSize,
                                              slot: quant.attention, widthVariesByTensor: true)
                }
                try checker.requireAffine("\(linearPrefix).out_proj.weight",
                                          rows: config.hiddenSize, columns: valueWidth,
                                          slot: quant.attention, widthVariesByTensor: true)
                // The MLX conversion carries the depthwise kernel as
                // [channels, taps, 1] and the runtime reads that axis order
                // (`docs/qwen35moe/10-MLX4BIT-AUDIT.md` §4).
                try checker.requireBF16("\(linearPrefix).conv1d.weight",
                                        dims: [qkvWidth, linear.convKernelDim, 1])
                try checker.requireBF16("\(linearPrefix).A_log", count: linear.numValueHeads)
                try checker.requireBF16("\(linearPrefix).dt_bias", count: linear.numValueHeads)
                try checker.requireBF16("\(linearPrefix).norm.weight", count: linear.valueHeadDim)
            }

            // The router: `mlp.gate` under this family's spelling, left
            // unquantized by the oQ conversion.
            if quant.router.weightBits == 16 {
                try checker.requireBF16Matrix("\(prefix).mlp.gate.weight",
                                              rows: config.numExperts, columns: config.hiddenSize)
            } else {
                try checker.requireAffine("\(prefix).mlp.gate.weight",
                                          rows: config.numExperts, columns: config.hiddenSize,
                                          slot: quant.router)
            }
            try checker.requireAffine("\(prefix).mlp.shared_expert_gate.weight",
                                      rows: 1, columns: config.hiddenSize,
                                      slot: quant.sharedExpert)
            for (role, rows, columns) in [
                ("gate_proj", config.intermediateSize, config.hiddenSize),
                ("up_proj", config.intermediateSize, config.hiddenSize),
                ("down_proj", config.hiddenSize, config.intermediateSize),
            ] {
                try checker.requireAffine("\(prefix).mlp.shared_expert.\(role).weight",
                                          rows: rows, columns: columns,
                                          slot: quant.sharedExpert)
            }
        }

        // `layerCount` is what a reader budgets recurrent state from without
        // walking `layerKinds`; a manifest whose two statements disagree would
        // size the state wrong rather than fail.
        guard linearLayers == linear.layerCount else {
            throw ModelError.indexCorrupt(
                detail: "arch.linearAttention.layerCount \(linear.layerCount) disagrees with "
                    + "the \(linearLayers) non-full layers in fullAttentionLayerMask")
        }
    }

    private static func validateRoutedExpertSchema(checker: ResidentSchemaChecker,
                                                   layout: PackedExpertsLayout,
                                                   quant: ManifestQuant,
                                                   config: ArchConfig) throws {
        let routedShapes: [(String, Int, Int)] = [
            ("gate", config.moeIntermediateSize, config.hiddenSize),
            ("up", config.moeIntermediateSize, config.hiddenSize),
            ("down", config.hiddenSize, config.moeIntermediateSize),
        ]
        for layer in layout.layers {
            guard let reference = layer.experts.first else {
                throw ModelError.indexCorrupt(
                    detail: "routed layer \(layer.layer) has no experts")
            }
            for (role, rows, columns) in routedShapes {
                let sizes = try checker.affineSizes(
                    rows: rows, columns: columns,
                    slot: quant.routedExpert,
                    field: "routed layer \(layer.layer) \(role)")
                var expectedRoles: [(String, String, [UInt32], Int?, UInt64, UInt64)] = [
                    (role, "U32", [sizes.shape.0, sizes.shape.1],
                     quant.routedExpert.weightBits, sizes.weight,
                     UInt64(MemoryLayout<UInt32>.alignment)),
                    ("\(role)_scales", "BF16",
                     [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                     nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)),
                ]
                // A `sym` expert blob carries no bias sub-tensor at all: the
                // schema requires its absence as strictly as it requires the
                // other two to be present.
                if quant.routedExpert.storesBias {
                    expectedRoles.append(
                        ("\(role)_biases", "BF16",
                         [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                         nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)))
                } else if reference.subTensors["\(role)_biases"] != nil {
                    throw ModelError.indexCorrupt(
                        detail: "routed layer \(layer.layer) declares scheme "
                            + "\(quant.routedExpert.scheme) but still carries \(role)_biases")
                }
                for (name, dtype, shape, bits, size, alignment) in expectedRoles {
                    guard let expected = reference.subTensors[name] else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) is missing role \(name)")
                    }
                    let (end, overflow) = expected.offset.addingReportingOverflow(expected.size)
                    guard expected.dtype == dtype,
                          expected.shape == shape,
                          expected.bits == bits,
                          expected.size == size,
                          expected.offset % alignment == 0,
                          !overflow,
                          end <= reference.size,
                          end <= UInt64(UInt32.max) + 1 else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) does not match the required schema")
                    }
                    for expert in layer.experts.dropFirst()
                        where expert.subTensors[name] != expected {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) metadata differs across experts")
                    }
                }
            }
        }
    }
}
