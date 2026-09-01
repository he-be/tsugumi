import Foundation
import Testing
@testable import Tsugumi
@testable import MoEPackFormat
@testable import TsugumiRepackCore

/// A tiny `.moepack` shaped like Ornith-1.5-35B-A3B: three layers in four hold a
/// recurrent state, the fourth attends, the word embedding is untied, the router
/// is unquantized BF16 — and the attention weights are **not one width**.
///
/// The last point is the reason this exists. `oQ4e-g64` spends 4 bits in some
/// layers and 8 in others inside the same `quant.attention` slot
/// (`docs/qwen35moe/13-PHASE1-REPACK.md` §4-2), so a fixture whose every tensor
/// agreed with its slot would pass the schema for the wrong reason. Payload
/// bytes are not meaningful: nothing here runs a forward pass.
enum QwenToyInstall {

    static let groupSize = 64

    static let arch = ArchConfig(
        hiddenSize: 128,
        intermediateSize: 128,
        moeIntermediateSize: 64,
        numHeads: 2,
        numKVHeads: 1,
        numFullKVHeads: 1,
        headDim: 64,
        fullHeadDim: 64,
        vocabSize: 256,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 4,
        numExperts: 2,
        topKExperts: 2,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: [0, 0, 0, 1],
        hiddenActivation: "silu")

    static let numKeyHeads = 2
    static let numValueHeads = 4
    static let keyHeadDim = 32
    static let valueHeadDim = 32
    static let convKernelDim = 4
    static var keyWidth: Int { numKeyHeads * keyHeadDim }
    static var valueWidth: Int { numValueHeads * valueHeadDim }
    static var qkvWidth: Int { 2 * keyWidth + valueWidth }

    /// Width of each attention weight, chosen so that one role (`in_proj_qkv`)
    /// mixes across layers and the widths disagree between roles as well.
    static func attentionBits(layer: Int, role: String) -> Int {
        switch role {
        case "in_proj_qkv": return layer == 1 ? 8 : 4
        case "q_proj": return 8
        case "k_proj", "v_proj", "o_proj": return 4
        default: return 8
        }
    }

    struct Options {
        /// What the manifest's attention slot claims. The default is the widest
        /// tensor present, which is what the repacker writes; a test that lowers
        /// it is asking the runtime to catch a tensor wider than its bound.
        var attentionSlotBits = 8
        var directoryName = "moepack-qwen-toy-\(UUID().uuidString)"
    }

    struct ResidentSpec {
        let name: String
        let dtype: UInt8
        let shape: [UInt32]
        let weightBytes: UInt64
        let scaleBytes: UInt64
        let biasBytes: UInt64
    }

    static func bf16(_ name: String, _ dims: [Int]) -> ResidentSpec {
        var shape = dims.map { UInt32($0) }
        while shape.count < 4 { shape.append(0) }
        return ResidentSpec(name: name, dtype: 1, shape: shape,
                            weightBytes: UInt64(dims.reduce(1, *) * 2),
                            scaleBytes: 0, biasBytes: 0)
    }

    static func affine(_ name: String, rows: Int, columns: Int, bits: Int) -> ResidentSpec {
        let aux = UInt64(rows * (columns / groupSize) * 2)
        return ResidentSpec(name: name, dtype: 0,
                            shape: [UInt32(rows), UInt32(columns), 0, 0],
                            weightBytes: UInt64(rows * columns * bits / 8),
                            scaleBytes: aux, biasBytes: aux)
    }

    static func residentSpecs() -> [ResidentSpec] {
        let d = arch.hiddenSize
        var specs: [ResidentSpec] = [
            affine("language_model.model.embed_tokens.weight",
                   rows: arch.vocabSize, columns: d, bits: 8),
            affine("language_model.lm_head.weight",
                   rows: arch.vocabSize, columns: d, bits: 8),
            bf16("language_model.model.norm.weight", [d]),
        ]
        for layer in 0..<arch.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            specs.append(bf16("\(prefix).input_layernorm.weight", [d]))
            specs.append(bf16("\(prefix).post_attention_layernorm.weight", [d]))
            if arch.fullAttentionLayerMask[layer] != 0 {
                let query = arch.numHeads * arch.fullHeadDim
                let kv = arch.numFullKVHeads * arch.fullHeadDim
                // Twice the query width: the second half is the output gate.
                specs.append(affine("\(prefix).self_attn.q_proj.weight",
                                    rows: 2 * query, columns: d,
                                    bits: attentionBits(layer: layer, role: "q_proj")))
                for role in ["k_proj", "v_proj"] {
                    specs.append(affine("\(prefix).self_attn.\(role).weight",
                                        rows: kv, columns: d,
                                        bits: attentionBits(layer: layer, role: role)))
                }
                specs.append(affine("\(prefix).self_attn.o_proj.weight",
                                    rows: d, columns: query,
                                    bits: attentionBits(layer: layer, role: "o_proj")))
                specs.append(bf16("\(prefix).self_attn.q_norm.weight", [arch.fullHeadDim]))
                specs.append(bf16("\(prefix).self_attn.k_norm.weight", [arch.fullHeadDim]))
            } else {
                let linear = "\(prefix).linear_attn"
                specs.append(affine("\(linear).in_proj_qkv.weight",
                                    rows: qkvWidth, columns: d,
                                    bits: attentionBits(layer: layer, role: "in_proj_qkv")))
                specs.append(affine("\(linear).in_proj_z.weight",
                                    rows: valueWidth, columns: d, bits: 8))
                for role in ["in_proj_a", "in_proj_b"] {
                    specs.append(affine("\(linear).\(role).weight",
                                        rows: numValueHeads, columns: d, bits: 8))
                }
                specs.append(affine("\(linear).out_proj.weight",
                                    rows: d, columns: valueWidth, bits: 8))
                specs.append(bf16("\(linear).conv1d.weight", [qkvWidth, convKernelDim, 1]))
                specs.append(bf16("\(linear).A_log", [numValueHeads]))
                specs.append(bf16("\(linear).dt_bias", [numValueHeads]))
                specs.append(bf16("\(linear).norm.weight", [valueHeadDim]))
            }
            specs.append(bf16("\(prefix).mlp.gate.weight", [arch.numExperts, d]))
            specs.append(affine("\(prefix).mlp.shared_expert_gate.weight",
                                rows: 1, columns: d, bits: 8))
            for (role, rows, columns) in [
                ("gate_proj", arch.intermediateSize, d),
                ("up_proj", arch.intermediateSize, d),
                ("down_proj", d, arch.intermediateSize),
            ] {
                specs.append(affine("\(prefix).mlp.shared_expert.\(role).weight",
                                    rows: rows, columns: columns, bits: 8))
            }
        }
        return specs
    }

    /// One routed expert: the same three roles both families write, 4-bit.
    static func expertTensors() -> (size: Int, tensors: [String: [String: Any]]) {
        var tensors: [String: [String: Any]] = [:]
        var offset = 0
        for (role, rows, columns) in [
            ("gate", arch.moeIntermediateSize, arch.hiddenSize),
            ("up", arch.moeIntermediateSize, arch.hiddenSize),
            ("down", arch.hiddenSize, arch.moeIntermediateSize),
        ] {
            let aux = rows * (columns / groupSize) * 2
            tensors[role] = ["offset": offset, "size": rows * columns / 2,
                             "dtype": "U32", "shape": [rows, columns], "bits": 4]
            offset += rows * columns / 2
            for component in ["scales", "biases"] {
                tensors["\(role)_\(component)"] = [
                    "offset": offset, "size": aux, "dtype": "BF16",
                    "shape": [rows, columns / groupSize],
                ]
                offset += aux
            }
        }
        return (offset, tensors)
    }

    static func write(options: Options = Options()) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(options.directoryName)
        let expertsDir = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: expertsDir, withIntermediateDirectories: true)

        // 1. model_weights.bin: index header, entry table, string table, payload.
        let specs = residentSpecs()
        let names = specs.map(\.name)
        let stringTable = names.joined().data(using: .utf8)!
        let entriesBase = MoEPackBinary.indexHeaderBytes
        let stringTableBase = entriesBase + names.count * MoEPackBinary.indexEntryBytes
        var nameOffsets: [UInt32] = []
        var cursor = 0
        for name in names {
            nameOffsets.append(UInt32(stringTableBase + cursor))
            cursor += name.utf8.count
        }
        let alignment = MoEPackFormatV1.alignmentBytes
        let rawIndexBytes = UInt64(stringTableBase + stringTable.count)
        let indexBytes = ((rawIndexBytes + alignment - 1) / alignment) * alignment

        var entries: [ResidentEntry] = []
        var payloadCursor = indexBytes
        for spec in specs {
            let payloadAlignment = UInt64(spec.dtype == 0
                ? MemoryLayout<UInt32>.alignment : MemoryLayout<UInt16>.alignment)
            payloadCursor = ((payloadCursor + payloadAlignment - 1) / payloadAlignment)
                * payloadAlignment
            let weightOffset = payloadCursor
            let scaleOffset = spec.scaleBytes > 0 ? weightOffset + spec.weightBytes : 0
            let biasOffset = spec.biasBytes > 0 ? scaleOffset + spec.scaleBytes : 0
            entries.append(ResidentEntry(
                name: spec.name, dtype: spec.dtype, logicalShape4: spec.shape,
                fileOffset: weightOffset, sizeBytes: spec.weightBytes,
                scaleOffset: scaleOffset, scaleSize: spec.scaleBytes,
                biasOffset: biasOffset, biasSize: spec.biasBytes,
                quantSpec: nil,
                sourceWeight: ModelLoaderTests.dummySource(spec.name),
                sourceScales: nil, sourceBiases: nil))
            payloadCursor += spec.weightBytes + spec.scaleBytes + spec.biasBytes
        }
        let residentSize = payloadCursor - indexBytes
        var fileBuf = [UInt8](repeating: 0, count: Int(indexBytes + residentSize))
        fileBuf.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            MoEPackBinary.writeIndexHeader(into: base,
                                          indexSize: indexBytes,
                                          residentSize: residentSize,
                                          entryCount: UInt64(entries.count))
            for (i, entry) in entries.enumerated() {
                MoEPackBinary.writeIndexEntry(
                    into: base.advanced(by: entriesBase + i * MoEPackBinary.indexEntryBytes),
                    entry: entry, nameOffset: nameOffsets[i])
            }
            _ = stringTable.withUnsafeBytes {
                memcpy(base.advanced(by: stringTableBase), $0.baseAddress!, stringTable.count)
            }
        }
        let weightsURL = dir.appendingPathComponent("model_weights.bin")
        try Data(fileBuf).write(to: weightsURL)
        let weightsSha = try Sha256Verifier.hashFile(at: weightsURL)

        // 2. packed_experts: one file per layer, all forty of a real install's
        // layers routed, at a stride the page allocator accepts.
        let expertStride: UInt64 = 16384
        let expert = expertTensors()
        precondition(expert.size <= Int(expertStride), "toy expert blob exceeds stride")
        let layerBytes = Int(expertStride) * arch.numExperts
        var layerShaByName: [String: String] = [:]
        for layer in 0..<arch.numLayers {
            let basename = String(format: "layer_%02d.bin", layer)
            let url = expertsDir.appendingPathComponent(basename)
            try Data(count: layerBytes).write(to: url)
            layerShaByName["packed_experts/\(basename)"] = try Sha256Verifier.hashFile(at: url)
        }

        // 3. layout.json
        var layersArray: [[String: Any]] = []
        for layer in 0..<arch.numLayers {
            let experts = (0..<arch.numExperts).map { index in
                [
                    "expert": index,
                    "offset": UInt64(index) * expertStride,
                    "size": expertStride,
                    "tensors": expert.tensors,
                ] as [String: Any]
            }
            layersArray.append([
                "layer": layer,
                "file": String(format: "layer_%02d.bin", layer),
                "experts": experts,
            ])
        }
        let layoutData = try JSONSerialization.data(withJSONObject: [
            "expertStride": expertStride,
            "numLayers": arch.numLayers,
            "expertsPerLayer": arch.numExperts,
            "layers": layersArray,
        ] as [String: Any], options: [.sortedKeys])
        let layoutURL = expertsDir.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)

        // 4. manifest.json
        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": fileBuf.count, "sha256": weightsSha],
            "packed_experts/layout.json": [
                "size": layoutData.count,
                "sha256": try Sha256Verifier.hashFile(at: layoutURL),
            ],
        ]
        for (path, sha) in layerShaByName {
            files[path] = ["size": layerBytes, "sha256": sha]
        }
        func slot(_ bits: Int) -> [String: Any] {
            bits == 16
                ? ["weightBits": 16, "scheme": "bf16", "scaleType": "none",
                   "biasType": "none", "groupSize": groupSize]
                : ["weightBits": bits, "scheme": "affine", "scaleType": "BF16",
                   "biasType": "BF16", "groupSize": groupSize]
        }
        let layerKinds = arch.fullAttentionLayerMask.map {
            $0 != 0 ? "full_attention" : "linear_attention"
        }
        let manifestRoot: [String: Any] = [
            "magic": "MOEPACK",
            "versionMajor": 1,
            "versionMinor": MoEPackFormatV1.versionMinorLinearAttention,
            "flags": ["streamingPresent": true, "turboQuantKV": false,
                      "aneSharedExpert": false, "linearAttention": true],
            "modelID": "qwen-toy",
            "arch": [
                "hiddenSize": arch.hiddenSize,
                "ffnIntermediate": arch.intermediateSize,
                "moeIntermediateSize": arch.moeIntermediateSize,
                "numHeads": arch.numHeads,
                "numKVHeads": arch.numKVHeads,
                "numFullKVHeads": arch.numFullKVHeads,
                "headDim": arch.headDim,
                "fullHeadDim": arch.fullHeadDim,
                "vocabSize": arch.vocabSize,
                "slidingWindow": arch.slidingWindow,
                "finalLogitSoftcap": arch.finalLogitSoftcap,
                "ropeTheta": arch.ropeTheta,
                "fullRopeTheta": arch.fullRopeTheta,
                "partialRotaryFactor": arch.partialRotaryFactor,
                "numLayers": arch.numLayers,
                "numExperts": arch.numExperts,
                "topKExperts": arch.topKExperts,
                "tieWordEmbeddings": arch.tieWordEmbeddings,
                "attentionKEqV": arch.attentionKEqV,
                "hiddenActivation": arch.hiddenActivation,
                "fullAttentionLayerMask": arch.fullAttentionLayerMask.map { Int($0) },
                "family": "qwen3_5_moe",
                "layerKinds": layerKinds,
                "linearAttention": [
                    "numKeyHeads": numKeyHeads,
                    "numValueHeads": numValueHeads,
                    "keyHeadDim": keyHeadDim,
                    "valueHeadDim": valueHeadDim,
                    "convKernelDim": convKernelDim,
                    "layerCount": layerKinds.filter { $0 == "linear_attention" }.count,
                ],
            ] as [String: Any],
            "quant": [
                "embedding": slot(8),
                "attention": slot(options.attentionSlotBits),
                "router": slot(16),
                "sharedExpert": slot(8),
                "routedExpert": slot(4),
            ],
            "files": files,
            "expertsPerLayer": arch.numExperts,
            "numLayers": arch.numLayers,
            "expertStride": expertStride,
        ]
        let manifestData = try JSONSerialization.data(
            withJSONObject: manifestRoot, options: [.sortedKeys, .withoutEscapingSlashes])
        try manifestData.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }
}
