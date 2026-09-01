import Foundation

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
struct ArchInfo: Sendable, Equatable {
    /// Fixed-size recurrent layers (Qwen3.5-MoE's Gated DeltaNet). Absent for a
    /// family whose every layer attends.
    struct LinearAttention: Sendable, Equatable {
        let numKeyHeads: Int
        let numValueHeads: Int
        let keyHeadDim: Int
        let valueHeadDim: Int
        let convKernelDim: Int
        let layerCount: Int
    }

    /// `gemma4` is the family this repository was built for; a manifest written
    /// for it omits the family key entirely, so its bytes never moved.
    static let gemma4Family = "gemma4"
    static let qwen35MoeFamily = "qwen3_5_moe"

    let family: String
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 otherwise. What the zeros mean depends on the
    /// family, which is why `layerKinds` exists alongside it.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String
    /// Per-layer kind, verbatim from `config.json -> layer_types`.
    let layerKinds: [String]
    let linearAttention: LinearAttention?

    init(family: String, hiddenSize: Int, intermediateSize: Int, moeIntermediateSize: Int,
         numHeads: Int, numKVHeads: Int, numFullKVHeads: Int, headDim: Int, fullHeadDim: Int,
         vocabSize: Int, slidingWindow: Int, finalLogitSoftcap: Double, ropeTheta: Double,
         fullRopeTheta: Double, partialRotaryFactor: Double, numLayers: Int, numExperts: Int,
         topKExperts: Int, tieWordEmbeddings: Bool, attentionKEqV: Bool,
         fullAttentionLayerMask: [UInt8], hiddenActivation: String,
         layerKinds: [String] = [], linearAttention: LinearAttention? = nil) {
        self.family = family
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
        self.layerKinds = layerKinds
        self.linearAttention = linearAttention
    }

    var isGemma4: Bool { family == ArchInfo.gemma4Family }

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tc = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        // The family decides which keys exist at all: Qwen3.5-MoE names no
        // sliding window, no softcap and no global head count, and Gemma names
        // no linear-attention geometry. Reading either config with the other
        // family's parser fails on a missing key rather than inventing one.
        let modelType = (root["model_type"] as? String) ?? ""
        if modelType.hasPrefix("qwen3_5_moe") {
            return try loadQwen35Moe(configPath: configPath, textConfig: tc)
        }
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let ropeFull = (rope["full_attention"] as? [String: Any]) ?? [:]
        let ropeSWA  = (rope["sliding_attention"] as? [String: Any]) ?? [:]
        let prf = (ropeFull["partial_rotary_factor"] as? Double)
            ?? (ropeFull["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0.25
        let fullTheta = (ropeFull["rope_theta"] as? Double)
            ?? (ropeFull["rope_theta"] as? NSNumber)?.doubleValue ?? 1_000_000.0
        let swaTheta = (ropeSWA["rope_theta"] as? Double)
            ?? (ropeSWA["rope_theta"] as? NSNumber)?.doubleValue ?? 10_000.0
        let kEqV = (tc["attention_k_eq_v"] as? Bool) ?? false
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let act = (tc["hidden_activation"] as? String) ?? "gelu_pytorch_tanh"
        return ArchInfo(
            family: gemma4Family,
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_global_key_value_heads"),
            headDim: try i("head_dim"),
            fullHeadDim: try i("global_head_dim"),
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window"),
            finalLogitSoftcap: try d("final_logit_softcapping"),
            ropeTheta: swaTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("top_k_experts"),
            tieWordEmbeddings: tie,
            attentionKEqV: kEqV,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            layerKinds: layerTypes,
            linearAttention: nil)
    }

    /// `Qwen3_5MoeForConditionalGeneration`. 30 of its 40 layers hold a
    /// fixed-size recurrent state and 10 attend; no layer slides
    /// (`docs/qwen35moe/01-MODEL.md` §1).
    private static func loadQwen35Moe(configPath: String,
                                      textConfig tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        guard !layerTypes.isEmpty else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing layer_types")
        }
        guard layerTypes.count == (try i("num_hidden_layers")) else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "layer_types has \(layerTypes.count) entries, num_hidden_layers "
                    + "says \(try i("num_hidden_layers"))")
        }
        guard layerTypes.allSatisfy({ $0 == "full_attention" || $0 == "linear_attention" }) else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "unexpected layer_types entry")
        }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        guard let theta = (rope["rope_theta"] as? Double)
                ?? (rope["rope_theta"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.rope_theta")
        }
        let prf = (rope["partial_rotary_factor"] as? Double)
            ?? (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue
            ?? (tc["partial_rotary_factor"] as? Double)
            ?? (tc["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0.25
        // One head geometry: this family has no second, "global" attention
        // shape, so the full-attention fields mirror the plain ones.
        let headDim = try i("head_dim")
        let kvHeads = try i("num_key_value_heads")
        let linear = LinearAttention(
            numKeyHeads: try i("linear_num_key_heads"),
            numValueHeads: try i("linear_num_value_heads"),
            keyHeadDim: try i("linear_key_head_dim"),
            valueHeadDim: try i("linear_value_head_dim"),
            convKernelDim: try i("linear_conv_kernel_dim"),
            layerCount: layerTypes.filter { $0 == "linear_attention" }.count)
        guard linear.layerCount > 0 else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "qwen3_5_moe with no linear_attention layer")
        }
        return ArchInfo(
            family: qwen35MoeFamily,
            hiddenSize: try i("hidden_size"),
            // The shared expert is this family's dense FFN.
            intermediateSize: try i("shared_expert_intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: kvHeads,
            numFullKVHeads: kvHeads,
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            // No layer slides, so there is no window to name and no softcap.
            slidingWindow: 0,
            finalLogitSoftcap: 0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: (tc["tie_word_embeddings"] as? Bool) ?? false,
            attentionKEqV: false,
            fullAttentionLayerMask: layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) },
            hiddenActivation: (tc["hidden_act"] as? String) ?? "silu",
            layerKinds: layerTypes,
            linearAttention: linear)
    }
}
