import Foundation
import Metal

/// Resident-tensor accessors in Qwen3.5-MoE's spelling.
///
/// `Model`'s own accessors carry the names the Gemma 4 repack writes. Some of
/// them happen to be the names this family uses too — `input_layernorm`,
/// `self_attn.q_proj`, `model.norm` — and those are reused rather than
/// duplicated here. What follows is only what differs, which is three groups:
///
/// - the MoE block keeps the upstream spelling (`mlp.gate` for the router,
///   `mlp.shared_expert.*` for the dense branch) instead of Gemma's flattened
///   `router.proj` / `mlp.gate_proj`, and adds `shared_expert_gate`
/// - the word embedding is not tied, so `lm_head` is a tensor of its own
/// - 30 of the 40 layers hold no K/V at all and reach the recurrence through
///   `linear_attn.*` (`docs/qwen35moe/01-MODEL.md` §3-2)
///
/// Nothing here validates: `validateRuntimeSchema` already ran over every one
/// of these names at load, with the geometry the manifest states
/// (`docs/qwen35moe/18-MIXED-BITS.md` §4).
extension Model {

    // MARK: - Head and MoE block

    /// Not tied to the embedding on this family, and 8-bit on the production
    /// checkpoint (`docs/qwen35moe/19-LM-HEAD-INT8.md`).
    public var qwenLMHead: TensorView {
        get throws { try resident(name: "language_model.lm_head.weight") }
    }

    /// The router. Upstream calls it `mlp.gate`, which reads like a projection
    /// of the dense branch and is not one.
    public func qwenRouter(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.gate.weight")
    }

    /// `sigmoid(w . x)` over the whole shared-expert output — one row, no
    /// Gemma counterpart (`docs/qwen35moe/03-DESIGN.md` §2-4).
    public func qwenSharedExpertGate(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.shared_expert_gate.weight")
    }

    public func qwenSharedExpertGateProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.shared_expert.gate_proj.weight")
    }

    public func qwenSharedExpertUp(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.shared_expert.up_proj.weight")
    }

    public func qwenSharedExpertDown(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.shared_expert.down_proj.weight")
    }

    // MARK: - Linear-attention (Gated DeltaNet) layers

    /// `[2*Hk*Dk + Hv*Dv, D]` — q, k and v in one projection, split after the
    /// depthwise convolution rather than before it.
    public func qwenInProjQKV(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_qkv.weight")
    }

    /// `[Hv*Dv, D]` — the gate `RMSNormGated` multiplies by, through SiLU.
    public func qwenInProjZ(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_z.weight")
    }

    /// `[Hv, D]` — feeds the decay gate through `softplus` and a second `exp`.
    public func qwenInProjA(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_a.weight")
    }

    /// `[Hv, D]` — feeds `beta` through `sigmoid`.
    public func qwenInProjB(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_b.weight")
    }

    public func qwenOutProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.out_proj.weight")
    }

    /// Depthwise causal kernel, BF16, carried by MLX as `[channels, taps, 1]`
    /// — the axis order the runtime reads and the one negative control in
    /// `docs/qwen35moe/17-PHASE2-KERNELS.md` §2 swaps.
    public func qwenConv1D(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.conv1d.weight")
    }

    public func qwenALog(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.A_log")
    }

    public func qwenDtBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.dt_bias")
    }

    /// `RMSNormGated`'s weight — **not** a `1 + w` tensor, unlike every other
    /// norm in this checkpoint (`docs/qwen35moe/12-OQ4E-G64-AUDIT.md` §2).
    public func qwenDeltaNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.norm.weight")
    }

    // MARK: - Geometry

    /// The recurrent layers' head geometry, which `ArchConfig` cannot carry:
    /// its head fields describe the other 10 layers.
    public var qwenLinearAttention: ManifestLinearAttention? {
        manifest.arch.linearAttention
    }

    /// `arch.family` as the manifest states it. Absent means Gemma 4, which is
    /// the family this format was written for.
    public var modelFamily: String { manifest.arch.family ?? "gemma4" }

    /// The family an install *declares*, read from `manifest.json` alone.
    ///
    /// For the one decision that has to be made before loading: which
    /// tokenizer and which runner an install needs. It is a hint, not a
    /// verification — the manifest is not authenticated here, and nothing is
    /// mapped. `Model.load` still runs every gate afterwards, including
    /// `expecting:`, so a manifest that lies about its family fails there
    /// instead of here. Unreadable or unparsable means "the family this format
    /// was written for", which is what an old install with no `family` field
    /// also means.
    public static func declaredFamily(at directoryURL: URL) -> String {
        struct Peek: Decodable {
            struct Arch: Decodable { let family: String? }
            let arch: Arch?
        }
        let url = directoryURL.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: url, options: [.mappedIfSafe]),
              let peek = try? JSONDecoder().decode(Peek.self, from: data),
              let family = peek.arch?.family, !family.isEmpty
        else { return "gemma4" }
        return family
    }
}
