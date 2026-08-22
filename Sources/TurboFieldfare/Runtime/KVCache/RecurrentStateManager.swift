import Foundation
import Darwin
import Metal

/// Fixed-size per-layer state for Qwen3.5-MoE's 30 Gated DeltaNet layers
/// (`docs/qwen35moe/03-DESIGN.md` §3-3).
///
/// `KVCacheManager` cannot hold this. Every field it has — `kSlot`, `stride`,
/// `capacity`, the ring's `position % capacity` — assumes a cache indexed by
/// token. A recurrent layer has no such index: it carries one state that every
/// token folds into, whose size does not depend on the context length at all
/// (`01-MODEL.md` §3-4).
///
/// Two tensors per recurrent layer:
///
///   `S`     FP32 `[Hv, Dv, Dk]` — the recurrence itself, 2 MiB a layer.
///           FP32 is not a choice: the decay gate multiplies it every token, so
///           the rounding accumulates over the whole context
///           (`15-PHASE2-GDN.md`).
///   `conv`  FP16 `[K-1, C]`     — the last three tokens the depthwise causal
///           convolution still needs, oldest row first, matching the
///           reference's `concat(state, qkv)` (`17-PHASE2-KERNELS.md` §1).
///
/// Both live in one buffer each with a per-layer offset, so the runner binds an
/// offset rather than an array element and a snapshot is one contiguous copy.
///
/// **This state cannot be rewound or truncated.** Dropping a token from the
/// middle is not an operation the recurrence has; the only way back to an
/// earlier point is a snapshot taken at that point (`03-DESIGN.md` §3-4). The
/// snapshot machinery that the server and speculative decoding will need is
/// deliberately not here yet: Phase 3 decodes forward and nothing else.
public final class RecurrentStateManager {
    /// Layers that hold a recurrent state, in model order. Index into
    /// `stateOffset(layer:)` with the *model* layer number, not this array's.
    public let recurrentLayers: [Int]
    public let numValueHeads: Int
    public let valueHeadDim: Int
    public let keyHeadDim: Int
    /// Channels the depthwise convolution runs over: `2*Hk*Dk + Hv*Dv`.
    public let convChannels: Int
    /// Taps the convolution keeps from the past: `convKernelDim - 1`.
    public let convHistory: Int

    /// FP32 `[Hv, Dv, Dk]` per recurrent layer.
    ///
    /// `var`, not `let`, for one reason: `adoptShadow()` swaps it with the
    /// speculative copy rather than blitting 61.4 MiB. Every reader binds it at
    /// encode time, so the swap is invisible to them.
    public private(set) var stateBuffer: MTLBuffer
    /// FP16 `[convHistory, convChannels]` per recurrent layer.
    public private(set) var convBuffer: MTLBuffer

    public let stateBytesPerLayer: Int
    public let convBytesPerLayer: Int

    /// Model layer -> slot in the two buffers; -1 for the full-attention layers.
    private let slotOfLayer: [Int]

    public init(device: MTLDevice,
                config: ArchConfig,
                linear: ManifestLinearAttention) throws {
        guard linear.numValueHeads > 0, linear.valueHeadDim > 0,
              linear.numKeyHeads > 0, linear.keyHeadDim > 0,
              linear.convKernelDim > 1 else {
            throw ModelError.indexCorrupt(detail: "linear-attention geometry is degenerate")
        }
        var layers: [Int] = []
        var slots = [Int](repeating: -1, count: config.numLayers)
        for layer in 0..<config.numLayers where config.fullAttentionLayerMask[layer] == 0 {
            slots[layer] = layers.count
            layers.append(layer)
        }
        // Two statements of the same fact. `layerCount` is what a reader sizes
        // this state from without walking the mask, so a disagreement would
        // allocate the wrong number of layers rather than fail.
        guard layers.count == linear.layerCount else {
            throw ModelError.indexCorrupt(
                detail: "\(layers.count) recurrent layers in fullAttentionLayerMask, "
                    + "\(linear.layerCount) in arch.linearAttention")
        }
        self.recurrentLayers = layers
        self.slotOfLayer = slots
        self.numValueHeads = linear.numValueHeads
        self.valueHeadDim = linear.valueHeadDim
        self.keyHeadDim = linear.keyHeadDim
        self.convChannels = 2 * linear.numKeyHeads * linear.keyHeadDim
            + linear.numValueHeads * linear.valueHeadDim
        self.convHistory = linear.convKernelDim - 1

        let stateElements = linear.numValueHeads * linear.valueHeadDim * linear.keyHeadDim
        self.stateBytesPerLayer = stateElements * MemoryLayout<Float>.size
        self.convBytesPerLayer = convHistory * convChannels * MemoryLayout<Float16>.size

        guard let state = device.makeBuffer(length: max(1, stateBytesPerLayer * layers.count),
                                            options: .storageModeShared),
              let conv = device.makeBuffer(length: max(1, convBytesPerLayer * layers.count),
                                           options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        state.label = "qwen.recurrentState"
        conv.label = "qwen.convState"
        self.stateBuffer = state
        self.convBuffer = conv
        reset()
    }

    /// Total bytes this manager allocates — what `ExpertCacheBudget` counts in
    /// place of the K/V it does not allocate for these layers.
    public var totalBytes: UInt64 {
        UInt64(stateBuffer.length) &+ UInt64(convBuffer.length)
    }

    public func holdsState(layer: Int) -> Bool { slotOfLayer[layer] >= 0 }

    // MARK: - Checkpoints (SPEC CACHE-8)

    /// Bytes one checkpoint of this state costs.
    ///
    /// **It does not depend on the context length.** The recurrent state is what
    /// every token folds into, so a checkpoint taken at token 30000 is the same
    /// size as one taken at token 8 — which is what makes CACHE-8 affordable on
    /// this machine at all.
    public var checkpointBytes: Int { stateBuffer.length + convBuffer.length }

    /// A copy of the state as it stands. The buffers are `storageModeShared`,
    /// so this is a memcpy; the caller must not have GPU work in flight over
    /// them (the server captures right after a prefill has been read back).
    public func captureState() -> Data {
        var data = Data(count: checkpointBytes)
        let stateLength = stateBuffer.length
        data.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            memcpy(base, stateBuffer.contents(), stateLength)
            memcpy(base + stateLength, convBuffer.contents(), convBuffer.length)
        }
        return data
    }

    /// Put a captured state back. Refuses a payload that is not this manager's
    /// shape rather than writing a partial state — a half-restored recurrence
    /// produces a continuation that is wrong without being detectably wrong.
    @discardableResult
    public func restoreState(_ data: Data) -> Bool {
        guard data.count == checkpointBytes else { return false }
        let stateLength = stateBuffer.length
        let convLength = convBuffer.length
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            memcpy(stateBuffer.contents(), base, stateLength)
            memcpy(convBuffer.contents(), base + stateLength, convLength)
        }
        return true
    }

    /// Byte offset of `layer`'s recurrent state inside `stateBuffer`.
    public func stateOffset(layer: Int) -> Int {
        let slot = slotOfLayer[layer]
        precondition(slot >= 0, "layer \(layer) is a full-attention layer and holds no state")
        return slot * stateBytesPerLayer
    }

    /// Byte offset of `layer`'s convolution history inside `convBuffer`.
    public func convOffset(layer: Int) -> Int {
        let slot = slotOfLayer[layer]
        precondition(slot >= 0, "layer \(layer) is a full-attention layer and holds no state")
        return slot * convBytesPerLayer
    }

    /// Back to the start of a conversation. Both tensors are zero there: the
    /// recurrence starts empty and the convolution's past is padding
    /// (`Scripts/qwen35/reference_forward.py` `State`).
    ///
    /// This is a real zeroing, not `KVCacheManager.reset`'s `MADV_DONTNEED`.
    /// Those buffers are only read below a cursor that also went to zero; these
    /// are read whole on the very next token.
    public func reset() {
        memset(stateBuffer.contents(), 0, stateBuffer.length)
        memset(convBuffer.contents(), 0, convBuffer.length)
    }

    // MARK: - The second copy a speculative row writes into

    /// Where the **speculative** row's state goes.
    ///
    /// A width-2 verify pass runs two rows: row 0 is a token the body has
    /// already committed to, row 1 is the head's guess. If the guess is wrong,
    /// what has to survive is the state **after row 0** — which is neither the
    /// state before the pass nor the state after it. A snapshot taken before
    /// the pass restores too much and loses row 0
    /// (`docs/qwen35moe/36-MTP-DECODE.md` §3-2, where exactly that produced a
    /// degenerate loop).
    ///
    /// So the recurrence is split instead, and nothing is copied
    /// (`33-MTP-ACCEPTANCE.md` §3-6): row 0 writes in place, row 1 writes here,
    /// and acceptance is `adoptShadow()` — two pointers. `qwen_delta_rule` and
    /// `qwen_delta_qkv_prepare` have taken `stateIn` and `stateOut` as separate
    /// arguments since Phase 2, so no kernel changes.
    public private(set) var shadowStateBuffer: MTLBuffer?
    public private(set) var shadowConvBuffer: MTLBuffer?

    /// Bytes the shadow costs once it exists — one more copy of the 61.4 MiB.
    public var shadowBytes: UInt64 {
        guard let shadowStateBuffer, let shadowConvBuffer else { return 0 }
        return UInt64(shadowStateBuffer.length) &+ UInt64(shadowConvBuffer.length)
    }

    /// Allocate the shadow. Called once, by the first speculative step; a run
    /// that never speculates never pays the 61.4 MiB.
    public func ensureShadow(device: MTLDevice) throws {
        if shadowStateBuffer != nil { return }
        guard let state = device.makeBuffer(length: stateBuffer.length,
                                            options: .storageModeShared),
              let conv = device.makeBuffer(length: convBuffer.length,
                                           options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        state.label = "qwen.recurrentState.shadow"
        conv.label = "qwen.convState.shadow"
        // Zeroed for the same reason `reset()` zeroes: a shadow that is read
        // before it is written would be read whole.
        memset(state.contents(), 0, state.length)
        memset(conv.contents(), 0, conv.length)
        shadowStateBuffer = state
        shadowConvBuffer = conv
    }

    /// The speculative row was accepted: its state becomes the state.
    ///
    /// A pointer swap, not a copy. What lands in the shadow is the state after
    /// row 0, which the next pass overwrites before reading.
    public func adoptShadow() {
        guard let shadowStateBuffer, let shadowConvBuffer else { return }
        let oldState = stateBuffer
        let oldConv = convBuffer
        stateBuffer = shadowStateBuffer
        convBuffer = shadowConvBuffer
        self.shadowStateBuffer = oldState
        self.shadowConvBuffer = oldConv
    }
}
