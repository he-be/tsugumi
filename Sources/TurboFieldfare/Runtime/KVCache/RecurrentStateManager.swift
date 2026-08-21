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
    public let stateBuffer: MTLBuffer
    /// FP16 `[convHistory, convChannels]` per recurrent layer.
    public let convBuffer: MTLBuffer

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
}
