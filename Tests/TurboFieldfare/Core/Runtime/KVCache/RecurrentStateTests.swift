import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// The structural half of `docs/qwen35moe/04-PHASES.md` Phase 3: where a
/// recurrent layer's state lives, and what the K/V cache does about the 30
/// layers that no longer have one (`03-DESIGN.md` §3-3).
///
/// The numeric half is `--qwen-decode`, which needs the 19 GB install and
/// therefore cannot run here.
@Suite struct RecurrentStateTests {

    private let config = ArchConfig.ornith1_5_35B_A3B

    /// The production geometry, as `arch.linearAttention` states it.
    private let linear = ManifestLinearAttention(numKeyHeads: 16,
                                                 numValueHeads: 32,
                                                 keyHeadDim: 128,
                                                 valueHeadDim: 128,
                                                 convKernelDim: 4,
                                                 layerCount: 30)

    private var recurrentLayers: Set<Int> {
        Set((0..<config.numLayers).filter { config.fullAttentionLayerMask[$0] == 0 })
    }

    private func makeState() throws -> (MetalContext, RecurrentStateManager) {
        let ctx = try MetalContext()
        return (ctx, try RecurrentStateManager(device: ctx.device,
                                               config: config,
                                               linear: linear))
    }

    @Test func statesAreSizedByGeometryNotByContext() throws {
        let (_, state) = try makeState()
        #expect(state.recurrentLayers.count == 30)
        // [Hv, Dv, Dk] FP32 = 32 * 128 * 128 * 4 = 2 MiB a layer.
        #expect(state.stateBytesPerLayer == 32 * 128 * 128 * 4)
        // [K-1, C] FP16 with C = 2*16*128 + 32*128.
        #expect(state.convChannels == 8192)
        #expect(state.convHistory == 3)
        #expect(state.convBytesPerLayer == 3 * 8192 * 2)
        // 60 MiB + 48 KiB x 30 — the whole recurrent state of the model, and it
        // does not grow with the context (`01-MODEL.md` §3-4).
        let perLayer: Int = state.stateBytesPerLayer + state.convBytesPerLayer
        let expected: UInt64 = UInt64(30 * perLayer)
        #expect(state.totalBytes == expected)
    }

    @Test func offsetsPartitionTheBuffersWithoutOverlap() throws {
        let (_, state) = try makeState()
        var seenState = Set<Int>()
        var seenConv = Set<Int>()
        for layer in state.recurrentLayers {
            #expect(state.holdsState(layer: layer))
            #expect(seenState.insert(state.stateOffset(layer: layer)).inserted)
            #expect(seenConv.insert(state.convOffset(layer: layer)).inserted)
        }
        // Every full-attention layer is absent, and slot numbering is dense:
        // the highest offset plus one layer's worth is the whole buffer.
        for layer in 0..<config.numLayers where config.fullAttentionLayerMask[layer] != 0 {
            #expect(!state.holdsState(layer: layer))
        }
        #expect(seenState.max()! + state.stateBytesPerLayer == state.stateBuffer.length)
        #expect(seenConv.max()! + state.convBytesPerLayer == state.convBuffer.length)
    }

    @Test func resetZeroesBothTensors() throws {
        let (_, state) = try makeState()
        let floats = state.stateBuffer.contents()
            .assumingMemoryBound(to: Float.self)
        let halves = state.convBuffer.contents()
            .assumingMemoryBound(to: UInt16.self)
        floats[0] = 3.5
        floats[state.stateBuffer.length / 4 - 1] = -1.25
        halves[7] = 0x3C00
        state.reset()
        #expect(floats[0] == 0)
        #expect(floats[state.stateBuffer.length / 4 - 1] == 0)
        #expect(halves[7] == 0)
    }

    @Test func aGeometryThatDisagreesWithTheMaskIsRefused() throws {
        // `layerCount` is what a reader budgets from without walking the mask.
        // Two statements of the same fact must not be allowed to differ: the
        // wrong one here allocates 29 layers of state and the 30th layer then
        // reads whatever follows the buffer.
        let ctx = try MetalContext()
        let wrong = ManifestLinearAttention(numKeyHeads: 16, numValueHeads: 32,
                                            keyHeadDim: 128, valueHeadDim: 128,
                                            convKernelDim: 4, layerCount: 29)
        #expect(throws: ModelError.self) {
            _ = try RecurrentStateManager(device: ctx.device, config: config, linear: wrong)
        }
    }

    @Test func aDegenerateGeometryIsRefused() throws {
        // A kernel width of 1 would make the convolution history empty, and
        // `[0, C]` is a buffer no allocation catches.
        let ctx = try MetalContext()
        let flat = ManifestLinearAttention(numKeyHeads: 16, numValueHeads: 32,
                                           keyHeadDim: 128, valueHeadDim: 128,
                                           convKernelDim: 1, layerCount: 30)
        #expect(throws: ModelError.self) {
            _ = try RecurrentStateManager(device: ctx.device, config: config, linear: flat)
        }
    }

    // MARK: - The K/V cache's side

    @Test func recurrentLayersAllocateNoKeyValueStorage() throws {
        let ctx = try MetalContext()
        let maxContext = 4096
        let kv = try KVCacheManager(device: ctx.device,
                                    config: config,
                                    maxContext: maxContext,
                                    fp16RingEnabled: false,
                                    recurrentLayers: recurrentLayers)
        var bytes = 0
        for layer in 0..<config.numLayers {
            if config.fullAttentionLayerMask[layer] == 0 {
                #expect(kv.layerKind(layer) == .linear)
                #expect(kv.stride(layer: layer) == 0)
                // One byte apiece, because Metal will not make a zero-length
                // buffer; the point is that it is not `maxContext * stride`.
                #expect(kv.bufferLength(layer: layer) == 0)
            } else {
                #expect(kv.layerKind(layer) == .full)
                #expect(kv.stride(layer: layer)
                        == config.numFullKVHeads * config.fullHeadDim * 2)
                bytes += kv.bufferLength(layer: layer) * 2
            }
        }
        // 10 layers, not 40: sizing the recurrent ones as if they cached would
        // ask for 3.9x these bytes and no kernel would ever read them.
        let fullStride: Int = config.numFullKVHeads * config.fullHeadDim * 2
        #expect(bytes == 10 * maxContext * fullStride * 2)
    }

    @Test func aRecurrentModelCannotBeRewound() throws {
        let ctx = try MetalContext()
        let kv = try KVCacheManager(device: ctx.device,
                                    config: config,
                                    maxContext: 128,
                                    fp16RingEnabled: false,
                                    recurrentLayers: recurrentLayers)
        kv.advance(by: 16)
        // Linear storage would say "all 16": every K/V row is still there. The
        // recurrent state is what makes the answer 0, and it is not in this
        // object to look at (`03-DESIGN.md` §3-4).
        #expect(kv.maximumSafeRewind == 0)
    }

    // MARK: - What the memory guard is told

    @Test func theBudgetChargesTheStateInsteadOfKeyValue() throws {
        // The toy install is the only Qwen-family model this target can open.
        // Its shape is the real one in miniature: 3 recurrent layers, 1 full.
        let directory = try QwenToyInstall.write()
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: directory,
                                   device: device,
                                   expecting: QwenToyInstall.arch)
        let arch = QwenToyInstall.arch
        let maxContext = 1024
        let kv = model.kvCacheByteEstimate(maxContext: maxContext, fp16RingEnabled: false)
        let fullStride: Int = arch.numFullKVHeads * arch.fullHeadDim * 2
        // One layer of the four, K and V.
        #expect(kv == UInt64(maxContext * fullStride * 2))

        let stateBytes: Int = QwenToyInstall.numValueHeads * QwenToyInstall.valueHeadDim
            * QwenToyInstall.keyHeadDim * 4
        let convBytes: Int = (QwenToyInstall.convKernelDim - 1) * QwenToyInstall.qkvWidth * 2
        #expect(model.recurrentStateByteEstimate == UInt64(3 * (stateBytes + convBytes)))

        // The two numbers together must be what the manager actually asks the
        // device for, or the guard is protecting a different program.
        let context = try MetalContext()
        let state = try RecurrentStateManager(
            device: context.device,
            config: arch,
            linear: try #require(model.qwenLinearAttention))
        #expect(state.totalBytes == model.recurrentStateByteEstimate)
    }

    @Test func gemmaKeepsItsWholeRewind() throws {
        // The same property, on the family that has no recurrent layer: adding
        // the third kind must not have narrowed Gemma's answer.
        let ctx = try MetalContext()
        let gemma = ArchConfig.gemma4_26B_A4B
        let kv = try KVCacheManager(device: ctx.device,
                                    config: gemma,
                                    maxContext: 128,
                                    fp16RingEnabled: false)
        kv.advance(by: 16)
        #expect(kv.maximumSafeRewind == 16)
    }
}
