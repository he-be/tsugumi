import Foundation
import Metal
import Tsugumi
import TsugumiValidationSupport

// MARK: - Prefill dense int4 QMM, off the prefill timeline
//
// `--dense-qmm-bench` (docs/m6-prefill/31). Times one `MPPPrefillInt4QMM`
// dispatch per kernel on the prefill projection and shared-MLP shapes at a full
// prompt chunk (M = 2048), symmetric int4, synthetic resident weights. Each
// case reports the median GPU time of one dispatch in its own command buffer
// and the rate it implies. MLX's `affine_qmm_t_nax` runs these shapes at
// 16.4-17.5 TFLOP/s (docs/m6-prefill/31 §1).

func runDenseQMMBench(groupSize: Int, iterations: Int) throws {
    affineScheme = .sym
    let context = try makeContext(groupSize: groupSize)
    let device = context.device
    var rng = SplitMix64(seed: 0x716d6d)
    let m = 2048
    let shapes: [(String, Int, Int)] = [
        ("shared.gate", 2112, 2816), ("shared.down", 2816, 2112),
        ("q.swa", 4096, 2816), ("k.swa", 2048, 2816), ("o.swa", 2816, 4096),
        ("q.full", 8192, 2816), ("o.full", 2816, 8192),
    ]
    let maxN = shapes.map(\.1).max()!, maxK = shapes.map(\.2).max()!
    let wBytes = (0..<(maxN * maxK / 2)).map { _ in UInt8(truncatingIfNeeded: rng.next()) }
    let sHalves = [UInt16](repeating: 0x3C24, count: maxN * maxK / groupSize)   // bf16 0.01
    let xHalves = (0..<(m * maxK)).map { _ in Float16(Float(rng.next() % 2001) / 1000 - 1) }
    guard let w = device.makeBuffer(bytes: wBytes, length: wBytes.count, options: .storageModeShared),
          let s = device.makeBuffer(bytes: sHalves, length: sHalves.count * 2, options: .storageModeShared),
          let x = device.makeBuffer(bytes: xHalves, length: xHalves.count * 2, options: .storageModeShared),
          let y = device.makeBuffer(length: m * maxN * 2, options: .storageModePrivate) else {
        fatalError("dense qmm bench fixture failed")
    }
    let kernels: [MPPPrefillInt4QMM.Kernel] = [.tile64, .nax]
    let qmms = kernels.map { MPPPrefillInt4QMM(context: context, kernel: $0) }
    print("dense qmm bench: M \(m), g\(groupSize) sym, median of \(iterations) (after 2 warm), 1 dispatch per command buffer")
    print("shape\tN\tK\tkernel\tgpu_us\tTFLOP/s")
    for (name, n, k) in shapes {
        let flops = 2.0 * Double(m) * Double(n) * Double(k)
        for (kernel, qmm) in zip(kernels, qmms) {
            guard qmm.isAvailable else {
                print("\(name)\t\(n)\t\(k)\t\(kernel.rawValue)\tSKIP")
                continue
            }
            var samples: [Double] = []
            for iteration in 0..<(iterations + 2) {
                guard let cmd = context.queue.makeCommandBuffer() else { fatalError() }
                let path = qmm.encode(commandBuffer: cmd, weights: w, scales: s, biases: s,
                                      x: x, y: y, m: m, n: n, k: k)
                guard path != .unavailable else { fatalError("\(name) \(kernel): unavailable") }
                waitAndCheck(cmd, "dense qmm bench \(name)")
                if iteration >= 2 { samples.append(cmd.gpuEndTime - cmd.gpuStartTime) }
            }
            samples.sort()
            let t = samples[samples.count / 2]
            print(String(format: "%@\t%d\t%d\t%@\t%.1f\t%.2f",
                         name, n, k, kernel.rawValue, t * 1e6, flops / t / 1e12))
        }
    }
}

// MARK: - Prefill BF16 router, off the prefill timeline
//
// `--router-bench` (docs/m6-prefill/33). The level-2 `post.router` segment ends
// with a command buffer that runs beside the shared expert's, so its seconds
// are not the router's. This times the row kernel and the GEMM alone at a full
// prompt chunk (M = 2048), D 2816, 128 experts, top-8.

func runRouterBench(groupSize: Int, iterations: Int) throws {
    let context = try makeContext(groupSize: groupSize)
    let device = context.device
    let router = try PrefillRouter(context: context, routerWeightBits: 16)
    var rng = SplitMix64(seed: 0x726f7574)
    let m = 2048, d = 2816, experts = 128, topK = 8
    let wBits = (0..<(experts * d)).map { _ in
        Quantization.bf16Bits(Float(rng.next() % 2001) / 10000 - 0.1)
    }
    let esBits = [UInt16](repeating: Quantization.bf16Bits(1), count: d)
    let pesBits = [UInt16](repeating: Quantization.bf16Bits(1), count: experts)
    let xHalves = (0..<(m * d)).map { _ in Float16(Float(rng.next() % 2001) / 1000 - 1) }
    guard let w = device.makeBuffer(bytes: wBits, length: wBits.count * 2, options: .storageModeShared),
          let es = device.makeBuffer(bytes: esBits, length: esBits.count * 2, options: .storageModeShared),
          let pes = device.makeBuffer(bytes: pesBits, length: pesBits.count * 2, options: .storageModeShared),
          let x = device.makeBuffer(bytes: xHalves, length: xHalves.count * 2, options: .storageModeShared),
          let ids = device.makeBuffer(length: m * topK * 4, options: .storageModeShared),
          let wts = device.makeBuffer(length: m * topK * 2, options: .storageModeShared) else {
        fatalError("router bench fixture failed")
    }
    let flops = 2.0 * Double(m) * Double(experts) * Double(d)
    print("router bench: M \(m), D \(d), \(experts) experts, top-\(topK), median of \(iterations) (after 2 warm), 1 dispatch per command buffer")
    print("kernel\tgpu_us\tTFLOP/s")
    let kinds = ["row", "gemm"] + (router.hasTensorOpsGEMM ? ["nax"] : [])
    for kind in kinds {
        let gemm = kind != "row"
        var samples: [Double] = []
        for iteration in 0..<(iterations + 2) {
            guard let cmd = context.queue.makeCommandBuffer() else { fatalError() }
            if gemm {
                router.encodeGemma4BF16GEMM(commandBuffer: cmd, weights: w, hidden: x,
                                            effectiveScale: es, perExpertScale: pes,
                                            outIndices: ids, outWeights: wts,
                                            queryCount: UInt32(m), numExperts: UInt32(experts),
                                            d: UInt32(d), topK: UInt32(topK),
                                            hiddenStrideElements: UInt32(d),
                                            tensorOps: kind == "nax")
            } else {
                router.encodeGemma4BF16Block(commandBuffer: cmd, weights: w, hidden: x,
                                             effectiveScale: es, perExpertScale: pes,
                                             outIndices: ids, outWeights: wts,
                                             queryCount: UInt32(m), numExperts: UInt32(experts),
                                             d: UInt32(d), topK: UInt32(topK),
                                             hiddenStrideElements: UInt32(d))
            }
            waitAndCheck(cmd, "router bench")
            if iteration >= 2 { samples.append(cmd.gpuEndTime - cmd.gpuStartTime) }
        }
        samples.sort()
        let t = samples[samples.count / 2]
        print(String(format: "%@\t%.1f\t%.2f", kind, t * 1e6, flops / t / 1e12))
    }
}
