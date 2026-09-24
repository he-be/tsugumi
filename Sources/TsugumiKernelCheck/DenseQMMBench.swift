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
