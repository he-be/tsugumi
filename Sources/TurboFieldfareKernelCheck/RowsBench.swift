import Foundation
import Metal
import TurboFieldfare
import TurboFieldfareValidationSupport

// MARK: - k-row dense GEMV throughput
//
// `docs/mtp/19-M4.7-RESULTS.md` §5: with the expert cache at 80 slots the
// verify block is GPU-bound, and the stages that grew fastest with k are the
// ones whose *weights do not depend on k* — the shared expert (4 → 9 → 16 ms)
// and the LM head (3 → 7 → 13 ms). Both run `dequant_int4_gemv_rows_simd`,
// which already reads W once per block, so the growth is arithmetic, not bytes.
//
// This bench isolates that kernel at the production shapes so the k-scaling can
// be read without loading the 26B target or touching expert I/O: one dispatch,
// no cache, no host profile. `bytes` is what the kernel must read (packed
// nibbles + BF16 scale/bias), so `GB/s` says how far the shape is from the
// machine's ~150 GB/s roof, and `Δ/row` says what one extra verify row costs.

private struct RowsShape {
    let name: String
    /// Output rows (the weight matrix's M).
    let m: Int
    /// Reduction length (the weight matrix's N).
    let n: Int
    /// Times this shape runs in one verify block, for the per-block column.
    let perBlock: Int
}

/// Bytes `dequant_int4_gemv_rows_simd` reads for one (m, n) weight tensor.
private func rowsWeightBytes(m: Int, n: Int, groupSize: Int) -> Double {
    let packed = Double(m) * Double(n) / 2
    let affine = Double(m) * Double(n / groupSize) * 2 * 2  // BF16 scale + bias
    return packed + affine
}

/// A weight tensor of the right size and layout, filled with a cheap pattern.
///
/// Timing does not depend on the nibble values, and quantizing 738 M random
/// floats on the CPU for the head shape would dominate the run; the scales are
/// kept small and positive so the accumulator stays finite and the GPU never
/// takes a denormal/NaN path that the real weights would not.
private func syntheticInt4Tensor(device: MTLDevice, m: Int, n: Int, groupSize: Int)
    -> (weights: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer) {
    let packedBytes = m * (n / 2)
    let groups = m * (n / groupSize)
    guard let weights = device.makeBuffer(length: packedBytes, options: .storageModeShared),
          let scales = device.makeBuffer(length: groups * 2, options: .storageModeShared),
          let biases = device.makeBuffer(length: groups * 2, options: .storageModeShared) else {
        fatalError("buffer allocation failed for m=\(m) n=\(n)")
    }
    let bytes = weights.contents().bindMemory(to: UInt8.self, capacity: packedBytes)
    var state: UInt64 = 0x9E3779B97F4A7C15
    for index in 0..<packedBytes {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        bytes[index] = UInt8(truncatingIfNeeded: state >> 33)
    }
    // BF16 0x3C00 == 0.0078125, 0x3B00 == 0.001953125.
    let scaleWords = scales.contents().bindMemory(to: UInt16.self, capacity: groups)
    let biasWords = biases.contents().bindMemory(to: UInt16.self, capacity: groups)
    for index in 0..<groups {
        scaleWords[index] = 0x3C00
        biasWords[index] = 0x3B00
    }
    return (weights, scales, biases)
}

/// One row's worth of activation, repeated for the block's `maxRows`.
private func syntheticActivation(device: MTLDevice, rows: Int, n: Int) -> MTLBuffer {
    let halves = (0..<(rows * n)).map { index in
        Float16(Float((index % 61)) * 0.01 - 0.3)
    }
    guard let buffer = Fp16Buffer.make(device, halves: halves) else {
        fatalError("activation allocation failed")
    }
    return buffer
}

/// One benchmarked encoding of the k-row GEMV.
private struct RowsVariant {
    let name: String
    /// nil = the shipped `dequant_int4_gemv_rows_simd`.
    let rowsPerSIMDGroup: Int?
    let specializeT: Bool
}

private func encodeVariant(_ variant: RowsVariant,
                           kernel: DequantInt4GEMV,
                           cmd: MTLCommandBuffer,
                           tensor: (weights: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer),
                           x: MTLBuffer, y: MTLBuffer,
                           t: Int, m: Int, n: Int) throws {
    guard let rowsPerSIMDGroup = variant.rowsPerSIMDGroup else {
        kernel.encodeRows(commandBuffer: cmd,
                          weights: tensor.weights, scales: tensor.scales, biases: tensor.biases,
                          x: x, xStrideElements: n,
                          y: y, yStrideElements: m,
                          t: t, m: UInt32(m), n: UInt32(n))
        return
    }
    try kernel.encodeRowsWide(commandBuffer: cmd,
                              weights: tensor.weights, scales: tensor.scales,
                              biases: tensor.biases,
                              x: x, xStrideElements: n,
                              y: y, yStrideElements: m,
                              t: t, m: UInt32(m), n: UInt32(n),
                              rowsPerSIMDGroup: rowsPerSIMDGroup,
                              specializeT: variant.specializeT)
}

/// Every wide variant against the shipped k-row kernel, bit for bit.
///
/// The point of `dequant_int4_gemv_rows_wide_simd` is that it reorders nothing:
/// the affine sum it reads from `xsum` is the same eight adds in the same
/// order, and the dot product is untouched. So the bar here is equality, not a
/// tolerance — anything else means the reordering claim in the kernel comment
/// is wrong.
func runRowsVerify(groupSize: Int) throws -> [CaseResult] {
    let context = try makeContext(groupSize: groupSize)
    let kernel = try DequantInt4GEMV(context: context)
    var cases: [CaseResult] = []
    // 2816 is a whole number of 256-element blocks; 2112 and 704 leave a
    // scalar tail, and 704 leaves one that is shorter than a block.
    let shapes = [(m: 512, n: 2816), (m: 264, n: 2112), (m: 130, n: 704)]
    let variants = [
        RowsVariant(name: "wide r1", rowsPerSIMDGroup: 1, specializeT: false),
        RowsVariant(name: "wide r1 T", rowsPerSIMDGroup: 1, specializeT: true),
        RowsVariant(name: "wide r2 T", rowsPerSIMDGroup: 2, specializeT: true),
        RowsVariant(name: "wide r4 T", rowsPerSIMDGroup: 4, specializeT: true),
    ]
    for shape in shapes {
        var rng = SeedTree(0x4707).key("rows-wide-\(shape.m)-\(shape.n)-\(groupSize)")
        let rows = quantizedRows(count: shape.m, n: shape.n, groupSize: groupSize, rng: &rng)
        let (packed, scales, biases) = packRows(rows)
        let xHalves = (0..<(DequantInt4GEMV.maxRows * shape.n)).map { _ in
            Float16(rng.uniform(-1.0, 1.0))
        }
        guard let wBuf = context.device.makeBuffer(bytes: packed, length: packed.count,
                                                   options: .storageModeShared),
              let sBuf = context.device.makeBuffer(
                bytes: scales, length: scales.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared),
              let bBuf = context.device.makeBuffer(
                bytes: biases, length: biases.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared),
              let xBuf = Fp16Buffer.make(context.device, halves: xHalves),
              let yRef = Fp16Buffer.make(context.device,
                                         count: DequantInt4GEMV.maxRows * shape.m),
              let yGot = Fp16Buffer.make(context.device,
                                         count: DequantInt4GEMV.maxRows * shape.m) else {
            fatalError("buffer allocation failed")
        }
        let tensor = (weights: wBuf, scales: sBuf, biases: bBuf)
        for t in [1, 2, 4, 8] {
            guard let refCB = context.queue.makeCommandBuffer() else {
                fatalError("command buffer allocation failed")
            }
            kernel.encodeRows(commandBuffer: refCB,
                              weights: wBuf, scales: sBuf, biases: bBuf,
                              x: xBuf, xStrideElements: shape.n,
                              y: yRef, yStrideElements: shape.m,
                              t: t, m: UInt32(shape.m), n: UInt32(shape.n))
            waitAndCheck(refCB, "rows-verify reference")
            let reference = Fp16Buffer.read(yRef, count: t * shape.m)
            for variant in variants {
                guard let cmd = context.queue.makeCommandBuffer() else {
                    fatalError("command buffer allocation failed")
                }
                try encodeVariant(variant, kernel: kernel, cmd: cmd, tensor: tensor,
                                  x: xBuf, y: yGot, t: t, m: shape.m, n: shape.n)
                waitAndCheck(cmd, "rows-verify \(variant.name)")
                let actual = Fp16Buffer.read(yGot, count: t * shape.m)
                let mismatches = zip(actual, reference).filter { $0 != $1 }.count
                cases.append(result("\(variant.name) m=\(shape.m) n=\(shape.n) t=\(t)",
                                    groupSize: groupSize,
                                    rel: mismatches == 0 ? 0 : 1,
                                    tolerance: 0,
                                    detail: "mismatched elements \(mismatches)/\(t * shape.m)"))
            }
        }
    }
    return cases
}

/// `--rows-bench`: the k-row dense GEMV at the shapes a verify block runs.
func runRowsBench(groupSize: Int, iterations: Int) throws {
    let context = try makeContext(groupSize: groupSize)
    let kernel = try DequantInt4GEMV(context: context)
    let shapes = [
        // The LM head, tied to the embedding table: the single largest read in
        // the block, and the whole of the `head` stage.
        RowsShape(name: "head 262144x2816", m: 262144, n: 2816, perBlock: 1),
        // Shared expert gate and up (F=2112 from D=2816), then down.
        RowsShape(name: "shared.gate 2112x2816", m: 2112, n: 2816, perBlock: 60),
        RowsShape(name: "shared.down 2816x2112", m: 2816, n: 2112, perBlock: 30),
        // Q/K/V and O, the dense half of the `attn` stage.
        RowsShape(name: "attn.qkv 8192x2816", m: 8192, n: 2816, perBlock: 30),
        RowsShape(name: "attn.o 2816x4096", m: 2816, n: 4096, perBlock: 30),
    ]
    let variants = [
        RowsVariant(name: "rows (shipped)", rowsPerSIMDGroup: nil, specializeT: false),
        RowsVariant(name: "wide r1 T", rowsPerSIMDGroup: 1, specializeT: true),
        RowsVariant(name: "wide r2 T", rowsPerSIMDGroup: 2, specializeT: true),
        RowsVariant(name: "wide r4 T", rowsPerSIMDGroup: 4, specializeT: true),
    ]
    let rowCounts = [1, 2, 3, 4, 5, 6, 7, 8]

    print("=== k-row dense GEMV (group \(groupSize), \(iterations) iterations) ===")
    print("ms per dispatch; ms/blk is the shape's whole contribution to one verify block")
    for shape in shapes {
        let tensor = syntheticInt4Tensor(device: context.device,
                                         m: shape.m, n: shape.n, groupSize: groupSize)
        let x = syntheticActivation(device: context.device,
                                    rows: DequantInt4GEMV.maxRows, n: shape.n)
        guard let y = Fp16Buffer.make(context.device,
                                      count: DequantInt4GEMV.maxRows * shape.m) else {
            fatalError("output allocation failed")
        }
        let bytes = rowsWeightBytes(m: shape.m, n: shape.n, groupSize: groupSize)
        print("")
        print("  \(shape.name)  x\(shape.perBlock)/block  "
                + String(format: "%.1f MB", bytes / 1e6))
        var header = "    variant          "
        for t in rowCounts { header += String(format: "     t=%d", t) }
        header += "   | ms/blk t=4"
        print(header)
        for variant in variants {
            var line = String(format: "    %-16@", variant.name as NSString)
            var atFour = 0.0
            for t in rowCounts {
                let seconds = try gpuSecondsThrowing(
                    context: context, iterations: iterations,
                    label: "\(variant.name) \(shape.name) t=\(t)") { cmd in
                    try encodeVariant(variant, kernel: kernel, cmd: cmd, tensor: tensor,
                                      x: x, y: y, t: t, m: shape.m, n: shape.n)
                }
                if t == 4 { atFour = seconds }
                line += String(format: " %8.3f", seconds * 1e3)
            }
            line += String(format: "   | %8.3f", atFour * 1e3 * Double(shape.perBlock))
            print(line)
        }
    }
}

/// `gpuSeconds` for an encoder that can throw (pipeline construction).
func gpuSecondsThrowing(context: MetalContext, iterations: Int,
                        label: String,
                        encode: (MTLCommandBuffer) throws -> Void) throws -> Double {
    guard let warmup = context.queue.makeCommandBuffer() else {
        fatalError("command buffer allocation failed")
    }
    try encode(warmup)
    waitAndCheck(warmup, "\(label) warmup")

    guard let cmd = context.queue.makeCommandBuffer() else {
        fatalError("command buffer allocation failed")
    }
    for _ in 0..<iterations { try encode(cmd) }
    waitAndCheck(cmd, label)
    return (cmd.gpuEndTime - cmd.gpuStartTime) / Double(iterations)
}
