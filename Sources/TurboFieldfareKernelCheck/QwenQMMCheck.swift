import Foundation
import Metal
import TurboFieldfare

// MARK: - INT8 の QMM の検査 (`docs/qwen35moe/04-PHASES.md` Phase 4)
//
// `QwenHeadCheck.swift` と同じ立て方: チェックポイントも fixtures も開かず、
// 実物と同じ形の合成入力に対して GPU / CPU float32 (床) / CPU double (真値) を
// 走らせる。**このカーネルだけ経路が 2 本ある**ので、見るものが 1 つ増える:
//
//   scalar   1 スレッドが (token, row) を持ち、重みは FP32 のまま積む
//   tiled    64x64 タイル、逆量子化した重みを FP16 で threadgroup に置く
//
// 真値に対しては両方を採点し、**互いにも突き合わせる。**両者の差は staging の
// 丸め (重み 1 個につき 1 回) だけであるはずなので、そこが開いていれば
// タイルの幾何を疑う場所が一意に決まる。
//
// **形は 4 つ。**64 の倍数でない T と N を必ず 1 つ入れる — タイルの端の
// 処理は写し間違えても中央が正しければ「だいたい合う」ので、実物の形
// (2048 / 4096 / 8192、どれも 64 の倍数) だけを見ていると通ってしまう。
// `N=1` は shared expert のゲートで、tiled 版はタイルの 64 列中 63 列を
// 捨てるので scalar に落ちることを併せて見る。
//
// **負例 4 本。**INT4 の QMM (`prefill_int4_qmm_simdgroup_f16`) を 8-bit に
// 写すときに静かに壊れる道は、LM head を写したとき (`QwenHeadCheck.swift`) と
// 同じ 3 本 + タイル特有の 1 本:
//
//   行の刻み   K/2 のまま (ニブルの幾何) → 2 行に 1 行しか読まない
//   符号       バイトを signed で読む (MLX affine は 0..255 の unsigned)
//   bias 項    b·Σx を落とす (affine の零点は INT8 では本物のデータ)
//   重みの向き  `W` を `[K, N]` として読む (Bs を K-major に置く段の取り違え)

private func qmmBuffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer {
    values.withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: max(raw.count, 4),
                          options: .storageModeShared)!
    }
}

private func qmmRead<T>(_ buffer: MTLBuffer, count: Int, as: T.Type) -> [T] {
    Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: T.self, capacity: count),
                              count: count))
}

/// NaN を素通しさせない相対誤差 (`QwenHeadCheck.swift` と同じもの)。
private func qmmRelative(_ actual: [Double], _ reference: [Double]) -> Double {
    precondition(actual.count == reference.count, "shape mismatch — harness bug")
    var maxDiff = 0.0
    var refNorm = 0.0
    for i in 0..<actual.count {
        precondition(reference[i].isFinite, "reference is not finite — harness bug")
        refNorm = Swift.max(refNorm, abs(reference[i]))
        if !actual[i].isFinite { return .infinity }
        maxDiff = Swift.max(maxDiff, abs(actual[i] - reference[i]))
    }
    precondition(refNorm > 1e-4, "reference has no signal — harness bug")
    return maxDiff / refNorm
}

private struct QMMShape {
    var t: Int
    var n: Int
    var k: Int
    var groupSize = 64
    /// 期待する経路。呼んだあとで実際に走った経路と突き合わせる。
    var expected: QwenPrefillInt8QMM.Path

    var groupsPerRow: Int { k / groupSize }
    var label: String { "T=\(t) N=\(n) K=\(k)" }
}

private struct QMMInputs {
    var x: [Float16]
    var quantized: [UInt8]
    var scales: BF16Vector
    var biases: BF16Vector
}

private func makeQMMInputs(_ shape: QMMShape, seed: UInt64) -> QMMInputs {
    var rng = QwenRNG(state: seed)
    let x = (0..<(shape.t * shape.k)).map { _ in Float16(rng.normal()) }
    var quantized = [UInt8](repeating: 0, count: shape.n * shape.k)
    for i in 0..<quantized.count { quantized[i] = UInt8(rng.next() & 0xFF) }
    // 8-bit affine: `w = q * s + b`、q は 0..255 なので bias は負側に寄る
    // (`QwenHeadCheck.swift` と同じ桁)。
    var scaleRNG = QwenRNG(state: seed &+ 1)
    var scales = BF16Vector(count: shape.n * shape.groupsPerRow, scale: 0.0, rng: &scaleRNG)
    var biases = BF16Vector(count: shape.n * shape.groupsPerRow, scale: 0.0, rng: &scaleRNG)
    for i in 0..<scales.bits.count {
        let s = Float(0.0002 + 0.0004 * scaleRNG.uniform())
        let b = Float(-128.0 * Double(s) - 0.01 * scaleRNG.uniform())
        scales.bits[i] = Quantization.bf16Bits(s)
        scales.values[i] = Quantization.bf16ToFloat(scales.bits[i])
        biases.bits[i] = Quantization.bf16Bits(b)
        biases.values[i] = Quantization.bf16ToFloat(biases.bits[i])
    }
    return QMMInputs(x: x, quantized: quantized, scales: scales, biases: biases)
}

private enum QMMVariant {
    case correct
    /// INT4 の行の刻みを残す (`K / 2` バイト/行)。
    case nibbleRowStride
    /// 量子化されたバイトを符号つきで読む。
    case signedBytes
    /// affine の零点 (`b * Σx`) を落とす。
    case noBiasTerm
    /// `W` を `[K, N]` として読む (staging の向きの取り違え)。
    case transposedWeights
}

private func qmmReference<T: BinaryFloatingPoint>(
    _ inputs: QMMInputs, _ shape: QMMShape,
    as: T.Type, variant: QMMVariant = .correct
) -> [Double] {
    let rowStride = variant == .nibbleRowStride ? shape.k / 2 : shape.k
    var out = [Double](repeating: 0, count: shape.t * shape.n)
    for t in 0..<shape.t {
        for n in 0..<shape.n {
            var acc: T = 0
            for group in 0..<shape.groupsPerRow {
                let s = T(inputs.scales.values[n * shape.groupsPerRow + group])
                let b = T(inputs.biases.values[n * shape.groupsPerRow + group])
                var dot: T = 0
                var sum: T = 0
                for i in (group * shape.groupSize)..<((group + 1) * shape.groupSize) {
                    let raw = variant == .transposedWeights
                        ? inputs.quantized[i * shape.n + n]
                        : inputs.quantized[n * rowStride + i]
                    let q: T = variant == .signedBytes
                        ? T(Int8(bitPattern: raw))
                        : T(raw)
                    let xv = T(inputs.x[t * shape.k + i])
                    dot += q * xv
                    sum += xv
                }
                acc += s * dot
                if variant != .noBiasTerm { acc += b * sum }
            }
            // GPU は FP16 で書き出すので、参照もそこで丸める。
            out[t * shape.n + n] = Double(Float16(Float(acc)))
        }
    }
    return out
}

private func runQMMOnGPU(_ context: MetalContext,
                         _ qmm: QwenPrefillInt8QMM,
                         _ inputs: QMMInputs,
                         _ shape: QMMShape,
                         forcedPath: QwenPrefillInt8QMM.Path? = nil)
    throws -> (values: [Double], path: QwenPrefillInt8QMM.Path) {
    let device = context.device
    let w = qmmBuffer(device, inputs.quantized)
    let s = qmmBuffer(device, inputs.scales.bits)
    let b = qmmBuffer(device, inputs.biases.bits)
    let x = qmmBuffer(device, inputs.x)
    let y = device.makeBuffer(length: shape.t * shape.n * MemoryLayout<Float16>.size,
                              options: .storageModeShared)!
    guard let commandBuffer = context.queue.makeCommandBuffer() else {
        throw QwenCheckError.noCommandBuffer
    }
    let path = qmm.encode(commandBuffer: commandBuffer,
                          weights: w, scales: s, biases: b,
                          x: x, y: y,
                          t: shape.t, n: shape.n, k: shape.k,
                          forcedPath: forcedPath)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        throw QwenCheckError.dispatchFailed("\(error)")
    }
    let values = qmmRead(y, count: shape.t * shape.n, as: Float16.self).map { Double($0) }
    return (values, path)
}

func runQwenQMMCheck(context: MetalContext) throws -> [CaseResult] {
    var results: [CaseResult] = []
    let qmm = try QwenPrefillInt8QMM(context: context)

    // **K だけ実物のまま** (hidden 2048)。affine の group 64 とタイルの 32 の
    // 両方の倍数であることがこのカーネルの前提なので、そこは動かさない。
    // T と N は縮める — 参照が O(T·N·K) の Swift ループで、実物の N (2048 /
    // 4096) だと検査に分が要る。どちらの向きもタイル 64 を**跨ぐ**ことだけは
    // 守る (`QwenHeadCheck.swift` が語彙を縮めるのと同じ判断)。
    let shapes = [
        // タイルを両方向に跨ぐ (T は 2 タイル、N は 3 タイル)。
        QMMShape(t: 72, n: 192, k: 2048, expected: .simdgroupMatrix),
        // ちょうど 1 タイル × 2 タイル。
        QMMShape(t: 64, n: 128, k: 2048, expected: .simdgroupMatrix),
        // 端。T も N も 64 の倍数でない。
        QMMShape(t: 100, n: 130, k: 2048, expected: .simdgroupMatrix),
        // `shared_expert_gate` — 1 行しか無いので tiled に行かせない。
        QMMShape(t: 96, n: 1, k: 2048, expected: .scalarBlock),
    ]

    for shape in shapes {
        let inputs = makeQMMInputs(shape, seed: 0x9E11_0300 &+ UInt64(shape.n))
        let truth = qmmReference(inputs, shape, as: Double.self)
        let floor32 = qmmReference(inputs, shape, as: Float.self)
        let floorRel = qmmRelative(floor32, truth)

        let got = try runQMMOnGPU(context, qmm, inputs, shape)
        // 経路そのものを採点する。tiled のつもりで scalar が走っていれば、
        // 数字はどちらも正しいので気づけない。
        results.append(result("qwen_int8_qmm 経路 \(shape.label)",
                              groupSize: 0,
                              rel: got.path == shape.expected ? 0 : 1,
                              tolerance: 0,
                              detail: "\(got.path.rawValue) (期待 \(shape.expected.rawValue))"))
        results.append(result("qwen_int8_qmm \(shape.label)",
                              groupSize: 0,
                              rel: qmmRelative(got.values, truth),
                              tolerance: 4e-3,
                              detail: String(format: "float32 の床 %.2e", floorRel)))

        // 2 本目の経路。tiled が使える形でだけ、両方走らせて突き合わせる。
        if shape.expected == .simdgroupMatrix {
            let scalar = try runQMMOnGPU(context, qmm, inputs, shape, forcedPath: .scalarBlock)
            precondition(scalar.path == .scalarBlock, "forcedPath が効いていない")
            results.append(result("qwen_int8_qmm scalar \(shape.label)",
                                  groupSize: 0,
                                  rel: qmmRelative(scalar.values, truth),
                                  tolerance: 4e-3,
                                  detail: "重みを FP32 のまま積む経路"))
            results.append(result("qwen_int8_qmm tiled 対 scalar \(shape.label)",
                                  groupSize: 0,
                                  rel: qmmRelative(got.values, scalar.values),
                                  tolerance: 4e-3,
                                  detail: "差は staging の FP16 丸めだけのはず"))
        }
    }

    // 検出力。実物の形 1 つに対して、静かに壊れる 4 通りの参照を当てる。
    let shape = shapes[0]
    let inputs = makeQMMInputs(shape, seed: 0x9E11_0300 &+ UInt64(shape.n))
    let got = try runQMMOnGPU(context, qmm, inputs, shape)
    for (name, variant) in [
        ("行の刻みが K/2 (INT4 の幾何)", QMMVariant.nibbleRowStride),
        ("バイトを signed で読む", .signedBytes),
        ("affine の bias 項を落とす", .noBiasTerm),
        ("重みを [K, N] として読む", .transposedWeights),
    ] {
        let wrong = qmmReference(inputs, shape, as: Double.self, variant: variant)
        results.append(detectionResult("qwen_int8_qmm 検出力 (\(name))",
                                       groupSize: 0,
                                       rel: qmmRelative(got.values, wrong),
                                       floor: 1e-2))
    }

    return results
}
