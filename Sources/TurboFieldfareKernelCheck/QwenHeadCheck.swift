import Foundation
import Metal
import TurboFieldfare

// MARK: - INT8 の LM head chain の検査 (`docs/qwen35moe/03-DESIGN.md` §2-8)
//
// `QwenKernelCheck.swift` と同じ立て方: チェックポイントも fixtures も開かず、
// 実物と同じ形の合成入力に対して GPU / CPU float32 (床) / CPU double (真値) を
// 走らせる。違うのは**採点対象が logit そのものではない**ことで、この chain は
// 語彙幅のベクトルをどこにも書かない。代わりに読むのは:
//
//   summaries  8 行ごとの最大 logit — 採点した語彙行を**全部**覆う FP32 の配列
//   token      それを畳んだ argmax
//
// 前者が数値の物差し、後者が経路全体の答え。
//
// **負例 4 本。**INT4 の chain を 8-bit に写すときに静かに壊れる道は決まっている:
//
//   行の刻み     N/2 のまま (ニブルの幾何) → 2 行に 1 行しか読まない
//   符号         バイトを signed で読む (MLX affine は 0..255 の unsigned)
//   bias 項      b·Σx を落とす (affine の零点は INT8 では本物のデータ)
//   語彙の末尾   248,077 ではなく 248,320 行を採点する (末尾 243 行は未学習)
//
// 最後の 1 本は算術ではなく**引数**の間違いなので、参照ではなく入力側に仕掛ける:
// 未学習の行にわざと大きな値を置き、chain がそれを選ばないことを見る。

// 小道具。`QwenKernelCheck.swift` と `GatedDeltaNetCheck.swift` が同じ名前の
// ものをファイル内に持っているので、ここも自前で持つ (どれも数行)。

private func headBuffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer {
    values.withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: raw.count,
                          options: .storageModeShared)!
    }
}

private func headRead<T>(_ buffer: MTLBuffer, count: Int, as: T.Type) -> [T] {
    Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: T.self, capacity: count),
                              count: count))
}

/// NaN を素通しさせない相対誤差。参照に信号が無ければ物差しのバグ。
private func headRelative(_ actual: [Double], _ reference: [Double]) -> Double {
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

private struct HeadShape {
    /// 実物と同じ。語彙だけは検査の速さのために小さくする (語彙方向は純粋な
    /// 行並列なので、経路は同じものが通る)。
    var d = 2048
    /// 採点する行数。
    var scored = 8192
    /// 表に載っている行数。`scored` を超える分が「未学習の末尾」にあたる。
    var total = 8192 + 243
    var groupSize = 64
    var rmsEps: Float = 1e-6

    var groupsPerRow: Int { d / groupSize }
}

/// 量子化された lm_head の表と、参照が使うまったく同じ値。
private struct HeadInputs {
    var hidden: [Float16]
    var normWeight: BF16Vector
    var quantized: [UInt8]
    var scales: BF16Vector
    var biases: BF16Vector
}

private func makeHeadInputs(_ shape: HeadShape, seed: UInt64) -> HeadInputs {
    var rng = QwenRNG(state: seed)
    let hidden = (0..<shape.d).map { _ in Float16(rng.normal()) }
    let normWeight = BF16Vector(count: shape.d, scale: 1.0, rng: &rng)
    var quantized = [UInt8](repeating: 0, count: shape.total * shape.d)
    for i in 0..<quantized.count { quantized[i] = UInt8(rng.next() & 0xFF) }
    // 8-bit affine の scale は正、bias は零点なので負側に寄る
    // (`w = q * s + b`、q は 0..255)。実物の桁に合わせておく。
    var scaleRNG = QwenRNG(state: seed &+ 1)
    var scales = BF16Vector(count: shape.total * shape.groupsPerRow, scale: 0.0, rng: &scaleRNG)
    var biases = BF16Vector(count: shape.total * shape.groupsPerRow, scale: 0.0, rng: &scaleRNG)
    for i in 0..<scales.bits.count {
        let s = Float(0.0002 + 0.0004 * scaleRNG.uniform())
        let b = Float(-128.0 * Double(s) - 0.01 * scaleRNG.uniform())
        scales.bits[i] = Quantization.bf16Bits(s)
        scales.values[i] = Quantization.bf16ToFloat(scales.bits[i])
        biases.bits[i] = Quantization.bf16Bits(b)
        biases.values[i] = Quantization.bf16ToFloat(biases.bits[i])
    }
    return HeadInputs(hidden: hidden, normWeight: normWeight,
                      quantized: quantized, scales: scales, biases: biases)
}

private enum HeadVariant {
    case correct
    /// INT4 の行の刻みを残す (`d / 2` バイト/行)。
    case nibbleRowStride
    /// 量子化されたバイトを符号つきで読む。
    case signedBytes
    /// affine の零点 (`b * Σx`) を落とす。
    case noBiasTerm
}

/// 参照。GPU と同じ順序で 2 段: RMSNorm を FP16 に落としてから内積を取る。
/// (chain も `xNormed` を FP16 で持つので、そこを double のまま流すと
/// カーネルの罪でない差が出る。)
private func headReference<T: BinaryFloatingPoint>(
    _ inputs: HeadInputs, _ shape: HeadShape, rows: Int,
    as: T.Type, variant: HeadVariant = .correct
) -> (normed: [Float16], logits: [Double], best: Int) {
    var square: T = 0
    for value in inputs.hidden { square += T(value) * T(value) }
    let inv = T(1) / T(Double(square / T(shape.d) + T(shape.rmsEps)).squareRoot())
    // 焼き済みの `1+w` がそのまま重みなので、ここで 1 を足してはいけない
    // (`docs/qwen35moe/10-MLX4BIT-AUDIT.md` §3)。
    let normed = (0..<shape.d).map {
        Float16(T(inputs.hidden[$0]) * inv * T(inputs.normWeight.values[$0]))
    }
    let rowStride = variant == .nibbleRowStride ? shape.d / 2 : shape.d
    var logits = [Double](repeating: 0, count: rows)
    var best = 0
    var bestValue = -Double.infinity
    for row in 0..<rows {
        var acc: T = 0
        for group in 0..<shape.groupsPerRow {
            let s = T(inputs.scales.values[row * shape.groupsPerRow + group])
            let b = T(inputs.biases.values[row * shape.groupsPerRow + group])
            var dot: T = 0
            var sum: T = 0
            for i in (group * shape.groupSize)..<((group + 1) * shape.groupSize) {
                let raw = inputs.quantized[row * rowStride + i]
                let q = variant == .signedBytes ? T(Int8(bitPattern: raw)) : T(raw)
                let x = T(normed[i])
                dot += q * x
                sum += x
            }
            acc += s * dot
            if variant != .noBiasTerm { acc += b * sum }
        }
        logits[row] = Double(acc)
        if logits[row] > bestValue { bestValue = logits[row]; best = row }
    }
    return (normed, logits, best)
}

/// 8 行ごとの最大値 — GPU の summaries と同じ畳み方。
private func headSummaries(_ logits: [Double], rowsPerGroup: Int) -> [Double] {
    stride(from: 0, to: logits.count, by: rowsPerGroup).map { start in
        logits[start..<Swift.min(start + rowsPerGroup, logits.count)].max()!
    }
}

private struct HeadRun {
    var normed: [Float16]
    var logits: [Float16]
    var summaries: [Double]
    var token: Int
}

private func runHeadOnGPU(_ context: MetalContext,
                          _ chain: QwenLMHeadChainInt8,
                          _ gemv: DequantInt8GEMV,
                          _ inputs: HeadInputs,
                          _ shape: HeadShape,
                          vocab: Int) throws -> HeadRun {
    let device = context.device
    let hidden = headBuffer(device, inputs.hidden)
    let normWeight = headBuffer(device, inputs.normWeight.bits)
    let weights = headBuffer(device, inputs.quantized)
    let scales = headBuffer(device, inputs.scales.bits)
    let biases = headBuffer(device, inputs.biases.bits)
    let token = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                  options: .storageModeShared)!
    guard let commandBuffer = context.queue.makeCommandBuffer() else {
        throw QwenCheckError.noCommandBuffer
    }
    guard chain.encodeGreedyDecode(commandBuffer: commandBuffer,
                                   hidden: hidden,
                                   normWeight: normWeight,
                                   weights: weights,
                                   scales: scales,
                                   biases: biases,
                                   outToken: token,
                                   d: UInt32(shape.d),
                                   vocab: UInt32(vocab),
                                   rmsEps: shape.rmsEps) else {
        throw QwenCheckError.encodeRefused("qwen_lm_head_greedy_int8")
    }
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        throw QwenCheckError.dispatchFailed("\(error)")
    }
    let groups = chain.rowGroupCount(vocab: vocab)
    let raw = headRead(chain.rowSummaries, count: groups * 2, as: Float.self)

    // 二人目の証人。`dequant_int8_gemv_simd` は Gemma 4 の router と shared
    // expert がずっと使ってきた INT8 の読み方で、こちらは logit を全部書き出す。
    // chain が同じ表を同じように読めているなら、両者は FP16 の刻みまで一致する。
    let logits = device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                   options: .storageModeShared)!
    guard let second = context.queue.makeCommandBuffer() else {
        throw QwenCheckError.noCommandBuffer
    }
    gemv.encode(commandBuffer: second,
                weights: weights, scales: scales, biases: biases,
                x: chain.normalizedHidden, y: logits,
                m: UInt32(vocab), n: UInt32(shape.d))
    second.commit()
    second.waitUntilCompleted()
    if let error = second.error {
        throw QwenCheckError.dispatchFailed("\(error)")
    }

    return HeadRun(normed: headRead(chain.normalizedHidden, count: shape.d, as: Float16.self),
                   logits: headRead(logits, count: vocab, as: Float16.self),
                   summaries: stride(from: 0, to: raw.count, by: 2).map { Double(raw[$0]) },
                   token: Int(headRead(token, count: 1, as: UInt32.self)[0]))
}

/// 一致 / 不一致を `CaseResult` の物差しに載せる。トークン ID は連続量ではないので、
/// 誤差ではなく 0/1 で採点する。
private func agreement(_ actual: Int, _ expected: Int) -> Double {
    actual == expected ? 0 : 1
}

func runQwenHeadCheck(context: MetalContext) throws -> [CaseResult] {
    let shape = HeadShape()
    let chain = try QwenLMHeadChainInt8(context: context,
                                        maxD: shape.d,
                                        maxVocab: shape.total)
    let gemv = try DequantInt8GEMV(context: context)
    var results: [CaseResult] = []

    // ---- 1. 正例 ------------------------------------------------------------
    let inputs = makeHeadInputs(shape, seed: 0x1_5EAD_0001)
    let truth = headReference(inputs, shape, rows: shape.scored, as: Double.self)
    let floor32 = headReference(inputs, shape, rows: shape.scored, as: Float.self)
    let got = try runHeadOnGPU(context, chain, gemv, inputs, shape, vocab: shape.scored)

    let truthSummaries = headSummaries(truth.logits,
                                       rowsPerGroup: QwenLMHeadChainInt8.rowsPerThreadgroup)
    let floorSummaries = headSummaries(floor32.logits,
                                       rowsPerGroup: QwenLMHeadChainInt8.rowsPerThreadgroup)
    let floorRel = headRelative(floorSummaries, truthSummaries)
    results.append(result("qwen_lm_head_int8 summaries (\(truthSummaries.count) 本)",
                          groupSize: shape.groupSize,
                          rel: headRelative(got.summaries, truthSummaries),
                          // FP32 で出るものは床の 20 倍 (17-PHASE2-KERNELS §2)。
                          tolerance: Swift.max(20 * floorRel, 1e-6),
                          detail: String(format: "float32 の床 %.2e", floorRel)))
    results.append(result("qwen_lm_head_int8 前段の RMSNorm",
                          groupSize: shape.groupSize,
                          rel: headRelative(got.normed.map { Double($0) },
                                            truth.normed.map { Double($0) }),
                          tolerance: 4e-3,
                          detail: String(format: "float32 の床 %.2e",
                                         headRelative(floor32.normed.map { Double($0) },
                                                      truth.normed.map { Double($0) }))))
    // 二人目の証人を 2 通りに使う。まず既存カーネルが同じ参照に乗ることを見て
    // (= 参照の算式が INT8 の規約どおりだという証拠)、次にその logit を chain の
    // summaries と**直に**突き合わせる (= chain が同じ表を同じように読んでいる
    // という証拠)。片方だけでは、両方が同じように間違っている場合を排除できない。
    results.append(result("dequant_int8_gemv_simd の logit \(shape.scored) 行 (参照との一致)",
                          groupSize: shape.groupSize,
                          rel: headRelative(got.logits.map { Double($0) }, truth.logits),
                          tolerance: 4e-3,
                          detail: "既存の INT8 経路が同じ算式に乗る"))
    results.append(result("qwen_lm_head_int8 summaries 対 dequant_int8_gemv_simd",
                          groupSize: shape.groupSize,
                          rel: headRelative(got.summaries,
                                            headSummaries(got.logits.map { Double($0) },
                                                          rowsPerGroup: QwenLMHeadChainInt8.rowsPerThreadgroup)),
                          tolerance: 4e-3,
                          detail: "gemv の出力は FP16 なのでその刻みが下限"))
    results.append(result("qwen_lm_head_int8 token (argmax)",
                          groupSize: shape.groupSize,
                          rel: agreement(got.token, truth.best),
                          tolerance: 0,
                          detail: "GPU \(got.token) / 参照 \(truth.best)"))

    // ---- 2. 負例 3 本 (算術) -------------------------------------------------
    for (label, variant, floor) in [
        ("行の刻みが N/2 (INT4 の幾何)", HeadVariant.nibbleRowStride, 1e-2),
        ("バイトを signed で読む", HeadVariant.signedBytes, 1e-2),
        ("affine の bias 項を落とす", HeadVariant.noBiasTerm, 1e-2),
    ] {
        let wrong = headReference(inputs, shape, rows: shape.scored,
                                  as: Double.self, variant: variant)
        let wrongSummaries = headSummaries(
            wrong.logits, rowsPerGroup: QwenLMHeadChainInt8.rowsPerThreadgroup)
        results.append(detectionResult("qwen_lm_head_int8 検出力 (\(label))",
                                       groupSize: shape.groupSize,
                                       rel: headRelative(got.summaries, wrongSummaries),
                                       floor: floor))
    }

    // ---- 3. 未学習の末尾 243 行 ----------------------------------------------
    //
    // `vocab_size` は 248,320、tokenizer の語彙は 248,077。差の 243 行は
    // 学習されていないので採点しない (10-MLX4BIT-AUDIT §3)。ここでは末尾の行に
    // **必ず勝つ**大きさの重みを置いて、`vocab` に何を渡したかを可視にする。
    var loud = inputs
    for row in shape.scored..<shape.total {
        for group in 0..<shape.groupsPerRow {
            let index = row * shape.groupsPerRow + group
            let s = Float(1.0)
            loud.scales.bits[index] = Quantization.bf16Bits(s)
            loud.scales.values[index] = Quantization.bf16ToFloat(loud.scales.bits[index])
            loud.biases.bits[index] = Quantization.bf16Bits(0)
            loud.biases.values[index] = 0
        }
        for i in 0..<shape.d { loud.quantized[row * shape.d + i] = 255 }
    }
    let scoredRun = try runHeadOnGPU(context, chain, gemv, loud, shape, vocab: shape.scored)
    let allRun = try runHeadOnGPU(context, chain, gemv, loud, shape, vocab: shape.total)
    results.append(result("qwen_lm_head_int8 末尾 243 行を採点しない",
                          groupSize: shape.groupSize,
                          rel: scoredRun.token < shape.scored ? 0 : 1,
                          tolerance: 0,
                          detail: "vocab=\(shape.scored) で選んだのは \(scoredRun.token)"))
    results.append(detectionResult("qwen_lm_head_int8 検出力 (末尾まで採点する)",
                                   groupSize: shape.groupSize,
                                   rel: agreement(allRun.token, scoredRun.token),
                                   floor: 1,
                                   detail: "vocab=\(shape.total) なら \(allRun.token)"))
    return results
}

/// 実物の語彙で 1 トークンぶんを測る。508 MB の表を 1 回読むのがこの経路の全部
/// なので、数字は帯域そのものになるはず。
func runQwenHeadBench(context: MetalContext, iterations: Int) throws {
    var shape = HeadShape()
    shape.scored = QwenLMHeadChainInt8.ornithScoredVocab
    shape.total = 248_320
    let chain = try QwenLMHeadChainInt8(context: context,
                                        maxD: shape.d,
                                        maxVocab: shape.total)
    let inputs = makeHeadInputs(shape, seed: 0x1_5EAD_BE7C)
    let device = context.device
    let hidden = headBuffer(device, inputs.hidden)
    let normWeight = headBuffer(device, inputs.normWeight.bits)
    let weights = headBuffer(device, inputs.quantized)
    let scales = headBuffer(device, inputs.scales.bits)
    let biases = headBuffer(device, inputs.biases.bits)
    let token = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                  options: .storageModeShared)!

    var samples: [Double] = []
    for _ in 0..<iterations {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw QwenCheckError.noCommandBuffer
        }
        guard chain.encodeGreedyDecode(commandBuffer: commandBuffer,
                                       hidden: hidden, normWeight: normWeight,
                                       weights: weights, scales: scales, biases: biases,
                                       outToken: token,
                                       d: UInt32(shape.d), vocab: UInt32(shape.scored),
                                       rmsEps: shape.rmsEps) else {
            throw QwenCheckError.encodeRefused("qwen_lm_head_greedy_int8")
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QwenCheckError.dispatchFailed("\(error)")
        }
        samples.append((commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1e3)
    }
    samples.sort()
    let median = samples[samples.count / 2]
    let bytes = Double(shape.scored * shape.d)
        + Double(shape.scored * shape.groupsPerRow * 2 * 2)
    print("  \("qwen_lm_head_int8".padding(toLength: 30, withPad: " ", startingAt: 0))"
          + "\(String(format: "%6d", 1))"
          + "\(String(format: "%12.3f", median))"
          + "\(String(format: "%14.1f", median))"
          + String(format: "   %.0f GB/s (%.0f MB)", bytes / (median * 1e-3) / 1e9, bytes / 1e6))
}
