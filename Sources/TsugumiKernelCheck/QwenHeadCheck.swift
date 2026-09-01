import Foundation
import Metal
import Tsugumi

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

/// 許可マスクを 1 行 1 bit に詰める。`reverseBitOrder` は負例のためのもので、
/// ワード内の並びだけを逆さにする (32 本の塊の中で ID が入れ替わる)。
private func packedMask(_ allowed: [Bool], reverseBitOrder: Bool = false) -> [UInt32] {
    let words = QwenLMHeadChainInt8.maskWordCount(vocab: allowed.count)
    var packed = [UInt32](repeating: 0, count: words)
    for (index, isAllowed) in allowed.enumerated() where isAllowed {
        let bit = reverseBitOrder ? 31 - (index % 32) : index % 32
        packed[index / 32] |= UInt32(1) << UInt32(bit)
    }
    return packed
}

/// 許可された行だけの argmax。参照側。
private func maskedBest(_ logits: [Double], _ allowed: [Bool]) -> Int {
    var best = -1
    var bestValue = -Double.infinity
    for row in 0..<logits.count where allowed[row] {
        if logits[row] > bestValue { bestValue = logits[row]; best = row }
    }
    return best
}

/// 融合ヘッドを 1 回通してから、同じ hidden をマスクつきでもう一度畳む。
///
/// 2 本目は前段の RMSNorm を走らせない (`encodeMaskedRescore` の約束) ので、
/// 1 本目が `xNormed` を書いていることがこの順序の前提そのものである。
private func runMaskedHeadOnGPU(_ context: MetalContext,
                                _ chain: QwenLMHeadChainInt8,
                                _ inputs: HeadInputs,
                                _ shape: HeadShape,
                                vocab: Int,
                                allowed: [Bool],
                                reverseBitOrder: Bool = false) throws -> Int {
    let device = context.device
    let hidden = headBuffer(device, inputs.hidden)
    let normWeight = headBuffer(device, inputs.normWeight.bits)
    let weights = headBuffer(device, inputs.quantized)
    let scales = headBuffer(device, inputs.scales.bits)
    let biases = headBuffer(device, inputs.biases.bits)
    let bits = headBuffer(device, packedMask(allowed, reverseBitOrder: reverseBitOrder))
    let token = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                  options: .storageModeShared)!

    guard let first = context.queue.makeCommandBuffer() else {
        throw QwenCheckError.noCommandBuffer
    }
    guard chain.encodeGreedyDecode(commandBuffer: first,
                                   hidden: hidden, normWeight: normWeight,
                                   weights: weights, scales: scales, biases: biases,
                                   outToken: token,
                                   d: UInt32(shape.d), vocab: UInt32(vocab),
                                   rmsEps: shape.rmsEps) else {
        throw QwenCheckError.encodeRefused("qwen_lm_head_greedy_int8")
    }
    first.commit()
    first.waitUntilCompleted()
    if let error = first.error { throw QwenCheckError.dispatchFailed("\(error)") }

    guard let second = context.queue.makeCommandBuffer() else {
        throw QwenCheckError.noCommandBuffer
    }
    guard chain.encodeMaskedRescore(commandBuffer: second,
                                    weights: weights, scales: scales, biases: biases,
                                    allowedBits: bits,
                                    outToken: token,
                                    d: UInt32(shape.d), vocab: UInt32(vocab)) else {
        throw QwenCheckError.encodeRefused("qwen_lm_head_greedy_int8_masked")
    }
    second.commit()
    second.waitUntilCompleted()
    if let error = second.error { throw QwenCheckError.dispatchFailed("\(error)") }
    return Int(headRead(token, count: 1, as: UInt32.self)[0])
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

    // ---- 4. マスクつきの畳み込み (GEN-7 の棄却経路) --------------------------
    //
    // `docs/qwen35moe/25-CLI-TOOLS.md` §2。融合ヘッドは logit を書かないので、
    // 文法が argmax を拒んだときは**同じ hidden をマスクつきでもう一度**畳む。
    // 行の演算は `_raw` と共有しているので、ここで採点しているのは
    // **どの行を採点したか**だけである。
    let everything = [Bool](repeating: true, count: shape.scored)
    let allMasked = try runMaskedHeadOnGPU(context, chain, inputs, shape,
                                           vocab: shape.scored, allowed: everything)
    results.append(result("qwen_lm_head_int8 マスク全許可 == 素の argmax",
                          groupSize: shape.groupSize,
                          rel: agreement(allMasked, got.token),
                          tolerance: 0,
                          detail: "マスク \(allMasked) / 素 \(got.token)"))

    var withoutWinner = everything
    withoutWinner[truth.best] = false
    let runnerUp = try runMaskedHeadOnGPU(context, chain, inputs, shape,
                                          vocab: shape.scored, allowed: withoutWinner)
    let expectedRunnerUp = maskedBest(truth.logits, withoutWinner)
    results.append(result("qwen_lm_head_int8 勝者を落とすと 2 位",
                          groupSize: shape.groupSize,
                          rel: agreement(runnerUp, expectedRunnerUp),
                          tolerance: 0,
                          detail: "GPU \(runnerUp) / 参照 \(expectedRunnerUp)"))

    // 疎なマスク。64 行に 1 本ほどしか通さないので、threadgroup (8 行) の中に
    // 許可行が 1 本も無い塊が生まれる — そこが `-INFINITY` のまま畳まれ、
    // 畳み込みの番兵 (0xFFFFFFFF) が答えに出てこないことまで見る。
    var sparse = [Bool](repeating: false, count: shape.scored)
    var sparseRNG = QwenRNG(state: 0x1_5EAD_0A55)
    for row in 0..<shape.scored where sparseRNG.next() % 64 == 0 { sparse[row] = true }
    if !sparse.contains(true) { sparse[7] = true }
    let sparseAllowed = sparse.filter { $0 }.count
    let sparseToken = try runMaskedHeadOnGPU(context, chain, inputs, shape,
                                             vocab: shape.scored, allowed: sparse)
    let expectedSparse = maskedBest(truth.logits, sparse)
    results.append(result("qwen_lm_head_int8 疎なマスク (\(sparseAllowed) 行)",
                          groupSize: shape.groupSize,
                          rel: agreement(sparseToken, expectedSparse),
                          tolerance: 0,
                          detail: "GPU \(sparseToken) / 参照 \(expectedSparse)"))

    // 文法が 1 本しか許さない状態 (綴りが決まりきっている途中) は実際に起きる。
    var single = [Bool](repeating: false, count: shape.scored)
    let onlyRow = (truth.best + 1_234) % shape.scored
    single[onlyRow] = true
    let singleToken = try runMaskedHeadOnGPU(context, chain, inputs, shape,
                                             vocab: shape.scored, allowed: single)
    results.append(result("qwen_lm_head_int8 許可が 1 本ならそれを返す",
                          groupSize: shape.groupSize,
                          rel: agreement(singleToken, onlyRow),
                          tolerance: 0,
                          detail: "GPU \(singleToken) / 許可 \(onlyRow)"))

    // ---- 5. 負例 2 本 (マスク) ----------------------------------------------
    //
    // 1 本目はマスクを読まないカーネル。勝者を落としても勝者が返るので、
    // 上の「2 位」の案がそれを捕まえられることをここで示す。
    results.append(detectionResult("qwen_lm_head_int8 検出力 (マスクを読まない)",
                                   groupSize: shape.groupSize,
                                   rel: agreement(runnerUp, truth.best),
                                   floor: 1,
                                   detail: "読まなければ \(truth.best) が返る"))
    // 2 本目はワード内の bit 順の取り違え。**落ちずに別の行を採点する**形の
    // 間違いで、マスクが密なうちは答えが変わらないこともあるので、1 本だけ
    // 許す形に当てる (逆順なら許されるのは別の ID になる)。
    let flipped = try runMaskedHeadOnGPU(context, chain, inputs, shape,
                                         vocab: shape.scored, allowed: single,
                                         reverseBitOrder: true)
    results.append(detectionResult("qwen_lm_head_int8 検出力 (ワード内の bit 順が逆)",
                                   groupSize: shape.groupSize,
                                   rel: agreement(flipped, onlyRow),
                                   floor: 1,
                                   detail: "逆順で選ばれたのは \(flipped)"))

    // ---- 6. logit を書き出す版 (サンプリングの土台) --------------------------
    //
    // `docs/qwen35moe/42-SAMPLING.md` §2-1。要件 S1 (公式推奨サンプラを実際に
    // 使う) には分布が要る。`_multi` が捨てていた内積を同じパスで FP16 に
    // 書き出すのがこのカーネルで、**argmax は今までどおり出す** —
    // 評価経路 (S4) と投機の強制棄却の対照が argmax のままだからである。
    //
    // 2 行目は 1 行目の 0.5 倍にしてある。z = Σ w·x は x について線形なので
    // 参照は「1 行目の半分」— 行の独立を、別の参照を作らずに採点できる
    // (2 の冪なので FP16 でも刻みが動かない)。行のオフセットを取り違えた
    // カーネルはここで落ちる。
    let logitsRun = try runLogitsHeadOnGPU(context, chain, inputs, shape,
                                           vocab: shape.scored,
                                           normed: got.normed)
    let row0 = logitsRun.logits[0].map { Double($0) }
    let row1 = logitsRun.logits[1].map { Double($0) }
    results.append(result("qwen_lm_head_logits_int8 の logit \(shape.scored) 行 (参照との一致)",
                          groupSize: shape.groupSize,
                          rel: headRelative(row0, truth.logits),
                          tolerance: 4e-3,
                          detail: "書き出しは FP16 なのでその刻みが下限"))
    results.append(result("qwen_lm_head_logits_int8 == dequant_int8_gemv_simd",
                          groupSize: shape.groupSize,
                          rel: headRelative(row0, got.logits.map { Double($0) }),
                          tolerance: 1e-6,
                          detail: "同じ hidden・同じ表・同じ FP16 の刻み"))
    results.append(result("qwen_lm_head_logits_int8 の argmax == 融合ヘッド",
                          groupSize: shape.groupSize,
                          rel: agreement(logitsRun.tokens[0], got.token),
                          tolerance: 0,
                          detail: "logits \(logitsRun.tokens[0]) / 融合 \(got.token)"))
    let hostArgmax = row0.enumerated().max(by: { $0.element < $1.element })?.offset ?? -1
    results.append(result("qwen_lm_head_logits_int8 書き出した logit の argmax も同じ",
                          groupSize: shape.groupSize,
                          rel: agreement(hostArgmax, got.token),
                          tolerance: 0,
                          detail: "ホストで畳むと \(hostArgmax)"))
    results.append(result("qwen_lm_head_logits_int8 2 行目は独立 (x の 0.5 倍)",
                          groupSize: shape.groupSize,
                          rel: headRelative(row1, row0.map { $0 * 0.5 }),
                          tolerance: 4e-3,
                          detail: "z は x について線形。行を取り違えたら落ちる"))
    // 検出力: 1 行目の参照をわざと壊したものに当てる。上の 4e-3 が「何とでも
    // 一致する緩さ」ではないことの証拠 (`PLAN_VISION.md` §6-3)。
    let wrongStride = headReference(inputs, shape, rows: shape.scored,
                                    as: Double.self, variant: .nibbleRowStride)
    results.append(detectionResult("qwen_lm_head_logits_int8 検出力 (行の刻みが N/2)",
                                   groupSize: shape.groupSize,
                                   rel: headRelative(row0, wrongStride.logits),
                                   floor: 1e-2))
    results.append(detectionResult("qwen_lm_head_logits_int8 検出力 (2 行目に 1 行目を書く)",
                                   groupSize: shape.groupSize,
                                   rel: headRelative(row1, row0),
                                   floor: 1e-2,
                                   detail: "0.5 倍でない行が来たら気付く"))
    return results
}

/// `logits` を書き出す版を走らせる。前段の RMSNorm は呼ばない (このカーネルは
/// 正規化済みの行を取る) ので、`normed` — 素の chain が書いた正規化済みの行 —
/// をそのまま 1 行目に、その 0.5 倍を 2 行目に置く。
private func runLogitsHeadOnGPU(_ context: MetalContext,
                                _ chain: QwenLMHeadChainInt8,
                                _ inputs: HeadInputs,
                                _ shape: HeadShape,
                                vocab: Int,
                                normed: [Float16]) throws
    -> (logits: [[Float16]], tokens: [Int]) {
    let device = context.device
    let rows = 2
    var stacked = normed
    stacked.append(contentsOf: normed.map { Float16(Float($0) * 0.5) })
    let hidden = headBuffer(device, stacked)
    let weights = headBuffer(device, inputs.quantized)
    let scales = headBuffer(device, inputs.scales.bits)
    let biases = headBuffer(device, inputs.biases.bits)
    let tokens = device.makeBuffer(length: rows * MemoryLayout<UInt32>.size,
                                   options: .storageModeShared)!
    let logits = device.makeBuffer(length: rows * vocab * MemoryLayout<Float16>.size,
                                   options: .storageModeShared)!
    guard let commandBuffer = context.queue.makeCommandBuffer() else {
        throw QwenCheckError.noCommandBuffer
    }
    guard chain.encodeLogitsDecodeRows(commandBuffer: commandBuffer,
                                       hiddenNormed: hidden,
                                       weights: weights,
                                       scales: scales,
                                       biases: biases,
                                       outTokens: tokens,
                                       logits: logits,
                                       rows: rows,
                                       d: UInt32(shape.d),
                                       vocab: UInt32(vocab)) else {
        throw QwenCheckError.encodeRefused("qwen_lm_head_logits_int8_rows_chunk_multi")
    }
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        throw QwenCheckError.dispatchFailed("\(error)")
    }
    let flat = headRead(logits, count: rows * vocab, as: Float16.self)
    let ids = headRead(tokens, count: rows, as: UInt32.self).map { Int($0) }
    return ((0..<rows).map { Array(flat[$0 * vocab..<($0 + 1) * vocab]) }, ids)
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

    // The rejection path reads the same table with a bit test in front of each
    // row, so the two lines answer one question: does consulting the mask cost
    // anything on top of the read? (`docs/qwen35moe/25-CLI-TOOLS.md` §2.)
    // Every row is allowed here — a sparse mask would skip rows and measure a
    // best case rather than the one a refused token actually pays.
    let allowed = headBuffer(device, [UInt32](
        repeating: 0xFFFF_FFFF,
        count: QwenLMHeadChainInt8.maskWordCount(vocab: shape.scored)))

    func measure(_ label: String, _ encode: (MTLCommandBuffer) -> Bool) throws {
        var samples: [Double] = []
        for _ in 0..<iterations {
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                throw QwenCheckError.noCommandBuffer
            }
            guard encode(commandBuffer) else {
                throw QwenCheckError.encodeRefused(label)
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
        print("  \(label.padding(toLength: 30, withPad: " ", startingAt: 0))"
              + "\(String(format: "%6d", 1))"
              + "\(String(format: "%12.3f", median))"
              + "\(String(format: "%14.1f", median))"
              + String(format: "   %.0f GB/s (%.0f MB)",
                       bytes / (median * 1e-3) / 1e9, bytes / 1e6))
    }

    try measure("qwen_lm_head_int8") { commandBuffer in
        chain.encodeGreedyDecode(commandBuffer: commandBuffer,
                                 hidden: hidden, normWeight: normWeight,
                                 weights: weights, scales: scales, biases: biases,
                                 outToken: token,
                                 d: UInt32(shape.d), vocab: UInt32(shape.scored),
                                 rmsEps: shape.rmsEps)
    }
    try measure("qwen_lm_head_int8 masked") { commandBuffer in
        chain.encodeMaskedRescore(commandBuffer: commandBuffer,
                                  weights: weights, scales: scales, biases: biases,
                                  allowedBits: allowed,
                                  outToken: token,
                                  d: UInt32(shape.d), vocab: UInt32(shape.scored))
    }
}
