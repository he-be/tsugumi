import Foundation
import Metal
import TurboFieldfare

// MARK: - サンプラの検査 (`docs/qwen35moe/42-SAMPLING.md`)
//
// 採点するものが 2 つある。
//
//   1. **切り詰めが仕様どおりか。**`QwenSampler` は GPU の `sample` カーネルの
//      規則をホストに写したもの (投機サンプリングが p と q の**値**を要るため、
//      §2-2)。ここでは写した先ではなく**独立に double で作った参照**と突き合わせる
//      — カーネルとホストが同じように間違っている場合を排除するため。
//   2. **投機サンプリングの出力分布が本当に p か。**カーネルの話ではなく
//      手続きの話なので、モデルを開かずに合成分布で採点できる。42 §4 の
//      受入条件 (段取り 3) の前倒しで、実機に触る前にここで落とせる。
//
// 負例は 2 本。「Gemma の softcap を掛けたまま」と「argmax 一致で受理する」
// — どちらも**静かに別の分布になる**形で、要件 S1 が壊れる道そのものである。

private func samplerBuffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer {
    values.withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: raw.count,
                          options: .storageModeShared)!
    }
}

/// 実物の logit の桁に寄せた合成分布。上位が数本だけ抜けている形にする
/// (何も切らない分布では切り詰めの検査にならない)。
private func syntheticLogits(vocab: Int, seed: UInt64) -> [Float16] {
    var rng = QwenRNG(state: seed)
    var out = [Float16](repeating: 0, count: vocab)
    for i in 0..<vocab { out[i] = Float16(Float(rng.normal()) * 3.0) }
    out[vocab / 3] = Float16(18.0)
    out[vocab / 7] = Float16(16.5)
    out[vocab / 11] = Float16(15.0)
    return out
}

/// 平坦な分布。上位 20 本を取っても 0.95 の質量に届かないので、top_k が
/// 切る側の枝を踏む。
private func flatLogits(vocab: Int, seed: UInt64) -> [Float16] {
    var rng = QwenRNG(state: seed)
    return (0..<vocab).map { _ in Float16(Float(rng.normal()) * 0.35) }
}

/// double の参照。softmax → (top_p → top_k) → 温度 → 正規化、42 §2-1 の順。
/// `softcap` が正なら Gemma の cap を掛ける (負例のため)。
private func referenceCategorical(_ logits: [Float16],
                                  temperature: Double,
                                  topK: Int,
                                  topP: Double,
                                  softcap: Double = 0) -> (ids: [Int32], weights: [Double]) {
    let capped = logits.map { value -> Double in
        let z = Double(value)
        return softcap > 0 ? softcap * tanh(z / softcap) : z
    }
    let m = capped.max() ?? 0
    let exps = capped.map { Foundation.exp($0 - m) }
    let total = exps.reduce(0, +)
    let probs = exps.map { $0 / total }
    let sorted = probs.enumerated().sorted { lhs, rhs in
        lhs.element == rhs.element ? lhs.offset < rhs.offset : lhs.element > rhs.element
    }
    var ids: [Int32] = []
    var kept: [Double] = []
    var cumulative = 0.0
    for (offset, p) in sorted.prefix(topK) {
        ids.append(Int32(offset))
        kept.append(p)
        cumulative += p
        if topP > 0, topP < 1, cumulative >= topP { break }
    }
    let reweighted = kept.map { Foundation.pow($0, 1.0 / temperature) }
    let sum = reweighted.reduce(0, +)
    return (ids, reweighted.map { $0 / sum })
}

/// 経験分布と目標分布の全変動距離。
private func totalVariation(_ counts: [Int32: Int], _ target: QwenCategorical,
                            draws: Int) -> Double {
    var seen = Set<Int32>()
    var tv = 0.0
    for (index, id) in target.ids.enumerated() {
        seen.insert(id)
        let empirical = Double(counts[id] ?? 0) / Double(draws)
        tv += abs(empirical - Double(target.weights[index]))
    }
    for (id, count) in counts where !seen.contains(id) {
        tv += Double(count) / Double(draws)
    }
    return tv / 2
}

func runQwenSamplerCheck(context: MetalContext) throws -> [CaseResult] {
    var results: [CaseResult] = []
    let vocab = 4096
    let groupSize = context.affineGroupSize

    // 公式推奨設定 (S1)。検査もこの 3 つで走らせる。
    let config = GenerationConfig(maxNewTokens: 1,
                                  temperature: 0.6,
                                  topK: 20,
                                  topP: 0.95,
                                  seed: 0x5A11_0001)
    let logitValues = syntheticLogits(vocab: vocab, seed: 0x5A3D_0001)
    // 2 行目は**平坦な**分布。1 行目は尖っていて top_p が先に切るので、
    // それだけでは `top_k=20` の枝を一度も踏まない (最初に書いたときは 2 本しか
    // 残らなかった)。両方の枝を踏ませるために 2 行流す。
    let flatValues = flatLogits(vocab: vocab, seed: 0x5A3D_0002)
    let logits = samplerBuffer(context.device, logitValues + flatValues)

    let sampler = try QwenSampler(context: context, vocab: vocab, maxRows: 2)
    guard let commandBuffer = context.queue.makeCommandBuffer() else {
        throw QwenCheckError.noCommandBuffer
    }
    sampler.encodeProbabilities(commandBuffer: commandBuffer, logits: logits, rows: 2)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error { throw QwenCheckError.dispatchFailed("\(error)") }

    let got = try sampler.categorical(row: 0, config: config)
    let truth = referenceCategorical(logitValues, temperature: 0.6, topK: 20, topP: 0.95)

    // ---- 1. 切り詰め ---------------------------------------------------------
    results.append(result("QwenSampler 残す ID 列 (top_p=0.95 → top_k=20)",
                          groupSize: groupSize,
                          rel: got.ids == truth.ids ? 0 : 1,
                          tolerance: 0,
                          detail: "GPU 経路 \(got.ids.count) 本 / 参照 \(truth.ids.count) 本"))
    var weightError = 0.0
    for (index, w) in got.weights.enumerated() where index < truth.weights.count {
        weightError = Swift.max(weightError, abs(Double(w) - truth.weights[index]))
    }
    results.append(result("QwenSampler 温度 0.6 の重み (対 double)",
                          groupSize: groupSize,
                          rel: weightError,
                          // 確率は FP16 で GPU から降りてくる。その刻みが下限。
                          tolerance: 4e-3,
                          detail: String(format: "最大差 %.2e", weightError)))
    let weightSum = got.weights.reduce(0, +)
    results.append(result("QwenSampler 重みの和が 1",
                          groupSize: groupSize,
                          rel: Double(abs(weightSum - 1)),
                          tolerance: 1e-5,
                          detail: String(format: "%.7f", weightSum)))

    // ---- 1b. top_k が効く側の枝 (平坦な分布) ---------------------------------
    let flatGot = try sampler.categorical(row: 1, config: config)
    let flatTruth = referenceCategorical(flatValues, temperature: 0.6, topK: 20, topP: 0.95)
    results.append(result("QwenSampler 平坦な分布では top_k=20 が効く",
                          groupSize: groupSize,
                          rel: flatGot.ids == flatTruth.ids && flatGot.ids.count == 20 ? 0 : 1,
                          tolerance: 0,
                          detail: "残ったのは \(flatGot.ids.count) 本 / 参照 \(flatTruth.ids.count) 本"))
    var flatWeightError = 0.0
    for (index, w) in flatGot.weights.enumerated() where index < flatTruth.weights.count {
        flatWeightError = Swift.max(flatWeightError, abs(Double(w) - flatTruth.weights[index]))
    }
    results.append(result("QwenSampler 平坦な分布の重み (対 double)",
                          groupSize: groupSize,
                          rel: flatWeightError,
                          tolerance: 4e-3,
                          detail: String(format: "最大差 %.2e", flatWeightError)))

    // ---- 2. 負例: Gemma の softcap を切り忘れる ------------------------------
    let cappedTruth = referenceCategorical(logitValues, temperature: 0.6,
                                           topK: 20, topP: 0.95, softcap: 30)
    var cappedError = 0.0
    for (index, w) in got.weights.enumerated() where index < cappedTruth.weights.count {
        cappedError = Swift.max(cappedError, abs(Double(w) - cappedTruth.weights[index]))
    }
    results.append(detectionResult("QwenSampler 検出力 (Gemma の softcap を掛けたまま)",
                                   groupSize: groupSize,
                                   rel: cappedError,
                                   floor: 1e-2,
                                   detail: "30*tanh(z/30) を掛けると分布が変わる"))

    // ---- 3. greedy は argmax (評価経路 S4) -----------------------------------
    var greedyConfig = config
    greedyConfig.temperature = 0
    let greedy = try sampler.categorical(row: 0, config: greedyConfig)
    results.append(result("QwenSampler temperature 0 は argmax 1 本",
                          groupSize: groupSize,
                          rel: greedy.ids == [truth.ids[0]] && greedy.weights == [1] ? 0 : 1,
                          tolerance: 0,
                          detail: "greedy \(greedy.ids) / 参照 \(truth.ids[0])"))

    // ---- 4. 投機サンプリングが p を保つ (42 §2-2) ----------------------------
    //
    // draft q と本体 p を別の分布にして、受理・棄却の手続きを 200,000 回まわす。
    // 経験分布が **p** に一致しなければ、要件 S1 と S2 は両立していない。
    let p = QwenCategorical(ids: [10, 11, 12, 13],
                            weights: [0.50, 0.25, 0.15, 0.10])
    let q = QwenCategorical(ids: [10, 11, 14],
                            weights: [0.30, 0.60, 0.10])
    let residual = p.residual(minus: q)
    results.append(result("残差 (p-q)+ の重みの和が 1",
                          groupSize: groupSize,
                          rel: Double(abs(residual.weights.reduce(0, +) - 1)),
                          tolerance: 1e-5,
                          detail: "残す ID \(residual.ids)"))

    let draws = 200_000
    var counts: [Int32: Int] = [:]
    var rng = QwenRNG(state: 0x5A3D_5EC0)
    for _ in 0..<draws {
        let d = q.draw(u: Float(rng.uniform()))
        let accept = Float(rng.uniform())
        let pd = p.probability(of: d)
        let qd = q.probability(of: d)
        if qd > 0, accept <= Swift.min(1, pd / qd) {
            counts[d, default: 0] += 1
        } else {
            counts[residual.draw(u: Float(rng.uniform())), default: 0] += 1
        }
    }
    let tv = totalVariation(counts, p, draws: draws)
    results.append(result("投機サンプリングの出力分布 == p (\(draws) 回)",
                          groupSize: groupSize,
                          rel: tv,
                          // 4 本の多項分布、n=200,000 の標本誤差は 1e-3 の桁。
                          tolerance: 6e-3,
                          detail: String(format: "全変動距離 %.4f", tv)))

    // 負例: 受理規則を「argmax が一致したら受理」(= 今の MTP) にすると分布が
    // p から外れる。要件 S1 が壊れる形そのもの。
    var greedyCounts: [Int32: Int] = [:]
    var greedyRNG = QwenRNG(state: 0x5A3D_5EC0)
    for _ in 0..<draws {
        let d = q.draw(u: Float(greedyRNG.uniform()))
        _ = greedyRNG.uniform()
        greedyCounts[d == p.ids[0] ? d : p.ids[0], default: 0] += 1
    }
    results.append(detectionResult("検出力 (argmax 一致で受理すると p から外れる)",
                                   groupSize: groupSize,
                                   rel: totalVariation(greedyCounts, p, draws: draws),
                                   floor: 1e-1,
                                   detail: "今の MTP の受理規則をサンプリング下で使った場合"))
    return results
}
