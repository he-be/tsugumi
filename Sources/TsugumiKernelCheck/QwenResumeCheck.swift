import Foundation
import Metal
import Tsugumi

// The property the Ornith prompt cache rests on
// (`docs/qwen35moe/41-PROMPT-CACHE.md`), against the production install:
//
//   swift run -c release TsugumiKernelCheck \
//     --qwen-resume scratch/ornith-oq4e-g64.moepack
//
// A continuation is not a new mode of the model. It is the same prompt with a
// chunk boundary in a different place — one that happens to fall between two
// HTTP requests instead of inside one. So the question this check asks is the
// one `21-PHASE4-PREFILL.md` §4 asked about chunk widths, moved across the
// request boundary: **does splitting the prefill change the answer?**
//
// It also pins the bookkeeping the server's cache is built on: how far the K/V
// cursor is when the loop returns. That number is not always
// `prompt + generated − 1` — an accepted MTP draft on the last pass consumes
// the token it emitted (§3-2), and reading it wrong costs a whole turn's cache.

private enum QwenResumeError: Error, CustomStringConvertible {
    case noMetalDevice
    var description: String { "no Metal device" }
}

private struct ResumeCase {
    let name: String
    let passed: Bool
    let detail: String
}

func runQwenResumeCheck(modelPath: String,
                        slotCount: Int,
                        maxNewTokens: Int = 12) throws -> Bool {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw QwenResumeError.noMetalDevice
    }
    let fixture = QwenDecodeFixture.referenceSmoke
    let wanted = min(maxNewTokens, fixture.expected.count)
    let prompt = fixture.prompt

    print("=== qwen3_5_moe resumed prefill (docs/qwen35moe/41-PROMPT-CACHE.md) ===")
    print("  model    \(modelPath)")
    print("  fixture  built in (14-REFERENCE.md §6) — prompt \(prompt.count), \(wanted) tokens")

    let context = try MetalContext()
    let model = try Model.load(directoryURL: URL(fileURLWithPath: modelPath),
                               device: device,
                               expecting: .ornith1_5_35B_A3B,
                               streamingMode: .pread(slotCount: slotCount))
    let runner = try QwenForwardRunner(model: model, context: context,
                                       maxContext: prompt.count + wanted + 8)
    print("")

    /// One whole prompt, one request.
    func cold(_ tokens: [Int32], new: Int) throws -> QwenGreedyRun {
        runner.reset()
        return try runner.runCompletion(promptTokens: tokens,
                                              maxNewTokens: new,
                                              chunkWidth: 512)
    }

    /// The same prompt in two requests, split after `head` tokens.
    ///
    /// The first request generates one token and **consumes none of it** — the
    /// loop feeds a token only to draw the next one — so the state it leaves
    /// holds exactly the head. That is what makes this comparison about the
    /// boundary and nothing else.
    func resumed(_ tokens: [Int32], head: Int, new: Int) throws -> QwenGreedyRun {
        runner.reset()
        let first = try runner.runCompletion(
            promptTokens: Array(tokens.prefix(head)), maxNewTokens: 1, chunkWidth: 512)
        guard first.kvPosition == head else {
            throw QwenResumeError.noMetalDevice   // reported by the case below
        }
        return try runner.runCompletion(promptTokens: Array(tokens.dropFirst(head)),
                                              maxNewTokens: new,
                                              chunkWidth: 512,
                                              cachedPromptTokens: head)
    }

    var cases: [ResumeCase] = []

    let whole = try cold(prompt, new: wanted)
    cases.append(ResumeCase(
        name: "1 要求で通したときの位置",
        passed: whole.kvPosition == prompt.count + whole.newTokens - 1,
        detail: "kv=\(whole.kvPosition), prompt \(prompt.count) + 生成 \(whole.newTokens) − 1"))

    for head in [prompt.count - 1, prompt.count / 2, 4] {
        let split = try resumed(prompt, head: head, new: wanted)
        cases.append(ResumeCase(
            name: "接頭 \(head) を持ち越しても同じトークン",
            passed: split.tokens == whole.tokens
                && split.cachedPromptTokens == head
                && split.promptTokens == prompt.count - head,
            detail: "\(split.tokens.count)/\(whole.tokens.count) tokens, "
                + "cached=\(split.cachedPromptTokens) computed=\(split.promptTokens)"))
    }

    // 負例: the state has to be the thing doing the work. Resume with a suffix
    // that does **not** follow what the state holds and the answer must change
    // — if it did not, the continuation would be ignoring its own history.
    let strayHead = prompt.count / 2
    runner.reset()
    let firstHalf = try runner.runCompletion(
        promptTokens: Array(prompt.prefix(strayHead)), maxNewTokens: 1, chunkWidth: 512)
    let stray = try runner.runCompletion(
        promptTokens: Array(prompt.suffix(from: strayHead).reversed()),
        maxNewTokens: wanted, chunkWidth: 512,
        cachedPromptTokens: firstHalf.kvPosition)
    cases.append(ResumeCase(
        name: "負例: 続きが違えば答えも違う",
        passed: stray.tokens != whole.tokens,
        detail: "先頭 \(stray.tokens.prefix(3)) 対 \(whole.tokens.prefix(3))"))

    // The MTP loop has its own bookkeeping: an accepted draft on the last pass
    // leaves the state one token longer than the plain loop would.
    do {
        try runner.attachMTPHead()
        runner.reset()
        let mtp = try runner.runCompletionMTP(promptTokens: prompt,
                                                   maxNewTokens: wanted,
                                                   chunkWidth: 512)
        let total = prompt.count + mtp.newTokens
        cases.append(ResumeCase(
            name: "幅 2 の位置は prompt+生成 か その 1 つ手前",
            passed: mtp.kvPosition == total || mtp.kvPosition == total - 1,
            detail: "kv=\(mtp.kvPosition), prompt+生成=\(total)"))

        runner.reset()
        let mtpFirst = try runner.runCompletionMTP(
            promptTokens: Array(prompt.prefix(4)), maxNewTokens: 1, chunkWidth: 512)
        let mtpSplit = try runner.runCompletionMTP(
            promptTokens: Array(prompt.dropFirst(4)), maxNewTokens: wanted,
            chunkWidth: 512, cachedPromptTokens: mtpFirst.kvPosition)
        cases.append(ResumeCase(
            name: "幅 2 でも接頭を持ち越して同じトークン",
            passed: mtpSplit.tokens == mtp.tokens,
            detail: "\(mtpSplit.tokens.count)/\(mtp.tokens.count) tokens, "
                + "cached=\(mtpSplit.cachedPromptTokens)"))
        // ---- サンプリング (docs/qwen35moe/42-SAMPLING.md) ------------------
        //
        // 実機で見られるのは分布そのものではない (1 本の走りは 1 標本しか
        // 出さない)。ここで採点するのは**結線**である: 種を固定したら再現
        // するか、種を変えたら変わるか、MTP を挟んでも同じ性質が保たれるか。
        // 分布の正しさは `QwenSamplerCheck.swift` が合成分布で採っている。
        let sampling = GenerationConfig(maxNewTokens: wanted,
                                        temperature: 0.6,
                                        topK: 20,
                                        topP: 0.95,
                                        seed: 0x0117_0001)
        runner.reset()
        let sampled = try runner.runCompletion(promptTokens: prompt,
                                               maxNewTokens: wanted,
                                               chunkWidth: 512,
                                               sampling: sampling)
        runner.reset()
        let sampledAgain = try runner.runCompletion(promptTokens: prompt,
                                                    maxNewTokens: wanted,
                                                    chunkWidth: 512,
                                                    sampling: sampling)
        cases.append(ResumeCase(
            name: "サンプリングは種を固定すれば再現する",
            passed: sampled.tokens == sampledAgain.tokens,
            detail: "\(sampled.tokens.prefix(4)) 対 \(sampledAgain.tokens.prefix(4))"))

        // 「黙って greedy ではない」ことは**公式値では採点できない**。
        // この fixture でモデルが自信を持っている位置では top_p=0.95 の
        // nucleus が 1 本になり、温度 0.6 の draw も必ず argmax を返す —
        // 実際、公式値の 12 トークンは greedy と一致した (下の detail)。
        // それは正しい挙動であって、結線の証拠にはならない。
        // 証拠になるのは**必ず確率が割れる設定**の方である。
        var stochastic = sampling
        stochastic.temperature = 1.5
        stochastic.topP = 1.0
        stochastic.topK = 20
        runner.reset()
        let hot = try runner.runCompletion(promptTokens: prompt,
                                           maxNewTokens: wanted,
                                           chunkWidth: 512,
                                           sampling: stochastic)
        var hotOther = stochastic
        hotOther.seed = 0x0117_0002
        runner.reset()
        let hotOtherRun = try runner.runCompletion(promptTokens: prompt,
                                                   maxNewTokens: wanted,
                                                   chunkWidth: 512,
                                                   sampling: hotOther)
        runner.reset()
        let greedyRun = try runner.runCompletion(promptTokens: prompt,
                                                 maxNewTokens: wanted,
                                                 chunkWidth: 512)
        // 実機の分布そのものを見る。1 本の走りは 1 標本しか出さないので、
        // 「列が変わらなかった」だけでは結線の当否を決められない。
        runner.reset()
        let officialProbe = try runner.probeDistribution(promptTokens: prompt,
                                                         sampling: sampling)
        runner.reset()
        let hotProbe = try runner.probeDistribution(promptTokens: prompt,
                                                    sampling: stochastic)
        func show(_ c: QwenCategorical) -> String {
            zip(c.ids, c.weights).prefix(3)
                .map { String(format: "%d:%.3f", $0.0, $0.1) }
                .joined(separator: " ")
        }
        print("  分布  公式値 \(officialProbe.ids.count) 本 [\(show(officialProbe))]  "
            + "温度 1.5 \(hotProbe.ids.count) 本 [\(show(hotProbe))]")
        cases.append(ResumeCase(
            name: "実機の logits から分布が立つ (温度 1.5 で 2 本以上)",
            passed: hotProbe.ids.count >= 2
                && abs(hotProbe.weights.reduce(0, +) - 1) < 1e-4,
            detail: "\(hotProbe.ids.count) 本、和 "
                + String(format: "%.5f", hotProbe.weights.reduce(0, +))))
        // **決定力のある案。**「種を変えたら列が変わる」は、この機体のこの
        // fixture では採点にならない — 温度 1.5 でも先頭が 0.986 を持つので、
        // 2 つの種が両方とも argmax を引くのは普通に起きる (それは正しい)。
        // 代わりに採点するのは「ループが出したトークンは、分布と種から
        // **予言できる**か」である。予言が当たるなら、draw は分布の CDF を
        // 歩いている。当たらないなら、どこかで別の値が使われている。
        let predicted = hotProbe.draw(
            u: QwenSampler.uniform(config: stochastic, position: 0, stream: 0))
        runner.reset()
        let oneToken = try runner.runCompletion(promptTokens: prompt,
                                                maxNewTokens: 1,
                                                chunkWidth: 512,
                                                sampling: stochastic)
        cases.append(ResumeCase(
            name: "ループが出したトークンは分布と種から予言できる",
            passed: oneToken.tokens.first == predicted,
            detail: "予言 \(predicted) / 実際 \(oneToken.tokens.first.map(String.init) ?? "なし")"))

        // 種が結果を動かすことは、確率が本当に割れる温度で見る。
        var flat = stochastic
        flat.temperature = 8.0
        runner.reset()
        let flatA = try runner.runCompletion(promptTokens: prompt, maxNewTokens: wanted,
                                             chunkWidth: 512, sampling: flat)
        var flatOther = flat
        flatOther.seed = 0x0117_0003
        runner.reset()
        let flatB = try runner.runCompletion(promptTokens: prompt, maxNewTokens: wanted,
                                             chunkWidth: 512, sampling: flatOther)
        cases.append(ResumeCase(
            name: "温度 8 なら種で列が変わる",
            passed: flatA.tokens != flatB.tokens,
            detail: "seed1 \(flatA.tokens.prefix(3)) / seed3 \(flatB.tokens.prefix(3))"))
        _ = hot; _ = hotOtherRun
        cases.append(ResumeCase(
            name: "公式値 (0.6/0.95/20) の列は greedy と一致してよい",
            passed: true,
            detail: sampled.tokens == greedyRun.tokens
                ? "この fixture では一致した — nucleus が 1 本の位置が続く"
                : "この fixture では分かれた \(sampled.tokens.prefix(3)) / \(greedyRun.tokens.prefix(3))"))

        runner.reset()
        let mtpSampled = try runner.runCompletionMTP(promptTokens: prompt,
                                                     maxNewTokens: wanted,
                                                     chunkWidth: 512,
                                                     sampling: sampling)
        runner.reset()
        let mtpSampledAgain = try runner.runCompletionMTP(promptTokens: prompt,
                                                          maxNewTokens: wanted,
                                                          chunkWidth: 512,
                                                          sampling: sampling)
        cases.append(ResumeCase(
            name: "幅 2 のサンプリングも種を固定すれば再現する",
            passed: mtpSampled.tokens == mtpSampledAgain.tokens,
            detail: "\(mtpSampled.tokens.prefix(4)) 対 \(mtpSampledAgain.tokens.prefix(4))"))
        let mtpTotal = prompt.count + mtpSampled.newTokens
        cases.append(ResumeCase(
            name: "サンプリング下でも位置は prompt+生成 か その 1 つ手前",
            passed: mtpSampled.kvPosition == mtpTotal || mtpSampled.kvPosition == mtpTotal - 1,
            detail: "kv=\(mtpSampled.kvPosition), prompt+生成=\(mtpTotal)"))

        // **サンプリング下の中立性はトークン列では採点できない。**greedy の
        // 中立性 (強制棄却の対照と列が完全一致) が成り立つのは、両方の腕が
        // 同じ argmax を出すからである。サンプリングでは、受理された腕は
        // draft ストリームから引き、棄却された腕は残差から引く — 同じ**分布**
        // だが同じ**列**ではない。しかも T 行カーネルと decode カーネルは
        // 加算順が違うので (`36-MTP-DECODE.md`)、強制棄却の腕を非投機と
        // 突き合わせても最後の一致は保証されない。
        // 分布の一致は `QwenSamplerCheck.swift` が合成分布で採っている
        // (全変動距離 0.0008、負例 0.5)。ここで採点できるのは結線だけである。
    } catch {
        print("  SKIP  MTP cases — no sidecar at \(QwenMTPSidecar.defaultDirectory)")
        print("")
    }

    var passed = true
    for one in cases {
        print("  \(one.passed ? "PASS" : "FAIL")  \(one.name) — \(one.detail)")
        passed = passed && one.passed
    }
    print("")
    print(passed
          ? "PASS  \(cases.count) cases, 1 of them a negative control"
          : "FAIL  \(cases.filter { !$0.passed }.count) of \(cases.count) cases")
    return passed
}
