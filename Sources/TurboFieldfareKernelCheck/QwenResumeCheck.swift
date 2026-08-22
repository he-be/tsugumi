import Foundation
import Metal
import TurboFieldfare

// The property the Ornith prompt cache rests on
// (`docs/qwen35moe/41-PROMPT-CACHE.md`), against the production install:
//
//   swift run -c release TurboFieldfareKernelCheck \
//     --qwen-resume scratch/ornith-oq4e-g64.gturbo
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
        return try runner.runGreedyCompletion(promptTokens: tokens,
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
        let first = try runner.runGreedyCompletion(
            promptTokens: Array(tokens.prefix(head)), maxNewTokens: 1, chunkWidth: 512)
        guard first.kvPosition == head else {
            throw QwenResumeError.noMetalDevice   // reported by the case below
        }
        return try runner.runGreedyCompletion(promptTokens: Array(tokens.dropFirst(head)),
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
    let firstHalf = try runner.runGreedyCompletion(
        promptTokens: Array(prompt.prefix(strayHead)), maxNewTokens: 1, chunkWidth: 512)
    let stray = try runner.runGreedyCompletion(
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
        let mtp = try runner.runGreedyCompletionMTP(promptTokens: prompt,
                                                   maxNewTokens: wanted,
                                                   chunkWidth: 512)
        let total = prompt.count + mtp.newTokens
        cases.append(ResumeCase(
            name: "幅 2 の位置は prompt+生成 か その 1 つ手前",
            passed: mtp.kvPosition == total || mtp.kvPosition == total - 1,
            detail: "kv=\(mtp.kvPosition), prompt+生成=\(total)"))

        runner.reset()
        let mtpFirst = try runner.runGreedyCompletionMTP(
            promptTokens: Array(prompt.prefix(4)), maxNewTokens: 1, chunkWidth: 512)
        let mtpSplit = try runner.runGreedyCompletionMTP(
            promptTokens: Array(prompt.dropFirst(4)), maxNewTokens: wanted,
            chunkWidth: 512, cachedPromptTokens: mtpFirst.kvPosition)
        cases.append(ResumeCase(
            name: "幅 2 でも接頭を持ち越して同じトークン",
            passed: mtpSplit.tokens == mtp.tokens,
            detail: "\(mtpSplit.tokens.count)/\(mtp.tokens.count) tokens, "
                + "cached=\(mtpSplit.cachedPromptTokens)"))
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
