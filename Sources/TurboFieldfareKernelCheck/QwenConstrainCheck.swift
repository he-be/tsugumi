import Foundation
import Metal
import TurboFieldfare

// GEN-7 on the production install: the constraint hook of `QwenForwardRunner`,
// driven with stub constraints instead of a grammar
// (`docs/qwen35moe/25-CLI-TOOLS.md` §2).
//
//   swift run -c release TurboFieldfareKernelCheck \
//     --qwen-constrain scratch/ornith-oq4e-g64.gturbo
//
// What is already proven elsewhere, and deliberately not re-proven here: the
// masked kernel picks the largest allowed row (`--qwen`, six cases against a
// CPU reference), and the tool grammar accepts what the template writes
// (`--qwen-tools`, 36 cases against the real vocabulary). What neither of them
// touches is the seam between them — the whole-vocabulary mask being packed
// into bits at 248,077 rows, the rejected token being re-scored against the
// *same* hidden row, and `ConstraintGate`'s rule for the stop tokens.
//
// A stub rather than a grammar is the point: a grammar's verdict is a property
// of the model's text, so a run that never rejects proves nothing about the
// rejection path. These constraints reject on demand.

private enum QwenConstrainError: Error, CustomStringConvertible {
    case noMetalDevice
    var description: String { "no Metal device" }
}

/// A constraint whose verdict is set by the check rather than derived from
/// anything. `allowed` nil means "everything".
private final class StubSetConstraint: GenerationConstraint, @unchecked Sendable {
    var allowed: Set<Int32>?
    var mayEndHere: Bool
    /// Every token the decode loop kept, in order — the sequence `accept` is
    /// contracted to see.
    private(set) var accepted: [Int32] = []
    /// A deliberately inconsistent witness: `allows` answers this instead of
    /// reading `allowed`, so the probe and the mask can be made to disagree.
    var probeOverride: Bool?

    init(allowed: Set<Int32>? = nil, mayEndHere: Bool = true) {
        self.allowed = allowed
        self.mayEndHere = mayEndHere
    }

    func allows(tokenID: Int32) -> Bool {
        if let probeOverride { return probeOverride }
        guard let allowed else { return true }
        return allowed.contains(tokenID)
    }

    func fillAllowedMask(_ mask: UnsafeMutableBufferPointer<Bool>) throws {
        guard let allowed else {
            mask.update(repeating: true)
            return
        }
        mask.update(repeating: false)
        for id in allowed where id >= 0 && Int(id) < mask.count {
            mask[Int(id)] = true
        }
    }

    func accept(tokenID: Int32) throws { accepted.append(tokenID) }
}

private struct ConstrainCase {
    let name: String
    let passed: Bool
    let detail: String
}

func runQwenConstrainCheck(modelPath: String,
                           slotCount: Int,
                           maxNewTokens: Int = 8) throws -> Bool {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw QwenConstrainError.noMetalDevice
    }
    let fixture = QwenDecodeFixture.referenceSmoke
    let wanted = min(maxNewTokens, fixture.expected.count)
    let reference = Array(fixture.expected.prefix(wanted))

    print("=== qwen3_5_moe constrained greedy (docs/qwen35moe/25-CLI-TOOLS.md §2) ===")
    print("  model    \(modelPath)")
    print("  fixture  built in (14-REFERENCE.md §6) — prompt \(fixture.prompt.count), "
          + "\(wanted) tokens")

    let context = try MetalContext()
    let model = try Model.load(directoryURL: URL(fileURLWithPath: modelPath),
                               device: device,
                               expecting: .ornith1_5_35B_A3B,
                               streamingMode: .pread(slotCount: slotCount))
    let runner = try QwenForwardRunner(model: model, context: context,
                                       maxContext: fixture.prompt.count + wanted + 8)
    print("  scoring \(runner.scoredVocab) vocabulary rows")
    print("")

    func generate(_ constraint: StubSetConstraint,
                  stopTokens: Set<Int32> = [],
                  tokens: Int? = nil) throws -> (produced: [Int32], rescores: Int) {
        runner.reset()
        runner.resetProfile()
        let produced = try runner.generateGreedyPrefilled(
            promptTokens: fixture.prompt,
            maxNewTokens: tokens ?? wanted,
            chunkWidth: 512,
            stopTokens: stopTokens,
            constraint: constraint)
        return (produced, runner.constraintRescores)
    }

    /// The same generation through the width-2 MTP loop
    /// (`docs/qwen35moe/40-MTP-GRAMMAR.md`). Nil when the sidecar is not on
    /// this machine, which is a skip and not a failure: the head is a 480 MB
    /// artifact beside the checkpoints, not part of the install.
    func generateMTP(_ constraint: StubSetConstraint,
                     stopTokens: Set<Int32> = [],
                     tokens: Int? = nil) throws -> (produced: [Int32], rescores: Int) {
        runner.reset()
        runner.resetProfile()
        let run = try runner.runCompletionMTP(
            promptTokens: fixture.prompt,
            maxNewTokens: tokens ?? wanted,
            chunkWidth: 512,
            stopTokens: stopTokens,
            constraint: constraint)
        return (run.tokens, runner.constraintRescores)
    }

    var cases: [ConstrainCase] = []

    // 1. A constraint that never rejects must be invisible. Same tokens as the
    //    unconstrained reference, and not one extra head pass.
    let transparent = StubSetConstraint()
    let clear = try generate(transparent)
    cases.append(ConstrainCase(
        name: "許可が全部なら参照と同じトークン",
        passed: clear.produced == reference && clear.rescores == 0,
        detail: "rescored=\(clear.rescores), \(clear.produced.count) tokens"))
    // 2. …and the constraint saw exactly what was emitted, in order.
    cases.append(ConstrainCase(
        name: "accept は出たトークンを順に受け取る",
        passed: transparent.accepted == clear.produced,
        detail: "accepted \(transparent.accepted.count) of \(clear.produced.count)"))

    // 3. Forbid the token the model wants, and only that one. The answer has to
    //    change at step 0 and nowhere else has to be paid for.
    var everythingElse = Set<Int32>(0..<Int32(runner.scoredVocab))
    everythingElse.remove(reference[0])
    let withoutWinner = StubSetConstraint(allowed: everythingElse)
    let second = try generate(withoutWinner)
    cases.append(ConstrainCase(
        name: "勝者を落とすと 1 回だけ畳み直して別のトークンが出る",
        passed: second.produced.first != reference[0]
            && second.rescores == 1
            && second.produced.count == wanted,
        detail: "step 0 = \(second.produced.first.map(String.init) ?? "-") "
            + "(参照 \(reference[0])), rescored=\(second.rescores)"))

    // 4. One allowed row out of 248,077, at every step. Nothing but the packing
    //    decides which row that is, so a bit index off by anything at all
    //    produces a different id — this is the case that scales the synthetic
    //    "許可が 1 本" case up to the real vocabulary width.
    let onlyID = Int32((Int(reference[0]) + 4_099) % runner.scoredVocab)
    let single = StubSetConstraint(allowed: [onlyID])
    let pinned = try generate(single, tokens: 4)
    cases.append(ConstrainCase(
        name: "許可が 1 本なら毎手それが出る (語彙 \(runner.scoredVocab) 行)",
        passed: pinned.produced == [Int32](repeating: onlyID, count: 4)
            && pinned.rescores == 4,
        detail: "produced \(pinned.produced), rescored=\(pinned.rescores)"))

    // 5. An empty mask is an error, not a stop (GEN-7).
    let nothing = StubSetConstraint(allowed: [])
    var emptyThrew = ""
    do {
        _ = try generate(nothing, tokens: 1)
    } catch let error as GenerationConstraintError {
        emptyThrew = "\(error)"
    }
    cases.append(ConstrainCase(
        name: "許可が 0 本なら noAllowedToken",
        passed: emptyThrew.contains("allows no token"),
        detail: emptyThrew.isEmpty ? "落ちなかった" : emptyThrew))

    // 6. 負例: the probe and the mask are the same constraint answering twice.
    //    A constraint that refuses everything through `allows` while its mask
    //    permits everything must be caught, not silently obeyed — the token the
    //    masked pass returns is one the probe rejects.
    let inconsistent = StubSetConstraint()
    inconsistent.probeOverride = false
    var mismatchThrew = ""
    do {
        _ = try generate(inconsistent, tokens: 1)
    } catch let error as GenerationConstraintError {
        mismatchThrew = "\(error)"
    }
    cases.append(ConstrainCase(
        name: "負例: 単票とマスクが食い違えば落とす",
        passed: mismatchThrew.contains("rejected token"),
        detail: mismatchThrew.isEmpty ? "落ちなかった" : mismatchThrew))

    // 7. The stop tokens are decided by `mayEndHere` alone (`ConstraintGate`).
    //    The same run, twice, differing only in that flag: it stops where the
    //    model wanted to, or the stop id is masked and the run continues.
    let stopID = reference[0]
    let mayEnd = StubSetConstraint(mayEndHere: true)
    let ended = try generate(mayEnd, stopTokens: [stopID])
    let mayNotEnd = StubSetConstraint(mayEndHere: false)
    let continued = try generate(mayNotEnd, stopTokens: [stopID])
    cases.append(ConstrainCase(
        name: "mayEndHere が真なら停止トークンは通る",
        passed: ended.produced == [stopID] && ended.rescores == 0,
        detail: "produced \(ended.produced)"))
    cases.append(ConstrainCase(
        name: "mayEndHere が偽なら停止トークンは隠れる",
        passed: continued.produced.first != stopID
            && !continued.produced.contains(stopID)
            && continued.rescores >= 1,
        detail: "step 0 = \(continued.produced.first.map(String.init) ?? "-"), "
            + "rescored=\(continued.rescores)"))
    // …and the constraint never ruled on the stop id it let through.
    cases.append(ConstrainCase(
        name: "通した停止トークンは accept に渡らない",
        passed: mayEnd.accepted.isEmpty,
        detail: "accepted \(mayEnd.accepted)"))

    // --- the same questions through the width-2 verify pass ---------------
    //
    // What changes with the MTP head in the loop is *where* the row being
    // rescored lives: the two draws come out of the verify scratch, not out of
    // `head.normalizedHidden` (`docs/qwen35moe/40-MTP-GRAMMAR.md` §2). A
    // rescore that read the wrong row would still return a legal token — one
    // the mask allows — so the cases that catch it are the ones that pin the
    // answer: the transparent constraint (must reproduce the reference) and the
    // single-allowed-row one (must produce that row at every step).
    let headAttached: Bool
    do {
        try runner.attachMTPHead()
        headAttached = true
    } catch {
        headAttached = false
        print("  SKIP  MTP cases — no sidecar at \(QwenMTPSidecar.defaultDirectory)")
        print("")
    }
    if headAttached {
        let transparentMTP = StubSetConstraint()
        let clearMTP = try generateMTP(transparentMTP)
        cases.append(ConstrainCase(
            name: "幅 2: 許可が全部なら参照と同じトークン",
            passed: clearMTP.produced == reference && clearMTP.rescores == 0,
            detail: "rescored=\(clearMTP.rescores), \(clearMTP.produced.count) tokens"))
        cases.append(ConstrainCase(
            name: "幅 2: accept は出たトークンを順に受け取る",
            passed: transparentMTP.accepted == clearMTP.produced,
            detail: "accepted \(transparentMTP.accepted.count) "
                + "of \(clearMTP.produced.count)"))

        // The rejected row is row 0 of the verify scratch. The answer must be
        // the one the per-token loop drew from the same hidden state.
        let withoutWinnerMTP = StubSetConstraint(allowed: everythingElse)
        let secondMTP = try generateMTP(withoutWinnerMTP)
        cases.append(ConstrainCase(
            name: "幅 2: 勝者を落とすと素の decode と同じ別トークンが出る",
            passed: secondMTP.produced == second.produced && secondMTP.rescores == 1,
            detail: "step 0 = \(secondMTP.produced.first.map(String.init) ?? "-") "
                + "(素の decode \(second.produced.first.map(String.init) ?? "-")), "
                + "rescored=\(secondMTP.rescores)"))

        let singleMTP = StubSetConstraint(allowed: [onlyID])
        let pinnedMTP = try generateMTP(singleMTP, tokens: 4)
        cases.append(ConstrainCase(
            name: "幅 2: 許可が 1 本なら毎手それが出る",
            passed: pinnedMTP.produced == [Int32](repeating: onlyID, count: 4),
            detail: "produced \(pinnedMTP.produced), rescored=\(pinnedMTP.rescores)"))

        // 負例, twice: an empty mask is still an error, and the probe and the
        // mask still have to agree — both verdicts now come from a row the
        // speculative loop chose.
        let nothingMTP = StubSetConstraint(allowed: [])
        var emptyThrewMTP = ""
        do {
            _ = try generateMTP(nothingMTP, tokens: 1)
        } catch let error as GenerationConstraintError {
            emptyThrewMTP = "\(error)"
        }
        cases.append(ConstrainCase(
            name: "負例: 幅 2 でも許可 0 本は noAllowedToken",
            passed: emptyThrewMTP.contains("allows no token"),
            detail: emptyThrewMTP.isEmpty ? "落ちなかった" : emptyThrewMTP))

        let inconsistentMTP = StubSetConstraint()
        inconsistentMTP.probeOverride = false
        var mismatchThrewMTP = ""
        do {
            _ = try generateMTP(inconsistentMTP, tokens: 1)
        } catch let error as GenerationConstraintError {
            mismatchThrewMTP = "\(error)"
        }
        cases.append(ConstrainCase(
            name: "負例: 幅 2 でも単票とマスクの食い違いは落とす",
            passed: mismatchThrewMTP.contains("rejected token"),
            detail: mismatchThrewMTP.isEmpty ? "落ちなかった" : mismatchThrewMTP))
    }

    var passed = true
    for one in cases {
        print("  \(one.passed ? "PASS" : "FAIL")  \(one.name) — \(one.detail)")
        passed = passed && one.passed
    }
    print("")
    print(passed
          ? "PASS  \(cases.count) cases, \(headAttached ? 3 : 1) of them negative controls"
          : "FAIL  \(cases.filter { !$0.passed }.count) of \(cases.count) cases")
    return passed
}
