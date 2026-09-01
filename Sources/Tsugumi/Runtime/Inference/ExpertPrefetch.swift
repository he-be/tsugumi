import Foundation

/// 層 L の router 入力に層 L+d の router 重みを当てて出した予測を、**実際に
/// fetch へ渡す**経路 (`docs/mtp/29-M8-B-PROBE.md` §6)。
///
/// 計器 (`RouterPreviewProbe`) が測ったのは「当たるか」だけで、当たった分を
/// 前倒しに読むところは入っていなかった。29 §4 の実測では予測を全部読むと
/// バイトが 1.72 倍になって 32 スロットでは自滅するので、**予測重みの強い順に
/// 上から N 本だけ**投げる。N=1 なら的中 70% / バイト 1.07 倍で、
/// 導出でブロック 82 → 約 66 ms である (§5)。
///
/// 先読みはヒントで、実行は正規の routing のまま — 読むバイトの中身も、どの
/// エキスパートを使うかも変わらない。変わるのは**どのスロットに置くか**と
/// **いつ読み始めるか**だけなので、出力は 1 バイトも動かない (§6 の「正しさ」)。
enum ExpertPrefetch {
    static let isEnabled =
        ProcessInfo.processInfo.environment["TF_MTP_EXPERT_PREFETCH"] == "1"

    /// 層あたり何本まで先読みするか。29 §4/§5 の実測と導出から既定は 1:
    /// N=2〜4 は露出が減るぶんバイトが増えて横ばいで、N=1 だけがバイトを
    /// 7% しか増やさない = 32 スロットの追い出し合戦のリスクが最小である。
    /// `0` は「予測はするが 1 本も投げない」= 予測そのものの費用 (層 L+d の
    /// router をもう 1 本 GPU で回す分) だけを測るための設定である。
    static let topN: Int = {
        guard let raw = ProcessInfo.processInfo.environment["TF_MTP_EXPERT_PREFETCH_N"],
              let value = Int(raw), value >= 0 else { return 1 }
        return min(value, RouterPreviewProbe.maxRank)
    }()

    /// 何層先を読むか。d=1 は実ミスの 66% に名前が付き、d=2 は 57% (29 §2)。
    /// リードタイムは d=2 のほうが長いので、実機で振れるようにしてある。
    static let distance: Int = {
        guard let raw = ProcessInfo.processInfo.environment["TF_MTP_EXPERT_PREFETCH_D"],
              let value = Int(raw), value >= 1 else { return 1 }
        return min(value, 4)
    }()
}

/// 飛行中の先読みの控え。対象の層ごとに 1 本まで。
///
/// **待つのは必須である。**`executeExpertCachePlan` は読み終わってから
/// `expertResidency` を上げるので、完了した先読みは層 L+d のプランに
/// ただのヒットとして見える — ここは何も足さなくてよい。危ないのは飛行中で、
/// そのときスロットはまだ空きに見えるため、層 L+d のプランが**同じスロットを
/// 犠牲に選んで二重書きしうる**。だから層 L+d のプランを作る前に、
/// そしてチャンクを抜ける前に、必ず `wait()` を通す (29 §6)。
struct ExpertPrefetchState {
    private struct Pending {
        let handle: RoutedExpertFetchHandle
        /// 読みに行ったエキスパート。層 L+d が本当に要求したかを後で突き合わせる。
        let experts: [Int]
    }
    private var pending: [Int: Pending] = [:]
    /// 発行した層の数と、そのとき読むことにしたエキスパートの延べ本数。
    private var issuedLayers = 0
    private var issuedReads = 0
    /// 「直近のラウンドで使ったスロットしか空いていない」で諦めた回数。
    /// ここが大きいなら 32 スロットに先読みの余地が無いということである。
    private var declined = 0
    /// 先読みの読み出しが失敗した回数。先読みはヒントなので握り潰して先へ
    /// 進む — 失敗したスロットは planning で空きに戻されており、層 L+d が
    /// 普通に読み直す。数だけ残して summary に出す。
    private var failed = 0
    /// 先読みしたエキスパートのうち、その層が本当に要求した数。
    /// **これが先読みの的中率で、hit 率ではない** — 先読みの read は miss として
    /// 数えられるので telemetry の hit 率は必ず下がる (29-M8-B §7)。
    private var useful = 0

    /// ブロック 1 回の始まり。前のラウンドの控えは `executePrefillChunk` の
    /// `defer` で回収済みのはずだが、待たずに捨てるのがいちばん危ない扱いな
    /// ので、残っていたらここで待つ。
    mutating func beginBlock() {
        drain()
    }

    func isPending(layer: Int) -> Bool { pending[layer] != nil }

    mutating func issued(layer: Int, experts: [Int], handle: RoutedExpertFetchHandle) {
        pending[layer] = Pending(handle: handle, experts: experts)
        issuedLayers += 1
        issuedReads += experts.count
    }

    mutating func decline() { declined += 1 }

    /// 層 `layer` の先読みを待って手放す。プランを作る直前に呼ぶ。
    /// 返るのは読みに行ったエキスパート — 呼び手はそれを `score` に渡す。
    @discardableResult
    mutating func wait(layer: Int) -> [Int] {
        guard let entry = pending.removeValue(forKey: layer) else { return [] }
        do { _ = try entry.handle.wait() } catch { failed += 1 }
        return entry.experts
    }

    /// 待った先読みが当たっていたかを、その層が実際に要求した集合で採点する。
    mutating func score(prefetched: [Int], requested: Set<Int>) {
        for expert in prefetched where requested.contains(expert) { useful += 1 }
    }

    /// 残っている控えを全部待つ。チャンクを抜けるときに必ず通す:
    /// 待たずに捨てると、飛行中の読み出しが別の用途に割り当て直された
    /// スロットへ書き込みうる。
    mutating func drain() {
        for layer in pending.keys.sorted() { wait(layer: layer) }
    }

    var summary: String? {
        guard ExpertPrefetch.isEnabled, issuedLayers > 0 || declined > 0 else { return nil }
        return String(
            format: "[expert prefetch n=%d d=%d layers=%d reads=%d useful=%.0f%%(%d) "
                + "declined=%d failed=%d]",
            ExpertPrefetch.topN, ExpertPrefetch.distance,
            issuedLayers, issuedReads,
            issuedReads > 0 ? Double(useful) / Double(issuedReads) * 100 : 0, useful,
            declined, failed)
    }

    mutating func reset() {
        issuedLayers = 0
        issuedReads = 0
        declined = 0
        failed = 0
        useful = 0
    }
}
