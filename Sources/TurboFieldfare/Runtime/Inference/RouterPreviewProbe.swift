import Foundation

/// 「層 L の router 入力に層 L+d の router 重みを当てると、層 L+d が実際に選ぶ
/// エキスパートをどれだけ当てられるか」だけを測る計器。**出力は 1 バイトも
/// 変えない** — 予測は数えるだけで、routing にも fetch にも渡さない。
///
/// 何のためにあるか (`docs/mtp/28-M8-PROPOSAL.md` §3 の果実 B):
/// 32 スロットのブロックは 82 ms のうち 33 ms が expert の読み出しの露出で、
/// その 33 ms は「読むバイト数」ではなく「読み出しを発行できる時刻」で決まる
/// (27-M7 §7)。層 L+1 のルーティングは層 L の出力を待つので、今は層 L+1 の
/// fetch を層 L+1 に入ってからしか出せない。層 L の時点で層 L+d の router を
/// **先に**当てて当たるなら、その分だけ発行を前倒しできる。
///
/// 27 §7 の「予測はできない」は**履歴**からの予測 (直前ラウンドのミスの再要求は
/// 0.0%) の話で、これは計算そのものからの予測なので別物である。実際、
/// エキスパート番号の層間相関はランダムと区別がつかない (層 L の集合が層 L+1 を
/// 当てる率 5.9%、ランダム 6.2%) ので、番号の再利用では代わりにならない。
///
/// `TF_MTP_ROUTER_PREVIEW=1` のときだけ動く。距離は `TF_MTP_ROUTER_PREVIEW_D`
/// (既定 2 = 層 L で L+1 と L+2 を測る)。
struct RouterPreviewProbe {
    static let isEnabled =
        ProcessInfo.processInfo.environment["TF_MTP_ROUTER_PREVIEW"] == "1"
    /// いくつ先の層まで測るか。1 だけだと「先読みできる時間」が層の前半 (約
    /// 0.5 ms) しか取れないので、既定で 2 まで見る。
    static let maxDistance: Int = {
        guard let raw = ProcessInfo.processInfo.environment["TF_MTP_ROUTER_PREVIEW_D"],
              let value = Int(raw), value >= 1 else { return 2 }
        return min(value, 4)
    }()

    /// 距離 1 つぶんの積み上げ。すべて「エキスパート個数」の合計で、率は
    /// 最後に割って出す。
    private struct Counters {
        /// 比べられた層の数 (= 予測が残っていた層の数)。
        var layers = 0
        /// 層が実際に要求したエキスパートの延べ数。
        var actual = 0
        /// そのうち予測が名前を挙げていた数。
        var hit = 0
        /// 実際に要求したうちスロットに載っていなかった数 = 読み出しが要る数。
        var miss = 0
        /// **本題**: そのミスのうち予測が名前を挙げていた数。前倒しで隠せる分。
        var missCovered = 0
        /// 予測が挙げた延べ数。`predicted - hit` が外れ = 無駄読みの上限。
        var predicted = 0
        /// 予測のうち、予測した時点で層 L+d のスロットに載っていなかった数 =
        /// 先読みするなら**実際に SSD を叩くことになる**数。
        var issued = 0
        /// その先読みのうち、層 L+d が本当に要求した数。`issued - useful` が
        /// 捨て読み — 32 スロットではこれがスロットの取り合いになる。
        var useful = 0
        /// 先読み候補を router の予測重みで強い順に並べたときの、順位ごとの
        /// 発行数と的中数。**全部読むと 1.7 倍のバイトになるので、上から何本
        /// までなら割に合うか**がここで決まる。
        var issuedAtRank = [Int](repeating: 0, count: maxRank)
        var usefulAtRank = [Int](repeating: 0, count: maxRank)
    }

    /// 順位別に見る本数。層あたりの候補は平均 6 本なので 8 で足りる。
    static let maxRank = 8

    private var counters: [Int: Counters] = [:]
    /// 層 -> 距離 -> その層について前もって出しておいた予測。
    /// ブロックの中でしか生きない (層をまたぐだけで、ラウンドはまたがない)。
    private struct Prediction {
        /// 予測した集合ぜんぶ。
        let experts: Set<Int>
        /// そのうち予測時点で常駐していなかったもの = 先読みが読むことになる分。
        /// 予測重みの強い順。先頭ほど「読んで当たる」見込みが高いはず、という
        /// 仮説をここで検定する。
        let nonResident: [Int]
    }
    private var pending: [Int: [Int: Prediction]] = [:]

    /// ブロック 1 回の始まり。前のラウンドの予測を持ち越さない。
    mutating func beginBlock() {
        guard Self.isEnabled else { return }
        pending.removeAll(keepingCapacity: true)
    }

    /// 層 `layer` の router 入力から出した「層 `layer + distance` の予測」。
    /// `nonResident` は予測した**その時点**で層 `layer + distance` のスロットに
    /// 載っていなかったもの — 先読みを入れたら実際に読むのはここだけである。
    mutating func record(fromLayer layer: Int,
                         distance: Int,
                         predicted: Set<Int>,
                         nonResident: [Int]) {
        guard Self.isEnabled else { return }
        pending[layer + distance, default: [:]][distance] =
            Prediction(experts: predicted, nonResident: nonResident)
    }

    /// 層 `layer` が実際に要求したエキスパートと、その時点の常駐状況。
    /// `resident` は `actual` と同じ並び。
    mutating func compare(layer: Int, actual: [Int], resident: [Bool]) {
        guard Self.isEnabled, actual.count == resident.count else { return }
        guard let predictions = pending.removeValue(forKey: layer) else { return }
        for (distance, prediction) in predictions {
            var c = counters[distance] ?? Counters()
            c.layers += 1
            c.actual += actual.count
            c.predicted += prediction.experts.count
            c.issued += prediction.nonResident.count
            let wanted = Set(actual)
            for (rank, expert) in prediction.nonResident.enumerated()
            where rank < Self.maxRank {
                c.issuedAtRank[rank] += 1
                if wanted.contains(expert) { c.usefulAtRank[rank] += 1 }
            }
            let usefulSet = Set(prediction.nonResident).intersection(wanted)
            c.useful += usefulSet.count
            for (index, expert) in actual.enumerated() {
                if prediction.experts.contains(expert) { c.hit += 1 }
                if !resident[index] {
                    c.miss += 1
                    if prediction.experts.contains(expert) { c.missCovered += 1 }
                }
            }
            counters[distance] = c
        }
    }

    /// 累積の 1 行。`cover` が果実 B の期待値をそのまま決める数字で、
    /// 露出している読み出しはおよそ `1 - cover` 倍になる。
    /// `waste` は当たらなかった先読みの割合 = SSD 帯域とスロットの浪費の上限。
    var summary: String? {
        guard Self.isEnabled, !counters.isEmpty else { return nil }
        let lines = counters.keys.sorted().map { distance -> String in
            let c = counters[distance]!
            func pct(_ n: Int, _ d: Int) -> Double { d > 0 ? Double(n) / Double(d) * 100 : 0 }
            return String(
                format: "[router preview d=%d layers=%d top8=%.1f%% cover=%.1f%%(%d/%d) "
                    + "issue=%.2f/layer useful=%.1f%% read=%.2fx]",
                distance, c.layers,
                pct(c.hit, c.actual),
                pct(c.missCovered, c.miss), c.missCovered, c.miss,
                c.layers > 0 ? Double(c.issued) / Double(c.layers) : 0,
                pct(c.useful, c.issued),
                // 先読みを入れたときに層が読む総バイトが今の何倍になるか。
                // 分子は「先読みが読む分 + 先読みが当てられなかった残りのミス」。
                c.miss > 0
                    ? Double(c.issued + c.miss - c.useful) / Double(c.miss) : 0)
        }
        let ranks = counters.keys.sorted().map { distance -> String in
            let c = counters[distance]!
            // 「上から n 本だけ先読みする」を積み上げで読む: 的中率と、
            // そのときブロックが読むバイトが今の何倍になるか。
            var issued = 0, useful = 0
            var cells: [String] = []
            for rank in 0..<Self.maxRank where c.issuedAtRank[rank] > 0 {
                issued += c.issuedAtRank[rank]
                useful += c.usefulAtRank[rank]
                let precision = Double(c.usefulAtRank[rank])
                    / Double(c.issuedAtRank[rank]) * 100
                let cover = c.miss > 0 ? Double(useful) / Double(c.miss) * 100 : 0
                let read = c.miss > 0
                    ? Double(issued + c.miss - useful) / Double(c.miss) : 0
                cells.append(String(format: "%d:p=%.0f%%,cov=%.0f%%,read=%.2fx",
                                    rank + 1, precision, cover, read))
            }
            return "[router preview d=\(distance) rank " + cells.joined(separator: " ") + "]"
        }
        return (lines + ranks).joined(separator: "\n")
    }
}
