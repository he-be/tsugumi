import Foundation
import Metal

/// MTP の受理率の**上限**を、ドラフターなしで測るための診断。
///
/// D5 (same-seed sample-and-compare) では、位置 t の提案が受理される確率は
/// `P(ターゲットが引いたトークン == 提案)` であり、提案をどう作っても
/// **ターゲット自身の抽選分布の最大確率 `p_max` を超えられない**。
/// したがって `p_max` の列は、任意のドラフターに対する受理率の上限を与える。
/// ここで測るのは実際に抽選に使われる分布 — softcap+softmax のあと
/// top-k 64 と top-p で切り、`sample_topk64_final` と同じ規約で再正規化したもの。
///
/// `TF_ACCEPT_PROBE=<path>` が立っているときだけ動く。立っていなければ
/// decode 経路のコストは `shared == nil` の判定 1 個。
/// 可変状態 (`handle` / `top`) は `lock` で守るので `@unchecked Sendable`。
final class AcceptanceProbe: @unchecked Sendable {
    /// 環境変数で 1 度だけ立ち上げる。以降は decode ホットパスから参照するだけ。
    static let shared: AcceptanceProbe? = {
        guard let path = ProcessInfo.processInfo.environment["TF_ACCEPT_PROBE"],
              !path.isEmpty else { return nil }
        return try? AcceptanceProbe(path: path)
    }()

    private let handle: FileHandle
    private let lock = NSLock()
    private var top = [(value: Float, index: Int)](repeating: (0, -1), count: 64)

    private init(path: String) throws {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard FileManager.default.createFile(atPath: path, contents: nil) else {
            throw GeneratorError.invalidGenerationConfig(
                "TF_ACCEPT_PROBE: cannot create \(path)")
        }
        handle = try FileHandle(forWritingTo: url)
        handle.write(Data("""
            # turbo-fieldfare acceptance probe v1
            # pmax_trunc = 抽選分布 (top-k64 + top-p 適用後、再正規化) の最大確率
            step\tdrawn\ttop1\tkept\tpmax_raw\tpmax_trunc\tp_drawn\trank_drawn\n
            """.utf8))
    }

    /// `probs` は softcap+softmax 済み FP16 [vocab] (`.storageModeShared`)。
    /// `sample_topk64_final` と同じ順序で切って、その分布の統計を 1 行書く。
    func record(step: Int, drawn: Int32, probs: MTLBuffer, vocab: Int,
                config: GenerationConfig) {
        lock.lock()
        defer { lock.unlock() }

        let p = probs.contents().bindMemory(to: Float16.self, capacity: vocab)
        let k = min(config.topK ?? 64, 64)

        // top-k を 1 パスで拾う。しきい値未満は挿入すら試さないので、
        // 262,144 要素でも比較がほぼ 1 回で済む。
        for i in 0..<k { top[i] = (-Float.infinity, -1) }
        var threshold = -Float.infinity
        for i in 0..<vocab {
            let v = Float(p[i])
            if v <= threshold { continue }
            var j = k - 1
            while j > 0, top[j - 1].value < v {
                top[j] = top[j - 1]
                j -= 1
            }
            top[j] = (v, i)
            threshold = top[k - 1].value
        }

        // top-p はカーネルと同じ「閾値を跨いだ 1 個を含める」規約。
        var kept = k
        if let topP = config.topP, topP > 0, topP < 1 {
            var cumulative: Float = 0
            for i in 0..<k {
                cumulative += top[i].value
                if cumulative >= topP { kept = i + 1; break }
            }
        }

        // temperature は確率に p^(1/T) で効く (カーネルと同じ)。
        let t = config.temperature
        var mass: Float = 0
        var weights = [Float](repeating: 0, count: kept)
        for i in 0..<kept {
            weights[i] = t == 1.0 ? top[i].value : powf(top[i].value, 1.0 / t)
            mass += weights[i]
        }

        var pDrawn: Float = 0
        var rank = -1
        for i in 0..<kept where top[i].index == Int(drawn) {
            pDrawn = weights[i] / mass
            rank = i
            break
        }

        let line = "\(step)\t\(drawn)\t\(top[0].index)\t\(kept)\t"
            + "\(top[0].value)\t\(weights[0] / mass)\t\(pDrawn)\t\(rank)\n"
        handle.write(Data(line.utf8))
    }
}
