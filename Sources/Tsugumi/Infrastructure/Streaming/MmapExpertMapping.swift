import Darwin
import Foundation
import Metal

/// D の経路 — 層ファイルを `MAP_SHARED` で張りっぱなしにし、エキスパート 1 個を
/// `bytesNoCopy` の `MTLBuffer` 1 本にして GPU に直接読ませる経路
/// (`docs/mtp/48-D-P1-P4-MMAP-RESIDENCY.md` §9 + `49-D-P5-RESIDENCY-SET.md` §9)。
///
/// **CLI とサーバーの既定はこちらである** (`docs/mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md`
/// §5a: advise を両腕に揃えた上で tok/s ×1.331 / ttft ×0.77 / peak −3.20 GB)。
/// `PreadExpertStreamer` はこれを持ち、私有スロットへの `pread` の代わりに
/// ここのバッファを返す。`TF_EXPERT_MMAP=0` で 51 までの私有スロット +
/// `pread` に戻せる — A/B のドライバ (`bench/mtp5*`) はその形で回す。
///
/// 形は 48/49 のプローブがそのまま決めている:
///
/// - **`MAP_SHARED` + `PROT_READ`** (48 §2)。`MAP_PRIVATE` はページを私有 anonymous に
///   変えてしまうので、消したかった 3.22 GB がそのまま戻る。
/// - **マップは層ごとに 1 本、`MTLBuffer` はエキスパート単位に 128 本** (48 §1)。
///   常駐になるのはコマンドバッファが名指したバッファのぶんだけ (1,640 ページ = 26.9 MB)
///   で、マッピング全体ではない。
/// - **`MTLResidencySet` を層ごとに 1 個**持ち、その層でスロットに載っている
///   エキスパートを入れて `commit()` + `requestResidency()` する (49 §2)。
///   これで初回タッチの上乗せは 0.61 → 0.09 ms になり、フォールトはコマンド
///   バッファの外に出る。
/// - **`useResource` は外さない** (49 §2 の腕 B\*: 効いているのは set のほうで、
///   `useResource` を残しても差が無い)。`MoE.swift` は 1 行も触らない。
/// - **CPU 事前タッチは入れない** (49 §5: 何もしないより遅い)。
///
/// `F_RDADVISE` は decode 経路が既存のまま出す (48 §9 で必須) が、prompt prefill は
/// **どちらの腕も出していなかった** (52 §3)。pread はミスを `concurrentPerform` で
/// 7.58 本同時に投げて深度を代替しており、mmap ではそれが `requestResidency()`
/// 1 本に潰れる。そこを `adviseMisses` が埋める (52 §4)。
public final class MmapExpertMapping: @unchecked Sendable {
    /// この経路を使うか。**既定 on** — 外すのは `TF_EXPERT_MMAP=0` のときだけである。
    ///
    /// 値を読むのは `Model` がストリーマーを開くところ 1 か所で、そこから先は
    /// ストリーマーごとの `mmap` が非 nil かどうかだけを見る。テストが直接
    /// 作るストリーマーは既定 (`useMmap: false`) のまま `pread` である。
    public static let isEnabled = ProcessInfo.processInfo.environment["TF_EXPERT_MMAP"] != "0"

    /// `executeExpertCachePlan` のミスに `F_RDADVISE` を出すか、の**明示指定**。
    /// `nil` = 指定なしで、そのときは**腕に従う** (mmap の腕だけ出す)。
    ///
    /// prompt prefill は**どちらの腕も advise を出していなかった** (`RealForwardRunner`
    /// の prefill/block 経路 1380-2686 に advise の呼びが 1 つも無い。出しているのは
    /// `:3127`/`:3129` の decode 経路だけ)。それでも pread が prefill で 7.9 GB/s
    /// 出るのは、タイルのミスを `concurrentPerform` で 7.58 本同時に投げていて
    /// **並列度がそのままキュー深度になる**からである。mmap の腕はそれが
    /// `requestResidency()` 1 本に潰れて深度 1 になり、52 §4 の実測で prefill は
    /// 5.56 GB/s しか出なかった。明示的に頼むと **11.99 GB/s / ttft ×0.59** になる。
    ///
    /// `pread` の腕には既定で出さない — 52 §5 の対照で prefill io ×1.03 /
    /// tok/s ×0.985 と**速くならない**からである。`TF_EXPERT_MMAP_ADVISE=1` で
    /// 明示すればどちらの腕でも出る (52 §5 のドライバがそう回す)。
    public static let adviseOverride: Bool? = {
        switch ProcessInfo.processInfo.environment["TF_EXPERT_MMAP_ADVISE"] {
        case "1": return true
        case "0": return false
        default: return nil
        }
    }()

    /// set を使うか。**既定 on** (49 §2 の測定で入った)。`TF_EXPERT_MMAP_RESIDENCY=0`
    /// で**付け替えを 1 回もしない**腕になる — バッファは `useResource` で
    /// 名指されたままなので正しさは動かず、ページはカーネルの中でフォールトする。
    ///
    /// 対照が要るのは、この家族では **`commit()` が 1 回 0.4 ms あり、層ごと
    /// トークンごとに呼ぶので decode の 1/3 を占める**からである
    /// (`docs/qwen35moe/27-PHASE6-THROUGHPUT.md` §9)。set が買っているのは
    /// 「フォールトをカーネルの外に出すこと」なので、**買値と売値をこの
    /// モデルで測り直す**必要がある。
    public static let residencySetEnabled =
        ProcessInfo.processInfo.environment["TF_EXPERT_MMAP_RESIDENCY"] != "0"

    /// `commit()` + `requestResidency()` を何回に 1 回出すか。既定 1 = 毎回。
    ///
    /// 足し引き自体は 0.02 ms/tok と**ただ同然**で、高いのは `commit()` (1 回
    /// 0.4 ms、層ごとトークンごと) である。間引くと set の中身は遅れて効く —
    /// 遅れている間に使われたページは**カーネルの中でフォールトする**ので、
    /// 「commit を減らす利得」と「フォールトの費用」の交換になる。
    /// どちらが勝つかはモデル次第なので測る (`docs/qwen35moe/27-PHASE6-THROUGHPUT.md` §9)。
    public static let commitEvery: Int = {
        guard let raw = ProcessInfo.processInfo.environment["TF_EXPERT_MMAP_COMMIT_EVERY"],
              let value = Int(raw), value > 1 else { return 1 }
        return value
    }()

    /// `commit()` を**呼び出しスレッドから外す**か。既定 off。
    ///
    /// mmap の腕では `executeExpertCachePlan` は**バイトを 1 つも読まない** —
    /// 返すのはマッピングへの view で、residency set の更新は
    /// 「フォールトをカーネルの外に出す」ための先回りでしかない
    /// (`docs/mtp/49-D-P5-RESIDENCY-SET.md` §2)。常駐の保証そのものは
    /// `useResource` が出しているので、**commit を待つ理由は 1 つも無い**。
    ///
    /// それでも既定が同期なのは、非同期にすると「まだ commit されていない
    /// ページをカーネルが踏む」窓が開くからである。窓の中のフォールトは
    /// カーネル内で払う — `TF_EXPERT_MMAP_RESIDENCY=0` の腕と同じ費用で、
    /// `docs/qwen35moe/27-PHASE6-THROUGHPUT.md` §9-2 はそれが**引き分け**だと
    /// 測っている。違うのは**この腕では commit も走る**ことで、先回りが
    /// 間に合ったページのぶんだけ得をする。どちらが勝つかは実測
    /// (`docs/qwen35moe/39-RESIDENCY-COMMIT.md`)。
    public static let residencyAsync =
        ProcessInfo.processInfo.environment["TF_EXPERT_MMAP_RESIDENCY_ASYNC"] == "1"

    /// **set から落とさない上限** (エキスパート/層)。0 = 落とす (従来どおり
    /// スロットキャッシュの中身をそのまま映す)。
    ///
    /// `docs/qwen35moe/27-PHASE6-THROUGHPUT.md` §9 の測定から来ている: Ornith の
    /// decode では expert の「io」の **85% が set の付け替えそのもの**で、
    /// バイトはほとんどページキャッシュから来ていた (`iostat` で decode 中の
    /// disk0 は 0.6〜0.85 GB/s、冷たい天井の 1 割強)。追い出しのたびに
    /// `removeAllocation` するのをやめ、上限まで足しっぱなしにすると、
    /// 2 度目以降の同じエキスパートは `added` が空になって sync ごと消える。
    ///
    /// 上限に当たったら **`desired` だけ残して作り直す** (一括の刈り取り)。
    /// 落とすものが在庫の大半なので、`removeAllocation` を 1 本ずつ呼ぶより
    /// `removeAllAllocations` 1 本のほうが安い。
    ///
    /// 値は**エキスパート数**である。1 本 1.69 MiB / 40 層なので、64 なら
    /// 常駐要求は 4.3 GB、96 なら 6.5 GB になる — ここは `ExpertCacheBudget` が
    /// 数えていない領域なので、**上げるのは運用点の判断**である。
    public static let stickyResidentExperts: Int = {
        guard let raw = ProcessInfo.processInfo.environment["TF_EXPERT_MMAP_RESIDENT"],
              let value = Int(raw), value > 0 else { return 0 }
        return value
    }()

    /// エキスパート 1 個 = 1 本。索引はエキスパート番号 (スロット番号ではない)。
    let expertBuffers: [MTLBuffer]
    /// この層の set。コマンドバッファ側は `useResidencySet(_:)` 1 行で受ける。
    let residencySet: any MTLResidencySet

    private let base: UnsafeMutableRawPointer
    private let mappedLength: Int
    /// いま set に入っているエキスパート。スロットキャッシュの中身を映す。
    private var residentExperts: Set<Int> = []
    /// `commitEvery` 用。この層で何回 set を編集したか。
    private var edits = 0
    private let lock = NSLock()
    /// `syncResidencyAsync` の行き先。層ごとに 1 本。
    private let residencyQueue = DispatchQueue(label: "tf.expert.residency",
                                               qos: .userInitiated)

    init(layout: StreamLayout, device: MTLDevice, fileDescriptor: Int32) throws {
        let pageSize = Int(getpagesize())
        let stride = Int(layout.expertStride)
        let expertCount = layout.expertsPerLayer
        let end = layout.streamOffset &+ layout.streamSize
        guard end <= UInt64(Int.max) else {
            throw StreamerError.invalidIOSplitConfiguration(
                "stream end \(end) is not addressable")
        }
        // 張るのはファイルの先頭から stream の末尾まで。運用の層ファイルは
        // `streamOffset == 0` / `streamSize == 128 * expertStride` なので全体である。
        let length = ((Int(end) + pageSize - 1) / pageSize) * pageSize
        guard stride % pageSize == 0 else {
            // 48 §1 の粒度はページ境界に乗っていることが前提である
            // (出荷形の expertStride 3,358,720 = 205 ページ)。乗っていない形式で
            // `bytesNoCopy` を切ると隣のエキスパートを巻き込む。
            throw StreamerError.invalidIOSplitConfiguration(
                "expertStride \(stride) is not a multiple of the page size \(pageSize)")
        }

        let raw = mmap(nil, length, PROT_READ, MAP_SHARED | MAP_FILE, fileDescriptor, 0)
        guard let raw, raw != MAP_FAILED else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        self.base = raw
        self.mappedLength = length

        var buffers: [MTLBuffer] = []
        buffers.reserveCapacity(expertCount)
        for expert in 0..<expertCount {
            let offset = layout.streamOffset &+ layout.expertOffset(layer: 0, expert: expert)
            guard offset % UInt64(pageSize) == 0,
                  Int(offset) + stride <= length else {
                munmap(raw, length)
                throw StreamerError.invalidIOSplitConfiguration(
                    "expert \(expert) offset \(offset) is not page-aligned inside the mapping")
            }
            guard let buffer = device.makeBuffer(
                bytesNoCopy: raw.advanced(by: Int(offset)),
                length: stride,
                options: .storageModeShared,
                deallocator: nil)
            else {
                munmap(raw, length)
                throw StreamerError.bufferWrapFailed
            }
            buffer.label = "mmap-expert-\(expert)"
            buffers.append(buffer)
        }
        self.expertBuffers = buffers

        let descriptor = MTLResidencySetDescriptor()
        descriptor.label = "routed-experts"
        descriptor.initialCapacity = expertCount
        self.residencySet = try device.makeResidencySet(descriptor: descriptor)
        Self.record(mappedBytes: length)
    }

    // マッピングは**プロセスの寿命ぶん張りっぱなし**である (48 §7 が測れていない
    // 「マッピングの寿命」を production の形にするのが P-6 の主眼)。`deinit` で
    // `munmap` しないのは、ここで剥がすとまだ他所が持っている `MTLBuffer` が
    // 消えたページを指しうるからで、プロセスが終わるまで持ったままにする。

    /// set の中身を `desired` に合わせる。呼ぶのは fetch の側 (I/O キュー) で、
    /// 前の層の GPU 仕事の陰に入る位置である (49 §3 の prep 0.59 ms)。
    ///
    /// 落とすのはスロットキャッシュが追い出したエキスパートだけなので、飛行中の
    /// コマンドバッファが読んでいるものは落ちない — `pread` の腕がスロットを
    /// 上書きしてよいのと同じ保証に乗っている (層 L の routed CB は層 L+1 が
    /// プランを作る前に完了している)。
    /// `desiredSnapshot` は**この lock を持ったまま**呼ぶ。スナップショットを
    /// 撮る順と set に当てる順が入れ替わると、新しい sync が足したエキスパートを
    /// 古い sync が落としうる (prefill はタイルごとに並列で fetch する)。
    /// `useResource` が残っているので落ちても壊れはしないが、測っている費用が
    /// 順序に依存してしまう。
    /// `syncResidency` を専用の直列キューに投げて、呼び出し側は待たない。
    ///
    /// 直列なのは `syncResidency` の中の `lock` と同じ理由 — スナップショットを
    /// 撮る順と set に当てる順が入れ替わってはいけない — に加えて、層ごとに
    /// スレッドを増やさないためである。
    func syncResidencyAsync(desiredSnapshot: @escaping @Sendable () -> Set<Int>) {
        residencyQueue.async { [self] in syncResidency(desiredSnapshot: desiredSnapshot) }
    }

    func syncResidency(desiredSnapshot: () -> Set<Int>) {
        lock.lock()
        defer { lock.unlock() }
        let desired = desiredSnapshot()
        // 対照の腕: 帳簿だけ進めて set には触らない。
        guard Self.residencySetEnabled else {
            Self.record(nanos: 0, added: 0, removed: 0, skipped: true)
            return
        }
        let cap = Self.stickyResidentExperts
        let added = desired.subtracting(residentExperts)
        // 落とさない腕: スロットから追い出されたぶんは set に残す。同じ
        // エキスパートが戻ってきたときに付け替えが 1 回も起きないのが狙いで、
        // 在庫が上限を超えたときだけ一括で刈る。
        let trimming = cap > 0 && residentExperts.count + added.count > cap
        let removed = cap > 0
            ? (trimming ? residentExperts.subtracting(desired) : [])
            : residentExperts.subtracting(desired)
        guard !added.isEmpty || !removed.isEmpty else {
            Self.record(nanos: 0, added: 0, removed: 0, skipped: true)
            return
        }
        residentExperts = cap > 0 && !trimming ? residentExperts.union(desired) : desired

        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        if trimming {
            residencySet.removeAllAllocations()
            for expert in desired where expert >= 0 && expert < expertBuffers.count {
                residencySet.addAllocation(expertBuffers[expert])
            }
        } else {
            for expert in removed where expert >= 0 && expert < expertBuffers.count {
                residencySet.removeAllocation(expertBuffers[expert])
            }
            for expert in added where expert >= 0 && expert < expertBuffers.count {
                residencySet.addAllocation(expertBuffers[expert])
            }
        }
        let edited = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        edits += 1
        // 間引く腕: 編集だけ積んで、K 回に 1 回だけ driver に渡す。
        guard edits % Self.commitEvery == 0 else {
            Self.record(nanos: edited - started,
                        editNanos: edited - started,
                        added: trimming ? desired.count : added.count,
                        removed: removed.count,
                        skipped: false)
            return
        }
        residencySet.commit()
        let committed = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        residencySet.requestResidency()
        Self.record(nanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started,
                    editNanos: edited - started,
                    commitNanos: committed - edited,
                    requestNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - committed,
                    added: trimming ? desired.count : added.count,
                    removed: removed.count,
                    skipped: false)
    }

    /// 測定用 (`--verify-cold` などがスロットキャッシュを空にしたときに合わせる)。
    func dropResidency() {
        lock.lock()
        let hadAny = !residentExperts.isEmpty
        residentExperts.removeAll()
        lock.unlock()
        guard hadAny else { return }
        residencySet.removeAllAllocations()
        residencySet.commit()
    }

    // MARK: - 計器 (プロセス全体)

    public struct Stats: Sendable {
        public var layers = 0
        public var mappedBytes = 0
        public var syncs = 0
        public var skipped = 0
        public var added = 0
        public var removed = 0
        public var nanos: UInt64 = 0
        /// `nanos` の内訳: 足し引き / `commit()` / `requestResidency()`。
        public var editNanos: UInt64 = 0
        public var commitNanos: UInt64 = 0
        public var requestNanos: UInt64 = 0
    }

    private static let statsLock = NSLock()
    nonisolated(unsafe) private static var _stats = Stats()

    public static var stats: Stats {
        statsLock.lock()
        defer { statsLock.unlock() }
        return _stats
    }

    private static func record(mappedBytes: Int) {
        statsLock.lock()
        _stats.layers += 1
        _stats.mappedBytes += mappedBytes
        statsLock.unlock()
    }

    private static func record(nanos: UInt64,
                               editNanos: UInt64 = 0,
                               commitNanos: UInt64 = 0,
                               requestNanos: UInt64 = 0,
                               added: Int,
                               removed: Int,
                               skipped: Bool) {
        statsLock.lock()
        if skipped {
            _stats.skipped += 1
        } else {
            _stats.syncs += 1
            _stats.added += added
            _stats.removed += removed
            _stats.nanos &+= nanos
            _stats.editNanos &+= editNanos
            _stats.commitNanos &+= commitNanos
            _stats.requestNanos &+= requestNanos
        }
        statsLock.unlock()
    }
}
