import Darwin
import Foundation
import Metal

/// D の試作 — 層ファイルを `MAP_SHARED` で張りっぱなしにし、エキスパート 1 個を
/// `bytesNoCopy` の `MTLBuffer` 1 本にして GPU に直接読ませる経路
/// (`docs/mtp/48-D-P1-P4-MMAP-RESIDENCY.md` §9 + `49-D-P5-RESIDENCY-SET.md` §9)。
///
/// **既定では作られない。**`TF_EXPERT_MMAP=1` のときだけ `PreadExpertStreamer` が
/// これを持ち、私有スロットへの `pread` の代わりにここのバッファを返す。
/// 測っているのは 49 §10 の P-6 (30 層・張りっぱなし・層ごとの residency set) で、
/// **速度の札としては「負けない」を確認するだけ**である (50 §7: 賞金は壁時計の約 2%)。
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
/// `F_RDADVISE` は `PreadExpertStreamer` 側の既存経路がそのまま出す (48 §9 で必須)。
/// 読む fd は同じファイルなので、advise の効き方はどちらの腕でも同じである。
public final class MmapExpertMapping: @unchecked Sendable {
    /// 試作の入口。**計器であって設定ではない** — 既定は今の `pread` のまま。
    public static let isEnabled = ProcessInfo.processInfo.environment["TF_EXPERT_MMAP"] == "1"

    /// 52 の実験: mmap の腕にも `F_RDADVISE` を出す。**計器であって設定ではない。**
    ///
    /// prompt prefill は**どちらの腕も advise を出していない** (`RealForwardRunner`
    /// の prefill/block 経路 1380-2686 に advise の呼びが 1 つも無い。出しているのは
    /// `:3127`/`:3129` の decode 経路だけ)。それでも pread が prefill で 7.9 GB/s
    /// 出るのは、タイルのミスを `concurrentPerform` で 7.58 本同時に投げていて
    /// **並列度がそのままキュー深度になる**からである。mmap の腕はそれが
    /// `requestResidency()` 1 本に潰れるので深度 1 で、52 の実測で 5.56 GB/s しか出ない。
    /// ここで先読みを明示的に頼めば、深度が原因かどうかが分かれる。
    public static let adviseMisses =
        ProcessInfo.processInfo.environment["TF_EXPERT_MMAP_ADVISE"] == "1"

    /// エキスパート 1 個 = 1 本。索引はエキスパート番号 (スロット番号ではない)。
    let expertBuffers: [MTLBuffer]
    /// この層の set。コマンドバッファ側は `useResidencySet(_:)` 1 行で受ける。
    let residencySet: any MTLResidencySet

    private let base: UnsafeMutableRawPointer
    private let mappedLength: Int
    /// いま set に入っているエキスパート。スロットキャッシュの中身を映す。
    private var residentExperts: Set<Int> = []
    private let lock = NSLock()

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
    // 消えたページを指しうるからで、試作としては解放しないほうが安全である。

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
    func syncResidency(desiredSnapshot: () -> Set<Int>) {
        lock.lock()
        defer { lock.unlock() }
        let desired = desiredSnapshot()
        let added = desired.subtracting(residentExperts)
        let removed = residentExperts.subtracting(desired)
        guard !added.isEmpty || !removed.isEmpty else {
            Self.record(nanos: 0, added: 0, removed: 0, skipped: true)
            return
        }
        residentExperts = desired

        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        for expert in removed where expert >= 0 && expert < expertBuffers.count {
            residencySet.removeAllocation(expertBuffers[expert])
        }
        for expert in added where expert >= 0 && expert < expertBuffers.count {
            residencySet.addAllocation(expertBuffers[expert])
        }
        residencySet.commit()
        residencySet.requestResidency()
        Self.record(nanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started,
                    added: added.count,
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

    private static func record(nanos: UInt64, added: Int, removed: Int, skipped: Bool) {
        statsLock.lock()
        if skipped {
            _stats.skipped += 1
        } else {
            _stats.syncs += 1
            _stats.added += added
            _stats.removed += removed
            _stats.nanos &+= nanos
        }
        statsLock.unlock()
    }
}
