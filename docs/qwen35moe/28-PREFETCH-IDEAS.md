# 28. 先読みの機序と、次に攻める手 (検討、2026-08-22)

[27 §9](27-PHASE6-THROUGHPUT.md) の層をまたぐ先読み (+1.9〜13.2%、既定 off) の続き。
**本書は実測を 1 つも足していない** — 引用する数字はすべて
[27](27-PHASE6-THROUGHPUT.md) / [docs/mtp/30](../mtp/30-M8-B-PREFETCH.md) /
[docs/mtp/33](../mtp/33-M8-IO-FLOOR.md) の実測か、そこからの**導出**である。
やることは 2 つ: **同じ機構が Gemma で負けて Ornith で勝った機序を費用構造まで
下ろす**こと、そして **Gemma で死んだ手のうち復活するもの・Ornith だから
攻められるもの**を、実装の形と判定基準つきで在庫にすること。
**実験は別コンテキストで行う** (§6 が手順)。既定は変えない。判断はユーザー
([04](04-PHASES.md) 次の一手 #29)。

> **結果は [31](31-PREFETCH-CHEAPER.md) にある** (2026-08-22、同日)。
> §5 の順で回した: 計器で **`declined`=0 / wait 1.1 ms/tok** → §3-2 のリトライと
> §3-3 の d=2 は**落ちた**。§3-1 の N>8 は**名指しは当たる (+23 ポイント) のに
> 実機で負ける** — 写像が陰に入りきらない。効いたのは副産物の方
> (§3-4 (a) preview の select を出さない、**+2.4〜3.1%**)。§3-4 (b) の融合は引き分け。

---

## 0. 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | 符号が反転した機序 | **待ちの「通貨」が違う。**Gemma の待ちはデバイスのバイト (I/O 帯域が床)、Ornith の待ちはホストのページ写像 (CPU 仕事)。先読みは読むバイトを減らさない機構なので、バイトが床の世界では外れのバイト +11% がそのまま費用になり ([30 §5](../mtp/30-M8-B-PREFETCH.md))、写像が床の世界では**写像を GPU の陰に移せる** (§1) |
| 2 | 一番攻められる点 | **N を上げる限界費用が GPU 側でゼロ。**preview の GEMV は N に依らず 256 logit 全部を計算済みで、k=8 に切っているのは select カーネルの固定 k だけ (`MoE.swift:43` の `maxStreamedExperts`)。**CPU が 256 logit から top-N を選べば** N>8 の対価は「無駄な写像の churn」だけになる (§3-1)。Gemma では N↑ = バイト↑で単調に負けた ([30 §6](../mtp/30-M8-B-PREFETCH.md)) が、その死因はここには無い |
| 3 | N では直らないもの | 予測の固定費 **GPU +1.2〜1.8 ms/tok** ([27 §9-5](27-PHASE6-THROUGHPUT.md)) は N に依らない。短いプロンプト (t1 +1.9%) の利得を食っているのはこれで、**安くする手は別に要る** (§3-4)。router 重みの読みは 1 MiB/層 = 導出で 0.3 ms/tok 程度なので、費用の主はディスパッチ約 80 本/tok の側とみる (**導出**、§3-4) |
| 4 | d=2 (2 層前倒し) | **計器を入れてから。**いまの実装に「先読みが間に合っているか」の計器が無い (`QwenForwardRunner.swift:817` の wait は時間を取っていない)。待ちが ~0 ならリードタイムは足りていて、d=2 は的中を落とすだけの手になる (§3-3) |
| 5 | 設計空間を机上に持ち込める | preview の top-16 と実際の top-8 を**トレースに 1 回だけ落とせば**、N・保護規則・リトライの全組み合わせを `expert_sim.py` の流儀でオフラインに引ける ([27 §6-1](27-PHASE6-THROUGHPUT.md) の「規則の再現」と同じ手)。実機のスイープは最後の確認だけでよい (§5) |
| 6 | 復活しないもの | 読みの分割 (深めるべきデバイス読みが無い)、`rdadvise` (両家族で死亡)、**prefill への先読み** (prefill は 6.2 GB/s でデバイス帯域の床に居る = Gemma と同じ世界)。§4 |
| 7 | この道の終点 | 先読みはトークン境界を越えられない (次トークンの embedding は LM head の後)。**越える唯一の形が投機 = MTP** で、verify ブロックは写像費用を 1/k に割る。本モデルは MTP ヘッド同梱・ドラフト用 256 エキスパート常駐可 ([README](README.md) #12)。Phase 6 の外 (§3-6) |

---

## 1. 機序 — 同じ機構、違う通貨

### 1-1. 2 つの世界の費用構造

[27 §9-6](27-PHASE6-THROUGHPUT.md) の表を、先読みの損益計算ができる形まで下ろす。

| | Gemma (2026-08-19、私有スロット) | Ornith (mmap、本線) |
| --- | --- | --- |
| 待ちの正体 | **デバイス帯域。**fetch 60.7 ms/ブロックで 7.5 GB/s、プローブの天井に張り付き ([30 §5](../mtp/30-M8-B-PREFETCH.md))。冷たい device の天井は 6.0〜6.5 GB/s ([33 §1](../mtp/33-M8-IO-FLOOR.md)) | **ホストのページ写像。**「io 15.8 ms/tok」の 85% が residency set の更新、その 94% が `commit()` (0.4 ms × 40 回/tok)。decode 中の disk0 は 0.6〜0.85 GB/s しか動かず、**バイトはページキャッシュに在る** ([27 §9-1](27-PHASE6-THROUGHPUT.md)) |
| 通貨 | バイト | 写像 (ページ 9,200 枚/tok、[27 §9-3](27-PHASE6-THROUGHPUT.md)) |
| 当たり 1 本の利得 | 読みが早く始まるだけ。**床がバイトなので読む総量が同じなら時間も同じ** | クリティカルパスの写像 1 本 (≈0.06 ms) が GPU の陰に移る |
| 外れ 1 本の費用 | **バイト +1 本ぶんが床に直撃** (32 スロットで +11% → fetch +1.7 ms/ブロック) | 陰の中の写像 1 本。**常駐していればゼロ** (プランはヒットとして見る) |
| N を上げる限界費用 | 的中率の低い rank のバイトが積み上がる → **単調に負け** ([30 §6](../mtp/30-M8-B-PREFETCH.md)) | GPU 側ゼロ (logit は計算済み)。増えるのは陰の写像と slot churn だけ (§3-1) |
| 隠す先 | **無い。**32 スロットは I/O 床、48 は GPU 床 ([33 §0-5](../mtp/33-M8-IO-FLOOR.md)) | **GPU 28.5 ms/tok が回っている間のホスト**。写像 15.8 ms はその陰に入る大きさ |

### 1-2. 先読みが勝つ条件 (一般形)

次の 3 つが**同時に**成り立つときだけ、この機構は表に出る:

1. **待ちがホスト仕事で、GPU と並行できる。**Gemma はここで既に偽
   (待ちはデバイス I/O で、先読みしても総バイトは減らない)。
2. **外れの費用が「バイト」ではなく「無駄仕事 1 回」で、それが陰に隠れる。**
   Ornith の外れは写像 1 本 ≈ 0.06 ms、しかも常駐なら 0。
3. **陰が足りる。**GPU 時間/層 (≈0.7 ms) ≧ 隠したい写像/層 (≈0.4 ms)。

将来別の family を載せるときは、**先に「待ちの通貨」を §9-1 の分解で測る**こと。
的中率 (Gemma 70% / Ornith 64.5%) は両者でほぼ同じだったのだから、
的中率をいくら測っても符号は予言できない — 通貨が予言する。

### 1-3. いまの上限 (導出)

先読み on の t4 でも decode hit は 75.2% ([27 §9-5](27-PHASE6-THROUGHPUT.md))。
残る費用は「**名指せなかった miss** (35.5%) の写像」+「予測 GEMV の固定費」で、
仮に名指しが 100%・固定費 0 なら io 15.8 ms/tok がほぼ全部 GPU の陰に入り、
decode は 50.0 → 約 34 ms/tok (≈29 tok/s) が机上の天井 (**導出**、m.json)。
§3 の手はすべて、この 2 つの残差のどちらかを削る手である。

---

## 2. Gemma で死んだ手の在庫と、復活の判定

| 手 | Gemma での死因 | Ornith での判定 |
| --- | --- | --- |
| N≥2 の先読み | バイト +11〜21% が I/O 床に直撃、**N 単調に負け** ([30 §6](../mtp/30-M8-B-PREFETCH.md)) | **復活有力。**死因 (バイト) がここには無い。しかも N>8 は select カーネルの都合で未踏 (§3-1) |
| d=2 | 的中 12 ポイント低下ぶんのバイト増 ([30 §6](../mtp/30-M8-B-PREFETCH.md)) | **条件付き復活。**バイトの罰は無いが、リードタイムが既に足りているなら利得も無い。計器が先 (§3-3) |
| 予測を安くする | [30 §7](../mtp/30-M8-B-PREFETCH.md) が挙げて未着手のまま | **そのまま有効。**固定費 +1.2〜1.8 ms/tok は Ornith でも払っている。短プロンプトの利得はほぼこれに食われている (§3-4) |
| 追い出し保護を K プランに広げる | [30 §7](../mtp/30-M8-B-PREFETCH.md) が挙げて未着手 (追加ミス 5.2 本/ブロック) | **計器から。**`declined` (投機プランの全体棄却) を数えていない。N>8 では all-or-nothing の棄却が効いてくる (§3-2) |
| スロット増 | 48→64 は io −17% でも t/s −3% (**GPU 床**、[33 §0-4](../mtp/33-M8-IO-FLOOR.md)) | **弱い。**48 で +3.5% の導出 ([27 §9-3](27-PHASE6-THROUGHPUT.md)) だが、先読みと**同じ 15.8 ms を狙う競合手**で、併用は劣加法のはず (**導出**)。候補追加自体がユーザー判断 ([04](04-PHASES.md) #28) |
| 読みの分割 (キュー深化) | どの深さでも帯域は平ら ([33 §1](../mtp/33-M8-IO-FLOOR.md)) | **復活しない。**深めるべきデバイス読みがそもそも無い |
| `rdadvise` 調律 | — | **死亡済み** ([27 §6-3](27-PHASE6-THROUGHPUT.md))。既定の腕は `F_RDADVISE` を出さない |

---

## 3. 攻め手 — 実装の形と判定基準

### 3-1. N>8: select を捨てて CPU が top-N を選ぶ

**機序。**preview の GEMV は層 L+1 の router 重み全 256 行を `normed` に当てて
**256 logit 全部を書いている**。top-8 に切っているのは後段の select カーネル
(`router_select_k8`、`MoE.swift:43` の `maxStreamedExperts = 8` に
precondition で固定、`MoE.swift:211/264/322`) であって、GEMV ではない。
Qwen 経路は per-expert scale に単位値を渡している (`QwenForwardRunner.swift`
`encodeRouter`) ので、**生 logit の大小順 = select カーネルの選ぶ順**である。

**実装の形。**preview 専用の logits バッファ (256 × Float、shared) を
runner に持たせ、GEMV だけを encode する入口を `MoE` に 1 本足す
(select は encode しない — **dispatch が 1 本減る**副産物つき)。既にある
pre-router の join (`QwenForwardRunner.swift:803`) の後で CPU が 256 要素から
top-N を選ぶ。256 要素の部分ソートはホスト費用 ≈ 0。**追加の往復もゼロ**
(読み戻す join は元からある)。K=8 の現行経路は既定のまま残し、
env (`TF_QWEN_EXPERT_PREFETCH` > 8) のときだけ広い経路に入る。

**期待値 (導出)。**rank 8 でも「使われる」44%・「miss を名指す」15.2%
([27 §9-4](27-PHASE6-THROUGHPUT.md)) と裾が厚いので、rank 9〜16 が
未名指しの 35.5% の一部を拾う見込みは十分ある。理論上は、外れが完全に陰に
隠れて slot churn が無い限り**的中率が低い rank でも期待値は正**
(当たり = クリティカルパス 0.06 ms 節約、外れ = 陰の 0.06 ms) — 実際の限界は
churn と、fetch スレッドの lock 直列化の側に出るはずである。

**リスク。**32 スロットのうち、実プラン 8 + 投機 N が毎トークン層ごとに動く。
N=16 なら 24/32 が回り、LFU が守っている常連を victim に選ぶ頻度が上がる
(投機プランは use count を上げず victim は「最安」から取る —
`PreadExpertStreamer.swift:341` 周辺)。**ヒット率の変化を必ず併記する**こと。

### 3-2. 投機プランの all-or-nothing を外す

`makeExpertCachePlan` は置けない miss が 1 本でもあると**プラン全体を捨てる**
(`guard misses.count <= victims.count else { return nil }`、
`PreadExpertStreamer.swift:341`)。N=8 では Gemma で `declined=0` だったが
([30 §1](../mtp/30-M8-B-PREFETCH.md))、Ornith では数えていない。N を上げるほど
この棄却は起きやすくなり、**一番価値のある rank 1 まで巻き添えで捨てる**。

手は 2 つ、どちらも小さい:
- **半減リトライ**: 棄却されたら N/2 の prefix でもう一度 (上位 rank ほど
  価値が高いので prefix が正しい)。プラン作成は µs 級なので費用は無視できる。
- **置けるぶんだけ置くプラン**: streamer に partial 版を足す。正攻法だが
  追い出し規則の検査 (負例 3 本、[30 §8](../mtp/30-M8-B-PREFETCH.md)) に手が入る。

まず `declined` を数える計器だけ入れ、ゼロならどちらも要らない。

### 3-3. リードタイムの計器 → d=2 の要否

いまの実装は層 L で L+1 の先読みを出し、L+1 のプラン直前で wait する
(`QwenForwardRunner.swift:817-819`)。**この wait の時間を測っていない**ので、
「先読みが間に合っているか」が分からない。

- wait ≈ 0 なら: リードタイム (層 L の MoE の GPU 時間 ≈ 0.7 ms) は足りている。
  **d=2 は的中を落とすだけ** (Gemma で 80→68%、[30 §6](../mtp/30-M8-B-PREFETCH.md))
  なのでやらない。残差は §3-1 / §3-4 の側にある。
- wait が写像 1 回分 (0.4 ms) を超えて出るなら: d=2 の形は「層 L の
  pre-router CB に L+2 の router も相乗り (L の `normed` に当てる)、
  in-flight ハンドル 2 本」。router GEMV がもう 40 本/tok 増えるので、
  §3-4 とセットでないと固定費が倍になる。

計器は wait の nanos と発行/棄却の本数を `RouterPreviewStats` の隣に足して
footer に出すだけ。方策は何も変えない。

### 3-4. 予測の固定費 +1.2〜1.8 ms/tok を安くする

router 重みは BF16 で 256 × 2048 × 2 B = **1 MiB/層 = 40 MiB/tok**。
150 GB/s 級で読めていれば **≈ 0.3 ms/tok** にしかならないので、実測 1.2〜1.8 ms
の主は**追加ディスパッチ約 80 本/tok (GEMV + select × 40 層) の立ち上がり**
とみる (**導出**。[27 §4](27-PHASE6-THROUGHPUT.md) で「バッファ本数が減ると
gpu 総和が減る」と同じ現象の encoder 版)。なら手は帯域ではなく本数:

| 手 | 削れる本数 | 変更の重さ |
| --- | --- | --- |
| (a) preview の select を encode しない (§3-1 の副産物) | 40 本/tok | 小 (encode 経路の分岐だけ) |
| (b) 実 router と preview を 1 dispatch に融合 (幅 2 倍で 2 層ぶんの GEMV) | さらに 40 本/tok | 中 (カーネル 1 本。[30 §7](../mtp/30-M8-B-PREFETCH.md) の「同じ dispatch で 2 層ぶん」案そのもの) |
| (c) 適応スキップ: miss が構造的に少ない層は preview を張らない | 層分布しだい | 小 (トレースの層別 miss 分布が先。`--dump-expert-trace` 1 本と `expert_sim.py` の拡張でオフラインに出る — モデル再実行不要) |

t1 (56 tok) の利得が +1.9% しかないのはこの固定費のせいなので、
**既定 on をユーザーに推せるかは §3-1 より §3-4 が握っている**可能性が高い。

### 3-5. 併用の相互作用 (測るなら最後)

- **× sticky residency** (`TF_EXPERT_MMAP_RESIDENT=64`): 単独では引き分け
  ([27 §9-2](27-PHASE6-THROUGHPUT.md))。先読みと併せると「外れの写像」自体が
  減る方向だが、期待値は小さい。128 は wired limit 8192 MB で GPU OOM
  ([27 §9-2](27-PHASE6-THROUGHPUT.md)) — limit を上げるのは機械ごとの
  ユーザー判断で、再起動で戻る類のもの。
- **× 48 スロット**: 両方とも同じ 15.8 ms を狙うので**劣加法のはず** (**導出**)。
  48 の候補追加が通ったときに 1 回だけ併用を測れば足りる。

### 3-6. この道の終点 — トークン境界と MTP

preview は層 L の `normed` があるから引ける。トークン境界を越えるには
次トークンの embedding が要り、それは LM head の後 — **予測ではなく投機**
になる。それが MTP で、verify ブロック k トークンは各層のエキスパート集合を
**ブロックに 1 回**しか写像しないから、写像費用が構造的に 1/k になる。
本モデルは MTP ヘッド同梱、ドラフト側 256 エキスパートは全常駐にできる
([README](README.md) #12)。~~ただしツール宣言のあるエージェント経路では
MTP は効かない (文法 → 投機オフ)~~ — **これは旧情報で、[29 §4-1](29-MTP-PREFETCH-OUTLOOK.md)
が訂正した** (Gemma 側は 2026-08-22 に文法込み投機が緑)。**Phase 6 の手ではない**が、
「写像を隠す」の先に「写像を割る」があることは記録しておく。
**この節の続き (在庫全体の MTP 下での再判定) は [29](29-MTP-PREFETCH-OUTLOOK.md)。**

---

## 4. 復活させない手

| 手 | 理由 |
| --- | --- |
| 読みの分割でキューを深くする | 深めるべきデバイス読みが無い (decode の disk0 は 0.6〜0.85 GB/s、バイトはページキャッシュ)。Gemma でも分割自体が無効だった ([33 §1](../mtp/33-M8-IO-FLOOR.md)) |
| `rdadvise` / `byteCap` 調律 | 既定の腕は `F_RDADVISE` を出さない ([27 §6-3](27-PHASE6-THROUGHPUT.md)) |
| prefill への層またぎ先読み | prefill は 13.4 GB を 6.2 GB/s で読んでいて**デバイス帯域の床に居る** ([27 §5](27-PHASE6-THROUGHPUT.md)) = Gemma と同じ通貨。§1-2 の条件 1 が偽。しかも 1 チャンク内の各エキスパートは 1 回しか要求されず、hit の概念自体が無い ([27 §6-1](27-PHASE6-THROUGHPUT.md)) |
| `commit()` の間引き・set 不使用 | 3 つとも引き分け済み ([27 §9-2](27-PHASE6-THROUGHPUT.md))。写像は誰かが払う — 消す手ではなく**陰に移す手** (= 先読み) だけが効く |

---

## 5. 別コンテキストでの実験手順

順序に意味がある: **計器 → 机上 → 実機**。実機のスイープは最後の確認だけ。

1. **計器 (方策を変えない)**: 先読みの wait nanos / issued / declined を
   footer に追加 (§3-3)。既存の n8 の A/B を 1 回取り直して
   (a) wait ≈ 0 か (b) declined = 0 か を見る。ここで d=2 (§3-3) と
   リトライ (§3-2) の要否が決まる。
2. **広い preview の 1 回どり**: §3-1 の GEMV-only 経路を入れ、
   `TF_QWEN_EXPERT_PREFETCH=0` + preview K=16 で m.json / t4 を 1 回ずつ。
   footer の rankMiss の 9〜16 位が **N>8 の期待値そのもの**。ついでに
   (layer, token, preview top-16, actual top-8) をトレースに落とせば、
   N・保護・リトライの全組み合わせを **offline で**引ける (§0-5)。
3. **実機スイープ**: 机上で残った N (例: off / 8 / 12 / 16) だけを
   `bench/qwen35.sh` の流儀 (RUNS=3、交互、COOLDOWN=4、熱ドリフト検定) で。
   m.json と t4 の 2 本。**ヒット率と GPU ms/tok を併記** (churn と固定費の監視)。
4. **検査**: `--qwen-decode` 55 本全一致を**先読み on・両腕**で
   ([27 §7](27-PHASE6-THROUGHPUT.md) と同じゲート)。prefill 経路は触らないが
   `--qwen-prefill` も 1 回。`TurboFieldfareKernelCheck --qwen` と
   `swift test --no-parallel`。Gemma の経路に触れないこと
   (`RealForwardRunner` / `MoE` の既存 encode の署名を変えない)。
5. **判定**: 反復 3 未満のセルに解釈を書かない。既定は変えず、
   「n8 のまま #29 の判断へ」「n12/n16 を候補に足す」「§3-4 を先にやる」の
   どれかをユーザーに出す。

## 6. コードの根拠 (2026-08-22 に確認した現物)

| 事実 | 場所 |
| --- | --- |
| select の k=8 固定 (`topK == maxStreamedExperts` の precondition) | `Sources/TurboFieldfare/Kernels/MoE/MoE.swift:43,211,264,322` |
| GEMV は 256 logit 全部を Float で書く (`maxRouterRows × 256`、shared) | `MoE.swift:176-183` |
| Qwen は per-expert scale に単位値 → 生 logit の順 = select の順 | `QwenForwardRunner.swift` `encodeRouter` (unitFeatureScale / unitExpertScale) |
| preview は実 router と同じ CB に相乗り、join は元からある | `QwenForwardRunner.swift:799-803` |
| 先読みは in-flight 1 本、L+1 のプラン直前に wait (時間は未計測) | `QwenForwardRunner.swift:812-841,874` |
| 投機プランは all-or-nothing (`guard misses.count <= victims.count else nil`) | `Infrastructure/Streaming/PreadExpertStreamer.swift:341` |
| 投機プランは clock も use count も進めず、直近プランのスロットを守る | 同 `makeExpertCachePlan(speculative:)` |
| residency 更新は fetch スレッド上、lock で直列 | `Infrastructure/Streaming/MmapExpertMapping.swift:200` |
| miss 0 の fetch はキューに乗らずその場で完了 | `Runtime/Inference/ModelExpertIO.swift:231` |
| footer の preview 統計 (rankMiss まで出る。prefetch の wait/declined は無い) | `Sources/TurboFieldfareCLI/RunQwen.swift:430-445` |
| ベンチの prefetch A/B は off / n8 の 2 腕のみ | `bench/qwen35.sh` `cmd_prefetch` |
