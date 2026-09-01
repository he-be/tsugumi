# 42. FreeToken からの持ち帰り — 境界 anchor・global cache・弾力予算 (検討、2026-08-22)

[FreeToken](https://github.com/FlashML-org/FreeToken)
([arXiv:2608.16157](https://arxiv.org/abs/2608.16157)、FlashML-org) は
CUDA (RTX 30/40/50) 向けの edge-native MoE serving エンジンで、
DeepSeek-V4-Flash / Qwen3.6-35B-A3B / GLM-5.2 級を消費者機で回す。
読む理由は家族の近さである: Qwen3.6-35B-A3B は本線の MTP ヘッドのドナー
([30 §6](30-MTP-HEAD-GRAFT.md)) であり、抱えている問題も同じ —
**ストリーミングされる expert、巻き戻せない再帰状態 (DeltaNet 系を名指し)、
エージェントによる文脈編集**。

**表記: 本書の「公表値(FreeToken)」は論文と README の記載である。**
[32](32-NVMAI-ADOPT.md) の 実測(NVMAI) より 1 段弱い — **ソースは読んでいない**。
そして数字は RTX の世界 (PCIe / VRAM) のもので、待ちの通貨が違う
([28 §1](28-PREFETCH-IDEAS.md) と同じ注意) 以前に、土俵の絶対値を 1 つも
共有しない。**写せるのは機構の形だけで、数字は 1 つも写せない。**
本書は実測を 1 つも足していない。既定は 1 つも変えない。
判断はユーザー ([04](04-PHASES.md) 次の一手 #34)。

---

## 0. 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | 一番の持ち帰り | **意味的境界の anchor checkpoint** (§2)。[41](41-PROMPT-CACHE.md) が残した全ミス 2 種 — 履歴を書き換えるクライアント ([41 §4-2](41-PROMPT-CACHE.md)) と 2 本目の会話 ([41 §8](41-PROMPT-CACHE.md) #1) — への設計済みの答え。FreeToken は DeltaNet 系の再帰状態を名指しで同じ構図を解いている |
| 2 | その費用 (**導出**) | **会話内の anchor は再帰状態の写真だけ ≈ 62 MiB/個** (GDN 61.41 MiB [34 §2-2](34-PROMPT-CACHE-ESTIMATE.md)、conv 込みの器 62.8 MiB [20](20-PHASE3-DECODE.md))。**KV は append-only なのでカーソルを戻すだけ** — 切り離した写真 (0.232 GB) が要るのは 2 本目の会話のときだけ。取る位置を [41](41-PROMPT-CACHE.md) の `publish` に重ねれば、整合の問題を解き直さずに済む |
| 3 | global expert cache | 公表値は大差 (miss 16% 対 41%) だが、**こちらには逆風の実測が既にある** ([31 §7](31-PREFETCH-CHEAPER.md): 層別 miss 分布に崖が無い = 層が均質なら均等割がほぼ最適)。`expert_sim.py` の流儀 ([27 §6-1](27-PHASE6-THROUGHPUT.md)) で **GPU 0 分**の白黒がつく。本命は予算の細い **Gemma 8 GB 側** (§3) |
| 4 | KV ↔ スロットの弾力予算 | M3 Pro 運用点は 2.84 対 12.88 GB ([34 §2-3](34-PROMPT-CACHE-ESTIMATE.md)) で圧が無く**急がない**。§2 の写真プールが予算に入る日に一緒に (§4) |
| 5 | CPU–GPU co-execution | FreeToken の核だが unified memory には**軸ごと存在しない**。翻訳して残るのは **miss した expert の decode GEMV を CPU にやらせ、ページ写像も residency commit も踏まない**案 — [04](04-PHASES.md) #31 (c) hit-fixup の変種であり独立の手ではない (§5) |
| 6 | 写さないもの | full-layer double-buffered prefill (prefill は帯域の床 [28 §6](28-PREFETCH-IDEAS.md))、FTW 形式 (`.gturbo` が同型)、warming 無し起動 (既にそう)、機種別の自動キャリブレーション (「この 1 台」の方針と衝突) (§6) |
| 7 | コードとライセンス | **1 行も写さない** (ソース未読、ライセンス**未確認**)。写すのは論文が公表した機構の形だけ。コードを写す日が来たら先にライセンスを確認し `THIRD_PARTY_NOTICES.md` に足す |

---

## 1. 機構の在庫と対応表

| FreeToken の機構 (公表値) | 本ランタイムの対応物 | 判定 |
| --- | --- | --- |
| 意味的境界の anchor checkpoint — 再帰状態の小プール + LRU、「編集後も位置が生き残る最深の checkpoint から restore」 | [41](41-PROMPT-CACHE.md) のその場保持 1 エントリ (厳密な延長のみ) | **写す価値が一番高い** (§2) |
| 全層共有の global LRU expert cache — Qwen3.6 で容量 37% のとき miss 16% (静的配置 41%)、DeepSeek-V4-Flash で容量 11% のとき 39% (対 59〜89%) | 層ごと 16/32 スロットの LFU (recency タイブレーク) | オフライン判定だけ写す (§3) |
| KV ページ ↔ expert スロットの弾力配分 — safe point で再構築、リロード無し | `ExpertCacheBudget` (load 時に確定) | 条件が来るまで寝かせる (§4) |
| 帯域適応の CPU–GPU co-execution | 対応する軸が無い (unified memory) | 方針だけ翻訳 (§5) |
| full-layer double-buffered prefill — routing を待たず層 l+1 を丸ごと転送、無効化で 19〜26% 損 | prefill のタイル読み先行 ([27](27-PHASE6-THROUGHPUT.md)) | 写さない (§6) |
| FTW 重み形式 / cold 起動 / pin 前の最終レイアウト直書き | `.gturbo` / 既に cold 起動 / mmap の世界 | 写さない (§6) |

## 2. 意味的境界の anchor — [41](41-PROMPT-CACHE.md) の「全部か無か」を「境界まで」に広げる

### 2-1. FreeToken の機構 (公表値)

checkpoint を任意の位置ではなく**意味的境界** — thinking 区間、ツール呼び出し、
ツール出力、会話ターン — に置く。エージェントフレームワークが文脈を編集する
のは**まさにその境界**だからである (論文の例は、thinking ブロックを剥がす
クライアントと、古いツール出力を placeholder に置き換えるクライアント)。
編集された要求が来たら「**編集後も位置が生き残っている最深の checkpoint**」
から restore し、full-attention の KV は編集点まで再利用、再帰層は anchor
から再開、**本当に新しい接尾だけ**を prefill する。再帰状態は小プール +
LRU eviction で持つ。

### 2-2. こちらへの写像 — FreeToken より安く済む (導出)

[41](41-PROMPT-CACHE.md) 後に残るミスは 2 種類しかない:
**(a) クライアントが履歴を書き換える** ([41 §4-2](41-PROMPT-CACHE.md) —
thinking を送り返さない形は `diverged_at` が assistant ターンの先頭を指して
全ミス)、**(b) 2 本目の会話** ([41 §8](41-PROMPT-CACHE.md) #1)。
anchor は (a) への直接の答えで、(b) には同じプールの器がそのまま使える。

こちらの条件は FreeToken の想定より恵まれている:

- **KV は append-only なので、境界への「巻き戻し」はカーソルを戻すだけ**
  ([32 §1-2](32-NVMAI-ADOPT.md) の投機の checkpoint/restore と同じ機序)。
  会話内の anchor に KV の写真は要らない
- 巻き戻せないのは再帰状態だけで、それは**文脈長に依らず固定** ≈ 62 MiB。
  4 個持っても 0.25 GB、運用点の余裕約 10 GB
  ([34 §2-3](34-PROMPT-CACHE-ESTIMATE.md)) の 2.5%
- 取る位置を [41](41-PROMPT-CACHE.md) の `publish` (ターンの確定点) に
  重ねれば、KV・再帰状態・投機の整合 ([41 §3-2](41-PROMPT-CACHE.md) の
  「受理して終わったパスは最後のトークンも食っている」) を**解き直さなくて
  よい** — publish の瞬間に blit を 1 本足すだけで、62 MiB の blit は帯域から
  見て 1 ms 級 (**導出**)。GDN の snapshot/restore がコピー 0 回級で済むこと
  自体は [33 §3-6](33-MTP-ACCEPTANCE.md) が実測済み
- restore 後は MTP ヘッドの 1 層キャッシュを落とす —
  [41 §3-3](41-PROMPT-CACHE.md) が開けた穴と同じ扱い

(b) の 2 本目の会話だけは KV も分ける必要があるので**切り離した写真**
(position 8,192 で 0.232 GB、[34 §2](34-PROMPT-CACHE-ESTIMATE.md)) になる。
FreeToken の「小プール + LRU」はここでも形として一致する。

### 2-3. 効かない場合 — 着手条件は技術ではなく運用

- **pi 形のクライアントには取り分が 0 である。**もう全部当たっている
  ([41 §4-1](41-PROMPT-CACHE.md))。anchor が効くのは [41 §4-2](41-PROMPT-CACHE.md)
  で全ミスになった形 — thinking を送り返さない・履歴を圧縮する — だけ
- したがって着手条件は「**その形のクライアントを実際に使うのか**」という
  ユーザーの運用判断が先で、要件を先取りして作らない
- 続きから走った答えが再計算とバイト一致するとは限らない話
  ([41 §6](41-PROMPT-CACHE.md)) は anchor 経由でも同じに起きる
- 判定線: [41 §4-2](41-PROMPT-CACHE.md) の全ミス検体の TTFT が、境界ヒットで
  どこまで [41 §5-2](41-PROMPT-CACHE.md) (−93%) に近づくか

## 3. global expert cache — 白黒はオフラインでつく

公表値(FreeToken) の差 (§1 の表) は大きいが、比較相手は「静的配置」であって
層ごと LFU ではない。こちらは層ごとに独立の LFU で、運用点 32 スロット /
層 256 本 = 容量 12.5%、hit 74.9% ([32 §3](32-NVMAI-ADOPT.md) が引く実測)。
global 化の意味は「ヒットの出る層へスロットが寄る」ことだが、**逆風の実測が
既にある**: [31 §7](31-PREFETCH-CHEAPER.md) は層別 miss 分布に**崖が無い**
ことを見た。層が均質なら均等割はほぼ最適で、寄せる先が無い。

ただし判定は **GPU 0 分**でできる: [27 §6-1](27-PHASE6-THROUGHPUT.md) の
トレース再現 (`expert_sim.py` の流儀) に「プール 1 本の LRU / LFU」を足して
同じトレースを流すだけ。層別ヒット率の温度差がそのまま答えになる。
**予算が本当に細いのは Gemma の 8 GB 機** (16 スロット / 層 128 本) なので、
やるならそちらのトレースが本命。実装まで進むならスロットと層別ファイル
記述子の紐付きを解くことになり、**Gemma と共有の経路なので Gemma のベンチを
取り直す** ([README](README.md) 運用ルール)。

## 4. KV ↔ expert スロットの弾力配分 — 写真プールを持つ日に

公表値(FreeToken): 非 expert 重みのロード後に残りを KV ページと expert
スロットへ割り、**その割りを起動時に固定しない**。スケジューラの safe point
で expert cache を作り直し、再起動もリロードもしない。

こちらの `ExpertCacheBudget` は load 時に確定したきりだが、M3 Pro の運用点
には圧が無く (§0 #4)、**急ぐ理由が無い**。効く日は 2 つ:
**(a)** 予算の細い 8 GB の Gemma 機、**(b)** §2 の写真プール (0.232 GB/本) が
予算に入り、文脈長・会話本数と取り合いになった日。縮小側は安いはず
(**仮定**) — スロットは「割り当て済みでも触らなければ非常駐」なので、帳簿から
外せば物理は返る。拡張側と KV の作り直しが FreeToken の言う safe point 問題で、
そこは設計が要る。

## 5. cold expert の CPU fixup — co-execution の Mac 翻訳

FreeToken の核 (帯域適応の CPU–GPU co-execution) は unified memory に**軸ごと
存在しない**。翻訳して残るのは方針 — **miss したものを別の実行資源に流す**。

根拠はこちらの実測に既にある: decode の取得待ちの正体はバイトではなく
**ホストのページ写像** ([27 §9](27-PHASE6-THROUGHPUT.md)、1 トークン約 9,200
ページ) と **residency の commit** ([39](39-RESIDENCY-COMMIT.md))。なら miss
した expert の decode GEMV (1 行 × 4-bit) を **CPU (Accelerate) が mmap
ページから直接計算**すれば、Metal の写像も residency も踏まない。形は
[32 §3](32-NVMAI-ADOPT.md) の hit-fixup ([04](04-PHASES.md) #31 (c)、未着手)
の fixup 側を CPU にした**変種**であり、独立の手ではない。
**CPU の GEMV が写像 + commit より安いかは測っていない (仮定)。**decode 専用
(prefill は帯域の床に居るので GPU のまま)。判定は interleaved A/B
([32 §5](32-NVMAI-ADOPT.md))。

## 6. 写さないもの

| 機構 | 写さない理由 |
| --- | --- |
| full-layer double-buffered prefill | prefill は 6.2 GB/s で**デバイス帯域の床に居る** ([28 §6](28-PREFETCH-IDEAS.md))。現行 prefill も次タイルの先行取得を既に重ねている ([27](27-PHASE6-THROUGHPUT.md))。床に居る間は、routing への依存を切っても取れる泡が無い |
| FTW 重み形式 | `.gturbo` が同型である (page-aligned・固定 stride・blob 単位の bind) |
| warming 無しの cold 起動 / pin 前の最終レイアウト直書き | 前者は既にそう (スロットは touch されるまで非常駐)。後者は pread + pin の世界の話で、mmap に対応物が無い |
| 機種ごとの帯域適応 (自動キャリブレーション) | 方針と衝突する — 本計画は**この 1 台**で速いことだけを狙う ([README](README.md))。Gemma 側の community 機に「提案値を出すだけ」の形はありうるが、既定は動かさない |

## 7. 着手順 (提案、判断はユーザー)

1. **§2 の anchor** — ただし「[41 §4-2](41-PROMPT-CACHE.md) の形のクライアント
   を使うか」の運用判断が先。使うなら実装は [41](41-PROMPT-CACHE.md) の器
   (`match` / `publish`) に blit と境界の帳簿を足す形で、
   [32 §1](32-NVMAI-ADOPT.md) の checkpoint 器とも重なる
2. **§3 のオフライン判定** — GPU 0 分・委譲可能。負けたらそこで終わり
3. **§5 の CPU fixup** — #31 (c) に着手する日に、GPU fixup と CPU fixup を
   同じ interleaved A/B に並べる
4. **§4 の弾力予算** — §2 の写真プールが予算に入る日まで寝かせる
