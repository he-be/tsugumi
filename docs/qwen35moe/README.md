# Qwen3.5-MoE (Ornith-1.5-35B-A3B) 対応 — 計画

QAT・Vision・MTP に続く 4 つ目の大改修。
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
(`architectures: ["Qwen3_5MoeForConditionalGeneration"]`, `model_type: qwen3_5_moe`) を
本ランタイムに載せる。M3 Pro 18GB / macOS 15.7.5 / ブランチ `macos15-support`。
2026-08-21 起草。旧 `PLAN_QWEN35.md` (単一ファイル) を分割したもの。
プラン内で積み重なっていた追記の層 (§15 → §16 → §17) は畳んであり、
**各文書は現在の結論だけを書く。**実測の経緯と数字は 10 番台の結果文書が持つ。

**現在地 (2026-08-22): Phase 8 が通り、HTTP から実物が答える。**
Gated DeltaNet (`qwen_delta_rule`) は 2048 トークン後の状態が **CPU float32 の床と
3 桁一致**、prefill 30 層 **125.7 ms** ([15](15-PHASE2-GDN.md))。その周辺 7 本
(因果 `conv1d` + l2norm / 減衰ゲート / `RMSNormGated` / partial RoPE / 出力ゲート /
shared ゲート / SiLU) も通り、**検査は 29 本すべて緑、うち 6 本は負例**
([17](17-PHASE2-KERNELS.md))。突き合わせ先は [14](14-REFERENCE.md) の float32 参照器
(上流実装と相対 6.4e-07 / top-1 一致 100%)。`.moepack` への repack は
`--verify-install` が緑 (20.49 GB、[13](13-PHASE1-REPACK.md))。
**本線は `oQ4e-g64`** — 同じ文章 4 本の平均 NLL が公式 MLX-4bit より 4 本とも低く、
MTP と vision の実物も入っている ([16 §1](16-QUALITY.md))。
**混在ビット幅は片づいた** ([18](18-MIXED-BITS.md)): 幅は manifest のスロットではなく
**常駐索引からテンソルごとに導く**ことにし、本線の `oQ4e-g64` が `Model.load` を
1,294 ms で通るようになった (`--qwen-open`)。形式は 1 バイトも変えていない。
**最後の 1 本だった LM head も書けた** ([19](19-LM-HEAD-INT8.md)): 本線の `lm_head` が
8-bit なので INT4 の specialization ではなく **INT8 の chain**で、実物の語彙で
1 トークン **4.0 ms / 134 GB/s**。`--qwen` の検査は 39 本すべて緑。
**Phase 3 の結線が通った** ([20](20-PHASE3-DECODE.md)): `QwenForwardRunner` (直列) と
`RecurrentStateManager` を新設し、固定プロンプトから参照と全一致、負例 5 本も落ちる。
実物で見えた食い違いは 3 つ (埋め込みと shared ゲートが 8-bit、routed の活性化が SiLU) で、
いずれも**落ちずにそれらしく間違う**形だった。`RealForwardRunner` は 1 行も動かしていない。
**Phase 3 は閉じ、Phase 4 の prefill が通った** ([21](21-PHASE4-PREFILL.md)):
参照を取り直したら **55 本目に `<|im_end|>` を出して止まっていた**ので、
「64 トークン」の出口条件は**その生成の全体と一致**という形で満たされた。
その 55 本が、**プロンプトを T 行の経路に通しても全部一致する** — チャンク幅
512 (1 チャンク) と 8 (3 チャンク) の両方で。要ったのは **INT8 の QMM**
(INT4 版は 8-bit を読めない) と T 行版の小物 3 本で、`--qwen` は **57 本**になった。
**Phase 5 が通った** ([22](22-PHASE5-TOKENIZER.md)): `QwenTokenizer` は
`GFTokenizer` の分岐ではなく**兄弟**で (Gemma の検査を緩めないため)、decode は
**ByteLevel** の streaming、framing は**上流の `chat_template.jinja` そのもの**。
`--qwen-tokenizer` の **223 本**は上流 `tokenizers` / `transformers` との
突き合わせで、うち 4 本は負例。CLI は `RunQwen.swift` という別経路で、
**推論 (`<think>`) は stderr、答えは stdout** に分かれる。
**Phase 5 の残りも片づいた** ([23](23-PHASE5-TOOLS.md)): XML 形の
ツール呼び出しに `QwenToolCallParser` と `QwenToolCallGrammar` が入り、
**テンプレート自身が描いた呼び出しを文法が受理し、パーサが同じ引数に戻す**
(往復 7 検体)。生の文字列値は **`\n</parameter>` を含まない任意のテキスト**を
13 状態で綴るので、`<b>bold</b>` のような引数が通る。`GrammarVocabulary` の
piece も ByteLevel 用の入口を足した — Gemma の規則だと**落ちずに日本語が
1 文字も通らなくなる**ので、負例で押さえてある。`--qwen-tools` は **36 本**、
うち 6 本は負例。**[22 §4-2](22-PHASE5-TOKENIZER.md) の「swift-jinja の
キー順は不定」は訂正した** — 昇順で、両方決定的である
([23 §5-1](23-PHASE5-TOOLS.md))。
**prefill の routed expert に 2 本目の経路が入った** ([24](24-PREFILL-MOE-PATH.md)):
`prefillRoutedPath` でタイル版 (`prefill_moe_gemm_int4`) を選べるようになり、
**同じ 55 トークンを出したうえで 4 通りとも速い** (GPU 時間で −22〜40%、
既定のチャンク 2048 で 1 トークン 5.46 → 3.93 ms)ので、**既定をタイル版にした**
(2026-08-22 ユーザー確定)。`TF_PREFILL_MOE=scalar` を立てると既定も `.perPair` に
戻る。同じ測定が [21 §5](21-PHASE4-PREFILL.md) の
**説明の付いていなかった行を片づけた**: 増えていたのは GPU 時間で、
**クールダウンを 4 秒取るだけで消える** (2048/512 は 17.80 ではなく **9.43 ms**)。
**CLI の `--tools` が入り、Phase 5 の箇条書きが全部片づいた**
([25](25-CLI-TOOLS.md)): 融合ヘッドは logit をどこにも書かないので、
文法が argmax を拒んだときは**同じ hidden をマスクつきでもう一度畳む**
(`qwen_lm_head_greedy_int8_rows_chunk_masked`)。常の手はマスクを 1 bit も
読まず、**拒まれたトークン 1 個につきヘッドをもう 1 回** (実測 4.086 ms、
素の 4.084 ms と差が測れない)。実物は宣言したツールを呼び、
**素の argmax がすでに整形式だった検体では文法が 1 回も効かなかった** —
外した検体では停止トークンが 1 回だけ拒まれ、モデルは自分で辻褄を
合わせてから呼び出しを書いた。`--qwen` は **63 本**、実物に当てる
`--qwen-constrain` が **9 本** (うち 1 本は負例)。
**サーバーも結線した** ([26](26-PHASE8-SERVER.md)): `QwenServerSession` は
`ServerModelSession` の**兄弟**で、**HTTP 層・要求検証・待ち行列・timings は
1 行も変えていない** — 家族を知っているのはバックエンドだけである。実物は
日本語で答え、ストリーミングで答え、**ツールを呼んでその応答を読んで答えた**。
決定は純粋型 `QwenGenerationPlan` に置いた (C0 で 12 本)。**prompt cache は
持たないと決めた** — 再帰状態を巻き戻せないので `cache_n` は常に 0 で、
[03 §5](03-DESIGN.md) の「slot あたり 62.8 MiB」という推奨は消え、
`ExpertCacheBudget` に足す勘定は無かった。できないこと 4 つは断るか
`approximations` に載せる: 画像は 400、投機は起動時、prompt cache は無い、
**サンプラは受理して無視** (R3 — 既定の `temperature` は 1.0 なので、断ると
既定の客が全部落ちる)。残った判断が 1 つある: **`tool_choice: required` に
ツールと無関係な質問を与えると、呼び出しを書かないまま `max_tokens` まで
散文が続く** ([26 §6-1](26-PHASE8-SERVER.md))。
**計測 (Phase 6) の 1 本目が済み、実物が 1.4 倍速くなった**
([27](27-PHASE6-THROUGHPUT.md))。運用点の数字を**実タスク 4 本**で取ったところ、
遅かったのは計算ではなく**待ち**だった: decode は 1 トークンに 81 本の
コマンドバッファを 1 本ずつ join していて、読みと GPU が 1 度も重なっていない。
join を次の join まで遅らせ、**shared 分岐を routed の読みに重ね**、prefill では
**タイルの読みを 1 枚先行**させたら、**生成 +31〜41% / prefill −25〜45%** になった
(短いプロンプトの TTFT **2.4 → 1.5 秒**、m.json で **14.6 → 20.0 tok/s**)。
**カーネルは 1 本も書いていない。**`TF_QWEN_PIPELINE=0` で元の直列に戻る。
条件のスイープも取れた: 既定 (32 スロット / lfu / チャンク 2048 / mmap) が
どの軸でも最良で、**`--rdadvise` は効かず**、**mmap の腕ではヒット率が壁時計の
代理にならない** ([27 §6](27-PHASE6-THROUGHPUT.md))。
**残っていた「取得 15.8 ms/tok」の正体も割れた** ([27 §9](27-PHASE6-THROUGHPUT.md)):
デバイスの転送ではなく**ホスト側のページ写像** (1 トークンに 9,200 ページ) で、
`commit()` を消しても**カーネル内フォールトに移るだけ** — 非投機的な 3 つの手は
どれも引き分けだった。**層をまたぐ先読みは効く**: 256-way でも **miss の 64.5% を
1 層前に名指せ**、投げると **+1.9〜13.2%** (長いプロンプトほど効く。出力もメモリも
動かない)。Gemma で同じ機構が負けた ([docs/mtp/30](../mtp/30-M8-B-PREFETCH.md)) のは
**待ちがデバイス帯域の床だった**からで、ここでは床が違う。**既定は off、
入れるかはユーザー判断** ([04](04-PHASES.md) 次の一手 #29)。
**Phase 7 の前提が 1 つ崩れ、その場で直した** ([30](30-MTP-HEAD-GRAFT.md)): 上流 discussion #10 の
指摘を手元のウェイトで検算したところ、**同梱の MTP ヘッドは乱数初期化のまま**だった —
上流 bf16 の `mtp.*` は形も役割も違う全テンソルが std ≈ 0.0200・尖度 ≈ 3.00 の
純ガウスで、**256 エキスパートの std のばらつきが CV 0.69%** (学習済みの本体は
4.07〜14.07%)。量子化の副作用ではない (上流 raw bf16 と逆量子化値が小数第 5 位まで一致)。
**[29 §0 #7](29-MTP-PREFETCH-OUTLOOK.md) が「一番大きい未知」と書いた受理長 a は、
測る前に向きが出た** — 出荷ヘッドのままでは Phase 7 の取り分は無い。
そこで **`shisa-ai/…-MTP-ONLY` (19 本 BF16、1.689 GB) に差し替えた**
([30 §6](30-MTP-HEAD-GRAFT.md)): 19 本が `oQ4e-g64` の 42 本にちょうど 1:1 で乗り
(融合 `gate_up_proj` の分割順は上流の実物で相関 **0.99501/0.99554**、norm は
zero-centered なので +1)、**42/42 がバイト一致で書け、index が参照するバイト
21,855,738,720 も `expertStride` 1,769,472 B も赤リスト 0 本も動かなかった**。
乱数の指紋だった **expert 間 std の CV は 0.69% → 9.46%** になり、`mtp.fc` の尖度は
3.05 → **504.89**。増えたディスクは差し替えシャード **503 MB** だけ
(`~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa-baked`)。**以降 MTP を読むものは
これだけを使う。**`.moepack` は元から `mtp.*` を持たない (repack が
`.excludedDraft` に落とす) ので、**pack も固定 digest も動かしていない**。
**そして受理率を測った** ([33](33-MTP-ACCEPTANCE.md)): 実タスク 4 本の生成 192
トークンを float32 参照器に教師強制で通し、**深さ 1 の受理率 P1 = 69.31〜87.83%
(平均 78.70%)、深さ 3 の受理長 a = 2.180〜2.534 (平均 2.344)**。
[29 §3-4](29-MTP-PREFETCH-OUTLOOK.md) の中止線 a<1.5 は大きく上回る。
**ただし a = 2.344 は「2.3 倍速くなる」ではない** — [32 §1-4](32-NVMAI-ADOPT.md) の
「検証費用は行ではなくエキスパートの**和集合**で伸びる」を実トレースで引き直すと
幅 2 で **12.43 experts/層 (1.554 倍、NVMAI の 12.68 と 2% 以内で一致)**、
幅 4 では 2.43 倍。**利得と費用が交差するので運用幅は k=2 (ドラフト 1 本)**、
取り分は **+15〜29%** (**実測**。幅 2 の 1 パスを実際に流すと **1.27〜1.30 倍**で、
和集合の予測 1.554 より軽い — GPU の 30 層はエキスパートを 1 枚も読まず、
ホストの費用はパス回数に比例する、[33 §3-7](33-MTP-ACCEPTANCE.md))。
机上の帯 (+8.5〜11.5% → +6.4〜9.5%) は**どちらも悲観しすぎだった**。**符号を握るのは受理率ではなくパスあたりの
固定費**で、NVMAI の実測はここで和集合の予測より 16% 悪かった — 当方では未測定。
pre/post-norm は**深さ 1 では区別がつかない** (2 勝 2 敗、±1.1 ポイント) ので
運用点では論点にならない。**カーネルは 1 本も書いていない。**

**そして実機で通しで回して決着させた** ([36](36-MTP-DECODE.md)): ヘッドを
**503 MB の sidecar 1 枚**で GPU に載せ (`.moepack` の repack は 0 回)、幅 2 の
検証パスと巻き戻しを書き、実タスクを 192 トークン生成した。**採る。ただし条件付き** —
エージェント形のターン (短いプロンプト、コード修正 ×1.146 / ツール JSON ×1.199) で
勝ち、コード生成 ×0.959、英語散文 ×0.941、**2,698 トークンの要約 ×0.843** で負ける。
勝ち負けを決めているのは受理率ではなく**エキスパート取得の相乗り**で、素の decode が
遅いタスクほど MTP が勝つ (MTP 側の tok/s は 16.3〜19.3 とほとんど動かない)。
**[33 §3-7](33-MTP-ACCEPTANCE.md) の「+15〜29%」は比が正しく土俵が違った** —
幅 2 / 幅 1 の比 (1.27〜1.31) はプレフィル経路で測ったもので、そこに掛けるべき
幅 1 の費用は decode の 1.52 倍だった。**34/35 がプレフィルの話ばかりなのは、
検証パスをプレフィルで代用していたからである。**費用の主犯は密射影のカーネルで
(タイル版が 8 行未満を断り、1 スレッド = 1 出力行の版に落ちていた)、decode の
SIMD/行 GEMV を T 行に広げて GPU 60.9 → 27.4 ms/tok。再帰状態の巻き戻しは
[33 §3-6](33-MTP-ACCEPTANCE.md) の二重バッファで**コピー 0 回**。
**投機は中立** (強制棄却の対照とトークン列が完全一致) だが、**素の decode とは
答えが変わる** — T 行カーネルと decode カーネルの加算順が違うためで、
文法・ツール呼び出しとの併用もまだできない。

**そして検討 6 本を負例として解剖し、次を決めた** ([37](37-MTP-POSTMORTEM-PLAN.md)):
腐ったのは**費用の量の移送**だけで (比をプレフィル経路で測って decode に掛けた
33 §3-7、NVMAI の和集合費用モデルを写して符号まで逆だった 32 §1-4)、
品質の量 (受理率・ablation) は全部実機を当てた。規則は「費用の量は proxy から
積まない。比を掛ける前に基底を実測する」。NVMAI の再読で **draft への
プロンプト履歴のチャンク相乗り** (`prefillChunkedWithMTP`) が未採用と分かり、
改善は **A: 幅 2 特化のハイブリッド検証パス** (attention/GDN は decode の 1 行
カーネル × 2、MoE だけ和集合 grouping — 残る 1.21 倍と長文脈の負けと出力差を
1 設計で潰す) を本丸に、B: 文法併用、C: 自動ゲートと途中離脱の順。

**そして A を測ってから直した** ([38](38-MTP-VERIFY-PATH.md)): 段ごとの計器を
入れると **37 の内訳は外れていた** — ホスト +11 ms と名指しした route grouping は
**0.25 ms/パス**で、1.21 倍のほぼ全部は `preRouter` の GPU、その中身は
**文脈に比例する attention** だった (同じ 2,640 位置に decode は +5.0 ms、
T 行経路は +26.0 ms)。プロンプト用の query-blocked カーネルは 16 クエリを
1 スレッドグループに詰めるので、幅 1〜2 では 16 グループが KV 全体を歩く。
既にある split-KV (`Attention.encodeRows`) に**行ごとに 1 発ずつ**差し替えて
(1 発で流すと投機の中立性が壊れる)、**2,698 トークンの要約が ×0.80 → ×1.11**、
5 本中 3 本が勝ち側になった。**新しい Metal カーネルは 0 本。**

**そして次の最大項 (`io`) を追い詰めた** ([39](39-RESIDENCY-COMMIT.md)):
検証パスの取得の待ちは**バイトではなく `MTLResidencySet.commit()`** で、
それを専用の直列キューへ投げるだけで `io` が 36.5 → 8.3 ms/パスになる
(`TF_EXPERT_MMAP_RESIDENCY_ASYNC=1`、mmap の腕は読むバイトが 0 で、常駐の
保証は `useResource` が出しているから待つ理由が無い)。**素の decode も
MTP も 4 タスク 4 腕すべてで勝ち** (×1.04〜1.12 / ×1.04〜1.08、MTP+async は
素の decode に対し a1 ×1.333 / t4 ×1.190)。**出力は 192/192 一致。**
[27 §9-2](27-PHASE6-THROUGHPUT.md) の「set を捨てると引き分け」は再現するので、
**先回りが無駄だったのではなく、払うスレッドが間違っていた**。同書は
層またぎ先読みを検証パスに入れて**負け**も測っている (予測 router の GPU が
幅 2 で +8.3 ms/パス、commit の通貨は回数なので予測は covering しない限り
1 回増やすだけ) — その負けの計器が非同期化の根拠になった。
**既定は変えていない** (共有経路なのでユーザー判断)。

**そして最後の受入条件を閉じ、サーバーに出した** ([40](40-MTP-GRAMMAR.md)):
[36 §5](36-MTP-DECODE.md) から残っていた「**文法・ツール呼び出しと併用できない**」は
設計の非互換ではなく**引数 1 つの不在**だった — マスクつきの再畳み込みが
ヘッド chain 内部の 1 行 (`xNormedBuffer`) しか読めず、検証パスの 2 行は
別のスクラッチに正規化されるので指せなかった。`encodeMaskedRescore` に
**どの行かを渡せるように**して (カーネルは同じもの、**新しい Metal は 0 本**)、
行 0 の判定 → emit → 行 1 の判定という順に当てた。**受理判定は「制約後」の
引きと比べる。**素の decode / MTP / 強制棄却の 3 腕が **55/55 と 63/63 で
トークン一致**し、そのうち 1 検体は文法が実際に引きを動かしている
(`rescored=1`)。サーバーは **`--draft-block-size 2` を受ける** (0 か 2 のみ、
sidecar は起動時に載せる) ようになり、**tools + thinking を宣言した
エージェント形のターンが投機で回る** — 2,935 トークンのプロンプトで
decode **13.9/14.3 → 16.0/16.1 tok/s** (n=2)、受理率 0.77〜0.94。
`--qwen-constrain` は **15 本** (負例 3)。**エージェント用途で一番効く制約は
投機ではなく prompt cache が無いこと**で、同じターンの prefill 11.3 秒は
毎ターン満額かかる ([34](34-PROMPT-CACHE-ESTIMATE.md) は未着手)。
**そして prompt cache を入れた** ([41](41-PROMPT-CACHE.md)): [26 §4-3](26-PHASE8-SERVER.md) の
「持たない」を撤回する。再帰状態は巻き戻せないので**部分再利用はできない**が、
**「新しいプロンプトが状態の食ったトークン列で始まる」ときだけ続ける**形なら
成立し、**追加メモリは 0 バイト** (エントリは生きている状態そのもの)。
[34 §4-3](34-PROMPT-CACHE-ESTIMATE.md) が「一番大きい危険」と呼んだ**再レンダの
継ぎ目は、実測ではずれなかった** — assistant ターンを `reasoning_content` と
`tool_calls` で送り返すクライアント (pi) では、テンプレートの描き直しが
モデルの書いたバイトと一致する。**エージェント形の会話は 2 ターン目以降が
全部ヒット**し (ツール応答の往復もストリーミングも)、**2,936 トークンの
文書を読ませた会話の TTFT は 10.33 → 0.72 秒 (−93%)**。34 の見積もり
(9.16 秒) は当たっていた。実装で一番効いた発見は「**状態は生成の最後の
1 トークンを食っていない**」が**常には真でない**こと — 幅 2 の投機が最後の
パスで受理して終わると、その行はもう再帰状態に入っている。`--qwen-resume`
**7 本** (負例 1) と C0 **9 本**を足した。**続きから走った答えは、全部計算し直した
答えとバイト一致するとは限らない** (decode と prefill のカーネルの丸めの差、
[41 §6](41-PROMPT-CACHE.md))。
**残るのは vision (Phase 9)** ([04](04-PHASES.md))。
**Gemma の既定 69 本と `swift test` の 1,359 件は緑のまま。**

方針は既存 PLAN と同じ: **汎用性を捨てる。**この 1 台 (M3 Pro / 18GB /
macOS 15.7.5) で速いことだけを目的にし、互換性・移植性・他アーキテクチャへの
一般化は最初から狙わない。

---

## 結論を先に

| # | 論点 | 結論 |
| --- | --- | --- |
| 1 | これは何か | **Qwen3.5-MoE。ただの「Qwen 版 Gemma」ではない。**40 層のうち **30 層が線形注意 (Gated DeltaNet)**、10 層だけが full attention。SWA は 1 層も無い (**実測(上流)**、[01 §1](01-MODEL.md)) |
| 2 | 一番大きい実装 | **Gated DeltaNet カーネル → 書けた** ([15](15-PHASE2-GDN.md))。omlx の blocked-sequential の幾何を写し、状態はレジスタ、ホットループに barrier 無し。検証 15 本が緑、prefill 30 層 125.7 ms。**周辺 7 本も書けた** (検査 29 本、[17](17-PHASE2-KERNELS.md))。**KV キャッシュではなく固定サイズの再帰状態を持つ層**という構造変更 ([03 §3-3](03-DESIGN.md)) も**入った** — `RecurrentStateManager` は 62.8 MiB で文脈長に依らない ([20 §1-1](20-PHASE3-DECODE.md)) |
| 3 | 重み変換 | **当初の「bf16 → MLX 4-bit 変換器を新規に書く」は消えた。**MLX 4-bit 量子化済みの候補が 2 本手元にある ([02](02-CHECKPOINTS.md))。残るのは焼き込み (`q_norm` × 1/16)・名前寄せ・repack ([03 §1](03-DESIGN.md)) |
| 4 | チェックポイントの選定 | **`oQ4e-g64` に決めた** (21.86 GB、imatrix + MTP + vision、router は BF16)。同じ文章 4 本の平均 NLL が公式 MLX-4bit より 4 本とも低い ([16 §1](16-QUALITY.md))。対価だった**混在ビット幅は片づいた** — 幅は索引から導く ([18](18-MIXED-BITS.md)) |
| 5 | MoE の形は乗るか | **乗る。ただし `numExperts <= 256` の precondition にちょうど乗る (余裕ゼロ)。**top-8 は一致、`D=2048` は 64 の倍数、prefill router のスクラッチは既に 256 で確保済み。**decode/prefill の MoE カーネルは無改造で正しく動く**見込み (専用化 PSO から汎用 PSO に落ちるだけ) ([03 §4](03-DESIGN.md)) |
| 6 | エキスパート 1 個のバイト数 | **1,769,472 B = 16 KiB × 108 ちょうど。パディング 0 バイト** ([01 §3-2](01-MODEL.md))。導出がのちに実物と 3 回バイト一致し、**実測に格上げ済み** ([10 §2](10-MLX4BIT-AUDIT.md))。Gemma は 205 ページ中 13,312 B が捨て札なので、そこは改善 |
| 7 | 1 トークンあたりのバイト | **導出で Gemma の 0.78 倍** (4K 文脈、全ヒット時 2.41 GB → 1.89 GB)。**decode は Gemma より速くなり得る**。ただしヒット率が落ちる要因が別にある (#8) ([01 §3-5](01-MODEL.md)) |
| 8 | 一番大きい性能リスク | **測った。出なかった** ([27 §6-1](27-PHASE6-THROUGHPUT.md))。母集団は 3,840 → 10,240 で被覆率は 25% → 12.5% に半減するのに、**32 スロットの decode ヒット率は Gemma の 71.0% に対し 74.9%** である。さらに **mmap の腕ではヒット率が壁時計の代理にならない** — 16 → 32 スロットでヒット率が 12.7 ポイント動いても tok/s は 4.8% しか動かない ([27 §6-2](27-PHASE6-THROUGHPUT.md))。`allowedExpertCacheSlots` は `[8,16,24,32]` のまま |
| 9 | 二番目に大きい性能リスク | **prefill の GEMM 占有率が半減する。**チャンク 2048 でエキスパート 1 個あたり平均 128 行 → 64 行。64 行ブロックがちょうど 1 個しか埋まらない ([05 §1-2](05-RISKS.md))。**測った** ([24 §3-1](24-PREFILL-MOE-PATH.md)): 費用は実在する (タイル版の GPU 時間だけがチャンク幅で 6.6 → 5.0 秒と動く) が、**一番埋まらないチャンク 512 でもタイル版が 22% 速い** |
| 10 | 一番大きい正しさリスク | **partial RoPE のペアの取り方が Gemma と違う。**本ランタイムは `(i, HD/2+i)` を回し周波数の分母に `HD` を使う。Qwen は `(i, 32+i)` を回し分母は `rotary_dim=64`。**既存カーネルを流用すると静かに間違う** ([03 §2-2](03-DESIGN.md)) |
| 11 | ただで貰えるもの | (a) `attention_prefill_causal_qblock_d256` は head_dim だけで選ばれるので**そのまま当たる** (b) RMSNorm は Qwen も `1+w` 規約なので既存カーネルのまま (c) 文脈長は 10 層ぶんしか KV が要らず、線形層の状態は**文脈長に依らず 62.8 MiB 固定** ([01 §3-4](01-MODEL.md)) |
| 12 | MTP | **同梱ヘッドは乱数初期化で使えなかったので、差し替えた** ([30](30-MTP-HEAD-GRAFT.md))。上流 bf16 の `mtp.*` は全テンソルが std ≈ 0.0200・尖度 ≈ 3.00 の純ガウスで、256 エキスパートが統計的に見分けられない。**「専用ドラフターを別リポジトリから取ってくる必要が無い」は崩れた** — `shisa-ai/…-MTP-ONLY` (19 本 BF16) が `oQ4e-g64` の 42 本にちょうど 1:1 で乗り、**差し替え済み** ([30 §6](30-MTP-HEAD-GRAFT.md))。以降 MTP は `~/LLM/Ornith-1.5-35B-A3B-oQ4e-g64-shisa-baked` **のみ**を使う。器の側 (1 層、`mtp.*` 一式 503 MB、うちエキスパート 453 MB、**全 256 エキスパート常駐でドラフトの I/O がゼロ**) は [03 §6](03-DESIGN.md) のまま変わらない |
| 13 | サーバーへの波及 | **prompt cache と投機デコードの前提が壊れる。**再帰状態は「途中を捨てる」「巻き戻す」ができない。**両方とも片づいた**: **投機は [40](40-MTP-GRAMMAR.md)** (`--draft-block-size 2`、tools + thinking と併用可)、**prompt cache は [41](41-PROMPT-CACHE.md)** — 巻き戻せないままでよい形 (**厳密な延長のみ**) で成立し、エージェント形の会話は 2 ターン目以降が全部当たる。**SPEC は書き換えていない** — 家族ごとの差は既存の語彙 (R3 / DEV-16 / EP-4 / CACHE-5) で表現できた |
| 14 | Vision | 後回しでよい。**tower の形は Gemma と偶然ほぼ同じ** (1152 / 27 層 / 16 head / 4304) だが、**位置符号化とマージが別物**。カーネルは書き直し ([04 §11](04-PHASES.md))。bf16 の tower 実物 (893 MB) は oQ4e に入っている |

---

## 読む順

| # | 文書 | 役割 |
| --- | --- | --- |
| 1 | [01-MODEL.md](01-MODEL.md) | 上流の事実 (config・テンソル・算式)、Gemma 4 26B-A4B との差分、数量 (パラメータ・ページ・バイト予算) |
| 2 | [02-CHECKPOINTS.md](02-CHECKPOINTS.md) | チェックポイント候補 2 本と手元の持ち物、供給源選定の経緯、oQ / omlx から取り入れるもの・取り入れないもの |
| 3 | [03-DESIGN.md](03-DESIGN.md) | 変換と repack の残作業、新規カーネル 8 本、ランタイムの構造変更、既存資産の再利用可否、サーバー波及、MTP |
| 4 | [04-PHASES.md](04-PHASES.md) | Phase 0〜9 の分解と出口条件、Vision の中身、次の一手 |
| 5 | [05-RISKS.md](05-RISKS.md) | 性能リスク、中止線、残る未確認、やらないこと |
| 6 | [10-MLX4BIT-AUDIT.md](10-MLX4BIT-AUDIT.md) | **実測(手元)。**公式 MLX-4bit の検証、上流 bf16 との規約照合、減衰ゲート (`in_proj_a`) の感度測定 |
| 7 | [11-OQ4E-G64-REBUILD.md](11-OQ4E-G64-REBUILD.md) | **実測(手元)。**oQ4e-mtp の取得と、非互換 248 本の 8-bit g64 打ち直し (`oQ4e-g64` の作成) |
| 8 | [12-OQ4E-G64-AUDIT.md](12-OQ4E-G64-AUDIT.md) | **実測(手元)。**`oQ4e-g64` の規約照合 (norm / `conv1d` / router) と、2 候補を同じ物差しで並べた表、`q_norm` の焼き込み |
| 9 | [13-PHASE1-REPACK.md](13-PHASE1-REPACK.md) | **実測(手元)。**`.moepack` への repack と `--verify-install`、形式に足した 3 つのセクション、混在ビット幅という Phase 3 の宿題 |
| 10 | [14-REFERENCE.md](14-REFERENCE.md) | **実測(手元)。**float32 の層ストリーミング参照器、逆量子化と算式の検証、実物の初回 forward と生成スモーク、fixtures |
| 11 | [15-PHASE2-GDN.md](15-PHASE2-GDN.md) | **実測(手元)。**Gated DeltaNet カーネル (`qwen_delta_rule`)、3 精度での検証 15 本、TB の 3 通りと 30 層の時間 |
| 12 | [16-QUALITY.md](16-QUALITY.md) | **実測(手元)。**2 候補の平均 NLL (本線の決定) と、`in_proj_a` の実活性再測 (未決着) |
| 13 | [17-PHASE2-KERNELS.md](17-PHASE2-KERNELS.md) | **実測(手元)。**`qwen.metal` の 7 本、負例 6 本を含む検査 29 本、fast math と減衰ゲート、conv のトークン分割 (50.0 → 21.9 ms)、LM head が INT8 だと分かった件 |
| 14 | [18-MIXED-BITS.md](18-MIXED-BITS.md) | **実測(手元)。**混在ビット幅を索引から導く決定、Qwen 用の常駐スキーマ、負例 5 本、本線が開いた記録 |
| 15 | [19-LM-HEAD-INT8.md](19-LM-HEAD-INT8.md) | **実測(手元)。**INT8 の LM head chain、負例 4 本、1 トークン 4.0 ms / 134 GB/s、天井が 79 → 71 tok/s になる話 |
| 16 | [20-PHASE3-DECODE.md](20-PHASE3-DECODE.md) | **実測(手元)。**decode の結線、トークンの一致、負例 5 本、実物で見えた 3 つの食い違い、再帰状態の置き場所 |
| 17 | [21-PHASE4-PREFILL.md](21-PHASE4-PREFILL.md) | **実測(手元)。**55 トークンの参照 (Phase 3 が閉じた)、INT8 の QMM と検査 18 本、prefill の結線、チャンクをまたぐもの 3 つ、チャンク幅の時間 |
| 18 | [22-PHASE5-TOKENIZER.md](22-PHASE5-TOKENIZER.md) | **実測(手元)。**ByteLevel のトークナイザと上流 jinja、上流との突き合わせ 223 本 (負例 4)、CLI の Ornith 経路、tools の JSON の食い違い (キー順の記述は [23 §5-1](23-PHASE5-TOOLS.md) が訂正) |
| 19 | [23-PHASE5-TOOLS.md](23-PHASE5-TOOLS.md) | **実測(手元)。**XML 形のツール呼び出しのパーサと GBNF、生の値の 13 状態、ByteLevel の piece 表、検査 36 本 (負例 6)、テンプレートとの往復で閉じたもの・残ったもの |
| 20 | [24-PREFILL-MOE-PATH.md](24-PREFILL-MOE-PATH.md) | **実測(手元)。**routed expert のタイル版、経路の A/B 4 通り、既定の切り替えとそのデメリット 4 つ、GPU / 取得 / ホストの 3 分割、クールダウンで消える GPU クロック低下 ([21 §5](21-PHASE4-PREFILL.md) の訂正) |
| 21 | [25-CLI-TOOLS.md](25-CLI-TOOLS.md) | **実測(手元)。**マスクつきの LM head と GEN-7 の棄却経路、CLI の `--tools` / `--tool-choice`、検査 6 + 9 本 (負例 3)、実物が書いた呼び出しと「文法が効かなかった」検体 |
| 22 | [26-PHASE8-SERVER.md](26-PHASE8-SERVER.md) | **実測(手元)。**`QwenServerSession` と `QwenGenerationPlan`、生成ループを 1 本にした話、できないこと 4 つ (画像 / 投機 / prompt cache / サンプラ)、実物が HTTP で答えた 13 件、`tool_choice: required` が前置きで回る件 |
| 23 | [27-PHASE6-THROUGHPUT.md](27-PHASE6-THROUGHPUT.md) | **実測(手元)。**実タスク 4 本の prefill と生成、直列だった 2 か所 (遅延 join / 読みの重ね)、A/B の数字、スロット・チャンク・read advice・腕のスイープ、机上のヒット率カーブ、**残った取得の解剖 (ページ写像) と層をまたぐ先読み** (§9) |
| 24 | [28-PREFETCH-IDEAS.md](28-PREFETCH-IDEAS.md) | **検討 (実測なし)。**先読みの符号が Gemma と Ornith で反転した機序 (待ちの「通貨」)、Gemma で死んだ手の復活判定、N>8 / d=2 / 予測の固定費の攻め手と実験手順。#29 の判断材料 |
| 25 | [29-MTP-PREFETCH-OUTLOOK.md](29-MTP-PREFETCH-OUTLOOK.md) | **検討 (実測なし)。**28 の在庫の Phase 7 (MTP) 下での再判定 (写像の固定費が受理長で割れる / スロット判定の反転 / partial プランの昇格)、MTP でしか出ない手 (ドラフトの陰・token id 事前分布・受理長の CPU 先行測定)、28 §3-6 の「エージェント経路では効かない」の訂正 |
| 26 | [30-MTP-HEAD-GRAFT.md](30-MTP-HEAD-GRAFT.md) | **実測。**同梱 MTP ヘッドが乱数初期化であることの検算 (上流 bf16 / oQ4e 42 本 / 256 エキスパートの CV)、zero-centered gamma の規約、差し替え候補の選別と**形式適合 19 本 → 42 本**、分割順・norm の +1・量子化の代償の実測、**差し替えの実行と検算** (§6)、mlx-lm#1740 の読み方と pre/post-norm の未確認 |
| 27 | [31-PREFETCH-CHEAPER.md](31-PREFETCH-CHEAPER.md) | **実測(手元)。**28 の在庫を回した結果 — 計器 (`declined=0`、wait 1.1 ms/tok) で d=2 とリトライが落ち、N>8 は名指しが +23 ポイント積むのに写像が陰に入りきらず負け、**preview の select を出さない**手が全 26 ペアで +2.4〜3.1%。#29 の数字。**適応スキップも机上で否定した** (§7: 層別 miss 分布に崖が無い、preview は 39 本/tok) |
| 29 | [33-MTP-ACCEPTANCE.md](33-MTP-ACCEPTANCE.md) | **実測(手元) + 導出。**Phase 7 の M0 — 受理率 P1 と受理長 a をタスク別に、実トレースから引いた検証幅ごとのエキスパート和集合とその 2 度の訂正 (§3-4 / §3-5)、**幅 2 の 1 パス費用の実測 (1.27〜1.30 倍) と取り分 +15〜29%** (§3-7、**取り分は [36](36-MTP-DECODE.md) が実測で置き換えた** — 比は正しく、掛ける先が違った)、**GDN の snapshot/restore はコピー 0 回にできる** (§3-6)、参照器とランタイムの 192 トークン一致率 |
| 37 | [42-FREETOKEN-IDEAS.md](42-FREETOKEN-IDEAS.md) | **検討 (公表値のみ)。**CUDA 向け edge-native MoE serving エンジン FreeToken (arXiv:2608.16157) の機構在庫と写し先 — **意味的境界の anchor checkpoint** ([41](41-PROMPT-CACHE.md) の全ミス 2 種への設計済みの答え、会話内は再帰状態の写真 ≈ 62 MiB だけで済む形)、global expert cache のオフライン判定 ([31 §7](31-PREFETCH-CHEAPER.md) の「崖が無い」が逆風)、KV ↔ スロットの弾力予算、cold expert の CPU fixup (#31 (c) の変種)、写さないもの 4 つ |
| 36 | [41-PROMPT-CACHE.md](41-PROMPT-CACHE.md) | **実測(手元)。**その場保持の 1 エントリ・全部か無かの prompt cache — 規則と純粋型、34 の式が外れていた 1 点 (**受理して終わった投機パスは最後のトークンも食っている**)、MTP ヘッドのキャッシュという 34 が挙げていなかった穴、**再レンダの継ぎ目は実測ではずれない**、取り分 (TTFT 10.33 → 0.72 秒)、続きと再計算が分岐する話、`--qwen-resume` 7 本 |
| 35 | [40-MTP-GRAMMAR.md](40-MTP-GRAMMAR.md) | **実測(手元)。**文法つきの MTP (再畳み込みに**行を渡す**だけで済んだ話、2 行への当て方と順序、受理判定を制約後の引きと比べる理由) と、サーバーの `--draft-block-size 2` (幅は 2 だけ・sidecar は起動時・`max_tokens` の 1 位置)、3 腕のトークン一致 55/55 と 63/63、実タスクの A/B (decode ×1.14)、`--qwen-constrain` の 6 本追加、**prompt cache が無いことがエージェントで一番効く**という残り |
| 34 | [39-RESIDENCY-COMMIT.md](39-RESIDENCY-COMMIT.md) | **実測(手元)。**[38 §7](38-MTP-VERIFY-PATH.md) の残り 2 件を測って手を 2 本試した — ドラフタの 6.0 ms は**89% が GPU** (融合の上限は 0.64 ms/パス)、層またぎ先読みは**負け** (予測 router が幅 2 で +8.3 ms/パス、commit の通貨は回数)。採ったのは **`syncResidency` を背景の直列キューへ**投げる 1 手で、`io` 36.5 → 8.3 ms/パス、**4 タスク 4 腕すべて勝ち** (素の decode ×1.047〜1.124、MTP ×1.039〜1.083)。**set を捨てる腕は ×0.975〜0.986** と 27 §9-2 を再現するので、機序は「先回りが無駄」ではなく「スレッドが違う」。語彙切り詰めは id 分布で**着手前に潰した** |
| 33 | [38-MTP-VERIFY-PATH.md](38-MTP-VERIFY-PATH.md) | **実測(手元)。**[37](37-MTP-POSTMORTEM-PLAN.md) の改善案 A を、着手前に段ごとの計器で検算した — **内訳は外れていた** (route grouping は 0.25 ms/パス、11 ms ではない)。真犯人は**プロンプト用 attention カーネルが幅 1〜2 で KV 全体を 16 スレッドグループに歩かせる**こと (文脈 +2,640 位置で decode +5.0 ms 対 T 行経路 +26.0 ms)。既存の split-KV に**行ごと**に差し替え (ブロック 1 発だと投機の中立性が 95/96 に落ちる)、**t4 ×0.802 → ×1.110**、5 本中 3 本が勝ち。**新カーネル 0 本。**A の残り 2 弾は取り下げ (§7-4) |
| 32 | [36-MTP-DECODE.md](36-MTP-DECODE.md) | **実測(手元)。**MTP を実機で通した決着 — ヘッドの sidecar (`.moepack` は無改造)、GPU 上の MTP ブロック、幅 2 の検証パスとコピー 0 回の巻き戻し、**強制棄却の対照が設計の誤りを捕まえた話** (§3-1)、費用の追い込み (コマンドバッファ本数は無罪、密射影のカーネルが犯人)、**実タスク 5 本の A/B (×1.199〜×0.843)**、[33 §3-7](33-MTP-ACCEPTANCE.md) の見積もりがどこで曲がったか |
| 31 | [35-PREFILL-CHUNK-WIDTH.md](35-PREFILL-CHUNK-WIDTH.md) | **実測(手元)。**`QwenPrefill` が要求幅ではなく scratch の幅でプロンプトを切っていた — 1 プロセスで幅を変えると 2 本目以降が最初の幅で走る。**[21 §4](21-PHASE4-PREFILL.md) の「チャンク 8 (3 チャンク)」は空振りしていた**。直して初めて引き継ぎが本当に走り、通った。答えは 1 度も間違っていない |
| 30 | [34-PROMPT-CACHE-ESTIMATE.md](34-PROMPT-CACHE-ESTIMATE.md) | **検討 (導出のみ)。**32 §2 の snapshot-restore 型 prompt cache の取り分を机上で — prefill の**床 1.30 秒**を含む正しい式、シナリオ別の取り分 (**最大 9.2 秒**)、1 エントリの費用 (KV 20 KiB/tok × 10 層 + GDN 61.41 MiB) と予算判定、**その場保持なら 0 バイト**、足りていないもの 5 つと実装規模、**再レンダの継ぎ目という一番大きい危険** |
| 28 | [32-NVMAI-ADOPT.md](32-NVMAI-ADOPT.md) | **検討 + 実測(NVMAI)。**同じ Ornith 1.5 を動かす兄弟ランタイム `~/LLM/NVMAI` からの移植候補 — MTP の checkpoint/restore と損益分岐 (p > 0.585)、pre-final-norm の決着、snapshot-restore 型 prompt cache、decode の hit-fixup、interleaved A/B の作法 |

## 表記

PLAN.md / PLAN_QAT.md / PLAN_VISION.md と同じ **実測** / **導出** / **未確認**。
ただし本計画の **実測** は 2 種類ある。混ぜないために区別する:

| 記号 | 意味 |
| --- | --- |
| **実測(上流)** | 上流リポジトリの実体を取得して確認した事実。`config.json`、`model.safetensors.index.json`、各シャードの safetensors ヘッダ (HTTP range で先頭のみ取得)、`tokenizer_config.json`、`chat_template.jinja`、`transformers` の `modeling_qwen3_5_moe.py`、omlx のソース |
| **実測(手元)** | この機械で数字を取ったもの。**CPU のもの** (ファイルを読む / 逆量子化する / float32 で流す) と、**GPU のもの** ([15](15-PHASE2-GDN.md) 以降のカーネル検査とマイクロベンチ) がある。**モデルを載せて動かしたのは [20](20-PHASE3-DECODE.md) が最初**で、そこの時間は直列・prefill 無しの n=1 なので運用値ではない |

## 運用ルール

- 各文書は**現在の結論**を書く。実測の経緯と数字は 10 番台の結果文書が持ち、
  **測定の結論を二重に書かない** (docs/mtp と同じ作法)。食い違ったら結果文書
  (番号の大きいもの) が正
- 反復 3 未満のセルには解釈を書かない。数字だけ置く
- 運用点 (スロット数・チャンク幅) の**既定は変えない**。候補追加の提案に留め、
  判断はユーザーが行う (Gemma の `[8,16,24,32]` は 2026-08-20 のユーザー確定値)
- Gemma 4 経路の実測値を動かさない。共有コードに手を入れるときは Gemma のベンチを取り直す

## 参照

- 上流 (bf16): [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
- 公式 MLX 4-bit: [`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit)
- oQ 4-bit (MTP / vision つき): [`scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp`](https://huggingface.co/scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp)
- MTP ヘッド (**採用・差し替え済み**、[30 §6](30-MTP-HEAD-GRAFT.md)):
  [`shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY`](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY) /
  [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (ドナー)。
  元の指摘は [discussion #10](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/discussions/10)
- MTP の配線の参考実装 (未マージ): [mlx-lm#1740](https://github.com/ml-explore/mlx-lm/pull/1740) /
  [#990](https://github.com/ml-explore/mlx-lm/pull/990)
- 兄弟ランタイム: `~/LLM/NVMAI` (Apache-2.0) — 同じ Ornith 1.5 35B-A3B の
  Swift/Metal ランタイム。移植候補の選定は [32](32-NVMAI-ADOPT.md)
- FreeToken: [`FlashML-org/FreeToken`](https://github.com/FlashML-org/FreeToken) / [arXiv:2608.16157](https://arxiv.org/abs/2608.16157) — CUDA 向け edge-native MoE serving。機構の在庫と写し先は [42](42-FREETOKEN-IDEAS.md) (ソース未読・公表値のみ)
- 量子化ツール: [`jundot/omlx`](https://github.com/jundot/omlx) (Apache-2.0) —
  `docs/oQ_Quantization.md` / `omlx/oq.py` / `omlx/custom_kernels/qwen35_prefill/gdn.py`
- `transformers` `models/qwen3_5_moe/modeling_qwen3_5_moe.py`
- 本リポジトリ: PLAN.md / PLAN_QAT.md /
  PLAN_VISION.md / [docs/mtp/README.md](../mtp/README.md) /
  [docs/SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) / [docs/serving/SPEC.md](../serving/SPEC.md)
