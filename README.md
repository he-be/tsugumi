<p align="center">
  <img src="docs/assets/tsugumi.png" alt="Tsugumi のロゴ: 地面に立つツグミの横向きシルエット" width="280">
</p>

<h1 align="center">Tsugumi</h1>

<p align="center">
  <strong>16GB の Mac で、ちゃんと使えるローカル AI。</strong>
</p>

<p align="center">
  <img alt="Swift 6.2" src="https://img.shields.io/badge/Swift-6.2-F05138?logo=swift&logoColor=white">
  <img alt="Metal 3.2 or later" src="https://img.shields.io/badge/Metal-3.2%2B-5E5CE6">
  <img alt="macOS 15 or later" src="https://img.shields.io/badge/macOS-15%2B-000000?logo=apple&logoColor=white">
  <a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/License-Apache%202.0-2ea44f"></a>
</p>

<p align="center">
  <a href="#使ってみる">使ってみる</a> ·
  <a href="docs/MAC_APP.md">Mac アプリ</a> ·
  <a href="docs/WEB_SEARCH.md">Web 検索</a> ·
  <a href="docs/LOCAL_WIKIPEDIA.md">ローカル Wikipedia</a> ·
  <a href="docs/CLI.md">CLI</a> ·
  <a href="docs/OPENAI_SERVER.md">ローカルサーバ</a> ·
  <a href="docs/SYSTEM_DESIGN.md">仕組み</a> ·
  <a href="#upstream">Upstream</a>
</p>

![Gemma 4 26B-A4B で生成している Tsugumi](docs/assets/tsugumi-app.webp)

16GBのMacでローカルLLMといえば、これまでE4Bか12Bだった。Tsugumi は、生成結果を落とさずに
13.3 GiB まで絞った **Gemma 4 26B-A4B** を、普段使いの Mac で動かす。
RAM確保のために他のアプリを閉じる必要はない。メモリが混んでいれば遅くなるが、止まらない。

> **In English** — Tsugumi runs a 26B mixture-of-experts checkpoint on an
> ordinary Mac by asking the OS for memory instead of reserving it: the process
> itself needs 1.3 GB, and the rest of the 13.28 GiB lives in the page cache,
> which the OS reclaims whenever anything else wants it. Nothing leaves the
> machine. It is a fork of [TurboFieldfare](#upstream); the sections below are
> in Japanese, and the design notes under `docs/` are a mix of both.

## メモリが高い、という話

**Mac のメモリは後から足せない。** 
**これとは別の力として、DRAM 相場そのものが上がっている。** 

Tsugumi が節約するのは RAM の使用量ではなく、メモリを盛ったMacの差額**36,000 円のほう**である。

## 要る RAM と、借りる RAM

| | 量 | 性質 |
| --- | --- | --- |
| **要る RAM** | 1.3 GB | プロセスが確保する私有メモリ。これが下限 |
| **あれば使う RAM** | 最大 13.28 GiB | ページキャッシュ。**OS がいつでも取り返せる** |

13.28 GiB は「使わない」のではない。空いていれば載るし、載っていれば速い。
売りは *RAM を使わないこと* ではなく、**借りているだけで、いつでも返せること**。
他のアプリが要求すれば OS が取り上げ、取り上げられた分は遅くなる。それだけで、
落ちない・フリーズしない・スワップ地獄にもならない。

- 余裕のある機械 → 速い
- 余裕のない機械 → 遅い
- **どちらでも同じアプリ、同じウェイト、同じ出力**

24GB を買った人が速いのは本当で、それは否定しない。**16GB でも同じものが動いて、
遅くなるだけ**、というのがこのプロジェクトの主張である。

## 上流とは逆を向いている

Tsugumi は [TurboFieldfare](#upstream) のフォークで、動機は同じ「メモリが高い」
である。設計は逆を向いた。

**上流は予算を切った。** 2 GB という上限を決めて、自前のエキスパートキャッシュで
その中に収めた。**Tsugumi はその自前キャッシュを捨てて、OS のページキャッシュに
預けた。**上限を決めるのをやめた、と言ってもいい。だから「もっと節約した」ではなく、
同じ動機から反対側に歩いた結果として、上の表のような性質になっている。

## なぜ 26B なのか

ローカル LLM が動いた、という記事は氾濫している。ただし中身は E4B か 12B が多い。
12B は「12〜16GB の VRAM に載るのが利点」の中間サイズで、E4B は単機能特化である。
用途が書かれていない記事が多いのは、書けるほどの用途がないからだと思う。

本格的に使えるのは 26B の MoE からで、**Gemma 4 の日本語の強みがそのまま出るのも
このサイズ**である (日本語の自然さ・指示追従・長文処理については外部の評価が
すでに固まっているので、ここで証明はしない)。そして 26B は 16GB には載らない、
というのが定説だった — Q4_K_M で 16〜18 GB 要るからである。

Tsugumi が置いた石はここで、**QAT ウェイトを、生成結果を変えないまま 13.28 GiB
まで落とした**。生成結果はもちろん維持している。

ダウンロードが 15.7 GB あるのは、この本体に vision タワーと MTP ドラフタが
乗るからである。どちらも別ファイルで、使わない実行では読まれない。
上の表の 13.28 GiB は本体のぶんである。

**この設計は QAT が前提で成立している。** Gemma 4 は QAT の公式配布が全サイズ
揃っていて、26B-A4B には MTP ドラフタが同梱される。だからこの話は Gemma 4 でしか
語れない。

別モデル **Ornith-1.5 35B-A3B** も動くが、こちらはエージェントコーディング用。

| | [Gemma 4 26B-A4B QAT](https://ai.google.dev/gemma/docs/core/model_card_4) | [Ornith-1.5 35B-A3B](https://huggingface.co/ornith-ai) |
| --- | --- | --- |
| 位置づけ | 日本語チャット | コーディング |
| ダウンロード | 15.7 GB | 21.0 GB |
| 画像 | 1 メッセージに 4 枚まで | なし |
| 思考 | トグル、既定 OFF | トグル、既定 ON |
| 投機デコード | ドラフタ、ブロック 4 | MTP ヘッド、幅 2 |

## 使ってみる

```bash
git clone https://github.com/he-be/tsugumi.git
cd tsugumi
swift build -c release
.build/release/TsugumiMac
```

リリースビルドはアプリと、その隣で動く decode service を作る (アプリはモデルと
Metal をこのプロセスに持たせる)。初回は Swift Package Manager がトークナイザの
パッケージも取ってくる。

アプリが開いたら **Download** を選ぶ。既定は Gemma 4 で、完成品のパック
(15.7 GB) がそのまま降ってくる。全ファイルが SHA-256 のピンと照合され、
一致したものだけがインストール先に入る。あとは **Load Model** を選んで打ち始める。

### .app にする

チェックアウトの無い Mac に持っていくなら、バンドルを組む:

```bash
Scripts/app/make_app.sh --zip     # dist/Tsugumi.app と dist/Tsugumi-<版>.zip
Scripts/app/verify_app.sh         # 構造・署名・リソース解決・launchd を見る
```

署名は既定で ad-hoc なので、受け取った側で隔離属性を外す必要がある。
手順と、公証まわりの残作業は [docs/DISTRIBUTION.md](docs/DISTRIBUTION.md)。

### 動作要件

- Apple Silicon の Mac。狙いは普通の 16GB 機で、開発機は 18GB の M3 Pro
- macOS 15 (Sequoia) 以降
- Xcode 26 / Swift 6.2 以降
- モデルのディスク: Gemma 4 が 15.7 GB、Ornith が 21.0 GB
- 初回インストールのためのネットワーク

macOS 26 では Metal 4 / MSL 4.0 のテンソルカーネル (MPP の prefill matmul と
tensor-ops の prefill attention) が有効になる。macOS 15 ではシェーダが MSL 3.2
として compile され、それらは `__HAVE_TENSOR__` のガードで落ちて、可搬な
prefill カーネルに退く。decode は変わらず、prefill だけが macOS 15 で遅い。

## Mac アプリ

1. **Load Model** を選ぶ
2. プロンプトを書く
3. **Generate**、または <kbd>Command</kbd>+<kbd>Return</kbd>
4. 途中で止めるのは停止ボタンか <kbd>Escape</kbd>

会話はマルチターンで、左のサイドバーに並ぶ。
`~/Library/Application Support/Tsugumi/chats.json` に保存され、再起動しても
戻ってくる。

サンプラの既定は各チェックポイントの公式値である (Gemma は temp 1.0 /
top-k 64 / top-p 0.95 で編集可、Ornith は 0.6 / 20 / 0.95 で固定)。
コンテキスト長・エキスパートキャッシュ・prefill などは右ペイン。

画像は Gemma だけが受ける (1 メッセージ 4 枚まで)。Ornith はテキスト専用で、
音声と動画はどちらも対象外。ツールの実行はアプリも CLI もしない。

## CLI とローカルサーバ

同じ `.moepack` を、アプリを開かずに使える。

```bash
swift build -c release --product TsugumiServer
.build/release/TsugumiServer --model scratch/gemma4-qat-sym.moepack
```

`http://127.0.0.1:8080/v1` で Chat Completions、ストリーミング、function tools、
`data:` URI の画像、投機デコード、プロンプトキャッシュに応じる。

モデルを持つプロセスは、アプリ・CLI・サーバのどれか 1 つだけを同時に動かすこと。

## 仕組み

各層で、Metal が常駐ウェイトから attention とルータを計算する。CPU はルータの
上位 8 エキスパートを見て、その層のマッピングから必要なエキスパートだけを
`MTLResidencySet` に入れ、GPU はそのバッファを直接読む。**層のファイルは
`MAP_SHARED` で張りっぱなし**なので、常駐になるのはコマンドバッファが名指した
ぶん (1 層あたり 26.9 MB) だけで、残りは OS のページキャッシュの側にいる —
これが「借りて、返せる」の実装である。私有スロットに `pread` していた旧経路は
`TF_EXPERT_MMAP=0` で戻せる (A/B の測定用)。

## テストと貢献

```bash
Scripts/test.sh
```

モデルを走らせる前に、重いアプリを閉じて `memory_pressure -Q` を見ること。

## Upstream

Tsugumi は **Andrey Mikhaylov** の
**[TurboFieldfare](https://github.com/drumih/turbo-fieldfare)** (Apache-2.0) の
フォークである。エンジンの骨格はこちらのリポジトリの通りで、`.moepack` コンテナ (元は `.gturbo`)、
ストリーミング repack、境界のあるエキスパートキャッシュ、量子化 GEMV・
attention・MoE・正規化・RoPE・サンプリングの Metal カーネル、そして最初の Mac
アプリまでが彼の仕事である。これが無ければ以下のどれも存在しない。

フォークで足したもの: 2 つ目のアーキテクチャ (Qwen3.5-MoE / Ornith-1.5、線形
attention 層と自前の MTP ヘッド)、投機デコード、OpenAI 互換サーバ、プロンプトキャッシュ、
QAT の対称量子化経路、vision タワー、2 モデルを跨ぐマルチターンのチャットに作り直した Mac アプリ。

## ライセンスとモデルの条件

Tsugumi のソースとドキュメントは [Apache License 2.0](LICENSE) である
(派生元の上流も同じ)。

**ウェイトは同梱していない。**アプリが別途ダウンロードし、それぞれの条件に従う:

| ウェイト | 条件 |
| --- | --- |
| Gemma 4 26B-A4B | Google は Gemma 4 を [Apache License 2.0](https://ai.google.dev/gemma/apache_2) で配布している。ただし投機に使うドラフタは `license:gemma` の成果物なので、[Gemma の利用規約](https://ai.google.dev/gemma/terms)が付いてくる |
| Ornith-1.5 35B-A3B | MIT (上流のチェックポイント) |
| Ornith の MTP ヘッド | Apache 2.0 (shisa.ai の公開物から移植) |

Tsugumi は独立した研究プロジェクトであり、Google、Alibaba その他いかなる
モデル公開元とも提携・後援・推薦の関係にない。
