# NN. Q<k> <題> — <結論を 1 行で> (実測、YYYY-MM-DD)

証拠: [`bench/qfn/results/YYYY-MM-DD-pair-<label>/`](../../bench/qfn/results/) (2 機) または
[`bench/qfn/results/YYYY-MM-DD-Mac15-6-<label>/`](../../bench/qfn/results/) (MBP 単体)。
ホスト札 (各ディレクトリの `hostinfo.mbp.txt` / `hostinfo.m6.txt` / `link.txt` をそのまま貼る):

```
MBP: 測定: YYYY-MM-DD / Mac15,6 Apple M3 Pro 18GB / macOS 15.7.5 (24G624) / Swift 6.3.3 / SDK 26.5 / Xcode 26.6 / APPLE SSD AP1024Z / commit xxxxxxx / minos 15.0
M6:  測定: YYYY-MM-DD / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit xxxxxxx / minos 15.0
link: bridge0 169.254.x.x → 169.254.y.y / rtt 0.xx ms / 1 GiB in 0.xx s / 分け方 A k = NN | B σ = 0.NN
```

コマンド (`command.txt` をそのまま貼る)。反復 n = <数>。**反復 3 未満のセルには解釈を書かない。**

**台帳** ([06](06-LEDGER.md)): この変更は <既定 | 候補 (`Q38_…=…` で開く)>。
既定なら `ledger-<名>` の行「変更前 X s → 変更後 Y s、差 Z s」を、単体と 2 機の両方の列で貼る。候補なら「台帳 §1 は動かない」と書く。

## 0. 結論

1〜3 項目。数字は証拠ディレクトリの `summary.tsv` から中央値を引き、レンジを併記する。**倍率・%・tok/s は題にも本文にも書かない。**秒と、秒の差だけ。

## 1. 数字

| | 変更前 | 変更後 | 差 s (**導出**) |
| --- | ---: | ---: | ---: |
| prefill 壁時計 中央値 (レンジ)。単体 / 2 機、本文名、測った量 (a)(b)(c) を書く | | | |
| 1 run の読み (Pageins、`iostat` の disk ごとの MB、M6 は内蔵 + 外付け) | | | |
| 段の秒 (`Q38_ROUND_LOG` の StepProfile: preRouter / route / routed / ple。GPU 秒は膨らむので壁時計と足さない) | | | |
| M6 側の秒 (M6 のログ: A なら層 0〜k−1、B なら担当分の expert) と MBP 側の秒、link の待ち | | | |

## 2. 分かったこと / 外れたこと

[04-GATES.md](04-GATES.md) の合否に照らして書く。予測 ([08 §3](08-CANDIDATES.md) の式) が外れたなら、どの量が違ったかを 1 行で。

## 3. 次の Gate に渡すもの

分母になる数字、開いたまま残る未確認、やめると決めたもの。
