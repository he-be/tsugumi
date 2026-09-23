# NN. G<k> <題> — <結論を 1 行で> (実測、YYYY-MM-DD)

証拠: [`bench/m6/results/YYYY-MM-DD-Mac18-5-<label>/`](../../bench/m6/results/) (M6)、
[`bench/m6/results/YYYY-MM-DD-Mac15-6-<label>/`](../../bench/m6/results/) (MBP)。
ホスト札 (各ディレクトリの `hostinfo.txt` をそのまま貼る):

```
測定: YYYY-MM-DD / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit xxxxxxx / minos 15.0
```

コマンド (`command.txt` をそのまま貼る)。反復 n = <数>。**反復 3 未満のセルには解釈を書かない。**

**台帳** ([06](06-LEDGER.md)): この変更は <既定 | 候補 (`TF_…=…` で開く)>。
既定なら `ledger-<名>` の行「変更前 X s → 変更後 Y s、差 Z s」を貼る。候補なら「台帳 §1 は動かない」と書く。

## 0. 結論

1〜3 項目。数字は証拠ディレクトリの `summary.tsv` から中央値を引き、レンジを併記する。**倍率・%・「下限の何倍か」は題にも本文にも書かない。**秒と、秒の差だけ。

## 1. 数字

| | 変更前 | 変更後 | 差 s (**導出**) |
| --- | ---: | ---: | ---: |
| prefill 壁時計 中央値 (レンジ)、測った量 (a)(b)(c) を書く | | | |
| prefill io | | | |
| 段の GPU 秒 (profile。膨らむので壁時計と足さない) | | | |
| 段の下限 s ([04 B1](04-GATES.md)、**導出**) | | | |

## 2. 分かったこと / 外れたこと

[04-GATES.md](04-GATES.md) の合否に照らして書く。予測が外れたなら、どの前提が違ったかを 1 行で。

## 3. 次の Gate に渡すもの

分母になる数字、開いたまま残る未確認、やめると決めたもの。
