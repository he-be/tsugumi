# NN. G<k> <題> — <結論を 1 行で> (実測、YYYY-MM-DD)

証拠: [`bench/m6/results/YYYY-MM-DD-Mac18-5-<label>/`](../../bench/m6/results/) (M6)、
[`bench/m6/results/YYYY-MM-DD-Mac15-6-<label>/`](../../bench/m6/results/) (MBP)。
ホスト札 (各ディレクトリの `hostinfo.txt` をそのまま貼る):

```
測定: YYYY-MM-DD / Mac18,5 Apple M6 16GB / macOS 27.0 (26A428) / Swift 6.4 / SDK 27.0 / CLT / APPLE SSD AP0256Z / commit xxxxxxx / minos 15.0
```

コマンド (`command.txt` をそのまま貼る)。反復 n = <数>。**反復 3 未満のセルには解釈を書かない。**

## 0. 結論

1〜3 項目。数字は証拠ディレクトリの `summary.tsv` から中央値を引き、レンジを併記する。

## 1. 数字

| | M6 | MBP | 比 |
| --- | ---: | ---: | ---: |
| prefill 壁時計 中央値 (レンジ) | | | |
| pp | | | |
| prefill io | | | |
| GPU 側 (prefill − io、**導出**) | | | |

## 2. 分かったこと / 外れたこと

[04-GATES.md](04-GATES.md) の合否に照らして書く。予測が外れたなら、どの前提が違ったかを 1 行で。

## 3. 次の Gate に渡すもの

分母になる数字、開いたまま残る未確認、やめると決めたもの。
