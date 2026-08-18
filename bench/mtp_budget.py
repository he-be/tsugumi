#!/usr/bin/env python3
"""MTP の改修候補を、着手前に「完全勝利したらスコアが何倍か」に翻訳する。

`docs/mtp/22-GOAL-RESET.md` の感度表を出す。マイクロな実測 (verify ブロックの
費用 / 受理長 / decode 比率) は 1 つずつ意味を持たないので、ここで 1 本の関数に
通してから読む。手で書き換えないこと — 入力は全部フラグで、既定値の出典は
`SOURCES` に書いてある。

  ./bench/mtp_budget.py                      # 既定 (ゴール条件 cold 80 スロット)
  ./bench/mtp_budget.py --tasks              # タスク間の幅とレバーの大きさ
  ./bench/mtp_budget.py --fixed 0.48         # 1 シナリオだけ引く

モデル (docs/mtp/21-GOAL-CONDITION-RESULTS.md §5-6):

    verify(k) = F + M*k                        [decode ステップ]
    1 ラウンド = verify(k) + (k-1) * draft
    産出       = a(k) + 1
    decode 倍率 = 産出 / 1 ラウンド
    e2e 倍率    = 1 / ((1 - share) + share / decode 倍率)

`share` は wall clock のうち decode が占める割合。prefill・vision tower・load は
投機の対象外なので、ここが端から端の倍率の天井を決める。
"""

import argparse

SOURCES = """入力の出典 (すべて実測):
  F=0.96 M=0.36   21-GOAL-CONDITION §5 の当てはめ (ゴール条件 cold 80 スロット)
                  verify(2)=1.65 / verify(4)=2.44 (21 §3) を再現する
  draft=0.098     21 §6 (ドラフター 1 ステップ / decode ステップ)
  a               21 §6 (vis は本書が初測定)、14-M3.5 §4 (math / m / story)
  share=0.888     22-GOAL-RESET §4: sample_imgs 8 枚・Reasoning ON・temp 0・
                  80 スロットの中央値 (decode 45.8 s / wall 51.6 s)
"""

# 受理長 a[k] — temp 0
TASKS = {
    "math":  {2: 0.794, 3: 1.430, 4: 1.947},
    "m":     {2: 0.715, 3: 1.222, 4: 1.562},
    "vis":   {2: 0.739, 3: 1.243, 4: 1.596},
    "story": {2: 0.547, 3: 0.813, 4: 0.937},
}

# 感度表の行。(名前, F, M, タスク)
SCENARIOS = [
    ("現状 F=0.96 M=0.36",        0.96, 0.36, "vis"),
    ("F 半減 (0.48)",             0.48, 0.36, "vis"),
    ("F→0.20 (床付近)",           0.20, 0.36, "vis"),
    ("M 半減 (0.18)",             0.96, 0.18, "vis"),
    ("F 半減 + M 半減",            0.48, 0.18, "vis"),
    ("F→0.20 + M→0.18",          0.20, 0.18, "vis"),
    ("現状費用 + a が math 並み",   0.96, 0.36, "math"),
]


def decode_ratio(fixed, marginal, k, a, draft):
    """1 ラウンドの費用と産出から decode 倍率を引く。"""
    return (a[k] + 1) / (fixed + marginal * k + (k - 1) * draft)


def e2e_ratio(ratio, share):
    """decode 倍率を端から端の壁時計倍率に翻訳する。"""
    return 1.0 / ((1.0 - share) + share / ratio)


def required_decode_ratio(target, share):
    """e2e 目標に必要な decode 倍率。share が天井を決めるので、届かない目標もある。"""
    floor = 1.0 / (1.0 - share)
    if target >= floor:
        return None
    return 1.0 / ((1.0 / target - (1.0 - share)) / share)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixed", type=float, help="F をこの値に固定して 1 行だけ引く")
    p.add_argument("--marginal", type=float, default=0.36, help="M (--fixed と併用)")
    p.add_argument("--draft", type=float, default=0.098, help="ドラフター 1 ステップ")
    p.add_argument("--share", type=float, default=0.888, help="wall のうち decode の割合")
    p.add_argument("--target", type=float, default=1.33, help="e2e の目標倍率")
    p.add_argument("--task", default="vis", choices=sorted(TASKS))
    p.add_argument("--tasks", action="store_true",
                   help="タスク間の幅とレバーの大きさを出す")
    p.add_argument("--ks", default="2,3,4")
    args = p.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    need = required_decode_ratio(args.target, args.share)
    ceiling = 1.0 / (1.0 - args.share)

    print(SOURCES)
    print(f"decode 比率 share={args.share:.3f} → e2e の天井 {ceiling:.2f} 倍 "
          f"(decode が無限に速くなっても prefill/tower/load が残る)")
    if need is None:
        print(f"目標 e2e {args.target} は天井を超えている。目標かモデルが違う。")
    else:
        print(f"目標 e2e {args.target} に必要な decode 倍率: {need:.2f}")
    print()

    if args.fixed is not None:
        rows = [(f"F={args.fixed} M={args.marginal}", args.fixed, args.marginal, args.task)]
    else:
        rows = SCENARIOS

    if args.tasks:
        for name, fixed, marginal, _ in rows:
            print(name)
            best = {}
            for task, a in TASKS.items():
                vals = {k: e2e_ratio(decode_ratio(fixed, marginal, k, a, args.draft),
                                     args.share) for k in ks}
                bk = max(vals, key=vals.get)
                best[task] = vals[bk]
                cells = "  ".join(f"k={k} {vals[k]:.2f}" for k in ks)
                print(f"   {task:<6} {cells}   最良 k={bk} {vals[bk]:.2f}")
            lo, hi = min(best.values()), max(best.values())
            print(f"   → タスク間の幅 {lo:.2f}〜{hi:.2f} (差 {hi - lo:.2f})")
            print()
        return

    head = "".join(f"{'k=' + str(k):>9}" for k in ks)
    print(f"{'シナリオ':<26}{head}     (e2e 倍率)")
    for name, fixed, marginal, task in rows:
        a = TASKS[task]
        cells = "".join(
            f"{e2e_ratio(decode_ratio(fixed, marginal, k, a, args.draft), args.share):>9.2f}"
            for k in ks)
        print(f"{name:<26}{cells}")
    print()
    print("読み方: 着手前にその改修の行を引く。上限が中止条件 (e2e 1.10) を"
          "動かさないならやらない。")


if __name__ == "__main__":
    main()
