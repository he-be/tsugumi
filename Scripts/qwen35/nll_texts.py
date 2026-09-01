# -*- coding: utf-8 -*-
"""NLL 比較用の文章 4 本を作り、両チェックポイントのトークナイザで同じ ID に
なることを確かめて JSON に落とす (docs/qwen35moe/04 「次の一手」#9)。"""
import json, sys
from pathlib import Path
from tokenizers import Tokenizer

JA_PROSE = """朝の駅は雨の匂いがした。改札の前で傘をたたみながら、彼女は昨日の電話のことを思い出していた。
受話器の向こうの声は落ち着いていて、けれど言葉のあいだにわずかな間があった。その間が何を意味するのか、
考えはじめるときりがない。ホームに上がると、いつもの快速はまだ来ていなかった。
黄色い線の内側で、濡れたコートの肩を軽くはらう。向かいのホームで子どもが水たまりを踏み、
母親がそれをたしなめる声が、雨音のあいだから短く届いた。電車が入ってくると、
窓に映る自分の顔が一瞬だけ大きくなって、それから流れて消えた。席には座らず、
ドアの脇に立って外を見ることにした。灰色の街が後ろへ滑っていく。次の駅で降りて、
もう一度あの番号にかけてみようと思った。今度はこちらから、間を置かずに話そうと。"""

JA_TECH = """このモデルは 40 層のうち 30 層が線形注意で、残り 10 層だけが通常の注意機構である。
線形注意の層は KV キャッシュを持たず、代わりに文脈長に依らない固定サイズの再帰状態を持つ。
そのため長い文脈でもメモリの増え方が緩やかになる一方で、途中の状態を捨てたり巻き戻したりする操作が
そのままでは成り立たない。エキスパートは 1 層あたり 256 個あり、トークンごとに 8 個が選ばれる。
ルーターは全体の softmax を取ってから上位 8 個を選び、選ばれた重みを再正規化する。
量子化は affine で、グループサイズは 64、ビット幅は 4 と 8 が混在している。
埋め込みと注意の投影は 8 ビット、ルーティングされるエキスパートは 4 ビットで持つ。
回転位置符号は次元の 4 分の 1 にだけ掛かり、組の取り方と分母がほかの実装と異なるため、
既存のカーネルをそのまま流用すると静かに間違った結果が出る。"""

EN_TECH = """The runtime streams a single checkpoint from disk instead of staging the whole
model in memory. Each layer is opened, used, and released, so the peak resident set stays near
three gigabytes even though the file on disk is well over twenty. Expert weights dominate the
byte budget: for every token the router picks eight of two hundred and fifty-six experts, and only
those rows are read. The remaining sections, the attention projections, the normalisation vectors
and the router matrices, are small enough that they can be reloaded per layer without measurable
cost. Because the linear attention layers carry a fixed recurrent state rather than a growing key
value cache, the memory required for a long prompt does not grow with the prompt. That property
makes the design attractive on a laptop with limited unified memory, but it complicates prompt
caching, because a recurrent state cannot be truncated or rewound after the fact."""

CODE = '''func makeExpertPlan(for layer: Int, tokens: [Int], router: RouterOutput) -> ExpertPlan {
    var union = Set<Int>()
    union.reserveCapacity(tokens.count * topK)
    for token in tokens {
        for slot in 0..<topK {
            union.insert(router.index(token: token, slot: slot))
        }
    }
    let resident = cache.residentExperts(layer: layer)
    let missing = union.subtracting(resident).sorted()
    var tiles: [ExpertTile] = []
    tiles.reserveCapacity(missing.count)
    for expert in missing {
        let offset = layout.expertOffset(layer: layer, expert: expert)
        tiles.append(ExpertTile(expert: expert, offset: offset, length: layout.expertStride))
    }
    return ExpertPlan(layer: layer, union: union.sorted(), resident: resident, fetch: tiles)
}'''

TEXTS = {"ja_prose": JA_PROSE, "ja_tech": JA_TECH, "en_tech": EN_TECH, "code": CODE}

def main() -> int:
    roots = {
        "oq4e-g64": Path.home() / "LLM/Ornith-1.5-35B-A3B-oQ4e-g64",
        "mlx-4bit": Path.home() / "LLM/Ornith-1.5-35B-A3B-MLX-4bit",
    }
    toks = {k: Tokenizer.from_file(str(v / "tokenizer.json")) for k, v in roots.items()}
    out = {}
    ok = True
    for name, text in TEXTS.items():
        ids = {k: t.encode(text, add_special_tokens=False).ids for k, t in toks.items()}
        same = ids["oq4e-g64"] == ids["mlx-4bit"]
        ok &= same
        print(f"{name:9s} {len(ids['oq4e-g64']):5d} tok  両者一致 {same}")
        out[name] = ids["oq4e-g64"]
    Path("scratch/qwen35/nll_texts.json").write_text(json.dumps(out))
    return 0 if ok else 1

sys.exit(main())
