import html, json, statistics as st
ARMS = [("full", "フル（128 個）", True), ("k112", "112 個", False), ("k96", "96 個", True), ("k80", "80 個", True), ("k64", "64 個", False), ("iq3xxs", "IQ3_XXS", True)]
KIND = {"read": "読み", "item": "英名", "pick": "選択", "deed": "動作"}
e = html.escape
photos = {}
for l in open("kj_payloads96.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["k"] == 0:
        for m in r["payload"]["messages"]:
            if isinstance(m["content"], list):
                for x in m["content"]:
                    if x["type"] == "image_url": photos[r["play"]] = x["image_url"]["url"]
ans = {a: {} for a, _, _ in ARMS}
for a, _, _ in ARMS:
    for l in open(f"ans_{a}_sample.jsonl", encoding="utf-8"):
        r = json.loads(l); ans[a][(r["play"], r["name"])] = r
plays = list(dict.fromkeys(p for p, _ in ans["full"]))

def cell(r):
    if not r["valid"]:
        t = r["text"]
        return '<span class="bad">崩れ</span> <span class="raw">%s … （%d トークンで打ち切り）</span>' % (e(t[:220]), r["predicted_n"])
    d = json.loads(r["text"]); k = r["name"]
    if k == "read":
        return ('<div class="main">%s</div><div>%s</div><div class="sub">性別：%s</div><div class="sub">カテゴリ：%s</div><div class="sub">連想：%s</div>'
                % (e(str(d.get("正体", ""))), e(str(d.get("見立て", ""))), e(str(d.get("性別", ""))), e("・".join(d.get("カテゴリ", []))), e("・".join(d.get("連想する語", [])))))
    if k == "item": return '<div class="en">%s</div>' % e(str(d.get("英名", "")))
    if k == "pick": return '<div class="main">%s</div><div>%s</div>' % (e(str(d.get("選んだ一体", ""))), e(str(d.get("選んだわけ", ""))))
    return '<div class="en">%s</div>' % e(str(d.get("英文", "")))

cards = []
for i, p in enumerate(plays, 1):
    bad = any(not ans[a][(p, k)]["valid"] for a, _, _ in ARMS for k in KIND)
    rows = []
    for k, kj in KIND.items():
        tds = "".join('<td class="c-%s">%s</td>' % (a, cell(ans[a][(p, k)])) for a, _, _ in ARMS)
        rows.append('<tr><th scope="row">%s</th>%s</tr>' % (kj, tds))
    head = "".join('<th class="c-%s" scope="col">%s</th>' % (a, e(n)) for a, n, _ in ARMS)
    cards.append('<section class="card%s"><div class="photo"><img loading="lazy" src="%s" alt="鑑定品の写真 %d"><div class="pid">%d / %d<br>%s</div></div>'
                 '<div class="tw"><table><thead><tr><th></th>%s</tr></thead><tbody>%s</tbody></table></div></section>'
                 % (" hasbad" if bad else "", photos[p], i, i, len(plays), e(p), head, "".join(rows)))

# 英名 480 回の崩れ
item_rows, examples = [], []
for a, n, _ in ARMS:
    R = [json.loads(l) for l in open(f"item_{a}.jsonl", encoding="utf-8")]
    b = [r for r in R if not r["valid"]]
    item_rows.append((n, len(R), len(b), len({r["play"] for r in b})))
    for r in b[:2]: examples.append((n, r["text"][:260]))

toggles = "".join('<label><input type="checkbox" data-arm="%s"%s> %s</label>' % (a, " checked" if on else "", e(n)) for a, n, on in ARMS)
hide_css = "".join("body.hide-%s .c-%s{display:none}" % (a, a) for a, _, _ in ARMS)
init_hide = " ".join("hide-%s" % a for a, _, on in ARMS if not on)

T = lambda hdr, rows: '<div class="tw"><table class="num"><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (
    "".join("<th>%s</th>" % h for h in hdr), "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % c for c in r) for r in rows))
summary = T(["", "IQ3_XXS", "Q4_0・96 個", "Q4_0・80 個", "Q4_0・128 個（フル）"], [
    ["ファイル GiB", "10.63", "10.26", "8.77", "13.26"],
    ["M6：wired 最大 GiB", "13.4〜13.5", "12.70", "<b>11.23</b>", "載らない"],
    ["M6：Swapouts（2 周目、ページ）", "約 2〜4 万", "0", "<b>0</b>", "—"],
    ["M6：圧縮メモリ最大 GiB", "1.2", "1.55", "<b>0.06</b>", "—"],
    ["M6：鑑定 1 回の秒", "8.7〜8.9", "7.67", "<b>7.54</b>", "—"],
    ["M6：読みの最初の文字まで 秒", "0.87〜0.91", "0.79", "0.77", "—"],
    ["3090：英名の崩れ（480 回中）", "51（10.6%）", "11（2.3%）", "17（3.5%）", "3（0.6%）"],
    ["3090：フルの答えとの類似度<br>（読み / 英名 / 選択 / 動作）", "0.82 / 0.76 / 0.72 / 0.88", "0.90 / 0.90 / 0.82 / 0.91", "0.86 / 0.86 / 0.75 / 0.89", "1.00"],
])
cover = T(["残す個数 / 層", "ファイル GiB", "選択のうち残る割合", "最悪の層"], [
    ["128（現状）", "13.26", "100%", "—"], ["112", "11.75", "99.8%", "97.7%"], ["96", "10.26", "99.0%", "93.6%"],
    ["80", "8.77", "97.3%", "88.1%"], ["64", "7.27", "94.2%", "82.3%"], ["96（無作為に選んだ場合）", "10.26", "75.8%", "—"]])
items = T(["", "回数", "崩れ", "率", "崩れが出た写真の数"], [[e(n), t, b, "%.1f%%" % (100 * b / t), pl] for n, t, b, pl in item_rows])
ex = "".join('<li><b>%s</b>：<span class="raw">%s …</span></li>' % (e(n), e(t)) for n, t in examples)

page = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>鑑定機のダイエット検証 — 答えの読み比べ</title>
<style>
:root{{--bg:#f6f5f1;--card:#fff;--ink:#1d1c1a;--mut:#6b675f;--line:#dedbd3;--acc:#1f5f8b;--bad:#b3261e;--badbg:#fbe9e7;--head:#efede7}}
@media (prefers-color-scheme: dark){{:root{{--bg:#161614;--card:#1f1f1c;--ink:#ecebe6;--mut:#a09c92;--line:#35342f;--acc:#7fb8e0;--bad:#ff8a80;--badbg:#3a1f1c;--head:#272723}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 -apple-system,"Hiragino Sans","Hiragino Kaku Gothic ProN","Noto Sans JP",sans-serif}}
main{{max-width:1500px;margin:0 auto;padding:24px 16px 80px}}
h1{{font-size:24px;line-height:1.4;margin:0 0 4px}} h2{{font-size:18px;margin:36px 0 8px}} .lead{{color:var(--mut);margin:0 0 16px}}
.prose{{max-width:820px}} .prose li{{margin:4px 0}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid var(--line);padding:8px 10px;vertical-align:top;text-align:left}}
thead th{{background:var(--head);font-weight:600;white-space:nowrap}}
table.num{{width:auto;min-width:min(100%,640px);background:var(--card)}} table.num td{{font-variant-numeric:tabular-nums}}
.bar{{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 0;display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center}}
.bar label{{white-space:nowrap}} .bar .sep{{color:var(--mut)}}
.card{{display:grid;grid-template-columns:220px 1fr;gap:14px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:16px 0}}
.photo img{{width:100%;border-radius:6px;display:block}} .pid{{color:var(--mut);font-size:12px;margin-top:6px;font-variant-numeric:tabular-nums}}
.card td{{min-width:220px;font-size:14px;line-height:1.6}} .card tbody th{{white-space:nowrap;background:var(--head);width:1%}}
.main{{font-weight:700;font-size:15px}} .sub{{color:var(--mut);font-size:12.5px}} .en{{font-family:ui-serif,Georgia,serif}}
.bad{{display:inline-block;background:var(--badbg);color:var(--bad);font-weight:700;border-radius:4px;padding:0 6px;font-size:12px}}
.raw{{color:var(--mut);font-size:12.5px;overflow-wrap:anywhere}}
td:has(.bad){{background:var(--badbg)}}
body.onlybad .card:not(.hasbad){{display:none}}
{hide_css}
@media (max-width:760px){{.card{{grid-template-columns:1fr}} .photo img{{max-width:260px}}}}
</style></head>
<body class="{init_hide}"><main>
<h1>鑑定機のダイエット検証 — 答えの読み比べ</h1>
<p class="lead">2026-11-14〜15 測定。Gemma 4 26B-A4B（Huihui lattice Q4_0）から、鑑定で使うエキスパートだけを残した版の報告です。この HTML は来場者の写真を含みます。</p>

<div class="prose">
<h2>結論</h2>
<p><b>鑑定専用にエキスパートを刈るダイエットは成立します。</b>Q4_0 のまま層ごとに 80 個へ切った版は、M6（16 GB、ヘッドレス）でスワップなし・鑑定 1 回 7.5 秒で動きました。同じ場所に載る IQ3_XXS より速く、英名の崩れも少ないです。日本語の読みの質は数字では決められないので、下の読み比べで確かめてください。</p>
</div>
{summary}
<div class="prose">
<ul>
<li>M6 でスワップが止まる線は wired 12.7 GiB 付近です。96 個版はスワップ 0 でも圧縮メモリが 1.55 GiB あり、縁に立っています。</li>
<li>文脈長は効きません。IQ3_XXS で 8192 → 2048 にしても wired は 0.1 GiB しか動きませんでした。最低限は 2048 で足ります（最悪で 1,986 トークン）。</li>
<li>参考：Tsugumi（QAT）は M6 で鑑定 1 回 9.73 秒、SSD を毎回 11.7 GB 読んでいました。</li>
</ul>

<h2>鑑定ではエキスパートの使い方が偏っている</h2>
<p>96 枚 384 要求（約 20 万トークン）のルーターの選択を記録し、48 枚で残すエキスパートを選び、残りの 48 枚で確かめました。</p>
</div>
{cover}

<div class="prose">
<h2>英名の崩れ（96 枚 × 5 回）</h2>
<p>英名の問いだけを通常のサンプリングで 480 回ずつ引きました。崩れはどの版でも同じ型で、繰り返しに落ちて上限の 120 トークンまで書き、JSON が閉じません。フルモデルでも 0.6% 出ます。</p>
</div>
{items}
<div class="prose"><p>崩れた答えの例：</p><ul>{ex}</ul>

<h2>留保</h2>
<ul>
<li>類似度は「フルの答えにどれだけ近いか」で、良し悪しそのものではありません。</li>
<li>IQ3_XXS は unsloth 版、フルと切り出し版は Huihui lattice 版で、土台の違いが比較に混ざっています。</li>
<li>64 個（2.1%）が 80 個（3.5%）より崩れが少なく出ており、480 回でもこの程度の差は誤差の範囲です。</li>
<li>M6 の結果は 12 枚 × 2 周です。長時間運転は見ていません。</li>
</ul>

<h2>読み比べ（確かめる用の 48 枚）</h2>
<p>残すエキスパートを選ぶのに使っていない 48 枚です。各版が通常のサンプリング（temperature 1.0 / top-k 64）で 1 回ずつ出した答えを並べています。サンプリングなので、同じ版でも引くたびに文は変わります。</p>
<p>「選択」と「動作」の問いには、記録したときのフルモデルの読みが前提として入っています。各列の読みを受けた答えではありません。</p>
</div>
<div class="bar"><span>表示する列：</span>{toggles}<span class="sep">｜</span><label><input type="checkbox" id="onlybad"> 崩れのある写真だけ</label></div>
{"".join(cards)}
</main>
<script>
document.querySelectorAll('input[data-arm]').forEach(function(c){{c.addEventListener('change',function(){{document.body.classList.toggle('hide-'+c.dataset.arm,!c.checked)}})}});
document.getElementById('onlybad').addEventListener('change',function(ev){{document.body.classList.toggle('onlybad',ev.target.checked)}});
</script>
</body></html>"""
import os
out = "/Users/mh/LLM/kajimachine-diet-report"; os.makedirs(out, exist_ok=True)
open(out + "/index.html", "w", encoding="utf-8").write(page)
print(len(page) // 1024, "KB", len(plays), "plays", item_rows, sum(1 for c in cards if "hasbad" in c))
