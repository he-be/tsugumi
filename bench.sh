#!/usr/bin/env bash
# TurboFieldfare ベンチ v3 (M3 Pro 18GB / macOS 15.7.5)
#
#   ./bench.sh prompts     プロンプト生成 (m を 526tok 前後に調整済み)
#   ./bench.sh overhead    モデルロード時間の推定 (他の測定の減算基準)
#   ./bench.sh ptime       prefill 壁時計 (--max-new 1)。prefill on/off/chunk を比較
#   ./bench.sh policy      lfu vs lru。--prefill off の12%が説明できるか
#   ./bench.sh rdadvise    off/default/bounded/adaptive
#   ./bench.sh total       prefill on/off のトータル時間 (prefill+decode の損得)
#   ./bench.sh slots       スロット数 16/32/64/96 (Phase 2 の本題)
#   ./bench.sh trace [out] トレースを取ってヒット率カーブを机上で引く
#   ./bench.sh summarize   bench/results.tsv から中央値を出す
#   ./bench.sh one <ラベル> <json> [フラグ...]
#
# v3 (Phase 0 の計装後):
#  - footer に load/prefill/ttft/peak と expert キャッシュのヒット率が出るように
#    なったので、real からの引き算はもう要らない。結果 TSV に直接載せる。
#  - --verification trusted-install で層ファイルの SHA-256 (約 5s) を飛ばせる。
#
# v1 からの変更点:
#  - 熱ドリフト (7回で約4%低下) を観測したので、条件は繰り返しの内側で
#    インターリーブする。ドリフトが全条件に均等に乗るようにする。
#  - 実行間にクールダウンを入れる (COOLDOWN 秒、既定20)
#  - 結果を bench/results.tsv に追記して summarize で中央値を取る
#
# footer の実フォーマット (4 行):
#   [stop=maxTokens prefill=518tok new=128tok decode=5.96s tok/s=21.479]
#   [load=0.678s layerVerify=0.000s/30layers prefill=10.374s ttft=10.374s peak=2.29GB rss=1.71GB]
#   [expert prefill hit=0.2% 13/8612 io=5.472s | decode hit=71.0% 21629/30480 io=2.617s]
#   [decode/tok io=20.55ms cb1=0.67ms cb2=0.31ms head=3.30ms]
#   ※ 1 行目の prefill= はトークン数、2 行目の prefill= が秒数
#
# CLI が受け付ける値:
#   --expert-cache-slots   8|16|24|32|48|64|80|96|112  (既定64)
#                          1 スロット約 100MB。載らない設定は起動時に拒否される
#   --expert-cache-policy  lfu|lru      (既定lfu)
#   --prefill              on|off       (既定on、on は16スロット以上必須)
#   --prefill-chunk-tokens 32|64|128    (既定128、上限128)
#   --rdadvise             off|default|bounded|adaptive (既定off)
#   --verification         full-sha256|trusted-install (既定full-sha256)
#   --dump-expert-trace    <path>  bench/expert_sim.py に食わせる

set -euo pipefail

REPO="${REPO:-$PWD}"
CLI="${CLI:-$REPO/.build/release/TurboFieldfareCLI}"
MODEL="${MODEL:-$REPO/scratch/gemma4.gturbo}"
OUT="$REPO/bench"
LOGS="$OUT/logs"
TSV="$OUT/results.tsv"
RUNS="${RUNS:-3}"
COOLDOWN="${COOLDOWN:-20}"
MAXNEW="${MAXNEW:-128}"

preflight() {
  [[ -x "$CLI" ]] || { echo "CLI がない: $CLI" >&2; exit 1; }
  [[ -d "$MODEL" ]] || { echo "モデルがない: $MODEL" >&2; exit 1; }
  if pgrep -qf TurboFieldfareMac || pgrep -qf TurboFieldfareDecodeService \
     || pgrep -qf TurboFieldfareServer; then
    echo "他の TurboFieldfare プロセスが動いてる。落としてから。" >&2; exit 1
  fi
  mkdir -p "$LOGS"
  [[ -f "$TSV" ]] || printf 'label\tn\tptok\tntok\treal\tdecode\trate\tprefill_s\tttft_s\tio_ms\thit_pct\tpeak_gb\tflags\n' > "$TSV"
}

# $1=ラベル $2=messages.json $3=連番 残り=追加フラグ
run_once() {
  local label="$1" msgs="$2" n="$3"; shift 3
  local log="$LOGS/${label}.${n}.err"
  /usr/bin/time -p "$CLI" --model "$MODEL" --messages-file "$msgs" \
    --temperature 0 --seed 1 --max-new "$MAXNEW" --max-context 4096 "$@" \
    > "$LOGS/${label}.${n}.out" 2> "$log"

  local real decode rate ptok ntok pre ttft io hit peak
  real=$(awk '/^real/{print $2}' "$log")
  decode=$(grep -o 'decode=[0-9.]*' "$log" | head -1 | cut -d= -f2)
  rate=$(grep -o 'tok/s=[0-9.]*'   "$log" | head -1 | cut -d= -f2)
  ptok=$(grep -o 'prefill=[0-9]*tok' "$log" | head -1 | tr -dc '0-9')
  ntok=$(grep -o 'new=[0-9]*'      "$log" | head -1 | cut -d= -f2)
  # Phase 0 の計装から。引き算は要らない。
  pre=$(sed -n 's/.* prefill=\([0-9.]*\)s.*/\1/p'          "$log" | head -1)
  ttft=$(sed -n 's/.*ttft=\([0-9.]*\)s.*/\1/p'             "$log" | head -1)
  io=$(sed -n 's/.*decode\/tok io=\([0-9.]*\)ms.*/\1/p'    "$log" | head -1)
  hit=$(sed -n 's/.*| decode hit=\([0-9.]*\)%.*/\1/p'      "$log" | head -1)
  peak=$(sed -n 's/.*peak=\([0-9.]*\)GB.*/\1/p'            "$log" | head -1)

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$n" "$ptok" "$ntok" "$real" "$decode" "$rate" \
    "${pre:-}" "${ttft:-}" "${io:-}" "${hit:-}" "${peak:-}" "$*" >> "$TSV"
  printf '  %-20s #%s  decode=%ss %s tok/s  prefill=%ss ttft=%ss  io=%sms hit=%s%% peak=%sGB\n' \
    "$label" "$n" "$decode" "$rate" "${pre:-?}" "${ttft:-?}" "${io:-?}" "${hit:-?}" "${peak:-?}"
  sleep "$COOLDOWN"
}

# 条件をインターリーブして回す。specs は "ラベル|フラグ..." の配列。
# 繰り返しが外、条件が内。熱ドリフトが全条件に均等に乗る。
sweep() {
  local name="$1" msgs="$2"; shift 2
  local -a specs=("$@")
  echo "== $name  (RUNS=$RUNS, MAXNEW=$MAXNEW, COOLDOWN=${COOLDOWN}s) =="
  # ウォームアップ 1 回 (記録しない)
  local first="${specs[0]}"
  echo "-- warmup --"
  MAXNEW="$MAXNEW" run_once "${name}-warmup" "$msgs" 0 ${first#*|} >/dev/null 2>&1 || true
  for n in $(seq 1 "$RUNS"); do
    for spec in "${specs[@]}"; do
      run_once "${name}-${spec%%|*}" "$msgs" "$n" ${spec#*|}
    done
  done
  echo
}

cmd_prompts() {
  mkdir -p "$OUT"
  python3 - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
para = ("チャンク化した prefill が time to first token を短くしつつ、"
        "メモリ使用量を上限内に抑えられる理由を、KV キャッシュの確保タイミングと"
        "ルーティングされたエキスパートのフェッチ回数の観点から説明してください。")
tail = "\n\n上記の内容を踏まえて、要点を3つに整理してください。"
# 実測: 12 段落 = 616 tok なので 1 段落 ≒ 51 tok。10 段落で 520 前後。
specs = {
    "s": "こんにちは。",          # ロード時間の推定用。ほぼ prefill なし
    "m": para * 10 + tail,        # 520 tok 前後 (アプリ実測の 526 に合わせる)
    "l": para * 50 + tail,        # 2500 tok 前後
}
for k, v in specs.items():
    (out / f"{k}.json").write_text(
        json.dumps([{"role": "user", "content": v}], ensure_ascii=False))
    print(f"{out}/{k}.json  ({len(v)} 文字)")
PY
  echo "初回実行の prefill=Ntok を見て、m が 526 から外れていたら倍数を微調整。"
}

# ロード時間 = real - decode - (ごく短い prefill)。以降の減算基準。
cmd_overhead() {
  preflight; MAXNEW=1
  sweep overhead "$OUT/s.json" "load|--expert-cache-slots 32 --prefill off --rdadvise off"
  echo "この real-decode がモデルロード時間の上限見積り。以降ここから引く。"
}

# 本題。--max-new 1 で decode を潰し、real-decode をほぼ prefill にする。
cmd_ptime() {
  preflight; MAXNEW=1
  sweep ptime "$OUT/m.json" \
    "on128|--expert-cache-slots 32 --prefill on  --prefill-chunk-tokens 128 --rdadvise off" \
    "on64|--expert-cache-slots 32 --prefill on  --prefill-chunk-tokens 64  --rdadvise off" \
    "on32|--expert-cache-slots 32 --prefill on  --prefill-chunk-tokens 32  --rdadvise off" \
    "off|--expert-cache-slots 32 --prefill off --rdadvise off"
  cat <<'MSG'
real-decode-ロード時間 が prefill の実時間。アプリの 10.85s と突き合わせる。
  on128 が off より大幅に速い -> チャンク化は機能している
  ほぼ同じ / 逆転             -> チャンク化が何も買えていない
  128 > 64 > 32 と単調        -> 償却が効いており、上限128が天井
MSG
}

# --prefill off の decode +12% が LFU 汚染で説明できるかの検証。
cmd_policy() {
  preflight; MAXNEW=128
  sweep policy "$OUT/m.json" \
    "on-lfu|--expert-cache-slots 32 --expert-cache-policy lfu --prefill on --prefill-chunk-tokens 128 --rdadvise off" \
    "on-lru|--expert-cache-slots 32 --expert-cache-policy lru --prefill on --prefill-chunk-tokens 128 --rdadvise off" \
    "off-lfu|--expert-cache-slots 32 --expert-cache-policy lfu --prefill off --rdadvise off" \
    "off-lru|--expert-cache-slots 32 --expert-cache-policy lru --prefill off --rdadvise off"
  cat <<'MSG'
on-lru が on-lfu より速く off に近づくなら、チャンク化 prefill が LFU カウンタを
汚染して decode のヒット率を下げている、という説明が成立する。
その場合は upstream に投げる価値がある (ハード買い替えとは無関係に効く)。
on-lru でも改善しないなら別要因。
MSG
}

cmd_rdadvise() {
  preflight; MAXNEW=128
  sweep rdadvise "$OUT/m.json" \
    "off|--expert-cache-slots 32 --prefill on --prefill-chunk-tokens 128 --rdadvise off" \
    "default|--expert-cache-slots 32 --prefill on --prefill-chunk-tokens 128 --rdadvise default" \
    "bounded|--expert-cache-slots 32 --prefill on --prefill-chunk-tokens 128 --rdadvise bounded" \
    "adaptive|--expert-cache-slots 32 --prefill on --prefill-chunk-tokens 128 --rdadvise adaptive"
  echo "prefill と decode で最適値が割れる可能性がある。ptime 側でも同じ比較をすること。"
}

# prefill off は decode を速くするが prefill を遅くしているかもしれない。
# トータル (real) で損得を判定する。
cmd_total() {
  preflight; MAXNEW=128
  sweep total "$OUT/m.json" \
    "on|--expert-cache-slots 32 --prefill on --prefill-chunk-tokens 128 --rdadvise off" \
    "off|--expert-cache-slots 32 --prefill off --rdadvise off"
  echo "real の中央値で比較する。decode の12%はトータルで回収できているか。"
  echo "長プロンプト (l.json) でも同じ結論になるか要確認。prefill 側の比重が変わる。"
}

# Phase 2 の本題。スロット数を振る。--verification trusted-install で
# 層 SHA-256 の約 5s を落として、残りの差だけを見る。
cmd_slots() {
  preflight; MAXNEW="${MAXNEW:-128}"
  local common="--prefill on --prefill-chunk-tokens 128 --rdadvise off --verification trusted-install"
  sweep slots "$OUT/haiku.json" \
    "s16|--expert-cache-slots 16 $common" \
    "s32|--expert-cache-slots 32 $common" \
    "s64|--expert-cache-slots 64 $common" \
    "s96|--expert-cache-slots 96 $common"
  cat <<'MSG'
peak_gb が 12GB を超えていないこと、rate が単調に上がることの 2 つが合格条件。
rate が 64 で頭打ちなら I/O ではなく GPU 律速に移っている。その先は
スロットを増やしてもメモリを食うだけなので、そこで止める。
MSG
}

# 1 回のトレースからスロット数 vs ヒット率のカーブを机上で引く。
cmd_trace() {
  preflight
  local out="${1:-$OUT/trace.tsv}"
  "$CLI" --model "$MODEL" --messages-file "$OUT/haiku.json" \
    --temperature 0 --seed 1 --max-new "${MAXNEW:-128}" --max-context 4096 \
    --verification trusted-install --dump-expert-trace "$out" >/dev/null
  echo "trace: $out"
  ./bench/expert_sim.py "$out" --skew
}

cmd_one() {
  preflight
  local label="$1" msgs="$2"; shift 2
  run_once "$label" "$msgs" 1 "$@"
}

cmd_summarize() {
  python3 - "$TSV" <<'PY'
import sys, csv, statistics as st
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
g = defaultdict(list)
for r in rows:
    if r['label'].endswith('-warmup'):
        continue
    g[r['label']].append(r)
def med(v):
    try: return st.median([float(x) for x in v if x not in ('', None)])
    except Exception: return float('nan')
w = max((len(k) for k in g), default=10)
print(f"{'label':<{w}}  n  ptok    tok/s  decode  prefill    ttft   io_ms   hit%  peak_gb")
for k in sorted(g):
    v = g[k]
    def m(field): return med([x.get(field) for x in v])
    print(f"{k:<{w}} {len(v):>2}  {v[0]['ptok']:>4}  "
          f"{m('rate'):>7.2f} {m('decode'):>7.2f} {m('prefill_s'):>8.2f} "
          f"{m('ttft_s'):>7.2f} {m('io_ms'):>7.2f} {m('hit_pct'):>6.1f} {m('peak_gb'):>8.2f}")
PY
}

case "${1:-}" in
  prompts)   cmd_prompts ;;
  overhead)  cmd_overhead ;;
  ptime)     cmd_ptime ;;
  policy)    cmd_policy ;;
  rdadvise)  cmd_rdadvise ;;
  total)     cmd_total ;;
  slots)     cmd_slots ;;
  trace)     shift; cmd_trace "$@" ;;
  summarize) cmd_summarize ;;
  one)       shift; cmd_one "$@" ;;
  *)         sed -n '2,15p' "$0"; exit 1 ;;
esac