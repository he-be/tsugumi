#!/usr/bin/env bash
# Ornith (qwen3_5_moe) ベンチ — docs/qwen35moe/04-PHASES.md Phase 6
#
#   ./bench/qwen35.sh base      運用点 (32/lfu/2048) を RUNS 回 + 熱ドリフト検定
#   ./bench/qwen35.sh slots     16/24/32 スロット (インターリーブ。8 は chunked prefill が要る 16 未満)
#   ./bench/qwen35.sh chunk     prefill チャンク 512/1024/2048
#   ./bench/qwen35.sh policy    lfu vs lru
#   ./bench/qwen35.sh rdadvise  off/default/bounded/adaptive
#   ./bench/qwen35.sh pipeline  直列 (TF_QWEN_PIPELINE=0) vs 重ね (既定)
#   ./bench/qwen35.sh prefetch  層をまたぐ先読み off vs top-8
#   ./bench/qwen35.sh arm       mmap (既定) vs 私有スロット (TF_EXPERT_MMAP=0)
#   ./bench/qwen35.sh tasks     実タスク 4 本 (bench/qwen35/t*.json) を運用点で
#   ./bench/qwen35.sh tasksab   その 4 本を直列 vs 重ねで交互に
#   ./bench/qwen35.sh prompts   プロンプト長 s/m/l を運用点で
#   ./bench/qwen35.sh trace [out] トレース 1 本 + 机上のヒット率カーブ
#   ./bench/qwen35.sh summarize bench/qwen35-results.tsv の中央値
#
# host は残差 (wall - gpu - io) なので、読みと GPU が重なっていると負になる。
# それは異常ではなく、重なっている証拠である。
#
# bench.sh との違いは 3 つだけで、どれもこの家族の性質から来る:
#  1. **サンプリングは無い。**融合ヘッドは logit を書かないので Ornith 経路は
#     貪欲だけ (docs/qwen35moe/25-CLI-TOOLS.md §1)。--temperature 0 /
#     --repetition-penalty 1 が必須で、同じプロンプトなら出力もトークン数も
#     run 間で完全に同じになる。振れるのは時間だけである。
#  2. **クールダウンは 4 秒** (2026-08-22 ユーザー指定、この Phase 限定)。
#     0 秒だと GPU クロックが落ちて時間が 2.6 倍になる条件が実在する
#     (docs/qwen35moe/24-PREFILL-MOE-PATH.md §4-2)。Gemma 側の 20 秒とは別値。
#  3. **熱ドリフトの判定はこちらが持つ。**間隔を詰めたぶん、base を先頭と末尾で
#     2 回測り、|head/tail| が 5% を超えた run は捨てる (Phase 6 の作法)。
set -euo pipefail

REPO="${REPO:-$PWD}"
CLI="${CLI:-$REPO/.build/release/TurboFieldfareCLI}"
MODEL="${MODEL:-$REPO/scratch/ornith-oq4e-g64.gturbo}"
OUT="$REPO/bench"
LOGS="$OUT/logs/qwen35"
TSV="$OUT/qwen35-results.tsv"
RUNS="${RUNS:-3}"
COOLDOWN="${COOLDOWN:-4}"
MAXNEW="${MAXNEW:-128}"
MSGS="${MSGS:-$OUT/m.json}"
COMMON="--temperature 0 --repetition-penalty 1 --thinking off --max-context 4096 --verification trusted-install"

preflight() {
  [[ -x "$CLI" ]] || { echo "CLI がない: $CLI" >&2; exit 1; }
  [[ -d "$MODEL" ]] || { echo "モデルがない: $MODEL" >&2; exit 1; }
  if pgrep -qf TurboFieldfareMac || pgrep -qf TurboFieldfareDecodeService \
     || pgrep -qf TurboFieldfareServer; then
    echo "他の TurboFieldfare プロセスが動いてる。落としてから。" >&2; exit 1
  fi
  mkdir -p "$LOGS"
  [[ -f "$TSV" ]] || printf 'label\tn\tptok\tntok\treal\tdecode\trate\tprefill_s\tttft_s\thit_pct\tpeak_gb\tp_gpu\tp_io\tp_host\td_gpu\td_io\td_host\td_cb\tflags\n' > "$TSV"
}

# $1=ラベル $2=messages.json $3=連番 残り=追加フラグ
run_once() {
  local label="$1" msgs="$2" n="$3"; shift 3
  local log="$LOGS/${label}.${n}.err"
  # `env:KEY=VALUE` in a spec's flags sets one variable for that arm only —
  # the streaming arm is chosen by the environment, not by a flag, and an arm
  # that cannot be interleaved with the others cannot be compared with them.
  local -a envv=() flags=()
  local token
  for token in "$@"; do
    case "$token" in env:*) envv+=("${token#env:}") ;; *) flags+=("$token") ;; esac
  done
  /usr/bin/time -p env ${envv[@]+"${envv[@]}"} "$CLI" --model "$MODEL" --messages-file "$msgs" \
    $COMMON --max-new "$MAXNEW" ${flags[@]+"${flags[@]}"} \
    > "$LOGS/${label}.${n}.out" 2> "$log"

  local real decode rate ptok ntok pre ttft hit peak stop
  local pgpu pio phost dgpu dio dhost dcb
  real=$(awk '/^real/{print $2}' "$log")
  decode=$(grep -o 'decode=[0-9.]*s' "$log" | head -1 | tr -dc '0-9.')
  rate=$(grep -o 'tok/s=[0-9.]*'   "$log" | head -1 | cut -d= -f2)
  ptok=$(grep -o 'prefill=[0-9]*tok' "$log" | head -1 | tr -dc '0-9')
  ntok=$(grep -o 'new=[0-9]*tok'   "$log" | head -1 | tr -dc '0-9')
  pre=$(sed -n 's/.*\[load=[0-9.]*s prefill=\([0-9.]*\)s.*/\1/p'  "$log" | head -1)
  ttft=$(sed -n 's/.*ttft=\([0-9.]*\)s.*/\1/p'                    "$log" | head -1)
  pgpu=$(sed -n 's/.*\[prefill\/tok gpu=\([0-9.]*\)ms.*/\1/p'    "$log" | head -1)
  pio=$(sed -n 's/.*\[prefill\/tok gpu=[0-9.]*ms io=\([0-9.]*\)ms.*/\1/p' "$log" | head -1)
  phost=$(sed -n 's/.*\[prefill\/tok .* host=\(-\{0,1\}[0-9.]*\)ms.*/\1/p' "$log" | head -1)
  dgpu=$(sed -n 's/.*\[decode\/tok gpu=\([0-9.]*\)ms.*/\1/p'      "$log" | head -1)
  dio=$(sed -n 's/.*\[decode\/tok gpu=[0-9.]*ms io=\([0-9.]*\)ms.*/\1/p' "$log" | head -1)
  dhost=$(sed -n 's/.*\[decode\/tok .* host=\(-\{0,1\}[0-9.]*\)ms.*/\1/p' "$log" | head -1)
  dcb=$(sed -n 's/.*\[decode\/tok .* cb=\([0-9.]*\)\].*/\1/p'    "$log" | head -1)
  hit=$(sed -n 's/.*| decode hit=\([0-9.]*\)%.*/\1/p'             "$log" | head -1)
  peak=$(sed -n 's/.*peak=\([0-9.]*\)GB.*/\1/p'                   "$log" | head -1)
  stop=$(sed -n 's/.*\[stop=\([a-zA-Z]*\) .*/\1/p'                "$log" | head -1)

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$n" "$ptok" "$ntok" "$real" "$decode" "$rate" \
    "${pre:-}" "${ttft:-}" "${hit:-}" "${peak:-}" \
    "${pgpu:-}" "${pio:-}" "${phost:-}" "${dgpu:-}" "${dio:-}" "${dhost:-}" "${dcb:-}" \
    "$*" >> "$TSV"
  printf '  %-18s #%s  %s tok/s  prefill=%ss (gpu=%s io=%s host=%s)  decode/tok gpu=%s io=%s host=%s cb=%s  hit=%s%% peak=%sGB\n' \
    "$label" "$n" "${rate:-?}" "${pre:-?}" \
    "${pgpu:-?}" "${pio:-?}" "${phost:-?}" \
    "${dgpu:-?}" "${dio:-?}" "${dhost:-?}" "${dcb:-?}" "${hit:-?}" "${peak:-?}"
  sleep "$COOLDOWN"
}

# 繰り返しが外、条件が内。ドリフトが全条件に均等に乗る。
sweep() {
  local name="$1" msgs="$2"; shift 2
  local -a specs=("$@")
  echo "== $name  (RUNS=$RUNS, MAXNEW=$MAXNEW, COOLDOWN=${COOLDOWN}s, msgs=$(basename "$msgs")) =="
  echo "-- warmup --"
  run_once "${name}-warmup" "$msgs" 0 ${specs[0]#*|} >/dev/null 2>&1 || true
  run_once "${name}-drift" "$msgs" 0 ${specs[0]#*|}
  for n in $(seq 1 "$RUNS"); do
    for spec in "${specs[@]}"; do
      run_once "${name}-${spec%%|*}" "$msgs" "$n" ${spec#*|}
    done
  done
  run_once "${name}-drift" "$msgs" 9 ${specs[0]#*|}
  drift "${name}-drift"
  echo
}

# 先頭と末尾の base を比べる。5% を超えたらこの sweep の行は捨てる。
drift() {
  python3 - "$TSV" "$1" <<'PY'
import sys, csv
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter='\t')
        if r['label'] == sys.argv[2]]
if len(rows) < 2:
    sys.exit(0)
head, tail = float(rows[-2]['rate']), float(rows[-1]['rate'])
delta = abs(head - tail) / head * 100
verdict = "OK" if delta <= 5 else "捨てる (熱ドリフト)"
print(f"  drift: head={head:.2f} tail={tail:.2f} tok/s  |Δ|={delta:.1f}%  {verdict}")
PY
}

BASE="--expert-cache-slots 32 --expert-cache-policy lfu --prefill on --prefill-chunk-tokens 2048 --rdadvise off"

cmd_base()   { preflight; sweep base   "$MSGS" "op|$BASE"; }
cmd_slots()  { preflight; sweep slots  "$MSGS" \
                 "s16|--expert-cache-slots 16 --expert-cache-policy lfu --prefill on --prefill-chunk-tokens 2048 --rdadvise off" \
                 "s24|--expert-cache-slots 24 --expert-cache-policy lfu --prefill on --prefill-chunk-tokens 2048 --rdadvise off" \
                 "s32|$BASE"; }
cmd_chunk()  { preflight; sweep chunk  "$MSGS" \
                 "c512|--expert-cache-slots 32 --expert-cache-policy lfu --prefill on --prefill-chunk-tokens 512 --rdadvise off" \
                 "c1024|--expert-cache-slots 32 --expert-cache-policy lfu --prefill on --prefill-chunk-tokens 1024 --rdadvise off" \
                 "c2048|$BASE"; }
cmd_rdadvise() { preflight; sweep rdadvise "$MSGS" \
                 "off|$BASE" \
                 "default|--expert-cache-slots 32 --expert-cache-policy lfu --prefill on --prefill-chunk-tokens 2048 --rdadvise default" \
                 "bounded|--expert-cache-slots 32 --expert-cache-policy lfu --prefill on --prefill-chunk-tokens 2048 --rdadvise bounded" \
                 "adaptive|--expert-cache-slots 32 --expert-cache-policy lfu --prefill on --prefill-chunk-tokens 2048 --rdadvise adaptive"; }

# 既定は mmap + residency set。私有スロットの腕は環境変数でしか選べない。
cmd_arm() { preflight; sweep arm "$MSGS" \
                 "mmap|$BASE" \
                 "pread|env:TF_EXPERT_MMAP=0 $BASE"; }

# Phase 6 の本題。読みと GPU を重ねる前後を 1 プロセスずつ交互に回す。
cmd_pipeline() { preflight; sweep pipeline "$MSGS" \
                 "serial|env:TF_QWEN_PIPELINE=0 $BASE" \
                 "overlap|$BASE"; }

# 層をまたぐ先読み (TF_QWEN_EXPERT_PREFETCH=N)。予測は層 L の normed に
# 層 L+1 の router を当てたもの。
cmd_prefetch() { preflight; sweep prefetch "$MSGS" \
                 "off|$BASE" \
                 "n8|env:TF_QWEN_EXPERT_PREFETCH=8 $BASE"; }

cmd_policy() { preflight; sweep policy "$MSGS" \
                 "lfu|$BASE" \
                 "lru|--expert-cache-slots 32 --expert-cache-policy lru --prefill on --prefill-chunk-tokens 2048 --rdadvise off"; }

cmd_prompts() {
  preflight
  echo "== prompts  (RUNS=$RUNS, MAXNEW=$MAXNEW, COOLDOWN=${COOLDOWN}s) =="
  run_once "prompts-warmup" "$OUT/m.json" 0 $BASE >/dev/null 2>&1 || true
  for n in $(seq 1 "$RUNS"); do
    for p in s m l; do run_once "prompts-$p" "$OUT/$p.json" "$n" $BASE; done
  done
  echo
}

# 合成の反復ではなく、実際に人が投げる形の 4 本。t4 だけがこの機械の文書を
# 読ませる長いプロンプトで、prefill 側の代表になる。
cmd_tasks() {
  preflight
  echo "== tasks  (RUNS=$RUNS, MAXNEW=$MAXNEW, COOLDOWN=${COOLDOWN}s) =="
  run_once "tasks-warmup" "$OUT/qwen35/t1-ja-explain.json" 0 $BASE >/dev/null 2>&1 || true
  for n in $(seq 1 "$RUNS"); do
    for t in t1-ja-explain t2-code t3-en-prose t4-summarize; do
      run_once "tasks-$t" "$OUT/qwen35/$t.json" "$n" $BASE
    done
  done
  echo
}

# 実タスク 4 本を、直列と重ねの 2 本立てで交互に。Phase 6 の結論の表になる。
cmd_tasksab() {
  preflight
  echo "== tasksab  (RUNS=$RUNS, MAXNEW=$MAXNEW, COOLDOWN=${COOLDOWN}s) =="
  run_once "tasksab-warmup" "$OUT/qwen35/t1-ja-explain.json" 0 $BASE >/dev/null 2>&1 || true
  for n in $(seq 1 "$RUNS"); do
    for t in t1-ja-explain t2-code t3-en-prose t4-summarize; do
      run_once "tasksab-$t-serial"  "$OUT/qwen35/$t.json" "$n" env:TF_QWEN_PIPELINE=0 $BASE
      run_once "tasksab-$t-overlap" "$OUT/qwen35/$t.json" "$n" $BASE
    done
  done
  echo
}

cmd_trace() {
  preflight
  local out="${1:-$OUT/logs/qwen35/trace.tsv}"
  "$CLI" --model "$MODEL" --messages-file "$MSGS" $COMMON \
    --max-new "$MAXNEW" $BASE --dump-expert-trace "$out" >/dev/null
  echo "trace: $out"
  python3 "$OUT/expert_sim.py" "$out" --slots 16,24,32,48,64 --skew
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
print(f"{'label':<{w}}  n  ptok  ntok    tok/s  prefill  p_gpu   p_io  p_host   d_gpu    d_io  d_host   hit%  peak_gb")
for k in sorted(g):
    v = g[k]
    def m(f): return med([x.get(f) for x in v])
    print(f"{k:<{w}} {len(v):>2}  {v[0]['ptok']:>4}  {v[0]['ntok']:>4}  "
          f"{m('rate'):>7.2f} {m('prefill_s'):>8.2f} "
          f"{m('p_gpu'):>6.2f} {m('p_io'):>6.2f} {m('p_host'):>7.2f} "
          f"{m('d_gpu'):>7.2f} {m('d_io'):>7.2f} {m('d_host'):>7.2f} "
          f"{m('hit_pct'):>6.1f} {m('peak_gb'):>8.2f}")
PY
}

case "${1:-}" in
  base)      cmd_base ;;
  slots)     cmd_slots ;;
  chunk)     cmd_chunk ;;
  policy)    cmd_policy ;;
  rdadvise)  cmd_rdadvise ;;
  arm)       cmd_arm ;;
  pipeline)  cmd_pipeline ;;
  prefetch)  cmd_prefetch ;;
  prompts)   cmd_prompts ;;
  tasks)     cmd_tasks ;;
  tasksab)   cmd_tasksab ;;
  trace)     shift; cmd_trace "$@" ;;
  summarize) cmd_summarize ;;
  *)         sed -n '2,10p' "$0"; exit 1 ;;
esac
