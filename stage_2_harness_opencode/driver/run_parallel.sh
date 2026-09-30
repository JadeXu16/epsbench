#!/bin/bash
# Run the OpenCode harness on a targets file with N parallel workers.
# usage: run_parallel.sh <targets.csv> <slices> <workers> <run-name> <batch> [variant] [provider] [model] [concurrency]
set -u
cd "$(dirname "$0")/.."
EXP="../stage_4_experiments/opencode"

TARGETS="${1:?targets csv}"
SLICES="${2:?slices e.g. 30,7}"
NWORKERS="${3:-4}"
RUN="${4:?run-name e.g. my_run}"
BATCH="${5:?batch e.g. gpt-5.5_s1}"
VARIANT="${6:-v4}"
PROVIDER="${7:-azure}"
MODEL="${8:-gpt-5.4}"
CONCURRENCY="${9:-1}"

if [ "$PROVIDER" = "azure" ] && [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  echo "AZURE_OPENAI_API_KEY not set" >&2; exit 1
fi

_self_dir="$(cd "$(dirname "$0")/.." && pwd -P)"
[ "${FINBENCH_ALLOW_PARALLEL_ARMS:-0}" = 1 ] && _self_dir="__parallel_arms_allowed__"
for _pid in $(pgrep -f "tsx driver/run.ts" 2>/dev/null); do
  [ "$_pid" = "$$" ] && continue
  _cwd="$(readlink -f "/proc/$_pid/cwd" 2>/dev/null)" || continue
  if [ "$_cwd" = "$_self_dir" ]; then
    echo "ERROR: run.ts workers already running in $_self_dir (pid $_pid)." >&2
    echo "       Another harness on this node is fine and is ignored." >&2
    echo "       To clear a genuine leftover: kill $_pid" >&2
    exit 1
  fi
done

LOGDIR="${EXP}/logs/${BATCH}"
mkdir -p "$LOGDIR"
if [ -e "${EXP}/STOP_ALL" ] || [ -e "${EXP}/STOP_${BATCH}" ]; then
  echo "STOP file present (${EXP}/STOP_ALL or STOP_${BATCH}) — not launching workers for $BATCH" >&2
  exit 9
fi

EMB_PORT="${EMBEDDING_PORT:-8900}"
if [ "${NEWS_ABLATION:-}" = "1" ]; then
  export EMBEDDING_SERVER_URL=""
  echo "NEWS_ABLATION=1 -> no news tools, skipping embedding server (no GPU needed)"
elif curl -sf "http://127.0.0.1:${EMB_PORT}/health" > /dev/null 2>&1; then
  export EMBEDDING_SERVER_URL="http://127.0.0.1:${EMB_PORT}"
  echo "embedding server already running at $EMBEDDING_SERVER_URL"
else
  export EMBEDDING_SERVER_URL="http://127.0.0.1:${EMB_PORT}"
  echo "starting embedding server on port $EMB_PORT (loads model once, ~1-2 min)…"
  nohup python3 embedding_server.py --port "$EMB_PORT" > "$LOGDIR/embedding_server.log" 2>&1 &
  echo "  pid $!  (log: $LOGDIR/embedding_server.log; stays up for later batches)"
  for _ in $(seq 1 60); do
    curl -sf "$EMBEDDING_SERVER_URL/health" > /dev/null 2>&1 && break
    sleep 5
  done
  if ! curl -sf "$EMBEDDING_SERVER_URL/health" > /dev/null 2>&1; then
    echo "embedding server failed to become ready — see $LOGDIR/embedding_server.log" >&2
    exit 1
  fi
  echo "  embedding server ready"
fi

header=$(head -1 "$TARGETS")
for i in $(seq 1 "$NWORKERS"); do echo "$header" > "$LOGDIR/part$i.csv"; done
tail -n +2 "$TARGETS" | awk -v n="$NWORKERS" -v dir="$LOGDIR" \
  'NF { print >> (dir "/part" ((NR-1)%n+1) ".csv") }'

pids=()
for i in $(seq 1 "$NWORKERS"); do
  [ "$(wc -l < "$LOGDIR/part$i.csv")" -le 1 ] && continue
  XDG_DATA_HOME="$HOME/.opencode_workers/${RUN}_${BATCH}/w$i-job${SLURM_JOB_ID:-$$}" \
  OPENCODE_PORT=$(( ${OPENCODE_PORT_BASE:-4200} + i * 100 )) \
  npx tsx driver/run.ts \
    --schema-variant "$VARIANT" \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --targets "$LOGDIR/part$i.csv" \
    --slices "$SLICES" \
    --out "${EXP}/${RUN}/${BATCH}/2_records/outputs" \
    --records "${EXP}/${RUN}/${BATCH}/2_records/records" \
    --trajectoryDir "${EXP}/${RUN}/${BATCH}/2_records/trajectories" \
    --concurrency "$CONCURRENCY" \
    > "$LOGDIR/w$i.log" 2>&1 &
  pids+=($!)
  echo "worker $i: pid $! ($(($(wc -l < "$LOGDIR/part$i.csv")-1)) targets) -> $LOGDIR/w$i.log"
done

total_cells=$(( ($(tail -n +2 "$TARGETS" | grep -c .) ) * $(echo "$SLICES" | tr ',' '\n' | grep -c .) ))
echo "waiting for ${#pids[@]} workers ($total_cells cells)…  full logs: tail -f $LOGDIR/w*.log"

( while :; do
    sleep 30
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
    okn=$(grep -h "] OK " "$LOGDIR"/w*.log 2>/dev/null | wc -l)
    failn=$(grep -h "] FAIL" "$LOGDIR"/w*.log 2>/dev/null | wc -l)
    echo "  [$(date +%H:%M:%S)] progress: OK=$okn FAIL=$failn / $total_cells  (workers alive: $alive)"
    [ "$alive" -eq 0 ] && break
  done ) &
progress_pid=$!

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
kill "$progress_pid" 2>/dev/null; wait "$progress_pid" 2>/dev/null

echo ""
echo "=== summary ==="
grep -h "^Done:" "$LOGDIR"/w*.log
grep -h "FAIL" "$LOGDIR"/w*.log | head -20 || true
exit $fail
