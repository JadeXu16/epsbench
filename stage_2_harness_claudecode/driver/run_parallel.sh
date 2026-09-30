#!/bin/bash
# Run the Claude Code harness on a targets file with N parallel workers.
# usage: run_parallel.sh <targets.csv> <slices> <workers> <out-root> <model>
set -uo pipefail

TARGETS="${1:?usage: $0 <targets.csv> <slices> <workers> <out-root> <model>}"
SLICES="${2:?}"
WORKERS="${3:?}"
OUTROOT="${4:?}"
MODEL="${5:?}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

[ -s "$TARGETS" ] || { echo "FATAL: missing or empty $TARGETS" >&2; exit 1; }
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${ANTHROPIC_BASE_URL:-}" ]; then
  echo "note: no ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL set;" >&2
  echo "      falling back to whatever credentials the claude CLI finds itself." >&2
  echo "      Run 'claude auth status' and check authMethod if that is a surprise." >&2
fi

RECORDS="$OUTROOT/records"
SHARDDIR="$OUTROOT/.shards"
rm -rf "$SHARDDIR"; mkdir -p "$SHARDDIR" "$RECORDS"

python3 - "$TARGETS" "$RECORDS" "$SLICES" "$WORKERS" "$SHARDDIR" <<'PY'
import csv, sys
from pathlib import Path
targets, records, slices, workers, sharddir = sys.argv[1:6]
records, sharddir = Path(records), Path(sharddir)
slices = [s.strip() for s in slices.split(",") if s.strip()]
workers = int(workers)

rows = list(csv.DictReader(open(targets)))
todo = []
for r in rows:
    if all((records / r["ticker"] / r["fiscal_quarter"] / f"t-{s}.record.json").exists()
           for s in slices):
        continue
    todo.append(r)

print(f"{len(rows)} cells in list, {len(rows)-len(todo)} already done, {len(todo)} to run")
if not todo:
    sys.exit(0)

workers = min(workers, len(todo))
fields = list(rows[0].keys())
handles = []
for i in range(workers):
    fh = open(sharddir / f"shard{i}.csv", "w", newline="")
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
    handles.append((fh, w))
for n, r in enumerate(todo):
    handles[n % workers][1].writerow(r)
for fh, _ in handles:
    fh.close()
print(f"sharded into {workers} worker(s)")
PY
rc=$?
[ $rc -ne 0 ] && { echo "nothing to do."; exit 0; }

shopt -s nullglob
SHARDS=("$SHARDDIR"/shard*.csv)
[ ${#SHARDS[@]} -eq 0 ] && { echo "nothing to do."; exit 0; }

echo "=== launching ${#SHARDS[@]} worker(s), model=$MODEL, slices=$SLICES ==="
pids=()
for i in "${!SHARDS[@]}"; do
  (
    export TMPDIR="${FINBENCH_TMPDIR_BASE:-$OUTROOT/.tmp}/w$i"; mkdir -p "$TMPDIR"
    python3 driver/run.py \
      --targets "${SHARDS[$i]}" \
      --slices "$SLICES" \
      --model "$MODEL" \
      --out "$OUTROOT/outputs" \
      --records "$RECORDS" \
      --trajectory-dir "$OUTROOT/trajectories" \
      --tool-log-dir "$OUTROOT/tool_logs" \
      > "$OUTROOT/worker$i.log" 2>&1
  ) &
  pids+=($!)
  echo "  worker $i -> $OUTROOT/worker$i.log  (pid ${pids[-1]})"
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done

NSLICES=$(echo "$SLICES" | tr ',' '\n' | grep -c .)
NCELLS=$(($(grep -c . "$TARGETS") - 1))
WANT=$((NCELLS * NSLICES))
DONE=$(find "$RECORDS" -name '*.record.json' 2>/dev/null | wc -l)
echo "=== finished: $DONE/$WANT record(s) ==="
if [ "$DONE" -lt "$WANT" ]; then
  echo "$((WANT - DONE)) still missing. Check $OUTROOT/worker*.log for the reason,"
  echo "then re-run this exact command -- it retries only what is missing."
  exit 1
fi
echo "complete."
exit 0
