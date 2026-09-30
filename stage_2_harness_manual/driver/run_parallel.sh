#!/bin/bash
# Run the vanilla harness on a targets file with N parallel workers.
# usage: run_parallel.sh <targets.csv> <slices> <workers> <out-root> <model> [base-url]
set -uo pipefail

TARGETS="${1:?usage: $0 <targets.csv> <slices> <workers> <out-root> <model> [base-url]}"
SLICES="${2:?}"
WORKERS="${3:?}"
OUTROOT="${4:?}"
MODEL="${5:?}"
BASEURL="${6:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
[ -s "$TARGETS" ] || { echo "FATAL: missing or empty $TARGETS" >&2; exit 1; }
if [ -n "$BASEURL" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "FATAL: base-url given but OPENAI_API_KEY is not set" >&2; exit 1
fi
if [ -z "$BASEURL" ] && [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  echo "FATAL: AZURE_OPENAI_API_KEY is not set (or pass a base-url + OPENAI_API_KEY)" >&2; exit 1
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
rows = list(csv.DictReader(open(targets)))
todo = [r for r in rows if not all((records / r["ticker"] / r["fiscal_quarter"] / f"t-{s}.record.json").exists() for s in slices)]
print(f"{len(rows)} cells in list, {len(rows)-len(todo)} already done, {len(todo)} to run")
if not todo:
    sys.exit(0)
workers = min(int(workers), len(todo))
fields = list(rows[0].keys())
hs = []
for i in range(workers):
    fh = open(sharddir / f"shard{i}.csv", "w", newline=""); w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); hs.append((fh, w))
for n, r in enumerate(todo):
    hs[n % workers][1].writerow(r)
for fh, _ in hs:
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
    for s in $(echo "$SLICES" | tr ',' ' '); do
      python3 run_cell.py --targets "${SHARDS[$i]}" --slice-days "$s" --model "$MODEL" \
        --out-root "$OUTROOT" ${BASEURL:+--base-url "$BASEURL"}
    done > "$OUTROOT/worker$i.log" 2>&1
  ) &
  pids+=($!)
  echo "  worker $i -> $OUTROOT/worker$i.log  (pid ${pids[-1]})"
done
for p in "${pids[@]}"; do wait "$p"; done

NSLICES=$(echo "$SLICES" | tr ',' '\n' | grep -c .)
NCELLS=$(($(grep -c . "$TARGETS") - 1))
WANT=$((NCELLS * NSLICES))
DONE=$(find "$RECORDS" -name '*.record.json' 2>/dev/null | wc -l)
echo "=== finished: $DONE/$WANT record(s) ==="
if [ "$DONE" -lt "$WANT" ]; then
  echo "$((WANT - DONE)) still missing. Check $OUTROOT/worker*.log, then re-run this exact command."
  exit 1
fi
echo "complete."
