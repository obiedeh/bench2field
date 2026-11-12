#!/usr/bin/env bash
# Copy the board's run records into runs/ with private paths replaced (hygiene rule 9): the board's home directory
# becomes <home>. Re-run after every new measurement.
set -euo pipefail
SRC=/mnt/backup/alpamayo2-edge-edgellm-20261112/work/thor_results
DST=$(cd "$(dirname "$0")/.." && pwd)/runs
mkdir -p "$DST/stream" "$DST/results"
for f in "$SRC"/stream/*.json "$SRC"/*.json; do
  sed -E 's#/h[o]me/[A-Za-z0-9_.-]+#<home>#g' "$f" > "$DST/stream/$(basename "$f")"
done
for d in "$SRC"/results/*/; do
  set=$(basename "$d"); mkdir -p "$DST/results/$set"
  for c in "$d"*/; do
    clip=$(basename "$c"); mkdir -p "$DST/results/$set/$clip"
    for f in output.json profile.json compare.json; do
      [ -f "$c/$f" ] && sed -E 's#/h[o]me/[A-Za-z0-9_.-]+#<home>#g' "$c/$f" > "$DST/results/$set/$clip/$f"
    done
  done
done
for f in "$SRC"/results/cross_*.json; do [ -f "$f" ] && cp "$f" "$DST/results/"; done
grep -rl "/h[o]me/" "$DST" && { echo "home path left in runs/"; exit 1; } || echo "runs collected: $(find "$DST" -name '*.json' | wc -l) files"
