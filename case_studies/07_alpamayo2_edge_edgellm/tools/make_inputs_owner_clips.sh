#!/usr/bin/env bash
# Inputs for the eight clips of bench2field case study 06 (same clips, t0 and derived navigation text as the
# PyTorch runs of 2026-11-10), so the Edge-LLM engines are timed on the same decisions.
set -euo pipefail
export WORK=/mnt/backup/alpamayo2-edge-edgellm-20261112/work
export RECIPE=/mnt/backup/alpamayo2-edge-edgellm-20261112/recipes/recipes/alpamayo2edge_edgellm
export HF_HUB_ENABLE_HF_TRANSFER=0
cd "$WORK"
python3 -I -c "
import json
for s in json.load(open(__import__('os').path.expanduser('~/github/bench2field/case_studies/06_alpamayo2_edge/nav_clips.json')))['samples']:
    print(s['clip_id'], s['t0_us'], s['nav_text'])" | while read -r clip t0 nav; do
  short=${clip:0:8}
  [ -f "inputs/$short/request.json" ] && { echo "have $short"; continue; }
  env/inputs/bin/python "$RECIPE/host/make_inputs.py" --clip "$clip" "$t0" "$nav" downloads/Alpamayo2-Edge "inputs/$short" && echo "made $short: $nav"
done
echo OWNER_INPUTS_DONE
