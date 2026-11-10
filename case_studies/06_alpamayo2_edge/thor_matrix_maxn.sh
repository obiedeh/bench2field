#!/usr/bin/env bash
# Case study 06, Thor at the MAXN power mode (owner switched it, 2026-11-10). Repeats the
# key variants of the 90 W sets so the two modes compare; records are tagged "maxn".
# Refuses to run unless nvpmodel reports MAXN.
set -u
W=~/alpamayo2-edge-work
CS=$W/bench2field/case_studies/06_alpamayo2_edge
PY=~/.venv/alpamayo2-edge/bin/python
nvpmodel -q 2>/dev/null | grep -q "MAXN" || { echo "power mode is not MAXN; nothing run"; nvpmodel -q; exit 1; }
nvpmodel -q > $W/power_mode_maxn.txt
export B2F_HOST_LABEL=bench-thor PYTHONPATH=$CS
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
cd $CS
R="$PY run_decisions.py --board bench-thor --model $W/model --nav $W/nav_futures.json --sample-cache $W/sample_cache"
S="$PY run_stream.py --board bench-thor --model $W/model --nav $W/nav_futures.json --sample-cache $W/sample_cache"
run() { local tag=$1; shift; "$@" > $W/run3_thor_$tag.log 2>&1; echo "EXIT $? $tag" >> $W/thor_matrix_maxn.log; }
rm -f $W/thor_matrix_maxn.log
run eager $R --variant eager --prep cached --tag maxn
run tuned $R --variant tuned --prep cached --tag maxn
run tuned_fp8_steps5 $R --variant tuned --fp8 --diffusion-steps 5 --prep cached --tag fp8_steps5_maxn
run stream_tuned $S --variant tuned --tag maxn
run stream_tuned_fp8_steps5 $S --variant tuned --fp8 --diffusion-steps 5 --tag fp8_steps5_maxn
echo DONE >> $W/thor_matrix_maxn.log
