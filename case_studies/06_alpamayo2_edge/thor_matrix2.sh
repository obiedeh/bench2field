#!/usr/bin/env bash
# Case study 06, second Thor set: the accelerated variants that failed in the first set
# (Triton's bundled ptxas-blackwell does not know sm_110a; the system CUDA 13.2 ptxas
# is used instead), FP8, and the combinations with vision reuse.
set -u
W=~/alpamayo2-edge-work
CS=$W/bench2field/case_studies/06_alpamayo2_edge
PY=~/.venv/alpamayo2-edge/bin/python
export B2F_HOST_LABEL=bench-thor PYTHONPATH=$CS
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
cd $CS
R="$PY run_decisions.py --board bench-thor --model $W/model --nav $W/nav_futures.json --sample-cache $W/sample_cache"
S="$PY run_stream.py --board bench-thor --model $W/model --nav $W/nav_futures.json --sample-cache $W/sample_cache"
run() { local tag=$1; shift; "$@" > $W/run2_thor_$tag.log 2>&1; echo "EXIT $? $tag" >> $W/thor_matrix2.log; }
run tuned $R --variant tuned --prep cached --tag prepcached
run tuned_steps5 $R --variant tuned --diffusion-steps 5 --prep cached --tag steps5_prepcached
run tuned_fp8_steps5 $R --variant tuned --fp8 --diffusion-steps 5 --prep cached --tag fp8_steps5
run stream_tuned $S --variant tuned
run stream_tuned_steps5 $S --variant tuned --diffusion-steps 5 --tag steps5
run stream_tuned_fp8_steps5 $S --variant tuned --fp8 --diffusion-steps 5 --tag fp8_steps5
echo DONE >> $W/thor_matrix2.log
