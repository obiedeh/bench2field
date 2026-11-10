#!/usr/bin/env bash
# Case study 06 optimization runs on the Jetson AGX Thor, in order, after the
# flash-attn build has finished. Each run is one variant; logs go next to the work tree.
set -u
W=~/alpamayo2-edge-work
CS=$W/bench2field/case_studies/06_alpamayo2_edge
PY=~/.venv/alpamayo2-edge/bin/python
export B2F_HOST_LABEL=bench-thor PYTHONPATH=$CS
until grep -q "^EXIT" $W/flashattn_build.log; do sleep 30; done
grep "^EXIT" $W/flashattn_build.log
$PY -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || echo "flash_attn not importable"
cd $CS
R="$PY run_decisions.py --board bench-thor --model $W/model --nav $W/nav_futures.json --sample-cache $W/sample_cache"
S="$PY run_stream.py --board bench-thor --model $W/model --nav $W/nav_futures.json --sample-cache $W/sample_cache"
run() { local tag=$1; shift; "$@" > $W/run_thor_$tag.log 2>&1; echo "EXIT $? $tag" >> $W/thor_matrix.log; }
run eager_steps5_prepcached $R --variant eager --diffusion-steps 5 --prep cached --tag steps5_prepcached
run tuned $R --variant tuned --prep cached --tag prepcached
run tuned_steps5 $R --variant tuned --diffusion-steps 5 --prep cached --tag steps5_prepcached
run stream_eager $S --variant eager
run stream_tuned $S --variant tuned
run stream_tuned_steps5 $S --variant tuned --diffusion-steps 5 --tag steps5
echo DONE >> $W/thor_matrix.log
