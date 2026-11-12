#!/usr/bin/env bash
# NVFP4 language model variant of the host steps (own addition to the recipe). W4A16 export (FP16 activations,
# Edge-LLM's Thor-optimized NVFP4-A16 kernels) and W4A4 export side by side; the FP8 action graph is reused.
set -euo pipefail
export WORK=/mnt/backup/alpamayo2-edge-edgellm-20261112/work
export RECIPE=/mnt/backup/alpamayo2-edge-edgellm-20261112/recipes/recipes/alpamayo2edge_edgellm
cd "$WORK"; T0S=$(date +%s); step() { echo "[host-nvfp4] $1: $(( $(date +%s) - T0S ))s"; }
[ -d quant/nvfp4 ] || env/quant/bin/python quantize_nvfp4.py shim-hf quant/nvfp4 inputs/ea7bbd31
step "nvfp4 quantization"
[ -f shim-nvfp4/shim.json ] || env/host/bin/python "$RECIPE/host/shim.py" quant/nvfp4 shim-nvfp4 --expert shim \
    --fp16-modules "shim:visual.embeddings.patch_embedding,visual.encoder.layers.0.,visual.encoder.layers.1."
step "shim"
source env/host/bin/activate
for v in w4a16 w4a4; do
  d=onnx/nvfp4-$v
  flag=""; [ $v = w4a16 ] && flag="--no-quantize-activations"
  [ -d $d/visual ] || tensorrt-edgellm-export shim-nvfp4 $d --task reasoning --target-sm 110 $flag
  [ -f $d/llm/config.json ] || python "$RECIPE/host/export.py" llm shim-nvfp4 $d/llm --vocab vocab/llm
  [ -e $d/action ] || ln -s ../fp8/action $d/action
  step "export $v"
done
du -sh onnx/nvfp4-*/llm quant/nvfp4
echo HOST_NVFP4_DONE
