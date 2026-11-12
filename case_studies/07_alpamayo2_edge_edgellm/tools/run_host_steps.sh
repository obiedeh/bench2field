#!/usr/bin/env bash
# Host (RTX 5090) steps of NVIDIA's Alpamayo2-Edge Edge-LLM recipe, as written in
# recipes/alpamayo2edge_edgellm/README.md, steps 3 to 7. FP8, with the reduced output
# vocabulary. The DFlash2 drafter is not released, so the draft steps are skipped
# (greedy decoding on the board).
# Environments were created by setup_host.log (env/host, env/quant, env/inputs).
# Weights: downloads/Alpamayo2-Edge -> /mnt/backup/alpamayo2-edge-20261110/model,
# sha256 of both safetensors files equal to Hugging Face revision aaaa32c6 (model_sha256.txt).
# Usage: bash run_host_steps.sh [CLIP_ID T0_US "NAV TEXT"]   (default: the recipe's example clip)
set -euo pipefail
export WORK=/mnt/backup/alpamayo2-edge-edgellm-20261112/work
export RECIPE=/mnt/backup/alpamayo2-edge-edgellm-20261112/recipes/recipes/alpamayo2edge_edgellm
export UV_CACHE_DIR=/mnt/backup/uv-cache HF_HUB_ENABLE_HF_TRANSFER=0
CLIP=${1:-ea7bbd31-b7a5-4972-8dbd-7089e6b53de4}; T0=${2:-4000000}; NAV=${3:-"Turn right in 30m"}
SHORT=${CLIP:0:8}
cd "$WORK"
T0S=$(date +%s); step() { echo "[host] $1: $(( $(date +%s) - T0S ))s"; }

# 3. inputs (one clip from the gated PhysicalAI-AV dataset through the public alpamayo2_edge package)
[ -f "inputs/$SHORT/request.json" ] || env/inputs/bin/python "$RECIPE/host/make_inputs.py" --clip "$CLIP" "$T0" "$NAV" downloads/Alpamayo2-Edge "inputs/$SHORT"
step "inputs $SHORT"

# 4. shim: release layout -> exporter layout, and -> HF layout for the quantizer
[ -f shim/shim.json ] || env/host/bin/python "$RECIPE/host/shim.py" downloads/Alpamayo2-Edge shim
[ -f shim-hf/shim.json ] || env/host/bin/python "$RECIPE/host/shim.py" downloads/Alpamayo2-Edge shim-hf --hf
step "shim"

# 5. FP8 quantization, calibrated on the clip's frames, FP8 K/V-cache scales
[ -d quant/fp8 ] || env/quant/bin/python "$RECIPE/host/quantize.py" shim-hf quant/fp8 "inputs/$SHORT" --kv-fp8
[ -f shim-fp8/shim.json ] || env/host/bin/python "$RECIPE/host/shim.py" quant/fp8 shim-fp8 --expert shim \
    --fp16-modules "shim:visual.embeddings.patch_embedding,visual.encoder.layers.0.,visual.encoder.layers.1."
step "fp8 quantization"

# 6. reduced output vocabulary (2211 chain-of-causation token ids padded to 8192 rows)
[ -d vocab/llm ] || env/host/bin/python "$RECIPE/host/vocab.py" shim-fp8 "$RECIPE/host/coc_vocab.json" 8192 vocab/llm
step "vocab"

# 7. ONNX export for SM 110 (Thor)
source env/host/bin/activate
[ -d onnx/fp8/visual ] || tensorrt-edgellm-export shim-fp8 onnx/fp8 --task reasoning --target-sm 110
[ -f onnx/fp8/llm/config.json ] || python "$RECIPE/host/export.py" llm shim-fp8 onnx/fp8/llm --vocab vocab/llm
[ -f onnx/fp8/action/config.json ] || python "$RECIPE/host/action_graph.py" shim-fp8 onnx/fp8/action 5 3584 --fp8-kv --fp8 "$RECIPE/host/expert_scales.json"
step "onnx export"
du -sh onnx inputs quant 2>/dev/null
echo HOST_STEPS_DONE
