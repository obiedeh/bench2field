#!/usr/bin/env bash
# Build the case study 06 environment (Alpamayo 2 Edge) on the Jetson AGX Thor (JetPack 7.1).
#
# Creates ~/.venv/alpamayo2-edge with the system Python 3.12.3 (Edge pins 3.12),
# installs torch 2.11.0+cu130 and torchvision 0.26.0+cu130 from the PyTorch cu130
# index, because Edge's lock pins torch 2.8.0 for x86_64 CUDA 12 and that build has
# no wheel for this aarch64 CUDA 13 stack (the same substitution case study 02 made
# for the teacher), then transformers 5.17.0 and the rest of Edge's dependencies,
# the Edge package from the copied source with --no-deps so its torch pin is not
# enforced, physical_ai_av for the clips, nvidia-ml-py, and bench2field (editable,
# from the copied tree) for the tegrastats sampler. Prints elapsed time per step for
# the friction log. Nothing outside the home directory is touched: no apt, no
# JetPack, no nvpmodel.
#
# The weights (5.0 GB) are copied from the 5090 host with rsync rather than
# downloaded, because Hugging Face downloads stalled on this board in case study 02.
#
# Usage on the Thor, with ~/alpamayo2-edge-work/{code,model,bench2field} copied over:
#   bash ~/alpamayo2-edge-work/bench2field/case_studies/06_alpamayo2_edge/setup_env_thor.sh
set -euo pipefail
VENV=~/.venv/alpamayo2-edge
WORK=~/alpamayo2-edge-work
T0=$(date +%s)
elapsed() { echo "[thor-setup] $1: $(( $(date +%s) - T0 ))s since start"; }
python3 -m venv "$VENV"
elapsed "venv created"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu130
elapsed "torch installed"
"$VENV/bin/pip" install -q "transformers==5.17.0" "accelerate>=1.12.0" "huggingface-hub>=1.0,<2" \
    "safetensors>=0.8.0" "einops>=0.8.1" "hydra-core>=1.3.2" "numpy>=2.0" "pillow>=11.0" \
    "scipy>=1.16.0" "colorlog>=6.0" "physical-ai-av==0.2.0" nvidia-ml-py pyyaml
"$VENV/bin/pip" install -q --no-deps "$WORK/code"
"$VENV/bin/pip" install -q -e "$WORK/bench2field"
elapsed "alpamayo2_edge, transformers and bench2field installed"
"$VENV/bin/python" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda runtime", torch.version.cuda, "available", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      "bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)
print("transformers", transformers.__version__)
from alpamayo2_edge.checkpoint import load_model  # noqa: F401  import check only
from bench2field.telemetry.tegrastats import TegrastatsSampler
print("alpamayo2_edge imports; tegrastats available:", TegrastatsSampler.available())
PY
elapsed "verified"
