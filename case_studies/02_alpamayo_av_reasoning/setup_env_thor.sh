#!/usr/bin/env bash
# Build the case study 02 environment on the Jetson AGX Thor developer kit (JetPack 7.1).
#
# Creates ~/.venv/alpamayo with the system Python 3.12.3 (which meets Alpamayo 1's
# ==3.12.* pin), installs torch 2.11.0+cu130 and torchvision 0.26.0+cu130 (the
# manylinux_2_28_aarch64 wheels from the PyTorch cu130 index; torch 2.8.0, the
# upstream pin, has no wheel for this CUDA 13 aarch64 stack, see PLAN.md and
# FRICTION_LOG.md), transformers 4.57.1, the Alpamayo dependencies except
# flash-attn, the alpamayo_r1 package at the pinned commit with --no-deps so pip
# never tries to build flash-attn, and bench2field (editable, from a copy of this
# repository) for the tegrastats sampler and provenance. Prints elapsed time per
# step for the friction log. Nothing outside the home directory is touched: no
# apt, no JetPack, no nvpmodel.
#
# The 22 GB of weights are not downloaded here. On this board the Hugging Face
# download stalled on name resolution; the weights were copied from the 5090's
# Hugging Face cache with rsync into the same cache layout (~/.cache/huggingface/hub).
#
# Usage on the Thor, with the repository copied to ~/alpamayo-work/bench2field:
#   bash ~/alpamayo-work/bench2field/case_studies/02_alpamayo_av_reasoning/setup_env_thor.sh
set -euo pipefail

VENV=~/.venv/alpamayo
REPO=~/alpamayo-work/bench2field
ALPAMAYO_COMMIT="11a0e01c13a5622377c45ee37d653351453ec43b"
T0=$(date +%s)
elapsed() { echo "[thor-setup] $1: $(( $(date +%s) - T0 ))s since start"; }

python3 -m venv "$VENV"
elapsed "venv created"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu130
elapsed "torch installed"
"$VENV/bin/pip" install -q "transformers==4.57.1" colorlog einops hydra-core "physical_ai_av>=0.2.0" "pillow>=12" scipy nvidia-ml-py matplotlib pyyaml numpy
"$VENV/bin/pip" install -q --no-deps "alpamayo_r1 @ git+https://github.com/NVlabs/alpamayo@${ALPAMAYO_COMMIT}"
"$VENV/bin/pip" install -q -e "$REPO"
elapsed "alpamayo_r1, transformers and bench2field installed"

"$VENV/bin/python" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda runtime", torch.version.cuda, "available", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("transformers", transformers.__version__)
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1  # noqa: F401  import check only, no download
from bench2field.telemetry.tegrastats import TegrastatsSampler
print("alpamayo_r1 imports; tegrastats available:", TegrastatsSampler.available())
PY
elapsed "verified"
echo "[thor-setup] next: $VENV/bin/python $REPO/case_studies/02_alpamayo_av_reasoning/capture_env.py --board bench-thor"
