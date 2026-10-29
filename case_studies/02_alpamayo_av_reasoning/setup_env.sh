#!/usr/bin/env bash
# Build the case study 02 environment on an x86_64 host with an NVIDIA GPU.
#
# Creates ./.venv (Python 3.12, which Alpamayo 1 requires exactly), installs
# torch 2.8.0 for CUDA 12.8, the prebuilt flash-attn 2.8.3 wheel that matches
# it, transformers 4.57.1, the NVlabs/alpamayo package pinned to one commit,
# and bench2field itself (editable) for provenance, background and NVML
# telemetry. Prints the elapsed time after each step so the setup time in
# FRICTION_LOG.md can be reproduced.
#
# Needs uv on PATH. On the 5090 host every command runs under
# `env -u PYTHONPATH` because the shell sources ROS 2, whose PYTHONPATH
# would shadow the venv; the script does that itself.
#
# Usage, from the repository root:
#   bash case_studies/02_alpamayo_av_reasoning/setup_env.sh
#
# This script is for the 5090 host. The Thor needs a Jetson torch wheel in
# place of the cu128 one and no flash-attn; see SETUP.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV="$HERE/.venv"
ALPAMAYO_COMMIT="11a0e01c13a5622377c45ee37d653351453ec43b"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
T0=$(date +%s)
elapsed() { echo "[setup] $1: $(( $(date +%s) - T0 ))s since start"; }

run() { env -u PYTHONPATH VIRTUAL_ENV="$VENV" "$@"; }

run uv venv -p 3.12 "$VENV"
elapsed "venv created"

run uv pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128
elapsed "torch installed"

run uv pip install "$FLASH_ATTN_WHEEL"
elapsed "flash-attn wheel installed"

run uv pip install "transformers==4.57.1" \
    "alpamayo_r1 @ git+https://github.com/NVlabs/alpamayo@${ALPAMAYO_COMMIT}" \
    "nvidia-ml-py>=12" "matplotlib>=3.10"
run uv pip install -e "$REPO_ROOT"
elapsed "alpamayo_r1, transformers and bench2field installed"

env -u PYTHONPATH "$VENV/bin/python" - <<'PY'
import torch, transformers, flash_attn
print("torch", torch.__version__, "cuda runtime", torch.version.cuda,
      "available", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("transformers", transformers.__version__, "flash_attn", flash_attn.__version__)
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1  # noqa: F401  import check only, no download
print("alpamayo_r1 imports")
PY
elapsed "verified"
echo "[setup] next: env -u PYTHONPATH $VENV/bin/python $HERE/capture_env.py --board bench-5090"
