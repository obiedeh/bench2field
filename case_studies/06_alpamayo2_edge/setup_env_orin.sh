#!/usr/bin/env bash
# Try to build an Alpamayo 2 Edge environment on the rover's Jetson Orin NX 8 GB (JetPack 6,
# R36.4.7, CUDA 12.6, Python 3.10 only). User space only: a venv under ~/.venv, no apt, no
# service changes, no files removed. Every step prints elapsed time and the disk left, and
# keeps going far enough to record where it stops.
#
# Edge declares Python 3.12; the only Jetson torch wheels for this JetPack are cp310, so the
# Edge package is installed with --ignore-requires-python to find out whether the code runs
# on 3.10. pip runs with --no-cache-dir because the disk had 5.8 GB free.
set -u
VENV=~/.venv/alpamayo2-edge
WORK=~/alpamayo2-edge-work
T0=$(date +%s)
step() { echo "[orin-setup] $1: $(( $(date +%s) - T0 ))s, disk free $(df -h / | tail -1 | awk '{print $4}')"; }
step "start"
# JetPack 6 ships Python 3.10 without the python3-venv package (ensurepip is missing), so
# `python3 -m venv` fails; virtualenv from the user site works without apt.
python3 -m pip install -q --user --no-cache-dir virtualenv && python3 -m virtualenv -q "$VENV" && step "venv created (virtualenv)"
P="$VENV/bin/pip"
$P install -q --no-cache-dir --upgrade pip
$P install -q --no-cache-dir "torch==2.8.0" "torchvision==0.23.0" --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 ; echo "torch exit $?"; step "torch"
# SciPy 1.16 (Edge's pin) needs Python 3.11; 1.15.3 is the last for 3.10. NumPy stays below 2
# because the Jetson torch 2.8 wheel is built against NumPy 1.x. bitsandbytes for the low-bit loads.
$P install -q --no-cache-dir "transformers==5.17.0" "accelerate>=1.12.0" "huggingface-hub>=1.0,<2" "safetensors>=0.8.0" \
    "einops>=0.8.1" "hydra-core>=1.3.2" "numpy<2" "pillow>=11.0" "scipy==1.15.3" "colorlog>=6.0" pyyaml nvidia-ml-py
$P install -q --no-cache-dir bitsandbytes --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 ; echo "deps exit $?"; step "dependencies"
$P install -q --no-cache-dir --no-deps --ignore-requires-python "$WORK/code" ; echo "edge exit $?"; step "alpamayo2_edge"
$VENV/bin/python - <<'PY'
import sys, torch
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
try:
    import transformers; print("transformers", transformers.__version__)
except Exception as e:
    print("transformers import failed:", type(e).__name__, e)
try:
    from alpamayo2_edge.checkpoint import load_model  # noqa: F401
    print("alpamayo2_edge imports on", sys.version.split()[0])
except Exception as e:
    print("alpamayo2_edge import failed:", type(e).__name__, e)
PY
step "verified"
