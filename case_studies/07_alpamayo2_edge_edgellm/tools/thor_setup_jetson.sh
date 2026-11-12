#!/usr/bin/env bash
# Jetson AGX Thor (JetPack 7.1) adaptation of recipes/alpamayo2edge_edgellm/thor/setup.sh.
# Differences from the DRIVE OS script: TensorRT is the JetPack system package (headers and
# libraries under /usr, Python bindings in /usr/lib/python3.12/dist-packages), so no SDK
# wheel; CUDA 13.0 toolkit selected explicitly (the cuda alternative points at 13.2, the
# TensorRT package is built against 13.0, Edge-LLM's JetPack 7.1 build uses 13.0).
set -euo pipefail
BASE=~/alpamayo2-edge-edgellm-work
RECIPE_DIR=$BASE/recipes/recipes/alpamayo2edge_edgellm
WORK=$BASE/work
mkdir -p "$WORK/bin" "$WORK/cache/tmp" && cd "$WORK"
T0=$(date +%s); step() { echo "[thor-setup] $1: $(( $(date +%s) - T0 ))s"; }
if [ ! -x bin/uv ]; then
  wget -qO- https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-unknown-linux-gnu.tar.gz | tar xz --strip-components=1 -C bin
fi
step "uv"
[ -x venv/bin/python ] || bin/uv venv --python /usr/bin/python3.12 venv >/dev/null
UV_CACHE_DIR="$WORK/cache/uv" UV_PROJECT_ENVIRONMENT="$WORK/venv" bin/uv sync --project "$RECIPE_DIR" --frozen --inexact --no-install-project --no-install-package alpamayo-recipes --quiet
step "uv sync (cmake, ninja, torch cu130, tokenizers, curand headers)"
# system TensorRT Python bindings into the venv (no wheel ships with JetPack)
SP=$(venv/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
for d in /usr/lib/python3.12/dist-packages/tensorrt*; do ln -sfn "$d" "$SP/$(basename "$d")"; done
cat > env.sh <<EOT
export WORK=$WORK TENSORRT_ROOT=/usr CUDA_HOME=/usr/local/cuda-13.0
export PATH=$WORK/venv/bin:$WORK/bin:/usr/local/cuda-13.0/bin:\$PATH
export CUDA_CACHE_PATH=$WORK/cache/cuda UV_CACHE_DIR=$WORK/cache/uv TMPDIR=$WORK/cache/tmp
EOT
source env.sh
python - <<'EOT' > board.json
import json, re, subprocess, tensorrt, torch
v = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
print(json.dumps({"platform": "Jetson AGX Thor", "label": "bench-thor", "l4t": v("head -1 /etc/nv_tegra_release"),
  "jetpack": v("apt-cache policy nvidia-jetpack | grep -m1 Installed"), "kernel": v("uname -r"),
  "cuda": re.search(r"release ([\d.]+)", v("nvcc --version")).group(1), "tensorrt": tensorrt.__version__,
  "torch": torch.__version__, "power_mode": v("nvpmodel -q | head -1"), "cpus": int(v("nproc"))}, indent=1))
EOT
cat board.json
step "done"
