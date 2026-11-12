#!/usr/bin/env python3
"""NVFP4 language model, FP8 vision tower, FP8 K/V cache: the recipe's quantize.py with the language model's
weights in NVFP4 (lm_head stays FP16, as in Embedl's published build), calibrated on the same driving frames.
Usage: python quantize_nvfp4.py SHIM_HF OUT_DIR CLIP_DIR [CLIP_DIR ...]"""
import sys
from pathlib import Path
sys.path.insert(0, "/mnt/backup/alpamayo2-edge-edgellm-20261112/recipes/recipes/alpamayo2edge_edgellm/host")
from quantize import driving_frames
from tensorrt_edgellm.quantization.quantize import quantize_and_export

shim_hf, out, clips = Path(sys.argv[1]), Path(sys.argv[2]), [Path(c) for c in sys.argv[3:]]
frames = sum(len(list((c / "images").iterdir())) for c in clips)
quantize_and_export(model_dir=str(shim_hf), output_dir=str(out), quantization="nvfp4", visual_quantization="fp8",
                    kv_cache_quantization="fp8", image_dataset=driving_frames(clips), num_samples=frames)
