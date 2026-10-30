"""Tests for the case study 02 preprocessing kernels that need no GPU.

Three things are checked on the CPU: the resampling tables preprocess_ext
builds (normalized, in range, within MAX_TAPS, and matching torchvision's
antialiased bicubic resize on a small image), the patch index mapping the
kernel scatters into (against the processor's own view, permute and reshape,
pixel by pixel), and the committed measurement records (dataset input,
measured label, both kernel variants within one uint8 step of the PyTorch GPU
reference, every timing present). One opt-in test builds and runs both
kernel variants on the GPU when ALPAMAYO_KERNEL_GPU_TEST=1 and a CUDA device
and toolkit are present; it is the same gate the benchmark applies.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

CASE = Path(__file__).resolve().parents[1] / "case_studies" / "02_alpamayo_av_reasoning"
sys.path.insert(0, str(CASE))

import preprocess_ext as pe  # noqa: E402

ONE_STEP = 2.0 / 255.0
RECORDS = sorted(CASE.glob("runs/preprocess_*.json"))


def test_tables_are_normalized_in_range_and_within_max_taps():
    for in_size, out_size in ((pe.IN_H, pe.OUT_H), (pe.IN_W, pe.OUT_W)):
        starts, weights = pe.bicubic_aa_table(in_size, out_size)
        taps = (weights != 0).sum(1)
        assert int(taps.max()) <= pe.MAX_TAPS
        assert torch.allclose(weights.sum(1), torch.ones(out_size), atol=1e-6)
        assert int(starts.min()) >= 0 and int((starts + taps).max()) <= in_size


def _resample_with_tables(img: torch.Tensor, in_h: int, in_w: int, out_h: int, out_w: int) -> torch.Tensor:
    ys, yw = pe.bicubic_aa_table(in_h, out_h)
    xs, xw = pe.bicubic_aa_table(in_w, out_w)
    out = torch.zeros(out_h, out_w)
    for y in range(out_h):
        for x in range(out_w):
            acc = 0.0
            for j in range(pe.MAX_TAPS):
                if yw[y, j] == 0:
                    continue
                row = img[min(int(ys[y]) + j, in_h - 1)]
                for i in range(pe.MAX_TAPS):
                    acc += float(yw[y, j]) * float(xw[x, i]) * float(row[min(int(xs[x]) + i, in_w - 1)])
            out[y, x] = acc
    return out


def test_tables_reproduce_torchvision_antialiased_bicubic():
    """A 54x96 image downscaled 3.375x (the case study's scale) by the tables
    equals torchvision's antialiased bicubic resize to within float rounding."""
    F = pytest.importorskip("torchvision.transforms.v2.functional")
    g = torch.Generator().manual_seed(0)
    img = torch.rand(1, 1, 54, 96, generator=g) * 255.0
    ref = F.resize(img, [16, 28], interpolation=F.InterpolationMode.BICUBIC, antialias=True)[0, 0]
    # 54/16 = 3.375 = 1080/320, 96/28 = 3.43 ~ 1920/576 = 3.33: same tap counts as the kernel's geometry.
    mine = _resample_with_tables(img[0, 0], 54, 96, 16, 28)
    assert float((mine - ref).abs().max()) < 1e-2


def test_patch_index_mapping_equals_processor_layout():
    """The kernel's closed-form patch row and column for a pixel match the
    processor's view/permute/reshape, checked on a labelled image."""
    img = torch.arange(3 * pe.OUT_H * pe.OUT_W, dtype=torch.float32).reshape(1, 3, pe.OUT_H, pe.OUT_W)
    lay = pe.reference_layout(img)
    assert tuple(lay.shape) == (pe.PATCHES_PER_IMAGE, pe.ROW_LEN)
    for (c, y, x) in ((0, 0, 0), (1, 17, 33), (2, 319, 575), (0, 160, 288), (2, 31, 32)):
        bh, mh, py = y // 32, (y % 32) // 16, y % 16
        bw, mw, px = x // 32, (x % 32) // 16, x % 16
        row = ((bh * (pe.GRID_W // pe.MERGE) + bw) * pe.MERGE + mh) * pe.MERGE + mw
        for t in range(pe.TEMPORAL):
            col = ((c * pe.TEMPORAL + t) * pe.PATCH + py) * pe.PATCH + px
            assert float(lay[row, col]) == float(img[0, c, y, x])


def test_committed_preprocess_records_are_measured_dataset_runs():
    assert RECORDS, "no committed preprocess records"
    for p in RECORDS:
        text = p.read_text()
        assert '"input": "dataset"' in text and '"label": "measured"' in text, p.name


@pytest.mark.parametrize("record", RECORDS, ids=[p.name for p in RECORDS])
def test_committed_record_passes_the_parity_gate(record: Path):
    """Both kernel variants agree with the PyTorch GPU reference within one
    uint8 step after normalization, and every timing path is present."""
    rec = json.loads(record.read_text())
    err = rec["errors"]
    assert err["kernel_vs_torch_gpu"] <= ONE_STEP + 1e-6, record.name
    assert err["separable_vs_torch_gpu"] <= ONE_STEP + 1e-6, record.name
    assert err["gate_passed"] is True, record.name
    assert set(rec["kernel"]["variants"]) == set(pe.VARIANTS)
    for key in ("kernel", "kernel_separable", "kernel+upload", "kernel_separable+upload", "torch_gpu", "processor_cpu"):
        assert key in rec["timings_per_16_frames"], f"{record.name}: {key} timing missing"
    for key in ("fused", "separable", "torch_gpu"):
        assert rec["profiler_per_16_frames"][key]["device_time_per_batch_us"] > 0, f"{record.name}: {key} profile missing"


@pytest.mark.skipif(os.environ.get("ALPAMAYO_KERNEL_GPU_TEST") != "1" or not torch.cuda.is_available(),
                    reason="set ALPAMAYO_KERNEL_GPU_TEST=1 on a machine with a CUDA device and toolkit")
def test_kernels_match_torch_gpu_within_one_uint8_step():
    """Builds the extension and runs both variants on random frames."""
    g = torch.Generator().manual_seed(0)
    frames = torch.randint(0, 256, (2, 3, pe.IN_H, pe.IN_W), generator=g, dtype=torch.uint8).cuda()
    ref = pe.reference_torch(frames)
    outs = {v: pe.preprocess(frames, variant=v) for v in pe.VARIANTS}
    for v, out in outs.items():
        assert pe.max_abs_error(out, ref) <= ONE_STEP + 1e-6, v
    assert pe.max_abs_error(outs["fused"], outs["separable"]) == 0.0
