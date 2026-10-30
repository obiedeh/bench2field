"""CPU-only checks for case study 02's preprocessing kernel support code.

The kernel itself needs a GPU and a compiler; these tests pin down the two
things it depends on that can be checked anywhere: the antialiased bicubic
weight tables must reproduce torchvision's resize, and the patch index
mapping the kernel hard-codes must equal the processor's view/permute/
reshape. If either drifts, the GPU comparison in bench_preprocess.py would
fail in a way that is hard to place; here it fails with a name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
CASE = Path(__file__).resolve().parents[1] / "case_studies" / "02_alpamayo_av_reasoning"


def _ext():
    spec = importlib.util.spec_from_file_location("preprocess_ext", CASE / "preprocess_ext.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_tables_are_normalized_in_range_and_within_max_taps():
    pe = _ext()
    for in_size, out_size in ((pe.IN_H, pe.OUT_H), (pe.IN_W, pe.OUT_W)):
        starts, weights = pe.bicubic_aa_table(in_size, out_size)
        taps = (weights != 0).sum(1)
        assert int(taps.max()) <= pe.MAX_TAPS
        assert torch.allclose(weights.sum(1), torch.ones(out_size), atol=1e-6)
        assert int(starts.min()) >= 0 and int((starts + taps).max()) <= in_size


def _dense(starts: torch.Tensor, weights: torch.Tensor, in_size: int) -> torch.Tensor:
    """Expand a (start, weights) table into a dense [out, in] resampling matrix."""
    m = torch.zeros(weights.shape[0], in_size)
    for i in range(weights.shape[0]):
        s0 = int(starts[i])
        n = int((weights[i] != 0).sum())
        m[i, s0:s0 + n] = weights[i, :n]
    return m


def _resample_with_tables(pe, img: torch.Tensor) -> torch.Tensor:
    """Apply the kernel's tables on the CPU: rows then columns, the same weighted sums the kernel does."""
    wy = _dense(*pe.bicubic_aa_table(img.shape[-2], pe.OUT_H), img.shape[-2])
    wx = _dense(*pe.bicubic_aa_table(img.shape[-1], pe.OUT_W), img.shape[-1])
    x = img.float()
    return wy @ x @ wx.T


def test_tables_reproduce_torchvision_antialiased_bicubic():
    """Resample a float image with the tables and with torchvision; they must
    agree to float precision. (On uint8 input torchvision's CPU path uses
    fixed-point weights and lands one grey level away on about a tenth of the
    pixels; the GPU path, which the kernel is compared against, casts to float.
    Testing on float isolates the tables from that implementation detail.)"""
    pytest.importorskip("torchvision")
    from torchvision.transforms.v2 import functional as F

    pe = _ext()
    g = torch.Generator().manual_seed(0)
    img = torch.randint(0, 256, (1, 3, pe.IN_H, pe.IN_W), generator=g, dtype=torch.uint8).float()
    ref = F.resize(img, [pe.OUT_H, pe.OUT_W], interpolation=F.InterpolationMode.BICUBIC, antialias=True)
    mine = _resample_with_tables(pe, img)
    # Same weights, different float32 summation order: torch accumulates per tap, this test
    # uses two matrix products. Observed maximum 0.0032 on the 0 to 255 scale; the bound is a
    # hundredth of a grey level, far below the rounding step the kernel is judged against.
    assert float((mine - ref).abs().max()) < 1e-2


def test_patch_index_mapping_equals_processor_layout():
    pe = _ext()
    img = torch.arange(3 * pe.OUT_H * pe.OUT_W, dtype=torch.float32).view(1, 3, pe.OUT_H, pe.OUT_W)
    lay = pe.reference_layout(img)
    assert tuple(lay.shape) == (pe.PATCHES_PER_IMAGE, pe.ROW_LEN)
    for y, x, c in ((0, 0, 0), (37, 100, 2), (319, 575, 1), (16, 32, 0)):
        bh, mh, py = y // 32, (y % 32) // 16, y % 16
        bw, mw, px = x // 32, (x % 32) // 16, x % 16
        row = ((bh * (pe.GRID_W // pe.MERGE) + bw) * pe.MERGE + mh) * pe.MERGE + mw
        for t in range(pe.TEMPORAL):
            col = ((c * pe.TEMPORAL + t) * pe.PATCH + py) * pe.PATCH + px
            assert float(lay[row, col]) == float(img[0, c, y, x])


def test_committed_preprocess_records_are_measured_dataset_runs():
    for p in (CASE / "runs").glob("preprocess_*.json"):
        text = p.read_text()
        assert '"input": "dataset"' in text and '"label": "measured"' in text, p.name
