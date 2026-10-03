"""Retention over repeats: medians, spread, and the comparisons it refuses."""

import pytest

from bench2field import cli
from bench2field.metrics import MIN_REPEATS, field_retention, latency_stats, repeat_stat
from bench2field.schema import RunReport, TierResult
from test_core import BASE, OPT


def rep(variant, env, p95, nvpmodel=None, deadline_ms=33.3, drop_late=False, ort=None):
    lat = latency_stats([p95 * 0.8] * 94 + [p95] * 6)
    tier = TierResult(30.0, 30.0, 10.0, deadline_ms, 0, lat, {}, response=lat, drop_late=drop_late)
    platform = {k: v for k, v in (("nvpmodel", nvpmodel), ("onnxruntime", ort)) if v is not None}
    return RunReport(variant, env, [tier], platform=platform)


def group(variant, env, p95s, **kw):
    return [rep(variant, env, p, **kw) for p in p95s]


def four(bb=(9.0, 9.2, 8.9), bo=(3.0, 3.1, 2.9), fb=(12.0, 12.5, 11.8), fo=(6.0, 6.2, 5.9)):
    return (group(BASE, "bench-idle", bb), group(OPT, "bench-idle", bo),
            group(BASE, "field", fb), group(OPT, "field", fo))


def test_retention_uses_the_median_of_each_group_and_reports_the_spread():
    r = field_retention(*four(), 30.0)
    assert r.bench_speedup == pytest.approx(3.0, rel=1e-3)
    assert r.field_speedup == pytest.approx(2.0, rel=1e-3)
    assert r.retention == pytest.approx(0.5, rel=1e-3)
    g = r.groups["bench_baseline"]
    assert (g.n, g.median, g.lo, g.hi) == (3, pytest.approx(9.0), pytest.approx(8.9), pytest.approx(9.2))
    assert g.spread == pytest.approx(0.3)
    assert r.bench_gain_is_finding and r.field_gain_is_finding and r.warnings == []
    assert "3 repeats" in r.summary() and "range 8.900 to 9.200 ms" in r.summary()


def test_a_gain_smaller_than_the_spread_is_not_a_finding():
    r = field_retention(*four(bb=(10.0, 9.0, 11.0), bo=(9.5, 9.0, 10.5)), 30.0)
    assert r.bench_speedup > 1.0  # the medians do differ ...
    assert r.bench_gain_is_finding is False  # ... by less than the repeats disagree
    assert any("bench gain is no larger than the spread" in w for w in r.warnings)
    assert "WARNING" in r.summary()


def test_too_few_repeats_is_flagged_not_hidden():
    assert MIN_REPEATS == 3
    bb, bo, fb, fo = four()
    r = field_retention(bb[:2], bo, fb, fo, 30.0)
    assert any("fewer than 3 repeats (bench_baseline: 2)" in w for w in r.warnings)
    single = field_retention(bb[0], bo[0], fb[0], fo[0], 30.0)  # one run each still computes
    assert single.retention == pytest.approx(0.5, rel=1e-3)
    assert single.bench_gain_is_finding is None and single.warnings


def test_repeat_stat_on_one_run():
    s = repeat_stat([rep(BASE, "field", 12.0)], 30.0)
    assert (s.n, s.spread) == (1, 0.0) and s.median == pytest.approx(12.0)


def test_different_deadlines_are_refused():
    bb, bo, fb, fo = four()
    fo = group(OPT, "field", (6.0, 6.2, 5.9), deadline_ms=50.0)
    with pytest.raises(ValueError, match="differ in deadline"):
        field_retention(bb, bo, fb, fo, 30.0)


def test_different_drop_late_policies_are_refused():
    bb, bo, fb, fo = four()
    fb = group(BASE, "field", (12.0, 12.5, 11.8), drop_late=True)
    with pytest.raises(ValueError, match="differ in drop-late policy"):
        field_retention(bb, bo, fb, fo, 30.0)


def test_power_mode_must_match_between_baseline_and_optimized():
    bb = group(BASE, "bench-idle", (9.0, 9.2, 8.9), nvpmodel="NV Power Mode: 120W")
    bo = group(OPT, "bench-idle", (3.0, 3.1, 2.9), nvpmodel="NV Power Mode: MAXN")
    _, _, fb, fo = four()
    with pytest.raises(ValueError, match="bench baseline and optimized runs used different power modes"):
        field_retention(bb, bo, fb, fo, 30.0)


def test_power_mode_must_match_across_repeats():
    bb = [rep(BASE, "bench-idle", 9.0, nvpmodel="NV Power Mode: 120W"),
          rep(BASE, "bench-idle", 9.1, nvpmodel="NV Power Mode: 90W")]
    _, bo, fb, fo = four()
    with pytest.raises(ValueError, match="repeats of bench_baseline differ in power mode"):
        field_retention(bb, bo, fb, fo, 30.0)


def test_bench_and_field_on_different_boards_may_differ_in_power_mode_with_a_warning():
    bb = group(BASE, "bench-idle", (9.0, 9.2, 8.9), nvpmodel="NV Power Mode: 120W")
    bo = group(OPT, "bench-idle", (3.0, 3.1, 2.9), nvpmodel="NV Power Mode: 120W")
    fb = group(BASE, "field", (12.0, 12.5, 11.8), nvpmodel="NV Power Mode: 25W")
    fo = group(OPT, "field", (6.0, 6.2, 5.9), nvpmodel="NV Power Mode: 25W")
    r = field_retention(bb, bo, fb, fo, 30.0)
    assert r.retention == pytest.approx(0.5, rel=1e-3)
    assert any("bench and field used different power modes" in w for w in r.warnings)


def test_mixed_variants_or_environments_within_repeats_are_refused():
    bb, bo, fb, fo = four()
    with pytest.raises(ValueError, match="repeats of bench_baseline differ in variant"):
        field_retention([bb[0], bo[0]], bo, fb, fo, 30.0)
    with pytest.raises(ValueError, match="repeats of field_baseline differ in environment"):
        field_retention(bb, bo, [fb[0], bb[0]], fo, 30.0)
    with pytest.raises(ValueError, match="no runs"):
        field_retention([], bo, fb, fo, 30.0)


def test_cli_retention_takes_globs_and_lists(tmp_path, capsys):
    for name, g in zip(("bb", "bo", "fb", "fo"), four()):
        for i, r in enumerate(g):
            r.save(tmp_path / f"{name}_r{i}.json")
    args = [str(tmp_path / "bb_r*.json"), str(tmp_path / "bo_r*.json"), str(tmp_path / "fb_r*.json"),
            ",".join(str(tmp_path / f"fo_r{i}.json") for i in range(3))]
    assert cli.main(["retention", *args, "--hz", "30"]) == 0
    out = capsys.readouterr().out
    assert "retention 50%" in out and out.count("3 repeats") == 4
    assert cli.main(["retention", str(tmp_path / "nothing_*.json"), *args[1:], "--hz", "30"]) == 2
    assert "no reports match" in capsys.readouterr().err


def test_onnxruntime_version_gets_the_same_treatment_as_power_mode():
    bb = group(BASE, "bench-idle", (9.0, 9.2, 8.9), ort="1.30.0")
    bo = group(OPT, "bench-idle", (3.0, 3.1, 2.9), ort="1.30.0")
    fb = group(BASE, "field", (12.0, 12.5, 11.8), ort="1.24.0")
    fo = group(OPT, "field", (6.0, 6.2, 5.9), ort="1.24.0")
    r = field_retention(bb, bo, fb, fo, 30.0)  # 5090 bench vs Jetson field: allowed, flagged
    assert any("different onnxruntime versions ('1.30.0' vs '1.24.0')" in w for w in r.warnings)
    with pytest.raises(ValueError, match="field baseline and optimized runs used different onnxruntime"):
        field_retention(bb, bo, fb, group(OPT, "field", (6.0, 6.2, 5.9), ort="1.23.0"), 30.0)
    with pytest.raises(ValueError, match="repeats of bench_baseline differ in onnxruntime version"):
        field_retention(bb[:2] + group(BASE, "bench-idle", (9.1,), ort="1.29.0"), bo, fb, fo, 30.0)
