"""Record the environment case study 02 runs in, as one committed JSON file.

Every result file in this case study points at one of these records, so a
reader can tell exactly which driver, CUDA runtime, torch, transformers,
flash-attn and Alpamayo package commit produced a number, and what else the
machine was doing at the time. The record is written once per board after
setup and again whenever anything in the environment changes.

What it records:

    board            the board label (bench-5090, bench-thor; never a hostname)
    captured_at      UTC timestamp
    os, python       kernel release, machine type and interpreter version
    gpu              name, driver, compute capability, total memory, via torch and NVML
    packages         torch (with its CUDA runtime), torchvision, transformers,
                     flash_attn, physical_ai_av, alpamayo_r1 (version and the git
                     commit it was installed from), bench2field (version and commit)
    attention        which attention implementation flash_attn makes available
    resources        system RAM and free disk under the Hugging Face cache
    background       bench2field.background.snapshot(): containers, load average
                     and the busiest processes reduced to their executable names

Hostnames, LAN addresses and home directory paths are never written: the
only identifier is the board label, and any absolute path under the user's
home is shortened to "~".

    env -u PYTHONPATH .venv/bin/python capture_env.py --board bench-5090
    # writes runs/env_bench-5090.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BOARDS = ("bench-5090", "bench-thor", "field-orin")


def _package(name: str, dist: str | None = None) -> dict[str, Any]:
    """Version and, when installed from git, the commit of one package."""
    info: dict[str, Any] = {"installed": False}
    try:
        d = md.distribution(dist or name)
    except md.PackageNotFoundError:
        return info
    info.update(installed=True, version=d.version)
    direct = d.read_text("direct_url.json")
    if direct:
        try:
            url = json.loads(direct)
            info["source"] = url.get("url")
            commit = (url.get("vcs_info") or {}).get("commit_id")
            if commit:
                info["commit"] = commit
        except json.JSONDecodeError:
            pass
    return info


def _gpu() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import torch

        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            out.update(
                name=props.name,
                compute_capability=f"{props.major}.{props.minor}",
                total_memory_gib=round(props.total_memory / 2**30, 2),
                multiprocessors=props.multi_processor_count,
            )
    except Exception as exc:  # torch missing or broken: say so, do not guess
        out["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import pynvml

        pynvml.nvmlInit()
        out["driver"] = pynvml.nvmlSystemGetDriverVersion()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        out["nvml_name"] = pynvml.nvmlDeviceGetName(h)
        pynvml.nvmlShutdown()
    except Exception as exc:
        out["nvml_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _attention() -> dict[str, Any]:
    out: dict[str, Any] = {"flash_attn_importable": False}
    try:
        fa = importlib.import_module("flash_attn")
        out.update(flash_attn_importable=True, flash_attn_version=fa.__version__)
    except Exception as exc:
        out["flash_attn_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _resources() -> dict[str, Any]:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    probe = hf_home if hf_home.exists() else Path.home()
    usage = shutil.disk_usage(probe)
    out: dict[str, Any] = {
        "disk_free_gib_under_hf_cache": round(usage.free / 2**30, 1),
        "disk_total_gib_under_hf_cache": round(usage.total / 2**30, 1),
    }
    try:
        pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        out["ram_total_gib"] = round(pages / 2**30, 1)
    except (ValueError, OSError):
        pass
    return out


def _background() -> dict[str, Any] | None:
    """bench2field's background snapshot, with each process reduced to the
    name of its executable: a public record should say that a simulator or a
    browser was running, not which private project it was running from."""
    try:
        from bench2field import background

        snap = background.snapshot()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    procs = []
    for proc in snap.get("top_processes") or []:
        cmd = str(proc.get("cmd", "")).split()
        exe = Path(cmd[0]).name if cmd else ""
        procs.append({**{k: v for k, v in proc.items() if k != "cmd"}, "exe": exe})
    snap["top_processes"] = procs
    return snap


def _scrub(obj: Any, home: str) -> Any:
    """Replace the home directory in every string with "~"."""
    if isinstance(obj, str):
        return obj.replace(home, "~")
    if isinstance(obj, dict):
        return {k: _scrub(v, home) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v, home) for v in obj]
    return obj


def capture(board: str) -> dict[str, Any]:
    """The full environment record for `board`."""
    torch_info = _package("torch")
    try:
        import torch

        torch_info["cuda_runtime"] = torch.version.cuda
    except Exception:
        pass
    record = {
        "schema": "bench2field.case_study_02.env/1",
        "board": board,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": platform.python_version(),
        "gpu": _gpu(),
        "packages": {
            "torch": torch_info,
            "torchvision": _package("torchvision"),
            "transformers": _package("transformers"),
            "flash_attn": _package("flash_attn", "flash-attn"),
            "physical_ai_av": _package("physical_ai_av", "physical-ai-av"),
            "alpamayo_r1": _package("alpamayo_r1", "alpamayo-r1"),
            "bench2field": _package("bench2field"),
        },
        "attention": _attention(),
        "resources": _resources(),
        "background": _background(),
    }
    try:
        from bench2field import provenance

        record["packages"]["bench2field"]["commit"] = provenance.git_state()["commit"]
        record["packages"]["bench2field"]["dirty"] = provenance.git_state()["dirty"]
    except Exception:
        pass
    return _scrub(record, str(Path.home()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", required=True, choices=BOARDS)
    ap.add_argument("--out", type=Path, default=None, help="default runs/env_<board>.json")
    args = ap.parse_args(argv)
    out = args.out or HERE / "runs" / f"env_{args.board}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = capture(args.board)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    g = record["gpu"]
    print(f"{args.board}: {g.get('name')} driver {g.get('driver')} "
          f"torch {record['packages']['torch'].get('version')} "
          f"transformers {record['packages']['transformers'].get('version')} "
          f"flash_attn {record['packages']['flash_attn'].get('version')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
