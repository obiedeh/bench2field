"""Which code produced a report.

Every RunReport and sweep manifest records the git commit of the Bench2Field
checkout it ran from and whether that checkout had uncommitted changes, so
a number can always be traced to the exact code. `b2f sweep --expect-commit`
refuses to start on a checkout that is not the commit the operator meant to
run, which is the check to make before launching on a remote board.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent


def git_state(path: str | Path = PACKAGE_DIR) -> dict[str, Any]:
    """{"commit": full hash or None, "dirty": bool or None, "root": repo root or None}.

    None values mean the code is not running from a git checkout (an
    installed wheel, say) or git is not available; that is recorded, not hidden.
    """
    try:
        root = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, timeout=10)
        if root.returncode != 0:
            return {"commit": None, "dirty": None, "root": None}
        commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        status = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "root": None}
    return {"commit": commit or None, "dirty": bool(status.strip()), "root": root.stdout.strip() or None}


def check_expected_commit(expected: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Raise unless the checkout is exactly `expected` (a full hash or an
    unambiguous prefix) with no uncommitted changes. Returns the state."""
    state = state or git_state()
    commit = state.get("commit")
    if not commit:
        raise RuntimeError(f"expected commit {expected}, but this code is not running from a git checkout")
    if not commit.startswith(expected.lower()):
        raise RuntimeError(f"expected commit {expected}, but the checkout is at {commit[:12]}")
    if state.get("dirty"):
        raise RuntimeError(f"checkout is at {commit[:12]} as expected but has uncommitted changes")
    return state
