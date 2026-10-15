"""Require an actual clean checkout of the report's fixed SB3 source."""

import json
import platform
import subprocess
import sys
from pathlib import Path

BASELINE = "7cfb4dd6055e74b5caa4ed4d6777492209946e26"


def source_metadata(package_file):
    package = Path(package_file).resolve().parent

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(package), *args], text=True, stderr=subprocess.PIPE
        ).strip()

    try:
        root = Path(git("rev-parse", "--show-toplevel")).resolve()
        head = git("rev-parse", "HEAD")
        if package != root / "stable_baselines3":
            raise RuntimeError("import must come from the checkout, not site-packages")
        if head != BASELINE:
            raise RuntimeError(f"expected {BASELINE}, got {head}")
        # Inspect the complete checkout: changes to configuration also affect experiments.
        if git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("source checkout has uncommitted or untracked changes")
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        raise RuntimeError(f"SB3_BASELINE_REJECTED: {error}") from error
    return {
        "source_commit": head,
        "source_root": str(root),
        "sb3_import_path": str(Path(package_file).resolve()),
    }


def require_baseline():
    import gymnasium
    import numpy
    import stable_baselines3
    import torch

    metadata = source_metadata(stable_baselines3.__file__)
    metadata.update(
        {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        }
    )
    print("SB3_BASELINE_VERIFIED " + json.dumps(metadata, sort_keys=True), flush=True)
    return metadata
