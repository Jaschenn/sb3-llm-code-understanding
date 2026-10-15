"""Record the runtime used by issue #6.

Run: uv run --no-project python scripts/issue_06_system_info.py
Purpose: capture versions and the imported SB3 location before other experiments.
Expected: Python, PyTorch, Gymnasium and SB3 versions plus platform details.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import gymnasium
import stable_baselines3
import torch


def main() -> None:
    details: dict[str, str] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_import_path": str(Path(stable_baselines3.__file__).resolve()),
    }
    print(json.dumps(details, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
