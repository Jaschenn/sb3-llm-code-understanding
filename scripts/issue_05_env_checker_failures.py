#!/usr/bin/env python
"""Issue #05 (T05)：环境检查与失败路径。

构造一系列不合法或处于边界的环境，对每个环境运行
``stable_baselines3.common.env_checker.check_env()``，记录实际异常类型与完整报错，
用于反推 Gymnasium 环境接口在 SB3 中的检查边界。

运行方式（在 SB3 固定基线的虚拟环境中）::

    cd stable-baselines3
    PYTHONPATH=. uv run --no-project --python .venv/bin/python python \
        /path/to/sb3-llm-code-understanding/scripts/issue_05_env_checker_failures.py

输出：
    artifacts/runs/issue-05/<日期>-env-checker-failures/
        - summary.md        人读结果汇总（每个用例：触发条件 → 实际结果）
        - results.json      机器可读结果（含版本信息与完整报错）

上游基线：DLR-RM/stable-baselines3 @ 7cfb4dd6055e74b5caa4ed4d6777492209946e26
本文档引用的 env_checker.py 行号均为该固定 commit 下的行号。
"""

from __future__ import annotations

import argparse
import datetime
import inspect
import json
import platform
import shlex
import subprocess
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3 import __version__ as SB3_VERSION
from stable_baselines3.common.env_checker import check_env

import torch

RUN_NAME = "env-checker-failures"
SOURCE_REF = "7cfb4dd6055e74b5caa4ed4d6777492209946e26"


class GoodEnv(gym.Env):
    """基线环境：满足 Gymnasium API 与 SB3 约定，check_env 应当通过。"""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.array([0.5, 0.5, 0.5], dtype=np.float32), {}

    def step(self, action):
        obs = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return obs, 1.0, False, False, {}


class NotAnEnv:
    """F01：没有继承 gymnasium.Env 的类。预期 check_env L482-484 断言失败。"""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        return np.array([0.5, 0.5, 0.5], dtype=np.float32), {}

    def step(self, action):
        return np.array([0.5, 0.5, 0.5], dtype=np.float32), 1.0, False, False, {}


class BadActionSpaceEnv(GoodEnv):
    """F02：action_space 不是 gymnasium.spaces.Space 实例。预期 L424-427 断言失败。"""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = [0, 1]  # list 而非 spaces.Discrete


class ResetWithoutSeedEnv(GoodEnv):
    """F03：reset() 不接受 seed 参数（旧版 Gym 签名）。预期 L493-496 抛 TypeError。"""

    def reset(self):  # type: ignore[override]
        return np.array([0.5, 0.5, 0.5], dtype=np.float32), {}


class OldApiStepEnv(GoodEnv):
    """F04：step() 返回 4 元组（旧版 Gym API）。预期 L367-370 断言失败。"""

    def step(self, action):  # type: ignore[override]
        obs = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return obs, 1.0, False, {}  # 4 个返回值


class ResetNoInfoEnv(GoodEnv):
    """F05：reset() 只返回 obs，不返回 (obs, info) 元组。预期 L337-338 断言失败。"""

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)


class BadRewardTypeEnv(GoodEnv):
    """F06：step() 返回的 reward 是字符串。预期 L401 断言失败。"""

    def step(self, action):  # type: ignore[override]
        obs = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return obs, "1.0", False, False, {}


class BadTerminatedTypeEnv(GoodEnv):
    """F07：step() 返回的 terminated 是 int 而非 bool。预期 L402 断言失败。"""

    def step(self, action):  # type: ignore[override]
        obs = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return obs, 1.0, 1, False, {}  # terminated 为 int


class ObsOutOfBoundsEnv(GoodEnv):
    """F08：reset() 返回的 obs 超出观测空间上界。预期 L285-303 断言失败（含逐索引明细）。"""

    def __init__(self) -> None:
        super().__init__()
        # 观测空间上界压到 [0, 1]，但 reset 返回 [0.5, 1.5, 0.5]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        return np.array([0.5, 1.5, 0.5], dtype=np.float32), {}


class SequenceObsEnv(GoodEnv):
    """W09：Sequence 观测空间（变长观测）。

    预期仅产生 UserWarning（L156-162），随后 check_env 跳过返回值检查并直接返回
    （L531-532）——边界：检查"通过"但返回值根本没有被验证。
    """

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Sequence(spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32))

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        # 故意返回完全错误的观测——验证返回值检查确实被跳过
        return "not-an-observation", {}

    def step(self, action):  # type: ignore[override]
        return "not-an-observation", 1.0, False, False, {}


class AsymmetricActionEnv(GoodEnv):
    """W10：Box 动作空间非对称且未归一化（[-2, 1]）。预期 L510-518 仅警告，不失败。"""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Box(low=-2.0, high=1.0, shape=(1,), dtype=np.float32)


class InfiniteBoundActionEnv(GoodEnv):
    """F11：Box 动作空间含无限边界。预期 L520-523 断言失败（warn=True 时才检查）。"""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Box(low=-np.inf, high=1.0, shape=(1,), dtype=np.float32)


# 用例注册表：(编号, 分类, 预期结果, env_checker.py 源码位置, 工厂, warn)
Case = tuple[str, str, str, str, Callable[[], Any], bool]
CASES: list[Case] = [
    ("F01", "A 环境类型", "AssertionError", "env_checker.py:L482-484", NotAnEnv, True),
    ("F02", "A 空间声明", "AssertionError", "env_checker.py:L424-427", BadActionSpaceEnv, True),
    ("F03", "B API 签名", "TypeError", "env_checker.py:L493-496", ResetWithoutSeedEnv, True),
    ("F04", "C 返回值结构", "AssertionError", "env_checker.py:L367-370", OldApiStepEnv, True),
    ("F05", "C 返回值结构", "AssertionError", "env_checker.py:L337-338", ResetNoInfoEnv, True),
    ("F06", "C 返回值结构", "AssertionError", "env_checker.py:L401", BadRewardTypeEnv, True),
    ("F07", "C 返回值结构", "AssertionError", "env_checker.py:L402", BadTerminatedTypeEnv, True),
    ("F08", "D 观测内容", "AssertionError", "env_checker.py:L285-303", ObsOutOfBoundsEnv, True),
    ("W09", "E 警告边界", "UserWarning + 跳过返回值检查", "env_checker.py:L156-162, L531-532", SequenceObsEnv, True),
    ("W10", "E 警告边界", "UserWarning，不失败", "env_checker.py:L510-518", AsymmetricActionEnv, True),
    ("F11", "E 警告边界", "AssertionError", "env_checker.py:L520-523", InfiniteBoundActionEnv, True),
    ("F12", "E 警告边界", "AssertionError，无警告", "env_checker.py:L500-502, L535", SequenceObsEnv, False),
]


def run_case(case: Case) -> dict[str, Any]:
    """对单个用例运行 check_env，捕获异常、警告和完整 traceback。"""
    case_id, category, expected, source_ref, factory, warn = case
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            check_env(factory(), warn=warn)
            outcome, exc_type, exc_message = "通过（未抛异常）", None, None
        except Exception as exc:  # noqa: BLE001 - 我们就是要捕获所有失败类型
            outcome = f"失败：{type(exc).__name__}"
            exc_type = type(exc).__name__
            exc_message = f"{exc}"
            tb = traceback.format_exc()
        else:
            tb = ""
    warn_messages = [str(w.message) for w in caught]
    return {
        "case_id": case_id,
        "category": category,
        "expected": expected,
        "source_ref": source_ref,
        "warn": warn,
        "outcome": outcome,
        "exception_type": exc_type,
        "exception_message": exc_message,
        "warnings": warn_messages,
        "traceback": tb,
    }


def source_metadata() -> dict[str, Any]:
    """验证实际导入的检查器来自固定 commit 的未修改源码。"""
    checker_source = Path(inspect.getfile(check_env)).resolve()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(checker_source.parent), *args],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()

    try:
        source_root = Path(git("rev-parse", "--show-toplevel")).resolve()
        if checker_source != source_root / "stable_baselines3/common/env_checker.py":
            raise RuntimeError(f"检查器并非 SB3 源码仓库文件：{checker_source}")
        actual_commit = git("rev-parse", "HEAD")
        if actual_commit != SOURCE_REF:
            raise RuntimeError(f"SB3 实际 commit {actual_commit} 与固定基线 {SOURCE_REF} 不符")
        if git("status", "--porcelain", "--untracked-files=normal", "--", str(source_root / "stable_baselines3")):
            raise RuntimeError(f"SB3 源码有未提交修改：{source_root}")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(
            f"无法验证 SB3 源码基线（实际导入：{checker_source}）；"
            "请 checkout 固定 commit，并通过 PYTHONPATH 指向该源码仓库"
        ) from exc
    return {
        "sb3_commit": actual_commit,
        "sb3_source_root": str(source_root),
        "env_checker_source": str(checker_source),
        "sb3_source_clean": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="独立运行目录；默认使用当日实验目录")
    args = parser.parse_args()
    source = source_metadata()  # 先验证来源，失败时不生成带有错误基线的运行记录
    script_dir = Path(__file__).resolve().parent
    out_root = script_dir.parent / "artifacts" / "runs" / "issue-05"
    date_tag = datetime.date.today().isoformat()
    out_dir = args.output_dir or out_root / f"{date_tag}-{RUN_NAME}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [run_case(case) for case in CASES]

    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": shlex.join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
        **source,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gymnasium": gym.__version__,
        "numpy": np.__version__,
        "stable_baselines3": SB3_VERSION,
        "seed_policy": "check_env 内部固定使用 reset(seed=0) 与 action_space.sample()；"
        "本实验的失败均为确定性触发，不依赖随机种子",
    }

    (out_dir / "results.json").write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# Issue #05 运行记录：{out_dir.name}",
        "",
        f"- 运行时间：{meta['date']}",
        f"- SB3 实际 commit：`{meta['sb3_commit']}`（已核对固定基线，源码无未提交修改）",
        f"- 实际检查器路径：`{meta['env_checker_source']}`",
        f"- Python：{sys.version.split()[0]}｜PyTorch：{torch.__version__}｜"
        f"Gymnasium：{gym.__version__}｜numpy：{np.__version__}｜SB3(本地源码版本)：{SB3_VERSION}",
        f"- 运行命令：`{meta['command']}`",
        f"- 工作目录 / PYTHONPATH 源码根目录：`{meta['cwd']}` / `{meta['sb3_source_root']}`",
        f"- 平台：{meta['platform']}",
        "",
        "| 用例 | 分类 | warn | 预期 | 实际结果 | 源码位置 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        actual = r["exception_type"] or ("通过" + (f"（{len(r['warnings'])} 条警告）" if r["warnings"] else ""))
        lines.append(
            f"| {r['case_id']} | {r['category']} | {r['warn']} | {r['expected']} | {actual} | `{r['source_ref']}` |"
        )
    lines += ["", "## 完整报错与警告", ""]
    for r in results:
        lines.append(f"### {r['case_id']}（{r['category']}，预期 {r['expected']}）")
        lines.append("")
        if r["exception_message"]:
            lines.append("```")
            lines.append(str(r["exception_message"]))
            lines.append("```")
        for w in r["warnings"]:
            lines.append(f"> 警告：{w}")
        if not r["exception_message"] and not r["warnings"] and r["outcome"].startswith("通过"):
            lines.append("（无异常、无警告）")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"完成：{len(results)} 个用例，结果写入 {out_dir}")
    for r in results:
        print(f"  {r['case_id']}: {r['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
