#!/usr/bin/env python
"""Issue #8 复核脚本：核对 PR #2174 在固定基线中的落点。

只读取 SB3 源码目录中的文本文件，不联网、不导入 SB3、不运行训练。
结论见 docs/evidence/issues/issue-08.md（C-08-01 ~ C-08-07）。

共 16 项检查。其中 V-08-14 ~ V-08-16 为 2026-10-15 复核后补入，用于机械校验两条被复核纠正的表述：
「告警与跳过只在 warn=True 下发生」以及「只检查直接子空间、不存在任意层级递归」。

用法::

    python scripts/issue_08_verify_chain.py --sb3-source /path/to/stable-baselines3
    python scripts/issue_08_verify_chain.py --sb3-source ../sb3-src \\
        --output-dir artifacts/runs/issue-08/2026-10-14-baseline-verification

若 SB3 目录是 Git 仓库且 HEAD 等于固定基线 commit，脚本会额外校验行号；
HEAD 不等于基线时只校验标记是否存在，并把该情况标为 WARN。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:  # Windows 控制台默认不是 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

PINNED_COMMIT = "7cfb4dd6055e74b5caa4ed4d6777492209946e26"

ENV_CHECKER = "stable_baselines3/common/env_checker.py"
TEST_FILE = "tests/test_env_checker.py"
CHANGELOG = "docs/misc/changelog.md"


@dataclass
class Check:
    cid: str
    conclusion: str
    description: str
    rel_path: str
    marker: str
    expected_line: int | None = None
    # 当结论是「某标记只应出现固定次数」时使用（例如：函数体内无递归调用）。
    expected_occurrences: int | None = None
    status: str = "PENDING"
    actual_line: int | None = None
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "check": self.cid,
            "conclusion": self.conclusion,
            "description": self.description,
            "file": self.rel_path,
            "marker": self.marker,
            "expected_line": self.expected_line,
            "expected_occurrences": self.expected_occurrences,
            "actual_line": self.actual_line,
            "status": self.status,
            "note": self.note,
        }


CHECKS: list[Check] = [
    Check(
        "V-08-01",
        "C-08-02",
        "新增的 sequence_space 标志被初始化（改动前的写法只有 graph_space）",
        ENV_CHECKER,
        "should_skip = graph_space = sequence_space = False",
        102,
    ),
    Check(
        "V-08-02",
        "C-08-04",
        "维护者新增的 OneOf 守卫函数存在",
        ENV_CHECKER,
        "def _is_oneof_space(space: spaces.Space) -> bool:",
        12,
    ),
    Check(
        "V-08-03",
        "C-08-04",
        "守卫函数内含 gym<1.0 的兼容分支（AttributeError 走 return False）",
        ENV_CHECKER,
        "# Gym < v1.0",
        20,
    ),
    Check(
        "V-08-04",
        "C-08-02",
        "Dict 分支内识别嵌套 Sequence",
        ENV_CHECKER,
        "            elif isinstance(space, spaces.Sequence):",
        110,
    ),
    Check(
        "V-08-05",
        "C-08-02",
        "Tuple 分支内识别嵌套 Sequence",
        ENV_CHECKER,
        "        # Check for Sequence spaces inside Tuple",
        139,
    ),
    Check(
        "V-08-06",
        "C-08-03",
        "OneOf 被单独处理（告警 + 置 should_skip）",
        ENV_CHECKER,
        "    if _is_oneof_space(observation_space):",
        147,
    ),
    Check(
        "V-08-07",
        "C-08-03",
        "OneOf 分支确实把 should_skip 置为 True",
        ENV_CHECKER,
        '            "Note: The checks for returned values are skipped."',
        150,
    ),
    Check(
        "V-08-08",
        "C-08-02",
        "Sequence 判断同时覆盖顶层与嵌套两种情形",
        ENV_CHECKER,
        "    if isinstance(observation_space, spaces.Sequence) or sequence_space:",
        156,
    ),
    Check("V-08-09", "C-08-05", "回归测试：Sequence 嵌在 Dict 的环境类", TEST_FILE, "class SequenceInDictEnv(CustomEnv):", 221),
    Check("V-08-10", "C-08-05", "回归测试：Sequence 嵌在 Tuple 的环境类", TEST_FILE, "class SequenceInTupleEnv(CustomEnv):", 231),
    Check("V-08-11", "C-08-05", "回归测试：Sequence 嵌在 OneOf 的环境类", TEST_FILE, "class SequenceInOneOfEnv(CustomEnv):", 239),
    Check("V-08-12", "C-08-03", "回归测试：OneOf 场景断言告警（C-08-03 的可复现落点）", TEST_FILE, "def test_check_env_oneof():", 266),
    Check(
        "V-08-13",
        "C-08-06",
        "变更日志条目存在（v2.7.1 Bug Fixes，署名 @copilot）",
        CHANGELOG,
        "Fixed env checker to properly handle `Sequence` observation spaces when nested inside composite spaces",
        158,
    ),
    # --- 以下 3 项由复核意见补入（2026-10-15） ---
    Check(
        "V-08-14",
        "C-08-03",
        "OneOf/Sequence 的告警与跳过只在 warn=True 下发生：调用点在 check_env 的 `if warn:` 分支内",
        ENV_CHECKER,
        "    if warn:\n        should_skip = _check_unsupported_spaces(env, observation_space, action_space)",
        501,
    ),
    Check(
        "V-08-15",
        "C-08-02",
        "无任意层级递归：`_check_unsupported_spaces` 在文件中只出现 2 次（定义 + check_env 内唯一调用点）",
        ENV_CHECKER,
        "_check_unsupported_spaces",
        None,
        expected_occurrences=2,
    ),
    Check(
        "V-08-16",
        "C-08-02",
        "Dict 分支遍历的是直接子空间（`observation_space.spaces.items()`），不深入下一层",
        ENV_CHECKER,
        "        for key, space in observation_space.spaces.items():",
        105,
    ),
]


def read_lines(root: Path, rel: str) -> tuple[list[str] | None, str]:
    path = root / rel
    if not path.is_file():
        return None, f"文件不存在：{rel}"
    try:
        return path.read_text(encoding="utf-8").splitlines(), ""
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace").splitlines(), ""


def git_head(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def run(root: Path) -> tuple[list[Check], dict]:
    head = git_head(root)
    head_is_baseline = head == PINNED_COMMIT
    context = {
        "sb3_source": str(root),
        "pinned_commit": PINNED_COMMIT,
        "git_head": head,
        "head_matches_baseline": head_is_baseline,
        "line_check_enabled": head_is_baseline,
    }

    for chk in CHECKS:
        lines, err = read_lines(root, chk.rel_path)
        if lines is None:
            chk.status = "FAIL"
            chk.note = err
            continue
        if "\n" in chk.marker:  # 多行标记：按整段文本定位起始行
            text = "\n".join(lines)
            idx = text.find(chk.marker)
            hits = [] if idx < 0 else [text[:idx].count("\n") + 1]
        else:
            hits = [i + 1 for i, text in enumerate(lines) if chk.marker in text]

        if chk.expected_occurrences is not None:
            chk.actual_line = hits[0] if hits else None
            if len(hits) == chk.expected_occurrences:
                chk.status = "PASS"
            else:
                chk.status = "FAIL"
                chk.note = (
                    f"出现次数不符：期望 {chk.expected_occurrences} 次，实际 {len(hits)} 次"
                )
            continue

        if not hits:
            chk.status = "FAIL"
            chk.note = "未找到标记"
            continue
        chk.actual_line = hits[0]
        if len(hits) > 1:
            chk.note = f"标记在该文件出现 {len(hits)} 次，取首次命中 L{hits[0]}"
        if head_is_baseline and chk.expected_line is not None:
            if chk.actual_line == chk.expected_line:
                chk.status = "PASS"
            else:
                chk.status = "FAIL"
                extra = f"行号不符：期望 L{chk.expected_line}，实际 L{chk.actual_line}"
                chk.note = f"{chk.note}；{extra}" if chk.note else extra
        else:
            chk.status = "PASS"
            if not head_is_baseline:
                extra = "标记存在（HEAD 非基线，未校验行号）"
                chk.note = f"{chk.note}；{extra}" if chk.note else extra
    return CHECKS, context


def to_markdown(checks: list[Check], context: dict) -> str:
    counts = {s: sum(1 for c in checks if c.status == s) for s in ("PASS", "FAIL")}
    overall = "PASS" if counts["FAIL"] == 0 else "FAIL"
    lines = [
        "# Issue #8 基线核对运行记录",
        "",
        f"- 结果：**{overall}**（PASS {counts['PASS']} / FAIL {counts['FAIL']}）",
        f"- SB3 源码目录：`{context['sb3_source']}`",
        f"- 固定基线 commit：`{context['pinned_commit']}`",
        f"- 实际 HEAD：`{context['git_head'] or '未检测到 Git 仓库'}`",
        f"- 行号校验：{'已启用（HEAD 等于基线）' if context['line_check_enabled'] else '未启用（HEAD 非基线，仅校验标记存在）'}",
        "",
        "| 检查 | 对应结论 | 文件 | 标记 | 期望行 | 实际行 | 状态 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in checks:
        marker = c.marker.strip()
        if len(marker) > 52:
            marker = marker[:49] + "..."
        lines.append(
            f"| {c.cid} | {c.conclusion} | `{c.rel_path}` | `{marker}` | "
            f"{'L' + str(c.expected_line) if c.expected_line else '-'} | "
            f"{'L' + str(c.actual_line) if c.actual_line else '-'} | {c.status} |"
        )
    notes = [f"- {c.cid}：{c.note}" for c in checks if c.note]
    if notes:
        lines += ["", "## 说明", ""] + notes
    lines += [
        "",
        "## 复现命令",
        "",
        "```bash",
        "python scripts/issue_08_verify_chain.py \\",
        f"  --sb3-source {context['sb3_source']} \\",
        "  --output-dir artifacts/runs/issue-08/2026-10-14-baseline-verification",
        "```",
        "",
        f"本记录对应的结论与链接见 `docs/evidence/issues/issue-08.md`；PR #2174 的 merge commit 为 `7883ed4d20bc9edf3a22a053c77b3146c518a937`。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="核对 PR #2174 在固定基线中的落点")
    parser.add_argument("--sb3-source", required=True, help="SB3 源码目录（应检出到固定基线 commit）")
    parser.add_argument("--output-dir", default=None, help="可选：写出 summary.md 与 results.json")
    args = parser.parse_args()

    root = Path(args.sb3_source).expanduser().resolve()
    if not root.is_dir():
        print(f"[FAIL] 目录不存在：{root}", file=sys.stderr)
        return 2

    checks, context = run(root)
    summary = to_markdown(checks, context)
    print(summary)

    failed = [c for c in checks if c.status == "FAIL"]
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.md").write_text(summary, encoding="utf-8")
        payload = {
            "context": context,
            "overall": "FAIL" if failed else "PASS",
            "checks": [c.as_dict() for c in checks],
        }
        (out / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[写入] {out / 'summary.md'}")
        print(f"[写入] {out / 'results.json'}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
