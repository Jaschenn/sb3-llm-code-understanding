# Issue #8 基线核对运行记录

- 结果：**PASS**（PASS 13 / FAIL 0）
- SB3 源码目录：`C:\Users\Yang\WorkBuddy\2026-10-14-16-39-25\sb3-src`
- 固定基线 commit：`7cfb4dd6055e74b5caa4ed4d6777492209946e26`
- 实际 HEAD：`7cfb4dd6055e74b5caa4ed4d6777492209946e26`
- 行号校验：已启用（HEAD 等于基线）

| 检查 | 对应结论 | 文件 | 标记 | 期望行 | 实际行 | 状态 |
|---|---|---|---|---|---|---|
| V-08-01 | C-08-02 | `stable_baselines3/common/env_checker.py` | `should_skip = graph_space = sequence_space = False` | L102 | L102 | PASS |
| V-08-02 | C-08-04 | `stable_baselines3/common/env_checker.py` | `def _is_oneof_space(space: spaces.Space) -> bool:` | L12 | L12 | PASS |
| V-08-03 | C-08-04 | `stable_baselines3/common/env_checker.py` | `# Gym < v1.0` | L20 | L20 | PASS |
| V-08-04 | C-08-02 | `stable_baselines3/common/env_checker.py` | `elif isinstance(space, spaces.Sequence):` | L110 | L110 | PASS |
| V-08-05 | C-08-02 | `stable_baselines3/common/env_checker.py` | `# Check for Sequence spaces inside Tuple` | L139 | L139 | PASS |
| V-08-06 | C-08-03 | `stable_baselines3/common/env_checker.py` | `if _is_oneof_space(observation_space):` | L147 | L147 | PASS |
| V-08-07 | C-08-03 | `stable_baselines3/common/env_checker.py` | `"Note: The checks for returned values are skipped."` | L150 | L150 | PASS |
| V-08-08 | C-08-02 | `stable_baselines3/common/env_checker.py` | `if isinstance(observation_space, spaces.Sequence)...` | L156 | L156 | PASS |
| V-08-09 | C-08-05 | `tests/test_env_checker.py` | `class SequenceInDictEnv(CustomEnv):` | L221 | L221 | PASS |
| V-08-10 | C-08-05 | `tests/test_env_checker.py` | `class SequenceInTupleEnv(CustomEnv):` | L231 | L231 | PASS |
| V-08-11 | C-08-05 | `tests/test_env_checker.py` | `class SequenceInOneOfEnv(CustomEnv):` | L239 | L239 | PASS |
| V-08-12 | C-08-03 | `tests/test_env_checker.py` | `def test_check_env_oneof():` | L266 | L266 | PASS |
| V-08-13 | C-08-06 | `docs/misc/changelog.md` | `Fixed env checker to properly handle `Sequence` o...` | L158 | L158 | PASS |

## 说明

- V-08-07：标记在该文件出现 3 次，取首次命中 L150

## 复现命令

```bash
python scripts/issue_08_verify_chain.py \
  --sb3-source C:\Users\Yang\WorkBuddy\2026-10-14-16-39-25\sb3-src \
  --output-dir artifacts/runs/issue-08/2026-10-14-baseline-verification
```

本记录对应的结论与链接见 `docs/evidence/issues/issue-08.md`；PR #2174 的 merge commit 为 `7883ed4d20bc9edf3a22a053c77b3146c518a937`。
