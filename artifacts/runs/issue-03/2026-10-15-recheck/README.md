# 第三轮复核记录

2026-10-15，复核人：@Jaschenn。检查 PR #17 `b83fcb6b448719ab47e6f24df6260d74ad71b803` 与 PR #18 `99dea3c233172adbaf796171b6ed6b893b9c28ff`，两者依次合并到 main `07a0a02342da345cc3eb343b6006856a2bbd557f` 的临时副本 `/private/tmp/sb3-pr17-18-final-check`。没有合并远端 PR。

[主脚本输出](output.log)记录实际固定源码导入和环境版本；脚本与第二轮快照相同，本轮只修正文档。[联合测试结果](combined-tests.log)为 29 个针对性用例加 25 个上游用例，共 54 passed；其中 3 个上游组合直接返回，不代表全部执行了预测断言。

按第二轮 README 设置 SB3_SOURCE 与 SB3_PYTHON，在包含两个 PR 的合并副本中执行：

```bash
PYTHONPATH="$SB3_SOURCE" PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python "$SB3_PYTHON" python scripts/issue_03_predict_trace.py
PYTHONPATH="$SB3_SOURCE" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-project --python "$SB3_PYTHON" python -m pytest -q -p no:cacheprovider scripts/issue_03_contract_checks.py scripts/issue_04_contract_checks.py "$SB3_SOURCE/tests/test_predict.py::test_mixing_gym_vecenv_api" "$SB3_SOURCE/tests/test_train_eval_mode.py::test_predict_with_dropout_batch_norm" "$SB3_SOURCE/tests/test_train_eval_mode.py::test_a2c_ppo_collect_rollouts_with_batch_norm" "$SB3_SOURCE/tests/test_run.py::test_advantage_normalization" "$SB3_SOURCE/tests/test_run.py::test_ppo"
```
