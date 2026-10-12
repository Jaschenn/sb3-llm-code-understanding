# PR #19 修复验证

本目录保留复核修复后的 12 用例运行；原始 11 用例和原始 pytest 输出保留在相邻的 `2026-10-12-env-checker-failures/`。

## 实验与来源校验

- 固定基线的干净源码：实际导入路径和 Git HEAD 验证通过。
- 12 个实验：异常类型与预期一致；W09/W10 各 1 条警告；F12 无警告且拒绝非法 Sequence 观测。
- 预期 commit 与实际 HEAD 不符：拒绝。
- 在实际导入的 `env_checker.py` 中添加未提交修改：拒绝；验证后恢复原始内容。
- 不设置 PYTHONPATH，使用本地 PyPI SB3 2.9.0：拒绝，退出码 1，指定的输出目录未创建。

## 上游测试

命令（运行目录 `/private/tmp/pr19-fix`）：

```bash
PYTHONPATH=/tmp/pr19-sb3-baseline uv run --no-project \
  --python /Users/jaschen/sb3-llm-code-understanding/intro/.venv/bin/python \
  --with pytest python -m pytest /tmp/pr19-sb3-baseline/tests/test_env_checker.py \
  -v --override-ini addopts=
```

结果：**25 passed in 0.94s**。Python 3.13.5，pytest 9.1.1；SB3 源码为固定 commit `7cfb4dd6055e74b5caa4ed4d6777492209946e26`，运行依赖版本见本目录 `results.json`。
