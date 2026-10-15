# T04 复核与修正后的运行

2026-10-15，复核人：@Jaschenn。原提交 `7481cf67b48367de3c147e1e973b16e4f91fc5b3` 的运行材料保留于相邻的 2026-10-09 目录；本目录保存修正版脚本快照与完整输出。

环境：macOS 27.0.1 / arm64，Python 3.12.14，PyTorch 2.14.0，Gymnasium 1.3.0，NumPy 2.5.3，SB3 2.9.2a0。SB3 由干净的上游检出 `7cfb4dd6055e74b5caa4ed4d6777492209946e26` 构建 wheel；安装方式同[基线准备](../../../../docs/project-baseline.md)，本次用 wheel 替代 editable。CPU 单线程，seed=0，n_steps=32，batch_size=16，n_epochs=2，total_timesteps=128。

## 命令

在本 PR 根目录执行，`SB3_PYTHON` 指向已安装固定基线 SB3 与依赖的解释器（本次为 `/private/tmp/sb3-pr-review/pr12-validation-venv/bin/python`），`SB3_SOURCE` 为对应上游检出目录（本次 `/Users/jaschen/stable-baselines3`）：

```bash
uv run --no-project --python "$SB3_PYTHON" python scripts/issue_04_learn_trace.py
uv run --no-project --python "$SB3_PYTHON" ruff check scripts/issue_04_learn_trace.py
uv run --no-project --python "$SB3_PYTHON" ruff format --check scripts/issue_04_learn_trace.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-project --python "$SB3_PYTHON" python -m pytest -q -p no:cacheprovider "$SB3_SOURCE/tests/test_train_eval_mode.py::test_a2c_ppo_collect_rollouts_with_batch_norm" "$SB3_SOURCE/tests/test_run.py::test_advantage_normalization" "$SB3_SOURCE/tests/test_run.py::test_ppo"
uv run --no-project --python "$SB3_PYTHON" python artifacts/runs/issue-04/2026-10-15-review/check_disabled_training.py scripts/issue_04_learn_trace.py
```

实际运行在临时隔离检出 `/private/tmp/sb3-pr18-review-fix` 完成；标准输出保存到 `/private/tmp/sb3-pr-review/pr18-fixed-output.log`、`pr18-upstream-tests.log`、`pr18-disabled-training.log` 后复制到本目录。反向验证时使用的脚本位于 `/private/tmp/sb3-pr-review/check_disabled_training.py`，此处保留其副本。

## 结果

- [修正版脚本](issue_04_learn_trace.py)执行成功，见 [output.log](output.log)。
- reset 重新分配 observations 且归零，pos/full/generator_ready 重置的断言通过。
- 完整 callback 序列、32/64/96/128 的 train 边界与 buffer 满状态断言通过；最终总步数 128、`_n_updates=8`。
- learn 前后有 **12 个参数张量变化**。恢复 learn 前快照的评估均分 222.3，训练后 79.0，差值 **-143.3**；参数更新不保证回报提升。
- 上游相关测试 **14 passed**，见 [upstream-tests.log](upstream-tests.log)。Ruff 0.16.9 lint 与格式检查通过。
- [反向验证](check_disabled_training.py)模拟只增加更新计数、完全不修改参数的 train；参数断言应拒绝该情形，输出见 [disabled-training.log](disabled-training.log)。
- Mermaid CLI 12.0.0 已将修正后的训练图渲染为临时 SVG，语法通过；本次通过 `-p` 指定本机 Chrome 的 Puppeteer 配置。命令：`pnpm dlx @mermaid-js/mermaid-cli@12.0.0 -i artifacts/diagrams/issue-04/learn-call-chain.mmd -o /private/tmp/sb3-pr-review/pr18-diagram.svg -p /private/tmp/sb3-pr-review/puppeteer.json -b white`。

KL 早停路径按固定源码核对，本次默认 target_kl=None，未声称本运行触发 KL 早停。callback False 分支也仅作源码边界说明。
