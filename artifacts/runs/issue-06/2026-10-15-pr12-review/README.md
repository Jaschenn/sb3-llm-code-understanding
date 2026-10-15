# PR #12 合并前复核运行

执行日期：2026-10-15。验证对象：`626423d41499a5a0d9bb4d610b6f130d4d686dc5`。
执行者：Codex，受 @Jaschenn 委托；未取得非提交者 GitHub 审批。管理员合并授权及例外原因见 [Issue #6 复核记录](../../../../docs/evidence/issues/issue-06.md#审批例外与合并授权)。

## 环境与实际命令

本次使用 Python 3.12.14、PyTorch 2.14.0、Gymnasium 1.3.0、NumPy 2.5.3、SB3 2.9.2a0；全部依赖按原快照固定，仅替换 SB3 本地源码路径。环境详情见 [system-info.json](system-info.json)，实际安装要求见 [requirements.txt](requirements.txt)。macOS 为 27.0.1 / arm64，与原记录 27.0 有差异。

上游本机目录 `/Users/jaschen/stable-baselines3` 的 HEAD 为固定 commit `7cfb4dd6055e74b5caa4ed4d6777492209946e26`，工作区干净；安装时构建 wheel，没有使用 PyPI 的其他 SB3 版本。

实际命令（原 PR 的快照位于 `/private/tmp/sb3-pr-review/pr12`）：

```bash
uv venv --python 3.12.14 /private/tmp/sb3-pr-review/pr12-validation-venv
uv pip install --python /private/tmp/sb3-pr-review/pr12-validation-venv/bin/python -r /private/tmp/sb3-pr-review/pr12-validation-requirements.txt
uv run --no-project --python /private/tmp/sb3-pr-review/pr12-validation-venv/bin/python python /private/tmp/sb3-pr-review/validate_pr12.py > /private/tmp/sb3-pr-review/pr12-validation.log 2>&1
```

[validate_pr12.py](validate_pr12.py) 是实际执行的验证器副本，内部通过独立子进程执行四个原始实验脚本，再比较输出。脚本保留当时的临时绝对路径；重跑时应先修改 `root`、`out` 路径，并将 requirements.txt 中的上游源码路径替换为自己的固定 commit 检出目录。它最后要求整条曲线逐值一致，本次该断言失败，退出码为 1。

另在上述 PR 快照根目录执行了以下检查，均通过：

```bash
uv run --no-project --python /private/tmp/sb3-pr-review/pr12-validation-venv/bin/python ruff check scripts/issue_06_cartpole_effect.py scripts/issue_06_predict.py scripts/issue_06_short_training.py scripts/issue_06_system_info.py
uv run --no-project --python /private/tmp/sb3-pr-review/pr12-validation-venv/bin/python ruff format --check scripts/issue_06_cartpole_effect.py scripts/issue_06_predict.py scripts/issue_06_short_training.py scripts/issue_06_system_info.py
```

## 结果与限制

四个实验脚本均正常退出。预测输出 [predict.json](predict.json) 和两轮短训练 [events.jsonl](events.jsonl) 与原始记录完全一致。效果实验的检查点、种子、回合奖励/步数和统计值一致性校验通过，但整条曲线精确比对未通过，原始验证日志保留于 [validation.log](validation.log)。

| 累计训练步数 | 原始运行均值 | 本次复核均值 |
|---:|---:|---:|
| 0 | 98.1 | 98.1 |
| 4096 | 84.9 | 84.9 |
| 8192 | 91.1 | 91.1 |
| 16384 | 455.5 | 455.5 |
| 32768 | 161.3 | 485.6 |

逐回合数据与计时见 [cartpole-effect.json](cartpole-effect.json)。本次未复现原记录最后的性能下降，不将两条轨迹合并，也不覆盖原始输出。原因未确定；对齐 Python 和依赖未消除差异，平台/硬件数值差异尚未排除。固定种子并不足以把任一单次性能轨迹推广为 PPO 的一般规律。
