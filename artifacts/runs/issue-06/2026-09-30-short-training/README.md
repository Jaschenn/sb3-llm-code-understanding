# 两轮 PPO 短训练

源码 commit：`7cfb4dd6055e74b5caa4ed4d6777492209946e26`。
运行版本：Python 3.12.14、PyTorch 2.14.0、Gymnasium 1.3.0、SB3 2.9.2a0；详见[环境信息](../2026-09-30-system-info/output.json)。
随机种子：42。参数：单环境、`n_steps=16`、`batch_size=16`、`n_epochs=1`、`total_timesteps=32`、CPU、单 PyTorch 线程。
实际命令（协作仓库根目录）：

```bash
UV_CACHE_DIR=/private/tmp/uv-cache-issue06 uv run --no-project python scripts/issue_06_short_training.py > artifacts/runs/issue-06/2026-09-30-short-training/events.jsonl
```

目的：在 `collect_rollouts()` 和 `train()` 边界记录时间步、buffer 位置/满状态和更新次数。关键输出见 [events.jsonl](events.jsonl)：每轮 `collect_end` 时 buffer 满，位置 16；随后 `train_start`，最后更新次数从 0 到 1、再从 1 到 2。最终时间步为 32，脚本对完整事件顺序和次数作断言。

`collect_start` 在调用父类方法前记录，因此第二轮入口仍显示上一轮的满 buffer；父类进入 `collect_rollouts()` 后会先重置 buffer。这一行不表示第二轮跳过采样。
