# Issue #6 运行环境与复现入口

执行日期：2026-09-30。上游源码位于本机 `/Users/jas/stable-baselines3`，`git rev-parse HEAD` 为
`7cfb4dd6055e74b5caa4ed4d6777492209946e26`，与[项目基线](../../../docs/project-baseline.md)一致。
本协作仓库不复制上游源码。安装时从该目录构建 wheel，而非 editable 安装。

本机为 macOS 27.0、arm64；使用 `uv 0.12.19` 和 Python 3.12.14。
主要依赖为 PyTorch 2.14.0、Gymnasium 1.3.0、SB3 2.9.2a0、NumPy 2.5.3。
完整安装结果见 [pip-freeze.txt](pip-freeze.txt)；其中 SB3 的 `file://` 路径是本机路径，复核时需替换成自己的固定 commit 检出目录。

实际准备命令（在本协作仓库根目录）：

```bash
git -C /Users/jas/stable-baselines3 rev-parse HEAD
UV_CACHE_DIR=/private/tmp/uv-cache-issue06 uv venv --python 3.12 .venv
UV_CACHE_DIR=/private/tmp/uv-cache-issue06 uv pip install --python .venv/bin/python /Users/jas/stable-baselines3 ruff pytest
```

复核者先按项目基线检出上游 commit，再将安装命令中的上游绝对路径替换为自己的克隆目录。
`UV_CACHE_DIR` 只是本机缓存位置，可以改为任何可写目录。随后在协作仓库根目录运行以下四个目录中记录的命令。
实验脚本不保存模型文件；控制流实验训练 32 步，效果评估最多训练 32,768 步。

| 实验 | 记录 |
|---|---|
| 环境信息 | [2026-09-30-system-info](2026-09-30-system-info/README.md) |
| 一次预测和接口边界 | [2026-09-30-predict](2026-09-30-predict/README.md) |
| 两轮短训练 | [2026-09-30-short-training](2026-09-30-short-training/README.md) |
| CartPole 实际持续步数 | [2026-09-30-cartpole-effect](2026-09-30-cartpole-effect/README.md) |

同一随机种子与依赖版本有助于重跑，跨操作系统和硬件时动作值或数值日志仍可能不同；程序流程的复核以接口形状、异常类别、调用顺序和 buffer 状态为准，效果评估按逐回合步数查看。
