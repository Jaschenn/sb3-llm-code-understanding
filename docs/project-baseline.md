# 上游源码基线

本报告的源码定位和图表统一以以下固定版本为准。若需更新基线，先在 Issue 中讨论，再统一更新本文件与受影响的证据。

| 项目 | 固定值 |
|---|---|
| 上游仓库 | https://github.com/DLR-RM/stable-baselines3 |
| 分析 commit | `7cfb4dd6055e74b5caa4ed4d6777492209946e26` |
| 基线记录日期 | 2026-09-29 |
| 本地源码 | 小组成员各自克隆到自己的目录；本协作仓库不复制上游源码 |

源码引用示例：`stable_baselines3/common/base_class.py:L67`，并提供固定 commit 链接：

`https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L67`

复现准备：

```bash
git clone https://github.com/DLR-RM/stable-baselines3.git
cd stable-baselines3
git checkout 7cfb4dd6055e74b5caa4ed4d6777492209946e26
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[tests]'
```

上表固定的是源码。每个实验仍须在自己的运行目录记录操作系统、Python、PyTorch、Gymnasium、SB3 版本、实际命令、随机种子和输出。课程材料与依赖版本若需要统一，再由 T06 提出并更新本文件。
