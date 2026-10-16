# 1. 项目简介

## 1.1 SB3 是什么

Stable Baselines3（SB3）是一组基于 PyTorch 的强化学习算法实现，官方 README 把它定位为 Stable Baselines 的下一个主版本，目标是让研究者和工业界更容易复现、改进和比较强化学习方法（[README](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/README.md#L11)）。使用者是已经了解强化学习基础、希望直接调用成熟实现的人；README 明确提醒库“假定你有一定强化学习知识”（[README#L18](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/README.md#L18)）。

固定基线下，包版本为 `2.9.2a0`，要求 Python ≥ 3.10、`gymnasium>=0.29.1,<2.0`、`torch>=2.8,<3.0`（[setup.py#L82-L84](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/setup.py#L82)、[#L140](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/setup.py#L140)）。顶层 `__init__.py` 导出 A2C、DDPG、DQN、PPO、SAC、TD3 六个算法（[__init__.py#L1-L9](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/__init__.py#L1)）；`HER` 已改为一个会抛出 ImportError 的占位函数，对应能力是 `HerReplayBuffer`（[#L18-L22](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/__init__.py#L18)）。

README 的算法表还列了 ARS、CrossQ、TQC、TRPO、RecurrentPPO 等，但注脚说明它们属于 [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) 或 SBX，**不在本仓库的包内**（[README#L203-L205](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/README.md#L203)）。本报告只讨论包内的六个算法，避免把表格里的名字误当成基线中的代码。

## 1.2 项目的典型使用方式

用户提供符合 Gymnasium 接口的环境，选择算法与策略名，调用 `learn()` 训练，调用 `predict()` 得到动作。示例见[第 2 章 2.3 节](02-project-map.md#23-关键入口)。SB3 的卖点包括：统一的算法接口、自定义环境与策略、`Dict` 观测支持、回调、TensorBoard 支持和较高的测试覆盖（[README Main Features](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/README.md#L20)）。

## 1.3 为什么选它作为代码理解对象

这是小组的选择理由，不是官方目标：

- 包体量适中（64 个 Python 文件、15,693 行，见 [C-01-01](../evidence/evidence-matrix.md)），一个小组能在课程周期内读完主干。
- 层次清晰：公共基类、中间层、具体算法，加上 Policy、Buffer、VecEnv、Callback、Logger 等可单独阅读的组件（第 2、3 章）。
- 有可运行的最小例子和配套测试，便于“读源码 → 写假设 → 运行验证”（第 5 章）。
- 有公开的 Issue/PR 历史，可以观察真实维护过程（第 7 章）。

## 1.4 研究范围与边界

主问题：**SB3 如何将一个符合 Gymnasium 接口的环境，组织为可验证的预测和训练软件流程？**

范围内：对象协作、控制流、接口约束与失败路径、测试与运行证据、维护历史、以及大语言模型（LLM）在理解过程中的作用与错误。

范围外：不证明 PPO 的数学收敛性，不推导损失函数；不评估算法的最终性能（第 5 章仅报告一次固定种子的 CartPole 轨迹，且明确限定）；不覆盖 SB3 Contrib 与 SBX；不分析 GPU、多进程（`SubprocVecEnv`）和图像观测的专用路径。

## 1.5 方法与证据规则

- **固定基线**：所有源码引用都使用同一个上游 commit `7cfb4dd6055e74b5caa4ed4d6777492209946e26`（基线记录日期 2026-09-29），见[基线说明](../project-baseline.md)。
- **两类独立证据**：重要结论需要源码、测试、运行、Issue/PR 历史中的至少两类；只有一类的标“待验证”（[CONTRIBUTING](../../CONTRIBUTING.md)）。
- **AI 回答是假设**：LLM 输出必须由源码、测试或运行核验后才能引用；记录见[第 6 章](06-ai-evidence.md)与 [AI 日志](../evidence/ai-log.md)。
- **证据可回溯**：每章结论指向[证据矩阵](../evidence/evidence-matrix.md)，再到各 Issue 的原始证据文件、脚本和运行记录。
- **状态词**：`待验证`、`已支持`、`已否定`、`需限定`。

## 1.6 报告结构

| 章 | 内容 |
|---|---|
| 2 项目地图 | 仓库入口、模块职责、推荐阅读顺序 |
| 3 领域对象 | 核心对象及继承/持有/调用关系 |
| 4 调用链 | `predict()` 与 `learn()` 的端到端流程 |
| 5 测试与边界 | 环境契约、失败路径、实际运行与效果 |
| 6 AI 证据日志 | 提问、核验与纠错过程 |
| 7 维护观察 | 一次完整的 Issue→PR→合并案例及对照案例 |
| 8 学习反思 | AI 的帮助与风险、协作过程的得失 |
