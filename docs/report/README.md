# 实训报告写作骨架

## 1. 项目简介

项目定位、用户、问题、边界，以及选择 SB3 的原因。

## 2. 项目地图与阅读顺序

展示仓库目录、模块职责和从入口到核心模块的阅读路线。每项说明附代码证据。

## 3. 核心领域对象与类图

聚焦 `PPO`、`OnPolicyAlgorithm`、`BaseAlgorithm`、`BasePolicy`、`RolloutBuffer`、`VecEnv`、Gymnasium `Env` 与 `Callback`。标明继承、持有和调用关系。

## 4. 业务调用链

主链建议为 `env.reset() → model.predict() → policy → env.step()`；训练辅链为 `learn() → collect_rollouts() → RolloutBuffer → train()`。须包含至少一条失败路径。

## 5. 测试与边界

用现有测试反推环境契约、预测输入和保存加载等边界，并展示至少一个自建失败用例。

## 6. 证据矩阵与 AI 证据日志

链接 [证据矩阵](../evidence/evidence-matrix.md) 和 [AI 日志](../evidence/ai-log.md)。

## 7. 维护观察

选择一个 Issue/PR/commit，说明问题、设计取舍、代码或测试变化。

## 8. 学习反思

说明 AI 带来的效率、错误类型、验证手段和团队协作改进。
