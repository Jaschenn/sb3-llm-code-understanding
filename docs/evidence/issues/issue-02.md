# Issue #2：核心对象与领域类图

负责人：@Jaschenn
复核人：待指定（需非提交者）
上游基线：见 [project-baseline.md](../../project-baseline.md)

本文件对应[报告第 3 章](../../report/03-domain-objects.md)。原 PR #15 已关闭未合并；这里重新从固定基线核对。图源：[domain-inheritance.mmd](../../../artifacts/diagrams/issue-02/domain-inheritance.mmd)（继承与方法）、[domain-objects.mmd](../../../artifacts/diagrams/issue-02/domain-objects.mmd)（持有、调用与属性）。

## 可检验结论

| 本 Issue 内编号 | 结论 | 状态 | 证据链接 |
|---|---|---|---|
| C-02-01 | `OnPolicyAlgorithm`、`OffPolicyAlgorithm` 继承 `BaseAlgorithm`，`PPO` 继承 `OnPolicyAlgorithm`；PPO 的 `MlpPolicy` 是 `ActorCriticPolicy` 别名 | 待验证（仅源码） | 类定义行见第 3 章 3.1 表；[ppo/policies.py#L5](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/policies.py#L5) |
| C-02-02 | `ActorCriticPolicy` 继承 `BasePolicy`；`RolloutBuffer`、`ReplayBuffer` 继承 `BaseBuffer`；`DummyVecEnv` 继承 `VecEnv` | 待验证（仅源码） | [policies.py#L416](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L416)、[buffers.py#L343](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/buffers.py#L343)、[dummy_vec_env.py#L15](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/vec_env/dummy_vec_env.py#L15) |
| C-02-03 | 算法在 `_setup_model` 中创建并持有 buffer 与 policy；on-policy 持有 `rollout_buffer`，off-policy 持有 `replay_buffer` | 待验证（仅源码） | [on_policy_algorithm.py#L125-L138](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L125)、[off_policy_algorithm.py#L208-L224](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/off_policy_algorithm.py#L208) |
| C-02-04 | `_wrap_env` 在需要时先加 `Monitor` 再用 `DummyVecEnv` 包装；`DummyVecEnv` 持有环境列表 `envs` | 待验证（仅源码） | [base_class.py#L218-L224](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L218)、[dummy_vec_env.py#L31](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/vec_env/dummy_vec_env.py#L31) |
| C-02-05 | 回调是 `learn()` 的参数而非算法属性：`init_callback(model)` 让回调拿到算法引用，采样中调用 `on_step` 等钩子；采样用 `policy.forward`，用户 `predict()` 不进入 `forward` | 待验证（仅源码；predict 不进 forward 部分见 C-03-05，已有运行证据） | [callbacks.py#L68-L74](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/callbacks.py#L68)、[on_policy_algorithm.py#L192-L266](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L192)、[#L202](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L202) |

## 证据记录

行号由 Claude 在固定 commit 的本地检出上用 `grep -n` 与 `sed -n` 逐条核对，非凭记忆。没有运行实验：这些是结构性陈述，按 CONTRIBUTING 规定缺第二类独立证据时标“待验证”。复核人可在安装 SB3 后用 `issubclass(PPO, OnPolicyAlgorithm)`、`model.rollout_buffer`、`model.policy`、`type(model.env)` 补充运行证据。

## AI 使用记录

| AI 编号 | 模型与提问原文/链接 | AI 回答摘要 | 验证动作 | 观察结果 | 修正后的表述 |
|---|---|---|---|---|---|
| AI-02-01 | Claude（Claude Code），用户请求：补充完善报告并根据协作记录填写各章 | 给出对象职责表、类图与关系依据 | 在固定 commit 检出上逐条 `grep -n` 核对类定义与属性赋值行；核对中修正了初稿的行号（`policy(obs)` 由 L200-L202 改为 L202），并补充了此前缺失的引用（`reset()` L187、`DummyVecEnv.envs` L31） | 初稿行号有偏差、引用不全，已改正 | 见报告第 3 章 3.3 节 |

## 复核记录

待复核人填写。
