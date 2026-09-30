# Issue #6：实际运行与可复现实验

负责人：@Jaschenn

复核人：待 PR 中指定非提交者

上游基线：[project-baseline.md](../../project-baseline.md)，commit `7cfb4dd6055e74b5caa4ed4d6777492209946e26`。

## 可检验结论

| 编号 | 结论 | 状态 | 证据 |
|---|---|---|---|
| C-06-01 | 原生 Gymnasium `reset()` 的 `(observation, info)` 应先解包，再把 observation 交给 `PPO.predict()`；原生 `step()` 返回五项 | 已支持 | [预测输出](../../../artifacts/runs/issue-06/2026-09-30-predict/output.json)、[BaseAlgorithm.predict 源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L537)、[BasePolicy.predict 源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L331) |
| C-06-02 | SB3 包装出的 `DummyVecEnv` 使用批量 observation/action，`reset()` 只返回 observation，`step()` 返回四项；混传 Gymnasium reset 元组会报错 | 已支持 | [预测输出](../../../artifacts/runs/issue-06/2026-09-30-predict/output.json)、[包装源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L204)、[DummyVecEnv 源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/vec_env/dummy_vec_env.py#L56) |
| C-06-03 | 在这组短训练参数下，每次完成 16 步采样并填满 RolloutBuffer 后才调用 `train()`；两轮共 32 步、2 次更新 | 已支持，限定于本次运行与固定基线 | [事件日志](../../../artifacts/runs/issue-06/2026-09-30-short-training/events.jsonl)、[训练循环源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L300)、[buffer 写入源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L247) |
| C-06-04 | 原生 CartPole 通过本次 `check_env()`；这不足以证明任意自定义环境正确 | 已支持，但需限定 | [预测输出](../../../artifacts/runs/issue-06/2026-09-30-predict/output.json)、[检查器源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L467) |

## 运行现象与 T03/T04/T05 对照

| 实际运行现象 | 支持、否定或限定的结论 | 对应任务及其可引用证据 |
|---|---|---|
| `reset` 得 `[4]` observation 与 dict；`predict` 得动作 `1`，动作空间检查为 true；`step` 得下一 observation、reward、terminated、truncated、info | 支持 T03 的一次原生环境预测流程。动作来自未训练策略，不能据此评价策略效果 | [T03 #3](https://github.com/Jaschenn/sb3-llm-code-understanding/issues/3)：[预测脚本](../../../scripts/issue_06_predict.py)及[输出](../../../artifacts/runs/issue-06/2026-09-30-predict/output.json)，对应 [BaseAlgorithm.predict](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L537) 到 [BasePolicy.predict](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L331) |
| `DummyVecEnv` reset shape `[1,4]`；step 的 rewards/dones shape `[1]`，infos 数量 1；误传 `(observation, info)` 到 `predict()` 抛出带 API 混用提示的 `ValueError` | 支持 T03 须区分原生 Env 与 VecEnv；也为 T05 提供一条接口误用边界。此错误属于 `predict()` 的防护，不是 `check_env()` 的失败路径 | [T03 #3](https://github.com/Jaschenn/sb3-llm-code-understanding/issues/3)、[T05 #5](https://github.com/Jaschenn/sb3-llm-code-understanding/issues/5)：[预测输出](../../../artifacts/runs/issue-06/2026-09-30-predict/output.json)，对应 [DummyVecEnv.step_wait/reset](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/vec_env/dummy_vec_env.py#L56) 和 [BasePolicy 的元组检查](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L350) |
| 两次 `collect_end → train_start → train_end`；每次 `collect_end` buffer 满、位置 16；`updates` 依次为 0、1、2 | 支持 T04 的“采样后训练”顺序与 RolloutBuffer 在训练前已有数据。未验证 PPO loss 或奖励上升 | [T04 #4](https://github.com/Jaschenn/sb3-llm-code-understanding/issues/4)：[短训练脚本](../../../scripts/issue_06_short_training.py)及[事件日志](../../../artifacts/runs/issue-06/2026-09-30-short-training/events.jsonl)，对应 [collect_rollouts](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L162) 和 [learn](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L300) |
| `check_env(CartPole-v1)` 正常返回 | 仅支持该合法环境通过检查。T05 所需的两条不合法自定义环境失败路径须由 T05 自己验证 | [T05 #5](https://github.com/Jaschenn/sb3-llm-code-understanding/issues/5)：[预测输出](../../../artifacts/runs/issue-06/2026-09-30-predict/output.json)，对应 [check_env](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L467) 和 [返回值检查](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L331) |

## 复现与环境差异

[运行入口和准备命令](../../../artifacts/runs/issue-06/README.md)给出上游 commit、安装方式、版本与三个独立实验的实际命令；每个目录保存其种子、目标和输出。源码与运行环境可区分：基线固定了 SB3 源码，但 PyTorch/Gymnasium 等依赖是本次安装时解析出的版本。复核时若使用不同依赖或平台，应记录差异。动作值、奖励及浮点数值可能不同；关键核验对象为返回结构、异常类别、调用先后及 buffer 状态。

## AI 使用记录

| 编号 | 模型与提问原文 | AI 回答摘要 | 验证动作 | 观察结果 | 修正后的表述 |
|---|---|---|---|---|---|
| AI-06-01 | Codex（GPT-6）；用户提问：`帮我完成这个 issue：https://github.com/Jaschenn/sb3-llm-code-understanding/issues/6` | 提议用 CartPole 最小预测、合法环境检查及两轮短训练验证 T03/T04/T05 相关行为 | 核对固定源码，运行三个脚本并保存原始输出 | 预测接口差异和训练顺序得到支持；合法环境通过不等于覆盖 T05 自定义环境失败路径 | 仅把本次可观察的接口与控制流归入已支持；T05 失败路径继续由其任务单独验证 |

## 复核记录

待非提交者在 PR 中复跑并审阅。当前由提交者执行了脚本、Ruff lint/格式检查和 Python 编译检查；这不能代替独立复核。
