# Issue #6：实际运行与可复现实验

负责人：@Jaschenn

复核执行：Codex（受 @Jaschenn 委托）；未取得非提交者 GitHub 审批，用户已授权管理员合并，详见文末复核记录。

上游基线：[project-baseline.md](../../project-baseline.md)，commit `7cfb4dd6055e74b5caa4ed4d6777492209946e26`。

## 可检验结论

| 编号 | 结论 | 状态 | 证据 |
|---|---|---|---|
| C-06-01 | 原生 Gymnasium `reset()` 的 `(observation, info)` 应先解包，再把 observation 交给 `PPO.predict()`；原生 `step()` 返回五项 | 已支持 | [预测输出](../../../artifacts/runs/issue-06/2026-09-30-predict/output.json)、[BaseAlgorithm.predict 源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L537)、[BasePolicy.predict 源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L331) |
| C-06-02 | SB3 包装出的 `DummyVecEnv` 使用批量 observation/action，`reset()` 只返回 observation，`step()` 返回四项；混传 Gymnasium reset 元组会报错 | 已支持 | [预测输出](../../../artifacts/runs/issue-06/2026-09-30-predict/output.json)、[包装源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L204)、[DummyVecEnv 源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/vec_env/dummy_vec_env.py#L56) |
| C-06-03 | 在这组短训练参数下，每次完成 16 步采样并填满 RolloutBuffer 后才调用 `train()`；两轮共 32 步、2 次更新 | 已支持，限定于本次运行与固定基线 | [事件日志](../../../artifacts/runs/issue-06/2026-09-30-short-training/events.jsonl)、[训练循环源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L300)、[buffer 写入源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L247) |
| C-06-04 | 原生 CartPole 通过本次 `check_env()`；这不足以证明任意自定义环境正确 | 已支持，但需限定 | [预测输出](../../../artifacts/runs/issue-06/2026-09-30-predict/output.json)、[检查器源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L467) |
| C-06-05 | 原始 CartPole 运行中，10 回合平均持续步数从未训练的 98.1 变为训练 16,384 步时的 455.5，继续到 32,768 步后为 161.3；该轨迹效果不单调 | 需限定：原始记录如此，复核末检查点为 485.6，性能下降未独立复现 | [效果实验脚本](../../../scripts/issue_06_cartpole_effect.py)、[原始逐回合输出](../../../artifacts/runs/issue-06/2026-09-30-cartpole-effect/output.json)、[复核输出](../../../artifacts/runs/issue-06/2026-10-15-pr12-review/cartpole-effect.json) |

## CartPole 实际效果

[效果实验](../../../artifacts/runs/issue-06/2026-09-30-cartpole-effect/README.md)在训练前及各检查点，用相同的 10 个种子（100–109）运行确定性策略。记录的是每个回合保持平衡的步数；CartPole 每步奖励为 1，故本次回合奖励也等于步数。

| 累计训练步数 | 平均持续步数 | 最短–最长 | 备注 |
|---:|---:|---:|---|
| 0 | 98.1 | 79–126 | 初始化策略 |
| 4096 | 84.9 | 48–146 | 比未训练时低 |
| 8192 | 91.1 | 51–146 | 仍低于未训练时 |
| 16384 | 455.5 | 365–500 | 4/10 回合达到 500 步上限 |
| 32768 | 161.3 | 147–194 | 继续训练后下降 |

这一轨迹回答本例中 CartPole 实际持续了多久，但单次固定种子实验不能证明一般性的训练收益或最佳停止步数。此前 32 步的两轮实验只验证代码执行顺序，不用于评价策略表现。

## 运行现象与 T03/T04/T05 对照

| 实际运行现象 | 支持、否定或限定的结论 | 对应任务及其可引用证据 |
|---|---|---|
| `reset` 得 `[4]` observation 与 dict；`predict` 得动作 `1`，动作空间检查为 true；`step` 得下一 observation、reward、terminated、truncated、info | 支持 T03 的一次原生环境预测流程。动作来自未训练策略，不能据此评价策略效果 | [T03 #3](https://github.com/Jaschenn/sb3-llm-code-understanding/issues/3)：[预测脚本](../../../scripts/issue_06_predict.py)及[输出](../../../artifacts/runs/issue-06/2026-09-30-predict/output.json)，对应 [BaseAlgorithm.predict](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L537) 到 [BasePolicy.predict](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L331) |
| `DummyVecEnv` reset shape `[1,4]`；step 的 rewards/dones shape `[1]`，infos 数量 1；误传 `(observation, info)` 到 `predict()` 抛出带 API 混用提示的 `ValueError` | 支持 T03 须区分原生 Env 与 VecEnv；也为 T05 提供一条接口误用边界。此错误属于 `predict()` 的防护，不是 `check_env()` 的失败路径 | [T03 #3](https://github.com/Jaschenn/sb3-llm-code-understanding/issues/3)、[T05 #5](https://github.com/Jaschenn/sb3-llm-code-understanding/issues/5)：[预测输出](../../../artifacts/runs/issue-06/2026-09-30-predict/output.json)，对应 [DummyVecEnv.step_wait/reset](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/vec_env/dummy_vec_env.py#L56) 和 [BasePolicy 的元组检查](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L350) |
| 两次 `collect_end → train_start → train_end`；每次 `collect_end` buffer 满、位置 16；`updates` 依次为 0、1、2 | 支持 T04 的“采样后训练”顺序与 RolloutBuffer 在训练前已有数据。未验证 PPO loss 或奖励上升 | [T04 #4](https://github.com/Jaschenn/sb3-llm-code-understanding/issues/4)：[短训练脚本](../../../scripts/issue_06_short_training.py)及[事件日志](../../../artifacts/runs/issue-06/2026-09-30-short-training/events.jsonl)，对应 [collect_rollouts](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L162) 和 [learn](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L300) |
| 固定 10 回合的平均持续步数：0 步训练时 98.1，16,384 步时 455.5，32,768 步时 161.3 | 给 T04 增加可观察的训练效果；同时限定“训练越久奖励越高”这一说法，本次运行并不支持它 | [T04 #4](https://github.com/Jaschenn/sb3-llm-code-understanding/issues/4)：[效果实验记录](../../../artifacts/runs/issue-06/2026-09-30-cartpole-effect/README.md)及[逐回合输出](../../../artifacts/runs/issue-06/2026-09-30-cartpole-effect/output.json) |
| `check_env(CartPole-v1)` 正常返回 | 仅支持该合法环境通过检查。T05 所需的两条不合法自定义环境失败路径须由 T05 自己验证 | [T05 #5](https://github.com/Jaschenn/sb3-llm-code-understanding/issues/5)：[预测输出](../../../artifacts/runs/issue-06/2026-09-30-predict/output.json)，对应 [check_env](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L467) 和 [返回值检查](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L331) |

## 复现与环境差异

[运行入口和准备命令](../../../artifacts/runs/issue-06/README.md)给出上游 commit、安装方式、版本与四个独立实验的实际命令；每个目录保存其种子、目标和输出。源码与运行环境可区分：基线固定了 SB3 源码，但 PyTorch/Gymnasium 等依赖是本次安装时解析出的版本。复核时若使用不同依赖或平台，应记录差异。程序流程的关键核验对象为返回结构、异常类别、调用先后及 buffer 状态；策略表现按固定测试种子的逐回合步数比较。

## AI 使用记录

| 编号 | 模型与提问原文 | AI 回答摘要 | 验证动作 | 观察结果 | 修正后的表述 |
|---|---|---|---|---|---|
| AI-06-01 | Codex（GPT-6-Sol Medium）；用户提问：`帮我完成这个 issue：https://github.com/Jaschenn/sb3-llm-code-understanding/issues/6` | 提议用 CartPole 最小预测、合法环境检查及两轮短训练验证 T03/T04/T05 相关行为 | 核对固定源码，运行三个脚本并保存原始输出 | 预测接口差异和训练顺序得到支持；合法环境通过不等于覆盖 T05 自定义环境失败路径 | 仅把本次可观察的接口与控制流归入已支持；T05 失败路径继续由其任务单独验证 |
| AI-06-02 | Codex（GPT-6-Sol Medium）；用户追问：`实际效果在哪看？ 不只是步骤，而是实际的训练结果（在我们的例子里就是carpole 持续了多久）` | 原有 32 步日志缺少任务表现指标，需按固定测试回合测量 CartPole 持续步数 | 增加多检查点效果脚本；在 10 个相同种子上记录逐回合步数、奖励、终止状态 | 0、4096、8192、16384、32768 步检查点平均分别为 98.1、84.9、91.1、455.5、161.3 步 | 程序顺序证据与实际效果证据分开呈现；本次效果先升后降，不推断普遍规律 |

## 复核记录

### 2026-10-15：PR #12 合并前验证

执行者：Codex，受 @Jaschenn 委托。验证对象为 PR #12 提交 `626423d41499a5a0d9bb4d610b6f130d4d686dc5`。本次追加复核说明和原始输出，不修改四个实验脚本及原始运行记录。完整命令、环境与结果见[复核运行目录](../../../artifacts/runs/issue-06/2026-10-15-pr12-review/README.md)。

- 从无未提交修改的上游 commit `7cfb4dd6055e74b5caa4ed4d6777492209946e26` 构建并安装 SB3 wheel；使用 Python 3.12.14 和原 `pip-freeze.txt` 的固定依赖版本。平台为 macOS 27.0.1 / arm64，原记录为 macOS 27.0 / arm64；硬件及底层数值运算差异未排除。
- 四个实验脚本均正常退出。Ruff 0.16.9 lint 和格式检查通过；预测 JSON 和两轮短训练的完整事件 JSONL 与原始记录一致。
- 效果实验的检查点、评估种子、回合步数与奖励、统计值一致性检查通过。五个均值为 **98.1、84.9、91.1、455.5、485.6**；前四项与原记录一致，末项不同于原始 **161.3**。
- 忽略耗时后整条学习曲线的逐值一致性断言失败，见[验证日志](../../../artifacts/runs/issue-06/2026-10-15-pr12-review/validation.log)。不能声称所有验证通过，也不能将原记录中的最终性能下降视为已独立复现。此前 Python 3.13.5 复跑也得到末项 485.6，本次对齐 Python 与依赖后仍不同，因此不能仅归因于 Python 版本。差异原因尚未确定。

### 审批例外与合并授权

普通合并已被 GitHub 分支保护拒绝：`main` 要求至少一名非提交者批准，而 PR #12 没有审批；当前登录账号 @Jaschenn 也是 PR 作者，不能审批自己的 PR。本次 Codex 验证不冒充非提交者的 GitHub 批准。

用户在本次会话明确授权：

> 直接自己更新一下复核记录，然后合并即可。允许使用--admin 跳过审批，但是也需要留在复核记录。

据此，本次在保留上述验证限制及数值差异后，授权使用 `gh pr merge 12 --merge --admin --match-head-commit <更新后的提交>` 合并。这是由用户明确批准的审批例外，不代表已经取得非提交者审批，也不修改仓库后续 PR 的审批要求。实际合并结果以 [PR #12](https://github.com/Jaschenn/sb3-llm-code-understanding/pull/12) 的 GitHub 记录为准。
