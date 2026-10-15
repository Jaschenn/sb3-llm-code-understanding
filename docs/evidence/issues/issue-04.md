# Issue #4：learn() 框架级训练调用链

负责人：@dingdadabset

复核人：@Jaschenn

上游基线：commit `7cfb4dd6055e74b5caa4ed4d6777492209946e26`，见[基线说明](../../project-baseline.md)。

## 结论与逐项证据

范围：固定基线标准 PPO / MlpPolicy / RolloutBuffer 的软件控制流。所有运行脚本先核验实际导入目录、Git commit 和源码清洁状态。本轮把主路径实测、受控分支检查、上游测试与源码证据分开，不从局部 BN 观察推断任意模型或回调的行为。

当前验证材料见[第二轮运行记录](../../../artifacts/runs/issue-04/2026-10-15-audit/README.md)。状态使用“已支持 / 需限定 / 待验证 / 已否定”；状态针对表中限定后的命题，不等于该模块所有分支都已验证。

| 编号 | 限定后的可检验结论 | 状态 | 固定源码 | 对应运行 / 检查 |
|---|---|---|---|---|
| C-04-01 | PPO.learn 将调用转交 OnPolicyAlgorithm.learn；本轮只分析 on-policy 的 PPO | 已支持 | [PPO.learn](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/ppo.py#L302)、[主循环](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L300) | [真实调用输出](../../../artifacts/runs/issue-04/2026-10-15-audit/output.log) 的真实 learn/setup/rollout/train 方法边界，不以脚本打印的伪代码为证据 |
| C-04-02 | _setup_learn 初始化计时与 callback；缓冲 deque、环境 reset、计数器和 logger 均存在条件分支。reset_num_timesteps=False 时保留累计步数并把本次请求加到目标上，已有 last_obs 时不 reset | 已支持 | [_setup_learn](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L382) | [契约检查](../../../scripts/issue_04_contract_checks.py)::test_timestep_lower_bound_and_resume_without_reset；首次 learn 与续训只观察到一次 env.reset |
| C-04-03 | 完整 rollout 后先计算 returns，再更新 progress、按 log_interval 可选 dump_logs，最后 train；回调可提前结束 | 已支持 | [collect/learn](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L258) | [真实调用输出](../../../artifacts/runs/issue-04/2026-10-15-audit/output.log) 直接 profile 完整顺序；[契约检查](../../../scripts/issue_04_contract_checks.py)::test_collection_preserves_parameters_and_computes_returns_before_callback / test_callback_false_stops_without_training |
| C-04-04 | 固定基线标准采样代码在 eval/no_grad 下前向，不执行优化器更新；本次默认 policy、无修改参数的 callback 下，整个 state_dict 在采样前后相同 | 需限定 | [collect_rollouts](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L183) | [契约检查](../../../scripts/issue_04_contract_checks.py)::test_collection_preserves_parameters_and_computes_returns_before_callback 直接比较全部被测参数/缓冲并观察 forward 模式；BN 专项上游测试只补充 BN 证据，不推广到任意自定义 callback |
| C-04-05 | reset 重新分配归零 arrays 并清 pos/full/generator_ready；首次 get 展平部分字段，每次 get 都重新 permutation；actions 第三维为 action_dim，标量字段为 (n_steps,n_envs) | 已支持 | [reset](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/buffers.py#L391)、[get](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/buffers.py#L481) | [真实调用输出](../../../artifacts/runs/issue-04/2026-10-15-audit/output.log) 的初始字段与 reset 断言；[契约检查](../../../scripts/issue_04_contract_checks.py)::test_buffer_flattening_and_permutation_each_get 检查展平前后及每次 permutation |
| C-04-06 | collect_rollouts 在末步预测 values 并调用 compute_returns_and_advantage，然后才 on_rollout_end；本项目不验证 GAE 数学正确性 | 已支持 | [returns→callback](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L258) | [契约检查](../../../scripts/issue_04_contract_checks.py)::test_collection_preserves_parameters_and_computes_returns_before_callback 观察原计算函数返回后再进入 callback |
| C-04-07 | done、非空 terminal_observation、TimeLimit.truncated 三条件满足时，reward 加 gamma*V(terminal_obs)；真实终止不做此修正 | 已支持 | [bootstrap](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L236) | [契约检查](../../../scripts/issue_04_contract_checks.py)::test_time_limit_bootstrap_only_for_truncation：控制 V=2、gamma=0.9，截断得 2.8，真实终止得 1.0；这是分支验证，不是学得价值函数的正确性验证 |
| C-04-08 | PPO.train 更新学习率与 clip schedule 后遍历 epochs/minibatches；KL 超过 1.5*target_kl 时跳过该批优化并结束本轮 train；每个已进入 epoch 计数加一 | 已支持 | [PPO.train](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/ppo.py#L184) | [真实调用输出](../../../artifacts/runs/issue-04/2026-10-15-audit/output.log) 的默认 4 rollout×2 epochs 计数 8；[契约检查](../../../scripts/issue_04_contract_checks.py)::test_kl_multiplier_skips_optimizer_and_counts_epoch 用受控 log_ratio 验证阈值、optimizer.step 次数和早停计数，不声称测量真实策略 KL |
| C-04-09 | 仅 normalize_advantage=True 且实际 minibatch 样本数大于 1 时归一化 advantage；尾批单样本也跳过，不仅取决于配置 batch_size | 已支持 | [归一化条件](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/ppo.py#L216) | [契约检查](../../../scripts/issue_04_contract_checks.py)::test_advantage_normalization_flag_and_singleton_tail 观察 torch.std：启用时仅对双样本批调用，禁用时不调用，单样本尾批均不调用 |
| C-04-10 | 启用归一化时 batch_size 与 n_steps*n_envs 必须大于 1；禁用时允许单样本；buffer 样本数不整除 batch_size 时发出尾批警告 | 已支持 | [初始化断言/警告](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/ppo.py#L140) | [契约检查](../../../scripts/issue_04_contract_checks.py)::test_normalization_init_rejects_single_sample / test_normalization_disabled_allows_single_sample / test_remainder_minibatch_warning。test_ppo 的 clip_range_vf 断言不再冒充这些条件的证据 |
| C-04-11 | 本次 learn 后参数快照发生变化，但得分可下降；total_timesteps 为请求下界，完整 rollout 可越过目标，续训的目标从当前累计步数递增 | 已支持 | [训练循环](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L323) | [真实调用输出](../../../artifacts/runs/issue-04/2026-10-15-audit/output.log) 中 12 个参数张量变化，222.3→79.0，差值 -143.3；[契约检查](../../../scripts/issue_04_contract_checks.py)::test_timestep_lower_bound_and_resume_without_reset；反向验证拒绝仅增加计数却不改参数的 train |
| C-04-12 | dump_logs 从 ep_info_buffer 计算 r/l 平均值；time/iterations、time/time_elapsed、time/total_timesteps 排除 TensorBoard，实际输出目标由 logger 配置决定 | 已支持 | [dump_logs](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L277) | [契约检查](../../../scripts/issue_04_contract_checks.py)::test_logger_statistics_and_tensorboard_exclusions 直接检查记录的统计值及排除项；不将该例扩展为所有日志 writer 的验证 |

## AI 使用记录

本表保留原始提问和回答摘要。AI-04-03 的“限制偏离”属于算法动机说明，不构成 clip 能保证 KL 上界或避免过拟合的验证结论；本轮只核验实际软件分支。get 每次重新抽取随机排列，也不保证两次得到的排列一定不同。

| AI 编号 | 模型与提问原文 | AI 回答摘要 | 验证动作 | 观察结果 | 修正后的表述 |
|---|---|---|---|---|---|
| AI-04-01 | GPT-4o-mini，「stable-baselines3 里 PPO 的 learn() 到底是哪个对象在执行？PPO.learn 和 OnPolicy.learn 关系是啥？」 | AI 回答：「PPO.learn 里写了主要的训练 while 循环，OnPolicyAlgorithm 只是提供一些工具函数」 | 直接打开 `ppo/ppo.py:302` 看实现 + 打开 `on_policy_algorithm.py:300-342` 对比 | ❌ AI 把继承关系搞反了。`PPO.learn` 只有 `return super().learn(...)` 一行；真正 while 循环、rollout→train 调度全在 `OnPolicyAlgorithm.learn` 里，PPO 自己只实现 `train()`（损失计算 + 优化一步）。 | 正确继承链：`PPO` 重写 `__init__`、`_setup_model`、`train()`；`learn()` 完全复用父类 `OnPolicyAlgorithm`，后者依赖 BaseAlgorithm 提供的 `_setup_learn`、logger、callback 基础设施；PPO.learn 仅是透传，其额外参数只是提供默认值 `tb_log_name="PPO"` 给 logger。 |
| AI-04-02 | GPT-4o-mini，「RolloutBuffer.reset() 会把 observations/actions 这些 numpy 数组清成 zeros 吗？还是只移动 pointer？」 | AI：「RolloutBuffer.reset() 为了性能，不会 reallocate 大数组，只重置 pos=0/ptr；下一 add 会覆盖。」 | 核对 `common/buffers.py:391-401,481-507`；修正版 Section E 保留旧 observations 引用、填 9、reset 后断言新对象且全零 | AI 原回答被源码与运行证伪；此前“已核验基本正确”的记录也有误 | RolloutBuffer.reset 会重新分配并归零数组，重置 generator_ready，再调用 BaseBuffer.reset 重置 pos/full。generator_ready 控制 swap_and_flatten，不控制随机置换；get 每次调用都会重新 permutation。 |
| AI-04-03 | GPT-4o-mini，「train() 里为什么有 n_epochs 循环？是把同样的 rollout 数据过 n_epochs 次吗？会不会过拟合？」 | AI：「是的，每一轮 rollout 的数据拿来训练 n_epochs 次，这就是 PPO 的多 epoch 优化；因为样本是 on-policy 的，你多 epoch 后 policy 已经偏离当前 rollout 采样分布了，所以才需要 clip surrogate 限制不要飘太远。」 | 对照 `ppo/ppo.py:204 for epoch in range(self.n_epochs):` 和内层 `for rollout_data in self.rollout_buffer.get(batch_size):` + L222-L227 clip surrogate 的定义 | ✅ AI 这段是对的（多 epoch + clip 防 drift 是经典 PPO clip 版本的核心机制）。但 AI 没提「每一个 epoch 内部 buffer.get() 都会重新 shuffle indices」——这意味着不同 epoch 里 minibatch 的切分不是同一批（`buffers.py:484-488` 每次 get 重新 `indices = np.random.permutation`）。 | 保留 AI 原表述并补充：`RolloutBuffer.get()` 每次进入（每 epoch 一次）会重新 shuffle indices，因此同一 rollout 数据在不同 n_epochs 内部经历的 minibatch 划分是随机不同的；这种多 epoch 重排 + clip surrogate 共同构成 PPO 的 "sample reuse while staying close to old policy" 机制。 |

## 复核记录

2026-10-15，复核人：@Jaschenn。对 [PR #18](https://github.com/Jaschenn/sb3-llm-code-understanding/pull/18) 检查并修正，原提交为 `7481cf67b48367de3c147e1e973b16e4f91fc5b3`。以下保留源码核对与自动化实测结果，本次直接修正并提交，不合并 PR。

| 问题 | 修正与验证 |
|---|---|
| AI-04-02 把不重新分配数组记为已核实，且误解 generator_ready | 改为被源码与实测证伪；Section E 保留旧数组、填 9、reset 后断言新数组全零；说明 get 每次 permutation，flag 仅控制展平 |
| 用均分差异代替参数验证，且把 79.0 误称提升 | learn 前保存参数快照，learn 后直接断言存在变化；本次变化 12 个张量。恢复快照评估得 222.3，训练后 79.0，真实差值 -143.3，不要求分数上涨 |
| 训练图 KL 阈值、早停边及 epoch 计数不符源码 | 改为 1.5 × target_kl，明确本批优化跳过、每 epoch 末加 1，再判断是否继续；补 callback False 的退出分支和 log_interval=None 条件 |
| callback 只打印、没有断言完整训练节奏 | 增加 train 边界与 buffer 满状态观测；断言完整 callback 序列、32/64/96/128 的 train 边界、总步数 128 和 `_n_updates=8` |

环境：Python 3.12.14 / PyTorch 2.14.0 / Gymnasium 1.3.0 / SB3 2.9.2a0，SB3 从固定基线源码构建 wheel；CPU 单线程。修正版脚本正常退出，Ruff lint/格式检查通过；上游采样模式、advantage normalization 和 PPO 测试共 **14 passed**。完整命令、脚本快照、输出与测试日志见[复核运行目录](../../../artifacts/runs/issue-04/2026-10-15-review/README.md)。原运行目录保留，避免用新结果覆盖历史材料。训练图已通过 Mermaid CLI 12.0.0 实际渲染。

固定源码索引：[RolloutBuffer.reset](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/buffers.py#L391)、[RolloutBuffer.get](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/buffers.py#L481)、[KL 早停及计数](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/ppo.py#L267)、[learn 主循环](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L300)。

### 第二轮审慎检查

复核人：@Jaschenn。针对上一轮提交 `d305824`，重写逐结论证据表，补齐与命题直接对应的检查。

- 新增固定 Git 源码导入、commit 和清洁状态校验，普通安装版即使接口兼容也不得用于生成“固定基线验证成功”的记录。
- 通过真实方法 profile 检查 setup→collect→returns→progress→dump→train 顺序；静态打印的伪代码单独作为解释。
- 修正“BN 统计量不变足以证明任何参数不变”的推论；标准采样用例直接比较整个被测 state_dict，并明确自定义 callback 的副作用不受此结论约束。
- 为归一化开关/单样本尾批、两项初始化断言、关闭归一化的对照和尾批 warning 建立直接检查；不再拿 clip_range_vf 测试证明其他断言。
- 分别触发 KL 两个阈值区间、跳过 optimizer.step、epoch 计数、callback False、续训不 reset、请求步数下界、log_interval=None、logger 排除项及截断/真实终止的奖励对照。
- 受控 log_ratio 与受控 V 值仅验证软件分支；不会据此宣称 PPO/GAE 数学正确或实际性能改善。

本轮完整结果和命令见[第二轮运行记录](../../../artifacts/runs/issue-04/2026-10-15-audit/README.md)。旧运行目录保留；其中没有基线防护的脚本只作为历史快照，当前复现入口使用 scripts 中修正版。
