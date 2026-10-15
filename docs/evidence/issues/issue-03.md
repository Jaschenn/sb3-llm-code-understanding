# Issue #3：predict() 端到端调用链

负责人：@dingdadabset

复核人：@Jaschenn

上游基线：commit `7cfb4dd6055e74b5caa4ed4d6777492209946e26`，见[基线说明](../../project-baseline.md)。

## 结论与逐项证据

范围：固定基线的 PPO + ActorCriticPolicy，主实验为 CartPole-v1 / MlpPolicy。手动展开与 API 动作相同只证明该输入的输出一致；本轮另用 sys.setprofile 观察真实 API 方法调用，并与源码逐项对应。静态打印的调用链清单不充当运行证据。

当前验证材料见[第二轮运行记录](../../../artifacts/runs/issue-03/2026-10-15-audit/README.md)。状态使用“已支持 / 需限定 / 待验证 / 已否定”；状态针对表中限定后的命题，不等于该模块所有分支都已验证。

| 编号 | 限定后的可检验结论 | 状态 | 固定源码 | 对应运行 / 检查 |
|---|---|---|---|---|
| C-03-01 | PPO 沿用的 BaseAlgorithm.predict 将参数转交 self.policy.predict；不推广到自行重写 predict 的算法 | 已支持 | [BaseAlgorithm.predict](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L537) | [真实调用输出](../../../artifacts/runs/issue-03/2026-10-15-audit/output.log)：profile 观察两个入口；转发参数的实现由源码核对，不从动作相等推断 |
| C-03-02 | BasePolicy.predict 切 eval、拦截特定 tuple、转换观测，在 no_grad 内调用 _predict，再 reshape、按动作空间后处理并按 vectorized 标记 squeeze | 已支持 | [BasePolicy.predict](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L331) | [真实调用输出](../../../artifacts/runs/issue-03/2026-10-15-audit/output.log)：监测当前 CartPole 输入的真实方法顺序及 _predict 内 eval/no_grad；源码补充各分支；不宣称所有自定义 policy 都符合相同内部链 |
| C-03-03 | 长度 2 且第二项为 dict 的 Gymnasium reset tuple 被 predict 拒绝；step 五项用四变量接收则先在用户代码解包处失败 | 已支持 | [tuple 防护](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L356) | [真实调用输出](../../../artifacts/runs/issue-03/2026-10-15-audit/output.log)：分别实际触发并断言两个错误，不混为一条失败路径 |
| C-03-04 | obs_to_tensor 保留或补齐 batch 维，返回 tensor 与 vectorized 标记；本次覆盖 CartPole ndarray 单观测和 Dict 单观测/批量 | 已支持 | [BaseModel.obs_to_tensor](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L236) | [真实调用输出](../../../artifacts/runs/issue-03/2026-10-15-audit/output.log) 的 (4,)→(1,4)；[契约检查](../../../scripts/issue_03_contract_checks.py)::test_dict_and_batch_observations。图像转置和混合 Dict 形状不在本次运行覆盖内 |
| C-03-05 | PPO 的 ActorCriticPolicy 在本次 Discrete(2) 输入下走 _predict→get_distribution→特征/actor 网络→Categorical→确定性 mode；不调用 ActorCriticPolicy.forward | 已支持 | [_predict](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L709)、[get_distribution](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L743) | [真实调用输出](../../../artifacts/runs/issue-03/2026-10-15-audit/output.log)：profile 记录预处理、特征提取、分布构造及 mode 顺序，同时监测 forward；仅针对当前输入路径 |
| C-03-06 | 离散确定性动作取 Categorical argmax；Box 在 squash_output=False 时 clip，True 时 unscale 至环境边界 | 已支持 | [Box 后处理](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L372) | [真实调用输出](../../../artifacts/runs/issue-03/2026-10-15-audit/output.log) 的 argmax 断言；[契约检查](../../../scripts/issue_03_contract_checks.py)::test_box_clips_unbounded_actions / test_box_unscales_controlled_squashed_action。unscale 用固定 _predict 输出隔离后处理，不声称验证了 gSDE 随机采样 |
| C-03-07 | 原生 Gymnasium reset/step 为 2/5 项；DummyVecEnv reset 只给批量 obs，step 为 4 项，reset info 存于 reset_infos | 已支持 | [DummyVecEnv](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/vec_env/dummy_vec_env.py#L56) | [契约检查](../../../scripts/issue_03_contract_checks.py)::test_raw_gym_and_vecenv_return_contracts；本次为普通 Box 观测，不将 ndarray 形状推广到所有观测空间 |
| C-03-08 | 上游被测含 BatchNorm/Dropout 的网络在确定性重复预测时输出一致、指定 BN 统计量不变；不据此保证任意自定义网络状态均不变 | 需限定 | [eval/no_grad](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L352)、[上游测试](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/tests/test_train_eval_mode.py#L346) | 本轮重新执行 test_predict_with_dropout_batch_norm；该测试只检查选定统计量和输出，不是全模型任意状态的证明 |

## AI 使用记录

以下保留原始提问和回答摘要。原记录中“输出相同证明内部路径一致”的验证解释不足，现以本轮实际 profile 和上表的限定结论为准；输出相等本身只证明被测输入的动作一致。连续分布的确定性 mode 也不应混同为分类 argmax；Box clip/unscale 由空间与 squash_output 决定。

| AI 编号 | 模型与提问原文/链接 | AI 回答摘要 | 验证动作 | 观察结果 | 修正后的表述 |
|---|---|---|---|---|---|
| AI-03-01 | GPT-4o-mini，「stable-baselines3 的 model.predict(obs) 到动作之间的调用顺序是怎样的？每层文件路径是啥？」 | AI 回答的顺序是：`BaseAlgorithm.predict → BasePolicy.predict → _get_action_dist_from_latent → forward`，并且说 BasePolicy.predict 内部直接走 forward() | 实际对比 policies.py 源码 + 自己在脚本里手动展开每一步 | 原记录中的动作相等只支持输出一致，不能证明内部路径。第二轮 profile 直接观察 _predict→get_distribution→mode，并监测到本次调用未进入 ActorCriticPolicy.forward。 | 固定基线当前 PPO/CartPole 路径为 BaseAlgorithm.predict→BasePolicy.predict→obs_to_tensor→no_grad 下 _predict→get_distribution→CategoricalDistribution.mode→NumPy 后处理。get_actions 定义于 Distribution；mode 使用 argmax(probs)。自定义策略不自动继承此路径。 |
| AI-03-02 | GPT-4o-mini，「predict 里如果传 deterministic=False 怎么采样？deterministic=True 呢？对 Discrete(2) 会不会有区别？」 | AI 说：「deterministic=False 会从 Categorical 按 logits 采样；True 直接 argmax；两者对离散和连续都一样」 | 对比 policies.py `_predict` + `common/distributions.py` 中 CategoricalDistribution.get_actions 代码 + 脚本里 det=True 时确实拿到的是 argmax 的结果 | ⚠️ AI 的结论基本正确，但漏了一个边界：连续空间（Box）如果 `squash_output=True`（gSDE），还需要走 `unscale_action()`；另外 get_actions 返回 shape 对 Discrete 是 `(batch,)`，对 Box 是 `(batch, action_dim)`，最后 reshape 成 `(-1, *action_space.shape)` 这一步 AI 也没提。 | 保留 AI 对 det=True/False 核心区分；补：① reshape 到 action_space.shape 的对齐动作；② Box 空间的 clip/unscale 分支；③ Discrete 的 get_actions 不会返回 float 动作，所以在最后 squeeze 之前 actions shape 与 Box 不一样。 |

## 复核记录

2026-10-15，复核人：@Jaschenn。对 [PR #17](https://github.com/Jaschenn/sb3-llm-code-understanding/pull/17) 检查并修正，原提交为 `b91914c5ef8e5a778d4745b0eb51052a5fd09912`。以下保留源码核对与自动化实测结果，本次直接修正并提交，不合并 PR。

| 问题 | 修正与验证 |
|---|---|
| 四变量解包 Gymnasium step 被误归为 predict tuple 拦截 | 报告按实际失败位置拆开说明；脚本实际触发两种错误并断言信息，见[复核输出](../../../artifacts/runs/issue-03/2026-10-15-review/output.log) |
| 把 env_checker 写入 learn 自动调用链，并引用假设存在的 T05 脚本 | 区分显式 check_env 与 DummyVecEnv.step_wait 的底层解包，删除假设脚本名；按固定源码定位，不宣称运行了不存在的实验 |
| 图稿使用作者电脑的 file:// 链接 | 改为仓库相对链接；训练图由 PR #18 提供，明确标记跨 PR 依赖，合入前可通过其 PR 查看 |
| 将更新计数误写为循环外 `_n_updates += epoch` | 按固定源码改为每 epoch 末加 1，并说明 KL 早停的计数行为；同步修正条件断言、logger 排除项等相邻错误 |
| 手动展开没实际切 eval，且读取全局临时缓存导致运行输入不可控 | 手动展开先切 eval；每次从 seed=0 初始化并训练 512 步，CPU 单线程；实际输出 Categorical logits/probs 并断言动作是 argmax |

本次环境为 Python 3.12.14 / PyTorch 2.14.0 / Gymnasium 1.3.0 / SB3 2.9.2a0，SB3 从固定基线源码构建 wheel。修正版脚本正常退出，Ruff lint/格式检查通过；上游 tuple 混用和含 Dropout/BatchNorm 的预测测试共 **11 passed**。命令、脚本快照与原始测试输出见[复核运行目录](../../../artifacts/runs/issue-03/2026-10-15-review/README.md)。原运行目录保留，不用新输出覆盖历史记录。Mermaid CLI 12.0.0 已成功渲染预测图。

固定源码索引：[predict 转发](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L537)、[policy 预测](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L331)、[DummyVecEnv.step_wait](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/vec_env/dummy_vec_env.py#L56)、[PPO epoch 计数](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/ppo.py#L280)。

### 第二轮审慎检查

复核人：@Jaschenn。针对上一轮提交 `a1e87b3` 遗留的证据问题，逐项重写了当前结论表与报告正文。

- 新增导入来源核验：实际包必须来自固定 commit 的干净 Git 源码检出；拒绝普通 wheel、错误 commit 与未提交改动，不能只打印固定源码标签。测试路径与运行路径均记录到日志。
- 用 sys.setprofile 观察真实 predict 调用，检查 _predict 内 eval/no_grad；监测 ActorCriticPolicy.forward，并断言本次未调用。方法清单的打印和手动输出一致不再充当内部路径的证明。
- 针对 Dict 单条/批量、原生 Env 与 VecEnv、Box clip/unscale 分别运行检查；受控 unscale 与真实采样的边界写入结论表。
- 删除“高频”“所有网络状态不变”等无充分依据的推论；明确 BN 上游用例、当前标准策略和任意自定义策略之间的证据边界。
- T04 的对应结论、归一化条件、续训、数据展平、KL 与 callback 退出等说明在报告中同步收紧；报告明确 PR #18 的依赖。

本轮完整结果和命令见[第二轮运行记录](../../../artifacts/runs/issue-03/2026-10-15-audit/README.md)。旧目录中的无基线防护脚本仅作为历史快照，不作为当前复现入口。

### 第三轮复核

2026-10-15，复核人：@Jaschenn。复核提交 `b83fcb6b448719ab47e6f24df6260d74ad71b803`。AI 记录的当前观察/修正栏与真实 profile 证据统一；预测图限定为本次 CartPole 路径，Box 后处理改为旁注。

在隔离副本中，从 main `07a0a02342da345cc3eb343b6006856a2bbd557f` 依次模拟合并 PR #17 与 #18，无冲突。合并后两份主脚本正常退出；29 个针对性用例与 25 个上游用例合计 **54 passed**。其中 3 个上游参数组合直接返回，不计作实际执行了预测断言。结果与重现命令见[第三轮记录](../../../artifacts/runs/issue-03/2026-10-15-recheck/README.md)。本轮未发现新的运行故障，未合并远端 PR。提交签名因既有 1Password socket 故障不可用，本轮沿用单次禁用签名，不修改全局配置。
