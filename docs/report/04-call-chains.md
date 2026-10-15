# 4. 业务调用链

本章分析固定基线 `7cfb4dd6055e74b5caa4ed4d6777492209946e26` 的 PPO。主实验为 CartPole-v1 / MlpPolicy；连续动作与 Dict 输入通过单独的契约检查补充。自定义 policy、callback、CNN、循环策略及其他算法的重写不自动继承本章结论。源码基线见[说明](../project-baseline.md)。

证据分为三类：源码用于确认实现与条件；实际调用记录用于确认当前配置的执行路径；受控检查用于触发特定边界。打印方法名称或伪代码不是方法被实际调用的证据，最终动作相同也不足以证明内部路径相同。

## 4.1 预测调用链

[预测时序图](../../artifacts/diagrams/issue-03/predict-call-chain.mmd)与[T03 逐项证据](../evidence/issues/issue-03.md)对应。

### 4.1.1 从用户调用到动作

原生 Gymnasium 的最小循环为：

```python
obs, info = env.reset(seed=0)
action, state = model.predict(obs, deterministic=True)
obs, reward, terminated, truncated, info = env.step(int(action))  # CartPole Discrete
```

对当前 CartPole 输入，[固定源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L331)与 [sys.setprofile 实际调用记录](../../artifacts/runs/issue-03/2026-10-15-audit/output.log)共同支持：

| 阶段 | 入口及职责 | 输入/输出与边界 |
|---|---|---|
| 公共入口 | `BaseAlgorithm.predict → BasePolicy.predict` | PPO 复用此入口；`state` 在当前非循环策略中原样回传，默认 None |
| 模式与输入 | 切 eval，拦截长度为 2 且第二项为 dict 的 tuple | eval 控制 Dropout/BatchNorm 的标准行为；不是“任意自定义网络绝不改状态”的保证 |
| 观测转换 | `BaseModel.obs_to_tensor` | 单个 `(4,)` 转为 `(1,4)` tensor，vectorized=False；已带 batch 的合法输入保留批维；Dict 按 key 转换 |
| 策略 | no_grad 内调用 `ActorCriticPolicy._predict → get_distribution` | 本次 profile 直接检查 _predict 内 eval/no_grad 状态，同时监测到此路径未调用 ActorCriticPolicy.forward |
| 特征与 actor | `BaseModel.extract_features → preprocess_obs → FlattenExtractor.forward → MlpExtractor.forward_actor` | CartPole 特征提取器是 flatten；没有额外的“identity 线性层” |
| 分布 | `_get_action_dist_from_latent → action_net → CategoricalDistribution.proba_distribution` | 当前 Discrete(2) 场景构造分类分布；不推广为所有 PPO 动作空间均为 Categorical |
| 取动作 | `Distribution.get_actions → CategoricalDistribution.mode` | 当前 deterministic=True 使用 argmax；连续分布的 mode 不等于分类 argmax |
| 返回 | tensor→NumPy、reshape、按空间后处理、非批量 squeeze | CartPole 单观测最后得到 shape=() 动作；批量得到一维动作数组 |

实际记录只覆盖所监测的方法和当前输入，不是所有分支的穷尽调用图。另行手动展开并与 API 比较，证明该输入下动作一致，不能单独证明中间层的实现。

### 4.1.2 Box、Dict 与环境 API

[BasePolicy.predict 后处理](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L372)在 Box 空间中：squash_output=False 时 clip 到边界，True 时把 [-1,1] unscale 到环境边界。本轮 [T03 契约检查](../../scripts/issue_03_contract_checks.py)实际触发了超界动作 clip，并用固定 _predict 输出单独检查 unscale；后者不证明 gSDE 采样正确性。

Dict 单观测和批量观测的转换也有独立检查；图像转置、混合 Dict 形状和 CNN 分支未作本轮运行覆盖。

| 接口 | 原生 Gymnasium Env | 本次 DummyVecEnv |
|---|---|---|
| reset | `(obs, info)` | 批量 obs；info 存于 reset_infos |
| step | `(obs, reward, terminated, truncated, info)` | `(obs, rewards, dones, infos)`，dones 合并 terminated/truncated |
| 当前 Box 观测形状 | `(4,)` | `(1,4)`；不把该 ndarray 形状推广到所有空间 |

这些差异通过 `test_raw_gym_and_vecenv_return_contracts` 实际检查。无效 reset tuple 与 step 四变量解包是不同失败位置，见 4.3。

### 4.1.3 证据入口

[第二轮运行材料](../../artifacts/runs/issue-03/2026-10-15-audit/README.md)包含实际导入路径、commit、依赖版本、真实调用序列与契约检查结果。程序启动时拒绝错误 commit、脏源码和 wheel/其他非源码安装路径。上游预测测试的通过范围以具体用例和输出为准，不能推出任意网络状态不变。

## 4.2 训练调用链

T04 的[训练图由 PR #18 提供，合入后相对路径生效](../../artifacts/diagrams/issue-04/learn-call-chain.mmd)；合入前通过 [PR #18](https://github.com/Jaschenn/sb3-llm-code-understanding/pull/18) 查看。本节依赖其第二轮脚本与证据，不在 T03 分支复制 T04 文件。

### 4.2.1 主流程与数据协作

[PPO.learn](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/ppo.py#L302)转交 [OnPolicyAlgorithm.learn](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L300)。

1. `_setup_learn` 设置计时、callback 和目标步数。deque、环境 reset、计数器和 logger 的处理有条件；reset_num_timesteps=False 的续训保留当前步数，将请求量加到目标，并在已有 last_obs 时不 reset 环境。
2. `collect_rollouts` 切 eval、reset buffer、执行 rollout callback。在 no_grad 内用 `policy(obs)` 得到 actions/values/log_probs；此处走 ActorCriticPolicy.forward，与预测接口不同。
3. VecEnv.step 后累计 n_envs 步，调用 on_step；False 会立即结束本次采样，本轮不调用 train。正常路径处理 episode 信息、条件 bootstrap、buffer.add，再更新 last_obs 和 episode_starts。
4. 采满后 predict_values，并计算 returns/advantages，之后才 on_rollout_end。
5. learn 更新进度，按 log_interval 可选 dump_logs，再调用 PPO.train；随后进入下一轮。标准采样代码没有优化器步骤，但不能据此约束任意自定义 callback 的副作用。

T04 主实验用 n_steps=32、n_envs=1、batch_size=16、n_epochs=2、total_timesteps=128。第二轮 profile 观察 setup→四轮 collect/returns/progress/dump/train，直接断言顺序；不是只打印伪代码。完整 rollout 可超过请求步数；提前停止则可能不足请求量。续训应按当前累计步数加请求量判断目标，不简单套用从零开始的整倍数公式。

### 4.2.2 RolloutBuffer 生命周期

固定 [reset/get 实现](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/buffers.py#L391)支持：

| 字段 | reset 后形状 | 使用方式 |
|---|---|---|
| observations | `(n_steps,n_envs,*obs_shape)`，dtype 来自观测空间 | add 写入；get 首次展平为样本维 |
| actions | `(n_steps,n_envs,action_dim)`，dtype 来自动作空间 | get 样本转换为 float32；Discrete 在 train 中再 long/flatten |
| rewards/episode_starts/values/log_probs/returns/advantages | `(n_steps,n_envs)`，float32 | episode_starts 用 0/1 表示；GAE 末步使用 dones，其他步使用下一步 episode_starts |

buffer_size=n_steps，样本总数为 n_steps*n_envs。reset 重新分配并归零数组、清 generator_ready，并重置 pos/full；generator_ready 控制首次 swap_and_flatten，每次 get 的 permutation 独立执行。展平后的部分字段形状会变化，不能把表中 reset 后形状称为整个生命周期固定形状。

### 4.2.3 PPO.train 的条件

[PPO.train 源码](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/ppo.py#L184)将学习率与 clip schedule 用于当前进度；schedule 可以是常数，不保证线性衰减。

- 每个 epoch 从 get(batch_size) 重新打乱的数据中取 minibatch，evaluate_actions 后计算损失。
- 仅 normalize_advantage=True 且实际批内样本数大于 1 时归一化。配置 batch_size 大于 1 仍可能有单样本尾批，该尾批跳过归一化。
- KL 超过 1.5*target_kl 时，当前批不执行优化，结束该轮 train；之后外层 learn 仍可继续采样。每个已进入 epoch 末 `_n_updates += 1`，包括触发早停的 epoch；它不等于 optimizer.step 次数。
- 正常批执行 zero_grad→backward→梯度裁剪→step，最终记录 train/*；更新参数不保证评估回报增加。

初始化时仅在开启 advantage normalization 时要求 batch_size 和 n_steps*n_envs 大于 1；不整除 minibatch 则警告。本轮 T04 针对这些具体条件添加检查，原先引用的负 clip_range_vf 测试只能支持自己的条件，不能证明其他初始化断言。

T04 第二轮还单独触发：callback False、log_interval=None、续训不 reset、TimeLimit 截断奖励修正及真实终止对照、KL 阈值与优化步骤跳过。受控 KL 使用固定 log_ratio，bootstrap 使用固定 V 值，均用于验证软件分支，不验证学习算法数学正确性或真实策略质量。具体函数名与输出见 PR #18 的第二轮材料。

## 4.3 失败位置与证据边界

| 触发情形 | 实际失败/退出位置 | 本章证据范围 |
|---|---|---|
| 把 `(obs, info)` 直接传 predict | BasePolicy.predict 的特定两项 tuple 防护 | T03 主脚本实际触发 |
| 四变量接收 Gymnasium 五项 step 返回值 | 用户代码解包处抛 ValueError，尚未调用 predict | T03 主脚本实际触发 |
| 开启归一化但 batch_size=1 或 buffer 样本数=1 | PPO 构造器断言 | T04 分别触发两项，并验证关闭归一化的对照 |
| on_step 返回 False | collect_rollouts 返回 False，learn 不 train 本轮并调用 on_training_end | T04 实际触发 |
| KL 超阈值 | 本批优化前跳出 minibatch，epoch 计数后结束本轮 train | T04 受控输入检查 |
| 自定义环境接口不合规，用户显式运行 check_env | env_checker 的对应检查 | 由 T05 证据负责，本章不假设存在未提交的失败脚本 |
| 未检查的环境在训练中返回四项 step 结果 | DummyVecEnv.step_wait 解包底层五项返回值处 | 源码定位；本章未另建该自定义环境实测，不能写成 env_checker 自动检查 |

PPO 构造与 learn 不自动运行 check_env。所有“已支持”都针对明确范围的命题；尚未触发的分支应按源码证据说明，不将主路径测试通过扩展为全面覆盖。
