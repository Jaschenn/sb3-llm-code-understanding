# 4. 业务调用链

> 本章原始证据由 T03（predict 链，Issue #3）与 T04（learn 链，Issue #4）分头产出，对应证据文件分别位于 `docs/evidence/issues/issue-03.md` 与 `docs/evidence/issues/issue-04.md`。最终交叉校对与段落整合由组长完成；本节先把两条调用链的"最小复述 + 可定位源码锚点 + 验证方式"写出来，供后续全章排版时引用。源码基线统一为 commit `7cfb4dd`，引用方式参见 `docs/project-baseline.md`。

---

## 4.1 预测调用链：`reset → model.predict(obs) → env.step(action)`

**问题**：一次真实的、从用户一行代码出发的预测，中间经过了哪些对象 / 哪些包装 / 哪些边界检查？本项目分析的"预测"指 CartPole-v1 + PPO + MlpPolicy 的标准配置，Discrete 动作空间。对连续 Box 动作、Dict 观测、CNN 特征提取的分支会标注但不展开实现细节。

### 4.1.1 概览图（见 Issue #3 图稿）

Mermaid 可编辑源：[`artifacts/diagrams/issue-03/predict-call-chain.mmd`](file:///Users/dingquan/it/trae/project/sb3-llm-code-understanding/artifacts/diagrams/issue-03/predict-call-chain.mmd)。图内容对应下面 4.1.2 的步骤编号。

### 4.1.2 逐步复述

用户脚本里常见的写法是：
```python
env = gym.make("CartPole-v1")
model = PPO.load("ppo_cartpole.zip")
obs, info = env.reset(seed=0)        # ← 返回 (obs_ndarray, info_dict)
for _ in range(1000):
    action, _state = model.predict(obs, deterministic=True)   # ← 本条调用链入口
    obs, reward, terminated, truncated, info = env.step(int(action))
    if terminated or truncated: break
```

调用链按顺序经过下列对象与方法（每一步都可在固定 commit 下精确定位）：

| 步骤 | 入口 | 所在文件 / 行号 | 做了什么 | 关键点 / 边界 |
|---|---|---|---|---|
| 0 | `env.reset()` | `gymnasium`（非 SB3）→ Gymnasium 标准 2 元组 | 返回 `(obs: ndarray(4,), info: dict)` | 用户**不要**把这个 tuple 直接喂给 `predict()`，这是最常见错误（见 C-03-03） |
| 1 | `BaseAlgorithm.predict(obs, state, episode_start, deterministic)` | `common/base_class.py:536` | 原封不动转发给 `self.policy.predict(...)` | BaseAlgorithm 自身在此没有做任何形状校验、值校验；全部在 policy 里做 |
| 2 | `BasePolicy.predict(...)` 入口 | `common/policies.py:331` | ① `set_training_mode(False)` 切 eval，② 输入 tuple 检查，③ `obs_to_tensor`，④ `with no_grad: _predict`，⑤ numpy+reshape，⑥ Box 空间 clip/unscale，⑦ 是否 squeeze batch 维，⑧ 返回 `(action, state)` | **所有 predict 的用户可观测行为都在这里被决定**；`state` 在非 RNN 策略中始终原样回传 `None` |
| 2a | tuple 输入拦截 | `common/policies.py:356-363` | 若输入是 `(obs, info)` 这种长度为 2 且第二个元素是 dict 的 tuple，则抛 `ValueError` 并提示"你在混合 Gym API / VecEnv API" | **失败路径**之一；测试 `tests/test_predict.py:121 test_mixing_gym_vecenv_api` 专门验证这一异常 |
| 2b | `BasePolicy.obs_to_tensor(obs)` | `common/policies.py:236-277` | 对 ndarray / Dict 两种输入：① 图像 maybe_transpose；② reshape 补齐 batch 维 `(1, *obs_shape)`；③ `obs_as_tensor()` 转到 device 上成为 torch.Tensor；返回 `(obs_tensor, vectorized_env)` | `vectorized_env=False` 的唯一来源就是 reshape 之前没有 batch 维；它最后决定最终要不要 squeeze 掉 batch 这一维，输出 `shape=()` / 无 batch 维 |
| 2c | `ActorCriticPolicy._predict(obs_tensor, deterministic)` | `common/policies.py:709` | 等价于 `self.get_distribution(obs_tensor).get_actions(deterministic=...)` | 这里不走 `forward()`（forward 还会算 value_net 的 V 值和 log_prob；推理不需要） |
| 2c-1 | `ActorCriticPolicy.get_distribution(obs)` | `common/policies.py:743-752` | `extract_features(obs, pi_features_extractor)` → `mlp_extractor.forward_actor(features)` → `_get_action_dist_from_latent(latent_pi)` | `extract_features` = `preprocess_obs` 然后 `FlattenExtractor(obs)`（对 CartPole 基本就是 flatten + identity 线性层 0）|
| 2c-2 | `_get_action_dist_from_latent(latent_pi)` | `common/policies.py:684-707` | `mean_logits = self.action_net(latent_pi)`，对 Discrete(2)：`CategoricalDistribution.proba_distribution(action_logits=mean_logits)` | 连续 Box、MultiDiscrete、MultiBinary、DiagGaussian、StateDependentNoise 都在这里分流 |
| 2c-3 | `CategoricalDistribution.get_actions(deterministic)` | `common/distributions.py` | `deterministic=True → th.argmax(logits, dim=-1)`；否则按 logits 分类采样 | 连续 Box 空间则会有 `squash_output=False → Normal.sample()` 输出，后续还要 clip |
| 2d | tensor → numpy + reshape | `common/policies.py:370` | `actions.cpu().numpy().reshape((-1, *action_space.shape))` | 对 Discrete(2)，之前 actions.shape 是 `(1,)`，reshape 后仍是 `(1,)`（shape 匹配）|
| 2e | Box 空间 clip / unscale（本场景跳过） | `common/policies.py:372-379` | `squash_output → unscale_action`；否则 `np.clip(actions, low, high)` | CartPole Discrete **不进入本分支**；若把脚本换成 Pendulum-v1 就能看到 |
| 2f | 非 vectorized 时 squeeze 掉 batch 维 | `common/policies.py:382-384` | `if not vectorized_env: actions = actions.squeeze(axis=0)` | 这一步把 `shape=(1,)` 的 numpy 变成 `shape=()` 标量；如果用户一开始就给 `shape=(n,4)` 的 obs（已 batch），则不会执行这一步 |
| 3 | 用户 `env.step(int(action))` | `gymnasium` 标准 5 元组 | 返回 `(new_obs, reward, terminated, truncated, info)` | `terminated` 与 `truncated` 在 SB3 内部被合并成 `dones = terminated | truncated`（见 T04 collect_rollouts）|

### 4.1.3 两个 API 契约差异（原生 Gymnasium 与 SB3 VecEnv）

这是让 predict 出错率最高的知识点，必须在本章单独列出，与 T05 的"失败路径"互相呼应：

| 行为 | 原生 Gymnasium Env（用户直接 `gym.make` 得到的那个） | SB3 VecEnv（`model.get_env()` 返回的那个，默认包了 DummyVecEnv）|
|---|---|---|
| `reset()` 返回 | `(obs, info)` 2 元组 | **单独一个 obs** ndarray，shape 是 `(n_envs, *obs_shape)`；info 要另取 |
| `step(action)` 返回 | `(new_obs, reward, terminated, truncated, info)` 5 元组 | `(new_obs, rewards, dones, infos)` 4 元组；`dones[i] = terminated[i] or truncated[i]` |
| 动作维度 | `action.shape == action_space.shape` | `action.shape == (n_envs, *action_space.shape)` |
| 典型 predict 写法 | `action, _ = model.predict(single_obs, ...)` | `actions, _ = model.predict(vec_obs, ...)` 直接是 batch |

### 4.1.4 运行与测试证据

- **运行证据**：`scripts/issue_03_predict_trace.py`，复现结果见 `artifacts/runs/issue-03/2026-10-09-predict-trace/output.log`。脚本把上述 2a ~ 2f 的每一步手动展开，并在最后做了 `assert model.predict(obs) == 手动展开结果` 的一致性断言。运行环境：Python 3.12.13，SB3 2.9.2a0（固定 commit），gymnasium 1.4.0，torch 2.14.1，seed=0。
- **测试证据**：
  - `tests/test_predict.py:59 test_predict` —— 同一 model 既能吃单个 obs（shape 匹配 `action_space.shape` 且 `.contains(action) == True`），又能吃 2 条 env 的 batch obs（返回 shape[0] == batch）。
  - `tests/test_predict.py:121 test_mixing_gym_vecenv_api` —— 验证"把 reset 结果 tuple 直接传入 predict 会抛 ValueError"。
  - `tests/test_train_eval_mode.py:346 test_predict_with_dropout_batch_norm` —— predict 期间即使网络内部有 BN+Dropout，多次同 obs 的预测也要 `np.testing.assert_allclose` 全部一致；同时 batch_norm 统计量不变，证明 predict 不会修改网络状态。

### 4.1.5 常见失败路径清单（来源于 C-03-03 + 测试反例）

| 错误写法 | 实际会发生什么 | 如何在源码里定位 |
|---|---|---|
| `obs = env.reset(); action = model.predict(obs)`（忘了 reset 还会返回 info，结果 obs 是 2 元组） | 抛 `ValueError: You have passed a tuple to the predict() function ... mixing Gym API` | `common/policies.py:356-363` |
| `new_obs, reward, done, info = env.step(a)` 再把这个 4 元组传 predict（Gym<0.26 旧 API + 新 Gymnasium 混写） | 同上 tuple 检查也会误触发；或者 user 根本拿不到 `terminated/truncated` | `common/policies.py:356`；SB3 文档在 `VecEnv API vs Gym API` 一节有说明 |
| 把 `shape=(4,)` 的 obs 重复 `np.stack([obs, obs])` 得到 `(2,4)`，再 predict 后直接对返回的 action 做 `env.step(action[0])` 这种正确写法没问题，但容易犯「取整标量错拿 actions[0][0]」一类错误 | 对 Discrete(2) 这种 step() 可能报 `int 不能迭代` 之类环境内部错，不在 SB3 里 | 环境自身 check（T05 会深读 env_checker.py）|

---

## 4.2 训练调用链：`learn() → collect_rollouts() → RolloutBuffer → train()`

**问题**：用户调一次 `model.learn(total_timesteps=128_000)` 在软件控制流层面到底怎么推进？数据从 env 流到 buffer、再从 buffer 流到 optimizer 的顺序是什么、哪里是可配置点？注意：本章只讲**控制流与协作对象**，不推导 PPO loss 公式、不解释 GAE 数学、不证明收敛。

### 4.2.1 对象协作总览（见 Issue #4 图稿）

Mermaid 源：[`artifacts/diagrams/issue-04/learn-call-chain.mmd`](file:///Users/dingquan/it/trae/project/sb3-llm-code-understanding/artifacts/diagrams/issue-04/learn-call-chain.mmd)。图形形状分 4 种色块：
- 黄色：一次性初始化 `_setup_learn`
- 蓝色：每轮 rollout 采样
- 红色：每轮 PPO.train() 参数更新
- 绿色：返回与结束

### 4.2.2 逐步复述（PPO / OnPolicyAlgorithm 场景）

```python
model = PPO("MlpPolicy", "CartPole-v1", n_steps=2048, batch_size=64, n_epochs=10)
model.learn(total_timesteps=1_000_000)
```

#### Phase A：构造器 `PPO.__init__()`
- `PPO.__init__`（`ppo/ppo.py:80-171`）先 `super().__init__(...)` 跳到 `OnPolicyAlgorithm.__init__` 并传 `_init_setup_model=False`，因此暂时不建网络。
- 接下来做 3 个断言：
  - `normalize_advantage and batch_size > 1`（否则归一化会除 0 得 NaN），
  - `n_envs * n_steps > 1`（同上，防止 advantage normalization 退化），
  - 若 `n_steps × n_envs % batch_size != 0`，`warnings.warn` 提示最后一个 minibatch 会被截断，建议换 batch_size 为 buffer_size 的因数。
- 最后 `_init_setup_model=True` → 调用 `PPO._setup_model()`（`ppo/ppo.py:173-183`）把 `clip_range` / `clip_range_vf` 包成 `FloatSchedule`；父类 `OnPolicyAlgorithm._setup_model`（`on_policy_algorithm.py:115-141`）则真正：
  - 创建 `lr_schedule`、`set_random_seed`
  - 根据 obs_space 选 `RolloutBuffer / DictRolloutBuffer`，实例化 buffer_size=`n_steps`（注意不是乘 n_envs，因为 buffer 自身 shape 第二位就是 n_envs；这点容易误解）
  - `self.policy = self.policy_class(obs_space, act_space, lr_schedule, ...)`，PPO 默认 `policy_aliases["MlpPolicy"] = ActorCriticPolicy`（`ppo/ppo.py:74-78`）

#### Phase B：`learn()` 入口（`PPO.learn` 透传 → `OnPolicyAlgorithm.learn`）
- `PPO.learn`（`ppo/ppo.py:302`）只是 `super().learn(...)`，真正主循环在 `common/on_policy_algorithm.py:300-342`。
- 首件事是 `total_timesteps, callback = self._setup_learn(...)`（`base_class.py:382-435`）：
  1. `start_time = time.time_ns()`
  2. 初始化 `ep_info_buffer`、`ep_success_buffer` 两个 `deque(maxlen=stats_window_size)`，用于后面 dump_logs 算平均分/平均长度
  3. `_last_obs = self.env.reset()`、`_last_episode_starts = ones((n_envs,), dtype=bool)`。这两个字段贯穿 collect_rollouts 的每一步
  4. `_logger = utils.configure_logger(...)`；`callback.init_callback(self)`

#### Phase C：while 主循环（`num_timesteps < total_timesteps`）
每一次循环做 4 件事，严格按以下顺序：

1. **`collect_rollouts(env, callback, rollout_buffer, n_rollout_steps=n_steps)`**（`on_policy_algorithm.py:162-268`）
   - `policy.set_training_mode(False)`：**采样期间 policy 不更新、BN/dropout 关** → 这是测试 `test_a2c_ppo_collect_rollouts_with_batch_norm` 能断言 BN running_mean 不变的根因
   - `rollout_buffer.reset()`：`pos=0, full=False, generator_ready=False`
   - 若用 gSDE：`policy.reset_noise(n_envs)`；每 sde_sample_freq 步重置一次
   - `callback.on_rollout_start()`
   - 内层 while `n_steps < n_rollout_steps`：
     - `obs_as_tensor(_last_obs)` → `policy(obs_tensor)` 在 no_grad 下拿到 `(actions, values, log_probs)`（这里走的是 ActorCriticPolicy.forward()，推理链里的 predict 不会走到 forward）
     - Box 空间：`clip / unscale`
     - `env.step(clipped_actions)` → `new_obs, rewards, dones, infos`（VecEnv API，4 元组）
     - `num_timesteps += n_envs`；`callback.on_step()`；若 on_step 返回 False 立即整个 `return False` 提前结束训练（这是 EvalCallback、StopTrainingOnRewardThreshold 等回调的唯一中止路径）
     - **TimeLimit.truncated 特殊处理（GitHub #633）**：若 `done[i]=True` 但环境写入了 `TimeLimit.truncated=True`（比如 CartPole 在 500 步被自动截断而非真的结束），则用 `V(terminal_observation) × gamma` 当 bootstrap 加到 rewards[i]，避免把"截断"误当成"真实终止=0 价值"
     - `rollout_buffer.add(last_obs, actions, rewards, last_episode_starts, values, log_probs)`（`buffers.py:440`）→ 往 numpy 数组当前 pos 写
     - 更新 `_last_obs = new_obs; _last_episode_starts = dones`
   - 采满 n_steps 之后：`policy.predict_values(new_obs)` 拿 `last_values`，调 `rollout_buffer.compute_returns_and_advantage(last_values, dones)`（`buffers.py:403-439`）算 GAE；再 `callback.on_rollout_end()`
2. `iteration += 1`；`_update_current_progress_remaining(num_timesteps, total)` → `_current_progress_remaining = 1 - n/N`，后面 `lr_schedule`、`clip_range`、`clip_range_vf` 都靠这个数衰减
3. 可选：`if iteration % log_interval == 0: self.dump_logs(iteration)`。输出字段：`rollout/ep_rew_mean`、`rollout/ep_len_mean`、`time/fps`、`time/time_elapsed`、`time/total_timesteps`、`rollout/success_rate`。除了 time/total_timesteps 指定 exclude="tensorboard" 之外都同时写 stdout+tb
4. `self.train()`（在 PPO 中重写；见下文 D 阶段）

#### Phase D：`PPO.train()`（`ppo/ppo.py:184-301`）参数更新
- 切回 `set_training_mode(True)`；用 lr_schedule 更新 optimizer.lr；取 `clip_range = FloatSchedule(progress_remaining)`
- 外重循环 `for epoch in range(n_epochs)`：
  - 内重循环 `for rollout_data in rollout_buffer.get(batch_size)`（`buffers.py:481` 是 Generator，每进一次先 `np.random.permutation(buffer_size)` 打乱 indices，再按 batch_size 切块 yield）
  - 对 Discrete：`actions = rollout_data.actions.long().flatten()`
  - `policy.evaluate_actions(rollout_data.observations, actions)` → `(values, log_prob, entropy)`（这步会重新走 extract_features、得到当前 policy 下的值函数与 log 概率；注意这里与采样时的 old_log_prob 可能已经不一样了，这是多 epoch 的必然）
  - 若 `normalize_advantage and len(advantages) > 1`：`advantages = (A - A.mean()) / (A.std() + 1e-8)`
  - `ratio = exp(log_prob - old_log_prob)`；`policy_loss = - min(A·r, A·clip(r, 1-ε, 1+ε)).mean()`
  - value_loss：若 `clip_range_vf != None`，做 value 的 clip（`rollout_data.old_values + clamp(values-old_values, -ε_v, +ε_v)`），否则直接 MSE(returns, values)
  - `entropy_loss = -mean(entropy)`（离散连续都有解析熵；没有就用 -log_prob 近似）
  - `total_loss = pg_loss + ent_coef·H + vf_coef·value_loss`
  - `approx_kl = mean(exp(log_ratio)-1 - log_ratio)`（逆向 KL 的 Schulman 近似，issue #417）。如果设了 `target_kl` 且近似值超过 `1.5 × target_kl` → 提前 break，放弃后续 epoch
  - Optimizer：`zero_grad()` → `loss.backward()` → `clip_grad_norm_(policy.parameters(), max_grad_norm)` → `step()`
- 循环完后 `self._n_updates += epoch`（每个 epoch +1，不是每个 batch）；`explained_variance(values.flatten(), returns.flatten())`；最后一长串 `logger.record(train/...)` 把 entropy_loss / pg_loss / value_loss / approx_kl / clip_fraction / explained_variance / std / n_updates / clip_range / clip_range_vf 写出来

#### Phase E：退出 while + callback.on_training_end() + return self

只要 `num_timesteps >= total_timesteps` 就跳出 while；注意 Issue #1150 提到：`total_timesteps` 只是下界，因为每一次 collect_rollouts 是采满 `n_steps × n_envs` 才退，所以最终 `num_timesteps` 会是第一个 ≥ total 的 `n_steps × n_envs` 倍数。

### 4.2.3 关键数据对象：RolloutBuffer 的形状与使用节奏

| 字段 | shape（CartPole-v1, n_steps=2048, n_envs=1） | 什么时候写 | 什么时候读 |
|---|---|---|---|
| `observations` | `(2048, 1, 4)` float32 | `RolloutBuffer.add` 每步一条 | `train()` 中 `rollout_buffer.get(batch_size)` yield 的每一批 observations |
| `actions`      | `(2048, 1, 1)` int64（Discrete 经过 reshape 成 1）| add 每步 | train 里 `actions.long().flatten()` 再送 evaluate_actions |
| `rewards`      | `(2048, 1)` float32 | add 每步（**含** TimeLimit.truncated 修正后的奖励）| compute_returns_and_advantage 作为 r_t |
| `episode_starts` | `(2048, 1)` bool | add 每步 | GAE 里 `next_non_terminal = 1.0 - dones` 的直接来源 |
| `values`       | `(2048, 1)` float32 | add 每步 | compute_returns_and_advantage 作为 V(S_t)；PPO.train 里 explained_variance 用 |
| `log_probs`    | `(2048, 1)` float32 | add 每步，来自 policy.forward 的 log_prob | train 里 `old_log_prob = rollout_data.log_prob` 算 ratio |
| `returns`      | `(2048, 1)` float32 | compute_returns_and_advantage 末尾 | train 的 value_loss 作为 supervision（TD-GAE target）|
| `advantages`   | `(2048, 1)` float32 | 同上 | train 里 surrogate loss 的权重 A |

### 4.2.4 运行与测试证据

- **运行证据**：`scripts/issue_04_learn_trace.py`，以 `n_steps=32, batch_size=16, n_epochs=2, total=128` 跑通（4 轮完整 rollout）。输出 `artifacts/runs/issue-04/2026-10-09-learn-trace/output.log`：
  - callback 事件时序：`training_start → (rollout_start → rollout_end buf_len=32) × 4 → training_end`，每轮结束步数严格增加 32
  - RolloutBuffer 形状与 4.2.3 表一致（n_steps=32，n_envs=1）
  - 训练前后 predict 动作分布不同（Section H）
- **测试证据**：
  - `tests/test_train_eval_mode.py:320 test_a2c_ppo_collect_rollouts_with_batch_norm` —— 证明 collect_rollouts 期间 BN running_mean 不变 → 采样阶段不更新任何参数
  - `tests/test_run.py:40 test_advantage_normalization` —— normalize_advantage 的 T/F 两种都能顺利 learn
  - `tests/test_run.py:48 test_ppo(clip_range_vf)` —— 负的 clip_range_vf 触发 AssertionError（C-04-10）

### 4.2.5 控制流的失败/提前退出路径

| 触发条件 | 在哪一步退出 | 行为 |
|---|---|---|
| 用户给的 callback 在某个 on_step() 里返回 False | collect_rollouts 内的 `if not callback.on_step(): return False` → 立刻跳出 while，learn() 随之 break | 这是 StopTrainingOnRewardThreshold、EvalCallback 早期结束训练的路径 |
| 设了 `target_kl` 且 epoch 某一批后 `approx_kl > 1.5 × target_kl` | PPO.train 最内层的 `continue_training=False; break`（只跳出当前 epoch 循环）| 打印"Early stopping at step X due to reaching max kl"；本轮 train() 结束但不影响外层 while，下次 rollout 继续 |
| `batch_size <= 1 且 normalize_advantage=True` | PPO.__init__ 断言 AssertionError | 完全不进入训练，参数初始化阶段就失败 |
| `n_steps × n_envs == 1 且 normalize_advantage=True` | 同上 AssertionError | 同上 |
| `clip_range_vf < 0` 被传入 | `PPO._setup_model`（`ppo/ppo.py:179`）AssertionError | |
| 传入 env 的 observation_space / action_space 与已训练模型不一致（继续训练/加载模型 set_env 场景）| `BaseAlgorithm.set_env` 中 `check_for_correct_spaces` 抛 AssertionError | 见 base_class.py:497 |

---

## 4.3 成功 vs 失败路径的交叉参考（与 T05 呼应）

本章描述的都是"合法输入下的成功调用链"；若用户自己写的 Env 不符合契约，调用链会在以下位置提前炸掉，分别对应 T05 证据里的条目：

| 阶段 | 位置 | 检查什么 | T05 对应失败脚本（假设存在）|
|---|---|---|---|
| PPO.__init__ / set_env | `base_class.py:168 env = _wrap_env` 后 → `check_for_nested_spaces` | 不支持嵌套 Dict obs space | issue_05_nested_dict.py |
| PPO.__init__ | L180-202 | `supported_action_spaces` 断言；MlpPolicy + Dict space 报错；Box 必须有 finite bounds | |
| PPO.__init__（assert）| ppo/ppo.py:140-151 | normalize_advantage × buffer_size 断言 | issue_05_batchsize_1_assert.py |
| model.learn 第 1 次 step | env_checker.py（如果 env 是用户自定义的且未通过 check_env）| step 返回值、obs/action 形状、奖励标量等 —— T05 深挖 | issue_05_step_returns_4.py（返回 4 个不是 5 个）|
| rollout_buffer.add | buffers.py:460-479 | pos 越界、类型不一致 | 通常不会触发，除非手动改 |

> 注意：如果用户用 `gym.make("CartPole-v1")` 这种官方环境，env_checker 通常不需要跑；T05 的失败脚本都是**故意构造错误的自定义 Env**来验证 checker 会在对应位置抛错、给出可读异常信息、以及源码中行号对应哪一类契约。
