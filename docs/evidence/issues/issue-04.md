# Issue #4：learn() 框架级训练调用链

负责人：@dingdadabset  
复核人：@Jaschenn
上游基线：见 `docs/project-baseline.md`（stable-baselines3 commit `7cfb4dd6055e74b5caa4ed4d6777492209946e26`）

## 可检验结论

| 本 Issue 内编号 | 结论 | 状态 | 证据链接 |
|---|---|---|---|
| C-04-01 | `PPO.learn()` 本身不做任何实现，只是一行 `return super().learn(...)` 透传；真正的训练主循环在 `OnPolicyAlgorithm.learn()` 中 | 已支持 | `ppo/ppo.py:302-318`；运行脚本 Section B、C 明确列出该继承链 |
| C-04-02 | 主循环进入前会先执行 `BaseAlgorithm._setup_learn()`，一次性完成：`start_time`、ep_info_buffer/ep_success_buffer deque、首次 `env.reset()` 写入 `_last_obs`、logger + callback 初始化 | 已支持 | `common/base_class.py:382-435`；运行脚本 Section B 逐行解释 |
| C-04-03 | `OnPolicyAlgorithm.learn()` 的主循环严格按「while num_timesteps < total: (1) collect_rollouts → (2) iteration++ + 进度更新 → (3) 可选 dump_logs → (4) train()」顺序执行，没有并行步骤 | 已支持 | `common/on_policy_algorithm.py:323-338`；运行脚本 Section F 用 callback 事件序 + 累计步数打印，可观测 4×(rollout_start → rollout_end buf_len=32)，每轮结束后 num_timesteps += 32；最终 `num_timesteps = 128` 与 `TOTAL_TIMESTEPS` 严格相等 |
| C-04-04 | `collect_rollouts()` 每一轮严格采满 `n_steps × n_envs` 条 transition（除非 callback on_step 返回 False），采集中始终把 policy 设为 eval 模式且在 `no_grad()` 里走 `policy(obs_tensor)`，不会更新任何参数 | 已支持 | `common/on_policy_algorithm.py:162-268`；测试 `tests/test_train_eval_mode.py:320-341 test_a2c_ppo_collect_rollouts_with_batch_norm` 对含 BatchNorm 的 policy 连续 2 次 collect_rollouts 后断言 batch_norm running_mean 完全不变，证明 collect_rollouts 阶段不做参数更新 / BN 不切 train 模式 |
| C-04-05 | reset 后 observations 为 `(n_steps,n_envs,*obs_shape)`，actions 为 `(n_steps,n_envs,action_dim)`，rewards/returns/advantages/log_probs 为 `(n_steps,n_envs)`；reset 重新分配归零数组并重置 pos/full/generator_ready；get 首次将部分字段展平 | 已支持 | `common/buffers.py:391-401,481-507`；修正版 Section E 直接断言新旧 observations 非同一对象、新数组全零；本例 observations=(32,1,4)、actions=(32,1,1)，标量字段=(32,1) |
| C-04-06 | `collect_rollouts` 的收尾步骤 ⑦ 执行 `rollout_buffer.compute_returns_and_advantage(last_values, dones)` 算 GAE（含 gamma/gae_lambda），写入 returns 与 advantages 两个字段；然后才 callback.on_rollout_end | 已支持 | `common/on_policy_algorithm.py:258-266` + `common/buffers.py:403-439 compute_returns_and_advantage` |
| C-04-07 | 若 transition 中 `done=True` 且 `infos[idx]['TimeLimit.truncated']=True`（也就是「因步数上限被截断而非真失败」），collect_rollouts 会特殊处理：把 bootstrap 的 `gamma*V(terminal_obs)` 加到 reward[idx]，避免把截断误判为 episode 终点 | 已支持 | `common/on_policy_algorithm.py:236-245` + 注释引用 GitHub issue #633 |
| C-04-08 | `PPO.train()` 的参数更新严格按：切 train mode → 更新学习率、计算 clip_range → **n_epochs × 每 epoch 分 minibatch 遍历 rollout_buffer.get(batch_size)** → 每 batch 算 policy_loss(clip surrogate) + value_loss(可选 V clip) + entropy_loss → 反向传播 + grad_norm clip → optimizer.step → 记录 explained_variance + train/* logger 指标 | 已支持 | `ppo/ppo.py:184-301`；运行脚本 Section G 逐段对应 |
| C-04-09 | train() 中的 minibatch 是 `RolloutBuffer.get(batch_size)` yield 出来的；每个 minibatch 内部会做 advantages 归一化（normalize=(A-mean)/(std+1e-8)），但 batch_size==1 时跳过 | 已支持 | `common/buffers.py:481-520` + `ppo/ppo.py:216-219`（`if normalize_advantage and len(advantages) > 1`）；测试 `tests/test_run.py:40-44 test_advantage_normalization` 分别跑 normalize_advantage=True/False 且两者都能正常 learn(64) 不崩 |
| C-04-10 | `PPO.__init__` 会对两个常见组合错误做断言：① `normalize_advantage=True` 时 `batch_size` 必须 >1；② 仅在 `normalize_advantage=True` 时要求 `n_steps × n_envs > 1`（否则 advantage normalization 会出 NaN）；两者的 buffer_size 若不是 batch_size 的倍数还会给 warning 提示 | 已支持 | `ppo/ppo.py:140-162`；测试 `tests/test_run.py:48-59 test_ppo` 对 `clip_range_vf=-0.2` 抛 `AssertionError`（与同段 init 断言风格一致） |
| C-04-11 | 本次 learn 后至少一个 policy 参数张量与 learn 前快照不同；参数更新不保证回报提升 | 已支持，限定本次运行 | 修正版 Section H 对 named_parameters 的 learn 前后快照用 torch.equal 比较并断言存在变化；评估前模型恢复该参数快照，分别报告前后均分及真实差值，见文末复核运行 |
| C-04-12 | `dump_logs(iteration)` 在 `log_interval ≠ None and iteration % log_interval == 0` 时执行，会把 ep_info_buffer 里 `r/l` 统计成 `rollout/ep_rew_mean` / `rollout/ep_len_mean`；`fps` 由 `(num_timesteps - start) / elapsed` 计算；`time/total_timesteps` 排除 TensorBoard，其余输出取决于 logger 配置（exclude="tensorboard"） | 已支持 | `common/on_policy_algorithm.py:277-299`；log_interval=1 时每一轮 rollout 结束就会 dump 一次 |

## 证据记录

### C-04-01（PPO.learn 只是 super 透传）

源码：`ppo/ppo.py:302-318`：
```python
def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="PPO", ...):
    return super().learn(...)
```
继承链：`PPO → OnPolicyAlgorithm → BaseAlgorithm`。

### C-04-02（_setup_learn 四件事）

源码 `common/base_class.py:382-435`：
1. `start_time = time.time_ns()`
2. `self.ep_info_buffer = deque(maxlen=self._stats_window_size)` （默认 100）
3. `self._last_obs = self.env.reset()`（非 VecEnv 这里拿到的是 DummyVecEnv 的 obs，已经是 batch 维），`_last_episode_starts = np.ones((self.n_envs,), dtype=bool)`
4. `self._logger = utils.configure_logger(...)`  + `callback = self._init_callback(callback, progress_bar)`

### C-04-03（主循环顺序：rollout → dump → train）

源码 `common/on_policy_algorithm.py:323-338`：
```
while self.num_timesteps < total_timesteps:
    continue_training = collect_rollouts(...)
    if not continue_training: break
    iteration += 1
    _update_current_progress_remaining(...)
    if iteration % log_interval == 0: dump_logs(iteration)
    self.train()
```
运行证据：`artifacts/runs/issue-04/2026-10-09-learn-trace/output.log` Section F。关键输出：
```
rollout_start  timesteps=0
rollout_end    timesteps=32   buf_len=32
rollout_start  timesteps=32
rollout_end    timesteps=64   buf_len=32
rollout_start  timesteps=64
rollout_end    timesteps=96   buf_len=32
rollout_start  timesteps=96
rollout_end    timesteps=128  buf_len=32
training_end   timesteps=128
```
每一对 (rollout_start → rollout_end) 之间严格采满 32 步，之后才能推进 `iteration` 并执行 `train()`。

### C-04-04（collect_rollouts 全程 eval + no_grad）

源码 `on_policy_algorithm.py:183-203`：
```python
self.policy.set_training_mode(False)
...
with th.no_grad():
    obs_tensor = obs_as_tensor(self._last_obs, self.device)
    actions, values, log_probs = self.policy(obs_tensor)   # __call__ → forward()
```
测试佐证：`tests/test_train_eval_mode.py:320-341` 在 A2C/PPO 使用 FlattenBatchNormDropoutExtractor 做了两次连续 collect_rollouts，断言 BatchNorm 的 `bias` 和 `running_mean` 都 `th.isclose(..., atol=0).all()`。若 collect_rollouts 开了 train mode，则 BN running_mean 会移动——这就矛盾了。故测试直接证伪。

### C-04-05（RolloutBuffer 形状）

源码 `common/buffers.py:343 RolloutBuffer.__init__`：
```python
self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)
self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
...
```
运行日志 Section E：`observations shape=(32,1,4)`, `actions shape=(32,1,1)`，与公式 `buffer_size=n_steps * n_envs?` 注意——这里 buffer_size 直接等于 `n_steps`（不是乘），因为第二维已经是 `n_envs`，所以总样本数还是 `n_steps × n_envs`。

### C-04-06（GAE 在 collect_rollouts 末尾调用）

源码 `on_policy_algorithm.py:258-262`：
```python
with th.no_grad():
    values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
```
对应 `common/buffers.py:403-439`：按从 `t=n_steps-1` 往 `t=0` 反推 `next_non_terminal × next_values` + delta，`advantages = gae`，`returns = advantages + values`。

### C-04-07（TimeLimit.truncated 引导修正）

源码 `on_policy_algorithm.py:236-245`：
```python
for idx, done in enumerate(dones):
    if done and infos[idx].get("terminal_observation") is not None \
             and infos[idx].get("TimeLimit.truncated", False):
        terminal_obs = policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
        with th.no_grad():
            terminal_value = policy.predict_values(terminal_obs)[0]
        rewards[idx] += self.gamma * terminal_value
```
引用 issue `#633`。原理：如果 CartPole 在 500 步被截断（CartPole 默认的 max_episode_steps=500），那么第 500 步不应该被当成「状态值为 0」的 episode 终点——应该被当作「继续还有价值」的中间状态，所以把 V(terminal_obs) 的 bootstrap 加回来。

### C-04-08（PPO.train 的控制流）

源码 `ppo/ppo.py:184-301`，对应运行脚本 Section G 的 ① ~ ⑥ 标号逐段对齐。重要边界：
- `self._n_updates` 在每个 epoch 末尾 +1（不是每个 minibatch），见 L280
- `target_kl` 触发阈值是 `1.5 * target_kl`（不是 1×），见 L267
- `explained_variance` 用整个 rollout_buffer.values vs returns 计算（L284）

### C-04-09 & C-04-10（batch 的 normalize + init 断言）

- `RolloutBuffer.get(batch_size)` 是 `common/buffers.py:481` 开始的生成器：先 shuffle indices，然后 slice 每 batch_size 段 yield。
- `ppo/ppo.py:140-141` 断言 `batch_size > 1`（当 normalize_advantage）
- `ppo/ppo.py:148-151` 断言 `n_envs × n_steps > 1`（同样是避免 NaN）
- `ppo/ppo.py:153-162` 当 buffer_size % batch_size != 0 时 `warnings.warn` 提示"最后一个 mini-batch 会被截断，建议换 batch_size 为 factor"
测试佐证：`tests/test_run.py:40-44` 跑 normalize_advantage in (False, True) 两种设置都能完成一次 `model.learn(64)`。

### C-04-11（train() 真的改参数）

原运行日志 Section H：同 seed=0 下：
```
训练前（新构造的 PPO）: 平均得分 = 222.3
训练后（learn 完 128 步）: 平均得分 = 79.0
```
原脚本只打印均分，没有保存动作序列或断言参数变化，且把训练后绝对得分误称“提升”。修正版在 learn 前保存参数快照，在 learn 后直接比较并断言至少一个张量变化；评估前模型恢复该快照。分数仅作行为观察，不把“分数上升”作为更新成功条件。修正后的完整输出见文末复核运行。

### C-04-12（dump_logs 统计来源）

`on_policy_algorithm.py:277-299`：
```python
rollout/ep_rew_mean = safe_mean([ep_info["r"] for ep_info in ep_info_buffer])
rollout/ep_len_mean = safe_mean([ep_info["l"] for ep_info in ep_info_buffer])
time/fps = (num_timesteps - start) / time_elapsed
time/total_timesteps  —  exclude="tensorboard"（排除 TensorBoard，其他输出取决于 logger 配置）
```
ep_info_buffer 中每个元素来自 Monitor wrapper 写入 infos dict 的 "episode" 键（含 r/l/t），见 `common/monitor.py`。

## AI 使用记录

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
