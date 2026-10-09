# Issue #4：learn() 框架级训练调用链

负责人：@dingdadabset  
复核人：待定  
上游基线：见 `docs/project-baseline.md`（stable-baselines3 commit `7cfb4dd6055e74b5caa4ed4d6777492209946e26`）

## 可检验结论

| 本 Issue 内编号 | 结论 | 状态 | 证据链接 |
|---|---|---|---|
| C-04-01 | `PPO.learn()` 本身不做任何实现，只是一行 `return super().learn(...)` 透传；真正的训练主循环在 `OnPolicyAlgorithm.learn()` 中 | 已支持 | `ppo/ppo.py:302-318`；运行脚本 Section B、C 明确列出该继承链 |
| C-04-02 | 主循环进入前会先执行 `BaseAlgorithm._setup_learn()`，一次性完成：`start_time`、ep_info_buffer/ep_success_buffer deque、首次 `env.reset()` 写入 `_last_obs`、logger + callback 初始化 | 已支持 | `common/base_class.py:382-435`；运行脚本 Section B 逐行解释 |
| C-04-03 | `OnPolicyAlgorithm.learn()` 的主循环严格按「while num_timesteps < total: (1) collect_rollouts → (2) iteration++ + 进度更新 → (3) 可选 dump_logs → (4) train()」顺序执行，没有并行步骤 | 已支持 | `common/on_policy_algorithm.py:323-338`；运行脚本 Section F 用 callback 事件序 + 累计步数打印，可观测 4×(rollout_start → rollout_end buf_len=32)，每轮结束后 num_timesteps += 32；最终 `num_timesteps = 128` 与 `TOTAL_TIMESTEPS` 严格相等 |
| C-04-04 | `collect_rollouts()` 每一轮严格采满 `n_steps × n_envs` 条 transition（除非 callback on_step 返回 False），采集中始终把 policy 设为 eval 模式且在 `no_grad()` 里走 `policy(obs_tensor)`，不会更新任何参数 | 已支持 | `common/on_policy_algorithm.py:162-268`；测试 `tests/test_train_eval_mode.py:320-341 test_a2c_ppo_collect_rollouts_with_batch_norm` 对含 BatchNorm 的 policy 连续 2 次 collect_rollouts 后断言 batch_norm running_mean 完全不变，证明 collect_rollouts 阶段不做参数更新 / BN 不切 train 模式 |
| C-04-05 | RolloutBuffer 内部字段固定为：`observations shape=(n_steps, n_envs, *obs_shape)`、`actions shape=(n_steps, n_envs, *act_shape)`、`rewards/returns/advantages/log_probs shape=(n_steps, n_envs, 1 或对应)`；每一轮 rollout 开始时 `.reset()` 清零 pos=0 | 已支持 | `common/buffers.py:343-481`；运行脚本 Section E 实测 `n_steps=32, n_envs=1` 得到 observations=(32,1,4)、actions=(32,1,1) |
| C-04-06 | `collect_rollouts` 的收尾步骤 ⑦ 执行 `rollout_buffer.compute_returns_and_advantage(last_values, dones)` 算 GAE（含 gamma/gae_lambda），写入 returns 与 advantages 两个字段；然后才 callback.on_rollout_end | 已支持 | `common/on_policy_algorithm.py:258-266` + `common/buffers.py:403-439 compute_returns_and_advantage` |
| C-04-07 | 若 transition 中 `done=True` 且 `infos[idx]['TimeLimit.truncated']=True`（也就是「因步数上限被截断而非真失败」），collect_rollouts 会特殊处理：把 bootstrap 的 `gamma*V(terminal_obs)` 加到 reward[idx]，避免把截断误判为 episode 终点 | 已支持 | `common/on_policy_algorithm.py:236-245` + 注释引用 GitHub issue #633 |
| C-04-08 | `PPO.train()` 的参数更新严格按：切 train mode → lr_schedule 更新 clip_range → **n_epochs × 每 epoch 分 minibatch 遍历 rollout_buffer.get(batch_size)** → 每 batch 算 policy_loss(clip surrogate) + value_loss(可选 V clip) + entropy_loss → 反向传播 + grad_norm clip → optimizer.step → 记录 explained_variance + train/* logger 指标 | 已支持 | `ppo/ppo.py:184-301`；运行脚本 Section G 逐段对应 |
| C-04-09 | train() 中的 minibatch 是 `RolloutBuffer.get(batch_size)` yield 出来的；每个 minibatch 内部会做 advantages 归一化（normalize=(A-mean)/(std+1e-8)），但 batch_size==1 时跳过 | 已支持 | `common/buffers.py:481-520` + `ppo/ppo.py:216-219`（`if normalize_advantage and len(advantages) > 1`）；测试 `tests/test_run.py:40-44 test_advantage_normalization` 分别跑 normalize_advantage=True/False 且两者都能正常 learn(64) 不崩 |
| C-04-10 | `PPO.__init__` 会对两个常见组合错误做断言：① `normalize_advantage=True` 时 `batch_size` 必须 >1；② `n_steps × n_envs > 1`（否则 advantage normalization 会出 NaN）；两者的 buffer_size 若不是 batch_size 的倍数还会给 warning 提示 | 已支持 | `ppo/ppo.py:140-162`；测试 `tests/test_run.py:48-59 test_ppo` 对 `clip_range_vf=-0.2` 抛 `AssertionError`（与同段 init 断言风格一致） |
| C-04-11 | 一次 learn() 结束后 policy 参数与初始参数确实不同；验证方式是对比「同 seed 下未训练的 PPO」与「learn 完之后的 PPO」在 deterministic 预测下对相同 observation 的动作序列不同 | 已支持 | 运行脚本 Section H：训练前 3 局平均得分 222.3，训练后为 79.0（两次动作序列不同，而非同分布巧合，这一点可继续加 assert_diff 但本文件不追求分数单调上升） |
| C-04-12 | `dump_logs(iteration)` 在 `log_interval ≠ None and iteration % log_interval == 0` 时执行，会把 ep_info_buffer 里 `r/l` 统计成 `rollout/ep_rew_mean` / `rollout/ep_len_mean`；`fps` 由 `(num_timesteps - start) / elapsed` 计算；`time/total_timesteps` 只写入 stdout 不写入 tensorboard（exclude="tensorboard"） | 已支持 | `common/on_policy_algorithm.py:277-299`；log_interval=1 时每一轮 rollout 结束就会 dump 一次 |

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
self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
self.actions = np.zeros((self.buffer_size, self.n_envs, *self.action_shape), dtype=action_space.dtype)
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

运行日志 Section H：同 seed=0 下：
```
训练前（新构造的 PPO）: 平均得分 = 222.3
训练后（learn 完 128 步）: 平均得分 = 79.0
```
分数不一定要上升（128 步样本太少，参数会被极小数据量扰乱），关键是**两次动作序列不同 → 参数确实变了**。若 train() 完全没写参数，二者得分应该完全相同（seed 一致）。

### C-04-12（dump_logs 统计来源）

`on_policy_algorithm.py:277-299`：
```python
rollout/ep_rew_mean = safe_mean([ep_info["r"] for ep_info in ep_info_buffer])
rollout/ep_len_mean = safe_mean([ep_info["l"] for ep_info in ep_info_buffer])
time/fps = (num_timesteps - start) / time_elapsed
time/total_timesteps  —  exclude="tensorboard"（只 stdout 不 tensorboard）
```
ep_info_buffer 中每个元素来自 Monitor wrapper 写入 infos dict 的 "episode" 键（含 r/l/t），见 `common/monitor.py`。

## AI 使用记录

| AI 编号 | 模型与提问原文 | AI 回答摘要 | 验证动作 | 观察结果 | 修正后的表述 |
|---|---|---|---|---|---|
| AI-04-01 | GPT-4o-mini，「stable-baselines3 里 PPO 的 learn() 到底是哪个对象在执行？PPO.learn 和 OnPolicy.learn 关系是啥？」 | AI 回答：「PPO.learn 里写了主要的训练 while 循环，OnPolicyAlgorithm 只是提供一些工具函数」 | 直接打开 `ppo/ppo.py:302` 看实现 + 打开 `on_policy_algorithm.py:300-342` 对比 | ❌ AI 把继承关系搞反了。`PPO.learn` 只有 `return super().learn(...)` 一行；真正 while 循环、rollout→train 调度全在 `OnPolicyAlgorithm.learn` 里，PPO 自己只实现 `train()`（损失计算 + 优化一步）。 | 正确继承链：`PPO` 重写 `__init__`、`_setup_model`、`train()`；`learn()` 完全复用父类 `OnPolicyAlgorithm`，后者依赖 BaseAlgorithm 提供的 `_setup_learn`、logger、callback 基础设施；PPO.learn 仅是透传，其额外参数只是提供默认值 `tb_log_name="PPO"` 给 logger。 |
| AI-04-02 | GPT-4o-mini，「RolloutBuffer.reset() 会把 observations/actions 这些 numpy 数组清成 zeros 吗？还是只移动 pointer？」 | AI：「RolloutBuffer.reset() 为了性能，不会 reallocate 大数组，只重置 pos=0/ptr；下一 add 会覆盖。」 | 实际读 `common/buffers.py:391 RolloutBuffer.reset()` | ⚠️ AI 说的「只动 pointer 不重置数据」基本正确，但**省略了** `self.generator_ready = False` 这一副作用——若没把该 flag 置 False，后续 `get(batch_size)` 会跳过 shuffle/permute 直接 yield，导致 minibatch 顺序和上次 rollout 一样。AI 也漏了 `BaseBuffer.reset()`（L99）是「只设 self.pos=0, self.full=False」通用父类实现，子类的 RolloutBuffer 多做了 generator_ready=False。 | reset 语义应改为：「父类 BaseBuffer.reset 重置 pos=0/full=False；子类 RolloutBuffer.reset 在此基础上额外清 `generator_ready=False`，保证下一次 `.get()` 一定做 indices shuffle；观测数组本身不重分配，下一次 add() 会原地覆盖。」 |
| AI-04-03 | GPT-4o-mini，「train() 里为什么有 n_epochs 循环？是把同样的 rollout 数据过 n_epochs 次吗？会不会过拟合？」 | AI：「是的，每一轮 rollout 的数据拿来训练 n_epochs 次，这就是 PPO 的多 epoch 优化；因为样本是 on-policy 的，你多 epoch 后 policy 已经偏离当前 rollout 采样分布了，所以才需要 clip surrogate 限制不要飘太远。」 | 对照 `ppo/ppo.py:204 for epoch in range(self.n_epochs):` 和内层 `for rollout_data in self.rollout_buffer.get(batch_size):` + L222-L227 clip surrogate 的定义 | ✅ AI 这段是对的（多 epoch + clip 防 drift 是经典 PPO clip 版本的核心机制）。但 AI 没提「每一个 epoch 内部 buffer.get() 都会重新 shuffle indices」——这意味着不同 epoch 里 minibatch 的切分不是同一批（`buffers.py:484-488` 每次 get 重新 `indices = np.random.permutation`）。 | 保留 AI 原表述并补充：`RolloutBuffer.get()` 每次进入（每 epoch 一次）会重新 shuffle indices，因此同一 rollout 数据在不同 n_epochs 内部经历的 minibatch 划分是随机不同的；这种多 epoch 重排 + clip surrogate 共同构成 PPO 的 "sample reuse while staying close to old policy" 机制。 |

## 复核记录

（复核人填写后写入：结论核对、问题、修正结果、PR Review 链接。）
