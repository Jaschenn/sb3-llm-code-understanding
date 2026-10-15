# Issue #3：predict() 端到端调用链

负责人：@dingdadabset  
复核人：@Jaschenn
上游基线：见 `docs/project-baseline.md`（stable-baselines3 commit `7cfb4dd6055e74b5caa4ed4d6777492209946e26`）

## 可检验结论

| 本 Issue 内编号 | 结论 | 状态（待验证/已支持/已否定/需限定） | 证据链接 |
|---|---|---|---|
| C-03-01 | `BaseAlgorithm.predict()` 只是把调用原封不动转发给 `self.policy.predict()`，没有额外逻辑 | 已支持 | `stable_baselines3/common/base_class.py:536-556`；运行日志脚本 `scripts/issue_03_predict_trace.py` 步骤 2 与步骤 2.7 的一致性断言（最后 assert 通过） |
| C-03-02 | `BasePolicy.predict()` 内部执行顺序固定为：切 eval 模式 → tuple 输入拦截 → obs_to_tensor → no_grad 下的 `_predict` → tensor2numpy+reshape → Box 空间 clip/unscale → 非 vectorized squeeze batch → return (action, state) | 已支持 | `stable_baselines3/common/policies.py:331-386`；运行脚本 step2 各子步骤逐条对应 |
| C-03-03 | 若用户把 `env.reset()` 返回的 `(obs, info)` tuple 直接喂给 `model.predict()`，SB3 会在 `BasePolicy.predict` 开头抛 `ValueError`，提示「mixing Gym API / VecEnv API」 | 已支持 | `stable_baselines3/common/policies.py:356-363`；测试 `tests/test_predict.py:121-127 test_mixing_gym_vecenv_api` 就是验证这一断言 |
| C-03-04 | `obs_to_tensor()` 的核心作用是：补齐缺失的 batch 维度（shape `(4,)` → `(1,4)`）并转 `torch.Tensor`；同时返回的 `vectorized_env` 标记决定最终是否 squeeze | 已支持 | `stable_baselines3/common/policies.py:236-277`；运行日志 step 2.2 里 `obs_raw shape=(4,)` → `obs_tensor shape=(1,4)`，`vectorized_env=False` |
| C-03-05 | `ActorCriticPolicy._predict()`（PPO 默认 policy 类）等价于 `get_distribution(obs).get_actions(det)`；其中 `get_distribution` 会走 extract_features → mlp_extractor.forward_actor → action_net(logits) → 构造 CategoricalDistribution | 已支持 | `common/policies.py:709-717`、`743-752`、`684-707`；运行脚本 step 2.3 明确打印 distribution 类型为 `CategoricalDistribution` 及 det=True 返回 `argmax(logits)` |
| C-03-06 | `predict(..., deterministic=True)` 对离散动作走 `argmax(logits)`；对连续 Box 如果设置了 `squash_output=True` 会走 `unscale_action()` 把 tanh 后的 `[-1,1]` 反变换回 `[low, high]`；否则 `clip(low,high)`；对 Discrete 这一步跳过 | 已支持 | `common/policies.py:372-384`；运行日志 step 2.5 明确打印「CartPole 是 Discrete，本步不执行」；对连续动作的分支在同一段源码里可直接阅读 |
| C-03-07 | 「原生 Gymnasium Env」 与 「SB3 VecEnv」 的关键 API 差 异：前者 `reset()` 返回 `(obs, info)` 2 元组、`step(a)` 返回 `(obs, r, terminated, truncated, info)` 5 元组；后者 `reset()` **只返回 obs**（形状已经加了 n_envs 批维）、`step(a)` 返回 `(obs, r, dones, infos)` 4 元组，且 dones=bool[] / infos=list[dict] | 需限定 | 运行日志 step1 vs 源码 `common/vec_env/base_vec_env.py`（见 `base_class.py:204 _wrap_env` 将非 VecEnv 包成 DummyVecEnv）；需要区分"训练时内部用的 VecEnv"和"用户自己持有的普通 Gym Env"两种上下文 |
| C-03-08 | `predict()` 入口切 eval 模式，仅 `_predict()` 调用位于 `torch.no_grad()` 内；上游测试中同一模型、同一 observation 的确定性预测结果一致，且 BatchNorm 统计量不变 | 已支持，限定于被测网络与运行条件 | `common/policies.py:352,367`；测试 `tests/test_train_eval_mode.py:346-380 test_predict_with_dropout_batch_norm`；本次复跑结果见文末 |

## 证据记录

### C-03-01（BaseAlgorithm.predict → policy.predict 转发）

源码：`common/base_class.py:536-556`
```python
def predict(self, observation, state=None, episode_start=None, deterministic=False):
    return self.policy.predict(observation, state, episode_start, deterministic)
```
仅此一行，无其它逻辑。运行验证：脚本 step 2.7 中手动展开 BasePolicy 各步骤得到的 `final_action` 与 `model.predict()` API 返回 `final_from_api` 通过 `assert final_from_api == final_action`（数值完全一致）。

### C-03-02（BasePolicy.predict 八步顺序）

源码：`common/policies.py:331-386`，运行脚本 `scripts/issue_03_predict_trace.py` step 2 每一节对应源码位置都直接在 section 标题上标注了行号。脚本输出文件：`artifacts/runs/issue-03/2026-10-09-predict-trace/output.log`。

### C-03-03（tuple 输入拦截）

源码：`common/policies.py:356-363`
```python
if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
    raise ValueError("You have passed a tuple to the predict() function instead of a Numpy array or a Dict. ...")
```
测试佐证：`tests/test_predict.py:121-127 test_mixing_gym_vecenv_api`
```python
wrong_obs = env.reset()  # returns (obs, info) tuple
with pytest.raises(ValueError, match=r"mixing Gym API"):
    model.predict(wrong_obs)
```

### C-03-04（obs_to_tensor 补批维）

源码：`common/policies.py:236-277`
```python
observation = observation.reshape((-1, *self.observation_space.shape))
obs_tensor = obs_as_tensor(observation, self.device)
```
运行日志：`artifacts/runs/issue-03/2026-10-09-predict-trace/output.log` step 2.2：
```
obs_raw  shape=(4,) dtype=float32
obs_tensor shape=(1, 4) dtype=torch.float32
vectorized_env=False
```

### C-03-05（_predict → get_distribution → Categorical → argmax）

源码：
- `common/policies.py:709-717`
- `common/policies.py:743-752` → `_get_action_dist_from_latent L684-707` 对 Discrete 走 `self.action_dist.proba_distribution(action_logits=...)`，
- 最终 `get_actions(det=True)` 在离散分布里是 `th.argmax(logits, dim=-1)`。
运行日志 step 2.3：distribution 类型显示 `CategoricalDistribution`，deterministic=True 返回 `value=[0]`。

### C-03-06（Box 空间 clip / unscale 分支）

源码：`common/policies.py:372-384`。CartPole Discrete(2) 不进入 Box 分支（日志 step2.5 明确标注）。对连续空间（Pendulum-v1 / Box）的行为通过源码可读：`squash_output → unscale_action`，否则 `np.clip(actions, low, high)`。这一点在 `common/policies.py:388-413` 的 scale/unscale 工具函数上也有完整断言。

### C-03-07（原生 Gym vs VecEnv 接口差异）

- 原生 Gymnasium：`reset() -> (obs, info)`，`step(a) -> (new_obs, reward, terminated, truncated, info)` （5 元组）
- SB3 VecEnv：`reset() -> obs` 直接返回 ndarray shape=(n_envs, *obs_shape)；`step(a) -> (obs, rewards, dones, infos)` （4 元组），terminated/truncated 合并为 dones
证据来源：
- 脚本 step1 实测 `env.reset()` 返回长度 2；step3 实测 `env.step()` 返回长度 5。
- `common/base_class.py:204 _wrap_env` 对非 VecEnv 做 `DummyVecEnv([lambda: env])`；`common/vec_env/dummy_vec_env.py` 源码可看到 step 返回 4 元组。
- BasePolicy.predict 中针对 tuple 的报错正是为了拦住「用户把 Gym reset 返回的 tuple 直接用」的常见错误，C-03-03 的测试证明这是一条高频踩坑路径。
限制：如果用户把原始 Env 单独用于推理，就不要混淆 API；若用户走 `model.get_env().reset()`（拿到的就是 VecEnv API），则 reset 直接拿 obs 没 info。

### C-03-08（predict 幂等性 + eval mode）

源码：
- `predict()` 入口 `set_training_mode(False)`（policies.py:352），调用 `_predict` 用 `with th.no_grad()`（L367）。
测试佐证：`tests/test_train_eval_mode.py:346-380 test_predict_with_dropout_batch_norm` 对含 BatchNorm+Dropout 的网络，做：
```python
first_pred, _ = model.predict(observation, deterministic=True)
for _ in range(5):
    pred, _ = model.predict(observation, deterministic=True)
    np.testing.assert_allclose(first_pred, pred)
```
并断言 batch_norm 的 running_mean 前后不变 → 证明 predict 不会改变 BatchNorm 统计量，也不会走 dropout 随机分支。

## AI 使用记录

| AI 编号 | 模型与提问原文/链接 | AI 回答摘要 | 验证动作 | 观察结果 | 修正后的表述 |
|---|---|---|---|---|---|
| AI-03-01 | GPT-4o-mini，「stable-baselines3 的 model.predict(obs) 到动作之间的调用顺序是怎样的？每层文件路径是啥？」 | AI 回答的顺序是：`BaseAlgorithm.predict → BasePolicy.predict → _get_action_dist_from_latent → forward`，并且说 BasePolicy.predict 内部直接走 forward() | 实际对比 policies.py 源码 + 自己在脚本里手动展开每一步 | ❌ AI 错误了两处：① `BasePolicy.predict` 不调用 `forward()`，而是调用 `_predict()`；② `_predict` 对 ActorCriticPolicy 来说是 `get_distribution().get_actions()`，forward() 是同时返回 actions+values+log_prob 的训练接口（predict 里不需要 value）。测试脚本 step 2.7 用 `assert final_action == model.predict()` 验证了不经过 forward 这条路径也能拿到完全一致的输出。 | 正确顺序：`BaseAlgorithm.predict → BasePolicy.predict → (eval mode, 输入校验) → obs_to_tensor → no_grad下 _predict → (get_distribution → … → Categorical.argmax) → numpy.reshape → Box分支 clip/unscale → (非vectorized时squeeze batch)`；forward() 只在 collect_rollouts 等训练阶段会用（因为需要 values 和 log_prob），推理链路 predict 不会触发 forward。 |
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
