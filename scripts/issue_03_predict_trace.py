"""
T03 / Issue #3：追踪一次 `model.predict(obs)` 从用户调用到动作输出的完整链路。

运行位置：上游 SB3 源码仓库的 venv 激活后执行。
输出：按步骤打印输入、关键中间变量 shape、方法名和源码定位。
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

# ============ 配置 ============
ENV_ID = "CartPole-v1"
SEED = 0
TOTAL_TIMESTEPS_FOR_TRAIN = (
    512  # 只训练很短一段，目的是让 policy 不是完全随机，便于观察行为
)

COLUMN = 110


def bar(title: str) -> None:
    left = (COLUMN - len(title) - 2) // 2
    right = COLUMN - len(title) - 2 - left
    print()
    print("=" * left + f"  {title}  " + "=" * right)


def section(msg: str) -> None:
    print(f"\n-- {msg} " + "-" * max(2, COLUMN - len(msg) - 4))


def pretty(name: str, value) -> None:
    if isinstance(value, th.Tensor):
        info = f"Tensor shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
        if value.numel() <= 8:
            info += f" value={value.detach().cpu().numpy().tolist()}"
        print(f"   {name:<30s}: {info}")
    elif isinstance(value, np.ndarray):
        info = f"ndarray shape={value.shape} dtype={value.dtype}"
        if value.size <= 8:
            info += f" value={value.tolist()}"
        print(f"   {name:<30s}: {info}")
    elif isinstance(value, (tuple, list)) and len(value) <= 6:
        print(f"   {name:<30s}: {type(value).__name__} len={len(value)} value={value}")
    else:
        s = repr(value)
        if len(s) > 120:
            s = s[:117] + "..."
        print(f"   {name:<30s}: {s}")


# =====================================================================
# 1. 训练一个极小的 PPO（或者直接加载）——这样 policy 有可观察的权重
# =====================================================================
bar("Step 0 · 准备环境 + 从相同种子训练（不读取缓存模型）")

th.set_num_threads(1)
env = gym.make(ENV_ID)
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    seed=SEED,
    device="cpu",
    n_steps=128,
    batch_size=64,
    n_epochs=2,
)
print(f"  开始学习 {TOTAL_TIMESTEPS_FOR_TRAIN} 步；本实验不评价策略表现。")
t0 = time.time()
model.learn(total_timesteps=TOTAL_TIMESTEPS_FOR_TRAIN, progress_bar=False)
print(f"  训练耗时 {time.time() - t0:.2f}s。")

# =====================================================================
# 2. 手动走一遍 env.reset()，拿到 observation
# =====================================================================
bar("Step 1 · 用户：env.reset() 拿到初始 observation")

reset_out = env.reset(seed=SEED)
# Gymnasium: returns (obs, info)
obs_raw, info_raw = reset_out
print(f"   Gymnasium.reset() 返回长度 = {len(reset_out)}")
pretty("obs_raw", obs_raw)
pretty("info_raw", info_raw)
print(
    "   对比 VecEnv ：SB3 内部将 env 包到 DummyVecEnv 后，"
    " vec_env.reset() 只返回 observation（形状加了 batch 维度），**没有 info**。"
)
pretty("env.observation_space", env.observation_space)
pretty("env.action_space", env.action_space)

# =====================================================================
# 3. 进入 BaseAlgorithm.predict() → policy.predict()
# =====================================================================
bar("Step 2 · 用户：model.predict(obs_raw) —— 进入 BaseAlgorithm.predict()")

# 为了让读者能对上源码，我们按照真实的调用顺序「手动展开」每一层并打印关键变量。
policy: ActorCriticPolicy = model.policy  # type: ignore[assignment]
policy.set_training_mode(False)
assert not policy.training
print("   手动展开先切 eval 模式，与 BasePolicy.predict 入口一致。")

section("2.1 输入合法性检查（BasePolicy.predict L356-L363）")
try:
    model.predict((obs_raw, info_raw), deterministic=True)
except ValueError as error:
    assert "mixing Gym API" in str(error)
    print(f"   ✓ 实际传入 (obs, info) 被拦截：{error}")
else:
    raise AssertionError("predict 未拒绝 Gymnasium reset 元组")

section(
    "2.2 obs_to_tensor()：numpy → torch.Tensor + 判断是否加了 batch 维（BasePolicy.predict L365 + L236）"
)
with th.no_grad():
    obs_tensor, vectorized_env = policy.obs_to_tensor(obs_raw)
pretty("输入 obs_raw", obs_raw)
pretty("输出 obs_tensor", obs_tensor)
pretty("vectorized_env（是否已经是 batch）", vectorized_env)

section(
    "2.3 进入 no_grad + _predict()：BasePolicy.predict L367-L368  → ActorCriticPolicy._predict L709"
)
with th.no_grad():
    distribution = policy.get_distribution(obs_tensor)
    actions_tensor = distribution.get_actions(deterministic=True)
pretty("distribution 类型", type(distribution).__name__)
categorical = distribution.distribution
pretty("Categorical.logits（归一化后的对数概率）", categorical.logits)
pretty("Categorical.probs（动作概率）", categorical.probs)
assert th.equal(actions_tensor, th.argmax(categorical.logits, dim=-1))
pretty("distribution.get_actions(deterministic=True)", actions_tensor)

section("2.4 tensor → numpy + reshape 回 action_space.shape（BasePolicy.predict L370）")
actions_np = actions_tensor.cpu().numpy().reshape((-1, *policy.action_space.shape))
pretty("actions_np（带 batch 维）", actions_np)

section(
    "2.5 对 Box 空间的动作再做 clip/unscale（CartPole 是 Discrete，这步跳过）（BasePolicy.predict L372-L379）"
)
print(
    f"   当前 action_space 类型 = {type(policy.action_space).__name__}，因此本步不执行。"
)

section(
    "2.6 非 vectorized 时 squeeze 掉 batch 维，得到最终动作（BasePolicy.predict L382-L384）"
)
if not vectorized_env:
    final_action = actions_np.squeeze(axis=0)
else:
    final_action = actions_np
pretty("final_action（最终返回给用户的值）", final_action)

section("2.7 实际 model.predict() 调用 —— 对比展开结果")
final_from_api, state_out = model.predict(obs_raw, deterministic=True)
pretty("model.predict 返回的 action", final_from_api)
assert final_from_api == final_action, (
    f"手动展开和 api 结果不一致：{final_from_api} vs {final_action}"
)
assert env.action_space.contains(final_from_api)
print("   ✓ 手动展开结果 == model.predict() API 返回，链路一致。")

# =====================================================================
# 4. env.step(action) —— 闭环
# =====================================================================
bar("Step 3 · 用户：env.step(final_action) —— 环境推进并返回新观察")
step_out = (
    env.step(int(final_from_api))
    if ENV_ID == "CartPole-v1"
    else env.step(final_from_api)
)
new_obs, reward, terminated, truncated, step_info = step_out
try:
    legacy_obs, legacy_reward, legacy_done, legacy_info = step_out
except ValueError as error:
    assert "too many values to unpack" in str(error)
    print(f"   ✓ 四变量解包在用户代码处失败，不会进入 predict：{error}")
else:
    raise AssertionError("Gymnasium 五项 step 返回值不应被四变量解包")
print(f"   step() 返回长度 = {len(step_out)}（Gymnasium 标准 5 元组）")
pretty("new_obs", new_obs)
pretty("reward", reward)
pretty("terminated", terminated)
pretty("truncated", truncated)
pretty("step_info", step_info)

# =====================================================================
# 5. 附上每一步对应源码位置，方便引用到证据文件
# =====================================================================
bar("Step 4 · 调用链清单（按真实执行顺序）")
chain = [
    ("用户脚本", "model.predict(obs)", "model = PPO.load(...) 或 PPO(...).learn(...)"),
    (
        "common/base_class.py:536",
        "BaseAlgorithm.predict(observation,...)",
        "直接转发到 self.policy.predict(...)",
    ),
    (
        "common/policies.py:331",
        "BasePolicy.predict(observation,...)",
        "切 eval，输入检查，obs→tensor，调用 _predict，reshape，squeeze",
    ),
    (
        "common/policies.py:236",
        "BasePolicy.obs_to_tensor(observation)",
        "ndarray/Dict → 加 batch 维 → 转 torch tensor",
    ),
    (
        "common/policies.py:709",
        "ActorCriticPolicy._predict(obs_tensor, det)",
        "→ self.get_distribution(obs).get_actions(det)",
    ),
    (
        "common/policies.py:743",
        "ActorCriticPolicy.get_distribution(obs)",
        "extract_features → mlp_extractor.forward_actor → action_net → Categorical",
    ),
    (
        "common/policies.py:122",
        "BaseModel.extract_features(obs, featex)",
        "preprocess_obs 然后 features_extractor(obs)",
    ),
    (
        "common/distributions.py",
        "CategoricalDistribution.get_actions(det)",
        "det=True 用 argmax；否则按 logits 采样",
    ),
    (
        "common/policies.py:370",
        "BasePolicy.predict —— 把 tensor 转 numpy + reshape",
        "Box 空间在此 clip 或 unscale",
    ),
    (
        "common/policies.py:382",
        "BasePolicy.predict —— 非 vectorized  squeeze(batch)",
        "输出最终 shape=(action_space.shape,)",
    ),
    (
        "用户脚本",
        "env.step(action)",
        "Gymnasium 标准接口，得到 new_obs, reward, terminated, truncated, info",
    ),
]
for file, func, note in chain:
    print(f"   {file:<40s}  {func:<50s}  {note}")

env.close()
wrapped_env = model.get_env()
if wrapped_env is not None:
    wrapped_env.close()

bar("End · T03 predict trace 完成")
