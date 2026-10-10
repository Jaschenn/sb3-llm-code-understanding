"""
T04 / Issue #4：追踪 `model.learn(total_timesteps)` 在软件结构层面的执行路径。

不推导 PPO loss、GAE 或收敛性质，只记录：主循环顺序、数据对象协作、
各阶段方法名和源码位置、关键状态的形状。
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch as th

UPSTREAM = os.path.expanduser("~/it/trae/_sb3_upstream/stable-baselines3")
if os.path.isdir(UPSTREAM):
    sys.path.insert(0, UPSTREAM)

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.buffers import RolloutBuffer  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

ENV_ID = "CartPole-v1"
SEED = 0
# 故意用很小的参数，这样一眼能看清循环节奏
N_STEPS = 32         # 每轮 rollout 每个 env 走多少步
BATCH_SIZE = 16      # train() 里每个 minibatch 多大
N_EPOCHS = 2         # rollout 数据复用多少次
TOTAL_TIMESTEPS = 128  # 主循环退出条件：N_STEPS * 4 轮刚好够跑
LOG_DIR = "/tmp/t04_sb3_tb/"

COLUMN = 120

# 为了看清楚每个阶段的进入，用 callback 在关键节点打时间戳
class PhaseCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.events: list[tuple[float, int, str]] = []

    def _on_training_start(self) -> None:
        self.events.append((time.time(), self.model.num_timesteps, "training_start"))

    def _on_rollout_start(self) -> None:
        self.events.append((time.time(), self.model.num_timesteps, "rollout_start"))

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.events.append((time.time(), self.model.num_timesteps, f"rollout_end  buf_len={self.model.rollout_buffer.size()}"))

    def _on_training_end(self) -> None:
        self.events.append((time.time(), self.model.num_timesteps, "training_end"))


def bar(title: str) -> None:
    left = (COLUMN - len(title) - 2) // 2
    right = COLUMN - len(title) - 2 - left
    print()
    print("=" * left + f"  {title}  " + "=" * right)


def section(msg: str) -> None:
    print(f"\n── {msg} " + "─" * max(2, COLUMN - len(msg) - 4))


def pretty(name: str, value) -> None:
    if isinstance(value, th.Tensor):
        info = f"Tensor shape={tuple(value.shape)} dtype={value.dtype}"
        if value.numel() <= 10:
            info += f" ≈ {value.detach().float().mean().item():.4f}"
        print(f"   {name:<30s}: {info}")
    elif isinstance(value, np.ndarray):
        info = f"ndarray shape={value.shape} dtype={value.dtype}"
        if value.size <= 10:
            info += f" ≈ mean={value.mean():.4f}"
        print(f"   {name:<30s}: {info}")
    elif hasattr(value, "buffer_size"):
        print(f"   {name:<30s}: {type(value).__name__} buffer_size={value.buffer_size} n_envs={getattr(value,'n_envs','?')} obs_space={value.observation_space} act_space={value.action_space}")
    else:
        s = repr(value)
        if len(s) > 80:
            s = s[:77] + "..."
        print(f"   {name:<30s}: {s}")


# =============================================================
# 1. 构造一个 PPO，但让我们可以 inspect 每一步到底发生了什么
# =============================================================
bar("T04 · 框架级训练调用链追踪（CartPole-v1 + PPO MlpPolicy）")

env = gym.make(ENV_ID)
cb = PhaseCallback()

section("A. PPO.__init__ → 内部 _setup_model()（on_policy L115）")
t0 = time.time()
model = PPO(
    "MlpPolicy", env, verbose=0, seed=SEED,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    gamma=0.99, gae_lambda=0.95,
    ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
    normalize_advantage=True,
    tensorboard_log=None,
)
pretty("model 类型", type(model).__name__)
pretty("model.policy 类型", type(model.policy).__name__)
pretty("model.rollout_buffer", model.rollout_buffer)
pretty("n_steps（每轮 rollout 步数/env）", model.n_steps)
pretty("batch_size（minibatch 大小）", model.batch_size)
pretty("n_epochs（每轮 rollout 训练几轮）", model.n_epochs)
pretty("总 buffer 行数 = n_steps × n_envs", model.n_steps * model.n_envs)
print(f"   结论：__init__ 时 _setup_model 已经初始化了 lr_schedule、policy、rollout_buffer。耗时 {time.time()-t0:.3f}s")

section("B. learn() 入口（on_policy L300 → 先调用 BaseAlgorithm._setup_learn L382）")
print("   learn 签名：learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)")
print("   _setup_learn 做 4 件事：")
print("   ① time.time_ns() → start_time")
print("   ② (re)set deques ep_info_buffer / ep_success_buffer（默认 window=100）")
print("   ③ 首次或 reset=True 时调用 env.reset() 初始化 self._last_obs，self._last_episode_starts=ones")
print("   ④ 配置 logger、callback.init_callback，返回 (total_timesteps, callback)")
print("   PPO.learn (ppo L302) 只是 super().learn 透传；真正的主循环在 OnPolicyAlgorithm.learn。")

section("C. learn() 主循环（on_policy L323-L338）— 伪代码：")
print("""
   iteration = 0
   total_timesteps, callback = self._setup_learn(...)
   callback.on_training_start(locals(), globals())
   while self.num_timesteps < total_timesteps:
       if not self.collect_rollouts(env, callback, rollout_buffer, n_rollout_steps=self.n_steps):
           break
       iteration += 1
       self._update_current_progress_remaining(num_timesteps, total)
       if iteration % log_interval == 0: self.dump_logs(iteration)
       self.train()                          # ← PPO.train() 在这里真正更新参数
   callback.on_training_end()
   return self
""")

section("D. collect_rollouts() 内部做了什么（on_policy L162）")
print("   ① policy.set_training_mode(False) （预测前关 dropout）")
print("   ② rollout_buffer.reset()")
print("   ③ gSDE 则 policy.reset_noise(n_envs)")
print("   ④ callback.on_rollout_start()")
print("   ⑤ for n_steps in 0..n_rollout_steps-1:")
print("        · obs_as_tensor(_last_obs)  （numpy → torch）")
print("        · with no_grad: actions, values, log_probs = policy(obs_tensor)")
print("        · Box 空间：unscale/clipped_actions = clip/unscale")
print("        · new_obs, rewards, dones, infos = env.step(clipped_actions)")
print("        · num_timesteps += n_envs ; callback.on_step()")
print("        · _update_info_buffer(infos, dones) — 把 Monitor 的 r/l 放进 ep_info_buffer deque")
print("        · for idx, done: TimeLimit.truncated 特殊处理：reward += gamma * V(terminal_obs)（issue 633）")
print("        · rollout_buffer.add(last_obs, actions, rewards, last_episode_starts, values, log_probs)")
print("        · _last_obs = new_obs  ;  _last_episode_starts = dones")
print("   ⑥ 最后一步：with no_grad: last_values = policy.predict_values(obs_as_tensor(new_obs))")
print("   ⑦ rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)")
print("   ⑧ callback.on_rollout_end() ；return True")

section("E. rollout_buffer：关键字段（common/buffers.py:343 RolloutBuffer）")
rb: RolloutBuffer = model.rollout_buffer
pretty("observations 形状（未 reset 前）", rb.observations)
pretty("actions 形状", rb.actions if rb.actions is not None else "None（reset 后初始化）")
pretty("rewards", rb.rewards if rb.rewards is not None else "None")
pretty("returns", rb.returns if rb.returns is not None else "None")
pretty("advantages", rb.advantages if rb.advantages is not None else "None")
pretty("log_probs", rb.log_probs if rb.log_probs is not None else "None")

section("F. 真正开始 learn(TOTAL_TIMESTEPS)，并借助 callback 看每个阶段的顺序")
print(f"   TOTAL_TIMESTEPS={TOTAL_TIMESTEPS}, n_steps={N_STEPS} → 预期 rollout 轮数 ≈ { (TOTAL_TIMESTEPS + N_STEPS - 1) // N_STEPS }")
t1 = time.time()
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, log_interval=1, tb_log_name="t04_trace", reset_num_timesteps=True, progress_bar=False)
print(f"   learn 总耗时 {time.time()-t1:.3f}s；最终 num_timesteps = {model.num_timesteps}")
print("   callback 事件顺序（含累计步数）：")
t_ref = cb.events[0][0] if cb.events else 0
for t, n, desc in cb.events:
    print(f"     +{(t-t_ref)*1000:7.1f} ms  timesteps={n:<6d}  {desc}")

section("G. PPO.train() 内部（ppo L184）主要控制流")
print("   ① policy.set_training_mode(True) — 打开 train 模式")
print("   ② _update_learning_rate(policy.optimizer)  按 progress_remaining 线性衰减")
print("   ③ clip_range = self.clip_range(progress_remaining)")
print("   ④ for epoch in range(n_epochs):        # 默认 10 次，这里设为 2")
print("        for rollout_data in rollout_buffer.get(batch_size):")
print("            · Discrete: actions.long().flatten()")
print("            · values, log_prob, entropy = policy.evaluate_actions(rollout_data.obs, actions)")
print("            · advantages normalize （mean=0, std=1；mini-batch level）")
print("            · ratio = exp(log_prob - old_log_prob)")
print("            · policy_loss = - min(adv*ratio, adv*clamp(ratio,1-ε,1+ε)).mean()")
print("            · value_loss: 可选 clip_range_vf；否则 MSE(returns, values_pred)")
print("            · entropy_loss: -entropy.mean()（或近似 -log_prob）")
print("            · loss = pg + ent_coef*entropy + vf_coef*value")
print("            · approx_kl = mean(exp(log_ratio)-1 - log_ratio)（逆向 KL，issue 417）")
print("            · 若 target_kl 设置且 approx_kl > 1.5·target_kl：早停 break（可选）")
print("            · optimizer.zero_grad() → loss.backward() → clip_grad_norm_(max_grad_norm) → optimizer.step()")
print("        _n_updates += 1")
print("   ⑤ explained_variance(values.flatten(), returns.flatten())")
print("   ⑥ logger.record 一大堆指标（entropy_loss / pg_loss / value_loss / approx_kl / clip_fraction / …）")

section("H. 交叉证据：用训练前后 model.predict 对比得分，验证 train() 确实改变了 policy 参数")
eval_env = gym.make(ENV_ID)


def eval_once(title: str, m: PPO, n_episodes: int = 5, seed: int = 0) -> float:
    rewards = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        done = False
        r = 0.0
        while not done:
            a, _ = m.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = eval_env.step(int(a) if ENV_ID == "CartPole-v1" else a)
            done = terminated or truncated
            r += rew
        rewards.append(r)
    mean_r = float(np.mean(rewards))
    print(f"   {title}: 平均得分 = {mean_r:.1f}  （{n_episodes} 局，deterministic=True）")
    return mean_r


model_before = PPO("MlpPolicy", eval_env, verbose=0, seed=SEED, n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS)
eval_once("训练前（新构造的 PPO）", model_before, n_episodes=3, seed=42)
eval_after = eval_once("训练后（learn 完）", model, n_episodes=3, seed=42)
print(f"   ✓ 训练后得分提升 = {eval_after}（若 policy 没变则不会涨；随机 CartPole 一般 ~20 左右）")

section("I. 训练主循环源码位置索引（便于 issue-04.md 引用）")
ref = [
    ("ppo/ppo.py:18-171",         "PPO.__init__（字段注入，batch_size/n_epochs/clip_range 等断言）"),
    ("ppo/ppo.py:173-183",        "PPO._setup_model（FloatSchedule 包装 clip_range）"),
    ("common/on_policy_algorithm.py:61-141", "OnPolicyAlgorithm.__init__ + _setup_model（创建 RolloutBuffer + policy）"),
    ("common/base_class.py:382-435",        "BaseAlgorithm._setup_learn（reset env、logger、callback）"),
    ("common/on_policy_algorithm.py:162-268","OnPolicyAlgorithm.collect_rollouts（采样、写入 buffer、算 returns/adv）"),
    ("common/buffers.py:343",                "RolloutBuffer 类定义"),
    ("common/buffers.py:440",                "RolloutBuffer.add（写入单步 transition）"),
    ("common/buffers.py:403",                "RolloutBuffer.compute_returns_and_advantage（GAE）"),
    ("common/buffers.py:481",                "RolloutBuffer.get（按 batch_size yield minibatches）"),
    ("ppo/ppo.py:184-301",                   "PPO.train（n_epochs × minibatches × surrogate loss + 优化）"),
    ("common/on_policy_algorithm.py:300-342","OnPolicyAlgorithm.learn（主 while 循环：rollout → train → dump_logs）"),
    ("ppo/ppo.py:302-318",                   "PPO.learn → super().learn 透传"),
]
for loc, desc in ref:
    print(f"   {loc:<44s} {desc}")

eval_env.close()
env.close()
try:
    model.get_env().close()  # type: ignore[union-attr]
except Exception:
    pass

bar("End · T04 learn trace 完成")
