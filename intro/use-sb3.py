import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# ============ 配置区：想改东西先看这里 ============
ENV_ID = "CartPole-v1"        # 换成别的环境只需要改这里
MODEL_PATH = "ppo_cartpole"   # 保存时会自动加 .zip 后缀
TOTAL_TIMESTEPS = 50_000      # 训练总步数
SEED = 0                      # 固定随机种子，方便复现
# ================================================


def make_eval_env():
    """评估用环境：套一层 Monitor，evaluate_policy 才不会报警告。"""
    return Monitor(gym.make(ENV_ID))


def evaluate(model, title):
    """跑 10 局，返回平均得分。不弹窗，速度快。"""
    mean_reward, std_reward = evaluate_policy(
        model, make_eval_env(), n_eval_episodes=10, deterministic=True
    )
    print(f"[{title}] 平均得分 = {mean_reward:.1f} ± {std_reward:.1f}")
    return mean_reward


def train():
    """训练：环境不开渲染，速度最快。"""
    train_env = gym.make(ENV_ID)
    model = PPO("MlpPolicy", train_env, verbose=1, seed=SEED)

    evaluate(model, "训练前")            # 随机瞎蒙的基线
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    evaluate(model, "训练后")            # 看有没有变好

    model.save(MODEL_PATH)               # 保存到 ppo_cartpole.zip
    print(f"模型已保存到 {MODEL_PATH}.zip")
    train_env.close()
    return model


def watch(model, episodes=5):
    """展示：单独开一个带画面的环境，不影响训练速度。"""
    env = gym.make(ENV_ID, render_mode="human")
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"第 {ep} 局得分: {total_reward}")
    env.close()


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH + ".zip"):
        print("发现已保存的模型，直接加载（跳过训练）")
        model = PPO.load(MODEL_PATH)
        evaluate(model, "已加载模型")
    else:
        model = train()

    watch(model)
