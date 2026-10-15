"""Measure CartPole episode lengths across a small PPO learning curve.

Run: uv run --no-project python scripts/issue_06_cartpole_effect.py
Purpose: report actual task performance, measured as steps survived per episode.
Expected: ten episodes on the same fixed seeds at every checkpoint, with mean,
median, minimum and maximum lengths. Improvement is not assumed.
"""

from __future__ import annotations

import json
import statistics
import time

import gymnasium as gym
import torch
from stable_baselines3 import PPO

TRAINING_SEED = 42
EVALUATION_SEEDS = list(range(100, 110))
CHECKPOINTS = [0, 4096, 8192, 16384, 32768]


def evaluate(model: PPO, env: gym.Env, seeds: list[int]) -> dict[str, object]:
    episodes: list[dict[str, int | float | bool]] = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        steps = 0
        total_reward = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(int(action))
            steps += 1
            total_reward += float(reward)
        episodes.append(
            {
                "seed": seed,
                "steps": steps,
                "reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
            }
        )

    lengths = [int(episode["steps"]) for episode in episodes]
    return {
        "episodes": episodes,
        "mean_steps": statistics.mean(lengths),
        "median_steps": statistics.median(lengths),
        "min_steps": min(lengths),
        "max_steps": max(lengths),
    }


def main() -> None:
    torch.set_num_threads(1)
    training_env = gym.make("CartPole-v1")
    evaluation_env = gym.make("CartPole-v1")
    model = PPO(
        "MlpPolicy",
        training_env,
        seed=TRAINING_SEED,
        device="cpu",
        n_steps=128,
        batch_size=128,
        n_epochs=4,
        verbose=0,
    )
    try:
        measurements: list[dict[str, object]] = []
        previous_timesteps = 0
        for target_timesteps in CHECKPOINTS:
            started = time.perf_counter()
            if target_timesteps > previous_timesteps:
                # With reset_num_timesteps=False, SB3 adds the current count to
                # total_timesteps; pass the increment to land on the target.
                model.learn(
                    total_timesteps=target_timesteps - previous_timesteps,
                    reset_num_timesteps=False,
                )
            training_seconds = time.perf_counter() - started
            assert model.num_timesteps == target_timesteps
            measurements.append(
                {
                    "training_timesteps": model.num_timesteps,
                    "training_wall_seconds_since_previous": round(training_seconds, 3),
                    "evaluation": evaluate(model, evaluation_env, EVALUATION_SEEDS),
                }
            )
            previous_timesteps = target_timesteps

        result: dict[str, object] = {
            "training_seed": TRAINING_SEED,
            "evaluation_seeds": EVALUATION_SEEDS,
            "algorithm": "PPO",
            "environment": "CartPole-v1",
            "deterministic_evaluation": True,
            "checkpoints": measurements,
            "n_steps": 128,
            "batch_size": 128,
            "n_epochs": 4,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        evaluation_env.close()
        wrapped_env = model.get_env()
        if wrapped_env is not None:
            wrapped_env.close()


if __name__ == "__main__":
    main()
