"""Trace two short PPO rollout/training cycles without judging reward quality.

Run: uv run --no-project python scripts/issue_06_short_training.py
Purpose: verify collect_rollouts fills the buffer before each train call.
Expected: two collect_start/collect_end/train_start/train_end groups; 32 total
timesteps with n_steps=16, one environment, one epoch and no model artifact.
"""

from __future__ import annotations

import json

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

SEED = 42
ROLLOUT_STEPS = 16
TOTAL_TIMESTEPS = 32


class TracedPPO(PPO):
    """Log the same method boundaries that OnPolicyAlgorithm.learn invokes."""

    events: list[str]

    def _emit(self, event: str) -> None:
        self.events.append(event)
        print(
            json.dumps(
                {
                    "event": event,
                    "num_timesteps": self.num_timesteps,
                    "buffer_position": self.rollout_buffer.pos,
                    "buffer_full": self.rollout_buffer.full,
                    "updates": self._n_updates,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        self._emit("collect_start")
        completed = super().collect_rollouts(
            env, callback, rollout_buffer, n_rollout_steps
        )
        self._emit("collect_end")
        return completed

    def train(self) -> None:
        self._emit("train_start")
        super().train()
        self._emit("train_end")


def main() -> None:
    torch.set_num_threads(1)
    env = gym.make("CartPole-v1")
    model = TracedPPO(
        "MlpPolicy",
        env,
        seed=SEED,
        device="cpu",
        n_steps=ROLLOUT_STEPS,
        batch_size=ROLLOUT_STEPS,
        n_epochs=1,
    )
    model.events = []
    print(
        json.dumps(
            {
                "seed": SEED,
                "n_steps": ROLLOUT_STEPS,
                "total_timesteps": TOTAL_TIMESTEPS,
            },
            sort_keys=True,
        )
    )
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        assert (
            model.events
            == ["collect_start", "collect_end", "train_start", "train_end"] * 2
        )
        assert model.num_timesteps == TOTAL_TIMESTEPS
        assert model._n_updates == 2
        print(
            json.dumps(
                {"event": "finished", "num_timesteps": model.num_timesteps},
                sort_keys=True,
            )
        )
    finally:
        wrapped_env = model.get_env()
        if wrapped_env is not None:
            wrapped_env.close()


if __name__ == "__main__":
    main()
