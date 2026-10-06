"""Compare one Gymnasium prediction step with the model's VecEnv interface.

Run: uv run --no-project python scripts/issue_06_predict.py
Purpose: verify reset/predict/step values and the two environment return shapes.
Expected: raw Gymnasium reset returns (observation, info), step returns five
values; SB3's wrapped VecEnv reset returns observations and step four values.
Passing the entire Gymnasium reset tuple to predict raises an API-mix error.
No training is performed; actions come from an initialized policy.
"""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

SEED = 42


def main() -> None:
    raw_env = gym.make("CartPole-v1")
    model_env = gym.make("CartPole-v1")
    model: PPO | None = None
    try:
        check_env(raw_env, warn=False)
        observation, info = raw_env.reset(seed=SEED)
        model = PPO(
            "MlpPolicy",
            model_env,
            seed=SEED,
            device="cpu",
            n_steps=16,
            batch_size=16,
            n_epochs=1,
        )
        action, state = model.predict(observation, deterministic=True)
        next_observation, reward, terminated, truncated, step_info = raw_env.step(
            int(action)
        )

        # This guards against the common mistake of passing Gymnasium's full
        # reset result where predict expects only the observation.
        try:
            model.predict((observation, info), deterministic=True)
        except ValueError as error:
            api_mix_error = str(error)
        else:
            raise AssertionError(
                "predict unexpectedly accepted the Gymnasium reset tuple"
            )

        wrapped_env = model.get_env()
        assert wrapped_env is not None
        wrapped_env.seed(SEED)
        vector_observation = wrapped_env.reset()
        vector_action, _ = model.predict(vector_observation, deterministic=True)
        vector_next_observation, vector_rewards, vector_dones, vector_infos = (
            wrapped_env.step(vector_action)
        )

        result: dict[str, object] = {
            "seed": SEED,
            "check_env": "passed",
            "gym_reset": {
                "observation_shape": list(observation.shape),
                "info_type": type(info).__name__,
            },
            "predict": {
                "action": int(action),
                "action_valid": bool(raw_env.action_space.contains(int(action))),
                "state_is_none": state is None,
                "gym_reset_tuple_error": api_mix_error,
            },
            "gym_step": {
                "observation_shape": list(next_observation.shape),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info_type": type(step_info).__name__,
            },
            "vec_env": {
                "class": type(wrapped_env).__name__,
                "reset_observation_shape": list(vector_observation.shape),
                "action_shape": list(np.asarray(vector_action).shape),
                "step_observation_shape": list(vector_next_observation.shape),
                "rewards_shape": list(vector_rewards.shape),
                "dones_shape": list(vector_dones.shape),
                "infos_count": len(vector_infos),
            },
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        raw_env.close()
        # The model owns a wrapper around model_env, which closes its child.
        if model is not None:
            wrapped_env = model.get_env()
            if wrapped_env is not None:
                wrapped_env.close()
        else:
            model_env.close()


if __name__ == "__main__":
    main()
