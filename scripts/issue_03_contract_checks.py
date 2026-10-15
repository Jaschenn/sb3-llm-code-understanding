"""Targeted contract checks for the fixed SB3 baseline."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from issue_03_baseline import BASELINE, require_baseline, source_metadata
from stable_baselines3 import PPO

METADATA = require_baseline()
th.set_num_threads(1)


@pytest.fixture
def checkout(tmp_path):
    target = tmp_path / "sb3"
    subprocess.run(
        ["git", "clone", "--shared", "--quiet", METADATA["source_root"], str(target)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--quiet", "--detach", BASELINE],
        check=True,
    )
    return target


def test_source_guard_accepts_baseline():
    assert (
        source_metadata(
            Path(METADATA["source_root"]) / "stable_baselines3/__init__.py"
        )["source_commit"]
        == BASELINE
    )


def test_source_guard_rejects_non_checkout(tmp_path):
    with pytest.raises(RuntimeError, match="SB3_BASELINE_REJECTED"):
        source_metadata(tmp_path / "site-packages/stable_baselines3/__init__.py")


def test_source_guard_rejects_dirty_checkout(checkout):
    version = checkout / "stable_baselines3/version.txt"
    version.write_text(version.read_text() + "\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        source_metadata(checkout / "stable_baselines3/__init__.py")


def test_source_guard_rejects_other_commit(checkout):
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "--quiet", "--detach", BASELINE + "^"],
        check=True,
    )
    with pytest.raises(RuntimeError, match="expected " + BASELINE):
        source_metadata(checkout / "stable_baselines3/__init__.py")


from stable_baselines3.common.policies import MultiInputActorCriticPolicy


@pytest.mark.parametrize("batched", [False, True])
def test_dict_and_batch_observations(batched):
    space = gym.spaces.Dict({"x": gym.spaces.Box(-1, 1, (2,), dtype=np.float32)})
    policy = MultiInputActorCriticPolicy(space, gym.spaces.Discrete(2), lambda _: 3e-4)
    obs = {"x": np.zeros((2, 2) if batched else (2,), dtype=np.float32)}
    tensor, vectorized = policy.obs_to_tensor(obs)
    action, state = policy.predict(obs, deterministic=True)
    assert vectorized == batched
    assert tensor["x"].shape == ((2, 2) if batched else (1, 2))
    assert action.shape == ((2,) if batched else ())
    assert state is None


def test_box_clips_unbounded_actions():
    env = gym.make("Pendulum-v1")
    try:
        model = PPO("MlpPolicy", env, n_steps=4, batch_size=4, seed=0, device="cpu")
        with th.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.fill_(10)
        obs, _ = env.reset(seed=0)
        tensor, _ = model.policy.obs_to_tensor(obs)
        with th.no_grad():
            raw = model.policy._predict(tensor, deterministic=True).numpy().squeeze(0)
        assert np.all(raw > env.action_space.high)
        with patch.object(
            model.policy, "unscale_action", wraps=model.policy.unscale_action
        ) as unscale:
            action, _ = model.predict(obs, deterministic=True)
        np.testing.assert_array_equal(action, env.action_space.high)
        unscale.assert_not_called()
    finally:
        env.close()


def test_box_unscales_controlled_squashed_action():
    env = gym.make("Pendulum-v1")
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=4,
            batch_size=4,
            seed=0,
            device="cpu",
            use_sde=True,
            policy_kwargs={"squash_output": True},
        )
        obs, _ = env.reset(seed=0)
        # Isolate BasePolicy.predict's postprocessing; this is not a gSDE sampling test.
        with (
            patch.object(model.policy, "_predict", return_value=th.tensor([[0.5]])),
            patch.object(
                model.policy, "unscale_action", wraps=model.policy.unscale_action
            ) as unscale,
        ):
            action, _ = model.predict(obs, deterministic=True)
        np.testing.assert_allclose(action, [1.0])
        unscale.assert_called_once()
    finally:
        env.close()


def test_raw_gym_and_vecenv_return_contracts():
    raw = gym.make("CartPole-v1")
    model = PPO(
        "MlpPolicy", "CartPole-v1", n_steps=4, batch_size=4, seed=0, device="cpu"
    )
    vec = model.get_env()
    try:
        obs, info = raw.reset(seed=0)
        assert obs.shape == (4,) and isinstance(info, dict)
        action, _ = model.predict(obs, deterministic=True)
        assert len(raw.step(int(action))) == 5
        vec.seed(0)
        batch = vec.reset()
        assert batch.shape == (1, 4)
        actions, _ = model.predict(batch, deterministic=True)
        assert actions.shape == (1,)
        result = vec.step(actions)
        assert len(result) == 4
        assert result[1].shape == result[2].shape == (1,)
        assert len(result[3]) == 1 and isinstance(vec.reset_infos[0], dict)
    finally:
        raw.close()
        vec.close()
