"""Targeted contract checks for the fixed SB3 baseline."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from issue_04_baseline import BASELINE, require_baseline, source_metadata
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


from stable_baselines3.common.callbacks import BaseCallback


@pytest.fixture
def model():
    m = PPO(
        "MlpPolicy",
        "CartPole-v1",
        n_steps=4,
        batch_size=2,
        n_epochs=1,
        seed=0,
        device="cpu",
    )
    yield m
    m.get_env().close()


def collect(model, callback=None):
    _, cb = model._setup_learn(4, callback=callback)
    return model.collect_rollouts(
        model.get_env(), cb, model.rollout_buffer, model.n_steps
    )


def test_collection_preserves_parameters_and_computes_returns_before_callback(model):
    events = []
    modes = []

    class Callback(BaseCallback):
        def _on_step(self):
            return True

        def _on_rollout_end(self):
            events.append("rollout_end")

    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    original = model.rollout_buffer.compute_returns_and_advantage

    def compute(*args, **kwargs):
        result = original(*args, **kwargs)
        events.append("returns_computed")
        return result

    hook = model.policy.register_forward_pre_hook(
        lambda module, args: modes.append((module.training, th.is_grad_enabled()))
    )
    try:
        with patch.object(
            model.rollout_buffer, "compute_returns_and_advantage", side_effect=compute
        ):
            assert collect(model, Callback())
    finally:
        hook.remove()
    assert modes == [(False, False)] * 4
    assert all(th.equal(before[k], v) for k, v in model.policy.state_dict().items())
    assert events == ["returns_computed", "rollout_end"]


def test_buffer_flattening_and_permutation_each_get(model):
    assert collect(model)
    rb = model.rollout_buffer
    assert rb.observations.shape == (4, 1, 4) and not rb.generator_ready
    with patch.object(np.random, "permutation", wraps=np.random.permutation) as permute:
        batches1 = list(rb.get(2))
        batches2 = list(rb.get(2))
    assert permute.call_count == 2
    assert rb.generator_ready and rb.observations.shape == (4, 4)
    assert [b.observations.shape for b in batches1 + batches2] == [(2, 4)] * 4
    previous = rb.observations
    rb.reset()
    assert rb.observations is not previous and rb.observations.shape == (4, 1, 4)
    assert not rb.generator_ready and not rb.full and rb.pos == 0


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"n_steps": 4, "batch_size": 1}, "batch_size"),
        ({"n_steps": 1, "batch_size": 2}, "n_steps"),
    ],
)
def test_normalization_init_rejects_single_sample(kwargs, message):
    env = gym.make("CartPole-v1")
    try:
        with pytest.raises(AssertionError, match=message):
            PPO("MlpPolicy", env, normalize_advantage=True, **kwargs)
    finally:
        env.close()


def test_normalization_disabled_allows_single_sample():
    env = gym.make("CartPole-v1")
    try:
        PPO("MlpPolicy", env, n_steps=1, batch_size=1, normalize_advantage=False)
    finally:
        env.close()


def test_remainder_minibatch_warning():
    env = gym.make("CartPole-v1")
    try:
        with pytest.warns(UserWarning, match="truncated mini-batch of size 1"):
            PPO("MlpPolicy", env, n_steps=3, batch_size=2)
    finally:
        env.close()


@pytest.mark.parametrize("normalize", [False, True])
def test_advantage_normalization_flag_and_singleton_tail(normalize):
    env = gym.make("CartPole-v1")
    calls = []
    original_std = th.Tensor.std

    def observed_std(tensor, *args, **kwargs):
        calls.append(tensor.numel())
        return original_std(tensor, *args, **kwargs)

    try:
        with pytest.warns(UserWarning, match="truncated mini-batch"):
            m = PPO(
                "MlpPolicy",
                env,
                n_steps=3,
                batch_size=2,
                n_epochs=1,
                normalize_advantage=normalize,
                seed=0,
            )
        with patch.object(th.Tensor, "std", observed_std):
            m.learn(3)
        assert calls == ([2] if normalize else [])
    finally:
        env.close()


@pytest.mark.parametrize("target,epochs,steps", [(0.016, 3, 6), (0.014, 1, 0)])
def test_kl_multiplier_skips_optimizer_and_counts_epoch(model, target, epochs, steps):
    assert collect(model)
    model.n_epochs = 3
    model.target_kl = target
    model.rollout_buffer.log_probs.fill(0)
    original = model.policy.evaluate_actions

    def controlled(*args, **kwargs):
        values, log_prob, entropy = original(*args, **kwargs)
        return values, log_prob * 0 + 0.2, entropy

    with (
        patch.object(model.policy, "evaluate_actions", side_effect=controlled),
        patch.object(
            model.policy.optimizer, "step", wraps=model.policy.optimizer.step
        ) as optimizer,
    ):
        model.train()
    assert optimizer.call_count == steps
    assert model._n_updates == epochs
    assert float(model.logger.name_to_value["train/approx_kl"]) == pytest.approx(
        np.exp(0.2) - 1 - 0.2, abs=1e-6
    )


def test_callback_false_stops_without_training(model):
    class Stop(BaseCallback):
        ended = False

        def _on_step(self):
            return False

        def _on_training_end(self):
            self.ended = True

    callback = Stop()
    with patch.object(model, "train", wraps=model.train) as train:
        model.learn(8, callback=callback)
    train.assert_not_called()
    assert model.num_timesteps == 1 and model.rollout_buffer.pos == 0
    assert model._n_updates == 0 and callback.ended


def test_timestep_lower_bound_and_resume_without_reset(model):
    env = model.get_env()
    with patch.object(env, "reset", wraps=env.reset) as reset:
        model.learn(3)
        assert model.num_timesteps == 4
        model.learn(1, reset_num_timesteps=False)
        assert model.num_timesteps == 8
    reset.assert_called_once()


@pytest.mark.parametrize(
    "terminated,expected_reward,expected_values_calls",
    [(False, 2.8, 3), (True, 1.0, 1)],
)
def test_time_limit_bootstrap_only_for_truncation(
    terminated, expected_reward, expected_values_calls
):
    class Env(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 1.0, terminated, False, {}

    env = gym.wrappers.TimeLimit(Env(), max_episode_steps=1)
    try:
        m = PPO("MlpPolicy", env, n_steps=2, batch_size=2, gamma=0.9, seed=0)
        with patch.object(
            m.policy, "predict_values", return_value=th.tensor([[2.0]])
        ) as values:
            assert collect(m)
        np.testing.assert_allclose(m.rollout_buffer.rewards, expected_reward)
        assert values.call_count == expected_values_calls
    finally:
        env.close()


def test_logger_statistics_and_tensorboard_exclusions(model):
    model._setup_learn(4)
    model.ep_info_buffer.extend([{"r": 2.0, "l": 4}, {"r": 4.0, "l": 6}])
    with patch.object(model.logger, "dump"):
        model.dump_logs(1)
    assert model.logger.name_to_value["rollout/ep_rew_mean"] == 3
    assert model.logger.name_to_value["rollout/ep_len_mean"] == 5
    for key in ["time/iterations", "time/time_elapsed", "time/total_timesteps"]:
        assert "tensorboard" in model.logger.name_to_excluded[key]


def test_log_interval_none_suppresses_dump_but_still_trains(model):
    with (
        patch.object(model, "dump_logs", wraps=model.dump_logs) as dump,
        patch.object(model, "train", wraps=model.train) as train,
    ):
        model.learn(4, log_interval=None)
    dump.assert_not_called()
    train.assert_called_once()
    assert model._n_updates == 1
