"""Ensure the trace rejects a train() that increments counters but changes no parameters."""

import runpy
import sys
from unittest.mock import patch

from stable_baselines3 import PPO


def no_parameter_update(self):
    self._n_updates += self.n_epochs


with patch.object(PPO, "train", no_parameter_update):
    try:
        runpy.run_path(sys.argv[1], run_name="__main__")
    except AssertionError as error:
        assert str(error) == "learn 后没有任何 policy 参数变化", str(error)
        print(
            "PASS: disabled parameter updates were rejected, even with matching counters"
        )
    else:
        raise AssertionError("The trace accepted training without parameter updates")
