"""Run the report's prediction and TensorBoard examples without leaving artifacts.

Run against the fixed upstream source (see docs/project-baseline.md):
PYTHONPATH=/path/to/stable-baselines3 uv run --no-project --active \
    --with tensorboard python scripts/issue_01_validate_examples.py
"""

import contextlib
import io
from pathlib import Path
import re
import tempfile

import stable_baselines3
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    report = Path(__file__).resolve().parents[1] / "docs/report/02-project-map.md"
    text = report.read_text()
    snippets = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    prediction = next(code for code in snippets if 'print("原始观察:"' in code)
    logging = next(code for code in snippets if 'tensorboard_log="./logs/"' in code)
    print("SB3 source:", stable_baselines3.__file__)
    print("SB3 version:", stable_baselines3.__version__)
    with tempfile.TemporaryDirectory(prefix="sb3-issue-01-") as temporary:
        with contextlib.chdir(temporary):
            prediction_scope = {}
            exec(compile(prediction, str(report), "exec"), prediction_scope)
            prediction_scope["env"].close()
            # Execute the documented code unchanged; omit verbose training output.
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(logging, str(report), "exec"), {})
            logs = Path("logs")
            assert (logs / "best_model.zip").is_file()
            assert (logs / "evaluations.npz").is_file()
            events = list(logs.rglob("events.out.tfevents.*"))
            assert events, "The example did not create TensorBoard event files"
            accumulator = EventAccumulator(str(events[0].parent)).Reload()
            tags = accumulator.Tags()["scalars"]
            assert "eval/mean_reward" in tags, tags
            assert "rollout/ep_rew_mean" in tags, tags
            print("TensorBoard events:", len(events))
            print("Verified scalars: eval/mean_reward, rollout/ep_rew_mean")
            print("Verified outputs: best_model.zip, evaluations.npz")


if __name__ == "__main__":
    main()
