import json
import statistics
import subprocess
import sys
from pathlib import Path

root = Path('/private/tmp/sb3-pr-review/pr12')
out = Path('/private/tmp/sb3-pr-review/pr12-validation-output')
out.mkdir(exist_ok=True)

def run(name):
    result = subprocess.run([sys.executable, str(root / 'scripts' / f'issue_06_{name}.py')], capture_output=True, text=True, check=True)
    (out / f'{name}.json').write_text(result.stdout)
    if result.stderr:
        (out / f'{name}.stderr').write_text(result.stderr)
    return result.stdout

info = json.loads(run('system_info'))
assert info['python'] == '3.12.14', info
assert info['stable_baselines3'] == '2.9.2a0', info
print('Runtime:', json.dumps(info), flush=True)
prediction = json.loads(run('predict'))
saved = json.loads((root / 'artifacts/runs/issue-06/2026-09-30-predict/output.json').read_text())
assert prediction == saved, (prediction, saved)
print('Prediction matches saved JSON exactly', flush=True)
events = [json.loads(s) for s in run('short_training').splitlines()]
saved_events = [json.loads(s) for s in (root / 'artifacts/runs/issue-06/2026-09-30-short-training/events.jsonl').read_text().splitlines()]
assert events == saved_events, (events, saved_events)
print('Training events match saved JSONL exactly', flush=True)
effect = json.loads(run('cartpole_effect'))
saved_effect = json.loads((root / 'artifacts/runs/issue-06/2026-09-30-cartpole-effect/output.json').read_text())
assert effect['evaluation_seeds'] == list(range(100,110))
assert [p['training_timesteps'] for p in effect['checkpoints']] == [0,4096,8192,16384,32768]
for checkpoint in effect['checkpoints']:
    ev = checkpoint['evaluation']
    episodes = ev['episodes']
    assert [e['seed'] for e in episodes] == list(range(100,110))
    lengths = [e['steps'] for e in episodes]
    assert ev['mean_steps'] == statistics.mean(lengths)
    assert ev['median_steps'] == statistics.median(lengths)
    assert ev['min_steps'] == min(lengths) and ev['max_steps'] == max(lengths)
    assert all(1 <= e['steps'] <= 500 and e['reward'] == e['steps'] and (e['terminated'] or e['truncated']) for e in episodes)
for data in (effect, saved_effect):
    for checkpoint in data['checkpoints']:
        checkpoint.pop('training_wall_seconds_since_previous')
print('Learning-curve means:', [p['evaluation']['mean_steps'] for p in effect['checkpoints']], flush=True)
print('Exact saved learning curve (except timing):', effect == saved_effect, flush=True)
assert effect == saved_effect, 'Learning curve still differs with pinned Python and dependencies'
print('ALL CHECKS PASSED', flush=True)
