# Issue #05 运行记录：2026-10-12-env-checker-review-fixes

- 运行时间：2026-10-12T22:51:13
- SB3 实际 commit：`7cfb4dd6055e74b5caa4ed4d6777492209946e26`（已核对固定基线，源码无未提交修改）
- 实际检查器路径：`/private/tmp/pr19-sb3-baseline/stable_baselines3/common/env_checker.py`
- Python：3.13.5｜PyTorch：2.14.0｜Gymnasium：1.3.0｜numpy：2.5.3｜SB3(本地源码版本)：2.9.2a0
- 运行命令：`/Users/jaschen/sb3-llm-code-understanding/intro/.venv/bin/python scripts/issue_05_env_checker_failures.py --output-dir artifacts/runs/issue-05/2026-10-12-env-checker-review-fixes`
- 工作目录 / PYTHONPATH 源码根目录：`/private/tmp/pr19-fix` / `/private/tmp/pr19-sb3-baseline`
- 平台：macOS-27.0.1-arm64-arm-64bit-Mach-O

| 用例 | 分类 | warn | 预期 | 实际结果 | 源码位置 |
|---|---|---|---|---|---|
| F01 | A 环境类型 | True | AssertionError | AssertionError | `env_checker.py:L482-484` |
| F02 | A 空间声明 | True | AssertionError | AssertionError | `env_checker.py:L424-427` |
| F03 | B API 签名 | True | TypeError | TypeError | `env_checker.py:L493-496` |
| F04 | C 返回值结构 | True | AssertionError | AssertionError | `env_checker.py:L367-370` |
| F05 | C 返回值结构 | True | AssertionError | AssertionError | `env_checker.py:L337-338` |
| F06 | C 返回值结构 | True | AssertionError | AssertionError | `env_checker.py:L401` |
| F07 | C 返回值结构 | True | AssertionError | AssertionError | `env_checker.py:L402` |
| F08 | D 观测内容 | True | AssertionError | AssertionError | `env_checker.py:L285-303` |
| W09 | E 警告边界 | True | UserWarning + 跳过返回值检查 | 通过（1 条警告） | `env_checker.py:L156-162, L531-532` |
| W10 | E 警告边界 | True | UserWarning，不失败 | 通过（1 条警告） | `env_checker.py:L510-518` |
| F11 | E 警告边界 | True | AssertionError | AssertionError | `env_checker.py:L520-523` |
| F12 | E 警告边界 | False | AssertionError，无警告 | AssertionError | `env_checker.py:L500-502, L535` |

## 完整报错与警告

### F01（A 环境类型，预期 AssertionError）

```
Your environment must inherit from the gymnasium.Env class cf. https://gymnasium.farama.org/api/env/
```

### F02（A 空间声明，预期 AssertionError）

```
The action space must inherit from gymnasium.spaces (cf. https://gymnasium.farama.org/api/spaces/)
```

### F03（B API 签名，预期 TypeError）

```
The reset() method must accept a `seed` parameter
```

### F04（C 返回值结构，预期 AssertionError）

```
The `step()` method must return five values: obs, reward, terminated, truncated, info. Actual: 4 values returned.
```

### F05（C 返回值结构，预期 AssertionError）

```
`reset()` must return a tuple (obs, info)
```

### F06（C 返回值结构，预期 AssertionError）

```
The reward returned by `step()` must be a float
```

### F07（C 返回值结构，预期 AssertionError）

```
The `terminated` signal must be a boolean
```

### F08（D 观测内容，预期 AssertionError）

```
The observation returned by the `reset()` method does not match the bounds of the given observation space Box(0.0, 1.0, (3,), float32). 
1 invalid indices: 
Expected: 0.0 <= obs[1] <= 1.0, actual value: 1.5 

```

### W09（E 警告边界，预期 UserWarning + 跳过返回值检查）

> 警告：Sequence observation space is not supported by Stable-Baselines3. You can pad your observation to have a fixed size instead.
Note: The checks for returned values are skipped.

### W10（E 警告边界，预期 UserWarning，不失败）

> 警告：We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) cf. https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

### F11（E 警告边界，预期 AssertionError）

```
Continuous action space must have a finite lower and upper bound
```
> 警告：We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) cf. https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

### F12（E 警告边界，预期 AssertionError，无警告）

```
The observation returned by `reset()` method must be a numpy array
```
