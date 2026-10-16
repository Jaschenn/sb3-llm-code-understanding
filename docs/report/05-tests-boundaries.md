# 5. 测试与边界

本章回答两个问题：SB3 对自定义环境的接口契约是什么，以及这些契约在真实运行中能被验证到什么程度。契约由 [T05 证据](../evidence/issues/issue-05.md)从 `check_env()` 与上游测试反推，运行证据由 [T06 证据](../evidence/issues/issue-06.md)提供。主实验为 CartPole-v1 / MlpPolicy，基线为 `7cfb4dd6055e74b5caa4ed4d6777492209946e26`，源码基线见[说明](../project-baseline.md)。Dict、Graph、Sequence 等空间只给出受控检查的结论，不自动推广。

证据分三类：固定 commit 的源码（用于确认检查存在及其条件）、上游测试 `tests/test_env_checker.py`（用于确认哪些失败路径有回归保护）、本组自建运行（用于给出实际报错与观察结果）。**`check_env()` 通过不等于环境正确**，5.1.3 给出直接反例。

## 5.1 环境契约：从检查器反推

### 5.1.1 检查顺序

[`check_env`](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L467-L547) 的检查顺序固定，前置检查失败就不会执行后面的检查：

| 顺序 | 检查内容 | 源码位置 | 典型失败 |
|---|---|---|---|
| 1 | 环境必须继承 `gymnasium.Env` | `env_checker.py:L482-484` | `AssertionError`（F01） |
| 2 | 两个空间必须是 `gymnasium.spaces.Space` 实例 | `env_checker.py:L421-427` | `AssertionError`（F02） |
| 3 | `reset()` 必须接受 `seed` 参数 | `env_checker.py:L493-496` | `TypeError`（F03） |
| 4 | 观测空间支持性警告，仅 `warn=True` 时执行 | `env_checker.py:L500-502`、`L147-169` | `UserWarning`，并置跳过标志（W09） |
| 5 | 动作空间建议与无限边界断言，位于 `if warn:` 块内 | `env_checker.py:L510-523` | 建议仅警告（W10）；无限边界 `AssertionError`（F11） |
| 6 | 返回值结构与内容检查，跳过标志为真时整段略过 | `env_checker.py:L531-535`，内部 `L337-338`、`L367-370`、`L401-402`、`L285-303` | `AssertionError`（F04–F08、F12） |
| 7 | NaN 检查 | `env_checker.py:L545` | 本组未单独构造 |
| 8 | `render` 检查，默认 `skip_render_check=True` 跳过 | `env_checker.py:L467`、`L538-539` | 默认不执行 |

顺序在实验中被间接验证：F03 的 `seed` 失败发生在返回值检查之前，F04–F07 都在空间检查通过之后才报错，与上表一致。

### 5.1.2 自建失败用例

脚本 [`scripts/issue_05_env_checker_failures.py`](../../scripts/issue_05_env_checker_failures.py) 构造 12 个不合法环境，实际结果见[复核运行记录](../../artifacts/runs/issue-05/2026-10-12-env-checker-review-fixes/summary.md)（原始 11 用例见[首次运行](../../artifacts/runs/issue-05/2026-10-12-env-checker-failures/summary.md)）：

| 用例 | 构造的不合法点 | 类别 | `warn` | 实际报错 | 源码 | 上游测试是否覆盖 |
|---|---|---|---|---|---|---|
| F01 | 鸭子类型合法但不继承 `gym.Env` | 环境类型 | True | `must inherit from the gymnasium.Env class` | L482-484 | 否 |
| F02 | `action_space = [0, 1]` | 空间声明 | True | `must inherit from gymnasium.spaces` | L424-427 | 否 |
| F03 | `reset(self)` 无 `seed` 参数 | API 签名 | True | `reset() method must accept a seed` | L493-496 | 否 |
| F04 | `step()` 返回旧 API 4 元组 | 返回值结构 | True | `must return five values: obs, reward, terminated, truncated, info` | L367-370 | 否 |
| F05 | `reset()` 只返回观测 | 返回值结构 | True | `reset() must return a tuple (obs, info)` | L337-338 | 否 |
| F06 | reward 返回字符串 | 返回值类型 | True | `reward ... must be a float` | L401 | 否 |
| F07 | `terminated` 返回 int | 返回值类型 | True | `terminated signal must be a boolean` | L402 | 否 |
| F08 | 声明 `Box(0,1)` 却返回 `[0.5, 1.5, 0.5]` | 观测内容 | True | `Expected: 0.0 <= obs[1] <= 1.0, actual value: 1.5` | L285-303 | 是（`test_check_env_detailed_error`） |
| W09 | `Sequence` 空间且返回字符串观测 | 警告边界 | True | **通过**，仅警告返回值检查被跳过 | L156-162、L531-532 | 部分（只断言警告） |
| W10 | `Box(-2, 1)` 非对称动作空间 | 警告边界 | True | **通过**，仅建议警告 | L510-518 | 是 |
| F11 | 动作空间 `low=-inf` | 警告边界 | True | `must have a finite lower and upper bound` | L520-523 | 否 |
| F12 | 同 W09 环境但 `warn=False` | 警告边界 | False | `observation ... must be a numpy array` | L500-502、L535 | 否 |

F08 的报错带逐索引明细（`invalid_indices`），是唯一被上游测试以参数化用例覆盖的失败路径；其余七条失败路径只有源码断言、**没有回归测试**。

### 5.1.3 检查覆盖不到的地方

- **静默通过（W09）**：`Sequence`/`Graph`/`OneOf` 观测空间在 `warn=True` 下触发警告后，`should_skip` 置真，全部返回值检查被跳过（L531-535）。实验中环境返回字符串 `"not-an-observation"`，`check_env` 仍然通过。同一环境改用 `warn=False` 则断言失败（F12）。因此：**通过了 `check_env` 的环境仍可能返回完全非法的观测**；该结论仅适用于 `warn=True` 路径，不能反推 `warn=False` 能完整支持这些空间。
- **`warn=False` 丢失保护（F11）**：无限动作边界的断言写在 `if warn:` 块内（L520-523），使用 `check_env(env, warn=False)` 的调用方不会拦截该情形。
- **`render` 不检查**：默认 `skip_render_check=True`（L467），且 `_check_render` 标注 `# pragma: no cover`（L442），上游自身也承认该分支不在 CI 覆盖内。
- **上游测试缺口**：非 `gym.Env` 类型、非 `Space` 空间、`reset` 无 `seed`、`step` 4 元组、reward/terminated 类型、无限动作边界这 6 条路径属于"源码有检查、测试无断言"，由本组运行实验首次给出可复现证据。

## 5.2 运行证据：契约在真实运行中的表现

T06 在 CartPole-v1 上运行最小预测、两轮短训练与效果评估，脚本见 [`scripts/issue_06_predict.py`](../../scripts/issue_06_predict.py)、[`issue_06_short_training.py`](../../scripts/issue_06_short_training.py)、[`issue_06_cartpole_effect.py`](../../scripts/issue_06_cartpole_effect.py)。

### 5.2.1 预测接口契约

| 接口 | 原生 Gymnasium Env | SB3 `DummyVecEnv` |
|---|---|---|
| `reset` | `(obs, info)`，必须先解包再把观测交给 `predict()` | 只返回批量观测，info 存于 `reset_infos` |
| `step` | `(obs, reward, terminated, truncated, info)` 五项 | `(obs, rewards, dones, infos)` 四项，`dones` 合并终止与截断 |
| 观测形状 | `(4,)` | `(1, 4)` |
| 混用后果 | 把 `(obs, info)` 直接传给 `predict()` 抛带 API 混用提示的 `ValueError` | — |

该 `ValueError` 由 `BasePolicy.predict` 的元组防护抛出（源码 [policies.py:L350](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L350)），属于**预测接口自身的防护，不是 `check_env` 的失败路径**——未检查的环境不会被自动拦截。

### 5.2.2 训练循环的可观察顺序

两轮短训练（每轮 16 步填满 buffer，共 32 步）的事件日志显示：每次 `collect_end` 时 buffer 已满，随后才是 `train_start → train_end`，`updates` 依次为 0、1、2。这支持第 4 章"采样后训练"的顺序（见 [4.2](04-call-chains.md)），但**只验证代码执行顺序，不评价策略效果**，也未验证 PPO 损失或奖励上升。

### 5.2.3 实际效果与限定

效果实验在训练前及多个检查点用相同的 10 个种子（100–109）运行确定性策略，记录每回合保持平衡的步数：

| 累计训练步数 | 平均持续步数 | 最短–最长 |
|---:|---:|---:|
| 0 | 98.1 | 79–126 |
| 4096 | 84.9 | 48–146 |
| 8192 | 91.1 | 51–146 |
| 16384 | 455.5 | 365–500 |
| 32768 | 161.3 | 147–194 |

**该轨迹不单调**，且末项未能复现：复核时末项得到 485.6 而非原始记录的 161.3，两次复核（Python 3.13.5 与对齐依赖后的 3.12.14）结果一致，差异原因尚未确定（见 [issue-06 复核记录](../evidence/issues/issue-06.md)）。因此本章不声称"训练越久效果越好"，也不把该下降视为已独立复现的现象。

## 5.3 证据边界与复现

| 触发情形 | 实际失败位置 | 是否被自动检查 |
|---|---|---|
| 环境类型、空间声明、`reset` 签名错误 | `check_env` 的对应断言 | 是，但需用户显式调用 `check_env` |
| `step` 返回 4 元组、reward/terminated 类型错误 | `check_env` 返回值检查 | 是（`warn=True` 且观测空间受支持时） |
| `Sequence`/`Graph`/`OneOf` 空间下返回非法观测 | 不失败，检查被跳过 | **否** |
| 把 `(obs, info)` 传给 `predict()` | `BasePolicy.predict` 元组防护 | 是（`predict` 自身防护） |
| 未检查的环境在训练中返回 4 项 `step` | `DummyVecEnv.step_wait` 解包处 | 否（`check_env` 不会在训练前自动运行） |

PPO 构造与 `learn()` 都**不会自动运行 `check_env`**，环境契约的验证责任在调用方。

复现入口：T05 见 [issue-05 证据的环境与命令](../evidence/issues/issue-05.md)（macOS arm64 / Python 3.12.15 / Gymnasium 1.4.0 / PyTorch 2.11.0；脚本启动时会校验 SB3 实际 commit 与源码洁净，不匹配即拒绝运行）；T06 见 [issue-06 运行入口](../../artifacts/runs/issue-06/README.md)（Python 3.12.14 / Gymnasium 1.3.0 / PyTorch 2.14.0，依赖清单 `pip-freeze.txt`）。失败用例为确定性触发，不依赖随机种子；效果评估受平台与数值运算差异影响，跨机复核应逐回合比较而非要求逐值一致。

本章不做的事：不评估训练算法优劣，不把"训练效果差"归为软件缺陷；未运行带 GUI 的 `render` 检查；未修改 T06 脚本与全组证据汇总文件。
