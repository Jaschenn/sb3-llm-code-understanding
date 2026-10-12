# Issue #5：环境检查与失败路径

负责人：@ufo58022271
复核人：（待组内确认后补充）
上游基线：见 `docs/project-baseline.md`（DLR-RM/stable-baselines3 @ `7cfb4dd6055e74b5caa4ed4d6777492209946e26`）

## 可检验结论

| 本 Issue 内编号 | 结论 | 状态（待验证/已支持/已否定/需限定） | 证据链接 |
|---|---|---|---|
| C-05-01 | `check_env()` 按固定顺序检查：环境类型 → 空间声明 → `reset(seed=)` 签名 → 空间支持性警告（仅 `warn=True`）→ 返回值检查 → NaN 检查；`render` 检查默认跳过 | 已支持 | 源码 `env_checker.py:L482-547`；运行 [summary.md](../../../artifacts/runs/issue-05/2026-10-12-env-checker-failures/summary.md) |
| C-05-02 | 环境类必须继承 `gymnasium.Env`，否则 `AssertionError`（运行 F01） | 已支持 | 源码 L482-484；运行 summary.md F01 |
| C-05-03 | `observation_space` / `action_space` 必须是 `gymnasium.spaces.Space` 实例，否则 `AssertionError`（运行 F02） | 已支持 | 源码 L421-427；运行 summary.md F02 |
| C-05-04 | `reset()` 必须接受 `seed` 参数，否则抛 `TypeError`（运行 F03） | 已支持 | 源码 L493-496；运行 summary.md F03 |
| C-05-05 | `reset()` 必须返回 `(obs, info)` 二元组、`step()` 必须返回 5 元组，且 reward 为 float/int、terminated/truncated 为 bool、info 为 dict，否则 `AssertionError`（运行 F04-F07） | 已支持 | 源码 L337-340、L367-370、L401-404；运行 summary.md F04-F07 |
| C-05-06 | `Box` 观测超出声明的边界时，`AssertionError` 报错带逐索引明细（运行 F08）；该路径已被上游测试覆盖 | 已支持 | 源码 L285-303；测试 `test_env_checker.py:L56-143`（`test_check_env_detailed_error`，本地 25 passed）；运行 summary.md F08 |
| C-05-07 | 在 `warn=True` 下，`Sequence`/`Graph`/`OneOf` 观测空间会触发警告并**跳过全部返回值检查**后直接返回（W09）；`warn=False` 不设置跳过标志，同一非法 Sequence 观测会断言失败（F12） | 需限定 | 源码 L147-169、L500-502、L531-535；原始运行 W09；[复核运行 W09/F12](../../../artifacts/runs/issue-05/2026-10-12-env-checker-review-fixes/summary.md) |
| C-05-08 | `Box` 动作空间：非对称/未归一化仅警告（运行 W10）；含无限边界时在 `warn=True` 下 `AssertionError`（运行 F11） | 已支持 | 源码 L510-523；运行 summary.md W10、F11 |
| C-05-09 | 上游测试未覆盖的失败路径：非 `gym.Env` 类型、非 `Space` 动作空间、`reset()` 无 `seed`、`step()` 返回 4 元组、reward/terminated 类型错误、无限动作边界——由本任务运行实验首次给出可复现证据 | 已支持 | 测试文件全文比对（`test_env_checker.py`，273 行）；运行 summary.md F01-F07、F11 |
| C-05-10 | `skip_render_check` 默认为 `True`，`render` 检查默认不执行（源码默认参数 L467、分支 L538-539） | 已支持 | 源码 L467、L538-539 |

## 证据记录

### 运行环境

全部实验于 2026-10-12 在以下环境执行，命令、版本与完整输出见
`artifacts/runs/issue-05/2026-10-12-env-checker-failures/`（`summary.md` 人读汇总、`results.json` 机器可读、`pytest_baseline.txt` 上游测试基线输出）：

- 平台 macOS-12.6-arm64；Python 3.12.15；PyTorch 2.11.0；Gymnasium 1.4.0；numpy 2.5.4
- SB3 为固定 commit `7cfb4dd` 源码（version.txt 报 2.9.2a0），以 `PYTHONPATH` 方式加载（未改动源码）
- 失败均为确定性触发，不依赖随机种子；`check_env` 内部固定使用 `reset(seed=0)`
- 复现命令：`PYTHONPATH=. .venv/bin/python ../sb3-llm-code-understanding/scripts/issue_05_env_checker_failures.py`

### 逐条证据

**C-05-01 检查顺序**：源码 [env_checker.py:L467-547](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L467-L547) 依次调用 `_check_spaces`（L487）→ `env.reset(seed=0)`（L493）→ `_check_unsupported_spaces`（L502，`warn=True` 时）→ `_check_returned_values`（L535）→ `_check_nan`（L545）。运行侧：F03 在返回值检查之前即因 `seed` 失败，F04-F07 在空间检查之后失败，与该顺序一致。

**C-05-02 环境类型**：源码 [L482-484](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L482-L484)。运行 F01（`NotAnEnv`，鸭子类型完全合法但不继承 `gym.Env`）→ `AssertionError: Your environment must inherit from the gymnasium.Env class`。两类证据齐。

**C-05-03 空间声明**：源码 [L421-427](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L421-L427)。运行 F02（`action_space = [0, 1]`）→ `AssertionError: The action space must inherit from gymnasium.spaces`。两类证据齐。

**C-05-04 seed 签名**：源码 [L493-496](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L493-L496)，`try: env.reset(seed=0) except TypeError`。运行 F03（`reset(self)` 无 seed）→ `TypeError: The reset() method must accept a 'seed' parameter`。两类证据齐。

**C-05-05 返回值结构**：源码 [L337-340](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L337-L340)（reset 元组）、[L367-370](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L367-L370)（step 5 元组）、[L401-404](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L401-L404)（类型断言）。运行 F04（旧 API 4 元组）、F05（reset 无 info）、F06（reward 为字符串）、F07（terminated 为 int）分别触发对应断言，报错原文见 summary.md。两类证据齐。

**C-05-06 观测边界**：源码 [L285-303](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L285-L303)，`invalid_indices` 逐索引列出。运行 F08（声明 `Box(0,1)` 却返回 `[0.5, 1.5, 0.5]`）→ 报错含 `Expected: 0.0 <= obs[1] <= 1.0, actual value: 1.5`。上游测试 `test_check_env_detailed_error`（[test_env_checker.py:L56-143](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/tests/test_env_checker.py#L56-L143)）以 18 个参数化用例覆盖越界/dtype/shape/多空格场景，本地运行 25 passed（见 `pytest_baseline.txt`）。三类证据齐。

**C-05-07 静默通过边界（需限定）**：源码 [L156-162](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L156-L162)（Sequence 警告并置 `should_skip=True`）、[L500-502](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L500-L502)（仅 `warn=True` 时调用空间支持性检查）、[L531-535](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L531-L535)（skip 后直接 return，否则检查返回值）。运行 W09：`SequenceObsEnv` 的 `reset()`/`step()` **故意返回字符串** `"not-an-observation"`，`check_env(..., warn=True)` 仍然"通过"，仅警告 `checks for returned values are skipped`。复核运行 F12 使用同一环境但设置 `warn=False`，无警告且抛出 `AssertionError: The observation returned by reset() method must be a numpy array`。含义：`check_env` 通过 ≠ 环境正确；跳过返回值检查的结论仅适用于 `warn=True`，不应推论 `warn=False` 能完整支持这些空间。此边界同时解释了为什么 Graph/OneOf 场景在测试中以 `warn=True` 断言警告（[test_env_checker.py:L212-273](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/tests/test_env_checker.py#L212-L273)）。

**C-05-08 动作空间边界**：源码 [L510-518](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L510-L518)（非对称仅警告）、[L520-523](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L520-L523)（无限边界断言，注意位于 `if warn:` 块内）。运行 W10（`Box(-2, 1)`）通过+警告；F11（`low=-inf`）断言失败。推论：`warn=False` 时无限边界动作空间不会被拦截——使用 `check_env(env, warn=False)` 的调用方失去这层保护。

**C-05-09 测试覆盖缺口**：逐行比对 [test_env_checker.py](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/tests/test_env_checker.py)（273 行）全部 25 个用例：覆盖 obs 越界/dtype/shape 明细、dict action 警告、Graph/Sequence/OneOf 警告、单步环境、goal env（经 `test_env_checker` 未含，实为 `check_goal_env` 在 `test_make_env.py` 中，此处按本文件范围说明）。**本文件未覆盖** C-05-02/03/04/05（类型、空间声明、seed、返回值元数与类型）与 F11（无限边界）——这些路径仅有源码断言，无回归测试；本任务的运行实验（F01-F07、F11）给出可复现行为证据。建议（供报告第 5 章）：此为"源码有检查、测试无断言"的覆盖缺口清单。

**C-05-10 render 默认跳过**：源码 [L467](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L467)（默认参数 `skip_render_check=True`）、[L538-539](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L538-L539)（分支），且 `_check_render` 标注 `# pragma: no cover`（[L442](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/env_checker.py#L442)），说明上游自身也承认该分支不被 CI 覆盖。源码类单证据，标注为"已支持"仅指默认行为；render 合法性验证超出本任务运行范围。

## AI 使用记录

| AI 编号 | 模型与提问原文/链接 | AI 回答摘要 | 验证动作 | 观察结果 | 修正后的表述 |
|---|---|---|---|---|---|
| AI-05-01 | LLM（WorkBuddy 会话，2026-10-12）："列出 SB3 check_env() 对 Gymnasium 环境的全部检查点与检查顺序" | 给出检查点清单：类型、空间、seed、返回值、NaN、render | 逐行核对固定 commit 的 `env_checker.py` L467-547 | 清单与源码一致，但 AI 最初未强调"无限边界断言仅在 warn=True 分支内" | 修正：无限边界检查是条件性的（C-05-08），调用方用 warn=False 会失去该检查 |
| AI-05-02 | LLM（WorkBuddy 会话，2026-10-12）："Sequence/Graph/OneOf 观测空间下 check_env 的行为？" | 假设："会警告并跳过后续返回值检查，可能静默通过" | 1) 源码核对 L147-169、L531-532；2) 设计 W09 实验：返回非法字符串观测 | 假设被实验证实：返回 `"not-an-observation"` 仍通过检查 | 表述成立并升级为"需限定"结论（C-05-07）：check_env 通过不能视为环境正确的充分条件 |
| AI-05-03 | LLM（WorkBuddy 会话，2026-10-12）："上游 test_env_checker.py 覆盖了哪些失败路径？" | 初步假设："主要覆盖观测值明细报错" | 逐行比对测试文件全部 25 个参数化用例并本地运行 | 覆盖面比假设更宽（含 Graph/Sequence/OneOf/dict action），但确实缺 C-05-09 所列 6 条路径 | 修正为精确的覆盖/缺口清单（C-05-09），并以本地 25 passed 作证 |

## 复核记录

2026-10-12，PR #19 复核修正：

- C-05-07 增加 `warn=True` 限定和 F12（`warn=False`）对照；12 个用例的实际结果见[复核运行](../../../artifacts/runs/issue-05/2026-10-12-env-checker-review-fixes/summary.md)。原始 11 用例运行记录保留，两个运行目录各自记录依赖版本。
- 脚本从实际导入的 `check_env` 文件定位 SB3 Git 仓库，读取 HEAD 并核对固定基线，同时要求源码目录没有未提交修改。PyPI 安装版、其他 commit 或修改过的源码会在写出实验记录之前被拒绝；记录增加实际导入路径、工作目录和真实执行命令。
- 复现修正后的运行（在本仓库根目录，`SB3_SOURCE` 为固定 commit 的源码仓库，`SB3_PYTHON` 为已安装运行依赖的解释器路径）：

  ```bash
  PYTHONPATH="$SB3_SOURCE" uv run --no-project --python "$SB3_PYTHON" python \
    scripts/issue_05_env_checker_failures.py \
    --output-dir artifacts/runs/issue-05/2026-10-12-env-checker-review-fixes
  ```

复核对应 PR：[PR #19](https://github.com/Jaschenn/sb3-llm-code-understanding/pull/19)。

## 边界与不做的事

- 本任务只分析 `check_env` 对环境接口的检查行为；不评估训练效果，不把"训练收敛差"归为软件 bug。
- 未运行带 GUI 的 render 检查（`skip_render_check=True` 为默认且源码标注 no cover）；如需补充需图形环境，留待后续。
- 未修改 T06 的实验脚本与全组证据汇总文件；`docs/report/05-tests-boundaries.md` 待 T05/T06 材料齐后按 Issue 要求另行整合，不在本 PR 范围。
