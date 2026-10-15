# Issue #8：Issue、PR 与历史维护观察

负责人：@yangza0707
复核人：@langzim（对应本 Issue 正文的「建议复核人：成员 3」）
上游基线：见 `docs/project-baseline.md`（DLR-RM/stable-baselines3 @ `7cfb4dd6055e74b5caa4ed4d6777492209946e26`）

## 可检验结论

| 本 Issue 内编号 | 结论 | 状态（待验证/已支持/已否定/需限定） | 证据链接 |
|---|---|---|---|
| C-08-01 | 维护者把「env checker 无法处理嵌套在复合空间（`Dict`/`Tuple`/`OneOf`）内的 `Sequence` 观测空间」认定为真实缺陷，并走完「报告 → 开 PR → 合并关闭」的完整维护流程 | 已支持 | [Issue #2172](https://github.com/DLR-RM/stable-baselines3/issues/2172)；[PR #2174](https://github.com/DLR-RM/stable-baselines3/pull/2174)；merge commit `7883ed4d20bc9edf3a22a053c77b3146c518a937` |
| C-08-02 | 修复的方向**不是**让检查器支持 `Sequence`，而是在 `_check_unsupported_spaces()` 中检查 `Dict` / `Tuple` 的**直接子空间**（`observation_space.spaces`，只遍历一层）里有没有 `Sequence`，置 `sequence_space=True` 后统一告警并 `should_skip=True`（即放弃返回值检查）。**该实现不是任意层级的递归**：形如 `Dict → Tuple → Sequence` 的两层嵌套仍不会被识别 | 已支持（限定：仅直接子空间，非递归） | 基线 `env_checker.py:L103-111`（Dict，遍历 `observation_space.spaces.items()`）、`L140-144`（Tuple，遍历 `observation_space.spaces`）、`L156-162`（顶层 `Sequence` 或 `sequence_space`）；「无递归」与「只遍历直接子空间」分别由 V-08-15、V-08-16 机械校验 |
| C-08-03 | 合并前维护者自行改写了补丁，并由此**新增**了 `OneOf` 分支：`OneOf` 观测空间此前不在 `_check_unsupported_spaces()` 中被识别，此后会告警并 `should_skip=True`，等于新增一条「`check_env` 通过 ≠ 环境正确」的静默路径。需限定的是：**该分支只在 `warn=True` 时生效**——`_check_unsupported_spaces()` 仅在 `check_env()` 的 `if warn:` 分支内被调用（`L501-502`），`warn` 默认为 `True`；显式传 `warn=False` 时既不告警也不跳过检查 | 已支持（限定：`warn=True`） | 维护者 commit [`180ab8c`](https://github.com/DLR-RM/stable-baselines3/commit/180ab8cce695961248cb54863022729984af850a)；基线 `env_checker.py:L12-20`、`L146-152`、调用点 `L501-502`（由 V-08-14 机械校验） |
| C-08-04 | AI（GitHub Copilot 编码代理）提交的原始补丁缺少向后兼容保护：直接 `isinstance(observation_space, spaces.OneOf)`，在 Gymnasium < 1.0 会 `AttributeError`；维护者补上 `_is_oneof_space()`（`try/except AttributeError`）并追加 mypy 抑制 | 已支持 | commit [`9bab86d`](https://github.com/DLR-RM/stable-baselines3/commit/9bab86d932174d0510f7ffd1254da68bebbb6015) → [`180ab8c`](https://github.com/DLR-RM/stable-baselines3/commit/180ab8cce695961248cb54863022729984af850a) → [`9c052b2`](https://github.com/DLR-RM/stable-baselines3/commit/9c052b2f27ceb0e6aa4d8a943043d784b4bc4b34)；基线 `env_checker.py:L12-20` |
| C-08-05 | 测试同样被人工简化：AI 原始提交为三个独立 `gym.Env` 子类、并参数化 `stack=[False, True]`（+83 行）；维护者改为继承仓库既有 `CustomEnv`、去掉参数化（该 commit 净 −85/+58） | 已支持 | commit `9bab86d` 与 `180ab8c` 的 diff；基线 `tests/test_env_checker.py:L221-273` |
| C-08-06 | 该修复随 v2.7.1（2025-12-05）发布，并已存在于本报告固定基线 `7cfb4dd` 中；报告第 5 章关于 `Sequence`/`OneOf` 的边界结论与之不冲突 | 已支持 | 基线 `docs/misc/changelog.md:L158`（位于 `## Release 2.7.1 (2025-12-05)` 段内）；基线 `git log -1` = `7cfb4dd Fix for too many open files with subproc vec env (#2284)` |
| C-08-07 | 对照案例：Issue #1950（`predict()` 收到 `reset()` 返回的 tuple）中，维护者 araffin 的动作**仅限于标签层面**（2024-06-21 移除 `bug`、改加 `custom gym env` 与 `check the checklist`）；**把该 Issue 关闭的是报告者本人 KevinHan1209**（2024-06-23，`not_planned`），因此**不能**把它表述为「维护者判定不改并关闭」。该 Issue 全程无任何代码变更 | 已支持 | `closed_by = KevinHan1209`，与 `user`（报告者）同一账号；标签事件的 `actor` 均为 `araffin`；`state_reason=not_planned`、`closed_at=2024-06-23T19:01:34Z` |
| C-08-08 | 该对照案例的 2 条评论均由非维护者发出（`author_association` 皆为 `NONE`），内容是给出正确用法 `obs, info = env.reset(...)`；维护者本人未留下文字评论 | 已支持 | [Issue #1950 评论](https://github.com/DLR-RM/stable-baselines3/issues/1950#issuecomment-2880773977)（@nyakotey, 2025-05-14）、#issuecomment-2543030592（@sidrai111, 2024-12-14） |
| C-08-09 | 基线 `policies.py` 的报错文本自身承担了引导职责：直接说明「You are probably mixing Gym API with SB3 VecEnv API」并给出文档链接 | 已支持（源码类单证据） | 基线 `stable_baselines3/common/policies.py:L357-362` |
| C-08-10 | 推断（**非事实**）：维护者 araffin 在 #1950 上改标签（移除 `bug`、加 `check the checklist`）时的判断依据，可能与上述报错自带的说明有关。需要区分的是：**关闭动作由报告者本人作出**（见 C-08-07），其关闭动机无任何记录，不得归为维护者的判定 | 待验证 | Issue #1950 中无维护者原话可引；仅有标签事件，且 `closed_by` 为报告者本人 |

## 证据记录

### 观察对象与选取标准

Issue #8 要求「选与环境兼容、预测/训练流程或报告主线相关的一组已关闭 Issue 与对应 PR/commit，再核对固定基线中最终的代码状态」。据此定下两条筛选条件：

1. 与报告主线相关——本报告第 5 章的边界结论集中在 `check_env()` 与 `VecEnv` API（见 T05 的 C-05-07），因此优先取 `env_checker` 与 Gym/VecEnv 接口相关条目。
2. 「一组」要能形成对照——既要有**有代码变更**的完整链条，也要有**最终未产生任何代码变更**的关闭决策，才能说明维护者实际在做什么判断。

最终选取：

- **主案例 A**：Issue #2172 → PR #2174 → merge commit `7883ed4`（有完整代码 / 测试 / 变更日志改动）。
- **对照案例 B**：Issue #1950 → 无代码变更，由**报告者本人**以 `not_planned` 关闭；维护者的动作止于标签调整，不代表维护者作出过「判定不改」的结论（见 C-08-07）。

### 主案例 A：Issue #2172 → PR #2174 → merge commit `7883ed4`

**问题（可用维护者与提交者原话核对的起点）。** 报告者 FredBill1 于 2025-08-29 给出最小复现：当 `Sequence` 空间被放进 `Dict`、`Tuple` 或 `OneOf` 时，`check_env()` 抛 `AssertionError` 或 `TypeError`，而不是按既有约定给出警告并跳过返回值检查。原文结论句：「`Sequence` space is not properly handled when putting inside a composite space (`Dict`, `Tuple`, or `OneOf`), causing the `stable_baselines3.common.env_checker.check_env` to fail.」

**讨论中的方案（PR 正文，属 AI 生成内容，需人工核验）。** PR #2174 由 GitHub Copilot 编码代理（`copilot-swe-agent[bot]`）开出，正文中的根因判断为：「The `_check_unsupported_spaces()` function properly detected `Sequence` spaces at the top level and set `should_skip=True` to bypass return value checks. However, it didn't recursively check for `Sequence` spaces within composite spaces」——即原来只做了顶层判断，漏了复合空间内部的 `Sequence`。该正文同时把方案限定为「检测复合空间内的 `Sequence` 并置 `should_skip`」，而没有声称要让 SB3 支持 `Sequence`。

> **对 AI 用词的一处修正（2026-10-15 复核后补）。** 上述引文中的 "recursively" 是 AI 对其自身补丁的概括，与实际实现不符：无论 AI 的原始提交（`9bab86d`）还是维护者改写后的合并态，都只遍历 `Dict` / `Tuple` 的**直接子空间**一层，没有对任意层级递归（合并态见 `L105`、`L140`；「只遍历直接子空间」由 V-08-16 校验，「无递归」由 V-08-15 以「全文件仅出现 2 次 `_check_unsupported_spaces`（定义 + 唯一调用点）」机械校验）。因此本文件一律表述为「检查直接子空间」，不使用「递归」一词。

**最终决定与变更（4 个 commit，逐条可核）。**

| 顺序 | commit | 作者 | 提交信息 | 变更量 |
|---|---|---|---|---|
| 1 | `49389c2` | `copilot-swe-agent[bot]` | Initial plan | — |
| 2 | `9bab86d` | `copilot-swe-agent[bot]` | Fix env checker to handle Sequence spaces in composite spaces（含 `Co-authored-by: araffin`） | +105 / −1 |
| 3 | `180ab8c` | **Antonin RAFFIN（araffin，维护者）** | Simplify tests and checks | +58 / −85 |
| 4 | `9c052b2` | **Antonin RAFFIN** | Add fix for mypy for gym<1.0 | +2 / −2 |

这三步构成了本 Issue 最关键的一条维护观察：**AI 提出补丁，维护者没有直接合并，而是自己重写后才合并**。

- 第 2→3 步之间，`env_checker.py` 的 OneOf 处理从「`isinstance(observation_space, spaces.OneOf)` 后仅检测内部 `Sequence`」改为 `_is_oneof_space()` 守卫，并**改成对 OneOf 本身告警 + `should_skip=True`**（见 [commit `180ab8c` 的 env_checker.py 补丁](https://github.com/DLR-RM/stable-baselines3/commit/180ab8cce695961248cb54863022729984af850a)）。这一步扩大的是「跳过检查」的范围，属于维护者对边界的重新划定，而不是 AI 原始方案的一部分。
- 第 2→3 步之间，测试从三个独立 `gym.Env` 子类（各自复制 `reset`/`step`）改为继承仓库已有的 `CustomEnv`，并删除 `stack` 参数化；维护者提交信息自述为 "Simplify tests and checks"。
- 第 4 步只改两行：给 `_is_oneof_space()` 补 `# type: ignore[attr-defined]`，并去掉测试环境类多余的 `stack` 形参，提交信息自述为 "Add fix for mypy for gym<1.0"。这说明**兼容性与类型检查是合并前的实际门槛**。

**最终合并态**：PR #2174 于 2025-09-08T15:48:22Z 由 araffin 合并，merge commit `7883ed4d20bc9edf3a22a053c77b3146c518a937`；相对 base 共改动 3 个文件、+89 / −12：

| 文件 | 改动 |
|---|---|
| `stable_baselines3/common/env_checker.py` | +32 / −4 |
| `tests/test_env_checker.py` | +55 / −7 |
| `docs/misc/changelog.rst` | +2 / −1 |

**回归测试。** AI 原始提交新增了三个测试环境类；合并态保留为 `SequenceInDictEnv`、`SequenceInTupleEnv`、`SequenceInOneOfEnv`，并把原 `test_check_env_sequence_obs` 参数化为 `[CustomEnv, SequenceInDictEnv]`，另加 `test_check_env_sequence_tuple`（要求同时匹配 `Sequence.*` 与 `Tuple.*` 两条告警）与 `test_check_env_oneof`（用 `pytest.skip` 兜底 Gymnasium 版本不支持 `OneOf` 的情况）。基线中对应位置：`tests/test_env_checker.py:L221-273`。**这一层回归测试的存在，使 C-08-03 关于 OneOf 的新行为可以被复核人独立复现。**

**基线核对（本任务要求的「再核对固定基线中最终的代码状态」）。**

| 核对项 | 基线 `7cfb4dd` 的实际状态 | 是否与 PR 结论一致 |
|---|---|---|
| `should_skip = graph_space = sequence_space = False` | `env_checker.py:L102` | 一致 |
| `Dict` 内 `Sequence` 识别 | `env_checker.py:L110-111` | 一致 |
| `Tuple` 内 `Sequence`/`Graph` 识别 | `env_checker.py:L139-144` | 一致 |
| `OneOf` 告警 + `should_skip=True` | `env_checker.py:L146-152`（守卫函数 `L12-20`） | 一致 |
| `Sequence` 或嵌套 `sequence_space` 告警 + `should_skip=True` | `env_checker.py:L156-162` | 一致 |
| 新增测试用例 | `tests/test_env_checker.py:L221-273`（文件共 273 行） | 一致 |
| 变更日志条目 | `docs/misc/changelog.md:L158`（`## Release 2.7.1 (2025-12-05)` 段内 Bug Fixes） | 一致 |
| **（复核后补入）** `_check_unsupported_spaces()` 调用点的 `warn` 门控 | `env_checker.py:L501-502`，位于 `check_env` 的 `if warn:` 分支内 | 一致（V-08-14）——据此限定 C-08-03 |
| **（复核后补入）** 无任意层级递归 | `L95` 定义、`L502` 唯一调用点，全文件仅 2 处 `_check_unsupported_spaces` | 一致（V-08-15）——据此限定 C-08-02 |
| **（复核后补入）** 只遍历直接子空间 | `L105`（Dict：`observation_space.spaces.items()`）、`L140`（Tuple：`observation_space.spaces`） | 一致（V-08-16） |

**一处需要重新定位的路径变化。** PR #2174 改的文件是 `docs/misc/changelog.rst`，而基线中该文件已改名为 `docs/misc/changelog.md`。需要说明的是：**历史链接本身并未失效**——PR 页面上的文件链接是指向该 PR 自身 base/head 的 commit，按 commit 定位的引用仍然可以访问。发生变化的是**跨版本的文件路径**：若要按路径跳到固定基线上的同名文件，必须重新定位（`changelog.rst` → `changelog.md`）。这一点在「只摘录 Issue 页面」的做法下看不出来。

**可复现的核对脚本。** 上表各核对项对应 `scripts/issue_08_verify_chain.py` 的 **16 项**检查（含复核后补入的 V-08-14/15/16）：脚本逐项断言上述标记存在，并在 SB3 目录的 Git HEAD 等于固定基线时进一步校验行号。脚本只读取文本文件，不联网、不导入 SB3、不运行训练。

```bash
python scripts/issue_08_verify_chain.py \
  --sb3-source <检出到固定基线 commit 的 SB3 源码目录> \
  --output-dir artifacts/runs/issue-08/2026-10-15-review-fixes
```

本次运行环境为 Windows，SB3 目录 HEAD = `7cfb4dd6055e74b5caa4ed4d6777492209946e26`（等于基线，行号校验启用），**16 项检查全部 PASS**；记录见 [summary.md](../../../artifacts/runs/issue-08/2026-10-15-review-fixes/summary.md)、[results.json](../../../artifacts/runs/issue-08/2026-10-15-review-fixes/results.json)。复核前的 13 项版本保留在 `artifacts/runs/issue-08/2026-10-14-baseline-verification/`。脚本在 HEAD 不等于基线时只校验标记是否存在、不校验行号，并在记录中显式标注该情形，避免把「换了 commit 也能过」误读为结论成立。

### 对照案例 B：Issue #1950（报告者自行关闭，无代码变更）

- 2024-06-21 报告者 KevinHan1209 报 bug：在 `learn.py` 示例中调用 `model.predict(obs)` 抛
  `ValueError: You have passed a tuple to the predict() function instead of a Numpy array or a Dict.`
- 2024-06-21 维护者 araffin **移除 `bug` 标签**，改加 `custom gym env`（"Issue related to Custom Gym Env"）与 `check the checklist`（"You have checked the required items in the checklist but you didn't do what is written..."）两个标签。这是维护者在本 Issue 上**唯一**可核对的动作。
- 2024-06-23 该 Issue 以 `state_reason = not_planned` 关闭。**关闭者是报告者本人 KevinHan1209**（`closed_by = KevinHan1209`，与其 `user` 账号一致），不是维护者。因此本案例**不能**写成「维护者判定不改并关闭」——可确认的是「维护者调整了分类、报告者自行关闭」，而关闭动机无任何记录。
- 该 Issue 共 2 条评论，`author_association` 均为 `NONE`，即**都不是维护者**：@sidrai111 问 "Did you find the solution?"（2024-12-14），@nyakotey 在 2025-05-14 给出正确用法——把 `obs = test_env.reset(seed=42, options={})` 改为 `obs, info = test_env.reset(seed=42, options={})`。
- 基线 `stable_baselines3/common/policies.py:L357-362` 保留了这段报错文本，其中直接写明 API 混用原因并附文档链接。

### 维护者原话 vs 小组推断（按 Issue #8 要求显式区分）

**可直接引用的原话：**

- 维护者 araffin（commit message）：`Simplify tests and checks`；`Add fix for mypy for gym<1.0`。
- AI 代理的原话（PR #2174 正文）：「it didn't recursively check for `Sequence` spaces within composite spaces, so nested `Sequence` spaces weren't detected and the checker tried to validate their return values, leading to failures.」该表述中「顶层已识别、嵌套被漏掉」这一判断经与 commit diff、基线源码逐条对照后成立（C-08-02）；但其中的 **"recursively" 与实际实现不符**——实现只遍历直接子空间一层，并无递归。本文件因此不沿用该用词。
- 报告者原话：Issue #2172 的 `Sequence space is not properly handled when putting inside a composite space…`；Issue #1950 的报错文本。
- 关闭动作本身也是可核对的事实：Issue #1950 的 `closed_by = KevinHan1209`，即**报告者本人**完成了关闭（C-08-07）。
- 维护者的标签选择本身是可核对的动作（标签名与其描述文本即为原文），但**标签背后的理由没有文字记录**。

**属小组推断、不得写成事实：**

- C-08-10：维护者 araffin **改标签**（移除 `bug`、加 `check the checklist`）时的判断依据，可能与 `policies.py` 的报错说明有关。Issue 中无维护者文字评论，仅有标签事件，因此标为「待验证」。注意区分：**关闭 #1950 的是报告者本人，不是维护者**，关闭动机无记录，不得写成「维护者判定不改」。
- 维护者改写 Copilot 补丁的**动机**（是出于 `gym<1.0` 兼容、mypy、还是代码风格偏好）。commit message 只给出了动作描述（"Simplify"、"Add fix for mypy for gym<1.0"），可确认的只有「改了什么」，不是「为什么这么改」。

## AI 使用记录

| AI 编号 | 模型与提问原文/链接 | AI 回答摘要 | 验证动作 | 观察结果 | 修正后的表述 |
|---|---|---|---|---|---|
| AI-08-01 | LLM（WorkBuddy 会话，2026-10-14）："推荐一组与 env checker / predict 主线相关的已关闭 SB3 Issue，且要能串成 Issue→PR→commit" | 首先推荐 Issue #1950（`predict()` 收到 tuple）作为主案例，理由是它最贴合 T03 的「原生 `Env` vs SB3 `VecEnv`」边界 | 用 GitHub API 拉取 #1950 的 `state_reason`、标签事件与全部评论；并核对是否有关联 PR | **假设被否定**：#1950 无任何 PR/commit 改动，不满足 Issue #8「从 Issue 找到 PR/commit 再找到实际修改或回归测试」的验收标准 | 主案例改为 #2172 → PR #2174 → `7883ed4`；#1950 降级为**对照案例**（无代码变更），其结论改为 C-08-07 / C-08-08 |
| AI-08-02 | 同上会话："PR #2174 的 `_is_oneof_space` 是不是为支持 `OneOf` 而加的？" | AI 初步表述为「新增了对 `OneOf` 空间的支持」 | 逐条比对 PR 的 4 个 commit diff 与基线 `env_checker.py:L12-20`、`L146-152` | 表述不准确：`_is_oneof_space()` 只做类型判断，实际效果是**放弃对 OneOf 的返回值检查**（`should_skip=True`），方向与「支持」相反；且该分支由维护者 commit `180ab8c` 引入，非 AI 原始方案 | 改为 C-08-03：这是新增的一条**静默通过路径**，并据此与 T05 的 C-05-07 交叉印证 |
| AI-08-03 | 同上会话："这组改动进了哪个版本？基线里还在吗？" | AI 给出「应属于 2.7.x」 | 在本地固定基线源码中核对 `git log`、`env_checker.py`、`tests/test_env_checker.py`、`docs/misc/changelog.md` 的行号与所属版本段 | 与结论一致，且发现 AI 未提到的一点：PR 改的是 `changelog.rst`，基线中已改名为 `changelog.md` | 补入「跨版本路径变化」观察（当时误写为「链接失效」，已在 AI-08-04 中修正），并明确结论 C-08-06 的版本归属为 v2.7.1（2025-12-05） |
| AI-08-04 | 同一份材料经复核人审阅（2026-10-15），指出 AI 两处表述不准确 | AI 原稿写有「在 `_check_unsupported_spaces()` 中**递归**发现嵌套的 `Sequence`」，并把 #1950 概括为「**维护者判定不改并关闭**」 | ① 调 `issues/1950` 的 `closed_by` 与 timeline 标签事件；② 通读基线 `env_checker.py:L95-170` 与 `check_env` 调用点 `L501-502`；③ 新增 V-08-14/15/16 三项机械校验并重跑脚本（16 项全 PASS） | **两处均被证实为不准确**：①「递归」沿用了 PR 正文的用词，而实现只遍历 `Dict`/`Tuple` 的直接子空间；② #1950 的 `closed_by` 是报告者 `KevinHan1209`，维护者只改过标签，「判定不改」属把标签动作误读为关闭决定。此外复核还指出「OneOf 告警并跳过」只在 `warn=True` 下发生、以及「历史链接失效」应表述为「跨版本路径变化」 | 已在结论表与正文全面改写：C-08-02 限定为「直接子空间，非递归」，C-08-03 限定 `warn=True`，C-08-07 改为「维护者仅改标签、报告者自行关闭」，C-08-10 限定为「改标签的依据」；并把这两处修正固化成 V-08-14/15/16 三项可重跑的检查 |

## 复核记录

### 第一期复核（2026-10-15）

复核人反馈了三处表述瑕疵。三项均已接受，并逐处核实、修改。复核人署名以 GitHub PR #21 上的 review 为准。

| # | 复核指出的问题 | 我方核实过程 | 处理结果 |
|---|---|---|---|
| 1 | Issue #1950 的关闭者写错了：实际是**报告者 KevinHan1209 自行关闭**，维护者 araffin 只调整了标签。因此不能归纳为「维护者判定不改并关闭」 | `issues/1950` 返回 `closed_by = KevinHan1209`，与 `user`（报告者）为同一账号；timeline 中 3 条标签事件的 `actor` 均为 `araffin`，`closed` 事件的 actor 为 `KevinHan1209` | **成立**。改写 C-08-07（维护者仅改标签、报告者自行关闭）、C-08-10（限定为「改标签的依据」，并明确关闭者非维护者）、对照案例 B 正文、7.1 选取标准；`maintenance-chain.mmd` 的 B 链条同步修改并重渲染 |
| 2 | 「递归检测 `Sequence`」表述不准确：源码只检查 `Dict`、`Tuple` 的**直接子空间**，没有递归检查任意层级；另需注明 OneOf 告警后跳过检查发生在 `warn=True` 条件下 | 通读基线 `env_checker.py:L95-170`：Dict 分支遍历 `observation_space.spaces.items()`（`L105`），Tuple 分支遍历 `observation_space.spaces`（`L140`），均为单层；`_check_unsupported_spaces` 全文件仅出现 2 次（`L95` 定义、`L502` 调用），无自调用。调用点位于 `check_env` 的 `if warn:` 分支内（`L501-502`），`warn` 默认 `True` | **成立**。C-08-02 改为「检查直接子空间，非递归」并加限定；C-08-03 补「仅在 `warn=True` 下生效」；正文对 PR 正文中 AI 自述的 "recursively" 一词加注修正；新增 V-08-14 / V-08-15 / V-08-16 三项机械校验 |
| 3 | 「文件改名导致历史链接失效」表述不准确：**固定 commit 下的历史链接仍可访问**，建议改为「跨版本文件路径发生变化，需要重新定位」 | PR #2174 的文件链接指的是该 PR 自身 base/head 的 commit，不依赖默认分支的最新路径 | **成立**。已改写为「跨版本路径变化，按路径引用时需重新定位」，并明确「链接本身并未失效」 |

复核后重跑核对脚本，记录见 [`artifacts/runs/issue-08/2026-10-15-review-fixes/`](../../../artifacts/runs/issue-08/2026-10-15-review-fixes/summary.md)：**16 项全部 PASS**（原 13 项 + 新增 V-08-14/15/16）。复核前的运行记录保留在 `artifacts/runs/issue-08/2026-10-14-baseline-verification/`。

### 复核清单（供后续复核人沿用）

1. Issue #2172 → PR #2174 → merge commit `7883ed4` 的链接是否连通；
2. 基线 `env_checker.py:L102-162` 是否与 C-08-02 / C-08-03 的描述一致（可重跑 `scripts/issue_08_verify_chain.py`，本次 16 项全 PASS）；
3. `tests/test_env_checker.py:L221-273` 是否构成 C-08-03 的回归测试；
4. C-08-10 是否已按「待验证」处理、未被写成事实。

复核对应 PR：[#21](https://github.com/Jaschenn/sb3-llm-code-understanding/pull/21)（跨仓库提交：`yangza0707:task/8-maintenance-history` → `Jaschenn:main`，2026-10-14 开出）。已通过 GitHub 的 review request 正式指定 @langzim 复核，并在该 PR 下留言附上复核最短路径（含核对脚本的重跑命令）。本 Issue 的 Assignee 已设为 @yangza0707，状态标签为 `status: review`。

## 边界与不做的事

- 本任务只核对**已发生的维护历史**与固定基线中的最终状态，不评价该修复方案的技术优劣，也不修改上游 SB3 仓库。
- 未复盘 PPO 算法或训练效果；所选案例限定在 `env_checker` 与 `VecEnv` API 这条主线上。
- 未运行上游测试套件：本任务的证据类型是 Issue/PR/commit 历史 + 源码定位，不需要运行。已运行的是 `scripts/issue_08_verify_chain.py`（只读文本文件、不导入 SB3）。若复核人希望把 `test_check_env_oneof`、`test_check_env_sequence_tuple` 等用例实际跑一遍，可在固定基线下执行 `pytest tests/test_env_checker.py`，该运行记录不在本 PR 范围内。
- 未修改 `docs/evidence/evidence-matrix.md`、`docs/evidence/ai-log.md`、`tasks/README.md`、`docs/report/README.md` 等汇总文件。
- C-08-03 涉及的行为边界与 T05 的 C-05-07 同源，已按 Issue #8 要求把该可验证发现反馈给 T05 负责人（见 `docs/report/07-maintenance.md`）。
