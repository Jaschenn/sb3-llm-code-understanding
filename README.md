# SB3：使用大语言模型辅助代码理解

《软件测试和分析》课程小组实训协作仓库。我们以 Stable-Baselines3（SB3）为案例，研究大语言模型如何辅助理解陌生项目，并用源码、测试、运行和维护历史交叉验证结论。

## 研究范围

主问题：SB3 如何将一个符合 Gymnasium 接口的环境，组织为可验证的预测和训练软件流程？

本项目分析对象协作、控制流、接口约束与失败路径；不证明 PPO 的数学收敛性或推导其损失函数。

## 新成员从这里开始

1. 阅读[上游源码基线](docs/project-baseline.md)和[报告目录](docs/report/README.md)。所有源码引用使用同一个 SB3 commit；分析时在自己的电脑上克隆 SB3，本仓库只存放分析材料。
2. 从[任务列表](tasks/README.md)打开一个 `status: todo` 的 Issue。Issue 中写明第一步、要提交的文件、完成标准和不属于本任务的内容。
3. 在 Issue 中将自己设为 **Assignee**，评论认领计划，并将标签改为 `status: claimed`。建议负责人只是建议，实际负责人以 Assignee 为准。
4. 从最新 `main` 建立 `task/<issue号>-<英文短名>` 分支。先按[证据模板](docs/evidence/issues/TEMPLATE.md)写自己的 `issue-<编号>.md`，再补脚本、图和对应章节。
5. 提交 PR，在描述中写 `Closes #<issue号>`、验证命令和复核人。非提交者复核后合并，Issue 自动关闭。完整规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。

例如认领 [T05](https://github.com/Jaschenn/sb3-llm-code-understanding/issues/5)：先读 `env_checker.py` 与对应测试，构造一个错误环境并运行，然后在 `docs/evidence/issues/issue-05.md` 记录“触发条件 → 源码检查 → 实际结果 → 得出的边界”。代码放 `scripts/issue_05_*.py`，输出放 `artifacts/runs/issue-05/`。

## 任务先后关系

| 时机 | 任务 | 如何衔接 |
|---|---|---|
| 全员开始前 | 统一 SB3 commit，了解证据模板 | 这是所有任务的共同前提；T01 项目地图可同步开展，无需等它完成才开始其他任务 |
| 第一轮，可并行 | T01 项目地图、T02 对象关系、T05 失败路径、T06 运行实验、T08 维护历史 | 各自先产出独立的 Issue 证据文件，互相分享发现 |
| 第二轮，可开始探索，定稿前交叉核对 | T03 预测调用链、T04 训练调用链 | 与 T02 统一对象名称；请 T06 核对关键运行行为，T05 提供失败路径 |
| 持续记录，最后汇总 | T07 AI 日志与反思；组长整合报告 | 每个人从第一天记录自己的 AI 提问与核验；T07 等任务证据合并后汇总，组长完成跨章节校对 |

每个任务先交可复核的证据，再进入最终报告。普通任务 PR 只改自己的 Issue 证据文件及负责的材料；总证据矩阵、AI 汇总和跨任务章节由指定负责人集中整理。分析范围是软件行为、对象关系和测试边界；PPO 数学推导不在本次任务范围内。

## 目录

| 路径 | 放什么、什么时候改 |
|---|---|
| [`docs/project-baseline.md`](docs/project-baseline.md) | 固定上游 SB3 的 commit 和源码引用方式；全员先读。变更基线需要全组同步。 |
| [`tasks/`](tasks/README.md) | 任务链接和建议分工。认领、负责人和进度以 GitHub Issue 为准，不在这里反复改状态。 |
| [`docs/evidence/issues/`](docs/evidence/issues/README.md) | 每个 Issue 一份原始证据文件，记录 AI 假设、源码位置、测试、运行和复核结果；任务负责人主要编辑这里。 |
| [`docs/evidence/`](docs/evidence/) | `evidence-matrix.md` 与 `ai-log.md` 是全组汇总，由 T07 在任务证据合并后维护；普通任务 PR 不同时改它们。 |
| [`docs/report/`](docs/report/README.md) | 最终报告的八个章节草稿。各任务按章节职责写入；多人涉及同一章时，先交各自证据，再由该章负责人整合。 |
| [`scripts/`](scripts/README.md) | 能让复核人重新运行的验证脚本；文件名带 Issue 号，例如 `issue_05_env_checker_failures.py`。 |
| [`artifacts/diagrams/`](artifacts/diagrams/README.md) | 项目地图、类图、调用链图的可编辑源文件与导出图，按 `issue-<编号>/` 分目录。 |
| [`artifacts/runs/`](artifacts/runs/README.md) | 实验命令、版本信息、关键输出和必要截图，按 Issue 和实验日期分目录；这里存运行结果，不存大模型文件或依赖缓存。 |
| [`.github/`](.github/) | 新 Issue 和 PR 的填写模板，帮助记录任务目标、证据、验证与复核。 |
