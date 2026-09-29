# 协作规则

## 任务认领

任务以 GitHub Issue 为唯一事实来源。认领者必须：

1. 在 Issue 右侧 **Assignees** 选择自己；
2. 留言：`认领。计划在 YYYY-MM-DD 前提交：<材料>。`；
3. 将 Project 状态改为 `Claimed`，再开始建立分支。

同一 Issue 只设一名主负责人。提交前由另一位同学担任证据复核人。

## 提交与审阅

- 分支名：`task/<issue号>-<英文短名>`，例如 `task/7-env-checker-boundaries`。
- 一个 PR 只解决一个 Issue；PR 中必须使用 `Closes #编号`。
- 结论必须有源码、测试、运行、Issue/PR 历史中的至少两类证据，核心结论优先使用源码加运行或测试。
- 运行材料放在 `artifacts/runs/`，图放在 `artifacts/diagrams/`，并在报告中注明文件名与执行命令。
- AI 产出的结论不得直接作为事实；先记录在 `docs/evidence/ai-log.md`，再验证。
- 不自审、不自合并；复核人检查代码位置、证据可复现性和报告表述。

## GitHub Project 看板

状态流：`Todo → Claimed → In Progress → In Review → Done`。

Issue 创建时为 `Todo`；负责人认领后改为 `Claimed`；开始产出后为 `In Progress`；提交 PR 后为 `In Review`；PR 合并后为 `Done`。
