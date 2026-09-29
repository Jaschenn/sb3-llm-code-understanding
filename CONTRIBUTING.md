# 协作规则

## 任务认领

任务以 GitHub Issue 为唯一事实来源。认领者必须：

1. 在 Issue 右侧 **Assignees** 选择自己；
2. 留言：`认领。计划在 YYYY-MM-DD 前提交：<材料>。`；
3. 将 `status: todo` 标签改为 `status: claimed`，再开始建立分支。

同一 Issue 只设一名主负责人。提交前由另一位同学担任证据复核人。

## 提交与审阅

- 分支名：`task/<issue号>-<英文短名>`，例如 `task/7-env-checker-boundaries`。
- 一个 PR 只解决一个 Issue；PR 中必须使用 `Closes #编号`。
- 统一按[上游源码基线](docs/project-baseline.md)定位源码；引用源码时给出该 commit 下的文件与行号。
- 重要结论应提供源码、测试、运行、Issue/PR 历史中的两类独立证据。无法找到第二类证据时标为“待验证”，不伪造证据。
- 每个任务只修改自己的 `docs/evidence/issues/issue-<编号>.md` 和相关章节；AI 原始记录也写入该 Issue 文件。不要在任务 PR 中直接改汇总矩阵或汇总 AI 日志。
- 图放在 `artifacts/diagrams/issue-<编号>/`；运行材料放在 `artifacts/runs/issue-<编号>/<日期>-<实验名>/`。脚本以 `issue_<编号>_` 开头，避免重名。
- AI 产出的结论先作为待验证假设，记录原问题、模型、回答摘要和核验过程。
- 在 PR 中指定非提交者复核人；复核人检查代码位置、证据可复现性和报告表述。提交者不批准自己的 PR。
- 总证据矩阵与 AI 汇总由 T07 负责人在各任务材料合并后整理；最终报告由组长统一校对。

## Issue 标签状态

状态流：`status: todo → status: claimed → status: in progress → status: review → Issue closed`。

一个 Issue 同时只保留一个 `status:*` 标签。认领时设置 `claimed`，实际开始工作后设置 `in progress`，提交 PR 后设置 `review`。PR 合并并自动关闭 Issue 后移除状态标签；关闭状态就是完成状态。当前没有要求使用 GitHub Project。

## 减少合并冲突

- `tasks/README.md` 只保留任务链接与建议分工；实际负责人和进度以 Issue 为准，普通任务 PR 不更新该表。
- 报告正文按 `docs/report/01-*.md` 至 `08-*.md` 分文件。各任务先写自己的章节或 Issue 证据文件；跨章节修改先在 Issue 中说明并与该章节负责人协调。
- `docs/report/README.md`、`docs/evidence/evidence-matrix.md` 和 `docs/evidence/ai-log.md` 是汇总入口，待各任务 PR 合并后再集中更新。
- 创建 PR 前拉取最新 `main`，解决本分支冲突，再请复核人审阅最新版本。
