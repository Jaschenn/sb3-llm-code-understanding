# SB3：使用大语言模型辅助代码理解

《软件测试和分析》课程小组实训协作仓库。我们以 Stable-Baselines3（SB3）为案例，研究大语言模型如何辅助理解陌生项目，并用源码、测试、运行和维护历史交叉验证结论。

## 研究范围

主问题：SB3 如何将一个符合 Gymnasium 接口的环境，组织为可验证的预测和训练软件流程？

本项目分析对象协作、控制流、接口约束与失败路径；不证明 PPO 的数学收敛性或推导其损失函数。

## 协作入口

1. 从 [任务看板](tasks/README.md) 选择未认领任务。
2. 在 GitHub 创建/打开对应 Issue，将自己设为 **Assignee**，并评论“认领”。
3. 从 `main` 建立分支：`task/<issue号>-<英文短名>`。
4. 提交材料和证据，发起 PR；PR 描述必须包含 `Closes #<issue号>`。
5. 由非提交者复核证据后合并。

详见 [协作规则](CONTRIBUTING.md) 和 [报告结构](docs/report/README.md)。

## 目录

- `docs/report/`：最终报告草稿与图表说明。
- `docs/evidence/`：证据矩阵、AI 使用日志。
- `tasks/`：任务清单、分工和验收标准。
- `scripts/`：可重复执行的验证脚本及说明。
- `artifacts/`：图、运行日志和截图等可引用材料。
- `.github/`：Issue 与 PR 模板。
