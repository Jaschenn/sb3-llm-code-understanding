# T01 图表

- `architecture.mmd`：类继承关系，DDPG 继承 TD3。
- `module-collaboration.mmd`：训练调度、算法更新与公共预测入口。DQN 自己覆盖了 `predict()`；图中的公共预测路径适用于 PPO 等沿用基类实现的算法。
- 两份文档中的 Mermaid 块与对应源文件相同；修改时同步更新，再导出 PNG。

在协作仓库根目录重新导出（本次使用 Mermaid CLI 12.0.0、白色背景、2 倍分辨率）：

```bash
pnpm dlx @mermaid-js/mermaid-cli@12.0.0 -i artifacts/diagrams/issue-01/architecture.mmd -o artifacts/diagrams/issue-01/01architecture.png -c artifacts/diagrams/issue-01/mermaid-config.json -b white -s 2
pnpm dlx @mermaid-js/mermaid-cli@12.0.0 -i artifacts/diagrams/issue-01/module-collaboration.mmd -o artifacts/diagrams/issue-01/02module-collaboration.png -c artifacts/diagrams/issue-01/mermaid-config.json -b white -s 2
```

如果使用已安装的 Chrome，可通过 `-p <配置文件>` 指定 Puppeteer 配置，其中 `executablePath` 指向本机 Chrome 可执行文件。中文字体由操作系统提供，其他系统导出的字形可能略有不同。
