# T08 图表

- `maintenance-chain.mmd`：一组已关闭 Issue 的维护链条对照。上支是「有代码变更」（Issue #2172 → PR #2174 → merge commit `7883ed4`），下支是「判定不改」（Issue #1950 → `not_planned` 关闭）。每个节点的依据、链接与行号见 `docs/evidence/issues/issue-08.md`。
- Mermaid 源文件是图表的编辑入口；修改后重新导出 PNG，报告引用导出图，证据文件引用此目录。

在协作仓库根目录重新导出（本次使用 Mermaid CLI 12.0.0、白色背景、2 倍分辨率）：

```bash
npx --yes @mermaid-js/mermaid-cli@12.0.0 \
  -i artifacts/diagrams/issue-08/maintenance-chain.mmd \
  -o artifacts/diagrams/issue-08/maintenance-chain.png \
  -c artifacts/diagrams/issue-08/mermaid-config.json \
  -b white -s 2
```

若本机 Puppeteer 未自带浏览器，可加 `-p artifacts/diagrams/issue-08/puppeteer-config.json` 指向已安装的 Chrome（该文件中的 `executablePath` 为 Windows 默认安装路径，其他平台需自行修改；此文件仅为本机导出提供便利，不属于图表的可编辑源）。中文字体由操作系统提供，其他系统导出的字形可能略有不同。
