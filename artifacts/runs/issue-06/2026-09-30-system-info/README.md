# 环境信息

源码 commit：`7cfb4dd6055e74b5caa4ed4d6777492209946e26`。
随机种子：不适用，此脚本不调用随机操作。
实际命令（协作仓库根目录）：

```bash
UV_CACHE_DIR=/private/tmp/uv-cache-issue06 uv run --no-project python scripts/issue_06_system_info.py > artifacts/runs/issue-06/2026-09-30-system-info/output.json
```

目的：记录后续实验所用的运行版本和 SB3 实际导入位置。输出见 [output.json](output.json)：Python 3.12.14、PyTorch 2.14.0、Gymnasium 1.3.0、SB3 2.9.2a0。完整依赖见[共同环境记录](../pip-freeze.txt)。
