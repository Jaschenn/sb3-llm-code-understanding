# 验证脚本

每个脚本应：

1. 在开头说明目的、依赖、执行命令和预期结果；
2. 固定或记录随机种子；
3. 将可引用输出保存至 `artifacts/runs/issue-<编号>/<日期>-<实验名>/`，并记录实际运行命令和版本；
4. 在 `docs/evidence/issues/issue-<编号>.md` 引用输出和结论。

建议的首批脚本：

- `issue_03_predict_chain.py`：CartPole 上的 `reset → PPO.predict → step`；
- `issue_05_env_checker_failures.py`：构造不合法环境并运行 `check_env`；
- `issue_06_short_training.py`：短 PPO 训练，记录采样/训练阶段；
- `issue_06_system_info.py`：记录 Python、PyTorch、Gymnasium 与 SB3 版本。
