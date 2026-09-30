# 最小预测与接口边界

源码 commit：`7cfb4dd6055e74b5caa4ed4d6777492209946e26`。
运行版本：Python 3.12.14、PyTorch 2.14.0、Gymnasium 1.3.0、SB3 2.9.2a0；详见[环境信息](../2026-09-30-system-info/output.json)。
随机种子：42，用于原生环境 reset、模型初始化及 VecEnv reset。
实际命令（协作仓库根目录）：

```bash
UV_CACHE_DIR=/private/tmp/uv-cache-issue06 uv run --no-project python scripts/issue_06_predict.py > artifacts/runs/issue-06/2026-09-30-predict/output.json
```

目的：在 CartPole 上执行 `reset → predict → step`，同时观察 Gymnasium 原生环境与 SB3 VecEnv 的返回结构，并触发一次 reset 元组误传的错误。
关键输出见 [output.json](output.json)：`check_env` 通过；原生 reset 的 observation 形状为 `[4]` 且 info 为 dict，step 返回下一 observation、reward、terminated、truncated、info；预测动作 `1` 在动作空间内。`DummyVecEnv` 的 reset observation 形状为 `[1,4]`，step 的 rewards/dones 形状均为 `[1]`；误传整个 reset 元组产生 `ValueError`，错误信息指出 Gym API 与 VecEnv API 混用。未进行训练，动作来自初始化策略，动作值和奖励值不用于评估策略质量。
