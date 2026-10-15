# T03 第二轮审慎检查

日期：2026-10-15。复核人：@Jaschenn。PR #17；上一轮提交为 a1e87b3。本目录保留此次脚本、基线检查模块、契约检查与原始输出，旧运行目录保持不变。

## 来源与命令

实际环境：macOS 27.0.1 / arm64、Python 3.12.14、PyTorch 2.14.0、Gymnasium 1.3.0、NumPy 2.5.3；完整来源信息由程序在输出首行记录。依赖解释器沿用前轮临时环境，但这次必须用 PYTHONPATH 导入固定 commit 的干净源码，不能直接拿已安装 wheel 的相同版本号作为来源证明。

在 PR 根目录运行。SB3_SOURCE 为按[基线说明](../../../../docs/project-baseline.md)准备的干净检出，本次为 /Users/jaschen/stable-baselines3；SB3_PYTHON 本次为 /private/tmp/sb3-pr-review/pr12-validation-venv/bin/python。

```bash
PYTHONPATH="$SB3_SOURCE" PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python "$SB3_PYTHON" python scripts/issue_03_predict_trace.py
PYTHONPATH="$SB3_SOURCE" PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python "$SB3_PYTHON" python -m pytest -q -s -p no:cacheprovider scripts/issue_03_contract_checks.py
PYTHONPATH="$SB3_SOURCE" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-project --python "$SB3_PYTHON" python -m pytest -q -p no:cacheprovider "$SB3_SOURCE/tests/test_predict.py::test_mixing_gym_vecenv_api" "$SB3_SOURCE/tests/test_train_eval_mode.py::test_predict_with_dropout_batch_norm"
uv run --no-project --python "$SB3_PYTHON" ruff check scripts/issue_03_baseline.py scripts/issue_03_predict_trace.py scripts/issue_03_contract_checks.py
uv run --no-project --python "$SB3_PYTHON" ruff format --check scripts/issue_03_baseline.py scripts/issue_03_predict_trace.py scripts/issue_03_contract_checks.py
```

实际隔离检出为 /private/tmp/sb3-pr17-review-fix；标准输出先保存至 /private/tmp/sb3-pr-review/pr17-audit-*.log 后复制到本目录。脚本快照与 scripts 中的本次版本保持一致，快照旁附有对应 baseline 模块，可单独执行；pytest 的入口仍建议使用 scripts 中的检查文件。

## 结果与证明范围

- 主实验正常退出，直接记录实际调用边界，见 [output.log](output.log)。输出首行必须为 SB3_BASELINE_VERIFIED，含实际导入路径与完整 commit。
- 本轮新增针对性检查 **9 passed**，见 [contract-checks.log](contract-checks.log)；所有检查定义保存在 [issue_03_contract_checks.py](issue_03_contract_checks.py)。
- 对应上游测试重新执行，**11 passed**，见 [upstream-tests.log](upstream-tests.log)。这些上游用例的通过不代替本轮逐结论的针对性检查。
- 不设置 PYTHONPATH、直接使用安装版环境时，主脚本在训练前拒绝，见 [rejected-installed-package.log](rejected-installed-package.log)。该错误退出是预期的反向验证，不是验证通过的实验输出；恢复固定源码导入后才可运行实验。
- Ruff 0.16.9 lint/格式检查通过。Mermaid CLI 12.0.0 实际渲染本 PR 对应图稿，语法通过；不将渲染成功视为源码语义正确的证明。

本轮契约检查包括：干净基线接受、非源码路径/错误 commit/脏源码拒绝（4 项），Dict 单条/批量（2 项），Box clip、受控 unscale、原生 Gym/VecEnv 返回契约（各 1 项）。

上游 11 passed 是 pytest 的用例计数：其中 SAC/TD3 在 CartPole、DQN 在 Pendulum 的 3 个参数组合由上游函数直接 return，并未执行预测断言。该计数不代表 11 条不同分支都被实际断言，更不代表全部算法/网络已覆盖。

profile 只记录明确列出的 Python 方法，且只针对当前单条 CartPole 输入。动作一致断言与调用路径检查是不同证据。Box unscale 的 _predict 输出被固定为 0.5，用来隔离后处理，不验证 gSDE 采样。

未覆盖的分支和证据限制见 [Issue #3 逐结论表](../../../../docs/evidence/issues/issue-03.md)。源码检查只确认启动时来源与清洁状态，不保证运行期间任意外部程序不修改文件。
