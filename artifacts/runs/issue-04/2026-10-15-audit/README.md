# T04 第二轮审慎检查

日期：2026-10-15。复核人：@Jaschenn。PR #18；上一轮提交为 d305824。本目录保留此次脚本、基线检查模块、契约检查与原始输出，旧运行目录保持不变。

## 来源与命令

实际环境：macOS 27.0.1 / arm64、Python 3.12.14、PyTorch 2.14.0、Gymnasium 1.3.0、NumPy 2.5.3；完整来源信息由程序在输出首行记录。依赖解释器沿用前轮临时环境，但这次必须用 PYTHONPATH 导入固定 commit 的干净源码，不能直接拿已安装 wheel 的相同版本号作为来源证明。

在 PR 根目录运行。SB3_SOURCE 为按[基线说明](../../../../docs/project-baseline.md)准备的干净检出，本次为 /Users/jaschen/stable-baselines3；SB3_PYTHON 本次为 /private/tmp/sb3-pr-review/pr12-validation-venv/bin/python。

```bash
PYTHONPATH="$SB3_SOURCE" PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python "$SB3_PYTHON" python scripts/issue_04_learn_trace.py
PYTHONPATH="$SB3_SOURCE" PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python "$SB3_PYTHON" python -m pytest -q -s -p no:cacheprovider scripts/issue_04_contract_checks.py
PYTHONPATH="$SB3_SOURCE" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-project --python "$SB3_PYTHON" python -m pytest -q -p no:cacheprovider "$SB3_SOURCE/tests/test_train_eval_mode.py::test_a2c_ppo_collect_rollouts_with_batch_norm" "$SB3_SOURCE/tests/test_run.py::test_advantage_normalization" "$SB3_SOURCE/tests/test_run.py::test_ppo"
uv run --no-project --python "$SB3_PYTHON" ruff check scripts/issue_04_baseline.py scripts/issue_04_learn_trace.py scripts/issue_04_contract_checks.py
uv run --no-project --python "$SB3_PYTHON" ruff format --check scripts/issue_04_baseline.py scripts/issue_04_learn_trace.py scripts/issue_04_contract_checks.py
```

实际隔离检出为 /private/tmp/sb3-pr18-review-fix；标准输出先保存至 /private/tmp/sb3-pr-review/pr18-audit-*.log 后复制到本目录。脚本快照与 scripts 中的本次版本保持一致，快照旁附有对应 baseline 模块，可单独执行；pytest 的入口仍建议使用 scripts 中的检查文件。

## 结果与证明范围

- 主实验正常退出，直接记录实际调用边界，见 [output.log](output.log)。输出首行必须为 SB3_BASELINE_VERIFIED，含实际导入路径与完整 commit。
- 本轮新增针对性检查 **20 passed**，见 [contract-checks.log](contract-checks.log)；所有检查定义保存在 [issue_04_contract_checks.py](issue_04_contract_checks.py)。
- 对应上游测试重新执行，**14 passed**，见 [upstream-tests.log](upstream-tests.log)。这些上游用例的通过不代替本轮逐结论的针对性检查。
- 不设置 PYTHONPATH、直接使用安装版环境时，主脚本在训练前拒绝，见 [rejected-installed-package.log](rejected-installed-package.log)。该错误退出是预期的反向验证，不是验证通过的实验输出；恢复固定源码导入后才可运行实验。
- Ruff 0.16.9 lint/格式检查通过。Mermaid CLI 12.0.0 实际渲染本 PR 对应图稿，语法通过；不将渲染成功视为源码语义正确的证明。

本轮契约检查包括：基线接受/拒绝（4 项）、标准采样 state_dict/模式与 returns→callback 顺序、buffer 展平与每次 permutation、初始化断言（2 项）、关闭归一化单样本对照、尾批 warning、归一化开关与单样本尾批（2 项）、KL 阈值/优化次数/epoch 计数（2 项）、callback False、续训和步数下界、截断与真实终止（2 项）、logger 统计与 TensorBoard 排除项、log_interval=None，共 20 项。

KL 使用固定 log_ratio=0.2，改变 target_kl，使 approx_kl 处于 target_kl 与 1.5*target_kl 之间或超过后者；对应 optimizer.step 为 6 次或 0 次，epoch 计数为 3 或 1。bootstrap 用 V=2、gamma=0.9，对比截断奖励 2.8 与真实终止奖励 1.0。这些是受控分支检查，不是策略质量或 PPO/GAE 数学正确性的证明。

新增 [check_disabled_training.py](check_disabled_training.py) 保留“只增加计数而不更新参数”的反向验证，预期被主脚本拒绝，见 [disabled-training.log](disabled-training.log)。

未覆盖的分支和证据限制见 [Issue #4 逐结论表](../../../../docs/evidence/issues/issue-04.md)。源码检查只确认启动时来源与清洁状态，不保证运行期间任意外部程序不修改文件。
