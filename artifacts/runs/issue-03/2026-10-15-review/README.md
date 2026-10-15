# T03 复核与修正后的运行

2026-10-15，复核人：@Jaschenn。原提交 `b91914c5ef8e5a778d4745b0eb51052a5fd09912` 的运行材料保留于相邻的 2026-10-09 目录；本目录保存修正版脚本快照与完整输出。

环境：macOS 27.0.1 / arm64，Python 3.12.14，PyTorch 2.14.0，Gymnasium 1.3.0，NumPy 2.5.3，SB3 2.9.2a0。SB3 由干净的上游检出 `7cfb4dd6055e74b5caa4ed4d6777492209946e26` 构建 wheel；安装方式同[基线准备](../../../../docs/project-baseline.md)，本次用 wheel 替代 editable。CPU 单线程，seed=0，512 步短训练。

## 命令

在本 PR 根目录执行，`SB3_PYTHON` 指向已安装固定基线 SB3 与依赖的解释器（本次为 `/private/tmp/sb3-pr-review/pr12-validation-venv/bin/python`），`SB3_SOURCE` 为对应上游检出目录（本次 `/Users/jaschen/stable-baselines3`）：

```bash
uv run --no-project --python "$SB3_PYTHON" python scripts/issue_03_predict_trace.py
uv run --no-project --python "$SB3_PYTHON" ruff check scripts/issue_03_predict_trace.py
uv run --no-project --python "$SB3_PYTHON" ruff format --check scripts/issue_03_predict_trace.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-project --python "$SB3_PYTHON" python -m pytest -q -p no:cacheprovider "$SB3_SOURCE/tests/test_predict.py::test_mixing_gym_vecenv_api" "$SB3_SOURCE/tests/test_train_eval_mode.py::test_predict_with_dropout_batch_norm"
```

实际运行在临时隔离检出 `/private/tmp/sb3-pr17-review-fix` 完成；标准输出分别保存至 `/private/tmp/sb3-pr-review/pr17-fixed-output.log`、`pr17-upstream-tests.log` 后复制到本目录。

## 结果

- [修正版脚本](issue_03_predict_trace.py)执行成功：[output.log](output.log)。
- 实际拒绝 `(obs, info)`，并另行验证 Gymnasium 五项 step 返回值用四变量解包时在用户代码处失败。
- 手动展开先切 eval；Categorical argmax、手动动作与 `model.predict` 结果断言一致，动作属于环境动作空间。不读取缓存模型，避免上次运行改变本次输入。
- 上游相关测试 **11 passed**，见 [upstream-tests.log](upstream-tests.log)。Ruff 0.16.9 lint 与格式检查通过。
- Mermaid CLI 12.0.0 已将预测图渲染为临时 SVG，语法通过；本次通过 `-p` 指定本机 Chrome 的 Puppeteer 配置。命令：`pnpm dlx @mermaid-js/mermaid-cli@12.0.0 -i artifacts/diagrams/issue-03/predict-call-chain.mmd -o /private/tmp/sb3-pr-review/pr17-diagram.svg -p /private/tmp/sb3-pr-review/puppeteer.json -b white`。

手动展开结果一致是当前 CartPole 配置的运行证据；每层调用顺序还须结合固定源码判断，不能仅凭最终动作相等证明全部内部路径。训练图属于 PR #18，本 PR 的报告已标记该依赖。
