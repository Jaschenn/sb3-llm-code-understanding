# Issue #01：项目地图与推荐阅读顺序

负责人：@成员03；复核人：@成员01。上游 commit：`7cfb4dd6055e74b5caa4ed4d6777492209946e26`，见[基线说明](../../project-baseline.md)。

本文件保留结论依据、AI 纠错和运行记录；项目地图与阅读路线见[报告第2章](../../report/02-project-map.md)，图源及导出命令见[图表目录](../../../artifacts/diagrams/issue-01/README.md)。

## 结论与证据

| 编号 | 结论 | 状态与证据 |
|---|---|---|
| C-01-01 | 包内有 64 个 Python 文件、15,693 行代码 | 已支持：固定 commit 下执行下方统计命令 |
| C-01-02 | `BaseAlgorithm` 提供公共骨架，中间层实现训练循环，具体算法实现更新；DDPG 继承 TD3 | 已支持：[BaseAlgorithm.learn](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L512)、[OnPolicyAlgorithm.learn](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L300)、[OffPolicyAlgorithm.learn](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/off_policy_algorithm.py#L324)、[DDPG 类定义](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ddpg/ddpg.py#L14)；[开发指南](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/docs/guide/developer.md#L20)说明采集与训练的分工 |
| C-01-03 | PPO 公共预测入口转交 Policy；观察转换要求正确处理 Gymnasium reset 返回值 | 已支持：[BaseAlgorithm.predict](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L537)、[BasePolicy.predict](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L331)；下方预测练习运行成功，相关测试为[test_predict](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/tests/test_predict.py#L62) |
| C-01-04 | 训练 Buffer 的创建和交互采集位于 OnPolicy/OffPolicy，而非 BaseAlgorithm | 已支持：[OnPolicy._setup_model](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L115)、[collect_rollouts](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L162)；[OffPolicy._setup_model](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/off_policy_algorithm.py#L177)、[collect_rollouts](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/off_policy_algorithm.py#L526)；开发指南与下方训练运行结果交叉核对 |
| C-01-05 | TensorBoard 事件日志与 EvalCallback 的 NPZ 评估记录是不同产物 | 已支持：[configure_logger](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/utils.py#L265)、[EvalCallback](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/callbacks.py#L342)；下方脚本验证两类产物均生成 |

在检出的上游仓库根目录复现规模统计：

```bash
rg --files stable_baselines3 -g '*.py' | wc -l
rg --files stable_baselines3 -g '*.py' | xargs wc -l | tail -1
```

实际输出：`64`、`15693 total`。仓库目录职责根据 README、开发指南、测试、CI 和辅助脚本核对，代表文件的固定链接集中在报告 2.1 节；阅读顺序是本任务建议。

## AI 使用记录

原始记录未注明模型名称，以下保留当时的提问、回答摘要和核验结果。

| 提问 | AI回答 | 验证方式 | 结果 |
|------|--------|---------|------|
| "SB3有多少代码？" | 约64个文件，15000+行 | 用find和wc命令统计 | ✅ 准确：64文件，15693行 |
| "common目录有哪些模块？" | 列举了base_class、policies等 | ls命令查看目录 | ✅ 核心模块都在 |
| "继承关系是怎样的？" | 三层：Base→OnPolicy/OffPolicy→算法，6个算法都直接继承中间层 | 最初只读了PPO/OnPolicy的类定义；复核时搜索所有 `class XXX(` | ⚠️ 部分错误：DDPG实际继承TD3（`ddpg/ddpg.py:L14`），已修正 |
| "learn()和predict()在哪？" | 都在BaseAlgorithm中定义，learn()包含训练循环 | 复核时按行号核对base_class.py | ⚠️ 部分错误：BaseAlgorithm.learn()（L512）是抽象方法，训练循环在OnPolicy（L300）/OffPolicy（L324）中；predict()在L537，已修正 |

**纠错案例**：最初AI给出的继承关系和行号"看起来合理"，我只抽查了PPO这一条链就标成了"正确"。复核人对照基线源码后指出 DDPG 继承 TD3、多处行号对不上。教训是：AI 给出的结构性结论需要逐个核对，不能只抽查一个例子；行号必须按固定 commit 实际检索得出。

---

## 复核记录

**复核人**：成员01（对照 SB3 基线 7cfb4dd 源码）

[GitHub review 记录](https://github.com/Jaschenn/sb3-llm-code-understanding/pull/16#pullrequestreview-5467354756)。

| # | 复核意见 | 核实结果 | 修改 |
|---|---------|---------|------|
| 1 | DDPG 应继承 TD3，而非直接继承 OffPolicyAlgorithm | 属实（`ddpg/ddpg.py:L14`） | `architecture.mmd`/PNG 及报告已改为 OffPolicyAlgorithm → TD3 → DDPG |
| 2 | base_class.py L67-L202 不含 learn/predict | 属实（learn L512，predict L537） | 补充各方法行号和固定链接 |
| 3 | policies.py L39 是 BaseModel | 属实（ActorCriticPolicy 在 L416） | 02 文档阅读路径2已修正 |
| 4 | buffers.py L200-L350 主要是 ReplayBuffer | 属实（ReplayBuffer L158-L342，RolloutBuffer L343 起） | 02 文档阅读路径3已修正 |
| 5 | `env.reset()` 返回 `(obs, info)` | 属实 | 02 文档两处示例改为 `obs, info = env.reset()` |
| 6 | `preprocess_obs` 未 import 且要求 Tensor | 属实（`preprocessing.py:L115` 断言输入为 `th.Tensor`） | 练习代码补全 import，先用 `policy.obs_to_tensor()` 转换 |

**自查补充**（复核意见之外，同时修正）：
- `BaseAlgorithm.learn()` 是抽象方法，原文"base_class 中的训练循环"表述有误
- `ppo/policies.py` 并非"通常为空"，而是定义了 MlpPolicy 等别名
- `collect_rollouts()` 范围应为 L162-L268（原写 L162-L250）
- 中间层的 `train()` 只是接口，实现位于各算法

## 本轮修正

修正训练调度归属和 TensorBoard 配置，补齐仓库目录及图源。随后合并重复图表、入口示例和职责说明，删除无依据的阅读耗时与学习效果承诺；图表仅在 `.mmd` 中维护，文档引用 PNG。预测路线也区分了 `BaseModel.obs_to_tensor()` 与 `BasePolicy.predict()`，并定位 PPO 的 `_predict()` / `get_distribution()`。

## 示例运行验证

验证脚本：[scripts/issue_01_validate_examples.py](../../../scripts/issue_01_validate_examples.py)。脚本提取报告中的预测练习和 TensorBoard 示例并原样执行，在临时目录检查输出，退出后清理运行材料。

按[基线准备说明](../../project-baseline.md)检出并安装上游源码、激活虚拟环境后，在本协作仓库根目录运行（将路径替换为自己的上游目录）：

```bash
PYTHONPATH=/path/to/stable-baselines3 uv run --no-project --active --with tensorboard python scripts/issue_01_validate_examples.py
```

本次验证：macOS 27.0.1 / arm64，Python 3.13.5，PyTorch 2.14.0，Gymnasium 1.3.0；源码为固定 commit `7cfb4dd6055e74b5caa4ed4d6777492209946e26`，包版本为 `2.9.2a0`。示例未设随机种子，只验证接口、形状与日志产物，不以动作取值或回报衡量训练效果。

结果：原始观察形状 `(4,)`，预处理后形状 `(1, 4)`，预测成功；生成一个 TensorBoard 事件文件，包含 `eval/mean_reward` 和 `rollout/ep_rew_mean`，同时生成 `best_model.zip` 和 `evaluations.npz`。独立评估环境使用 Monitor 包装。

另外核对了文档引用的图表源文件及导出图、本地文件链接、固定源码链接的文件和行号；两张 PNG 已重新渲染并检查可读性。图表导出命令见[图表说明](../../../artifacts/diagrams/issue-01/README.md)。
