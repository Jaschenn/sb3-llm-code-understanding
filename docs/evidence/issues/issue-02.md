# Issue #2：核心对象与领域类图

负责人：@Jaschenn  
上游基线：`7cfb4dd6055e74b5caa4ed4d6777492209946e26`，见 [项目基线](../../project-baseline.md)。

## 对象词典

| 对象（与 T03/T04 统一名称） | 职责 | 生命周期 |
|---|---|---|
| `PPO` / `model` | 具体算法，定义 policy 别名、PPO 的 `train` 与参数检查 | 默认构造时 setup；继承预测入口；`learn` 可重复调用。 |
| `OnPolicyAlgorithm` | 创建 policy/buffer，组织采样与更新 | PPO 父类，不是被 PPO 另行持有的对象；每轮采样前 reset buffer。 |
| `BaseAlgorithm` | 环境包装、公共配置、callback 初始化、预测委托 | `env` 可为空；训练时必须有环境。 |
| `BaseModel → BasePolicy → ActorCriticPolicy` / `policy` | 特征处理、动作/value/log-probability 计算，policy 持有 optimizer | setup 中创建并移至 device；采样、更新均复用同一对象。 |
| `MlpPolicy` / `CnnPolicy` / `MultiInputPolicy` | PPO policy 类别名 | 分别是 ActorCriticPolicy、ActorCriticCnnPolicy、MultiInputActorCriticPolicy 的引用，不是额外继承层。 |
| `RolloutBuffer` / `rollout_buffer` | 保存本轮样本，计算 returns/advantages，提供训练 batch | 每轮 reset、add、计算与 get；下一轮覆盖数据，但对象复用。Dict 观测默认用 DictRolloutBuffer。 |
| `VecEnv` / `env` | SB3 的批量环境接口 | 普通 Gymnasium Env 默认经 Monitor 包装进入 DummyVecEnv；VecEnv 不继承 Gymnasium Env。 |
| `DummyVecEnv` / `VecEnvWrapper` | 前者保存环境列表并逐个执行；后者持有并转发一个 VecEnv | DummyVecEnv 结束时保存 terminal observation 后自动 reset；wrapper 保存 `venv`。 |
| Gymnasium `Env` | 单环境 reset/step 接口 | 由用户或环境工厂创建，再由 VecEnv 调用。 |
| `BaseCallback` / `callback` | 生命周期钩子与提前终止控制 | learn 初始化时保存 model 引用；作为 learn/collect_rollouts 的局部参数传递。 |

## 可检验结论

| 编号 | 结论 | 状态 | 证据 |
|---|---|---|---|
| C-02-01 | PPO 继承 OnPolicyAlgorithm 与 BaseAlgorithm；MlpPolicy 是类别名 | 已支持 | R01–R04、别名源码、运行断言 |
| C-02-02 | 默认 setup 保存 policy、buffer、VecEnv；普通 Env 在 DummyVecEnv 中；VecEnv 不继承 gym.Env | 已支持 | R09、R11–R14、运行断言 |
| C-02-03 | predict 委托 policy，且不写 rollout buffer；训练跨轮复用 policy/buffer | 已支持 | R17–R23、两轮训练断言 |
| C-02-04 | callback 可回指 model；on_step 为 false 时在 buffer.add 前停止 | 已支持 | R16、R21、提前终止断言 |
| C-02-05 | Dict 观测默认选择 DictRolloutBuffer 与 MultiInputActorCriticPolicy | 已支持 | R06、R08、R12、运行断言 |
| C-02-06 | CNN 与 VecEnvWrapper 关系 | 待验证 | R05、R10、R15：已核源码，尚无专项运行 |

## 类图箭头与源码证据

[Mermaid 源文件](../../../artifacts/diagrams/issue-02/domain-objects.mmd)。图中的引用边只表示保存引用，不表示独占所有权；R 编号与下表一一对应。链接均固定在基线 commit。

| 边 | 关系 | 源码 |
|---|---|---|
| R01 | 继承 PPO → OnPolicyAlgorithm | [ppo.py:L18](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/ppo.py#L18) |
| R02 | 继承 OnPolicyAlgorithm → BaseAlgorithm | [on_policy_algorithm.py:L21](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L21) |
| R03–R06 | 继承 ActorCriticPolicy → BasePolicy → BaseModel；CNN/MultiInput → ActorCriticPolicy | [policies.py:L280](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L280)、[L416](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L416)、[L766](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L766)、[L839](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py#L839) |
| R07–R08 | 继承 RolloutBuffer → BaseBuffer；DictRolloutBuffer → RolloutBuffer | [buffers.py:L343](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/buffers.py#L343)、[L697](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/buffers.py#L697) |
| R09–R10 | 继承 DummyVecEnv/VecEnvWrapper → VecEnv | [dummy_vec_env.py:L15](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/vec_env/dummy_vec_env.py#L15)、[base_vec_env.py:L360](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/vec_env/base_vec_env.py#L360) |
| R11–R12 | 持有 algorithm → policy/buffer | [on_policy_algorithm.py:L119-L138](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L119-L138) |
| R13–R15 | 持有 algorithm → env，DummyVecEnv → envs，wrapper → venv | [base_class.py:L164-L175](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L164-L175)、[dummy_vec_env.py:L31-L42](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/vec_env/dummy_vec_env.py#L31-L42)、[base_vec_env.py:L369-L385](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/vec_env/base_vec_env.py#L369-L385) |
| R16 | 持有 callback → model | [callbacks.py:L68-L74](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/callbacks.py#L68-L74) |
| R17 | 调用 BaseAlgorithm.predict → policy.predict | [base_class.py:L537-L557](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L537-L557) |
| R18–R21 | 调用 collect_rollouts → policy/env/buffer/callback | [on_policy_algorithm.py:L198-L225](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L198-L225)、[L247-L266](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py#L247-L266) |
| R22–R23 | 调用 PPO.train → buffer.get/policy | [ppo.py:L204-L213](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/ppo.py#L204-L213)、[L274-L278](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/ppo.py#L274-L278) |
| R24 | 调用 DummyVecEnv → Env.step/reset | [dummy_vec_env.py:L56-L80](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/vec_env/dummy_vec_env.py#L56-L80) |

别名定义见 [ppo/policies.py:L1-L7](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/policies.py#L1-L7)，普通环境包装见 [base_class.py:L203-L226](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L203-L226)。

## 运行证据

[验证脚本](../../../scripts/issue_02_object_probe.py) 对指定固定 checkout 执行：检查 commit/导入路径，构造 CartPole PPO，断言 MRO、别名、环境类型、predict 不写 buffer、两轮训练对象复用、callback 回指与提前停止，并覆盖 Dict 环境。命令与结果见 [运行记录](../../../artifacts/runs/issue-02/2026-10-04-object-probe/README.md) 和 [output.json](../../../artifacts/runs/issue-02/2026-10-04-object-probe/output.json)：退出码 0，`result` 为 `PASS`。该实验不衡量学习效果或数学收敛。

## AI 使用记录

| 编号 | 模型与提问 | 回答摘要 | 验证动作 | 修正后表述 |
|---|---|---|---|---|
| AI-02-01 | GPT-6（Codex）：“帮我完成这个 issue：https://github.com/Jaschenn/sb3-llm-code-understanding/issues/2” | 识别主线的继承、持有和调用 | 固定 commit 源码定位、运行对象实验、导出类图 | MlpPolicy 是别名；VecEnv 独立于 gym.Env；callback 在 add 前可中止；CNN/wrapper 标为待验证。 |
