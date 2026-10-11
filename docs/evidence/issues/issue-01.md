# Issue #01：项目地图与推荐阅读顺序

负责人：@成员03  
复核人：@成员01  
上游基线：见 `docs/project-baseline.md`

## 任务目标

为 Stable-Baselines3 项目梳理清晰的模块职责说明和代码阅读路径，帮助团队成员快速理解项目结构。

## 1. 项目基本情况

**统计数据**：
```bash
# 文件数量
find stable_baselines3/stable_baselines3 -type f -name "*.py" | wc -l
# 结果：64个Python文件

# 代码行数
find stable_baselines3/stable_baselines3 -type f -name "*.py" -exec wc -l {} + | tail -1
# 结果：15,693行代码
```

**目录结构**：
```
stable_baselines3/
├── __init__.py           # 入口文件
├── common/               # 公共模块（共享代码）
├── ppo/                  # PPO算法
├── a2c/                  # A2C算法
├── dqn/                  # DQN算法
├── ddpg/                 # DDPG算法
├── sac/                  # SAC算法
└── td3/                  # TD3算法
```

---

## 2. 项目地图

### 1.1 架构图

展示SB3的类继承关系：

```mermaid
graph TD
    A[BaseAlgorithm<br/>基类] --> B[OnPolicyAlgorithm<br/>on-policy基类]
    A --> C[OffPolicyAlgorithm<br/>off-policy基类]
    
    B --> D[PPO]
    B --> E[A2C]
    
    C --> F[DQN]
    C --> H[SAC]
    C --> I[TD3]
    I --> G[DDPG]
    
    style A fill:#e1f5ff,stroke:#333,stroke-width:2px
    style B fill:#fff4e6,stroke:#333,stroke-width:2px
    style C fill:#fff4e6,stroke:#333,stroke-width:2px
    style D fill:#e8f5e9,stroke:#333,stroke-width:2px
    style E fill:#e8f5e9,stroke:#333,stroke-width:2px
    style F fill:#f3e5f5,stroke:#333,stroke-width:2px
    style G fill:#f3e5f5,stroke:#333,stroke-width:2px
    style H fill:#f3e5f5,stroke:#333,stroke-width:2px
    style I fill:#f3e5f5,stroke:#333,stroke-width:2px
```

**图说明**：
- 🔵 **蓝色（BaseAlgorithm）**：顶层基类，定义所有算法的统一接口（learn、predict、save、load）
- 🟡 **黄色（中间层）**：按数据使用方式分为on-policy和off-policy两类
- 🟢 **绿色（PPO、A2C）**：on-policy算法，训练后丢弃旧数据
- 🟣 **紫色（DQN、DDPG、SAC、TD3）**：off-policy算法，可重用历史数据

**解释**：SB3采用三层继承设计，所有算法共享统一的API，用户使用任何算法都是同样的方式。唯一的例外是DDPG：它继承的是TD3（`ddpg/ddpg.py:L14`，`class DDPG(TD3)`），源码注释说明SB3把DDPG当作TD3的特例，所以DDPG比其他算法多一层。

---

### 1.2 模块协作地图

展示各模块在训练和预测时的协作关系：

```mermaid
graph TD
    A[用户代码]
    B[具体算法<br/>PPO/DQN/SAC]
    C[BaseAlgorithm<br/>统一接口]
    
    D[Policy<br/>策略网络]
    E[Buffer<br/>数据缓冲]
    F[VecEnv<br/>环境包装]
    G[Callback]
    
    H[Gymnasium]
    I[PyTorch]
    
    A --> B
    B --> C
    C --> D
    C --> E
    C --> F
    C -.-> G
    
    F --> H
    D --> I
    E --> I
    
    style A fill:#e8f5e9,stroke:#333,stroke-width:2px
    style B fill:#fff4e6,stroke:#333,stroke-width:2px
    style C fill:#e1f5ff,stroke:#333,stroke-width:2px
    style D fill:#f3e5f5,stroke:#333,stroke-width:2px
    style E fill:#f3e5f5,stroke:#333,stroke-width:2px
    style F fill:#f3e5f5,stroke:#333,stroke-width:2px
    style G fill:#fff9c4,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    style H fill:#f5f5f5,stroke:#333,stroke-width:2px
    style I fill:#f5f5f5,stroke:#333,stroke-width:2px
```

**图说明**：
- 🟢 **用户代码**：调用SB3的API（如`model.learn()`）
- 🟡 **具体算法**：用户选择的算法（PPO、DQN等）
- 🔵 **BaseAlgorithm**：协调各模块的核心控制器
- 🟣 **核心模块**：Policy负责决策，Buffer存储数据，VecEnv管理环境
- 🟨 **辅助模块**：Callback监控训练过程（虚线表示可选）
- ⚪ **外部依赖**：Gymnasium提供环境接口，PyTorch实现神经网络

**解释**：用户的简单调用，经过BaseAlgorithm协调，被分发到各个专业模块完成复杂的强化学习训练。

---

## 2. 项目基本情况

**统计数据**：
```bash
# 文件数量
find stable_baselines3/stable_baselines3 -type f -name "*.py" | wc -l
# 结果：64个Python文件

# 代码行数
find stable_baselines3/stable_baselines3 -type f -name "*.py" -exec wc -l {} + | tail -1
# 结果：15,693行代码
```

**目录结构**：
```
stable_baselines3/
├── __init__.py           # 入口文件
├── common/               # 公共模块（共享代码）
├── ppo/                  # PPO算法
├── a2c/                  # A2C算法
├── dqn/                  # DQN算法
├── ddpg/                 # DDPG算法
├── sac/                  # SAC算法
└── td3/                  # TD3算法
```

---

## 3. 核心模块职责

### 3.1 顶层入口（`__init__.py`）

**固定链接**：https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/__init__.py

**职责**（来自源码）：
- 导出6个算法类：PPO, A2C, DQN, DDPG, SAC, TD3
- 导出工具函数：get_system_info()

**解释**：这是用户使用的入口，`from stable_baselines3 import PPO` 就是从这里导入的。

---

### 3.2 基础层（`common/base_class.py`）

**固定链接**：https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py

**职责**（来自源码注释）：
- 定义 `BaseAlgorithm` 抽象基类
- 声明 `learn()` 训练方法（抽象方法，由中间层实现）
- 提供 `predict()` 预测方法
- 提供 `save()` 和 `load()` 保存/加载方法

**关键代码位置**：

| 内容 | 行号 | 固定链接 |
|------|------|---------|
| 类定义和 `__init__` | L67-L202 | [L67](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L67) |
| `learn()`（`@abstractmethod`） | L511-L535 | [L512](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L512) |
| `predict()` | L537-L557 | [L537](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L537) |
| `load()` | L643 | [L643](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L643) |
| `save()` | L819 | [L819](https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/base_class.py#L819) |

```python
class BaseAlgorithm(ABC):          # L67
    """The base of RL algorithms"""

    @abstractmethod
    def learn(self, total_timesteps: int, ...) -> SelfBaseAlgorithm:   # L512
        """Return a trained model."""

    def predict(self, observation, ...) -> tuple[np.ndarray, ...]:     # L537
        return self.policy.predict(observation, state, episode_start, deterministic)
```

**解释**：所有算法都继承这个类，就像一个模板，定义了算法必须具备的功能。`learn()` 在这里只规定参数，真正的训练循环在中间层；`predict()` 则直接把工作交给策略网络 `policy.predict()`。

---

### 3.3 中间层（`common/on_policy_algorithm.py` 和 `off_policy_algorithm.py`）

#### 3.3.1 OnPolicyAlgorithm

**固定链接**：https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/on_policy_algorithm.py

**职责**（来自源码注释）：
- On-policy算法的基类（A2C、PPO使用）
- 提供 `collect_rollouts()` 方法收集经验（L162-L268）
- 实现 on-policy 的 `learn()` 训练循环（L300）
- 使用 `RolloutBuffer` 存储数据

**解释**：On-policy算法每次更新后要丢弃旧数据，所以单独封装了收集数据的逻辑。这里的 `train()`（L270）只是抛出 `NotImplementedError` 的空壳，由PPO、A2C各自实现。

#### 3.3.2 OffPolicyAlgorithm

**固定链接**：https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/off_policy_algorithm.py

**职责**（来自源码注释）：
- Off-policy算法的基类（DQN、SAC、TD3直接继承；DDPG经由TD3间接继承）
- 实现 off-policy 的 `learn()` 训练循环（L324）
- 声明 `train()` 接口（L372，由具体算法实现）
- 使用 `ReplayBuffer` 存储历史数据

**解释**：Off-policy算法可以重复使用旧数据，所以用ReplayBuffer保存历史经验。

---

### 3.4 算法实现层（`ppo/`, `a2c/`, `dqn/` 等）

以PPO为例：

**固定链接**：https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/ppo/ppo.py

**目录结构**：
```
ppo/
├── __init__.py
├── ppo.py           # PPO算法实现
└── policies.py      # 定义 MlpPolicy/CnnPolicy/MultiInputPolicy 别名
```

**职责**（来自源码注释）：
- 实现PPO算法（Proximal Policy Optimization）
- 定义超参数：clip_range, n_epochs, batch_size等
- 实现 `train()` 方法计算PPO损失（L184）

**解释**：每个算法一个文件夹，主要实现train()方法，定义自己的损失函数。`ppo/policies.py` 没有新写网络，只是把 `common/policies.py` 里的 `ActorCriticPolicy` 等类起了 `MlpPolicy` 这样的别名。

---

### 3.5 公共工具模块（`common/` 目录）

| 文件 | 职责（来自文档/代码） | 理解 |
|------|---------------------|------|
| `policies.py` | 策略网络定义（Actor-Critic等） | 神经网络的结构 |
| `buffers.py` | 经验缓冲区（RolloutBuffer、ReplayBuffer） | 存储交互数据 |
| `env_checker.py` | 环境检查工具 | 验证环境是否符合Gymnasium接口 |
| `callbacks.py` | 训练回调机制 | 监控训练过程、保存模型等 |
| `logger.py` | 日志记录 | 输出训练指标到TensorBoard |
| `vec_env/` | 向量化环境 | 同时运行多个环境加速训练 |
| `preprocessing.py` | 数据预处理 | 图像归一化等 |
| `distributions.py` | 动作分布 | 高斯分布、分类分布等 |
| `torch_layers.py` | PyTorch网络层 | MLP、CNN等网络结构 |
| `utils.py` | 工具函数 | 各种辅助函数 |

**固定链接**（示例）：
- policies.py: https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/policies.py
- buffers.py: https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/stable_baselines3/common/buffers.py

---

## 4. 继承关系总结

```
BaseAlgorithm（基类）
    ├── OnPolicyAlgorithm（on-policy算法）
    │   ├── PPO
    │   └── A2C
    └── OffPolicyAlgorithm（off-policy算法）
        ├── DQN
        ├── SAC
        └── TD3
            └── DDPG（继承TD3）
```

**解释**：
- **第一层**：BaseAlgorithm定义所有算法共同的方法
- **第二层**：区分on-policy和off-policy两种类型
- **第三层**：具体算法实现各自的训练逻辑
- **特例**：DDPG继承TD3（`ddpg/ddpg.py:L14`），比其他算法多一层

---

## 5. AI使用记录

| 提问 | AI回答 | 验证方式 | 结果 |
|------|--------|---------|------|
| "SB3有多少代码？" | 约64个文件，15000+行 | 用find和wc命令统计 | ✅ 准确：64文件，15693行 |
| "common目录有哪些模块？" | 列举了base_class、policies等 | ls命令查看目录 | ✅ 核心模块都在 |
| "继承关系是怎样的？" | 三层：Base→OnPolicy/OffPolicy→算法，6个算法都直接继承中间层 | 最初只读了PPO/OnPolicy的类定义；复核时搜索所有 `class XXX(` | ⚠️ 部分错误：DDPG实际继承TD3（`ddpg/ddpg.py:L14`），已修正 |
| "learn()和predict()在哪？" | 都在BaseAlgorithm中定义，learn()包含训练循环 | 复核时按行号核对base_class.py | ⚠️ 部分错误：BaseAlgorithm.learn()（L512）是抽象方法，训练循环在OnPolicy（L300）/OffPolicy（L324）中；predict()在L537，已修正 |

**纠错案例**：最初AI给出的继承关系和行号"看起来合理"，我只抽查了PPO这一条链就标成了"正确"。复核人对照基线源码后指出 DDPG 继承 TD3、多处行号对不上。教训是：AI 给出的结构性结论需要逐个核对，不能只抽查一个例子；行号必须按固定 commit 实际检索得出。

---

## 6. 复核记录

**复核人**：成员01（对照 SB3 基线 7cfb4dd 源码）

| # | 复核意见 | 核实结果 | 修改 |
|---|---------|---------|------|
| 1 | DDPG 应继承 TD3，而非直接继承 OffPolicyAlgorithm | 属实（`ddpg/ddpg.py:L14`） | 两份文档的架构图、继承树、`architecture.mmd`/PNG 已改为 OffPolicyAlgorithm → TD3 → DDPG |
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
