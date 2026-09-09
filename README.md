# EasyRL Teaching Lab（强化学习教学实验箱）

一个面向教学的强化学习实验项目：**多环境可切换、多算法可对比、配置集中、过程可视化、带智能运维报告**。学生只需改一个 `configs/config.yaml` 即可切换环境与算法、调节超参数。

## 特性

- **多环境**：CartPole（倒立摆）、MountainCar（小车上山），通过注册表一键切换。
- **多算法**：`q_learning`（表格 Q）、`sarsa`（表格 SARSA）、`dqn`（深度 Q 网络）、`reinforce`（策略梯度），统一接口，配置切换。
- **集中配置**：所有训练/超参数、环境选择、可视化与运维开关集中在 `configs/config.yaml`。
- **训练可视化**：自动绘制 `reward` 曲线与 `loss` 曲线（表格法为 TD-error，DQN 为 MSE，REINFORCE 为策略梯度损失）。
- **动态图结果**：训练后自动生成 GIF 动画（贪心策略 + 随机策略对照，可选“弱训练”对照）。
- **智能运维**：训练后生成自包含的 HTML 运维报告（内嵌曲线图、健康状态、教学建议）。
- **输出分类**：`output/{环境}/{models,charts,gifs,ops}` 按环境、按类型分类存放。

## 目录结构

```
easyrl-TeachingLab/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml                # 集中配置（环境/算法/训练/可视化/运维）
├── main.py                         # 入口：python main.py [--env ...] [--algo ...]
├── rl_teach/
│   ├── environments/               # 环境层：base + cartpole + mountaincar + registry
│   ├── agents/                    # 算法层：q_learning / sarsa / dqn / reinforce
│   ├── training/trainer.py        # 通用训练主循环
│   ├── viz/                       # 曲线图 + 自绘 GIF
│   └── ops/                       # 监控 / 健康检查 / HTML 报告
└── output/                        # 运行产物（按 环境/类型 分类）
    ├── cartpole/{models,charts,gifs,ops}
    └── mountaincar/{models,charts,gifs,ops}
```

## 环境要求

```
pip install -r requirements.txt
```

依赖：`numpy`、`gym`、`matplotlib`、`pyyaml`、`imageio`、`Pillow`、`psutil`。

> 说明：本项目的自绘 GIF 与曲线图均使用 `matplotlib` 的 `Agg` 无显示器后端，**云端/无图形界面环境均可直接运行**。

## 快速开始

在项目根目录执行：

```bash
# 默认（config.yaml 中的 cartpole + q_learning）
python main.py

# 切换到 MountainCar 环境
python main.py --env mountaincar

# 切换到 DQN 算法
python main.py --algo dqn

# 同时切换环境与算法
python main.py --env mountaincar --algo dqn
```

首次运行会完整训练并将模型保存到 `output/{环境}/models/`；之后可将 `config.yaml` 中
`training.load_model` 设为 `true`，直接加载模型、跳过训练秒出 GIF。

## 配置文件说明（configs/config.yaml）

| 配置项 | 含义 |
|---|---|
| `environment.name` | `cartpole` / `mountaincar` |
| `environment.max_steps` | 单轮最大步数 |
| `environment.num_digitized` | 连续状态离散化区间数（表格法） |
| `agent.type` | `q_learning` / `sarsa` / `dqn` / `reinforce` |
| `agent.learning_rate` | 表格法学习率（DQN/REINFORCE 使用各自内置默认 lr） |
| `agent.gamma` | 折扣因子 |
| `training.episodes` | 最大训练轮数 |
| `training.consecutive_success` | 连续成功 N 轮即视为收敛，提前停止 |
| `visualization.gifs.*` | 是否生成 trained/random/weak 对照 GIF |
| `ops.enabled` | 是否生成 HTML 运维报告 |
| `output.base_dir` | 输出根目录 |

## 教学建议

1. 先跑 `cartpole + q_learning`，展示「随机策略 vs 训练后」GIF 对比，直观理解 RL 是“学”出来的。
2. 对比 `q_learning` 与 `sarsa` 的差异（on-policy vs off-policy）。
3. 将 `agent.type` 改为 `dqn` / `reinforce`，对比不同算法的 `reward` / `loss` 曲线。
4. 打开 `output/{环境}/ops/ops_report.html`，讲解训练健康度与收敛情况。

## 输出文件说明

- `output/{环境}/models/{环境}_{算法}.npy`：训练好的模型/参数。
- `output/{环境}/charts/reward.png`、`loss.png`：奖励与损失曲线。
- `output/{环境}/gifs/*_trained.gif`、`*_random.gif`（`*_weak.gif` 可选）：动态演示。
- `output/{环境}/ops/ops_report.html`：自包含运维报告（双击即可在浏览器查看）。
