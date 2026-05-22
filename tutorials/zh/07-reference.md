[← 全部教程](../README.md) · [主页](../../README.md) · [English](../en/07-reference.md) | **中文**

# 7. 参考 — 配置、结构与输出

配置文件、关键参数、13 个输入通道，以及项目整体布局的参考。

## 配置参考

### 配置文件概览

| 文件                           | 用途                                       |
| ------------------------------ | ------------------------------------------ |
| `configs/ukea_finetune.yaml`   | 用 Futian 预训练权重微调 → UKEA **（默认）** |
| `configs/ukea_scratch.yaml`    | 在 UKEA 上从随机初始化训练                 |
| `configs/region1_scratch.yaml` | 在 Futian / 自定义数据集上从头训练         |

### 关键参数及修改时机

```yaml
# ── 架构 ──────────────────────────────────────────────────────────────────────
tfno2d:
  n_modes_height: 100   # [finetune] 必须等于预训练值（100）
                        # [scratch]  设为约 H/4；增大可提高精度
  n_modes_width:  140   # 规则同 n_modes_height
  hidden_channels: 32   # 减到 16 可加快训练；32 对应论文精度（region1）
  n_layers: 4           # 减到 2 可加快训练；4 对应论文精度

# ── 微调 ──────────────────────────────────────────────────────────────────────
finetune:
  enabled: True                          # False = 从头训练
  pretrained_dir: "../../exp/<id>/..."   # region1 预训练检查点的路径
  state_dict_name: "<name>_state_dict.pt"

# ── 训练计划 ──────────────────────────────────────────────────────────────────
opt:
  n_epochs: 100         # 微调用 100；从头训练用 1000
  warm_up_iter: 1       # 微调用 1；从头训练用 10
  T_max: 100            # 必须等于 n_epochs（余弦退火周期）
  lr_max: 1e-2          # 峰值学习率
  lr_min: 1e-4          # 最小学习率
  training_loss: 'WMSE' # 'l2' | 'h1' | 'WMSE' | 'h1WMSE'
  window_size: 8        # 每次前向预测的时间步数

# ── 数据集 ────────────────────────────────────────────────────────────────────
data:
  train_location: "ukea_8m_5min"   # benchmark/urbanflood/flood/ 下的文件夹名
  train_list: "ukea_train.txt"     # 训练用事件名
  test_list:  "ukea_test.txt"      # 训练过程中评估用的事件名
  batch_size: 1                    # N 卡 DDP 时设为 N
  num_workers_train: 0             # Windows 上保持 0；Linux 上设为 4

# ── 评估 ──────────────────────────────────────────────────────────────────────
eval:
  flood_threshold: 0.03   # 高于该值（米）的单元视为"被淹"（用于 CSI）
  locations: "ukea_8m_5min"   # 逗号分隔；加 "ukea_2m_5min" 进行超分评估

# ── 分布式训练 ────────────────────────────────────────────────────────────────
distributed:
  use_distributed: False  # True + torchrun 用于多卡（仅 Linux）
```

### 13 个输入通道是什么？

| 通道     | 内容                                  |
| -------- | ------------------------------------- |
| 1 – 6    | 过去 6 个降雨场（归一化）            |
| 7 – 12   | 过去 6 个累积降雨场（归一化）        |
| 13       | DEM（归一化到 [0, 1]）               |

## 项目结构

```
LarNO/
├── benchmark/                          ← 下载数据后填充
│   └── urbanflood/
│       ├── flood/
│       │   ├── ukea_8m_5min/           ← 每个事件一个子文件夹
│       │   └── region1_20m/
│       └── geodata/
│           ├── ukea_8m_5min/
│           └── region1_20m/
│
├── exp/                                ← 训练时自动创建
│   └── <timestamp>/
│       ├── weights/                    ← 模型检查点（.pt）
│       ├── visualization/              ← 洪水图（PNG）+ 动态 GIF
│       ├── pred_results/              ← 预测水深数组（.npy）
│       └── test_metrics/              ← 性能表（.xlsx）
│
└── code/urbanflood_larfno/             ← 从此处运行所有脚本
    ├── train.py                        ← 训练入口
    ├── test.py                         ← 评估入口
    ├── run_train.py                    ← Windows 封装：日志写入 train_log.txt（与 train.py 一样传 --config）
    ├── pyproject.toml                  ← 包定义
    ├── requirements.txt                ← 额外的 pip 依赖
    │
    ├── assets/                         ← README 用的图片
    │
    ├── configs/
    │   ├── ukea_finetune.yaml          ← 场景 A：微调 UKEA（默认）
    │   ├── ukea_scratch.yaml           ← 场景 B：从头训练 UKEA
    │   ├── region1_scratch.yaml        ← 场景 C：训练 Futian / 自定义
    │   ├── ukea_train.txt              ← UKEA 训练事件（8 个）
    │   ├── ukea_test.txt               ← UKEA 测试事件（12 个）
    │   ├── region1_fulltrain.txt       ← Futian 训练事件（64 个）
    │   ├── region1_smalltrain.txt      ← Futian 训练事件子集（16 个）
    │   └── region1_test.txt            ← Futian 测试事件（16 个）
    │
    ├── utils/
    │   ├── torch_utils.py
    │   └── distributed_utils.py
    │
    └── neuralop/
        ├── models/fno.py               ← TFNO2d + CGRU 模型
        ├── layers/ConvRNN.py           ← CGRU 时序记忆单元
        ├── data/datasets/Dynamic2DFlood.py  ← 数据集加载器
        ├── training/trainer.py         ← 训练循环、评估、GIF 生成
        └── losses/data_losses.py       ← WMSE、H1、Lp 损失
```

---

上一篇：[← 6. 云 GPU — AutoDL](06-cloud-gpu-autodl.md) · 下一篇：[8. 常见问题 →](08-faq.md)
