# LarNO Script Dependency Chain

> 生成时间：2026-03-12
> 目的：梳理推理/训练入口与各模块之间的完整依赖关系，供 Colab notebook 设计参考。

---

## 顶层入口

```
code/urbanflood_larfno/
├── test.py          ← 推理入口（Colab 主要调用此脚本）
├── train.py         ← 训练入口
└── run_train.py     ← Windows 专用训练封装（subprocess + UTF-8 日志）
```

---

## test.py 依赖链

```
test.py
├── [stdlib]  os, sys, re, argparse, time, pathlib, collections
├── [pip]     torch, pandas
├── [pip]     configmypy  ← 解析 YAML + argparse 配置
│                            ConfigPipeline / YamlConfig / ArgparseConfig
│                            读取 configs/<yaml_file>
├── neuralop/                ← 本地包（pip install -e .）
│   ├── __init__.py          → 导出 H1Loss, LpLoss, get_model
│   ├── models/fno.py        ← TFNO2d 模型主体
│   │   ├── layers/fno_block.py       ← FNOBlocks（频域卷积块）
│   │   ├── layers/spectral_convolution.py  ← SpectralConv（FFT + 模式混合）
│   │   ├── layers/ConvRNN.py         ← CGRU_cell（时序隐状态更新）
│   │   ├── layers/channel_mlp.py     ← ChannelMLP（通道映射）
│   │   ├── layers/padding.py         ← DomainPadding
│   │   ├── layers/embeddings.py      ← GridEmbeddingND / GridEmbedding2D
│   │   ├── layers/complex.py         ← ComplexValued
│   │   └── models/base_model.py      ← BaseModel 基类
│   ├── training/trainer.py  ← Trainer 类
│   │   ├── evaluate_for_test()       ← 推理主循环（逐事件 autoregressive）
│   │   └── save_comparison_gif()     ← 生成 MIKE+ vs LarNO 对比 GIF
│   ├── data/datasets/Dynamic2DFlood.py  ← 数据集加载器
│   │   ├── 读取 configs/<test_list>.txt  ← 事件名列表
│   │   ├── 读取 benchmark/urbanflood/flood/<location>/<event>/
│   │   │   ├── rainfall.npy  (T, H, W)
│   │   │   └── h.npy         (T, H, W)
│   │   └── 读取 benchmark/urbanflood/geodata/<location>/dem.npy  (H, W)
│   └── losses/data_losses.py  ← WMSELoss, H1WMSELoss, LpLoss, H1Loss
└── utils/
    ├── torch_utils.py        ← select_device()（CPU/CUDA 自动选择）
    └── distributed_utils.py  ← init_seeds()（随机种子）
```

**数据流（推理）：**
```
YAML config
    → ConfigPipeline → config 对象
    → get_model(config) → TFNO2d 实例
    → 加载 checkpoint (.pt) → model.load_state_dict()
    → Dynamic2DFlood(test) → DataLoader
    → Trainer.evaluate_for_test()
        → 逐 batch autoregressive 推理（window_size=8 steps/pass，共 72 steps）
        → 计算 R², MAE, MSE, RMSE, PeakR², CSI
        → save_comparison_gif() → visualization/<location>/<event>.gif
    → save_metrics_to_excel() → test_metrics/<location>/metrics_*.xlsx
```

---

## train.py 额外依赖

```
train.py（在 test.py 基础上增加）
├── torch.distributed       ← DDP（Linux 多卡，Windows 不支持）
├── neuralop/losses/data_losses.py
│   └── WMSELoss, H1WMSELoss  ← 训练损失
└── utils/distributed_utils.py
    └── init_seeds()
```

---

## 关键路径（Colab 推理所需的最小文件集）

| 类别 | 路径 | 说明 |
|---|---|---|
| 入口脚本 | `code/urbanflood_larfno/test.py` | 推理主脚本 |
| 配置文件 | `code/urbanflood_larfno/configs/urbanflood_config_2d.yaml` | region1 Quick Test 配置 |
| 事件列表 | `code/urbanflood_larfno/configs/region1_test.txt` | 16 个测试事件名 |
| 预训练权重 | `exp/20251025_184115_247446/weights/model_epoch_978_*/` | Google Drive 下载 |
| 数据集 | `benchmark/urbanflood/flood/region1_20m/<event>/` | 官网下载 |
| 地理数据 | `benchmark/urbanflood/geodata/region1_20m/dem.npy` | 同上 |
| 本地包 | `code/urbanflood_larfno/` | pip install -e . |
| pip 依赖 | `code/urbanflood_larfno/requirements.txt` | tensorly, configmypy 等 |

---

## Colab 推理流程（test.py 调用方式）

```bash
# 工作目录必须为 code/urbanflood_larfno/
cd /content/LarNO/code/urbanflood_larfno

# 安装本地包
pip install -e . --no-deps -q
pip install -r requirements.txt -q

# 运行推理（使用 Quick Test 配置 + 预训练权重）
python test.py --config urbanflood_config_2d.yaml --expr_id <expr_id>
```

---

## 13 个输入通道构成

| 通道 | 内容 | 来源 |
|---|---|---|
| 1–6 | 过去 6 步降雨场（归一化） | `rainfall.npy` |
| 7–12 | 过去 6 步累积降雨场（归一化） | `rainfall.npy` 累积求和 |
| 13 | DEM 高程场（归一化到 [0,1]） | `dem.npy` |

> 注：排水管网数据（drainage）属于保密数据，**未在开源版本中发布**。

---

## 输出文件位置

```
exp/<new_timestamp>/
├── test_metrics/region1_20m/
│   └── metrics_epoch_N_n@M.xlsx   ← R², MAE, MSE, RMSE, PeakR², CSI
├── visualization/region1_20m/
│   └── epoch_N/
│       ├── event_*.png             ← 逐步快照（MIKE+ vs LarNO）
│       └── event_*.gif             ← 动画对比（50 fps）
└── pred_results/region1_20m/
    └── *.npy                       ← 原始预测深度数组
```
