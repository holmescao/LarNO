[← 全部教程](../README.md) · [主页](../../README.md) · [English](../en/05-training.md) | **中文**

# 5. 训练（微调 / 从头训练）

本教程涵盖如何选择配置，以及三种训练场景：在 UKEA 上微调（推荐）、在 UKEA 上从头训练、在 Futian / 自定义数据集上从头训练。

## 选择配置

所有超参数都由 `configs/` 中的 **YAML 配置文件** 控制。我们提供了三个开箱即用的配置。通过 `--config` 传入所需文件：

```bash
python train.py --config <yaml_file>   # 训练
python test.py  --config <yaml_file>   # 评估
```

### 一览对比

| 配置文件                   | 用途                               | n_modes (H×W)         | hidden_ch | n_layers | warm_up | n_epochs | finetune |
| -------------------------- | ---------------------------------- | --------------------- | --------- | -------- | ------- | -------- | -------- |
| **`ukea_finetune.yaml`** ✅ | 用 Futian 权重在 UKEA 上微调       | **100 × 140**（固定） | 32        | 4        | 1       | 100      | **True** |
| `ukea_scratch.yaml`        | 在 UKEA 上从头训练                 | 25 × 60               | 32        | 2        | 10      | 1000     | False    |
| `region1_scratch.yaml`     | 在 Futian / 自定义数据集上从头训练 | 40 × 56               | 16        | 2        | 10      | 1000     | False    |

> **注意：** scratch 配置使用轻量架构以缩短训练时间。要复现论文在 region1 上的精度，请在 YAML 中设置 `n_modes_height: 100`、`n_modes_width: 140`、`hidden_channels: 32`、`n_layers: 4`。

> **为什么先从微调开始？** Futian 预训练模型已经学到了通用的洪水动力学（地形汇流、径流累积）。微调仅用 100 个 epoch 就能把这些已学特征适配到 UKEA，远少于从头训练。这是新用户推荐的路径。

> **为什么 `ukea_finetune.yaml` 必须保持 `n_modes = 100 × 140`？** FNO 谱层存储的权重张量形状由 `n_modes` 决定。加载 region1 预训练检查点要求架构与 region1 训练时 **完全一致**（100 × 140 模式）。改动 `n_modes` 会导致权重形状不兼容。

---

## 场景 A — 在 UKEA 上微调（推荐）

> 使用 `configs/ukea_finetune.yaml` — 默认配置。

### 步骤 1 — 下载预训练权重

下载链接与放置说明见 [3. 预训练权重](03-pretrained-weights.md)。

### 步骤 2 — 编辑配置

⚙️ 打开 `configs/ukea_finetune.yaml`，**仅** 更新 `finetune` 块：

```yaml
finetune:
  enabled: True
  pretrained_dir: "../../exp/<expr_id>/weights/<checkpoint_name>"
  state_dict_name: "<checkpoint_name>_state_dict.pt"
```

将 `<expr_id>` 和 `<checkpoint_name>` 替换为实际的文件夹名和文件名。
**不要改动 `n_modes_height` 或 `n_modes_width`** — 它们必须保持 100 / 140 以匹配预训练架构。

### 步骤 3 — 训练

💻 在 `code/urbanflood_larfno/` 目录下：

```bash
# Linux / AutoDL：
python train.py --config ukea_finetune.yaml 2>&1 | tee train_log.txt

# Windows：
python run_train.py --config ukea_finetune.yaml
```

> **多卡提示（仅 Linux）：** 在 `ukea_finetune.yaml` 中设置 `distributed.use_distributed: True` 和 `data.batch_size: <你的 GPU 数量>`，然后用以下命令启动：
> ```bash
> torchrun --nproc_per_node=<你的 GPU 数量> train.py --config ukea_finetune.yaml 2>&1 | tee train_log.txt
> ```

### 步骤 4 — 评估

```bash
python test.py --config ukea_finetune.yaml --expr_id <timestamp>
```

> 默认的 `eval.locations: "ukea_8m_5min,ukea_2m_5min"` 会同时评估 8 m 训练分辨率和 2 m 零样本超分辨率网格。

---

## 场景 B — 在 UKEA 上从头训练

> 使用 `configs/ukea_scratch.yaml`。无需预训练权重。

### 步骤 1 —（可选）编辑配置

⚙️ `configs/ukea_scratch.yaml` 开箱即用。默认架构较轻量以缩短训练时间。你可以调整：

```yaml
tfno2d:
  n_modes_height: 25    # 提高到 50 可获得更好精度
  n_modes_width:  60    # 提高到 120 可获得更好精度
  hidden_channels: 32
  n_layers: 2           # 提高到 4 可获得更好精度

opt:
  n_epochs: 1000        # 从头训练需要更多 epoch
  warm_up_iter: 10
  T_max: 1000           # 保持 T_max = n_epochs
  lr_max: 1e-2
  lr_min: 1e-4
```

> **UKEA 没有指定的论文精度目标。** 上述默认值较轻量；可在 GPU 显存允许的范围内提高 `n_modes`、`hidden_channels` 和 `n_layers`。

### 步骤 2 — 训练

```bash
# Linux / AutoDL：
python train.py --config ukea_scratch.yaml 2>&1 | tee train_log.txt

# Windows：
python run_train.py --config ukea_scratch.yaml
```

### 步骤 3 — 评估

```bash
python test.py --config ukea_scratch.yaml --expr_id <timestamp>
```

> 默认的 `eval.locations: "ukea_8m_5min,ukea_2m_5min"` 会同时评估 8 m 训练分辨率和 2 m 零样本超分辨率网格。

---

## 场景 C — 在 Futian / 自定义数据集上从头训练

> 使用 `configs/region1_scratch.yaml`。用于 Futian 数据集或你自己的大尺度研究区域。

### 步骤 1 — 准备事件列表

✏️ 在 `configs/region1_fulltrain.txt`（或 `region1_smalltrain.txt`）和 `configs/region1_test.txt` 中填入你的事件名（每行一个事件名），与 `benchmark/urbanflood/flood/<location>/` 下的子文件夹名一致。

### 步骤 2 — 编辑配置

⚙️ 打开 `configs/region1_scratch.yaml`，更新 `data` 和 `eval` 块：

```yaml
tfno2d:
  n_modes_height: 40    # 默认轻量；提高到 100 以复现论文精度
  n_modes_width:  56    # 默认轻量；提高到 140 以复现论文精度
  hidden_channels: 16   # 默认轻量；提高到 32 以复现论文精度
  n_layers: 2           # 默认轻量；提高到 4 以复现论文精度

opt:
  n_epochs: 1000
  warm_up_iter: 10
  T_max: 1000           # 保持 T_max = n_epochs
  window_size: 4        # 每次前向预测的时间步数

data:
  train_location: "region1_20m"    # 或你自己的文件夹名
  train_list: "region1_smalltrain.txt"  # 16 个事件（快速起步）；用 region1_fulltrain.txt 进行完整的 64 事件训练
  test_list:  "region1_test.txt"

eval:
  locations: "region1_20m"         # 或你自己的文件夹名
```

> **要复现 region1 上的论文精度：** 设置 `n_modes_height: 100`、`n_modes_width: 140`、`hidden_channels: 32`、`n_layers: 4`。

**对于自定义数据集**，还需将数据放置在：

```
benchmark/urbanflood/
├── flood/<your_location>/<event_name>/
│   ├── dem.npy         形状 (H, W)
│   ├── rainfall.npy    形状 (T, H, W)
│   └── h.npy           形状 (T, H, W)
└── geodata/<your_location>/
    └── dem.npy         （用于可视化的同一 DEM）
```

然后在 YAML 中设置 `train_location: "<your_location>"`。

### 步骤 3 — 训练

```bash
# Linux / AutoDL：
python train.py --config region1_scratch.yaml 2>&1 | tee train_log.txt

# Windows：
python run_train.py --config region1_scratch.yaml
```

### 步骤 4 — 评估

```bash
python test.py --config region1_scratch.yaml --expr_id <timestamp>
```

---

上一篇：[← 4. 推理、评估与输出](04-inference.md) · 下一篇：[6. 云 GPU — AutoDL →](06-cloud-gpu-autodl.md)
