[← 全部教程](../README.md) · [主页](../../README.md) · [English](../en/04-inference.md) | **中文**

# 4. 推理、评估与输出

本教程涵盖：用预训练权重运行推理、完整的评估命令行接口，以及如何解读 LarNO 产生的输出。

## 用预训练权重快速测试（region1）

在花时间训练之前，先用预训练的 Futian 权重在 **Futian（region1_20m）测试集** 上运行推理，验证安装是否正常。该步骤约需 **3 分钟**，立即生成洪水图与指标。

**前置条件：** 完成教程 [1. 安装](01-installation.md)、[2. 数据集准备](02-datasets.md) 和 [3. 预训练权重](03-pretrained-weights.md)。预训练权重路径已写入 `configs/urbanflood_config_2d.yaml` — 无需手动编辑路径。

### 步骤 1 — 检查配置

打开 `configs/urbanflood_config_2d.yaml`，确认以下取值正确（默认应已正确）：

```yaml
tfno2d:
  hidden_channels: 32    # 必须与预训练架构一致

data:
  train_location: "region1_20m"
  train_list: "region1_fulltrain.txt"
  test_list: "region1_test.txt"

eval:
  locations: "region1_20m"
```

### 步骤 2 — 运行推理

💻 在 `code/urbanflood_larfno/` 目录下：

```bash
# Linux / AutoDL：
python test.py --config urbanflood_config_2d.yaml --expr_id 20260220_183648_006352

# Windows：
python test.py --config urbanflood_config_2d.yaml --expr_id 20260220_183648_006352
```

> 将 `20260220_183648_006352` 替换为你下载的预训练权重的实际实验文件夹名。

结果保存在 `exp/<new_timestamp>/`：

- `test_metrics/region1_20m/` — 每个事件的 R²、MAE、CSI 的 Excel 表
- `visualization/region1_20m/` — PNG 快照 + 动态 GIF
- `pred_results/region1_20m/` — 原始预测数组

## 评估命令行接口

💻 在 `code/urbanflood_larfno/` 目录下：

```bash
# 自动检测最新实验：
python test.py --config <yaml_file>

# 指定某个特定实验：
python test.py --config <yaml_file> --expr_id 20260301_120000_000000

# 覆盖数据 / 输出路径：
python test.py --config <yaml_file> \
  --data_root ../../benchmark/urbanflood \
  --exp_root  ../../exp \
  --expr_id   <expr_id>
```

要一次性在 **多个位置** 上评估，在 YAML 中设置 `eval.locations`：

```yaml
eval:
  locations: "ukea_8m_5min,ukea_2m_5min"   # 同时测试零样本 2m 分辨率
```

📁 结果写入：

```
exp/<timestamp>/
├── test_metrics/<location>/metrics_epoch_N_n@M.xlsx
├── visualization/<location>/epoch_N/       ← PNG 快照 + 动态 GIF
└── pred_results/<location>/               ← 原始预测数组（.npy）
```

## 输出与指标

### 洪水图与动画（`visualization/`）

每个事件生成：
- **PNG 文件** — 并排快照（左：MIKE+ 参考；右：LarNO 预测）
- **GIF 文件** — 在所有时间步上以 50 fps 的动态对比

干燥单元：白色。淹没越深：蓝色越深（色标：0–2 m）。

### 性能指标（`test_metrics/`）

每个位置一个 Excel 文件，每个事件一行，外加一个整体的 mean ± std 行。

| 指标           | 物理含义                                                        | 越优方向 |
| -------------- | --------------------------------------------------------------- | -------- |
| **R²**         | 解释方差比例（1.0 = 完美）。                                   | 越高越好 |
| **MSE / RMSE** | 平均 / 均方根水深误差（m²/m）。                                | 越低越好 |
| **MAE**        | 平均绝对水深误差（m）。                                        | 越低越好 |
| **PeakR²**     | 峰值淹没水深的 R² — 对洪水风险至关重要。                       | 越高越好 |
| **CSI**        | 干湿分类指数（阈值 = `flood_threshold`，默认 3 cm）。          | 越高越好 |

---

上一篇：[← 3. 预训练权重](03-pretrained-weights.md) · 下一篇：[5. 训练 →](05-training.md)
