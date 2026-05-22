[← 全部教程](../README.md) · [主页](../../README.md) · [English](../en/03-pretrained-weights.md) | **中文**

# 3. 预训练权重

我们提供了一个 **Futian（region1_20m）预训练检查点**，在深圳案例上训练至论文精度。该检查点用于：

- **快速测试**（[4. 推理](04-inference.md)）— 立即在 region1 上运行推理，无需训练
- **场景 A**（[5. 训练](05-training.md)）— 仅用 100 个 epoch 在 UKEA 上微调

## 架构

该检查点使用以下架构。任何加载它的配置 **都必须与这些取值完全一致**；改动会导致权重形状不匹配错误。

| 参数              | 取值  |
| ----------------- | ----- |
| `hidden_channels` | 32    |
| `n_modes_height`  | 100   |
| `n_modes_width`   | 140   |
| `n_layers`        | 4     |

## 下载

| 镜像                        | 链接                                                                                                            |
| --------------------------- | --------------------------------------------------------------------------------------------------------------- |
| HuggingFace                 | [holmescao/LarNO](https://huggingface.co/holmescao/LarNO)                                                       |
| Google Drive                | [下载（无密码）](https://drive.google.com/file/d/1ITPoTWQkm5v9kdZT9fqza2Xd4a6Lc-0t/view?usp=drive_link)         |
| 百度网盘（提取码：`LaNO`）  | [下载](https://pan.baidu.com/s/1bJuO5sBdt6kNm5dwOl58WQ?pwd=LaNO)                                                |

## 放置位置

📁 解压并将检查点放置在 `exp/` 目录下：

```
LarNO/
└── exp/
    └── <expr_id>/                           ← 例如 20260220_183648_006352
        └── weights/
            └── <checkpoint_name>/           ← 例如 model_epoch_992_error@0.000055821
                └── <checkpoint_name>_state_dict.pt
```

- **快速测试**：路径已经写在 `configs/urbanflood_config_2d.yaml` 中 — 无需手动编辑。
- **场景 A**：用你实际的 `<expr_id>` 和 `<checkpoint_name>` 更新 `configs/ukea_finetune.yaml` 的 `finetune` 块（见 [5. 训练](05-training.md)）。

---

上一篇：[← 2. 数据集准备](02-datasets.md) · 下一篇：[4. 推理、评估与输出 →](04-inference.md)
