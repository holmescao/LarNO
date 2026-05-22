[← All tutorials](../README.md) · [Home](../../README.md) · **English** | [中文](../zh/03-pretrained-weights.md)

# 3. Pre-trained Weights

We provides a **Futian (region1_20m) pre-trained checkpoint**, trained to paper accuracy on the Shenzhen case study. This checkpoint is used for:

- **Quick Test** ([4. Inference](04-inference.md)) — run inference on region1 immediately, no training required
- **Scenario A** ([5. Training](05-training.md)) — fine-tune on UKEA in just 100 epochs

## Architecture

The checkpoint uses the following architecture. Any config that loads it **must match these values exactly**; changing them will cause a weight-shape mismatch error.

| Parameter         | Value |
| ----------------- | ----- |
| `hidden_channels` | 32    |
| `n_modes_height`  | 100   |
| `n_modes_width`   | 140   |
| `n_layers`        | 4     |

## Download

| Mirror                     | Link                                                                                                            |
| -------------------------- | --------------------------------------------------------------------------------------------------------------- |
| HuggingFace                | [holmescao/LarNO](https://huggingface.co/holmescao/LarNO)                                                       |
| Google Drive               | [Download (no password)](https://drive.google.com/file/d/1ITPoTWQkm5v9kdZT9fqza2Xd4a6Lc-0t/view?usp=drive_link) |
| Baidu Cloud (code: `LaNO`) | [Download](https://pan.baidu.com/s/1bJuO5sBdt6kNm5dwOl58WQ?pwd=LaNO)                                            |

## Placement

📁 Extract and place the checkpoint under the `exp/` directory:

```
LarNO/
└── exp/
    └── <expr_id>/                           ← e.g. 20260220_183648_006352
        └── weights/
            └── <checkpoint_name>/           ← e.g. model_epoch_992_error@0.000055821
                └── <checkpoint_name>_state_dict.pt
```

- For **Quick Test**: the path is already embedded in `configs/urbanflood_config_2d.yaml` — no manual editing needed.
- For **Scenario A**: update the `finetune` block in `configs/ukea_finetune.yaml` with your actual `<expr_id>` and `<checkpoint_name>` (see [5. Training](05-training.md)).

---

Prev: [← 2. Dataset Preparation](02-datasets.md) · Next: [4. Inference, Evaluation & Outputs →](04-inference.md)
