[← All tutorials](../README.md) · [Home](../../README.md) · **English** | [中文](../zh/04-inference.md)

# 4. Inference, Evaluation & Outputs

This tutorial covers running inference with the pre-trained weights, the full evaluation CLI, and how to read the outputs LarNO produces.

## Quick Test with pre-trained weights (region1)

Before spending time on training, first verify that your installation works by running **inference on the Futian (region1_20m) test set** using the pre-trained Futian weights. This takes about **3 minutes** and produces flood maps and metrics immediately.

**Prerequisites:** complete tutorials [1. Installation](01-installation.md), [2. Dataset Preparation](02-datasets.md), and [3. Pre-trained Weights](03-pretrained-weights.md). The pre-trained weight path is already embedded in `configs/urbanflood_config_2d.yaml` — no manual editing of paths is needed.

### Step 1 — Verify the config

Open `configs/urbanflood_config_2d.yaml` and confirm these values are correct (they should be by default):

```yaml
tfno2d:
  hidden_channels: 32    # must match pretrained architecture

data:
  train_location: "region1_20m"
  train_list: "region1_fulltrain.txt"
  test_list: "region1_test.txt"

eval:
  locations: "region1_20m"
```

### Step 2 — Run inference

💻 From `code/urbanflood_larfno/`:

```bash
# Linux / AutoDL:
python test.py --config urbanflood_config_2d.yaml --expr_id 20260220_183648_006352

# Windows:
python test.py --config urbanflood_config_2d.yaml --expr_id 20260220_183648_006352
```

> Replace `20260220_183648_006352` with the actual experiment folder name from your downloaded pre-trained weights.

Results are saved to `exp/<new_timestamp>/`:

- `test_metrics/region1_20m/` — Excel table with R², MAE, CSI per event
- `visualization/region1_20m/` — PNG snapshots + animated GIFs
- `pred_results/region1_20m/` — raw prediction arrays

## Evaluation CLI

💻 From `code/urbanflood_larfno/`:

```bash
# Auto-detect latest experiment:
python test.py --config <yaml_file>

# Specify a particular experiment:
python test.py --config <yaml_file> --expr_id 20260301_120000_000000

# Override data / output paths:
python test.py --config <yaml_file> \
  --data_root ../../benchmark/urbanflood \
  --exp_root  ../../exp \
  --expr_id   <expr_id>
```

To evaluate on **multiple locations** at once, set `eval.locations` in the YAML:

```yaml
eval:
  locations: "ukea_8m_5min,ukea_2m_5min"   # also tests zero-shot 2m resolution
```

📁 Results are written to:

```
exp/<timestamp>/
├── test_metrics/<location>/metrics_epoch_N_n@M.xlsx
├── visualization/<location>/epoch_N/       ← PNG snapshots + animated GIF
└── pred_results/<location>/               ← raw prediction arrays (.npy)
```

## Outputs and Metrics

### Flood maps and animations (`visualization/`)

Each event produces:
- **PNG files** — side-by-side snapshots (left: MIKE+ reference, right: LarNO prediction)
- **GIF file** — animated comparison at 50 fps across all time steps

Dry cells: white. Deeper inundation: darker blue (colorbar: 0–2 m).

### Performance metrics (`test_metrics/`)

One Excel file per location, one row per event, plus an overall mean ± std row.

| Metric         | Physical meaning                                                            | Better when |
| -------------- | --------------------------------------------------------------------------- | ----------- |
| **R²**         | Variance explained (1.0 = perfect).                                         | Higher      |
| **MSE / RMSE** | Mean / root-mean-squared depth error (m²/m).                                | Lower       |
| **MAE**        | Mean absolute depth error (m).                                              | Lower       |
| **PeakR²**     | R² on peak inundation depth — critical for flood risk.                      | Higher      |
| **CSI**        | Wet/dry classification index (threshold = `flood_threshold`, default 3 cm). | Higher      |

---

Prev: [← 3. Pre-trained Weights](03-pretrained-weights.md) · Next: [5. Training →](05-training.md)
