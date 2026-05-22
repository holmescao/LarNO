[← All tutorials](../README.md) · [Home](../../README.md) · **English** | [中文](../zh/07-reference.md)

# 7. Reference — Configuration, Structure & Outputs

A reference for the configuration files, their key parameters, the 13 input channels, and the overall project layout.

## Configuration Reference

### Config files overview

| File                           | Purpose                                          |
| ------------------------------ | ------------------------------------------------ |
| `configs/ukea_finetune.yaml`   | Fine-tune Futian pretrained → UKEA **(default)** |
| `configs/ukea_scratch.yaml`    | Train UKEA from random initialisation            |
| `configs/region1_scratch.yaml` | Train Futian / custom dataset from scratch       |

### Key parameters and when to change them

```yaml
# ── Architecture ──────────────────────────────────────────────────────────────
tfno2d:
  n_modes_height: 100   # [finetune] must equal pretrained value (100)
                        # [scratch]  set to ~H/4; increase for better accuracy
  n_modes_width:  140   # same rule as n_modes_height
  hidden_channels: 32   # reduce to 16 for faster training; 32 for paper accuracy (region1)
  n_layers: 4           # reduce to 2 for faster training; 4 for paper accuracy

# ── Fine-tuning ───────────────────────────────────────────────────────────────
finetune:
  enabled: True                          # False = train from scratch
  pretrained_dir: "../../exp/<id>/..."   # path to region1 pretrained checkpoint
  state_dict_name: "<name>_state_dict.pt"

# ── Training schedule ─────────────────────────────────────────────────────────
opt:
  n_epochs: 100         # 100 for fine-tuning; 1000 for scratch
  warm_up_iter: 1       # 1 for fine-tuning; 10 for scratch
  T_max: 100            # MUST equal n_epochs (cosine annealing period)
  lr_max: 1e-2          # peak learning rate
  lr_min: 1e-4          # minimum learning rate
  training_loss: 'WMSE' # 'l2' | 'h1' | 'WMSE' | 'h1WMSE'
  window_size: 8        # time steps predicted per forward pass

# ── Dataset ───────────────────────────────────────────────────────────────────
data:
  train_location: "ukea_8m_5min"   # folder name under benchmark/urbanflood/flood/
  train_list: "ukea_train.txt"     # event names for training
  test_list:  "ukea_test.txt"      # event names for evaluation during training
  batch_size: 1                    # set to N for N-GPU DDP
  num_workers_train: 0             # keep 0 on Windows; 4 on Linux

# ── Evaluation ────────────────────────────────────────────────────────────────
eval:
  flood_threshold: 0.03   # metres above which a cell is "flooded" (for CSI)
  locations: "ukea_8m_5min"   # comma-separated; add "ukea_2m_5min" for super-res eval

# ── Distributed training ──────────────────────────────────────────────────────
distributed:
  use_distributed: False  # True + torchrun for multi-GPU (Linux only)
```

### What are the 13 input channels?

| Channels | Content                                        |
| -------- | ---------------------------------------------- |
| 1 – 6    | Past 6 rainfall fields (normalised)            |
| 7 – 12   | Past 6 cumulative-rainfall fields (normalised) |
| 13       | DEM (normalised to [0, 1])                     |

## Project Structure

```
LarNO/
├── benchmark/                          ← populate after downloading data
│   └── urbanflood/
│       ├── flood/
│       │   ├── ukea_8m_5min/           ← one sub-folder per event
│       │   └── region1_20m/
│       └── geodata/
│           ├── ukea_8m_5min/
│           └── region1_20m/
│
├── exp/                                ← created automatically during training
│   └── <timestamp>/
│       ├── weights/                    ← model checkpoints (.pt)
│       ├── visualization/              ← flood maps (PNG) + animated GIFs
│       ├── pred_results/              ← predicted depth arrays (.npy)
│       └── test_metrics/              ← performance tables (.xlsx)
│
└── code/urbanflood_larfno/             ← run all scripts from here
    ├── train.py                        ← training entry point
    ├── test.py                         ← evaluation entry point
    ├── run_train.py                    ← Windows wrapper: logs to train_log.txt (pass --config like train.py)
    ├── pyproject.toml                  ← package definition
    ├── requirements.txt                ← extra pip dependencies
    │
    ├── assets/                         ← figures for the README
    │
    ├── configs/
    │   ├── ukea_finetune.yaml          ← Scenario A: fine-tune UKEA (default)
    │   ├── ukea_scratch.yaml           ← Scenario B: train UKEA from scratch
    │   ├── region1_scratch.yaml        ← Scenario C: train Futian / custom
    │   ├── ukea_train.txt              ← UKEA training events (8 events)
    │   ├── ukea_test.txt               ← UKEA test events (12 events)
    │   ├── region1_fulltrain.txt       ← Futian training events (64 events)
    │   ├── region1_smalltrain.txt      ← Futian training events subset (16 events)
    │   └── region1_test.txt            ← Futian test events (16 events)
    │
    ├── utils/
    │   ├── torch_utils.py
    │   └── distributed_utils.py
    │
    └── neuralop/
        ├── models/fno.py               ← TFNO2d + CGRU model
        ├── layers/ConvRNN.py           ← CGRU temporal memory cell
        ├── data/datasets/Dynamic2DFlood.py  ← dataset loader
        ├── training/trainer.py         ← training loop, evaluation, GIF generation
        └── losses/data_losses.py       ← WMSE, H1, Lp losses
```

---

Prev: [← 6. Cloud GPU — AutoDL](06-cloud-gpu-autodl.md) · Next: [8. FAQ →](08-faq.md)
