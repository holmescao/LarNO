[← All tutorials](../README.md) · [Home](../../README.md) · **English** | [中文](../zh/05-training.md)

# 5. Training (Fine-tune / From Scratch)

This tutorial covers choosing a configuration and the three training scenarios: fine-tuning on UKEA (recommended), training UKEA from scratch, and training Futian / a custom dataset from scratch.

## Choosing a Configuration

All hyperparameters are controlled by a **YAML config file** in `configs/`. Three ready-to-use configs are provided. Pass the desired file with `--config`:

```bash
python train.py --config <yaml_file>   # training
python test.py  --config <yaml_file>   # evaluation
```

### At-a-glance comparison

| Config file                | Use case                           | n_modes (H×W)         | hidden_ch | n_layers | warm_up | n_epochs | finetune |
| -------------------------- | ---------------------------------- | --------------------- | --------- | -------- | ------- | -------- | -------- |
| **`ukea_finetune.yaml`** ✅ | Fine-tune Futian weights on UKEA   | **100 × 140** (fixed) | 32        | 4        | 1       | 100      | **True** |
| `ukea_scratch.yaml`        | Train UKEA from scratch            | 25 × 60               | 32        | 2        | 10      | 1000     | False    |
| `region1_scratch.yaml`     | Train Futian / custom from scratch | 40 × 56               | 16        | 2        | 10      | 1000     | False    |

> **Note:** The scratch configs use a lightweight architecture to reduce training time. To reproduce the paper's accuracy on region1, set `n_modes_height: 100`, `n_modes_width: 140`, `hidden_channels: 32`, `n_layers: 4` in the YAML.

> **Why start with fine-tuning?** The Futian pre-trained model has already learned general flood dynamics (terrain channelling, runoff accumulation). Fine-tuning adapts these learned features to UKEA with only 100 epochs, far fewer than training from scratch. This is the recommended path for new users.

> **Why must `ukea_finetune.yaml` keep `n_modes = 100 × 140`?** FNO spectral layers store weight tensors shaped by `n_modes`. Loading a pre-trained region1 checkpoint requires the architecture to be **identical** to the one used during region1 training (100 × 140 modes). Changing `n_modes` would make the weight shapes incompatible.

---

## Scenario A — Fine-tune on UKEA (Recommended)

> Uses `configs/ukea_finetune.yaml` — the default config.

### Step 1 — Download pre-trained weights

See [3. Pre-trained Weights](03-pretrained-weights.md) for download links and placement instructions.

### Step 2 — Edit the config

⚙️ Open `configs/ukea_finetune.yaml` and update **only** the `finetune` block:

```yaml
finetune:
  enabled: True
  pretrained_dir: "../../exp/<expr_id>/weights/<checkpoint_name>"
  state_dict_name: "<checkpoint_name>_state_dict.pt"
```

Replace `<expr_id>` and `<checkpoint_name>` with the actual folder and file names.
**Do not change `n_modes_height` or `n_modes_width`** — they must stay at 100 / 140 to match the pretrained architecture.

### Step 3 — Train

💻 From `code/urbanflood_larfno/`:

```bash
# Linux / AutoDL:
python train.py --config ukea_finetune.yaml 2>&1 | tee train_log.txt

# Windows:
python run_train.py --config ukea_finetune.yaml
```

> **Multi-GPU tip (Linux only):** Set `distributed.use_distributed: True` and `data.batch_size: <your GPU numbers>` in `ukea_finetune.yaml`, then launch with:
> ```bash
> torchrun --nproc_per_node=<your GPU numbers> train.py --config ukea_finetune.yaml 2>&1 | tee train_log.txt
> ```

### Step 4 — Evaluate

```bash
python test.py --config ukea_finetune.yaml --expr_id <timestamp>
```

> The default `eval.locations: "ukea_8m_5min,ukea_2m_5min"` evaluates both the 8 m training resolution and the 2 m zero-shot super-resolution grid simultaneously.

---

## Scenario B — Train UKEA from Scratch

> Uses `configs/ukea_scratch.yaml`. No pre-trained weights required.

### Step 1 — (Optional) Edit the config

⚙️ `configs/ukea_scratch.yaml` is ready to use out of the box. The default architecture is lightweight to reduce training time. You may adjust:

```yaml
tfno2d:
  n_modes_height: 25    # increase up to 50 for better accuracy
  n_modes_width:  60    # increase up to 120 for better accuracy
  hidden_channels: 32
  n_layers: 2           # increase to 4 for better accuracy

opt:
  n_epochs: 1000        # scratch training needs more epochs
  warm_up_iter: 10
  T_max: 1000           # keep T_max = n_epochs
  lr_max: 1e-2
  lr_min: 1e-4
```

> **UKEA has no designated paper accuracy target.** The above defaults are lightweight; increase `n_modes`, `hidden_channels`, and `n_layers` as GPU memory allows.

### Step 2 — Train

```bash
# Linux / AutoDL:
python train.py --config ukea_scratch.yaml 2>&1 | tee train_log.txt

# Windows:
python run_train.py --config ukea_scratch.yaml
```

### Step 3 — Evaluate

```bash
python test.py --config ukea_scratch.yaml --expr_id <timestamp>
```

> The default `eval.locations: "ukea_8m_5min,ukea_2m_5min"` evaluates both the 8 m training resolution and the 2 m zero-shot super-resolution grid simultaneously.

---

## Scenario C — Train Futian / Custom Dataset from Scratch

> Uses `configs/region1_scratch.yaml`. For the Futian dataset or your own large-scale study area.

### Step 1 — Prepare your event lists

✏️ Fill in `configs/region1_fulltrain.txt` (or `region1_smalltrain.txt`) and `configs/region1_test.txt` with your event names (one event name per line), matching the sub-folder names under `benchmark/urbanflood/flood/<location>/`.

### Step 2 — Edit the config

⚙️ Open `configs/region1_scratch.yaml` and update the `data` and `eval` blocks:

```yaml
tfno2d:
  n_modes_height: 40    # default lightweight; increase up to 100 to reproduce paper accuracy
  n_modes_width:  56    # default lightweight; increase up to 140 to reproduce paper accuracy
  hidden_channels: 16   # default lightweight; increase to 32 to reproduce paper accuracy
  n_layers: 2           # default lightweight; increase to 4 to reproduce paper accuracy

opt:
  n_epochs: 1000
  warm_up_iter: 10
  T_max: 1000           # keep T_max = n_epochs
  window_size: 4        # time steps predicted per forward pass

data:
  train_location: "region1_20m"    # or your own folder name
  train_list: "region1_smalltrain.txt"  # 16 events (fast start); use region1_fulltrain.txt for full 64-event training
  test_list:  "region1_test.txt"

eval:
  locations: "region1_20m"         # or your own folder name
```

> **To reproduce paper accuracy on region1:** set `n_modes_height: 100`, `n_modes_width: 140`, `hidden_channels: 32`, `n_layers: 4`.

**For a custom dataset**, also place your data under:

```
benchmark/urbanflood/
├── flood/<your_location>/<event_name>/
│   ├── dem.npy         shape (H, W)
│   ├── rainfall.npy    shape (T, H, W)
│   └── h.npy           shape (T, H, W)
└── geodata/<your_location>/
    └── dem.npy         (same DEM used for visualisation)
```

Then set `train_location: "<your_location>"` in the YAML.

### Step 3 — Train

```bash
# Linux / AutoDL:
python train.py --config region1_scratch.yaml 2>&1 | tee train_log.txt

# Windows:
python run_train.py --config region1_scratch.yaml
```

### Step 4 — Evaluate

```bash
python test.py --config region1_scratch.yaml --expr_id <timestamp>
```

---

Prev: [← 4. Inference, Evaluation & Outputs](04-inference.md) · Next: [6. Cloud GPU — AutoDL →](06-cloud-gpu-autodl.md)
