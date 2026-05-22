[← All tutorials](../README.md) · [Home](../../README.md) · **English** | [中文](../zh/06-cloud-gpu-autodl.md)

# 6. Cloud GPU — AutoDL Guide

If you do not have a local GPU, you can rent one from [AutoDL](https://www.autodl.com/) for approximately ¥1–3 per hour. The guide below uses the **browser-based JupyterLab** — no extra software needed on your local machine.

> ⚠️ **Download the dataset to your local machine first** (see [2. Dataset Preparation](02-datasets.md)) before creating a cloud instance.

---

### 🖥️ Step 1 — Create a GPU instance

1. Register and log in at [https://www.autodl.com/](https://www.autodl.com/).
2. Click **租用 → GPU 云服务器**.
3. Choose a card with **≥ 8 GB VRAM** (e.g., RTX 3090 24 GB, RTX 4090 24 GB).
4. Select the base image:
   - **RTX 4090 recommended:** `PyTorch 2.5.1 → Python 3.12(ubuntu22.04) → CUDA 12.4`
   - Other cards: any PyTorch ≥ 2.1 with matching CUDA is fine — we install neuralop dependencies with `--no-deps` so PyTorch is not upgraded automatically.
5. Click **立即创建** and wait for the instance to start.

---

### 🌐 Step 2 — Open JupyterLab

On the instance overview page, click the **JupyterLab** button. Use the **terminal** (Launcher → Terminal) to run shell commands.

---

### 💻 Step 3 — Clone the repository

```bash
cd /root/autodl-tmp/
git clone https://github.com/holmescao/LarNO
```

---

### 📥 Step 4 — Upload the dataset via SCP

Find your **SSH login command** on the AutoDL instance overview page, e.g.:

```
ssh -p 27407 root@connect.westb.seetacloud.com   # password shown on the page
```

> ⚠️ The host, port, and password above are **examples** — use your own dashboard values.

#### Linux / macOS

```bash
scp -P 27407 /path/to/benchmark.zip root@connect.westb.seetacloud.com:/root/autodl-tmp/
# or rsync for large folders (resumes on failure):
rsync -avz --progress -e "ssh -p 27407" /path/to/benchmark/ \
    root@connect.westb.seetacloud.com:/root/autodl-tmp/LarNO/benchmark/
```

#### Windows (PowerShell / Git Bash)

```powershell
scp -P 27407 C:\path\to\benchmark.zip root@connect.westb.seetacloud.com:/root/autodl-tmp/
```

Or use [WinSCP](https://winscp.net/) with Protocol = SCP.

#### Unzip on the cloud instance

```bash
cd /root/autodl-tmp/
unzip benchmark.zip -d LarNO/
ls LarNO/                               # check the extracted folder name
mv LarNO/benchmark_upload LarNO/benchmark  # rename if needed
ls LarNO/benchmark/urbanflood/flood/    # verify
```

---

### ⚙️ Step 5 — Install dependencies

```bash
cd /root/autodl-tmp/LarNO/code/urbanflood_larfno
pip install -e . --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorly tensorly-torch "torch-harmonics==0.7.3" \
    ruamel-yaml configmypy opt-einsum h5py zarr matplotlib \
    "numpy>=1.25" pandas tqdm scipy opencv-python openpyxl torchmetrics \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 📥 Step 6 — Download pre-trained Futian weights

| Mirror                     | Link                                                                                                            |
| -------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Google Drive               | [Download (no password)](https://drive.google.com/file/d/1ITPoTWQkm5v9kdZT9fqza2Xd4a6Lc-0t/view?usp=drive_link) |
| Baidu Cloud (code: `LaNO`) | [Download](https://pan.baidu.com/s/1bJuO5sBdt6kNm5dwOl58WQ?pwd=LaNO)                                            |

Upload to the cloud instance via SCP, then unzip:

```bash
scp -P 27407 /path/to/exp.zip root@connect.westb.seetacloud.com:/root/autodl-tmp/
# In JupyterLab terminal:
cd /root/autodl-tmp/ && unzip exp.zip -d LarNO/
ls LarNO/exp/   # verify the checkpoint directory exists
```

> 💡 Or download directly on the server with `gdown` if Google Drive is reachable:
> ```bash
> pip install gdown -q
> gdown <file_id> -O /root/autodl-tmp/exp.zip
> cd /root/autodl-tmp/ && unzip exp.zip -d LarNO/
> ```

---

### 🚀 Step 7 — Run inference first, then train

> 💡 **New user tip:** Always run inference (test) first — it only takes ~3 minutes and confirms that the model, weights, and dataset are all loaded correctly before you commit to a long training run.

#### 7a — Quick inference with pre-trained Futian weights

This step takes about **3 minutes** and verifies that the model, dataset, and weights are all loaded correctly. Use `urbanflood_config_2d.yaml` — it already has the pre-trained weight path embedded and is configured for region1.

**First**, confirm the config values are correct in `configs/urbanflood_config_2d.yaml`:

```yaml
tfno2d:
  hidden_channels: 32    # must match pre-trained architecture

data:
  train_location: "region1_20m"
  train_list: "region1_fulltrain.txt"
  test_list: "region1_test.txt"

eval:
  locations: "region1_20m"
```

**Then** run inference:

```bash
cd /root/autodl-tmp/LarNO/code/urbanflood_larfno

# Run inference on region1 with pre-trained Futian weights:
python test.py --config urbanflood_config_2d.yaml --expr_id 20260220_183648_006352
```

> Replace `20260220_183648_006352` with the actual experiment folder name from your downloaded weights.

Results appear in `exp/<new_timestamp>/`:
- `test_metrics/region1_20m/` — Excel with R², MAE, CSI per test event
- `visualization/region1_20m/` — PNG snapshots + animated GIFs

#### 7b — Fine-tune on UKEA (Scenario A, Recommended)

After confirming inference works, fine-tune the model on UKEA:

```bash
python train.py --config ukea_finetune.yaml 2>&1 | tee train_log.txt
```

Monitor progress in a second terminal tab:

```bash
tail -f /root/autodl-tmp/LarNO/code/urbanflood_larfno/train_log.txt
```

After training finishes (about 10 minutes for 100 epochs on an RTX 4090), evaluate:

```bash
# Replace <timestamp> with the folder name printed at training start:
python test.py --config ukea_finetune.yaml --expr_id <timestamp>
```

#### 7c — Train from scratch (Scenarios B and C)

If you prefer to train without pre-trained weights:

```bash
# Scenario B — Train UKEA from scratch:
python train.py --config ukea_scratch.yaml 2>&1 | tee train_log.txt

# Scenario C — Train Futian / custom dataset from scratch:
python train.py --config region1_scratch.yaml 2>&1 | tee train_log.txt
```

Evaluate after training:

```bash
python test.py --config ukea_scratch.yaml --expr_id <timestamp>
# or
python test.py --config region1_scratch.yaml --expr_id <timestamp>
```

---

### 📥 Step 8 — Download results

Compress and download from JupyterLab:

```bash
cd /root/autodl-tmp/LarNO
zip -r exp_results.zip exp/
# Right-click exp_results.zip in JupyterLab file browser → Download
```

---

> 💡 **Cost estimate**: inference on 12 test events takes ~3 minutes. A full 100-epoch fine-tuning run on an RTX 4090 takes about 10 minutes ≈ ¥2. A 1000-epoch scratch run takes about 100 minutes. **Remember to shut down the instance when done.**

---

Prev: [← 5. Training](05-training.md) · Next: [7. Reference →](07-reference.md)
