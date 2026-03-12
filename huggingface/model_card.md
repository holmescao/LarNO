---
license: mit
tags:
  - flood-forecasting
  - neural-operator
  - urban-hydrology
  - time-series-prediction
  - zero-shot-super-resolution
  - pytorch
datasets:
  - holmescao/LarNO-dataset
language:
  - en
pipeline_tag: time-series-prediction
---

# LarNO — Latent Autoregressive Neural Operator

**Paper:** Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO
**Journal:** Journal of Hydrology (Under Review)
**GitHub:** [holmescao/LarNO](https://github.com/holmescao/LarNO)
**Colab Demo:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I9TDBCC0rQU3dKMujRCCm8hRMSumGe7E)

---

## Model Description

LarNO is a memory-efficient, discretization-invariant neural operator for large-scale urban flood spatiotemporal forecasting. It learns continuous-space hydrodynamic mappings to predict water depth distributions based on dynamic rainfall and static terrain.

**Key capabilities:**
- **Zero-shot super-resolution**: trained at 20 m resolution, applied at 5 m with no retraining
- **~940× faster** inference than MIKE+ hydraulic solver (with TensorRT)
- **Few-shot transfer** to unseen catchments via fine-tuning
- **Large-scale**: handles ~100 km² at 5 m / 5 min resolution

---

## Performance

Benchmarked on Futian district, Shenzhen (~100 km²) — 5 m resolution, zero-shot super-resolution (trained at 20 m):

| Method | Params | Inference† | Speedup | MAE (m) ↓ | CSI ↑ |
|---|---|---|---|---|---|
| MIKE+ (hydraulic solver) | — | ~8.9 h | 1× | Reference | Reference |
| UNO | 109.1 M | 710 s | ~570× | 0.024 ± 0.007 | 0.343 ± 0.026 |
| FNO | 29.1 M | 760 s | ~530× | 0.019 ± 0.004 | 0.620 ± 0.027 |
| **LarNO (ours)** | **29.1 M** | **34 s** ‡ | **~940×** | **0.008 ± 0.003** | **0.722 ± 0.016** |

> † Single 6-hour event on NVIDIA RTX 4090.
> ‡ LarNO inference uses **TensorRT (TRT)** acceleration; UNO and FNO do not support TRT.
> **Note:** The released dataset is a **20 m downsampled version** for accessibility. Metrics on the released 20 m data will differ from the 5 m paper results above.

---

## Model Architecture

- **Backbone**: TFNO2d (Factorized Fourier Neural Operator)
- **Temporal memory**: CGRU (Convolutional GRU) for latent autoregression
- **Input channels**: 13 (6 rainfall + 6 cumulative-rainfall + 1 DEM)
- **Output**: water depth field (H × W), one time step at a time

**Checkpoint architecture (must match for loading):**

| Parameter | Value |
|---|---|
| `hidden_channels` | 32 |
| `n_modes_height` | 100 |
| `n_modes_width` | 140 |
| `n_layers` | 4 |

---

## Usage

### Quick inference (no local setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I9TDBCC0rQU3dKMujRCCm8hRMSumGe7E)

### Local inference

```bash
git clone https://github.com/holmescao/LarNO
cd LarNO/code/urbanflood_larfno
pip install -e .
pip install -r requirements.txt

# Download weights from this HuggingFace repo or Google Drive
# Place under: LarNO/exp/<expr_id>/weights/<checkpoint_name>/

python test.py --config urbanflood_config_2d.yaml --expr_id <expr_id>
```

See the [GitHub README](https://github.com/holmescao/LarNO) for full installation and training instructions.

---

## Files

| File | Description |
|---|---|
| `exp_weights.zip` | Pre-trained Futian (region1_20m) checkpoint, hidden_ch=32, epoch ~992 |

**Directory structure after unzipping:**
```
exp/
└── <expr_id>/
    └── weights/
        └── <checkpoint_name>/
            └── <checkpoint_name>_state_dict.pt
```

---

## Dataset

The benchmark dataset is available at [holmescao/LarNO-dataset](https://huggingface.co/datasets/holmescao/LarNO-dataset).

| Dataset | Resolution | Area | Events |
|---|---|---|---|
| UKEA (`ukea_8m_5min`) | 8 m / 5 min | ~0.4 km² | 8 train + 12 test |
| Futian (`region1_20m`) | 20 m / 5 min | ~100 km² | 64 train + 16 test |

---

## Citation

```bibtex
@article{larno2025,
  title   = {Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO},
  author  = {[TODO: authors]},
  journal = {Journal of Hydrology},
  year    = {2025},
  doi     = {[TODO: DOI]}
}
```
