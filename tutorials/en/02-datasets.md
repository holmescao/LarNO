[← All tutorials](../README.md) · [Home](../../README.md) · **English** | [中文](../zh/02-datasets.md)

# 2. Dataset Preparation

LarNO is evaluated on two benchmark datasets. **We recommend starting with the small UKEA case** to verify that your installation works, before moving on to the large Futian case.

## UKEA small case (`ukea_8m_5min`) — start here ✅

| Property                   | Value                                                        |
| -------------------------- | ------------------------------------------------------------ |
| Area                       | ~0.4 km² (small coastal catchment, UK Environment Agency)    |
| Grid (train)               | 50 × 120 at **8 m** resolution                               |
| Grid (test)                | 200 × 480 at **2 m** resolution (zero-shot super-resolution) |
| Zero-shot super-resolution | **8 m → 2 m** (4× finer, no retraining)                      |
| Training events            | 8                                                            |
| Test events                | 12                                                           |

## Futian large case (`region1_20m`) — for further research 🔬

| Property        | Value                                       |
| --------------- | ------------------------------------------- |
| Area            | ~100 km² (Futian district, Shenzhen, China) |
| Grid            | 400 × 560 at **20 m** resolution            |
| Training events | 64 (full) / 16 (small subset)               |
| Test events     | 16                                          |

## Download links

| Mirror       | Link                                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------------ |
| figshare     | [10.6084/m9.figshare.30529031.v4](https://doi.org/10.6084/m9.figshare.30529031.v4)                           |
| Google Drive | [Download (no password)](https://drive.google.com/file/d/13VRExXwoFznTLIQKApn5O0_fxKsi8ThI/view?usp=sharing) |
| HuggingFace  | [holmescao.github.io/datasets/LarNO](https://holmescao.github.io/datasets/LarNO)                             |

📁 Unzip and place data so the directory tree looks like:

```
LarNO/
├── benchmark/urbanflood/
│   ├── flood/
│   │   ├── ukea_8m_5min/    ← UKEA events 8m (one sub-folder per event)
│   │   ├── ukea_2m_5min/    ← UKEA events 2m
│   │   └── region1_20m/     ← Futian events
│   └── geodata/
│       ├── ukea_8m_5min/
│       ├── ukea_2m_5min/
│       └── region1_20m/
└── code/urbanflood_larfno/
```

## File format

| File           | Shape       | Unit       | Description                                 |
| -------------- | ----------- | ---------- | ------------------------------------------- |
| `dem.npy`      | `(H, W)`    | metres     | Digital Elevation Model. `NaN` = buildings. |
| `rainfall.npy` | `(T, H, W)` | mm / 5 min | Rainfall intensity per 5-minute step.       |
| `h.npy`        | `(T, H, W)` | metres     | Ground-truth water depth from **MIKE+**.    |

| Location       | H   | W   | T   | Duration          |
| -------------- | --- | --- | --- | ----------------- |
| `ukea_8m_5min` | 50  | 120 | 36  | 3 h (5-min steps) |
| `ukea_2m_5min` | 200 | 480 | 36  | 3 h (5-min steps) |
| `region1_20m`  | 400 | 560 | 72  | 6 h (5-min steps) |

> **Why does UKEA have T=36 while Futian has T=72?** The UKEA events cover a 3-hour simulation window; the Futian events cover 6 hours. Both use 5-minute time steps.

## Event lists

Edit the plain text files in `configs/` to control train/test splits:

```
configs/
├── ukea_train.txt      ← 8 UKEA training events
├── ukea_test.txt       ← 12 UKEA test events
├── region1_fulltrain.txt   ← 64 Futian training events
├── region1_smalltrain.txt   ← 16 Futian training events
└── region1_test.txt    ← 16 Futian test events
```

---

Prev: [← 1. Installation](01-installation.md) · Next: [3. Pre-trained Weights →](03-pretrained-weights.md)
