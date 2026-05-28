# [(Journal of Hydrology 2026) Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO](https://authors.elsevier.com/c/1nAnn52cumnSP)

<p align="center">
  <a href="https://authors.elsevier.com/c/1nAnn52cumnSP"><img src="https://img.shields.io/badge/Journal%20of%20Hydrology-Published-blue" alt="Journal of Hydrology"></a>
  <a href="https://holmescao.github.io/datasets/LarNO"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-LarNO-orange" alt="HuggingFace Dataset"></a>
  <a href="https://colab.research.google.com/drive/1I9TDBCC0rQU3dKMujRCCm8hRMSumGe7E"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
  <a href="https://github.com/holmescao/U-RNN"><img src="https://img.shields.io/badge/Related%20Work-U--RNN-brightgreen" alt="U-RNN"></a>
</p>

<p align="center"><strong>English</strong> | <a href="README_CN.md">中文</a></p>

---

<p align="center">
  <img src="code/urbanflood_larfno/assets/demo_event.gif" width="820"/>
  <br><em><strong>LarNO vs. MIKE+ reference — animated water depth comparison (5-min steps, 6-hour flood event, Futian district Shenzhen, ~100 km²).</strong> Left: MIKE+ hydraulic solver (reference). Right: LarNO prediction. LarNO delivers ~940× faster inference than MIKE+ with mm-level depth accuracy (20 m resolution, released dataset).</em>
</p>

**LarNO** models large-scale urban flooding with a **latent autoregressive neural operator** — predicting water-depth maps over millions of grid cells (~100 km²) at **O(mm)-level accuracy** and **~940× faster** than the MIKE+ hydraulic solver. Trained at low resolution, it generalizes **zero-shot to higher resolutions** (e.g. 8 m → 2 m, 20 m → 5 m) without retraining, and transfers to unseen catchments via few-shot fine-tuning.

> 🧪 **Want to try LarNO instantly — no installation needed?**
> Open our **[Google Colab notebook](https://colab.research.google.com/drive/1I9TDBCC0rQU3dKMujRCCm8hRMSumGe7E)** to run inference on the Futian flood dataset with pre-trained weights in ~15 minutes, entirely in your browser.
>
> 📚 **Want to reproduce the paper?** The complete step-by-step tutorials live in **[`tutorials/`](tutorials/README.md)** (English & 中文).

## Highlights

- LarNO introduces a latent autoregressive neural operator for **zero-shot**, **high-resolution spatiotemporal generalization** in urban flood modeling.
- Latent autoregression improves the representation of nonlinear spatiotemporal flood dynamics.
- LarNO enables large-scale flood forecasting over **millions of grid cells** (~100 km2) and **sub-billion spatiotemporal points** (5 m and 5 min resolution).
- LarNO achieves **O(mm)-level water-depth accuracy**.
- LarNO supports **few-shot transfers** to unseen catchments via **fine-tuning**.
- LarNO supports **multi-GPU distributed training** and **TensorRT-accelerated inference**.

## News

- **[15/05/2026]** 🎉🎉🎉[LarNO](https://authors.elsevier.com/c/1nAnn52cumnSP) is now online in **Journal of Hydrology**! Free access is available until **July 17, 2026**.
- **[12/03/2026]** Pre-trained weights and benchmark dataset published on **[HuggingFace](https://huggingface.co/holmescao/LarNO)** — download without Google Drive or Baidu Cloud.
- **[12/03/2026]** Interactive demo released — run LarNO inference in your browser with **[Google Colab](https://colab.research.google.com/drive/1I9TDBCC0rQU3dKMujRCCm8hRMSumGe7E)**, no installation needed.
- **[02/03/2026]** Full end-to-end reproduction tutorial released — train and test LarNO with **a single GPU on [AutoDL](https://www.autodl.com/)**.
- **[02/03/2026]** Code released on **[GitHub](https://github.com/holmescao/LarNO)** and benchmark **[dataset](https://holmescao.github.io/datasets/LarNO)** publicly released.

---

<p align="center">
  <img src="code/urbanflood_larfno/assets/overview.png" width="820"/>
  <br><em><strong>Memory-efficient neural operator training and zero-shot generalization to high resolutions for urban flood spatiotemporal forecasting.</strong>
    (a) Overview: a neural operator is trained at low resolution and applied zero-shot at higher resolutions by directly evaluating the learned continuous operator on finer grids.
    (b) Training stage: discretized inputs (rainfall, terrain, drain inlet) are fed to the neural operator, with outputs supervised by a numerical solver (MIKE Plus).
    (c) Zero-shot application of the trained operator at higher spatial resolutions without any retraining.</em>
</p>

<p align="center">
  <img src="code/urbanflood_larfno/assets/architecture.png" width="820"/>
  <br><em><strong>LarNO architecture for urban flood spatiotemporal forecasting.</strong>
    The model comprises three stages: (1) a lifting layer maps the input to a higher-dimensional hidden state; (2) N LarNO layers iteratively update the hidden state — each layer first applies a GRU-based convolutional update combining the previous time-step and previous-layer hidden states, then refines the state via frequency-domain Fourier transforms (forward FFT, low-mode linear mixing, inverse FFT) and a local time-domain linear operator; (3) a projection layer maps the final hidden state to the output water depth.</em>
</p>

## Quick Start

<div align="center">

| | |
|:---:|:---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I9TDBCC0rQU3dKMujRCCm8hRMSumGe7E) | **No local GPU? Try in your browser.** The Colab notebook runs inference on the Futian dataset with pre-trained weights in ~15 min — no installation, no dataset download. |

</div>

Or run locally in **3 steps** (full setup in the [tutorials](tutorials/README.md)):

```bash
# 1. Clone & install (details: tutorials/en/01-installation.md)
git clone https://github.com/holmescao/LarNO && cd LarNO/code/urbanflood_larfno
pip install -e . && pip install -r requirements.txt

# 2. Download the dataset (tutorials/en/02) and pre-trained weights (tutorials/en/03), then:

# 3. Run inference — results in exp/<timestamp>/
python test.py --config urbanflood_config_2d.yaml --expr_id 20260220_183648_006352
```

## Performance

### Futian benchmark

Comparison on the Futian district (~100 km², Shenzhen) benchmark — **5 m resolution, zero-shot super-resolution** (trained at 20 m, tested at 5 m). Results from Table 1 of the paper.

| Method                   | Params     | Inference† | Speedup vs MIKE+ | MAE (m) ↓         | CSI ↑             |
| ------------------------ | ---------- | ---------- | ---------------- | ----------------- | ----------------- |
| MIKE+ (hydraulic solver) | —          | ~8.9 h     | 1×               | Reference         | Reference         |
| UNO                      | 109.1 M    | 45 s       | ~710×            | 0.024 ± 0.007     | 0.343 ± 0.026     |
| FNO                      | 29.1 M     | 42 s       | ~760×            | 0.019 ± 0.004     | 0.620 ± 0.027     |
| **LarNO (ours)**         | **29.1 M** | **34 s** ‡ | **~940×**        | **0.008 ± 0.003** | **0.722 ± 0.016** |

> † Inference time for a single 6-hour flood event on NVIDIA RTX 4090.
> ‡ LarNO inference uses **TensorRT (TRT)** acceleration; UNO and FNO do not support TRT.
> **Note:** The released dataset is a **20 m downsampled version** for accessibility. Metrics on the released 20 m data will differ from the 5 m paper results above.

### UKEA benchmark (released dataset)

LarNO fine-tuned from Futian pre-trained weights on an unseen region (UKEA small case, `ukea_8m_5min`, 8 train events, 100 epochs). Evaluated on 12 test events.

| Resolution | Setting                         | R² ↑              | MAE (m) ↓           | CSI ↑             | PeakR² ↑      |
| ---------- | ------------------------------- | ----------------- | ------------------- | ----------------- | ------------- |
| **8 m**    | Fine-tune (train resolution)    | 0.948 ± 0.056     | 0.0093 ± 0.0074     | 0.741 ± 0.030     | 0.949 ± 0.049 |
| **2 m**    | Zero-shot super-resolution (4×) | 0.776 ± 0.129     | 0.0163 ± 0.0101     | 0.515 ± 0.052     | 0.820 ± 0.101 |

> Zero-shot 2 m results use the model trained at 8 m — no 2 m data seen during training.

## 📚 Documentation

Full reproduction tutorials live in **[`tutorials/`](tutorials/README.md)** — available in **English** and **中文**. Recommended order: Setup (1 → 3) → Inference (4) → Training (5).

| I want to… | Guide |
|---|---|
| Install the environment | [1. Installation](tutorials/en/01-installation.md) |
| Get the datasets | [2. Dataset Preparation](tutorials/en/02-datasets.md) |
| Get pre-trained weights | [3. Pre-trained Weights](tutorials/en/03-pretrained-weights.md) |
| Run inference & read outputs | [4. Inference, Evaluation & Outputs](tutorials/en/04-inference.md) |
| Train (fine-tune / scratch) | [5. Training](tutorials/en/05-training.md) |
| Use a rented cloud GPU | [6. Cloud GPU — AutoDL](tutorials/en/06-cloud-gpu-autodl.md) |
| Look up config, layout & outputs | [7. Reference](tutorials/en/07-reference.md) |
| Troubleshoot | [8. FAQ](tutorials/en/08-faq.md) |

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use LarNO in your research, please cite the Journal of Hydrology article (DOI: [10.1016/j.jhydrol.2026.135686](https://doi.org/10.1016/j.jhydrol.2026.135686)):

```bibtex
@article{cao2026large,
  title={Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO},
  author={Cao, Xiaoyan and Yao, Yao and Wang, Zhi and Zhao, Zhangxinyue and Borthwick, Alistair GL and Qin, Huapeng},
  journal={Journal of Hydrology},
  pages={135686},
  year={2026},
  doi={10.1016/j.jhydrol.2026.135686},
  publisher={Elsevier}
}
```

If you use the released benchmark dataset, please also cite:

```bibtex
@article{cao2025bench,
author = {Cao, Xiaoyan and Qin, Huapeng},
title = {Benchmark dataset of ``Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO''},
year = {2025},
url = "https://figshare.com/articles/dataset/Benchmark_dataset_of_Large-scale_urban_flood_modeling_and_zero-shot_high-resolution_generalization_with_LarNO_/30529031",
doi = {10.6084/m9.figshare.30529031.v4}
}
```

If your work involves high-resolution spatiotemporal nowcasting of urban flooding, you may also be interested in our related work **[U-RNN](https://github.com/holmescao/U-RNN)**, which focuses on urban flood nowcasting at high spatial-temporal resolution and was published in *Journal of Hydrology*:

```bibtex
@article{cao2025u,
  title={U-RNN high-resolution spatiotemporal nowcasting of urban flooding},
  author={Cao, Xiaoyan and Wang, Baoying and Yao, Yao and Zhang, Lin and Xing, Yanwen and Mao, Junqi and Zhang, Runqiao and Fu, Guangtao and Borthwick, Alistair GL and Qin, Huapeng},
  journal={Journal of Hydrology},
  pages={133117},
  year={2025},
  publisher={Elsevier}
}
```
