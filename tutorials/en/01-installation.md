[← All tutorials](../README.md) · [Home](../../README.md) · **English** | [中文](../zh/01-installation.md)

# 1. Installation

💻 **Step 1 — Clone the repository**

```bash
git clone https://github.com/holmescao/LarNO
cd LarNO
```

💻 **Step 2 — Create a conda environment**

```bash
conda create -n larno python=3.9
conda activate larno
```

💻 **Step 3 — Install PyTorch**

Install **PyTorch ≥ 2.1 with CUDA ≥ 11.8**. Select the command matching your CUDA driver version at:

👉 **[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)**

Common examples:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
or
# CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

💻 **Step 4 — Install project dependencies**

```bash
cd code/urbanflood_larfno
pip install -e .
pip install -r requirements.txt
```

> 🇨🇳 China users — use the Tsinghua mirror:
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

✅ **Step 5 — Verify the full installation**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You should see something like `2.6.0+cu126 True`.

---

Next: [2. Dataset Preparation →](02-datasets.md)
