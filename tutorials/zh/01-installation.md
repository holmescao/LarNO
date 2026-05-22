[← 全部教程](../README.md) · [主页](../../README.md) · [English](../en/01-installation.md) | **中文**

# 1. 安装

💻 **步骤 1 — 克隆仓库**

```bash
git clone https://github.com/holmescao/LarNO
cd LarNO
```

💻 **步骤 2 — 创建 conda 环境**

```bash
conda create -n larno python=3.9
conda activate larno
```

💻 **步骤 3 — 安装 PyTorch**

安装 **PyTorch ≥ 2.1，CUDA ≥ 11.8**。请根据你的 CUDA 驱动版本，在下面页面选择对应的安装命令：

👉 **[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)**

常见示例：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
或
# CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

💻 **步骤 4 — 安装项目依赖**

```bash
cd code/urbanflood_larfno
pip install -e .
pip install -r requirements.txt
```

> 🇨🇳 国内用户 — 使用清华镜像加速：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

✅ **步骤 5 — 验证完整安装**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

你应当看到类似 `2.6.0+cu126 True` 的输出。

---

下一篇：[2. 数据集准备 →](02-datasets.md)
