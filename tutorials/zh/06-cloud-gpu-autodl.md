[← 全部教程](../README.md) · [主页](../../README.md) · [English](../en/06-cloud-gpu-autodl.md) | **中文**

# 6. 云 GPU — AutoDL 指南

如果你没有本地 GPU，可以在 [AutoDL](https://www.autodl.com/) 上以约 ¥1–3/小时 租用一块。下面的指南使用 **基于浏览器的 JupyterLab** — 本地无需安装额外软件。

> ⚠️ **先把数据集下载到本地**（见 [2. 数据集准备](02-datasets.md)），再创建云实例。

---

### 🖥️ 步骤 1 — 创建 GPU 实例

1. 在 [https://www.autodl.com/](https://www.autodl.com/) 注册并登录。
2. 点击 **租用 → GPU 云服务器**。
3. 选择 **≥ 8 GB 显存** 的显卡（例如 RTX 3090 24 GB、RTX 4090 24 GB）。
4. 选择基础镜像：
   - **推荐 RTX 4090：** `PyTorch 2.5.1 → Python 3.12(ubuntu22.04) → CUDA 12.4`
   - 其他显卡：任何 PyTorch ≥ 2.1 且 CUDA 匹配即可 — 我们用 `--no-deps` 安装 neuralop 依赖，因此不会自动升级 PyTorch。
5. 点击 **立即创建** 并等待实例启动。

---

### 🌐 步骤 2 — 打开 JupyterLab

在实例概览页面，点击 **JupyterLab** 按钮。使用 **终端**（Launcher → Terminal）运行 shell 命令。

---

### 💻 步骤 3 — 克隆仓库

```bash
cd /root/autodl-tmp/
git clone https://github.com/holmescao/LarNO
```

---

### 📥 步骤 4 — 通过 SCP 上传数据集

在 AutoDL 实例概览页面找到你的 **SSH 登录命令**，例如：

```
ssh -p 27407 root@connect.westb.seetacloud.com   # 密码显示在页面上
```

> ⚠️ 上面的主机、端口和密码均为 **示例** — 请使用你自己面板上的值。

#### Linux / macOS

```bash
scp -P 27407 /path/to/benchmark.zip root@connect.westb.seetacloud.com:/root/autodl-tmp/
# 大文件夹可用 rsync（中断后可续传）：
rsync -avz --progress -e "ssh -p 27407" /path/to/benchmark/ \
    root@connect.westb.seetacloud.com:/root/autodl-tmp/LarNO/benchmark/
```

#### Windows（PowerShell / Git Bash）

```powershell
scp -P 27407 C:\path\to\benchmark.zip root@connect.westb.seetacloud.com:/root/autodl-tmp/
```

或使用 [WinSCP](https://winscp.net/)，协议选择 SCP。

#### 在云实例上解压

```bash
cd /root/autodl-tmp/
unzip benchmark.zip -d LarNO/
ls LarNO/                               # 检查解压出的文件夹名
mv LarNO/benchmark_upload LarNO/benchmark  # 如有需要则重命名
ls LarNO/benchmark/urbanflood/flood/    # 验证
```

---

### ⚙️ 步骤 5 — 安装依赖

```bash
cd /root/autodl-tmp/LarNO/code/urbanflood_larfno
pip install -e . --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorly tensorly-torch "torch-harmonics==0.7.3" \
    ruamel-yaml configmypy opt-einsum h5py zarr matplotlib \
    "numpy>=1.25" pandas tqdm scipy opencv-python openpyxl torchmetrics \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 📥 步骤 6 — 下载预训练 Futian 权重

| 镜像                        | 链接                                                                                                            |
| --------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Google Drive                | [下载（无密码）](https://drive.google.com/file/d/1ITPoTWQkm5v9kdZT9fqza2Xd4a6Lc-0t/view?usp=drive_link)         |
| 百度网盘（提取码：`LaNO`）  | [下载](https://pan.baidu.com/s/1bJuO5sBdt6kNm5dwOl58WQ?pwd=LaNO)                                                |

通过 SCP 上传到云实例，然后解压：

```bash
scp -P 27407 /path/to/exp.zip root@connect.westb.seetacloud.com:/root/autodl-tmp/
# 在 JupyterLab 终端中：
cd /root/autodl-tmp/ && unzip exp.zip -d LarNO/
ls LarNO/exp/   # 验证检查点目录存在
```

> 💡 若服务器能访问 Google Drive，也可用 `gdown` 直接下载：
> ```bash
> pip install gdown -q
> gdown <file_id> -O /root/autodl-tmp/exp.zip
> cd /root/autodl-tmp/ && unzip exp.zip -d LarNO/
> ```

---

### 🚀 步骤 7 — 先推理，再训练

> 💡 **新用户提示：** 始终先运行推理（test）— 仅需约 3 分钟，可在投入长时间训练之前确认模型、权重和数据集都已正确加载。

#### 7a — 用预训练 Futian 权重快速推理

该步骤约需 **3 分钟**，验证模型、数据集和权重均正确加载。使用 `urbanflood_config_2d.yaml` — 它已内嵌预训练权重路径并已为 region1 配置好。

**首先**，确认 `configs/urbanflood_config_2d.yaml` 中的配置值正确：

```yaml
tfno2d:
  hidden_channels: 32    # 必须与预训练架构一致

data:
  train_location: "region1_20m"
  train_list: "region1_fulltrain.txt"
  test_list: "region1_test.txt"

eval:
  locations: "region1_20m"
```

**然后** 运行推理：

```bash
cd /root/autodl-tmp/LarNO/code/urbanflood_larfno

# 用预训练 Futian 权重在 region1 上运行推理：
python test.py --config urbanflood_config_2d.yaml --expr_id 20260220_183648_006352
```

> 将 `20260220_183648_006352` 替换为你下载的权重的实际实验文件夹名。

结果出现在 `exp/<new_timestamp>/`：
- `test_metrics/region1_20m/` — 每个测试事件的 R²、MAE、CSI 的 Excel
- `visualization/region1_20m/` — PNG 快照 + 动态 GIF

#### 7b — 在 UKEA 上微调（场景 A，推荐）

确认推理可用后，在 UKEA 上微调模型：

```bash
python train.py --config ukea_finetune.yaml 2>&1 | tee train_log.txt
```

在第二个终端标签页中监控进度：

```bash
tail -f /root/autodl-tmp/LarNO/code/urbanflood_larfno/train_log.txt
```

训练完成后（RTX 4090 上 100 个 epoch 约 10 分钟），进行评估：

```bash
# 将 <timestamp> 替换为训练开始时打印的文件夹名：
python test.py --config ukea_finetune.yaml --expr_id <timestamp>
```

#### 7c — 从头训练（场景 B 和 C）

如果你希望不使用预训练权重进行训练：

```bash
# 场景 B — 在 UKEA 上从头训练：
python train.py --config ukea_scratch.yaml 2>&1 | tee train_log.txt

# 场景 C — 在 Futian / 自定义数据集上从头训练：
python train.py --config region1_scratch.yaml 2>&1 | tee train_log.txt
```

训练后评估：

```bash
python test.py --config ukea_scratch.yaml --expr_id <timestamp>
# 或
python test.py --config region1_scratch.yaml --expr_id <timestamp>
```

---

### 📥 步骤 8 — 下载结果

在 JupyterLab 中压缩并下载：

```bash
cd /root/autodl-tmp/LarNO
zip -r exp_results.zip exp/
# 在 JupyterLab 文件浏览器中右键 exp_results.zip → 下载
```

---

> 💡 **费用估算**：12 个测试事件的推理约需 3 分钟。RTX 4090 上完整的 100 epoch 微调约 10 分钟 ≈ ¥2。1000 epoch 的从头训练约 100 分钟。**用完记得关闭实例。**

---

上一篇：[← 5. 训练](05-training.md) · 下一篇：[7. 参考 →](07-reference.md)
