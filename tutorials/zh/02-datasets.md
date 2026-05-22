[← 全部教程](../README.md) · [主页](../../README.md) · [English](../en/02-datasets.md) | **中文**

# 2. 数据集准备

LarNO 在两个基准数据集上进行评估。**我们建议先从小规模的 UKEA 案例开始**，验证安装正确无误后，再处理大规模的 Futian 案例。

## UKEA 小案例（`ukea_8m_5min`）— 从这里开始 ✅

| 属性                 | 取值                                                   |
| -------------------- | ------------------------------------------------------ |
| 面积                 | ~0.4 km²（英国环境署的小型滨海汇水区）                 |
| 网格（训练）         | 50 × 120，**8 m** 分辨率                               |
| 网格（测试）         | 200 × 480，**2 m** 分辨率（零样本超分辨率）            |
| 零样本超分辨率       | **8 m → 2 m**（细化 4 倍，无需重新训练）              |
| 训练事件数           | 8                                                      |
| 测试事件数           | 12                                                     |

## Futian 大案例（`region1_20m`）— 用于进一步研究 🔬

| 属性         | 取值                                  |
| ------------ | ------------------------------------- |
| 面积         | ~100 km²（中国深圳福田区）            |
| 网格         | 400 × 560，**20 m** 分辨率           |
| 训练事件数   | 64（完整）/ 16（小子集）             |
| 测试事件数   | 16                                    |

## 下载链接

| 镜像         | 链接                                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------------------ |
| figshare     | [10.6084/m9.figshare.30529031.v4](https://doi.org/10.6084/m9.figshare.30529031.v4)                           |
| Google Drive | [下载（无密码）](https://drive.google.com/file/d/13VRExXwoFznTLIQKApn5O0_fxKsi8ThI/view?usp=sharing)         |
| HuggingFace  | [holmescao.github.io/datasets/LarNO](https://holmescao.github.io/datasets/LarNO)                             |

📁 解压并放置数据，使目录结构如下：

```
LarNO/
├── benchmark/urbanflood/
│   ├── flood/
│   │   ├── ukea_8m_5min/    ← UKEA 8m 事件（每个事件一个子文件夹）
│   │   ├── ukea_2m_5min/    ← UKEA 2m 事件
│   │   └── region1_20m/     ← Futian 事件
│   └── geodata/
│       ├── ukea_8m_5min/
│       ├── ukea_2m_5min/
│       └── region1_20m/
└── code/urbanflood_larfno/
```

## 文件格式

| 文件           | 形状        | 单位       | 说明                                       |
| -------------- | ----------- | ---------- | ------------------------------------------ |
| `dem.npy`      | `(H, W)`    | 米         | 数字高程模型。`NaN` = 建筑物。            |
| `rainfall.npy` | `(T, H, W)` | mm / 5 min | 每 5 分钟时间步的降雨强度。                |
| `h.npy`        | `(T, H, W)` | 米         | 来自 **MIKE+** 的真值水深。               |

| 位置           | H   | W   | T   | 时长              |
| -------------- | --- | --- | --- | ----------------- |
| `ukea_8m_5min` | 50  | 120 | 36  | 3 小时（5 分钟步长） |
| `ukea_2m_5min` | 200 | 480 | 36  | 3 小时（5 分钟步长） |
| `region1_20m`  | 400 | 560 | 72  | 6 小时（5 分钟步长） |

> **为什么 UKEA 是 T=36 而 Futian 是 T=72？** UKEA 事件覆盖 3 小时模拟窗口；Futian 事件覆盖 6 小时。两者都使用 5 分钟时间步。

## 事件列表

编辑 `configs/` 中的纯文本文件以控制训练/测试划分：

```
configs/
├── ukea_train.txt      ← 8 个 UKEA 训练事件
├── ukea_test.txt       ← 12 个 UKEA 测试事件
├── region1_fulltrain.txt   ← 64 个 Futian 训练事件
├── region1_smalltrain.txt   ← 16 个 Futian 训练事件
└── region1_test.txt    ← 16 个 Futian 测试事件
```

---

上一篇：[← 1. 安装](01-installation.md) · 下一篇：[3. 预训练权重 →](03-pretrained-weights.md)
