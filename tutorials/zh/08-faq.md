[← 全部教程](../README.md) · [主页](../../README.md) · [English](../en/08-faq.md) | **中文**

# 8. 常见问题

**问：我应该从哪个场景开始？**

答：**先做快速测试**（[4. 推理](04-inference.md)）— 用 `urbanflood_config_2d.yaml` 在 region1 上运行 3 分钟推理，确认一切正常。然后尝试 **场景 A（`ukea_finetune.yaml`）** — 从 Futian 预训练模型微调，100 个 epoch 即可收敛，并在 UKEA 上给出不错的结果。只有当你想探索 UKEA 专属架构时，才需要转到场景 B（从头训练）。

**问：为什么 UKEA 网格只有 50 × 120，`ukea_finetune.yaml` 却用 `n_modes = 100 × 140`？**

答：FNO 谱层的权重张量大小由 `n_modes` 决定。要加载 Futian 预训练检查点，架构必须与 Futian 训练时所用一致（100 × 140 模式）。FNO 实现会在运行时自动裁剪到可用的 FFT 模式，因此不会报错 — 多出的权重参数对较小的 UKEA 网格只是没有被用到而已。

**问：我的 GPU 只有 6 GB 显存。还能训练吗？**

答：可以，在 UKEA 上。使用 `ukea_scratch.yaml` — 其默认轻量架构（12×30 模式、16 通道、2 层）可在 6 GB 内运行。对于 Futian（400×560 网格），你可能需要 ≥ 8 GB 显存。可考虑租用云 GPU（[6. 云 GPU — AutoDL](06-cloud-gpu-autodl.md)）。

**问：如何添加我自己的研究区域？**

答：见 [5. 训练中的场景 C](05-training.md#场景-c--在-futian--自定义数据集上从头训练)。为每个事件准备 `dem.npy`、`rainfall.npy`、`h.npy`，放置在 `benchmark/urbanflood/flood/<your_location>/` 下，复制 `region1_scratch.yaml` 并更新 `train_location`、`train_list`、`test_list` 和 `eval.locations`。

**问：我能用其他水力求解器（如 SWMM、HEC-RAS）的输出吗？**

答：可以 — 将水深保存为形状 `(T, H, W)`、单位为米的 `h.npy`，数据加载器无需修改即可工作。

**问：训练太慢。如何加速？**

答：有以下几种方式，由易到难：

1. **缩小模型** — 在 YAML 中减小以下任意项：
   - `n_modes_height` / `n_modes_width` — 更少的傅里叶模式 = 更快的 FFT
   - `hidden_channels` — 更少的内部特征（如 16 而非 32）
   - `n_layers` — 更少的 FNO 块（如 2 而非 4）

   默认的 scratch 配置已使用轻量设置（UKEA 为 `12×30`、`ch=16`、`layers=2`；region1 为 `40×56`、`ch=16`、`layers=2`）。要恢复 region1 的论文精度，设置 `n_modes_height=100`、`n_modes_width=140`、`hidden_channels=32`、`n_layers=4`。

2. **加快数据加载** — 在 Linux 上，在 YAML 中设置 `num_workers_train: 4`。

3. **启用混合精度** — 设置 `amp_autocast: True` 以进行 FP16 训练。

4. **使用多卡 DDP** — 在带多块 GPU 的 Linux 上：
   ```bash
   torchrun --nproc_per_node=4 train.py --config region1_scratch.yaml 2>&1 | tee train_log.txt
   ```

**问：能在 Windows 上进行多卡训练吗？**

答：不太容易 — Windows 不支持 NCCL。请使用 Linux 云实例（[6. 云 GPU — AutoDL](06-cloud-gpu-autodl.md)）。

**问：`wall_height: 50` 是什么意思？**

答：DEM 中的 `NaN` 像素（建筑物、域边界）会被替换为 50 m 高程，使水流无法穿过。该值必须超过你研究区域内真实地形的最高高程。

**问：事件 GIF 太慢 / 太快。**

答：编辑 `neuralop/training/trainer.py` 中 `save_comparison_gif()` 的 `duration` 参数。默认是每帧 `100` ms（= 10 fps）。增大可放慢（如 `500` = 2 fps）。

**问：我在 GitHub 上看不到 README 里的图片。**

答：图片以 git 跟踪在 `code/urbanflood_larfno/assets/` 下。如果在全新克隆后缺失，请运行 `git lfs pull`（若配置了 LFS），或用 `git ls-files code/urbanflood_larfno/assets/` 验证。

---

上一篇：[← 7. 参考](07-reference.md) · [全部教程](../README.md)
