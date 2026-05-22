[← All tutorials](../README.md) · [Home](../../README.md) · **English** | [中文](../zh/08-faq.md)

# 8. FAQ

**Q: Which scenario should I start with?**

A: **Quick Test first** ([4. Inference](04-inference.md)) — run inference on region1 in 3 minutes using `urbanflood_config_2d.yaml` to confirm everything works. Then try **Scenario A (`ukea_finetune.yaml`)** — fine-tuning from the Futian pre-trained model converges in 100 epochs and gives good results on UKEA. Only move to Scenario B (scratch) if you want to explore UKEA-specific architectures.

**Q: Why does `ukea_finetune.yaml` use `n_modes = 100 × 140` when the UKEA grid is only 50 × 120?**

A: The FNO spectral layer weight tensors are sized by `n_modes`. To load the Futian pre-trained checkpoint, the architecture must be identical to the one used during Futian training (100 × 140 modes). The FNO implementation automatically clips to the available FFT modes at runtime, so no errors occur — the extra weight parameters simply aren't used for the smaller UKEA grid.

**Q: My GPU has only 6 GB VRAM. Can I still train?**

A: Yes, on UKEA. Use `ukea_scratch.yaml` — its default lightweight architecture (12×30 modes, 16 channels, 2 layers) fits in 6 GB. For Futian (400×560 grid), you likely need ≥ 8 GB VRAM. Consider renting a cloud GPU ([6. Cloud GPU — AutoDL](06-cloud-gpu-autodl.md)).

**Q: How do I add my own study area?**

A: See [Scenario C in 5. Training](05-training.md#scenario-c--train-futian--custom-dataset-from-scratch). Prepare `dem.npy`, `rainfall.npy`, `h.npy` per event, place them under `benchmark/urbanflood/flood/<your_location>/`, copy `region1_scratch.yaml` and update `train_location`, `train_list`, `test_list`, and `eval.locations`.

**Q: Can I use output from a different hydraulic solver (e.g., SWMM, HEC-RAS)?**

A: Yes — save water depth as `h.npy` with shape `(T, H, W)` in metres and the data loader works without modification.

**Q: Training is slow. How can I speed it up?**

A: Several options, from easiest to most impactful:

1. **Reduce model size** — in the YAML, decrease any of:
   - `n_modes_height` / `n_modes_width` — fewer Fourier modes = faster FFT
   - `hidden_channels` — fewer internal features (e.g., 16 instead of 32)
   - `n_layers` — fewer FNO blocks (e.g., 2 instead of 4)

   The default scratch configs already use a lightweight setting (`12×30`, `ch=16`, `layers=2` for UKEA; `40×56`, `ch=16`, `layers=2` for region1). To restore paper accuracy on region1, set `n_modes_height=100`, `n_modes_width=140`, `hidden_channels=32`, `n_layers=4`.

2. **Enable faster data loading** — on Linux, set `num_workers_train: 4` in the YAML.

3. **Enable mixed precision** — set `amp_autocast: True` for FP16 training.

4. **Use multi-GPU DDP** — on Linux with multiple GPUs:
   ```bash
   torchrun --nproc_per_node=4 train.py --config region1_scratch.yaml 2>&1 | tee train_log.txt
   ```

**Q: Can I run multi-GPU training on Windows?**

A: Not easily — NCCL is not supported on Windows. Use a Linux cloud instance ([6. Cloud GPU — AutoDL](06-cloud-gpu-autodl.md)).

**Q: What does `wall_height: 50` mean?**

A: `NaN` pixels in the DEM (buildings, domain boundaries) are replaced with 50 m elevation so water cannot flow through them. This value must exceed the maximum real terrain elevation in your study area.

**Q: The event GIF is too slow / too fast.**

A: Edit the `duration` parameter in `neuralop/training/trainer.py` → `save_comparison_gif()`. Default is `100` ms per frame (= 10 fps). Increase to slow down (e.g., `500` = 2 fps).

**Q: I don't see the images in the README on GitHub.**

A: The images are tracked in git under `code/urbanflood_larfno/assets/`. If they are missing after a fresh clone, run `git lfs pull` (if LFS is configured) or verify with `git ls-files code/urbanflood_larfno/assets/`.

---

Prev: [← 7. Reference](07-reference.md) · [All tutorials](../README.md)
