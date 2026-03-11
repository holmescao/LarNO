import os
import sys
import builtins
_builtin_open = builtins.open
def _utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None, **kw):
    if encoding is None and 'b' not in mode:
        encoding = 'utf-8'
    return _builtin_open(file, mode, buffering, encoding=encoding, errors=errors, **kw)
builtins.open = _utf8_open
import re
import argparse
import time
from pathlib import Path

# Ensure project root is first on sys.path so local modules are found correctly
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import pandas as pd
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

from neuralop import H1Loss, LpLoss, get_model
from neuralop.training.trainer import Trainer
from neuralop.data.datasets.Dynamic2DFlood import Dynamic2DFlood
from collections import OrderedDict

from utils.torch_utils import select_device


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default='ukea_finetune.yaml',
                        help='YAML config file in configs/ (default: ukea_finetune.yaml)')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--exp_root', type=str, default=None)
    parser.add_argument('--expr_id', type=str, default=None)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def find_latest_checkpoint(checkpoint_dir):
    """Find the checkpoint file with the highest epoch number in checkpoint_dir."""
    latest_epoch = -1
    latest_path = None
    latest_error = None

    for root, dirs, files in os.walk(checkpoint_dir):
        for fname in files:
            if fname.endswith("_state_dict.pt"):
                match = re.search(r"model_epoch_(\d+)_error@([0-9.eE+-]+)", fname)
                if match:
                    epoch = int(match.group(1))
                    error = float(match.group(2))
                    if epoch > latest_epoch:
                        latest_epoch = epoch
                        latest_error = error
                        latest_path = os.path.join(root, fname)

    return latest_path, latest_epoch, latest_error


def save_metrics_to_excel(all_metrics, summary, out_path,
                          metric_order=None, float_fmt="{:.6f}",
                          overall_label="Overall (mean±std)"):
    """Save per-event metrics and overall summary to an Excel file."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if metric_order is None:
        keys = set()
        for md in all_metrics.values():
            keys.update(md.keys())
        keys.update(summary.get("mean", {}).keys())
        keys.update(summary.get("std", {}).keys())
        preferred = ["R2", "MSE", "RMSE", "MAE", "PeakR2", "CSI"]
        metric_order = [m for m in preferred if m in keys] + \
                       [m for m in sorted(keys) if m not in preferred]

    header = ["Event"] + metric_order
    rows = []
    for ev in sorted(all_metrics):
        md = all_metrics[ev]
        row = [str(ev)] + [
            ("" if md.get(m) is None else float_fmt.format(md[m]))
            for m in metric_order
        ]
        rows.append(row)

    overall_row = [overall_label]
    for m in metric_order:
        mean_v = summary.get("mean", {}).get(m)
        std_v = summary.get("std", {}).get(m)
        if mean_v is None or std_v is None:
            overall_row.append("")
        else:
            overall_row.append(f"{float_fmt.format(mean_v)}±{float_fmt.format(std_v)}")

    df = pd.DataFrame(rows + [overall_row], columns=header)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="metrics")


def run_test(location, model, exp_root, expr_id, data_root, config, device):
    save_dir = os.path.join(exp_root, expr_id)
    visualization_dir = os.path.join(save_dir, f"visualization/{location}")
    results_dir = os.path.join(save_dir, f"pred_results/{location}")

    trainer = Trainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        device=device,
        data_processor=None,
        mixed_precision=config.opt.amp_autocast,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose,
        visualization_dir=visualization_dir,
        save_dir=save_dir,
        results_dir=results_dir,
        flood_threshold=config.eval.flood_threshold,
    )

    checkpoint_dir = os.path.join(exp_root, expr_id)
    checkpoint_path, epoch, error = find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in: {checkpoint_dir}")
    print(f"Loading checkpoint: {checkpoint_path}  (epoch={epoch})")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Strip DataParallel 'module.' prefix if present
    new_state_dict = OrderedDict(
        (k[7:] if k.startswith("module.") else k, v)
        for k, v in state_dict.items()
    )
    trainer.model.load_state_dict(new_state_dict)
    trainer.model.eval()

    test_dataset = Dynamic2DFlood(
        data_root=data_root,
        split='test',
        location=location,
        train_list=config.data.train_list,
        test_list=config.data.test_list,
        wall_height=config.tfno2d.wall_height,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=config.data.num_workers_test, pin_memory=True,
    )

    all_metrics, summary = trainer.evaluate_for_test(epoch=epoch, test_loader=test_loader)

    test_metrics_dir = os.path.join(save_dir, f"test_metrics/{location}")
    os.makedirs(test_metrics_dir, exist_ok=True)
    xlsx_path = os.path.join(
        test_metrics_dir,
        f"metrics_epoch_{epoch}_n@{len(test_loader)}.xlsx"
    )
    save_metrics_to_excel(all_metrics, summary, xlsx_path)

    print("Per-event metrics:", all_metrics)
    print("Overall summary:", summary)
    print("Done.")
    return all_metrics, summary


if __name__ == "__main__":
    args = _parse_args()

    pipe = ConfigPipeline([
        YamlConfig(args.config, config_name="default", config_folder="./configs"),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../configs"),
    ])
    config = pipe.read_conf()

    device = select_device(args.device, config.data.batch_size)

    data_root = args.data_root or config.data.data_root
    exp_root = args.exp_root or config.data.exp_root

    # expr_id: prefer CLI arg, otherwise find latest experiment directory
    if args.expr_id:
        expr_id = args.expr_id
    else:
        subdirs = [d for d in os.listdir(exp_root)
                   if os.path.isdir(os.path.join(exp_root, d))]
        if not subdirs:
            raise RuntimeError(f"No experiment directories found in {exp_root}")
        expr_id = sorted(subdirs)[-1]
    print(f"Testing experiment: {expr_id}")

    model = get_model(config)
    model = model.to(device)

    locations = [loc.strip() for loc in config.eval.locations.split(",")]
    for location in locations:
        t0 = time.time()
        run_test(location, model, exp_root, expr_id, data_root, config, device)
        print(f"location={location}  time={time.time()-t0:.2f}s")
        torch.cuda.empty_cache()
