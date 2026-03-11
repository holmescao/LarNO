import os
import sys
import builtins
_builtin_open = builtins.open
def _utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None, **kw):
    if encoding is None and 'b' not in mode:
        encoding = 'utf-8'
    return _builtin_open(file, mode, buffering, encoding=encoding, errors=errors, **kw)
builtins.open = _utf8_open
import argparse
import datetime
import math
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.losses.data_losses import WMSELoss, H1WMSELoss
from neuralop.data.datasets.Dynamic2DFlood import Dynamic2DFlood

from utils.torch_utils import select_device
from utils.distributed_utils import init_seeds

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

local_rank = int(os.getenv('LOCAL_RANK', -1))
rank = int(os.getenv('RANK', -1))
world_size = int(os.getenv('WORLD_SIZE', -1))


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default='ukea_finetune.yaml',
                        help='YAML config file in configs/ (default: ukea_finetune.yaml)')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--exp_root', type=str, default=None)
    parser.add_argument('--train_location', type=str, default=None)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def WarmUpCosineAnneal(optimizer, warm_up_iter, T_max, lr_max, lr_min):
    def lambda0(cur_iter):
        if cur_iter < warm_up_iter:
            return cur_iter / warm_up_iter
        progress = (cur_iter - warm_up_iter) / (T_max - warm_up_iter)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (lr_min + cosine * (lr_max - lr_min)) / lr_max
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)


def init_device(device, batch_size):
    if local_rank != -1:
        assert torch.cuda.device_count() > local_rank, (
            f"Insufficient CUDA devices for DDP: "
            f"device_count={torch.cuda.device_count()}, local_rank={local_rank}"
        )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo")
        return torch.device('cuda', local_rank)
    return select_device(device, batch_size)


def init_expr_path(exp_root):
    os.makedirs(exp_root, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    timestamp_path = os.path.join(exp_root, "timestamp.txt")
    with open(timestamp_path, "w") as f:
        f.write(timestamp)
    return timestamp


if __name__ == "__main__":
    args = _parse_args()

    pipe = ConfigPipeline([
        YamlConfig(args.config, config_name="default", config_folder="./configs"),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../configs"),
    ])
    config = pipe.read_conf()

    device = init_device(args.device, config.data.batch_size)

    init_seeds(config.distributed.seed)

    # Allow CLI to override YAML defaults
    data_root = args.data_root or config.data.data_root
    exp_root = args.exp_root or config.data.exp_root
    train_location = args.train_location or config.data.train_location

    timestamp = init_expr_path(exp_root)
    print("Experiment start! Now: ", timestamp)

    exp_dir = os.path.join(exp_root, timestamp)
    save_dir = exp_dir
    visualization_dir = os.path.join(exp_dir, "visualization")
    results_dir = os.path.join(exp_dir, "pred_results")

    if rank <= 0 and config.verbose:
        pipe.log()
        sys.stdout.flush()

    train_dataset = Dynamic2DFlood(
        data_root=data_root,
        split='train',
        location=train_location,
        train_list=config.data.train_list,
        test_list=config.data.test_list,
        wall_height=config.tfno2d.wall_height,
    )
    test_dataset = Dynamic2DFlood(
        data_root=data_root,
        split='test',
        location=train_location,
        train_list=config.data.train_list,
        test_list=config.data.test_list,
        wall_height=config.tfno2d.wall_height,
    )

    use_distributed = dist.is_initialized() and world_size > 1
    if use_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        train_batch_size = config.data.batch_size // world_size
    else:
        train_sampler = None
        test_sampler = None
        train_batch_size = config.data.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=config.data.num_workers_train,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        sampler=test_sampler,
        num_workers=config.data.num_workers_test,
        pin_memory=True,
    )

    model = get_model(config)

    # Load pre-trained weights if fine-tuning is enabled
    if config.finetune.enabled:
        pretrained_path = os.path.join(
            config.finetune.pretrained_dir, config.finetune.state_dict_name)
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(
                f"Pretrained weights not found: {pretrained_path}")
        print(f"Loading pretrained weights from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        # Strip DDP 'module.' prefix if the checkpoint was saved with DistributedDataParallel
        state_dict = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v)
            for k, v in state_dict.items()
        )
        model.load_state_dict(state_dict)
        print("Pretrained weights loaded successfully.")

    model = model.to(device)
    if config.distributed.use_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, static_graph=True)

    print(f"lr:{config.opt.lr_max * (config.data.batch_size/8)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.opt.lr_max * (config.data.batch_size/8))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.opt.T_max * (config.data.batch_size/8),
        eta_min=config.opt.lr_min * (config.data.batch_size/8)
    )

    l2loss = LpLoss(d=3, p=2, reduction='mean')
    h1loss = H1Loss(d=3, reduction='mean')
    wmse = WMSELoss(reduction='mean')
    h1wmse = H1WMSELoss(d=3)

    loss_map = {"l2": l2loss, "h1": h1loss, "WMSE": wmse, "h1WMSE": h1wmse}
    if config.opt.training_loss not in loss_map:
        raise ValueError(
            f"Unknown training_loss={config.opt.training_loss!r}. "
            f"Choose from {list(loss_map)}")
    train_loss = loss_map[config.opt.training_loss]
    eval_losses = {"h1": h1loss, "l2": l2loss}

    if rank <= 0 and config.verbose:
        print("\n MODEL \n", model)
        print("\n OPTIMIZER \n", optimizer)
        print("\n SCHEDULER \n", scheduler)
        print(f"\n LOSSES \n * Train: {train_loss}\n * Test: {eval_losses}")
        print("\n Beginning Training...\n")
        sys.stdout.flush()

    trainer = Trainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        device=device,
        data_processor=None,
        mixed_precision=config.opt.amp_autocast,
        verbose=config.verbose,
        visualization_dir=visualization_dir,
        save_dir=save_dir,
        results_dir=results_dir,
        window_size=config.opt.window_size,
        use_distributed=config.distributed.use_distributed,
        flood_threshold=config.eval.flood_threshold,
    )

    trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
        local_rank=local_rank,
        ddp=False,
    )
