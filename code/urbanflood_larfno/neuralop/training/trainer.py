import os
import csv
import sys
import warnings
import time
from pathlib import Path
from typing import Union
from timeit import default_timer

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from tqdm import tqdm

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import neuralop.mpu.comm as comm
from neuralop.losses import LpLoss
from .training_state import load_training_state, save_training_state
from neuralop.data.datasets.Dynamic2DFlood import MinMaxScaler, r_MinMaxScaler


def _to_device(inputs: dict, device):
    """Move all tensors in a dict to the given device."""
    return {k: v.to(device) for k, v in inputs.items()}

def _dist_get_rank():
    """Safe wrapper: returns 0 when distributed is not initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

class Trainer:
    """
    A general Trainer class to train neural operators on given datasets.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        n_epochs: int,
        device: str = '',
        mixed_precision: bool = False,
        data_processor: nn.Module = None,
        use_distributed: bool = True,
        verbose: bool = False,
        window_size: int = 10,
        save_every: int = 50,
        visualization_dir: str = "./visualization",
        save_dir: Union[str, Path] = "./checkpoints",
        results_dir: str = "./pred_results",
        y_range=None,
        flood_threshold: float = 0.03,
    ):
        """
        Initialize the Trainer.

        Parameters
        ----------
        model : nn.Module
            The model to be trained.
        n_epochs : int
            Total number of training epochs.
        device : str, default='cuda'
            Device to use for training, e.g., 'cpu' or 'cuda'.
        mixed_precision : bool, default=False
            Whether to use mixed precision training.
        data_processor : nn.Module, default=None
            Data processor for preprocessing and postprocessing data.
        use_distributed : bool, default=False
            Whether to use Distributed Data Parallel (DDP) training.
        verbose : bool, default=False
            Whether to print detailed logs.
        window_size : int, default=90
            Window size for splitting along the time dimension.
        """
        self.save_every = save_every
        self.model = model
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        self.window_size = window_size  # Initialize window_size
        # Handle autocast device type for mixed precision
        if isinstance(self.device, torch.device):
            self.autocast_device_type = self.device.type
        else:
            if "cuda" in self.device:
                self.autocast_device_type = "cuda"
            else:
                self.autocast_device_type = "cpu"
        self.mixed_precision = mixed_precision
        self.data_processor = data_processor

        # Track the starting epoch for checkpointing/resuming
        self.start_epoch = 1

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.visualization_dir = visualization_dir
        os.makedirs(self.visualization_dir, exist_ok=True)

        self.save_model_dir = os.path.join(self.save_dir, "weights")
        os.makedirs(self.save_model_dir, exist_ok=True)

        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.y_range = y_range if y_range is not None else {"h": [0, 10000]}
        self.flood_threshold = flood_threshold
        
    def train(
        self,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        regularizer=None,
        training_loss=None,
        eval_losses=None,
        save_best: int = None,
        resume_from_dir: Union[str, Path] = None,
        local_rank=0,
        ddp=True, 
    ):
        """Trains the given model on the given dataset.

        If a device is provided, the model and data processor are loaded to device here.

        Parameters
        -----------
        train_loader: torch.utils.data.DataLoader
            Training data loader.
        test_loader: dict[torch.utils.data.DataLoader]
            Testing data loaders.
        optimizer: torch.optim.Optimizer
            Optimizer to use during training.
        scheduler: torch.optim.lr_scheduler
            Learning rate scheduler.
        regularizer: callable, optional
            Regularization function.
        training_loss: callable, optional
            Training loss function.
        eval_losses: dict[Loss], optional
            Loss functions to use during evaluation.
        save_every: int, optional, default=1
            Interval (in epochs) at which to save checkpoints.
        save_best: str, optional, default=None
            Metric key to monitor and save the best model.
            Overrides save_every and saves on eval_interval.
        save_dir: str | Path, default="./ckpt"
            Directory to save checkpoints.
        resume_from_dir: str | Path, default=None
            Directory to resume training state from.

        Returns
        -------
        all_metrics: dict
            Metrics for the last validation epoch across all test loaders.
        """

        self.regularizer = regularizer if regularizer else None

        if training_loss is None:
            training_loss = LpLoss(d=2)

        # Warn the user if training loss is reducing across the batch
        if hasattr(training_loss, 'reduction'):
            if training_loss.reduction == "mean":
                warnings.warn(f"{training_loss.reduction=}. This means that the loss is "
                              "initialized to average across the batch dim. The Trainer "
                              "expects losses to sum across the batch dim.")

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        # Checkpoint-related attributes

        self.save_best = save_best

        # Load model and data_processor to device
        self.model = self.model.to(self.device)

        if self.use_distributed and dist.is_initialized():
            device_id = _dist_get_rank()
            self.model = DDP(self.model, device_ids=[
                             device_id], output_device=device_id)
            
        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)

        # Ensure save_best is a metric we collect
        if self.save_best is not None:
            metrics = []
            for name in test_loader.keys():
                for metric in eval_losses.keys():
                    metrics.append(f"{name}_{metric}")

            # Add R2Metric to monitored metrics
            for name in test_loader.keys():
                metrics.append(f"{name}_r2")
            assert self.save_best in metrics, \
                f"Error: expected a metric of the form <loader_name>_<metric>, got {save_best}"
            best_metric_value = float('inf')
            # Either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        if _dist_get_rank()==0:
            if self.verbose:
                print(f'Training on {len(train_loader.dataset)} samples')
                print(f'Testing on {len(test_loader.dataset)} samples')
                sys.stdout.flush()

        train_metrics_dir = os.path.join(self.save_dir, f'train_metrics')
        os.makedirs(train_metrics_dir, exist_ok=True)
        test_metrics_dir = os.path.join(self.save_dir, f'test_metrics')
        os.makedirs(test_metrics_dir, exist_ok=True)

        min_train_err = 1e15

        for epoch in range(self.start_epoch, self.n_epochs + 1):
            if isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            train_err,  epoch_train_time =\
                self.train_one_epoch(optimizer, scheduler, epoch, train_loader, training_loss)
            
            scheduler.step()

            if _dist_get_rank()==0:
                epoch_metrics = dict(
                    train_err=train_err,
                    epoch_train_time=epoch_train_time
                )
                self.save_metrics(epoch_metrics, epoch, train_metrics_dir)

                # Save checkpoint if save_every and save_best is not set
                if min_train_err > train_err:
                    min_train_err = train_err
                    self.checkpoint(optimizer, scheduler, self.save_model_dir, train_err)


    def train_one_epoch(self, optimizer, scheduler, epoch, train_loader, training_loss):
        """Trains the model for one epoch and returns training metrics.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        train_loader : torch.utils.data.DataLoader
            Training data loader.
        training_loss : callable
            Training loss function.

        Returns
        -------
        train_err : float
            Training error.
        avg_loss : float
            Average loss.
        avg_lasso_loss : float or None
            Average regularization loss.
        epoch_train_time : float
            Training time for the epoch.
        """
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0

        # Track the number of training examples
        self.n_samples = 0
        
        pbar = enumerate(train_loader)
        if _dist_get_rank()==0:
            pbar = tqdm(pbar, leave=False, total=len(train_loader))

        for idx, (inputs, labels, event_name) in pbar:
            with torch.no_grad():
                # Move to GPU
                inputs = _to_device(inputs, self.device)
                labels = labels.to(self.device)

                # Normalize labels
                labels = MinMaxScaler(labels, self.y_range["h"][1], self.y_range["h"][0])


                if self.verbose:
                    print(f"Processing batch {idx}")

                # Get time dimension size
                T = labels.shape[-1]
                window_size = self.window_size # window length
                num_windows = T // window_size # number of windows

                window_sample = {
                    'x': inputs, # raw, unnormalized inputs
                    'y': labels, # normalized full sequence BCTHW
                }

            for w in range(num_windows):
                optimizer.zero_grad(set_to_none=True)

                # Train on the current window
                ind = window_size*w
                loss = self.train_one_window(
                    idx, ind, window_size, window_sample, training_loss)

                # Backward pass and optimizer step
                loss.backward()

                optimizer.step()

                # Accumulate losses
                train_err += loss.item()

                del loss

                torch.cuda.empty_cache()

            del inputs
            del labels
            del window_sample

        epoch_train_time = default_timer() - t1

        train_err /= len(train_loader)

        # Get current learning rate
        lr = None
        for pg in optimizer.param_groups:
            lr = pg["lr"]

        if _dist_get_rank()==0:
            if self.verbose:
                self.log_training(
                    epoch=epoch,
                    time=epoch_train_time,
                    train_err=train_err,
                    lr=lr
                )

        return train_err,  epoch_train_time

    def train_one_window(self, idx, ind, window_size, window_sample, training_loss,):
        """Trains the model on a single window.

        Parameters
        ----------
        idx : int
            Batch index within the train_loader.
        window_sample : dict
            Dictionary containing data for a single window.
        training_loss : callable
            Training loss function.

        Returns
        -------
        loss : torch.Tensor
            Loss for the current window.
        """

        if self.regularizer:
            self.regularizer.reset()
        if self.data_processor is not None:
            window_sample = self.data_processor.preprocess(window_sample)

        # Forward pass
        B,_,H,W,_ = window_sample["y"].shape
        # print(f"++++++++++ B:{B}")
        K = window_size
        init_shape = (B,H,W,K)
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(ind, window_sample["x"],window_size,cache_key=idx)
        else:
            out = self.model(ind, window_sample["x"],self.device,cache_key=idx,init_shape=init_shape)

        if self.epoch == 0 and idx == 0 and self.verbose:
            print(f"Raw output shape: {out.shape}")

        if self.data_processor is not None:
            out, window_sample = self.data_processor.postprocess(
                out, window_sample)

        # Compute loss
        # loss = 0.0

        # (B,C,H,W,T)
        label_h = window_sample["y"][:, 0,:,:,ind:ind+window_size].float() # BCHWK -> BHWK
        # label_u = window_sample["y"][:, 1,:,:,ind:ind+window_size].float()
        # label_v = window_sample["y"][:, 2,:,:,ind:ind+window_size].float()

        out_h = out[:, 0].float()
        # out_u = out[:, 1].float()
        # out_v = out[:, 2].float()


        loss_h = training_loss(out_h, label_h)
        # loss_u = training_loss(out_u, label_u)
        # loss_v = training_loss(out_v, label_v)

        # loss = loss_h + loss_u + loss_v
        loss = loss_h

        # if self.mixed_precision:
        #     with torch.autocast(device_type=self.autocast_device_type):
        #         loss += training_loss(out, **window_sample)
        # else:
        #     loss += training_loss(out, **window_sample)

        if self.regularizer:
            loss += self.regularizer.loss

        return loss


    def evaluate_for_test(self, epoch, test_loader):
        """
        Returns:
        all_metrics: per-event metrics
            {
            "<eventA>": {"R2": float, "MSE": float, "RMSE": float, "MAE": float,
                        "PeakR2": float, "CSI": float},
            "<eventB>": {...},
            ...
            }
        summary: mean/std across all events
            {
            "mean": {"R2": float, "MSE": float, "RMSE": float, "MAE": float,
                    "PeakR2": float, "CSI": float},
            "std":  {same keys}
            }
        """
        device = self.device
        all_metrics = {}

        # Aggregation lists for global mean/std
        agg = {
            "R2":     [],
            "MSE":    [],
            "RMSE":   [],
            "MAE":    [],
            "PeakR2": [],
            "CSI":    []
        }
        flood_thr = self.flood_threshold

        def _event_to_str(ev):
            # Normalize event_name to a stable str (handles str/list/tuple/tensor/ndarray/bytes)
            if isinstance(ev, (bytes, bytearray)):
                try:
                    return ev.decode("utf-8")
                except Exception:
                    return str(ev)
            if isinstance(ev, (list, tuple)):
                return _event_to_str(ev[0]) if len(ev) > 0 else "unknown"
            if hasattr(ev, "tolist"):  # tensor / ndarray
                raw = ev.tolist()
                if isinstance(raw, list):
                    return _event_to_str(raw[0]) if len(raw) > 0 else "unknown"
                return _event_to_str(raw)
            return str(ev)

        epoch_visualization_dir = os.path.join(self.visualization_dir, f'epoch_{epoch}')
        os.makedirs(epoch_visualization_dir, exist_ok=True)
        epoch_pred_dir = os.path.join(self.results_dir, f'epoch_{epoch}')
        os.makedirs(epoch_pred_dir, exist_ok=True)

        with torch.no_grad():
            for idx, (inputs, labels, event_name) in enumerate(test_loader):
                ev = _event_to_str(event_name)
                if self.verbose:
                    print(f"Processing test sample {idx}, event_name: {ev}")

                # —— Move to GPU / normalize —— #
                inputs = _to_device(inputs, device)
                # labels = labels.to(device, dtype=torch.float32)
                labels = MinMaxScaler(labels, self.y_range["h"][1], self.y_range["h"][0])

                sample = {'x': inputs,'y':labels}   # y: [B, C, H, W, T]
                T = labels.shape[-1]

                # —— Run model inference over full sequence —— #
                full_pred = self.predict_full_sample(idx, sample, T)  # [B,C,H,W,T]

                # Evaluate only the water depth h channel: shape [B, H, W, T]
                y_true_h = labels[:, 0, :, :, :]  # [B, H, W, T]
                y_pred_h = full_pred[:, 0, :, :, :]

                # Inverse normalization
                unit = 1000
                h_min = self.y_range["h"][0] / unit
                h_max = self.y_range["h"][1] / unit
                y_true_h = r_MinMaxScaler(y_true_h, h_max, h_min)
                y_pred_h = r_MinMaxScaler(y_pred_h, h_max, h_min)

                # Flatten to 1D vector (all space and time)
                y_true_flat = y_true_h.flatten()
                y_pred_flat = y_pred_h.flatten()

                # —— Basic metrics (per event sample) —— #
                eps = 1e-12
                diff = y_pred_flat - y_true_flat

                mse_val  = torch.mean(diff ** 2)
                rmse_val = torch.sqrt(mse_val + eps)
                mae_val  = torch.mean(torch.abs(diff))

                ss_res = torch.sum(diff ** 2)
                mean_y = torch.mean(y_true_flat)
                ss_tot = torch.sum((y_true_flat - mean_y) ** 2)
                r2_val = 1.0 - ss_res / (ss_tot + eps)

                # —— PeakR2: at the timestep t* where spatially-averaged true depth is maximum,
                # compute R² over that frame (H, W) —— #
                # y_true_h, y_pred_h: [B, H, W, T]
                # Average over (B,H,W) to get global mean depth at each t
                # (more robust than single-pixel max; change mean->sum for total-volume peak)
                spatial_mean_per_t = y_true_h.mean(dim=(0, 1, 2))  # [T]
                t_star = int(torch.argmax(spatial_mean_per_t).item())

                y_true_peak = y_true_h[..., t_star].flatten()  # [B*H*W]
                y_pred_peak = y_pred_h[..., t_star].flatten()

                diff_peak = y_pred_peak - y_true_peak
                ss_res_peak = torch.sum(diff_peak ** 2)
                mean_peak = torch.mean(y_true_peak)
                ss_tot_peak = torch.sum((y_true_peak - mean_peak) ** 2)
                peak_r2_val = 1.0 - ss_res_peak / (ss_tot_peak + eps)

                # —— CSI: maximum inundation extent (a pixel is considered flooded if depth > threshold at any timestep) —— #
                # Take the temporal maximum depth per pixel
                y_true_max = torch.amax(y_true_h, dim=-1)  # [B, H, W]
                y_pred_max = torch.amax(y_pred_h, dim=-1)  # [B, H, W]

                true_flood = (y_true_max > flood_thr)
                pred_flood = (y_pred_max > flood_thr)

                tp = torch.sum((true_flood & pred_flood).to(torch.float32))
                fp = torch.sum((~true_flood & pred_flood).to(torch.float32))
                fn = torch.sum((true_flood & ~pred_flood).to(torch.float32))
                csi_val = tp / (tp + fp + fn + eps)

                # Convert to float
                r2_f     = float(r2_val.item())
                mse_f    = float(mse_val.item())
                rmse_f   = float(rmse_val.item())
                mae_f    = float(mae_val.item())
                peakr2_f = float(peak_r2_val.item())
                csi_f    = float(csi_val.item())

                # Write per-event metrics
                all_metrics[ev] = {
                    "R2":     r2_f,
                    "MSE":    mse_f,
                    "RMSE":   rmse_f,
                    "MAE":    mae_f,
                    "PeakR2": peakr2_f,
                    "CSI":    csi_f
                }
                print(all_metrics)

                # Accumulate for global aggregation
                agg["R2"].append(r2_f)
                agg["MSE"].append(mse_f)
                agg["RMSE"].append(rmse_f)
                agg["MAE"].append(mae_f)
                agg["PeakR2"].append(peakr2_f)
                agg["CSI"].append(csi_f)

                
                
                # Visualization and peak map (batch_index: if B>1, loop or pass a specific index)
                self.visualize_and_save_full(ev, y_true_h.numpy(), y_pred_h.numpy(), epoch, epoch_visualization_dir, idx)
                self.save_comparison_gif(ev, y_true_h.numpy(), y_pred_h.numpy(), epoch, epoch_visualization_dir)

                # Peak distribution map
                peak_path = os.path.join(epoch_visualization_dir, f"{ev}_peak_maps_epoch{epoch}_idx{idx}.png")
                visualize_peak_maps(
                    y_true_h=y_true_h,
                    y_pred_h=y_pred_h,
                    save_path=peak_path,
                    title=f"{ev} Peak Maps (epoch {epoch}, idx {idx})",
                    batch_index=0,        # if B>1, loop or pass specific index
                    peak_reduce="mean"    # or "sum"
                )

                # Maximum inundation extent error classification map
                csi_path = os.path.join(epoch_visualization_dir, f"{ev}_maxCSI_errmap_epoch{epoch}_idx{idx}.png")
                visualize_max_inundation_error_map(
                    y_true_h=y_true_h,
                    y_pred_h=y_pred_h,
                    flood_thr=self.flood_threshold,
                    save_path=csi_path,
                    title=f"{ev} Max Inundation Error Map (epoch {epoch}, idx {idx})",
                    batch_index=0
                )

                self.save_predictions_full(ev, y_pred_h.numpy(), epoch, epoch_pred_dir, idx)

        # —— Compute mean/std —— #
        summary = {
            "mean": {k: float(np.mean(v)) if len(v) > 0 else float("nan") for k, v in agg.items()},
            "std":  {k: float(np.std(v))  if len(v) > 0 else float("nan") for k, v in agg.items()},
        }

        if self.verbose:
            print("Evaluation metrics by event:")
            for ev, md in all_metrics.items():
                print(ev, md)
            print("Summary:", summary)

        return all_metrics, summary


    def predict_full_sample(self, idx, sample: dict, T):
        """
        Run windowed prediction on a single sample and reconstruct the full prediction sequence.
        """
        self.model.eval()
        self.model.to(self.device)

        B, C, H, W = sample['y'].shape[0], sample['y'].shape[1], sample['y'].shape[2], sample['y'].shape[3]

        K = T
        init_shape = (B,H,W,K)

        ind = 0
        # Generate predictions
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                # y_pred = self.model(**window_sample)
                full_pred = self.model(ind, sample["x"],self.device, window_size=T,cache_key=None)
        else:

            # y_pred = self.model(**window_sample)
            start_time = time.time()
            full_pred = self.model(ind, sample["x"], self.device, cache_key=None, init_shape=init_shape)
            end_time = time.time()
            print(f"Prediction time for sample {idx}: {end_time - start_time:.9f} seconds")

        if self.data_processor is not None:
            full_pred, sample = self.data_processor.postprocess(
                full_pred, sample)

        return full_pred.cpu()

    
    def visualize_and_save_full(self, event_name, labels, full_pred: torch.Tensor, epoch: int, epoch_dir: str, sample_idx: int):
        """
        Generate and save visualization images of ground-truth and predicted water depth H.
        """

        # Extract predicted y and ground-truth y
        y_pred = full_pred  # [B, C, H, W, T]

        # Select water depth h (assumed to be the first channel)
        y_h = labels
        y_pred_h = y_pred

        # unit = 1000
        # h_min = self.y_range["h"][0] / unit
        # h_max = self.y_range["h"][1] / unit
        # y_h = r_MinMaxScaler(y_h.cpu().numpy(), h_max, h_min)
        # y_pred_h = r_MinMaxScaler(y_pred_h.cpu().numpy(), h_max, h_min)

        # Select time points to visualize based on total duration T (6 evenly spaced frames)
        # Show 6 timesteps distributed across y_h's time range
        T = y_h.shape[-1]  # time dimension size
        time_points = [int(T * i / 6) for i in range(6)]

        # Convert minutes to indices (assume one timestep per minute, 0-indexed)
        # time_indices = [tp - 1 for tp in time_points]
        time_indices = time_points  # use indices directly

        # Compute min/max of labels and predictions to unify colorbar range
        # vmin = min(y_h.min(), y_pred_h.min()).item()
        # vmax = max(y_h.max(), y_pred_h.max()).item()
        vmin, vmax = 0, 2  # colorbar range

        # Create a figure with 2 rows and len(time_indices) columns
        fig, axes = plt.subplots(2, len(time_indices),
                                 figsize=(3 * len(time_indices), 6))
        cmap = plt.cm.get_cmap('coolwarm').copy()
        cmap.set_bad(color='white')  # masked or NaN values will display as white
        
        for i, (tp, idx) in enumerate(zip(time_points, time_indices)):
            if idx >= y_h.shape[-1]:
                print(
                    f"Time index {idx} is out of bounds for time dimension {y_h.shape[-1]}. Skipping.")
                continue

            # Retrieve ground-truth and predicted images
            label_img = y_h[0, :, :, idx]       # [H=400, W=400]
            pred_img = y_pred_h[0, :, :, idx]   # [H=400, W=400]

            # Option 1: mask regions below threshold (recommended)
            # label_img = np.ma.masked_less(label_img, 0.01)
            # pred_img  = np.ma.masked_less(pred_img, 0.01)


            # Plot ground-truth
            ax = axes[0, i]
            im = ax.imshow(label_img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            ax.set_title(f'Label {tp} min')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Plot prediction
            ax = axes[1, i]
            im = ax.imshow(pred_img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            ax.set_title(f'Prediction {tp} min')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        # Save figure
        save_path = os.path.join(
            epoch_dir, f'visualization_epoch_{epoch}_sample_{event_name}.png')
        plt.savefig(save_path, dpi=50)
        plt.close(fig)  # Close figure to free memory

        if self.verbose:
            print(f"Saved visualization to {save_path}")

    def save_comparison_gif(self, event_name, y_true_h, y_pred_h, epoch, epoch_dir):
        """Save an animated GIF comparing ground-truth (Reference) and LarNO prediction.

        Parameters
        ----------
        y_true_h, y_pred_h : np.ndarray, shape (B, H, W, T), unit: metres
        """
        import io
        from PIL import Image

        B, H, W, T = y_true_h.shape
        vmax = min(float(max(y_true_h.max(), y_pred_h.max())), 2.0)
        vmin = 0.0
        cmap = plt.cm.get_cmap('Blues').copy()
        cmap.set_bad(color='white')

        step = max(1, T // 36)   # at most 36 frames → compact GIF
        frames = []
        for t in range(0, T, step):
            fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=80)
            for ax, img, title in [
                (axes[0], y_true_h[0, :, :, t], f'Reference  ({t} min)'),
                (axes[1], y_pred_h[0, :, :, t], f'LarNO  ({t} min)'),
            ]:
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=9)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Depth (m)')
            plt.suptitle(event_name, fontsize=10)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()

        if frames:
            gif_path = os.path.join(epoch_dir, f'{event_name}_epoch{epoch}.gif')
            frames[0].save(
                gif_path, save_all=True, append_images=frames[1:],
                duration=50, loop=0,
            )
            if self.verbose:
                print(f"Saved comparison GIF to {gif_path}")

    def save_predictions_full(self, event_name, full_pred: torch.Tensor, epoch: int, epoch_pred_dir: str, sample_idx: int):
        """
        Save the full reconstructed prediction (h channel) as a single .npy file.
        """
        self.model.eval()
        self.model.to(self.device)

        # Convert to NumPy array
        if  isinstance(full_pred, torch.Tensor):
            y_pred_np = full_pred[0].cpu().numpy()  # [C=1, H=400, W=400, T=360]
        else:
            y_pred_np = full_pred[0]  # assumed to already be a numpy array

        # Build save path
        pred_filename = f'predictions_epoch_{epoch}_sample_{event_name}.npy'
        pred_path = os.path.join(epoch_pred_dir, pred_filename)

        # Save as .npy file
        np.save(pred_path, y_pred_np)


    def on_epoch_start(self, epoch):
        """Hook that runs at the beginning of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.

        Returns
        -------
        None
        """
        self.epoch = epoch
        if _dist_get_rank() == 0:
            if self.verbose:
                print(f"Starting epoch {epoch}")

    def log_training(self,
                     epoch: int,
                     time: float,
                     train_err: float,
                     lr: float = None
                     ):
        """Logs training metrics.

        Parameters
        ----------
        epoch: int
            Current epoch number.
        time: float
            Training time for the epoch.
        avg_loss: float
            Average loss.
        train_err: float
            Training error.
        avg_lasso_loss: float, optional
            Average regularization loss.
        lr: float, optional
            Current learning rate.
        """
        msg = f"[{epoch}] time={time:.2f}, "
        msg += f"lr [{lr}], "
        msg += f"train_err={train_err:.9f}"

        print(msg)
        sys.stdout.flush()

    def log_eval(self,
                 epoch: int,
                 eval_metrics: dict):
        """Logs evaluation metrics.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        eval_metrics : dict
            Metrics collected during evaluation, keyed by f"{test_loader_name}_{metric}".
        """
        msg = ""
        for metric, value in eval_metrics.items():
            if isinstance(value, float) or isinstance(value, torch.Tensor):
                msg += f"{metric}={value:.9f}, "

        msg = f"Eval: " + msg[:-2]  # Remove the last comma and space
        print(msg)
        sys.stdout.flush()

    def resume_state_from_dir(self, save_dir):
        """
        Resumes training from the given directory.

        Parameters
        ----------
        save_dir: Union[str, Path]
            Directory where the training state is saved.
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        # Check if a saved model exists
        if (save_dir / "best_model_state_dict.pt").exists():
            save_name = "best_model"
        elif (save_dir / "model_state_dict.pt").exists():
            save_name = "model"
        else:
            raise FileNotFoundError(
                "Error: resume_from_dir expects a model state dict named model.pt or best_model.pt.")

        # Load the training state
        self.model, self.optimizer, scheduler, self.regularizer, resume_epoch =\
            load_training_state(save_dir=save_dir, save_name=save_name,
                                model=self.model,
                                optimizer=self.optimizer,
                                regularizer=self.regularizer,
                                scheduler=scheduler)

        if resume_epoch is not None:
            if resume_epoch > self.start_epoch:
                self.start_epoch = resume_epoch
                if self.verbose:
                    print(
                        f"Trainer resuming training from epoch {resume_epoch}")

    def checkpoint(self, optimizer, scheduler, save_dir, train_err):
        """Saves the current training state to the specified directory.

        Parameters
        ----------
        save_dir : str | Path
            Directory to save the training state.
        """
        if comm.get_local_rank() == 0:
            if self.save_best is not None:
                save_name = 'best_model'
            else:
                save_name = "model"

            # Build a unique filename based on epoch number and error
            save_filename = f"{save_name}_epoch_{self.epoch}_error@{train_err:.9f}"
            save_path = os.path.join(save_dir, save_filename)

            # Ensure save directory exists
            os.makedirs(save_dir, exist_ok=True)

            save_training_state(save_dir=save_path,
                                save_name=save_filename,
                                model=self.model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                regularizer=self.regularizer,
                                epoch=self.epoch
                                )
            if self.verbose:
                print(f"[Rank 0]: Saved training state to {save_path}")


    def save_metrics(self, epoch_metrics: dict, epoch: int, test_metrics_dir):
        """
        Save metrics to a CSV file.

        Parameters:
        - eval_metrics: dict
            Evaluation metrics, including R2 and MSE.
        - epoch: int
            Current epoch number, used to name the saved file.
        """
        csv_filename = 'metrics.csv'
        csv_path = os.path.join(test_metrics_dir, csv_filename)

        file_exists = os.path.isfile(csv_path)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Write header
                header = ['Epoch'] + list(epoch_metrics.keys())
                writer.writerow(header)

            row = [
                epoch] + [f"{value:.9f}" if value is not None else "None" for value in epoch_metrics.values()]
            writer.writerow(row)

        if self.verbose:
            print(f"Saved metrics to {csv_path}")

    def save_test_metrics(self, epoch_metrics: dict, epoch: int, out_dir: str):
        """
        Save metrics to out_dir/metrics.csv.
        Supports two input formats:
        A) Per-event hierarchical structure:
            {
            "eventA": {"R2": 0.9, "MSE": 0.01, ...},
            "eventB": {...}
            }
            -> write rows as (Epoch, Event, Metric, Value)

        B) Flat structure (compatible with original training log format):
            {"train_err":..., "avg_loss":..., ...}
            -> write (Epoch, Event='__overall__', Metric, Value)
        """
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "metrics.csv")
        file_exists = os.path.isfile(csv_path)

        def _write_rows(writer, epoch_metrics, epoch):
            # Per-event structure
            if all(isinstance(v, dict) for v in epoch_metrics.values()):
                for ev, md in epoch_metrics.items():
                    for metric, val in md.items():
                        writer.writerow([epoch, ev, metric, f"{float(val):.6f}"])
            else:
                # Fallback: no event structure
                ev = "__overall__"
                for metric, val in epoch_metrics.items():
                    writer.writerow([epoch, ev, metric, f"{float(val):.6f}"])

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Epoch", "Event", "Metric", "Value"])
            _write_rows(writer, epoch_metrics, epoch)

        if self.verbose:
            print(f"[save_metrics] wrote to {csv_path}")


def visualize_max_inundation_error_map(
    y_true_h: torch.Tensor,
    y_pred_h: torch.Tensor,
    flood_thr: float = 1e-3,
    save_path: str = None,
    title: str = None,
    batch_index: int = 0,
):
    """
    Plot the maximum inundation extent error classification map:
    - Blue:  TP (correctly predicted flooded, True>thr & Pred>thr)
    - Red:   FN (missed detection, True>thr & Pred<=thr)
    - Yellow: FP (false alarm, Pred>thr & True<=thr)
    - Gray:  TN (neither flooded)
    """
    assert y_true_h.ndim == 4 and y_pred_h.ndim == 4, "Expect [B,H,W,T]"

    # Take temporal maximum per pixel
    y_true_max = torch.amax(y_true_h[batch_index], dim=-1).numpy()  # [H,W]
    y_pred_max = torch.amax(y_pred_h[batch_index], dim=-1).numpy()

    true_flood = (y_true_max > flood_thr)
    pred_flood = (y_pred_max > flood_thr)

    # Encode: 0=TN, 1=TP, 2=FN, 3=FP
    code = np.zeros_like(true_flood, dtype=np.uint8)
    tp = (true_flood & pred_flood)
    fn = (true_flood & ~pred_flood)
    fp = (~true_flood & pred_flood)
    code[tp] = 1
    code[fn] = 2
    code[fp] = 3

    # Color map (TN gray, TP blue, FN red, FP yellow)
    cmap = ListedColormap(["#D3D3D3", "#1f77b4", "#d62728", "#ffdf00"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    im = ax.imshow(code, cmap=cmap, norm=norm)
    # ax.set_xlabel("W")
    # ax.set_ylabel("H")
    if title:
        ax.set_title(title)

    # Custom legend
    legend_elems = [
        Patch(facecolor="#1f77b4", edgecolor='k', label="TP (correctly predicted flooded)"),
        Patch(facecolor="#d62728", edgecolor='k', label="FN (missed detection)"),
        Patch(facecolor="#ffdf00", edgecolor='k', label="FP (false alarm)"),
        Patch(facecolor="#D3D3D3", edgecolor='k', label="TN (true negative)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", frameon=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def visualize_peak_maps(
    y_true_h: torch.Tensor,
    y_pred_h: torch.Tensor,
    save_path: str = None,
    title: str = None,
    batch_index: int = 0,
    peak_reduce: str = "mean",  # "mean" or "sum", used to determine peak timestep t*
    vmin: float = None,
    vmax: float = None,
):
    """
    Plot peak 2D distribution maps: left=true, center=pred, right=error (pred-true).
    y_true_h, y_pred_h: tensors of shape [B, H, W, T] (water depth channel only).
    peak_reduce: use "mean" (global average) or "sum" (global total) to determine peak timestep t*.
    """
    assert y_true_h.ndim == 4 and y_pred_h.ndim == 4, "Expect [B,H,W,T]"
    device = y_true_h.device

    # Select t*: timestep where global average/total reaches its peak
    if peak_reduce == "sum":
        score_t = y_true_h.sum(dim=(0, 1, 2))      # [T]
    else:
        score_t = y_true_h.mean(dim=(0, 1, 2))     # [T]
    t_star = int(torch.argmax(score_t).item())

    # Extract frame for batch_index at t*
    true_peak = y_true_h[batch_index, :, :, t_star].numpy()
    pred_peak = y_pred_h[batch_index, :, :, t_star].numpy()
    err_peak  = pred_peak - true_peak

    # Color scale: unified for true/pred; symmetric for error
    if vmin is None or vmax is None:
        vmin = np.nanmin([true_peak.min(), pred_peak.min()]) if vmin is None else vmin
        vmax = np.nanmax([true_peak.max(), pred_peak.max()]) if vmax is None else vmax
    vmin = float(-np.max(np.abs(err_peak)))
    vmax = float(np.max(np.abs(err_peak)))
    eabs = np.nanmax(np.abs(err_peak)) + 1e-12

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    im0 = axs[0].imshow(true_peak, vmin=0, vmax=vmax)
    axs[0].set_title(f"True @ t*={t_star}")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(pred_peak, vmin=0, vmax=vmax)
    axs[1].set_title("Pred @ t*")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(err_peak, vmin=-0.5, vmax=0.5, cmap='coolwarm')
    axs[2].set_title("Error (Pred - True)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    # for ax in axs:
    #     ax.set_xlabel("W")
    #     ax.set_ylabel("H")

    if title:
        fig.suptitle(title, y=1.02)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

