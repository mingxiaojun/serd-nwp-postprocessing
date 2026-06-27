# -*- coding: utf-8 -*-
import os
import time
import glob
import argparse
from datetime import datetime

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import amp
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm

from serd.data.forecast_analysis_dataset import ForecastDataset
from serd.data.normalizer_forecast import DataNormalizer as DataNormalizer_fc
from serd.data.normalizer_analysis import DataNormalizer as DataNormalizer_err
from serd.models import forecast_mean_unet

# =========================================================
# Args
# =========================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="DDP Train Regression Mean Model with Pure MSE Loss"
    )

    # Basic training settings
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=7)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Data and output paths
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./outputs/checkpoints/serd_v1")
    parser.add_argument(
        "--data_root_glob",
        type=str,
        default="/path/to/CMA_gfs_time_order_3_72/*[0-9]",
    )
    parser.add_argument(
        "--topo_path",
        type=str,
        default="./data/topo_data_Normalization.npy",
    )
    parser.add_argument("--train_count", type=int, default=1292)
    parser.add_argument("--valid_count", type=int, default=92)

    # Model and data shape settings
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--num_surface_vars", type=int, default=5)
    parser.add_argument("--num_levels", type=int, default=9)
    parser.add_argument("--num_classes", type=int, default=24)

    # Evaluation settings
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument(
        "--phys_eval_channel_idx",
        type=int,
        default=-1,
        help="channel index used for physical-space MSE evaluation",
    )

    # Other settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="preferred device in non-DDP mode",
    )
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--eval_seed", type=int, default=20251103)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--disable_compile", action="store_true")

    parser.set_defaults(save_images=True)
    return parser


parser = build_parser()
args = parser.parse_args()


# =========================================================
# Utils
# =========================================================
def validate_args(args):
    assert args.batch_size >= 1, "batch_size must be >= 1"
    assert args.lr > 0, "lr must be > 0"
    assert args.epochs >= 1, "epochs must be >= 1"
    assert args.weight_decay >= 0, "weight_decay must be >= 0"
    assert args.num_workers >= 0, "num_workers must be >= 0"
    assert args.grad_clip > 0, "grad_clip must be > 0"
    assert args.train_count >= 1, "train_count must be >= 1"
    assert args.valid_count >= 1, "valid_count must be >= 1"
    assert args.height >= 1 and args.width >= 1, "height/width must be >= 1"
    assert args.num_surface_vars >= 1, "num_surface_vars must be >= 1"
    assert args.num_levels >= 1, "num_levels must be >= 1"
    assert args.num_classes >= 1, "num_classes must be >= 1"
    assert args.eval_every >= 1, "eval_every must be >= 1"
    assert args.save_every >= 1, "save_every must be >= 1"
    assert 0.0 < args.ema_decay < 1.0, "ema_decay must be in (0, 1)"


def setup_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_dist_initialized()) or dist.get_rank() == 0


def cleanup_distributed():
    if is_dist_initialized():
        dist.barrier()
        dist.destroy_process_group()


def setup_distributed(dist_backend="nccl", preferred_device="cuda"):
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if dist_backend == "nccl":
            assert torch.cuda.is_available(), "NCCL backend requires CUDA."

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend=dist_backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        rank, world_size, local_rank = 0, 1, 0
        if preferred_device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    return rank, world_size, local_rank, device


def build_grad_scaler(device_type, use_bf16):
    enabled = (device_type == "cuda") and (not use_bf16)
    try:
        return GradScaler(device=device_type, enabled=enabled)
    except TypeError:
        try:
            return GradScaler(device_type, enabled=enabled)
        except TypeError:
            return GradScaler(enabled=enabled)


# =========================================================
# Geo / Time
# =========================================================
def build_geo_channels(
    height,
    width,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    device,
    dtype=torch.float32,
    add_extras=False,
):
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    rel = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1,2,H,W]

    lons = torch.linspace(lon_min, lon_max, width, device=device, dtype=dtype).view(1, 1, 1, width).expand(1, 1, height, width)
    lats = torch.linspace(lat_min, lat_max, height, device=device, dtype=dtype).view(1, 1, height, 1).expand(1, 1, height, width)

    lonr = torch.deg2rad(lons)
    latr = torch.deg2rad(lats)
    geo = torch.cat(
        [torch.sin(lonr), torch.cos(lonr), torch.sin(latr), torch.cos(latr)],
        dim=1,
    )  # [1,4,H,W]

    if not add_extras:
        return torch.cat([rel, geo], dim=1)

    omega = 7.292115e-5
    coriolis = 2.0 * omega * torch.sin(latr)
    cos_lat = torch.cos(latr)
    extra = torch.cat([coriolis, cos_lat], dim=1)
    return torch.cat([rel, geo, extra], dim=1)


def encode_time_list_ymdh_to_tensor(time_list, device):
    feats = []
    for time_str in time_list:
        dt = datetime.strptime(time_str, "%Y-%m-%d-%H")
        hour = dt.hour
        doy = dt.timetuple().tm_yday
        theta_d = 2 * np.pi * hour / 24.0
        theta_y = 2 * np.pi * (doy - 1) / 365.2425
        feats.append([np.sin(theta_d), np.cos(theta_d), np.sin(theta_y), np.cos(theta_y)])
    return torch.tensor(feats, dtype=torch.float32, device=device)  # [B,4]


# =========================================================
# Visualization
# =========================================================
def show_samples(x, title="Samples", save_dir="./checkpoints", epoch=None, vmin=270, vmax=290):
    if not is_main_process():
        return

    x_cpu = x.detach().cpu().numpy()
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(min(x_cpu.shape[0], 9)):
        axes[i].imshow(x_cpu[i].squeeze(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if epoch is not None:
        filename = os.path.join(save_dir, f"{title.replace(' ', '_')}_epoch_{epoch}_serd_v1_stage1.png")
    else:
        filename = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")

    plt.savefig(filename)
    plt.close()
    print(f"[rank0] Saved samples to {filename}")


# =========================================================
# Data / Batch Helpers
# =========================================================
def load_topography(args, device):
    assert os.path.exists(args.topo_path), f"Topography file not found: {args.topo_path}"
    topo_np = np.load(args.topo_path)
    assert topo_np.ndim == 4, f"Expected topo npy to be 4D [1,C,H,W], got shape={topo_np.shape}"
    assert topo_np.shape[0] == 1, f"Expected topo batch dim to be 1 for broadcasting, got shape={topo_np.shape}"
    assert topo_np.shape[2] >= args.height and topo_np.shape[3] >= args.width, (
        f"Topography spatial size {topo_np.shape[2:]} is smaller than target {(args.height, args.width)}"
    )

    topo_base = torch.from_numpy(topo_np).to(device=device, dtype=torch.float32)
    topo_base = topo_base[:, :, :args.height, :args.width]

    coord_feats = build_geo_channels(
        args.height,
        args.width,
        lon_min=114.01,
        lon_max=119.74,
        lat_min=29.019999,
        lat_max=34.749999,
        device=device,
        dtype=topo_base.dtype,
        add_extras=False,
    )
    topo_base = torch.cat([topo_base, coord_feats], dim=1)
    return topo_base


def prepare_batch(fc, err, label, valid_time, init_time, topo_base, device, args):
    assert fc.ndim == 4, f"Expected fc to be 4D [B,C,H,W], got shape={tuple(fc.shape)}"
    assert err.ndim == 4, f"Expected err to be 4D [B,C,H,W], got shape={tuple(err.shape)}"

    batch_size = fc.shape[0]
    expected_fc_channels = args.num_surface_vars * args.num_levels
    expected_err_channels = args.num_surface_vars

    assert fc.shape[1] == expected_fc_channels, (
        f"fc channel mismatch: expected {expected_fc_channels}, got {fc.shape[1]}"
    )
    assert err.shape[1] == expected_err_channels, (
        f"err channel mismatch: expected {expected_err_channels}, got {err.shape[1]}"
    )
    assert fc.shape[2] >= args.height and fc.shape[3] >= args.width, (
        f"fc spatial size {tuple(fc.shape[2:])} smaller than target {(args.height, args.width)}"
    )
    assert err.shape[2] >= args.height and err.shape[3] >= args.width, (
        f"err spatial size {tuple(err.shape[2:])} smaller than target {(args.height, args.width)}"
    )

    target_error = err[:, :, :args.height, :args.width].to(device, non_blocking=True)
    forecast_surface_2d = fc[:, ::args.num_levels, :args.height, :args.width].to(device, non_blocking=True)

    forecast_raw = fc[:, :, :args.height, :args.width].to(device, non_blocking=True)
    forecast_3d = forecast_raw.reshape(
        batch_size, args.num_surface_vars, args.num_levels, args.height, args.width
    )

    topo_data = topo_base.expand(batch_size, -1, -1, -1).to(dtype=target_error.dtype)
    label_tensor = label.long().to(device, non_blocking=True)
    valid_time_tensor = encode_time_list_ymdh_to_tensor(valid_time, device)
    init_time_tensor = encode_time_list_ymdh_to_tensor(init_time, device)
    cond_2d = torch.cat([topo_data, forecast_surface_2d], dim=1)

    return {
        "batch_size": batch_size,
        "target_error": target_error,
        "forecast_surface_2d": forecast_surface_2d,
        "forecast_3d": forecast_3d,
        "label": label_tensor,
        "valid_time": valid_time_tensor,
        "init_time": init_time_tensor,
        "cond_2d": cond_2d,
    }


def run_model(model, batch):
    return model(
        cond_2d=batch["cond_2d"],
        forecast_3d=batch["forecast_3d"],
        y=batch["label"],
        obs_time=batch["valid_time"],
        init_time=batch["init_time"],
    )


# =========================================================
# Metrics / Eval
# =========================================================
@torch.no_grad()
def evaluate_regression(
    model,
    device,
    test_loader,
    topo_base,
    args,
    phys_mean,
    phys_std,
    current_epoch=None,
):
    rank = dist.get_rank() if is_dist_initialized() else 0
    cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.eval_seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.eval_seed + rank)

        model.eval()
        mse_total_phys = 0.0
        n_phys_points = 0

        mse_total_norm = 0.0
        n_norm_points = 0

        channel_idx = args.phys_eval_channel_idx
        last_pred_phys = None
        last_true_phys = None

        for fc, err, label, valid_time, init_time in tqdm(
            test_loader,
            total=len(test_loader),
            desc="Eval",
            unit="batch",
            disable=not is_main_process(),
        ):
            batch = prepare_batch(fc, err, label, valid_time, init_time, topo_base, device, args)
            pred_error = run_model(model, batch)
            target_error = batch["target_error"]

            mse_total_norm += float(torch.sum((pred_error - target_error) ** 2).item())
            n_norm_points += int(np.prod(target_error.shape))

            y_norm = target_error[:, channel_idx, :, :].detach().cpu().numpy()
            p_norm = pred_error[:, channel_idx, :, :].detach().cpu().numpy()

            height, width = y_norm.shape[-2], y_norm.shape[-1]
            n_phys_points += int(batch["batch_size"] * height * width)

            y_phys = np.empty((batch["batch_size"], height, width), dtype=np.float32)
            p_phys = np.empty((batch["batch_size"], height, width), dtype=np.float32)
            for i in range(batch["batch_size"]):
                y_phys[i] = y_norm[i] * phys_std + phys_mean
                p_phys[i] = p_norm[i] * phys_std + phys_mean

            mse_total_phys += float(np.sum((p_phys - y_phys) ** 2))

            if last_pred_phys is None and args.save_images and is_main_process():
                last_pred_phys = torch.from_numpy(p_phys)
                last_true_phys = torch.from_numpy(y_phys)

        reduce_tensor = torch.tensor(
            [
                mse_total_phys, float(n_phys_points),
                mse_total_norm, float(n_norm_points),
            ],
            dtype=torch.float64,
            device=device if torch.cuda.is_available() else torch.device("cpu"),
        )

        if is_dist_initialized():
            dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)

        mse_total_phys_r, n_phys_r, mse_total_norm_r, n_norm_r = reduce_tensor.tolist()
        n_phys_r = max(1.0, n_phys_r)
        n_norm_r = max(1.0, n_norm_r)

        avg_mse_phys = mse_total_phys_r / n_phys_r
        avg_mse_norm = mse_total_norm_r / n_norm_r

        if last_pred_phys is not None and args.save_images and is_main_process():
            show_samples(
                last_pred_phys,
                title=f"Pred_Err_Phys_Channel_{channel_idx}",
                save_dir=args.save_dir,
                epoch=current_epoch,
            )
            show_samples(
                last_true_phys,
                title=f"True_Err_Phys_Channel_{channel_idx}",
                save_dir=args.save_dir,
                epoch=current_epoch,
            )

        return avg_mse_phys, avg_mse_norm


# =========================================================
# Checkpoint
# =========================================================
def save_checkpoint(path, model, optimizer, scheduler, ema, epoch, best_metric, args):
    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "best_metric": best_metric,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


# =========================================================
# Main
# =========================================================
def main():
    validate_args(args)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = os.path.join(args.data_dir, "triton_cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(args.data_dir, "torch_inductor")

    rank, world_size, local_rank, device = setup_distributed(
        args.dist_backend,
        preferred_device=args.device,
    )
    setup_seed(args.seed + rank)

    if is_main_process():
        print(f"DDP training on device: {device}, world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print("Task: regression mean model for system-bias-like error")
        print("Loss = MSE")

    if is_main_process():
        print("Loading datasets...")

    forecast_scaler_path = os.path.join(args.data_dir, "scalers_forecast_zscore_two_step_unet_train.pkl")
    error_scaler_path = os.path.join(args.data_dir, "scalers_ana_zscore_two_step_unet_train.pkl")

    assert os.path.exists(forecast_scaler_path), f"Forecast normalizer file not found: {forecast_scaler_path}"
    assert os.path.exists(error_scaler_path), f"Error normalizer file not found: {error_scaler_path}"

    all_filepaths = sorted(glob.glob(args.data_root_glob))
    assert len(all_filepaths) > args.train_count + args.valid_count, (
        f"Total files={len(all_filepaths)} must be > "
        f"train_count + valid_count = {args.train_count + args.valid_count}"
    )

    normalizer_forecast = DataNormalizer_fc.load(forecast_scaler_path)
    normalizer_err = DataNormalizer_err.load(error_scaler_path)

    if is_main_process():
        print("Total files:", len(all_filepaths))

    train_files = all_filepaths[:args.train_count]
    valid_files = all_filepaths[args.train_count: args.train_count + args.valid_count]
    test_files = all_filepaths[args.train_count + args.valid_count:]
    assert len(train_files) > 0, "train_files is empty"
    assert len(valid_files) > 0, "valid_files is empty"
    assert len(test_files) > 0, "test_files is empty"

    train_dataset = ForecastDataset(train_files, normalizer_forecast, normalizer_err)
    valid_dataset = ForecastDataset(valid_files, normalizer_forecast, normalizer_err)
    test_dataset = ForecastDataset(test_files, normalizer_forecast, normalizer_err)

    num_workers = args.num_workers
    persistent_workers = num_workers > 0
    pin_memory = device.type == "cuda"

    if is_dist_initialized():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = None
        valid_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    if is_main_process():
        print(f"Split days: train={len(train_files)}, valid={len(valid_files)}, test={len(test_files)}")

    topo_base = load_topography(args, device)

    out_channels = args.num_surface_vars
    in_channels_3d = args.num_surface_vars
    in_channels_2d = topo_base.shape[1] + out_channels

    fixed_levels = torch.tensor(
        [1013.25, 925.0, 850.0, 700.0, 500.0, 300.0, 200.0, 150.0, 100.0],
        dtype=torch.float32,
    )

    base_model = forecast_mean_unet.ForecastMeanUNet3D2D(
        in_channels_2d=in_channels_2d,
        in_channels_3d=in_channels_3d,
        out_channels=out_channels,
        num_classes=args.num_classes,
        model_channels_2d=128,
        base_channels_3d=32,
        num_res_blocks_2d=1,
        num_res_blocks_3d=1,
        channel_mult_2d=(1, 2, 4, 4),
        dropout=0.2,
        ksize=3,
        dilations_2d=(1, 2),
        dilations_3d=(1, 1),
        attn_ds_2d=(4, 8, 16),
        attn_3d=True,
        use_label_cond=True,
        use_obs_time=True,
        pad_to_mult_of_32=True,
        fixed_z_coord_hpa=fixed_levels,
    ).to(device)

    if not args.disable_compile and hasattr(torch, "compile"):
        try:
            base_model = torch.compile(base_model)
        except Exception as exc:
            if is_main_process():
                print(f"torch.compile failed: {exc}, using uncompiled model")

    if is_dist_initialized():
        model = DDP(
            base_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=False,
        )
    else:
        model = base_model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=max(1e-6, args.lr * 0.1),
    )

    ema_params = model.module.parameters() if isinstance(model, DDP) else model.parameters()
    ema = ExponentialMovingAverage(ema_params, decay=args.ema_decay)

    scaler_stats = joblib.load(error_scaler_path)
    channel_idx = args.phys_eval_channel_idx
    assert -args.num_surface_vars <= channel_idx < args.num_surface_vars, (
        f"phys_eval_channel_idx={channel_idx} out of valid range for {args.num_surface_vars} channels"
    )
    phys_mean = float(scaler_stats["mean"][channel_idx])
    phys_std = float(scaler_stats["std"][channel_idx])

    all_train_losses = [] if is_main_process() else None
    all_eval_phys = [] if is_main_process() else None
    all_eval_norm = [] if is_main_process() else None

    best_metric = float("inf")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = build_grad_scaler(device_type, use_bf16)

    try:
        for epoch in range(args.epochs):
            model.train()
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            start_time = time.time()
            total_loss = 0.0
            num_batches = 0

            for fc, err, label, valid_time, init_time in tqdm(
                train_loader,
                desc=f"Train (epoch {epoch + 1})",
                unit="batch",
                disable=not is_main_process(),
            ):
                batch = prepare_batch(fc, err, label, valid_time, init_time, topo_base, device, args)

                optimizer.zero_grad(set_to_none=True)

                with amp.autocast(
                    device_type=device_type,
                    dtype=amp_dtype,
                    enabled=(device_type == "cuda"),
                ):
                    pred_error = run_model(model, batch)
                    loss = F.mse_loss(pred_error, batch["target_error"])

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                ema.update()

                total_loss += float(loss.item())
                num_batches += 1

            loss_sum_tensor = torch.tensor(
                [total_loss, num_batches],
                dtype=torch.float64,
                device=device if torch.cuda.is_available() else torch.device("cpu"),
            )

            if is_dist_initialized():
                dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)

            denom = torch.clamp_min(loss_sum_tensor[1], 1.0)
            avg_train_loss = (loss_sum_tensor[0] / denom).item()

            scheduler.step()

            if is_main_process():
                elapsed = time.time() - start_time
                print(
                    f"[epoch {epoch + 1}/{args.epochs}] "
                    f"Train MSE={avg_train_loss:.6e} | "
                    f"Time={elapsed:.2f}s | "
                    f"LR={scheduler.get_last_lr()[0]:.6g}"
                )
                all_train_losses.append(avg_train_loss)

            if (epoch + 1) % args.eval_every == 0:
                with ema.average_parameters():
                    eval_model = model.module if isinstance(model, DDP) else model
                    mse_phys, mse_norm = evaluate_regression(
                        eval_model,
                        device,
                        valid_loader,
                        topo_base,
                        args,
                        phys_mean=phys_mean,
                        phys_std=phys_std,
                        current_epoch=epoch + 1,
                    )

                if is_main_process():
                    print(
                        f"[epoch {epoch + 1}] "
                        f"Valid MSE (phys, channel={channel_idx}) = {mse_phys:.6e} | "
                        f"Valid MSE (norm, all) = {mse_norm:.6e}"
                    )

                    all_eval_phys.append(mse_phys)
                    all_eval_norm.append(mse_norm)

                    if (epoch + 1) % args.save_every == 0:
                        ckpt_path = os.path.join(args.save_dir, f"reg_sysbias_epoch_{epoch + 1}_serd_v1_stage1.pth")
                        save_checkpoint(
                            ckpt_path,
                            model,
                            optimizer,
                            scheduler,
                            ema,
                            epoch + 1,
                            best_metric,
                            args,
                        )

                    if mse_norm < best_metric:
                        best_metric = mse_norm
                        best_path = os.path.join(args.save_dir, "reg_sysbias_best_serd_v1_stage1.pth")
                        save_checkpoint(
                            best_path,
                            model,
                            optimizer,
                            scheduler,
                            ema,
                            epoch + 1,
                            best_metric,
                            args,
                        )
                        print(f"[rank0] New best checkpoint saved to {best_path}")

            if is_dist_initialized():
                dist.barrier()

    finally:
        if is_main_process():
            np.save(
                os.path.join(args.save_dir, "all_train_losses_reg_sysbias_serd_v1_stage1.npy"),
                np.array(all_train_losses if all_train_losses is not None else []),
            )
            np.save(
                os.path.join(args.save_dir, "all_valid_mses_phys_reg_sysbias_serd_v1_stage1.npy"),
                np.array(all_eval_phys if all_eval_phys is not None else []),
            )
            np.save(
                os.path.join(args.save_dir, "all_valid_mses_norm_reg_sysbias_serd_v1_stage1.npy"),
                np.array(all_eval_norm if all_eval_norm is not None else []),
            )

            final_path = os.path.join(args.save_dir, "reg_sysbias_final_serd_v1_stage1.pth")
            save_checkpoint(final_path, model, optimizer, scheduler, ema, args.epochs, best_metric, args)
            print(f"[rank0] Training finished. Models and metrics were saved to {args.save_dir}")

        cleanup_distributed()


if __name__ == "__main__":
    main()


