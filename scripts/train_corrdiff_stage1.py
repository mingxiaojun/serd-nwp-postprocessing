# -*- coding: utf-8 -*-
import os
import time
import glob
import math
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

from serd.data.forecast_error_dataset import ForecastDataset
from serd.data.normalizer_forecast import DataNormalizer as DataNormalizer_fc
from serd.data.normalizer_error import DataNormalizer as DataNormalizer_err

# 杩欓噷淇濇寔鍜屼綘鐜板湪鐨勬ā鍨嬫ā鍧椾竴鑷?
from serd.models import forecast_mean_unet


# =========================================================
# Args
# =========================================================
parser = argparse.ArgumentParser(
    description="DDP Train Regression Mean Model (Residual-oriented system-bias training)"
)

# 鍩虹璁粌
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--num_workers", type=int, default=7)
parser.add_argument("--grad_clip", type=float, default=1.0)

# 鏁版嵁涓庝繚瀛?
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--save_dir", type=str, default="./outputs/checkpoints/corrdiff")
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

# 妯″瀷/鏁版嵁褰㈢姸
parser.add_argument("--height", type=int, default=192)
parser.add_argument("--width", type=int, default=192)
parser.add_argument("--num_surface_vars", type=int, default=5)
parser.add_argument("--num_levels", type=int, default=9)
parser.add_argument("--num_classes", type=int, default=24)

# 璇勪及
parser.add_argument("--eval_every", type=int, default=1)
parser.add_argument("--save_images", action="store_true")
parser.add_argument("--save_every", type=int, default=1)

# 鍏跺畠
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dist_backend", type=str, default="nccl")
parser.add_argument("--eval_seed", type=int, default=20251103)
parser.add_argument("--ema_decay", type=float, default=0.999)

# 娈嬪樊瀵煎悜鎹熷け
parser.add_argument("--lambda_point", type=float, default=0.10)
parser.add_argument("--lambda_cond", type=float, default=0.5)
parser.add_argument("--lambda_reg", type=float, default=0.20)
parser.add_argument("--lambda_low", type=float, default=0.15)
parser.add_argument("--lambda_tv", type=float, default=0.03)

parser.add_argument("--huber_beta", type=float, default=0.30)
parser.add_argument("--reg_block", type=int, default=16)
parser.add_argument("--lowpass_kernel", type=int, default=9)
parser.add_argument("--cond_min_group", type=int, default=4)

parser.add_argument(
    "--var_weights",
    type=float,
    nargs=5,
    default=[1.0, 1.0, 1.0, 1.0, 1.0],   # [q2m, u10, v10, sp, t2m]
    help="channel weights for [q2m, u10, v10, sp, t2m]"
)

parser.set_defaults(save_images=True)
args = parser.parse_args()


# =========================================================
# DDP Utils
# =========================================================
def setup_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(dist_backend="nccl"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend=dist_backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()
    else:
        rank, world_size, local_rank = 0, 1, 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


# =========================================================
# Geo / Time
# =========================================================
def build_geo_channels(
    H, W, lon_min, lon_max, lat_min, lat_max, device, dtype=torch.float32, add_extras=False
):
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
        indexing="ij",
    )
    rel = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1,2,H,W]

    lons = torch.linspace(lon_min, lon_max, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(1, 1, H, W)
    lats = torch.linspace(lat_min, lat_max, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(1, 1, H, W)

    lonr = torch.deg2rad(lons)
    latr = torch.deg2rad(lats)
    geo = torch.cat(
        [torch.sin(lonr), torch.cos(lonr), torch.sin(latr), torch.cos(latr)],
        dim=1,
    )  # [1,4,H,W]

    if not add_extras:
        return torch.cat([rel, geo], dim=1)

    OMEGA = 7.292115e-5
    f = 2.0 * OMEGA * torch.sin(latr)
    cosphi = torch.cos(latr)
    extra = torch.cat([f, cosphi], dim=1)
    return torch.cat([rel, geo, extra], dim=1)


def encode_time_list_ymdh_to_tensor(time_list, device):
    feats = []
    for s in time_list:
        dt = datetime.strptime(s, "%Y-%m-%d-%H")
        hour = dt.hour
        doy = dt.timetuple().tm_yday
        theta_d = 2 * np.pi * hour / 24.0
        theta_y = 2 * np.pi * (doy - 1) / 365.2425
        feats.append(
            [np.sin(theta_d), np.cos(theta_d), np.sin(theta_y), np.cos(theta_y)]
        )
    return torch.tensor(feats, dtype=torch.float32, device=device)  # (B,4)


# =========================================================
# Visualization
# =========================================================
def show_samples(x, title="Samples", save_dir="./checkpoints", epoch=None, vmin=-3, vmax=3):
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
        fname = os.path.join(save_dir, f"{title.replace(' ', '_')}_epoch_{epoch}_corrdiff_stage1.png")
    else:
        fname = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")

    plt.savefig(fname)
    plt.close()
    print(f"[rank0] Saved samples to {fname}")


# =========================================================
# Losses
# =========================================================
def get_channel_weights(device, dtype, weights):
    w = torch.tensor(weights, device=device, dtype=dtype)
    return w / w.sum().clamp_min(1e-12)


def reduce_channelwise(loss_c, channel_weights):
    w = channel_weights.to(device=loss_c.device, dtype=loss_c.dtype)
    w = w / w.sum().clamp_min(1e-12)
    return torch.sum(loss_c * w)


def weighted_huber_channelwise(pred, target, channel_weights, beta=0.5):
    """
    pred/target: (B,C,H,W)
    """
    loss_map = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    loss_c = loss_map.mean(dim=(0, 2, 3))  # (C,)
    total = reduce_channelwise(loss_c, channel_weights)
    return total, loss_c


def pred_tv_loss_channelwise(pred, channel_weights):
    """
    瀵归娴嬬殑绯荤粺璇樊鍦哄仛寮卞钩婊戠害鏉?
    """
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    loss_c = dx.abs().mean(dim=(0, 2, 3)) + dy.abs().mean(dim=(0, 2, 3))
    total = reduce_channelwise(loss_c, channel_weights)
    return total, loss_c


def regional_residual_zero_mean_loss_channelwise(residual, channel_weights, block=16):
    """
    绾︽潫娈嬪樊鍦ㄥ尯鍩熷钩鍧囧悗鎺ヨ繎 0
    residual: (B,C,H,W)
    """
    assert block >= 1
    if block == 1:
        pooled = residual
    else:
        pooled = F.avg_pool2d(residual, kernel_size=block, stride=block)
    loss_c = pooled.abs().mean(dim=(0, 2, 3))
    total = reduce_channelwise(loss_c, channel_weights)
    return total, loss_c


def lowfreq_residual_zero_mean_loss_channelwise(residual, channel_weights, kernel_size=9):
    """
    绾︽潫娈嬪樊鐨勪綆棰戦儴鍒嗘帴杩?0
    """
    assert kernel_size % 2 == 1, "lowpass kernel size must be odd"
    pad = kernel_size // 2
    residual_pad = F.pad(residual, (pad, pad, pad, pad), mode="reflect")
    residual_low = F.avg_pool2d(residual_pad, kernel_size=kernel_size, stride=1)
    loss_c = residual_low.abs().mean(dim=(0, 2, 3))
    total = reduce_channelwise(loss_c, channel_weights)
    return total, loss_c


def conditional_mean_residual_loss_channelwise(
    residual, label, channel_weights, min_group_size=2
):
    """
    绾︽潫 batch 鍐呯浉鍚?lead 鐨勬畫宸潯浠跺潎鍊兼帴杩?0
    residual: (B,C,H,W)
    label: (B,)
    """
    uniq = torch.unique(label)
    loss_c_sum = torch.zeros(residual.shape[1], device=residual.device, dtype=residual.dtype)
    group_count = 0

    for g in uniq:
        mask = (label == g)
        n = int(mask.sum().item())
        if n < min_group_size:
            continue
        r_g = residual[mask].mean(dim=0)           # (C,H,W)
        cur_c = r_g.abs().mean(dim=(1, 2))         # (C,)
        loss_c_sum += cur_c
        group_count += 1

    if group_count == 0:
        zero = residual.new_tensor(0.0)
        zero_c = torch.zeros(residual.shape[1], device=residual.device, dtype=residual.dtype)
        return zero, zero_c

    loss_c = loss_c_sum / group_count
    total = reduce_channelwise(loss_c, channel_weights)
    return total, loss_c


# =========================================================
# Metrics / Eval
# =========================================================
@torch.no_grad()
def lowpass_field(x, kernel_size=9):
    assert kernel_size % 2 == 1
    pad = kernel_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x_pad, kernel_size=kernel_size, stride=1)


@torch.no_grad()
def evaluate_regression(model, device, test_loader, topo_base, args, current_epoch=None):
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
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

        low_total_norm = 0.0
        n_low_points = 0

        channel_idx = -1
        scalers = joblib.load(os.path.join(args.data_dir, "scalers_err_zscore_two_step_unet_train.pkl"))
        mean_c = scalers["mean"][channel_idx]
        std_c = scalers["std"][channel_idx]

        last_pred_phys = None
        last_true_phys = None

        for fc, err, label, valid_time, init_time in tqdm(
            test_loader,
            total=len(test_loader),
            desc="Eval",
            unit="batch",
            disable=not is_main_process(),
        ):
            B = fc.shape[0]

            target_err = err[:, :, :args.height, :args.width].to(device, non_blocking=True)  # (B,5,H,W)
            forecast_2d = fc[:, ::args.num_levels, :args.height, :args.width].to(device, non_blocking=True)  # (B,5,H,W)

            forecast_raw = fc[:, :, :args.height, :args.width].to(device, non_blocking=True)
            forecast_3d = forecast_raw.reshape(
                B, args.num_surface_vars, args.num_levels, args.height, args.width
            )

            topo_data = topo_base.expand(B, -1, -1, -1).to(dtype=target_err.dtype)

            label = label.long().to(device, non_blocking=True)
            valid_time = encode_time_list_ymdh_to_tensor(valid_time, device)
            init_time = encode_time_list_ymdh_to_tensor(init_time, device)

            cond_2d = torch.cat([topo_data, forecast_2d], dim=1)

            pred_err = model(
                cond_2d=cond_2d,
                forecast_3d=forecast_3d,
                y=label,
                obs_time=valid_time,
                init_time=init_time,
            )

            mse_total_norm += float(torch.sum((pred_err - target_err) ** 2).item())
            n_norm_points += int(np.prod(target_err.shape))

            pred_low = lowpass_field(pred_err, kernel_size=args.lowpass_kernel)
            true_low = lowpass_field(target_err, kernel_size=args.lowpass_kernel)
            low_total_norm += float(torch.sum((pred_low - true_low) ** 2).item())
            n_low_points += int(np.prod(pred_low.shape))

            y_norm = target_err[:, channel_idx, :, :].detach().cpu().numpy()
            p_norm = pred_err[:, channel_idx, :, :].detach().cpu().numpy()

            H, W = y_norm.shape[-2], y_norm.shape[-1]
            n_phys_points += int(B * H * W)

            y_phys = np.empty((B, H, W), dtype=np.float32)
            p_phys = np.empty((B, H, W), dtype=np.float32)
            for i in range(B):
                y_phys[i] = y_norm[i] * std_c + mean_c
                p_phys[i] = p_norm[i] * std_c + mean_c

            mse_total_phys += float(np.sum((p_phys - y_phys) ** 2))

            if last_pred_phys is None and args.save_images and is_main_process():
                last_pred_phys = torch.from_numpy(p_phys)
                last_true_phys = torch.from_numpy(y_phys)

        dev = device if torch.cuda.is_available() else torch.device("cpu")
        t = torch.tensor(
            [
                mse_total_phys, float(n_phys_points),
                mse_total_norm, float(n_norm_points),
                low_total_norm, float(n_low_points),
            ],
            dtype=torch.float64,
            device=dev,
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        mse_total_phys_r, n_phys_r, mse_total_norm_r, n_norm_r, low_total_norm_r, n_low_r = t.tolist()
        n_phys_r = max(1.0, n_phys_r)
        n_norm_r = max(1.0, n_norm_r)
        n_low_r = max(1.0, n_low_r)

        avg_mse_phys = mse_total_phys_r / n_phys_r
        avg_mse_norm = mse_total_norm_r / n_norm_r
        avg_low_mse_norm = low_total_norm_r / n_low_r

        if last_pred_phys is not None and args.save_images and is_main_process():
            show_samples(last_pred_phys, title="Pred_Err_Phys_Channel_-1", save_dir=args.save_dir, epoch=current_epoch)
            show_samples(last_true_phys, title="True_Err_Phys_Channel_-1", save_dir=args.save_dir, epoch=current_epoch)

        return avg_mse_phys, avg_mse_norm, avg_low_mse_norm


# =========================================================
# Checkpoint
# =========================================================
def save_checkpoint(path, model, optimizer, scheduler, ema, epoch, best_metric, args):
    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    ckpt = {
        "epoch": epoch,
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "best_metric": best_metric,
        "args": vars(args),
    }
    torch.save(ckpt, path)


# =========================================================
# Main
# =========================================================
def main():
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = os.path.join(args.data_dir, "triton_cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(args.data_dir, "torch_inductor")

    rank, world_size, local_rank, device = setup_distributed(args.dist_backend)
    setup_seed(args.seed + rank)

    if is_main_process():
        print(f"DDP training on device: {device}, world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print("Task: regression mean model for system-bias-like error")
        print(
            f"Loss = "
            f"{args.lambda_point}*point + "
            f"{args.lambda_cond}*cond_mean(residual) + "
            f"{args.lambda_reg}*regional(residual) + "
            f"{args.lambda_low}*lowfreq(residual) + "
            f"{args.lambda_tv}*tv(pred)"
        )
        print(f"Channel weights: {args.var_weights}")

    # --------------------- Data ---------------------
    if is_main_process():
        print("Loading datasets...")

    all_filepaths = sorted(glob.glob(args.data_root_glob))
    normalizer_forecast = DataNormalizer_fc.load(
        os.path.join(args.data_dir, "scalers_forecast_zscore_two_step_unet_train.pkl")
    )
    normalizer_err = DataNormalizer_err.load(
        os.path.join(args.data_dir, "scalers_err_zscore_two_step_unet_train.pkl")
    )

    if is_main_process():
        print("Total files:", len(all_filepaths))

    assert len(all_filepaths) > args.train_count + args.valid_count, (
        f"Total files={len(all_filepaths)} must be > train_count + valid_count"
    )

    train_files = all_filepaths[:args.train_count]
    valid_files = all_filepaths[args.train_count: args.train_count + args.valid_count]
    test_files = all_filepaths[args.train_count + args.valid_count:]

    train_dataset = ForecastDataset(train_files, normalizer_forecast, normalizer_err)
    test_dataset = ForecastDataset(valid_files, normalizer_forecast, normalizer_err)

    num_workers = args.num_workers
    persistent = num_workers > 0

    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=False,
    )

    if is_main_process():
        print("鏁版嵁瀵煎叆瀹屾垚")

    # --------------------- Topography + geo ---------------------
    topo_np = np.load(args.topo_path)
    topo_base = torch.from_numpy(topo_np).to(device=device, dtype=torch.float32)[:, :, :args.height, :args.width]

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
    topo_base = torch.cat([topo_base, coord_feats], dim=1)  # (1, topo_ch+6, H, W)

    # --------------------- Model ---------------------
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

    try:
        base_model = torch.compile(base_model)
    except Exception as e:
        if is_main_process():
            print(f"torch.compile failed: {e}, using uncompiled model")

    if dist.is_available() and dist.is_initialized():
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

    channel_weights = get_channel_weights(device, torch.float32, args.var_weights)

    all_train_losses = [] if is_main_process() else None
    all_eval_phys = [] if is_main_process() else None
    all_eval_norm = [] if is_main_process() else None
    all_eval_low = [] if is_main_process() else None

    best_metric = float("inf")
    torch.cuda.empty_cache()

    # AMP锛歜f16 浼樺厛锛宖p16 鍏滃簳锛沚f16 涓嶇敤 scaler 缂╂斁
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler(device_type, enabled=(device_type == "cuda") and (not use_bf16))

    # =========================================================
    # Train Loop
    # =========================================================
    for epoch in range(args.epochs):
        model.train()
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        start_time = time.time()
        total_loss = 0.0
        total_loss_point = 0.0
        total_loss_cond = 0.0
        total_loss_reg = 0.0
        total_loss_low = 0.0
        total_loss_tv = 0.0
        num_batches = 0

        for fc, err, label, valid_time, init_time in tqdm(
            train_loader,
            desc=f"Train (epoch {epoch + 1})",
            unit="batch",
            disable=not is_main_process(),
        ):
            B = fc.shape[0]

            target_err = err[:, :, :args.height, :args.width].to(device, non_blocking=True)  # (B,5,H,W)
            forecast_2d = fc[:, ::args.num_levels, :args.height, :args.width].to(device, non_blocking=True)  # (B,5,H,W)

            forecast_raw = fc[:, :, :args.height, :args.width].to(device, non_blocking=True)
            forecast_3d = forecast_raw.reshape(
                B, args.num_surface_vars, args.num_levels, args.height, args.width
            )  # (B,5,9,H,W)

            topo_data = topo_base.expand(B, -1, -1, -1).to(dtype=target_err.dtype)

            label = label.long().to(device, non_blocking=True)
            valid_time_tensor = encode_time_list_ymdh_to_tensor(valid_time, device)
            init_time_tensor = encode_time_list_ymdh_to_tensor(init_time, device)

            cond_2d = torch.cat([topo_data, forecast_2d], dim=1)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=(device_type == "cuda")):
                pred_err = model(
                    cond_2d=cond_2d,
                    forecast_3d=forecast_3d,
                    y=label,
                    obs_time=valid_time_tensor,
                    init_time=init_time_tensor,
                )  # (B,5,H,W)

                residual = target_err - pred_err

                # 1) 寮辩偣瀵圭偣绾︽潫锛氬彧璐熻矗绋宠缁冿紝涓嶄綔涓轰富瀵?
                loss_point, _ = weighted_huber_channelwise(
                    pred_err, target_err, channel_weights, beta=args.huber_beta
                )

                # 2) batch 鍐呯浉鍚?lead 鐨勬畫宸潯浠跺潎鍊?-> 0
                loss_cond, _ = conditional_mean_residual_loss_channelwise(
                    residual,
                    label,
                    channel_weights,
                    min_group_size=args.cond_min_group,
                )

                # 3) 娈嬪樊鍖哄煙鍧囧€?-> 0
                loss_reg, _ = regional_residual_zero_mean_loss_channelwise(
                    residual,
                    channel_weights,
                    block=args.reg_block,
                )

                # 4) 娈嬪樊浣庨 -> 0
                loss_low, _ = lowfreq_residual_zero_mean_loss_channelwise(
                    residual,
                    channel_weights,
                    kernel_size=args.lowpass_kernel,
                )

                # 5) 棰勬祴绯荤粺璇樊鍦哄仛寮?TV
                loss_tv, _ = pred_tv_loss_channelwise(
                    pred_err,
                    channel_weights,
                )

                loss = (
                    args.lambda_point * loss_point
                    + args.lambda_cond * loss_cond
                    + args.lambda_reg * loss_reg
                    + args.lambda_low * loss_low
                    + args.lambda_tv * loss_tv
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            total_loss += float(loss.item())
            total_loss_point += float(loss_point.item())
            total_loss_cond += float(loss_cond.item())
            total_loss_reg += float(loss_reg.item())
            total_loss_low += float(loss_low.item())
            total_loss_tv += float(loss_tv.item())
            num_batches += 1

        loss_sum_t = torch.tensor(
            [
                total_loss,
                total_loss_point,
                total_loss_cond,
                total_loss_reg,
                total_loss_low,
                total_loss_tv,
                num_batches,
            ],
            dtype=torch.float64,
            device=device if torch.cuda.is_available() else torch.device("cpu"),
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)

        denom = torch.clamp_min(loss_sum_t[-1], 1.0)
        avg_train_loss = (loss_sum_t[0] / denom).item()
        avg_train_loss_point = (loss_sum_t[1] / denom).item()
        avg_train_loss_cond = (loss_sum_t[2] / denom).item()
        avg_train_loss_reg = (loss_sum_t[3] / denom).item()
        avg_train_loss_low = (loss_sum_t[4] / denom).item()
        avg_train_loss_tv = (loss_sum_t[5] / denom).item()

        scheduler.step()

        if is_main_process():
            dt = time.time() - start_time
            print(
                f"[epoch {epoch + 1}/{args.epochs}] "
                f"Train total={avg_train_loss:.6e} | "
                f"point={avg_train_loss_point:.6e} | "
                f"cond={avg_train_loss_cond:.6e} | "
                f"reg={avg_train_loss_reg:.6e} | "
                f"low={avg_train_loss_low:.6e} | "
                f"tv={avg_train_loss_tv:.6e} | "
                f"Time={dt:.2f}s | LR={scheduler.get_last_lr()[0]:.6g}"
            )
            all_train_losses.append(avg_train_loss)

        # --------------------- Eval ---------------------
        if (epoch + 1) % args.eval_every == 0:
            with ema.average_parameters():
                eval_model = model.module if isinstance(model, DDP) else model
                mse_phys, mse_norm, low_mse_norm = evaluate_regression(
                    eval_model, device, test_loader, topo_base, args, current_epoch=epoch + 1
                )

            if is_main_process():
                print(
                    f"[epoch {epoch + 1}] "
                    f"Test MSE (phys, channel=-1) = {mse_phys:.6e} | "
                    f"Test MSE (norm, all) = {mse_norm:.6e} | "
                    f"Test LowFreq MSE (norm, all) = {low_mse_norm:.6e}"
                )

                all_eval_phys.append(mse_phys)
                all_eval_norm.append(mse_norm)
                all_eval_low.append(low_mse_norm)

                if (epoch + 1) % args.save_every == 0:
                    ckpt_path = os.path.join(args.save_dir, f"reg_sysbias_epoch_{epoch + 1}_corrdiff_stage1.pth")
                    save_checkpoint(
                        ckpt_path, model, optimizer, scheduler, ema, epoch + 1, best_metric, args
                    )

                # 鏇寸鍚堜綘鐩爣锛氱敤浣庨 MSE 浣滀负 best 鎸囨爣
                if low_mse_norm < best_metric:
                    best_metric = low_mse_norm
                    best_path = os.path.join(args.save_dir, "reg_sysbias_best_corrdiff_stage1.pth")
                    save_checkpoint(
                        best_path, model, optimizer, scheduler, ema, epoch + 1, best_metric, args
                    )
                    print(f"[rank0] New best checkpoint saved to {best_path}")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # --------------------- Final save ---------------------
    if is_main_process():
        np.save(os.path.join(args.save_dir, "all_train_losses_reg_sysbias_corrdiff_stage1.npy"), np.array(all_train_losses))
        np.save(os.path.join(args.save_dir, "all_valid_mses_phys_reg_sysbias_corrdiff_stage1.npy"), np.array(all_eval_phys))
        np.save(os.path.join(args.save_dir, "all_valid_mses_norm_reg_sysbias_corrdiff_stage1.npy"), np.array(all_eval_norm))
        np.save(os.path.join(args.save_dir, "all_test_low_mses_norm_reg_sysbias_corrdiff_stage1.npy"), np.array(all_eval_low))

        final_path = os.path.join(args.save_dir, "reg_sysbias_final_corrdiff_stage1.pth")
        save_checkpoint(final_path, model, optimizer, scheduler, ema, args.epochs, best_metric, args)
        print(f"[rank0] 璁粌瀹屾垚锛屾ā鍨嬪拰缁撴灉宸蹭繚瀛樺埌 {args.save_dir}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

