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
from serd.models import physcond_error_diffusion


torch.set_float32_matmul_precision("high")


# =========================================================
# Args
# =========================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="Train Table 2 direct total-error diffusion with score loss and fCRPS"
    )

    # 鍩虹璁粌
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=500, help="total target epochs, not extra epochs")
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=7)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # 鏁版嵁涓庝繚瀛?
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./outputs/checkpoints/direct_diffusion_fcrps")
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
    parser.add_argument(
        "--forecast_scaler_path",
        type=str,
        default="./data/scalers_forecast_zscore_two_step_unet_train.pkl",
    )
    parser.add_argument(
        "--err_scaler_path",
        type=str,
        default="./data/scalers_err_zscore_two_step_unet_train.pkl",
    )
    parser.add_argument("--train_count", type=int, default=1292)
    parser.add_argument("--valid_count", type=int, default=92)

    # resume
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint path to resume training from",
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="only load model weights, do not restore optimizer/scheduler/ema/scaler/epoch",
    )
    parser.add_argument(
        "--load_strict",
        action="store_true",
        help="strictly load model_state_dict; default is non-strict for easier architecture migration",
    )

    # 妯″瀷 / 鏁版嵁褰㈢姸
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--num_surface_vars", type=int, default=5)
    parser.add_argument("--num_levels", type=int, default=9)
    parser.add_argument("--num_classes", type=int, default=24)

    # sde
    parser.add_argument("--sigma_min", type=float, default=2e-2)
    parser.add_argument("--sigma_max", type=float, default=10.0)
    parser.add_argument("--sigma_cap", type=float, default=10.0)
    parser.add_argument("--sigma_jitter_log", type=float, default=0.0)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--edm_rho", type=float, default=7.0)

    # train loss
    parser.add_argument("--k_crps", type=int, default=6, help="Training ensemble size for fair CRPS (>=2)")
    parser.add_argument("--lambda_score", type=float, default=1.0)
    parser.add_argument("--lambda_fcrps", type=float, default=0.5)

    # eval sampler
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--eval_seed", type=int, default=20251103)
    parser.add_argument("--edm_steps", type=int, default=40)
    parser.add_argument("--edm_sigma_min", type=float, default=2e-2)
    parser.add_argument("--edm_sigma_max", type=float, default=10.0)
    parser.add_argument("--pc_corrector_steps", type=int, default=1)
    parser.add_argument("--pc_snr", type=float, default=0.16)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--phys_eval_channel_idx", type=int, default=-1)

    # 鍏跺畠
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="preferred device in non-DDP mode",
    )
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--disable_compile", action="store_true")

    # 妯″瀷瓒呭弬鏁?
    parser.add_argument("--model_channels_2d", type=int, default=128)
    parser.add_argument("--base_channels_fcst", type=int, default=32)
    parser.add_argument("--num_res_blocks_2d", type=int, default=2)
    parser.add_argument("--num_res_blocks_fcst", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ksize", type=int, default=3)

    parser.set_defaults(save_images=True)
    return parser


parser = build_parser()
args = parser.parse_args()


# =========================================================
# Utils
# =========================================================
def validate_args(args):
    assert args.batch_size >= 1
    assert args.lr > 0
    assert args.epochs >= 1
    assert args.weight_decay >= 0
    assert args.num_workers >= 0
    assert args.grad_clip > 0
    assert args.train_count >= 1
    assert args.valid_count >= 1
    assert args.height >= 1 and args.width >= 1
    assert args.num_surface_vars >= 1
    assert args.num_levels >= 1
    assert args.num_classes >= 1
    assert args.eval_every >= 1
    assert args.save_every >= 1
    assert 0.0 < args.ema_decay < 1.0
    assert args.sigma_min > 0
    assert args.sigma_max >= args.sigma_min
    assert args.sigma_cap >= args.sigma_min
    assert args.k_crps >= 2
    assert args.lambda_score >= 0
    assert args.lambda_fcrps >= 0
    if args.resume:
        assert os.path.exists(args.resume), f"Resume checkpoint not found: {args.resume}"


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


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def strip_state_dict_prefixes(state_dict):
    clean = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        clean[nk] = v
    return clean


def get_clean_model_state_dict(model: torch.nn.Module):
    return strip_state_dict_prefixes(unwrap_model(model).state_dict())


def load_training_checkpoint(path, device):
    assert os.path.exists(path), f"Checkpoint not found: {path}"
    ckpt = torch.load(path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"{path} does not contain 'model_state_dict'")
    ckpt["model_state_dict"] = strip_state_dict_prefixes(ckpt["model_state_dict"])
    return ckpt


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
    rel = torch.stack([xx, yy], dim=0).unsqueeze(0)

    lons = torch.linspace(lon_min, lon_max, width, device=device, dtype=dtype).view(1, 1, 1, width).expand(1, 1, height, width)
    lats = torch.linspace(lat_min, lat_max, height, device=device, dtype=dtype).view(1, 1, height, 1).expand(1, 1, height, width)

    lonr = torch.deg2rad(lons)
    latr = torch.deg2rad(lats)
    geo = torch.cat(
        [torch.sin(lonr), torch.cos(lonr), torch.sin(latr), torch.cos(latr)],
        dim=1,
    )

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
    return torch.tensor(feats, dtype=torch.float32, device=device)


# =========================================================
# Diffusion helpers
# =========================================================
class VESDE:
    def __init__(self, sigma_min, sigma_max, N):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.N = int(N)
        self.log_ratio = np.log(self.sigma_max / self.sigma_min)
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.N)
        )

    def sigma(self, t: torch.Tensor):
        return torch.exp(
            torch.log(torch.tensor(self.sigma_min, device=t.device, dtype=t.dtype))
            + t * torch.tensor(self.log_ratio, device=t.device, dtype=t.dtype)
        )


def get_karras_sigmas(num_steps, sigma_min, sigma_max, rho=7.0, device=torch.device("cpu")):
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def sample_sigma_karras(B, sigma_min, sigma_max, rho, device, dtype):
    inv_r = 1.0 / rho
    sigma_min_r = torch.as_tensor(sigma_min, device=device, dtype=dtype) ** inv_r
    sigma_max_r = torch.as_tensor(sigma_max, device=device, dtype=dtype) ** inv_r
    u = torch.rand(B, device=device, dtype=dtype)
    sig = (sigma_max_r + u * (sigma_min_r - sigma_max_r)) ** rho
    return sig.view(B, 1, 1, 1)


def ensemble_fcrps(ens: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    ens: (K, B, C, H, W)
    y:   (B, C, H, W)
    return: scalar fair CRPS averaged over all dims
    """
    K = int(ens.shape[0])
    if K < 2:
        raise ValueError("fCRPS requires ensemble size K>=2.")

    term1 = torch.mean(torch.abs(ens - y.unsqueeze(0)))

    term2 = ens.new_tensor(0.0)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            term2 = term2 + torch.mean(torch.abs(ens[i] - ens[j]))
    term2 = term2 / (2.0 * K * (K - 1))
    return term1 - term2


# =========================================================
# Visualization
# =========================================================
def show_samples(x, title="Samples", save_dir="./checkpoints", epoch=None, vmin=270, vmax=290):
    if not is_main_process():
        return

    x_cpu = x.detach().cpu().numpy()
    os.makedirs(save_dir, exist_ok=True)

    n_show = min(x_cpu.shape[0], 9)
    ncols = min(3, n_show)
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(n_show):
        axes[i].imshow(x_cpu[i].squeeze(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[i].axis("off")

    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if epoch is not None:
        filename = os.path.join(save_dir, f"{title.replace(' ', '_')}_epoch_{epoch}_direct_diffusion_fcrps.png")
    else:
        filename = os.path.join(save_dir, f"{title.replace(' ', '_')}_direct_diffusion_fcrps.png")

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
    assert fc.shape[2] >= args.height and fc.shape[3] >= args.width
    assert err.shape[2] >= args.height and err.shape[3] >= args.width

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
        "topo_data": topo_data,
        "label": label_tensor,
        "valid_time": valid_time_tensor,
        "init_time": init_time_tensor,
        "cond_2d": cond_2d,
    }


def run_model(model, x_t, sigma, batch, z_coord_hpa=None):
    return model(
        x_t=x_t,
        static_2d=batch["cond_2d"],
        sigma=sigma,
        forecast_3d=batch["forecast_3d"],
        z_coord_hpa=z_coord_hpa,
        y=batch["label"],
        obs_time=batch["valid_time"],
        init_time=batch["init_time"],
    )


# =========================================================
# Sampler / Eval
# =========================================================
@torch.no_grad()
def pc_sampler(
    model,
    shape,
    device,
    batch,
    z_coord_hpa,
    sigma_min=2e-2,
    sigma_max=10.0,
    rho=7.0,
    steps=40,
    snr=0.16,
    corrector_steps=1,
):
    B, C, H, W = shape
    sigmas = get_karras_sigmas(
        num_steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
    ).to(dtype=torch.float32)
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=sigmas.dtype)], dim=0)

    x = torch.randn(B, C, H, W, device=device, dtype=torch.float32) * sigmas[0]
    model.eval()

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_t = torch.full((B, 1, 1, 1), float(sigma.item()), device=device, dtype=x.dtype)

        if sigma.item() > 0:
            for _ in range(corrector_steps):
                score = run_model(model, x_t=x, sigma=sigma_t, batch=batch, z_coord_hpa=z_coord_hpa)
                noise = torch.randn_like(x)
                grad_norm = torch.norm(score.reshape(B, -1), dim=1).mean().clamp_min(1e-12)
                noise_norm = torch.norm(noise.reshape(B, -1), dim=1).mean().clamp_min(1e-12)
                step_size = (snr * noise_norm / grad_norm) ** 2 * 2.0
                x_mean = x + step_size * score
                x = x_mean + torch.sqrt(2.0 * step_size) * noise

        score = run_model(model, x_t=x, sigma=sigma_t, batch=batch, z_coord_hpa=z_coord_hpa)

        sigma_next_t = torch.full((B, 1, 1, 1), float(sigma_next.item()), device=device, dtype=x.dtype)
        sigma2 = sigma_t ** 2
        sigma_next2 = sigma_next_t ** 2
        x_mean = x + (sigma2 - sigma_next2) * score

        if sigma_next.item() > 0:
            std = torch.sqrt(
                (sigma_next2 * (sigma2 - sigma_next2) / sigma2.clamp_min(1e-12)).clamp_min(0.0)
            )
            x = x_mean + std * torch.randn_like(x)
        else:
            x = x_mean

    return x


@torch.no_grad()
def evaluate_diffusion(
    model,
    device,
    sde,
    test_loader,
    topo_base,
    args,
    phys_mean,
    phys_std,
    z_coord_hpa,
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
            pred_error = pc_sampler(
                model=model,
                shape=batch["target_error"].shape,
                device=device,
                batch=batch,
                z_coord_hpa=z_coord_hpa.expand(batch["batch_size"], -1),
                sigma_min=max(float(args.edm_sigma_min), float(sde.sigma_min)),
                sigma_max=min(float(args.edm_sigma_max), float(sde.sigma_max)),
                rho=args.edm_rho,
                steps=args.edm_steps,
                snr=args.pc_snr,
                corrector_steps=args.pc_corrector_steps,
            )
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
def save_training_checkpoint(path, model, optimizer, scheduler, ema, scaler, epoch, best_metric, args):
    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": get_clean_model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_metric": float(best_metric),
        "args": vars(args),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, path)


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
        print("Task: VE-SDE score+fCRPS training with first-script batch inputs")
        print(f"Loss = {args.lambda_score} * score + {args.lambda_fcrps} * fCRPS | k_crps={args.k_crps}")
        if args.resume:
            print(f"Resume path: {args.resume}")
            print(f"Resume weights only: {args.resume_weights_only}")

    assert os.path.exists(args.forecast_scaler_path), f"Forecast normalizer file not found: {args.forecast_scaler_path}"
    assert os.path.exists(args.err_scaler_path), f"Error normalizer file not found: {args.err_scaler_path}"

    all_filepaths = sorted(glob.glob(args.data_root_glob))
    assert len(all_filepaths) > args.train_count + args.valid_count, (
        f"Total files={len(all_filepaths)} must be > "
        f"train_count + valid_count = {args.train_count + args.valid_count}"
    )

    normalizer_forecast = DataNormalizer_fc.load(args.forecast_scaler_path)
    normalizer_err = DataNormalizer_err.load(args.err_scaler_path)

    if is_main_process():
        print("Total files:", len(all_filepaths))

    train_files = all_filepaths[:args.train_count]
    valid_files = all_filepaths[args.train_count: args.train_count + args.valid_count]
    test_files = all_filepaths[args.train_count + args.valid_count:]
    assert len(train_files) > 0
    assert len(valid_files) > 0
    assert len(test_files) > 0

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

    effective_sigma_max = float(min(args.sigma_cap, args.sigma_max))
    sde = VESDE(sigma_min=args.sigma_min, sigma_max=effective_sigma_max, N=args.N)

    fixed_levels = torch.tensor(
        [1013.25, 925.0, 850.0, 700.0, 500.0, 300.0, 200.0, 150.0, 100.0],
        dtype=torch.float32,
        device=device,
    )
    if args.num_levels != fixed_levels.numel():
        raise ValueError(
            f"args.num_levels={args.num_levels} but fixed pressure levels has {fixed_levels.numel()} levels; please edit fixed_levels."
        )

    in_channels_static = topo_base.shape[1] + args.num_surface_vars

    # 1) 鏋勫缓鍘熷妯″瀷
    base_model = physcond_error_diffusion.ForecastErrorUNet2D3D(
        in_channels_xt=args.num_surface_vars,
        in_channels_static=in_channels_static,
        in_channels_fcst=args.num_surface_vars,
        out_channels=args.num_surface_vars,
        num_classes=args.num_classes,
        sde=sde,
        model_channels_2d=args.model_channels_2d,
        base_channels_fcst=args.base_channels_fcst,
        num_res_blocks_2d=args.num_res_blocks_2d,
        num_res_blocks_fcst=args.num_res_blocks_fcst,
        channel_mult_2d=(1, 2, 4, 4),
        dropout=args.dropout,
        ksize=args.ksize,
        dilations_2d=(1, 2),
        dilations_3d=(1, 1),
        attn_ds_2d=(8, 16),
        attn_fcst=True,
        use_label_cond=True,
        use_obs_time=True,
        pad_to_mult_of_32=True,
        head_use_sigma=True,
        fixed_z_coord_hpa=fixed_levels,
    ).to(device)

    # 2) 鍏堝姞杞芥ā鍨嬫潈閲嶏紝鍐?compile / DDP
    resume_ckpt = None
    if args.resume:
        resume_ckpt = load_training_checkpoint(args.resume, device)
        incompatible = base_model.load_state_dict(
            resume_ckpt["model_state_dict"],
            strict=args.load_strict,
        )
        if is_main_process():
            print(f"[rank0] Loaded model weights from {args.resume}")
            if hasattr(incompatible, "missing_keys") and len(incompatible.missing_keys) > 0:
                print("[rank0] missing_keys:", incompatible.missing_keys)
            if hasattr(incompatible, "unexpected_keys") and len(incompatible.unexpected_keys) > 0:
                print("[rank0] unexpected_keys:", incompatible.unexpected_keys)

    # 3) compile
    if not args.disable_compile and hasattr(torch, "compile"):
        try:
            base_model = torch.compile(base_model)
            if is_main_process():
                print("[rank0] torch.compile enabled")
        except Exception as exc:
            if is_main_process():
                print(f"[rank0] torch.compile failed: {exc}, using uncompiled model")

    # 4) DDP
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
        eta_min=max(1e-7, args.lr * 0.1),
    )

    ema = ExponentialMovingAverage(unwrap_model(model).parameters(), decay=args.ema_decay)

    scaler_stats = joblib.load(args.err_scaler_path)
    channel_idx = args.phys_eval_channel_idx
    assert -args.num_surface_vars <= channel_idx < args.num_surface_vars, (
        f"phys_eval_channel_idx={channel_idx} out of valid range for {args.num_surface_vars} channels"
    )
    phys_mean = float(scaler_stats["mean"][channel_idx])
    phys_std = float(scaler_stats["std"][channel_idx])

    all_train_losses = [] if is_main_process() else None
    all_train_score_losses = [] if is_main_process() else None
    all_train_fcrps_losses = [] if is_main_process() else None
    all_eval_phys = [] if is_main_process() else None
    all_eval_norm = [] if is_main_process() else None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = build_grad_scaler(device_type, use_bf16)

    # 5) 鎭㈠璁粌鐘舵€?
    start_epoch = 0
    best_metric = float("inf")

    if resume_ckpt is not None and (not args.resume_weights_only):
        if "optimizer_state_dict" in resume_ckpt and resume_ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt and resume_ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if "ema_state_dict" in resume_ckpt and resume_ckpt["ema_state_dict"] is not None:
            ema.load_state_dict(resume_ckpt["ema_state_dict"])
        if "scaler_state_dict" in resume_ckpt and resume_ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

        start_epoch = int(resume_ckpt.get("epoch", 0))
        best_metric = float(resume_ckpt.get("best_metric", float("inf")))

        if is_main_process():
            print(f"[rank0] Resume full state from epoch={start_epoch}, best_metric={best_metric:.6e}")

    elif resume_ckpt is not None and args.resume_weights_only:
        if is_main_process():
            print("[rank0] Loaded model weights only. Optimizer/scheduler/ema/scaler are re-initialized.")

    if start_epoch >= args.epochs:
        raise ValueError(
            f"Checkpoint epoch={start_epoch}, but args.epochs={args.epochs}. "
            f"--epochs should be the TOTAL target epochs."
        )

    sigma_hi = effective_sigma_max

    try:
        # 6) 浠?start_epoch 寮€濮嬭缁冿紝鑰屼笉鏄粠 0
        for epoch in range(start_epoch, args.epochs):
            model.train()
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            start_time = time.time()
            total_loss = 0.0
            total_loss_score = 0.0
            total_loss_fcrps = 0.0
            num_batches = 0

            for fc, err, label, valid_time, init_time in tqdm(
                train_loader,
                desc=f"Train (epoch {epoch + 1}/{args.epochs})",
                unit="batch",
                disable=not is_main_process(),
            ):
                batch = prepare_batch(fc, err, label, valid_time, init_time, topo_base, device, args)
                y = batch["target_error"]

                optimizer.zero_grad(set_to_none=True)

                sigma_base = sample_sigma_karras(
                    B=batch["batch_size"],
                    sigma_min=args.sigma_min,
                    sigma_max=sigma_hi,
                    rho=args.edm_rho,
                    device=device,
                    dtype=torch.float32,
                )

                members = []
                score_losses = []
                num_pairs = args.k_crps // 2
                z_batch = fixed_levels.expand(batch["batch_size"], -1)

                for _ in range(num_pairs):
                    eps = torch.randn_like(y)

                    if args.sigma_jitter_log > 0:
                        log_jit = torch.empty(batch["batch_size"], 1, 1, 1, device=device).uniform_(
                            -args.sigma_jitter_log, args.sigma_jitter_log
                        )
                        sigma_pair = (sigma_base * torch.exp(log_jit)).clamp(args.sigma_min, sigma_hi)
                    else:
                        sigma_pair = sigma_base

                    for noise_k in (eps, -eps):
                        x_t = y + sigma_pair * noise_k
                        with amp.autocast(
                            device_type=device_type,
                            dtype=amp_dtype,
                            enabled=(device_type == "cuda"),
                        ):
                            score_k = run_model(
                                model=model,
                                x_t=x_t,
                                sigma=sigma_pair,
                                batch=batch,
                                z_coord_hpa=z_batch,
                            )
                            loss_score_k = torch.mean((sigma_pair * score_k + noise_k) ** 2)
                            x0_hat_k = x_t + (sigma_pair ** 2) * score_k

                        score_losses.append(loss_score_k.float())
                        members.append(x0_hat_k.float())

                if args.k_crps % 2 == 1:
                    noise_k = torch.randn_like(y)
                    if args.sigma_jitter_log > 0:
                        log_jit = torch.empty(batch["batch_size"], 1, 1, 1, device=device).uniform_(
                            -args.sigma_jitter_log, args.sigma_jitter_log
                        )
                        sigma_single = (sigma_base * torch.exp(log_jit)).clamp(args.sigma_min, sigma_hi)
                    else:
                        sigma_single = sigma_base

                    x_t = y + sigma_single * noise_k
                    with amp.autocast(
                        device_type=device_type,
                        dtype=amp_dtype,
                        enabled=(device_type == "cuda"),
                    ):
                        score_k = run_model(
                            model=model,
                            x_t=x_t,
                            sigma=sigma_single,
                            batch=batch,
                            z_coord_hpa=z_batch,
                        )
                        loss_score_k = torch.mean((sigma_single * score_k + noise_k) ** 2)
                        x0_hat_k = x_t + (sigma_single ** 2) * score_k

                    score_losses.append(loss_score_k.float())
                    members.append(x0_hat_k.float())

                ens = torch.stack(members, dim=0)
                loss_score = torch.stack(score_losses).mean()
                loss_fcrps = ensemble_fcrps(ens, y.float())
                loss = args.lambda_score * loss_score + args.lambda_fcrps * loss_fcrps

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                ema.update()

                total_loss += float(loss.item())
                total_loss_score += float(loss_score.item())
                total_loss_fcrps += float(loss_fcrps.item())
                num_batches += 1

            stats_t = torch.tensor(
                [total_loss, total_loss_score, total_loss_fcrps, num_batches],
                dtype=torch.float64,
                device=device if torch.cuda.is_available() else torch.device("cpu"),
            )
            if is_dist_initialized():
                dist.all_reduce(stats_t, op=dist.ReduceOp.SUM)

            denom = torch.clamp_min(stats_t[-1], 1.0)
            avg_train_loss = (stats_t[0] / denom).item()
            avg_train_score = (stats_t[1] / denom).item()
            avg_train_fcrps = (stats_t[2] / denom).item()
            scheduler.step()

            if is_main_process():
                elapsed = time.time() - start_time
                print(
                    f"[epoch {epoch + 1}/{args.epochs}] "
                    f"Train total={avg_train_loss:.6e} | "
                    f"score={avg_train_score:.6e} | "
                    f"fCRPS={avg_train_fcrps:.6e} | "
                    f"Time={elapsed:.2f}s | "
                    f"LR={scheduler.get_last_lr()[0]:.6g}"
                )
                all_train_losses.append(avg_train_loss)
                all_train_score_losses.append(avg_train_score)
                all_train_fcrps_losses.append(avg_train_fcrps)

            # 姣忎釜 epoch 閮戒繚瀛?latest_full锛屾渶閫傚悎涓柇缁
            if is_main_process():
                latest_full_ckpt_path = os.path.join(args.save_dir, "vesde_physcond_direct_diffusion_fcrps_latest_full.pth")
                save_training_checkpoint(
                    latest_full_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema,
                    scaler=scaler,
                    epoch=epoch + 1,
                    best_metric=best_metric,
                    args=args,
                )

            if (epoch + 1) % args.eval_every == 0:
                with ema.average_parameters():
                    eval_model = unwrap_model(model)
                    mse_phys, mse_norm = evaluate_diffusion(
                        eval_model,
                        device,
                        sde,
                        valid_loader,
                        topo_base,
                        args,
                        phys_mean=phys_mean,
                        phys_std=phys_std,
                        z_coord_hpa=fixed_levels.view(1, -1),
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
                        ckpt_path = os.path.join(args.save_dir, f"vesde_physcond_direct_diffusion_fcrps_epoch_{epoch + 1}.pth")
                        save_training_checkpoint(
                            ckpt_path,
                            model,
                            optimizer,
                            scheduler,
                            ema,
                            scaler,
                            epoch + 1,
                            best_metric,
                            args,
                        )

                    if mse_norm < best_metric:
                        best_metric = mse_norm
                        best_path = os.path.join(args.save_dir, "vesde_physcond_direct_diffusion_fcrps_best.pth")
                        save_training_checkpoint(
                            best_path,
                            model,
                            optimizer,
                            scheduler,
                            ema,
                            scaler,
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
                os.path.join(args.save_dir, "all_train_losses_direct_diffusion_fcrps.npy"),
                np.array(all_train_losses if all_train_losses is not None else []),
            )
            np.save(
                os.path.join(args.save_dir, "all_train_score_losses_direct_diffusion_fcrps.npy"),
                np.array(all_train_score_losses if all_train_score_losses is not None else []),
            )
            np.save(
                os.path.join(args.save_dir, "all_train_fcrps_losses_direct_diffusion_fcrps.npy"),
                np.array(all_train_fcrps_losses if all_train_fcrps_losses is not None else []),
            )
            np.save(
                os.path.join(args.save_dir, "all_valid_mses_phys_direct_diffusion_fcrps.npy"),
                np.array(all_eval_phys if all_eval_phys is not None else []),
            )
            np.save(
                os.path.join(args.save_dir, "all_valid_mses_norm_direct_diffusion_fcrps.npy"),
                np.array(all_eval_norm if all_eval_norm is not None else []),
            )

            final_path = os.path.join(args.save_dir, "vesde_physcond_direct_diffusion_fcrps_final.pth")
            save_training_checkpoint(final_path, model, optimizer, scheduler, ema, scaler, args.epochs, best_metric, args)
            print(f"[rank0] Training finished. Models and metrics were saved to {args.save_dir}")

        cleanup_distributed()


if __name__ == "__main__":
    main()



