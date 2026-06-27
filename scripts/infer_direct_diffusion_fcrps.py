# -*- coding: utf-8 -*-
"""
Inference script for the Table 2 direct total-error diffusion model.

Save format:
  output_root/
    init_time/
      init_time_lead.npy

Each saved .npy has shape:
  [K, C, H, W]
where K is ensemble_size and C is the number of predicted surface-error variables.
"""

import os
import glob
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch import amp
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

try:
    from torch_ema import ExponentialMovingAverage
except Exception:
    ExponentialMovingAverage = None

from serd.data.forecast_error_dataset import ForecastDataset
from serd.data.normalizer_forecast import DataNormalizer as DataNormalizer_fc
from serd.data.normalizer_error import DataNormalizer as DataNormalizer_err
from serd.models import physcond_error_diffusion




# =========================================================
# Fixed settings: must be consistent with training
# =========================================================
HEIGHT = 192
WIDTH = 192
NUM_SURFACE_VARS = 5
NUM_LEVELS = 9
NUM_CLASSES = 24
N_DISCRETE_SIGMAS = 256
EDM_RHO = 7.0
INIT_NOISE_TEMP = 1.0
EMA_DECAY = 0.999

FIXED_LEVELS_HPA = [1013.25, 925.0, 850.0, 700.0, 500.0, 300.0, 200.0, 150.0, 100.0]

FORECAST_SCALER_NAME = "scalers_forecast_zscore_two_step_unet_train.pkl"
ERR_SCALER_NAME = "scalers_err_zscore_two_step_unet_train.pkl"

# Model hyperparameters used in the training script.
MODEL_CHANNELS_2D = 128
BASE_CHANNELS_FCST = 32
NUM_RES_BLOCKS_2D = 2
NUM_RES_BLOCKS_FCST = 2
DROPOUT = 0.1
KSIZE = 3


# =========================================================
# Args: only inference-related options
# =========================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="DDP inference for Table 2 direct total-error diffusion"
    )

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=7)
    parser.add_argument("--data_dir", type=str, default="./data")
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
        "--ckpt_path",
        type=str,
        default="./outputs/checkpoints/direct_diffusion_fcrps/vesde_physcond_direct_diffusion_fcrps_best.pth",
        help="Training checkpoint path, e.g. vesde_physcond_direct_diffusion_fcrps_best.pth or latest_full.pth.",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="./outputs/predictions/direct_diffusion_fcrps",
        help="Output directory. Results are saved as output_root/init_time/init_time_lead.npy.",
    )

    parser.add_argument("--train_count", type=int, default=1292)
    parser.add_argument("--valid_count", type=int, default=92)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test", "all"],
        help="Dataset split to infer using the unified SERD split.",
    )

    parser.add_argument("--sigma_min", type=float, default=2e-2)
    parser.add_argument("--sigma_max", type=float, default=10.0)
    parser.add_argument("--edm_steps", type=int, default=60)
    parser.add_argument("--ensemble_size", type=int, default=16)
    parser.add_argument("--pc_corrector_steps", type=int, default=1)
    parser.add_argument("--pc_snr", type=float, default=0.16)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dist_backend", type=str, default="nccl")

    return parser


# =========================================================
# Basic utils
# =========================================================
def validate_args(args):
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    if args.sigma_min <= 0 or args.sigma_max < args.sigma_min:
        raise ValueError("Require 0 < sigma_min <= sigma_max")
    if args.edm_steps < 1:
        raise ValueError("--edm_steps must be >= 1")
    if args.ensemble_size < 1:
        raise ValueError("--ensemble_size must be >= 1")
    if args.pc_corrector_steps < 0:
        raise ValueError("--pc_corrector_steps must be >= 0")
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"ckpt_path not found: {args.ckpt_path}")
    if not os.path.exists(args.topo_path):
        raise FileNotFoundError(f"topo_path not found: {args.topo_path}")

    forecast_scaler_path = os.path.join(args.data_dir, FORECAST_SCALER_NAME)
    err_scaler_path = os.path.join(args.data_dir, ERR_SCALER_NAME)
    if not os.path.exists(forecast_scaler_path):
        raise FileNotFoundError(f"forecast scaler not found: {forecast_scaler_path}")
    if not os.path.exists(err_scaler_path):
        raise FileNotFoundError(f"error scaler not found: {err_scaler_path}")


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

        if dist_backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA.")

        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if local_rank >= n_gpus:
                raise RuntimeError(
                    f"local_rank={local_rank}, but only {n_gpus} CUDA devices are visible. "
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}"
                )
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


def sanitize_time_str(s: str) -> str:
    return str(s).replace(":", "").replace(" ", "_")


# =========================================================
# Checkpoint loading
# =========================================================
def extract_model_state_from_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def load_model_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = extract_model_state_from_checkpoint(ckpt)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Cannot find a valid state_dict in checkpoint: {ckpt_path}")

    state_dict = strip_state_dict_prefixes(state_dict)
    model_state = model.state_dict()

    matched = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            matched[k] = v
        else:
            skipped.append(k)

    model_state.update(matched)
    model.load_state_dict(model_state, strict=False)
    missing = [k for k in model_state.keys() if k not in matched]

    return ckpt, len(matched), len(skipped), len(missing)


def maybe_build_and_load_ema(model, ckpt):
    if ExponentialMovingAverage is None:
        if is_main_process():
            print("[rank0] torch_ema is not available; using raw model weights.")
        return None
    if not isinstance(ckpt, dict) or ckpt.get("ema_state_dict", None) is None:
        if is_main_process():
            print("[rank0] No ema_state_dict found in checkpoint; using raw model weights.")
        return None

    ema = ExponentialMovingAverage(model.parameters(), decay=EMA_DECAY)
    ema.load_state_dict(ckpt["ema_state_dict"])
    if is_main_process():
        print("[rank0] EMA state loaded. Inference will use EMA parameters.")
    return ema


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
    geo = torch.cat([torch.sin(lonr), torch.cos(lonr), torch.sin(latr), torch.cos(latr)], dim=1)

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
        dt = datetime.strptime(str(time_str), "%Y-%m-%d-%H")
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
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.N))

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


# =========================================================
# Data helpers
# =========================================================
def load_topography(args, device):
    topo_np = np.load(args.topo_path)
    if topo_np.ndim == 3:
        topo_np = topo_np[None, ...]
    if topo_np.ndim != 4:
        raise ValueError(f"Expected topo npy to be 4D [1,C,H,W] or 3D [C,H,W], got shape={topo_np.shape}")
    if topo_np.shape[0] != 1:
        raise ValueError(f"Expected topo batch dim to be 1 for broadcasting, got shape={topo_np.shape}")
    if topo_np.shape[2] < HEIGHT or topo_np.shape[3] < WIDTH:
        raise ValueError(f"Topography spatial size {topo_np.shape[2:]} is smaller than target {(HEIGHT, WIDTH)}")

    topo_base = torch.from_numpy(topo_np).to(device=device, dtype=torch.float32)
    topo_base = topo_base[:, :, :HEIGHT, :WIDTH]

    coord_feats = build_geo_channels(
        HEIGHT,
        WIDTH,
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


def prepare_batch(fc, err, label, valid_time, init_time, topo_base, device):
    del err

    if fc.ndim != 4:
        raise ValueError(f"Expected fc to be 4D [B,C,H,W], got shape={tuple(fc.shape)}")

    batch_size = fc.shape[0]
    expected_fc_channels = NUM_SURFACE_VARS * NUM_LEVELS
    if fc.shape[1] != expected_fc_channels:
        raise ValueError(f"fc channel mismatch: expected {expected_fc_channels}, got {fc.shape[1]}")
    if fc.shape[2] < HEIGHT or fc.shape[3] < WIDTH:
        raise ValueError(f"fc spatial size {tuple(fc.shape[2:])} is smaller than target {(HEIGHT, WIDTH)}")

    forecast_surface_2d = fc[:, ::NUM_LEVELS, :HEIGHT, :WIDTH].to(device, non_blocking=True)
    forecast_raw = fc[:, :, :HEIGHT, :WIDTH].to(device, non_blocking=True)
    forecast_3d = forecast_raw.reshape(batch_size, NUM_SURFACE_VARS, NUM_LEVELS, HEIGHT, WIDTH)

    topo_data = topo_base.expand(batch_size, -1, -1, -1).to(dtype=forecast_raw.dtype)
    label_tensor = label.long().to(device, non_blocking=True)
    valid_time_tensor = encode_time_list_ymdh_to_tensor(valid_time, device)
    init_time_tensor = encode_time_list_ymdh_to_tensor(init_time, device)
    cond_2d = torch.cat([topo_data, forecast_surface_2d], dim=1)

    if label_tensor.min().item() < 0 or label_tensor.max().item() >= NUM_CLASSES:
        raise ValueError(
            f"label out of range: min={label_tensor.min().item()}, "
            f"max={label_tensor.max().item()}, num_classes={NUM_CLASSES}. "
            "If labels are lead hours, map them to class IDs first."
        )
    if not torch.isfinite(forecast_3d).all():
        raise ValueError("forecast_3d contains NaN or Inf")
    if not torch.isfinite(cond_2d).all():
        raise ValueError("cond_2d contains NaN or Inf")

    return {
        "batch_size": batch_size,
        "forecast_3d": forecast_3d,
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
# Sampler
# =========================================================
@torch.no_grad()
def pc_sampler(
    model,
    shape,
    device,
    batch,
    z_coord_hpa,
    sigma_min,
    sigma_max,
    steps,
    snr,
    corrector_steps,
):
    B, C, H, W = shape
    sigmas = get_karras_sigmas(
        num_steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=EDM_RHO,
        device=device,
    ).to(dtype=torch.float32)
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=sigmas.dtype)], dim=0)

    x = torch.randn(B, C, H, W, device=device, dtype=torch.float32) * (INIT_NOISE_TEMP * sigmas[0])
    model.eval()

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_t = torch.full((B, 1, 1, 1), float(sigma.item()), device=device, dtype=x.dtype)

        if sigma.item() > 0 and corrector_steps > 0:
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
            std = torch.sqrt((sigma_next2 * (sigma2 - sigma_next2) / sigma2.clamp_min(1e-12)).clamp_min(0.0))
            x = x_mean + std * torch.randn_like(x)
        else:
            x = x_mean

    return x


@torch.no_grad()
def sample_ensemble(model, shape, device, batch, z_coord_hpa, args, sde):
    members = []
    sigma_min = max(float(args.sigma_min), float(sde.sigma_min))
    sigma_max = min(float(args.sigma_max), float(sde.sigma_max))

    for _ in range(args.ensemble_size):
        pred = pc_sampler(
            model=model,
            shape=shape,
            device=device,
            batch=batch,
            z_coord_hpa=z_coord_hpa,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            steps=args.edm_steps,
            snr=args.pc_snr,
            corrector_steps=args.pc_corrector_steps,
        )
        members.append(pred.float())

    if args.ensemble_size == 1:
        return members[0]
    return torch.stack(members, dim=0)


# =========================================================
# Saving
# =========================================================
def save_ensemble_batch(gens, init_time_list, label_tensor, output_root):
    gens_np = gens.detach().float().cpu().numpy()
    if gens_np.ndim == 4:
        gens_np = gens_np[None, ...]

    if isinstance(label_tensor, torch.Tensor):
        label_np = label_tensor.detach().cpu().numpy()
    else:
        label_np = np.asarray(label_tensor)

    _, B, _, _, _ = gens_np.shape

    for i in range(B):
        init_str = sanitize_time_str(init_time_list[i])
        lead = (int(label_np[i]) + 1) * 3

        save_dir = os.path.join(output_root, init_str)
        os.makedirs(save_dir, exist_ok=True)

        save_name = f"{init_str}_{lead:02d}.npy"
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, gens_np[:, i].astype(np.float32))


# =========================================================
# Inference loop
# =========================================================
@torch.no_grad()
def run_inference(model, loader, topo_base, device, sde, fixed_levels, args):
    model.eval()
    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for fc, err, label, valid_time, init_time in tqdm(
        loader,
        desc="Inference",
        unit="batch",
        disable=not is_main_process(),
    ):
        batch = prepare_batch(fc, err, label, valid_time, init_time, topo_base, device)
        shape = (batch["batch_size"], NUM_SURFACE_VARS, HEIGHT, WIDTH)
        z_batch = fixed_levels.expand(batch["batch_size"], -1)

        with amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=(device_type == "cuda")):
            gens = sample_ensemble(
                model=model,
                shape=shape,
                device=device,
                batch=batch,
                z_coord_hpa=z_batch,
                args=args,
                sde=sde,
            )

        save_ensemble_batch(
            gens=gens,
            init_time_list=init_time,
            label_tensor=batch["label"],
            output_root=args.output_root,
        )

    if is_dist_initialized():
        dist.barrier()


# =========================================================
# Main
# =========================================================
def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = os.path.join(args.data_dir, "triton_cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(args.data_dir, "torch_inductor")

    rank, world_size, local_rank, device = setup_distributed(args.dist_backend, preferred_device=args.device)
    setup_seed(args.seed + rank)

    if is_main_process():
        print(f"Inference on device: {device}, world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print(f"Checkpoint: {args.ckpt_path}")
        print(f"Output root: {args.output_root}")
        print(f"Ensemble size: {args.ensemble_size}")

    all_filepaths = sorted(glob.glob(args.data_root_glob))
    if len(all_filepaths) == 0:
        raise RuntimeError(f"No files matched data_root_glob: {args.data_root_glob}")

    normalizer_forecast = DataNormalizer_fc.load(os.path.join(args.data_dir, FORECAST_SCALER_NAME))
    normalizer_err = DataNormalizer_err.load(os.path.join(args.data_dir, ERR_SCALER_NAME))

    train_end = args.train_count
    valid_end = args.train_count + args.valid_count
    if args.split == "train":
        infer_files = all_filepaths[:train_end]
    elif args.split == "valid":
        infer_files = all_filepaths[train_end:valid_end]
    elif args.split == "test":
        infer_files = all_filepaths[valid_end:]
    else:
        infer_files = all_filepaths
    if len(infer_files) == 0:
        raise RuntimeError(
            f"No inference files for split={args.split}. "
            f"Total files={len(all_filepaths)}, train_count={args.train_count}, valid_count={args.valid_count}."
        )

    if is_main_process():
        print("Total files:", len(all_filepaths))
        print("Inference files:", len(infer_files))

    full_dataset = ForecastDataset(infer_files, normalizer_forecast, normalizer_err)

    # Exact split for DDP inference: no duplicated samples.
    if is_dist_initialized():
        indices = list(range(rank, len(full_dataset), world_size))
        infer_dataset = Subset(full_dataset, indices)
        if is_main_process():
            print("Using exact rank-based dataset split for DDP inference.")
    else:
        infer_dataset = full_dataset

    infer_loader = DataLoader(
        infer_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    topo_base = load_topography(args, device)
    fixed_levels = torch.tensor(FIXED_LEVELS_HPA, dtype=torch.float32, device=device)
    sde = VESDE(sigma_min=args.sigma_min, sigma_max=args.sigma_max, N=N_DISCRETE_SIGMAS)

    in_channels_static = topo_base.shape[1] + NUM_SURFACE_VARS
    model = physcond_error_diffusion.ForecastErrorUNet2D3D(
        in_channels_xt=NUM_SURFACE_VARS,
        in_channels_static=in_channels_static,
        in_channels_fcst=NUM_SURFACE_VARS,
        out_channels=NUM_SURFACE_VARS,
        num_classes=NUM_CLASSES,
        sde=sde,
        model_channels_2d=MODEL_CHANNELS_2D,
        base_channels_fcst=BASE_CHANNELS_FCST,
        num_res_blocks_2d=NUM_RES_BLOCKS_2D,
        num_res_blocks_fcst=NUM_RES_BLOCKS_FCST,
        channel_mult_2d=(1, 2, 4, 4),
        dropout=DROPOUT,
        ksize=KSIZE,
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

    ckpt, matched, skipped, missing = load_model_weights(model=model, ckpt_path=args.ckpt_path, device=device)
    if is_main_process():
        print(f"Loaded checkpoint weights: matched={matched}, skipped={skipped}, missing_current={missing}")
        if skipped > 0 or missing > 0:
            print("[rank0] Warning: non-strict checkpoint loading had skipped or missing parameters.")

    ema = maybe_build_and_load_ema(model=model, ckpt=ckpt)

    try:
        if ema is not None:
            with ema.average_parameters():
                run_inference(
                    model=model,
                    loader=infer_loader,
                    topo_base=topo_base,
                    device=device,
                    sde=sde,
                    fixed_levels=fixed_levels,
                    args=args,
                )
        else:
            run_inference(
                model=model,
                loader=infer_loader,
                topo_base=topo_base,
                device=device,
                sde=sde,
                fixed_levels=fixed_levels,
                args=args,
            )

        if is_main_process():
            print(f"Inference finished. Results saved to: {args.output_root}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()




