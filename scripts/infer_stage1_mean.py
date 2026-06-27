import os

# 蹇呴』鍦?import torch 涔嬪墠璁剧疆
os.environ.setdefault("TRITON_CACHE_DIR", "./outputs/cache/triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "./outputs/cache/torch_inductor")
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)

import os
import glob
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from datetime import datetime

from serd.data.forecast_analysis_dataset import ForecastDataset
from serd.models import forecast_mean_unet
from serd.data.normalizer_forecast import DataNormalizer as DataNormalizer_fc
from serd.data.normalizer_analysis import DataNormalizer as DataNormalizer_err


# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser(description="DDP Inference for Regression Mean Model (3D+2D UNet)")

parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--num_workers", type=int, default=7)

parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--data_root_glob", type=str, default="/path/to/CMA_gfs_time_order_3_72/*[0-9]")
parser.add_argument("--topo_path", type=str, default="./data/topo_data_Normalization.npy")

# 妯″瀷鏉冮噸
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="./outputs/checkpoints/serd_v1/reg_sysbias_best_serd_v1_stage1.pth",
    help="璁粌濂界殑妯″瀷鏉冮噸璺緞锛屼緥濡?reg_mean_final_unet.pth"
)

# 杈撳嚭鐩綍
parser.add_argument("--output_root", type=str,
                    default="./outputs/predictions/serd_v1/stage1_mean")

parser.add_argument("--train_count", type=int, default=1292)
parser.add_argument("--valid_count", type=int, default=92)
parser.add_argument(
    "--split",
    type=str,
    default="test",
    choices=["train", "valid", "test", "all"],
    help="Dataset split to infer using the unified SERD split.",
)

# 鍏跺畠
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dist_backend", type=str, default="nccl")

args = parser.parse_args()


# ---------------------------
# Utils
# ---------------------------
def setup_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_distributed(dist_backend="nccl"):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=dist_backend,
            init_method='env://',
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


# ---------------------------
# 缁忕含搴︽浣欏鸡缂栫爜
# ---------------------------
def build_geo_channels(H, W, lon_min, lon_max, lat_min, lat_max, device, dtype=torch.float32,
                       add_extras=False):
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
        indexing="ij"
    )
    rel = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1,2,H,W]

    lons = torch.linspace(lon_min, lon_max, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(1, 1, H, W)
    lats = torch.linspace(lat_min, lat_max, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(1, 1, H, W)

    lonr = torch.deg2rad(lons)
    latr = torch.deg2rad(lats)
    geo = torch.cat([torch.sin(lonr), torch.cos(lonr),
                     torch.sin(latr), torch.cos(latr)], dim=1)  # [1,4,H,W]

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
        feats.append([
            np.sin(theta_d), np.cos(theta_d),
            np.sin(theta_y), np.cos(theta_y)
        ])
    return torch.tensor(feats, dtype=torch.float32, device=device)  # (B,4)


def sanitize_time_str(s: str) -> str:
    # 鍘熷鏍煎紡涓€鑸氨鏄?YYYY-MM-DD-HH锛岃繖閲屾浛鎹㈡垚鏇撮€傚悎浣滀负璺緞鍚嶇殑鏍煎紡
    return s.replace(":", "").replace(" ", "_")


def save_prediction_batch(pred, init_time_list, label_tensor, err_tensor, output_root):
    """
    pred: (B, C, H, W), torch.Tensor on cpu or gpu
    init_time_list: list[str]
    label_tensor: (B,), torch.Tensor or ndarray
    err_tensor: (B, C, H, W), torch.Tensor or ndarray
    """
    pred_np = pred.detach().float().cpu().numpy()

    if isinstance(label_tensor, torch.Tensor):
        label_np = label_tensor.detach().cpu().numpy()
    else:
        label_np = np.asarray(label_tensor)

    if isinstance(err_tensor, torch.Tensor):
        err_np = err_tensor.detach().float().cpu().numpy()
    else:
        err_np = np.asarray(err_tensor)

    # 瀵归綈 shape锛岄伩鍏嶄笌 pred 鐨勭┖闂磋寖鍥翠笉涓€鑷?
    if err_np.shape != pred_np.shape:
        err_np = err_np[:, :pred_np.shape[1], :pred_np.shape[2], :pred_np.shape[3]]

    # diff_np = err_np - pred_np

    B = pred_np.shape[0]
    for i in range(B):
        init_str = sanitize_time_str(init_time_list[i])
        lead = (int(label_np[i]) + 1) * 3

        save_dir = os.path.join(output_root, init_str)
        os.makedirs(save_dir, exist_ok=True)

        # 鍘熸帹鐞嗙粨鏋?
        save_name = f"{init_str}_{lead:02d}.npy"
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, pred_np[i])

        # # 鏂板锛歟rr 涓庢ā鍨嬫帹鐞嗙粨鏋滅殑宸?
        # err_save_name = f"{init_str}_{lead:02d}_err.npy"
        # err_save_path = os.path.join(save_dir, err_save_name)
        # np.save(err_save_path, diff_np[i])


# ---------------------------
# 鎺ㄧ悊
# ---------------------------
@torch.no_grad()
def run_inference(model, loader, topo_base, device, output_root):
    model.eval()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for fc, err, label, Valid_time, init1_time in tqdm(
        loader,
        desc="Inference",
        unit="batch",
        disable=not is_main_process()
    ):
        B = fc.shape[0]

        # 2D 棰勬姤鍙橀噺
        forecast_2d = fc[:, ::9, :192, :192].to(device, non_blocking=True)   # (B,5,H,W)

        # 3D 棰勬姤鍙橀噺
        forecast_raw = fc[:, :, :192, :192].to(device, non_blocking=True)    # (B,45,H,W) or consistent with your dataset
        forecast_3d = forecast_raw.reshape(B, 5, 9, 192, 192)

        topo_data = topo_base.expand(B, -1, -1, -1).to(dtype=forecast_2d.dtype)

        label = label.long().to(device, non_blocking=True)
        valid_time_feat = encode_time_list_ymdh_to_tensor(Valid_time, device)
        init_time_feat = encode_time_list_ymdh_to_tensor(init1_time, device)

        cond_2d = torch.cat([topo_data, forecast_2d], dim=1)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            pred = model(
                cond_2d=cond_2d,
                forecast_3d=forecast_3d,
                y=label,
                obs_time=valid_time_feat,
                init_time=init_time_feat,
            )  # (B,5,H,W)

        save_prediction_batch(
            pred=pred,
            init_time_list=init1_time,
            label_tensor=label,
            err_tensor=err,
            output_root=output_root
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    rank, world_size, local_rank, device = setup_distributed(args.dist_backend)
    setup_seed(args.seed + rank)

    if is_main_process():
        print(f"Inference on device: {device}, world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print(f"Output root: {args.output_root}")

    # ---------------------------
    # Data
    # ---------------------------
    all_filepaths = sorted(glob.glob(args.data_root_glob))
    normalizer_forecast = DataNormalizer_fc.load(
        os.path.join(args.data_dir, "scalers_forecast_zscore_two_step_unet_train.pkl")
    )
    normalizer_err = DataNormalizer_err.load(
        os.path.join(args.data_dir, "scalers_ana_zscore_two_step_unet_train.pkl")
    )

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

    if is_main_process():
        print("Total files:", len(all_filepaths))
        print("Inference files:", len(infer_files))

    infer_dataset = ForecastDataset(infer_files, normalizer_forecast, normalizer_err)

    if dist.is_available() and dist.is_initialized():
        infer_sampler = DistributedSampler(
            infer_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
    else:
        infer_sampler = None

    infer_loader = DataLoader(
        infer_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=infer_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    # ---------------------------
    # topo + coord
    # ---------------------------
    topo_np = np.load(args.topo_path)
    topo_base = torch.from_numpy(topo_np).to(device=device, dtype=torch.float32)[:, :, :192, :192]

    H, W = 192, 192
    coord_feats = build_geo_channels(
        H, W,
        lon_min=114.01, lon_max=119.74,
        lat_min=29.019999, lat_max=34.749999,
        device=device, dtype=topo_base.dtype,
        add_extras=False
    )
    topo_base = torch.cat([topo_base, coord_feats], dim=1)

    # ---------------------------
    # Model
    # ---------------------------
    num_classes = 24
    out_channels = 5
    in_channels_3d = 5
    in_channels_2d = topo_base.shape[1] + out_channels

    fixed_levels = torch.tensor(
        [1013.25, 925.0, 850.0, 700.0, 500.0, 300.0, 200.0, 150.0, 100.0],
        dtype=torch.float32
    )

    base_model = forecast_mean_unet.ForecastMeanUNet3D2D(
        in_channels_2d=in_channels_2d,
        in_channels_3d=in_channels_3d,
        out_channels=out_channels,
        num_classes=num_classes,
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

    # ckpt = torch.load(args.ckpt_path, map_location=device)
    # base_model.load_state_dict(ckpt, strict=True)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]
    base_model.load_state_dict(state_dict, strict=True)

    if is_main_process():
        print(f"Loaded checkpoint: {args.ckpt_path}")

    if dist.is_available() and dist.is_initialized():
        model = DDP(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=False
        )
        infer_model = model.module
    else:
        model = base_model
        infer_model = model

    # ---------------------------
    # Inference
    # ---------------------------
    run_inference(
        model=infer_model,
        loader=infer_loader,
        topo_base=topo_base,
        device=device,
        output_root=args.output_root
    )

    if is_main_process():
        print(f"Prediction finished. Results saved to: {args.output_root}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


