# -*- coding: utf-8 -*-
"""
All-surface-variable traditional baseline: grid-lead climatological bias correction
with residual uncertainty estimated from training residual standard deviation.

This script generalizes the SP-only GridLeadBias baseline to all surface variables.

Problem formulation:
    error_v = analysis_v - forecast_v

For each surface variable v, grid point (i,j), and lead:
    ebar_v(i,j,lead) = mean_train[error_v(i,j,lead)]
    sigma_v(i,j,lead) = std_train[error_v(i,j,lead) - ebar_v(i,j,lead)]

Prediction:
    y_hat_v = forecast_v + ebar_v(i,j,lead)
    Y_v | forecast, lead, i, j ~ Normal(y_hat_v, sigma_v(i,j,lead)^2)

This is a deterministic-forecast statistical baseline.
It does not require ensemble spread or model orography.

Variable order:
    [q2m, u10, v10, sp, t2m]

Outputs:
    - allvars_gridlead_bias_params.npz
    - allvars_gridlead_bias_valid_metrics.npz, if --use_validation
    - allvars_gridlead_bias_test_metrics.npz
"""

import os
import glob
import argparse

import joblib
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

import torch
from torch.utils.data import DataLoader

from serd.data.forecast_analysis_dataset import ForecastDataset
from serd.data.normalizer_forecast import DataNormalizer as DataNormalizer_fc
from serd.data.normalizer_analysis import DataNormalizer as DataNormalizer_err


VAR_NAMES_DEFAULT = ["q2m", "u10", "v10", "sp", "t2m"]


# =========================================================
# Args
# =========================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="All-variable grid-lead climatological bias correction with residual uncertainty"
    )

    # Paths
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_allvars_gridlead_bias")
    parser.add_argument(
        "--data_root_glob",
        type=str,
        default="/path/to/CMA_gfs_time_order_3_72/*[0-9]",
    )

    # Data split, consistent with the NGR-like baseline
    parser.add_argument("--train_count", type=int, default=1292)
    parser.add_argument("--valid_count", type=int, default=92)
    parser.add_argument("--use_validation", action="store_true")

    # Shape
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--num_surface_vars", type=int, default=5)
    parser.add_argument("--num_levels", type=int, default=9)

    # Variable setup
    parser.add_argument(
        "--var_names",
        type=str,
        default=",".join(VAR_NAMES_DEFAULT),
        help="Comma-separated surface variable names. Default: q2m,u10,v10,sp,t2m",
    )

    # Data loading
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)

    # Residual uncertainty controls
    parser.add_argument(
        "--sigma_floor_phys",
        type=float,
        default=1e-6,
        help="Physical-space lower bound for sigma_v.",
    )
    parser.add_argument(
        "--min_count_for_gridlead_sigma",
        type=int,
        default=2,
        help="If count(v,i,j,lead) is below this value, use lead-level fallback sigma for that variable.",
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser


def parse_var_names(args):
    names = [s.strip() for s in args.var_names.split(",") if s.strip()]
    if len(names) != args.num_surface_vars:
        raise ValueError(
            f"Expected {args.num_surface_vars} variable names, got {len(names)}: {names}"
        )
    return names


# =========================================================
# Normalization stats
# =========================================================
def load_stats(args):
    forecast_scaler_path = os.path.join(
        args.data_dir,
        "scalers_forecast_zscore_two_step_unet_train.pkl",
    )
    error_scaler_path = os.path.join(
        args.data_dir,
        "scalers_err_zscore_two_step_unet_train.pkl",
    )

    if not os.path.exists(forecast_scaler_path):
        raise FileNotFoundError(forecast_scaler_path)
    if not os.path.exists(error_scaler_path):
        raise FileNotFoundError(error_scaler_path)

    fc_stats = joblib.load(forecast_scaler_path)
    err_stats = joblib.load(error_scaler_path)

    fc_mean = np.asarray(fc_stats["mean"], dtype=np.float32)
    fc_std = np.asarray(fc_stats["std"], dtype=np.float32)

    err_mean = np.asarray(err_stats["mean"], dtype=np.float32)
    err_std = np.asarray(err_stats["std"], dtype=np.float32)

    surface_idx = np.arange(
        0,
        args.num_surface_vars * args.num_levels,
        args.num_levels,
        dtype=np.int64,
    )

    return {
        "forecast_scaler_path": forecast_scaler_path,
        "error_scaler_path": error_scaler_path,
        "surface_idx": surface_idx,
        "surface_fc_mean": fc_mean[surface_idx],
        "surface_fc_std": fc_std[surface_idx],
        "err_mean": err_mean,
        "err_std": err_std,
    }


def get_surface_forecast_norm(fc_norm, stats, args):
    """Extract normalized surface forecast channels [B,5,H,W] from [B,45,H,W]."""
    fc_norm = fc_norm[:, :, :args.height, :args.width].astype(np.float32)
    return fc_norm[:, stats["surface_idx"], :, :]


def lead_to_key(lead_value):
    """Use an integer lead key, robust for lead tensors stored as float or int."""
    return int(round(float(lead_value)))


# =========================================================
# Probabilistic metrics
# =========================================================
def crps_gaussian(mu, sigma, y):
    """Analytic CRPS for Gaussian predictive distribution."""
    sigma = np.maximum(sigma, 1e-8)
    z = (y - mu) / sigma

    return sigma * (
        z * (2.0 * ndtr(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )


def nll_gaussian(mu, sigma, y):
    """Gaussian negative log-likelihood."""
    sigma = np.maximum(sigma, 1e-8)
    z = (y - mu) / sigma
    return np.log(sigma) + 0.5 * z ** 2 + 0.5 * np.log(2.0 * np.pi)


# =========================================================
# Fit grid-lead bias and residual sigma for all surface variables
# =========================================================
def fit_allvars_gridlead_bias(train_dataset, stats, args, var_names):
    """
    Estimate for all variables:
        bias[v, lead, i, j] = mean_train(error_v)
        sigma[v, lead, i, j] = std_train(error_v - bias_v)

    Since std(error - mean) can be computed by E[e^2] - E[e]^2,
    we only need one training pass.
    """
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    C = args.num_surface_vars
    H, W = args.height, args.width

    # Dict[lead_key] -> arrays [C,H,W]
    sum_err = {}
    sumsq_err = {}
    count = {}

    # Lead-level scalar fallback statistics per variable.
    # Dict[lead_key] -> arrays [C]
    lead_total_sum = {}
    lead_total_sumsq = {}
    lead_total_count = {}

    global_sum = np.zeros(C, dtype=np.float64)
    global_sumsq = np.zeros(C, dtype=np.float64)
    global_count = np.zeros(C, dtype=np.int64)

    for batch_idx, (_, err, lead, _, _) in enumerate(loader):
        err_np = err.numpy()[:, :, :H, :W].astype(np.float32)
        lead_np = lead.numpy()

        # Convert normalized surface error to physical-space error.
        # err_np is expected to contain surface errors in variable order [q2m,u10,v10,sp,t2m].
        err_phys = (
            err_np[:, :C]
            * stats["err_std"][None, :C, None, None]
            + stats["err_mean"][None, :C, None, None]
        ).astype(np.float32)

        B = err_phys.shape[0]
        for b in range(B):
            key = lead_to_key(lead_np[b])

            if key not in sum_err:
                sum_err[key] = np.zeros((C, H, W), dtype=np.float64)
                sumsq_err[key] = np.zeros((C, H, W), dtype=np.float64)
                count[key] = np.zeros((C, H, W), dtype=np.int64)
                lead_total_sum[key] = np.zeros(C, dtype=np.float64)
                lead_total_sumsq[key] = np.zeros(C, dtype=np.float64)
                lead_total_count[key] = np.zeros(C, dtype=np.int64)

            e = err_phys[b].astype(np.float64)  # [C,H,W]
            sum_err[key] += e
            sumsq_err[key] += e ** 2
            count[key] += 1

            lead_total_sum[key] += np.sum(e, axis=(1, 2))
            lead_total_sumsq[key] += np.sum(e ** 2, axis=(1, 2))
            lead_total_count[key] += H * W

            global_sum += np.sum(e, axis=(1, 2))
            global_sumsq += np.sum(e ** 2, axis=(1, 2))
            global_count += H * W

        if (batch_idx + 1) % 200 == 0:
            print(f"  Processed train batches: {batch_idx + 1}")

    lead_values = np.array(sorted(sum_err.keys()), dtype=np.int64)
    L = len(lead_values)

    # Saved layout: [C,L,H,W]
    bias = np.zeros((C, L, H, W), dtype=np.float32)
    sigma = np.zeros((C, L, H, W), dtype=np.float32)
    counts = np.zeros((C, L, H, W), dtype=np.int32)
    lead_sigma_fallback = np.zeros((C, L), dtype=np.float32)
    lead_bias_fallback = np.zeros((C, L), dtype=np.float32)

    global_mean = global_sum / np.maximum(global_count, 1)
    global_var = global_sumsq / np.maximum(global_count, 1) - global_mean ** 2
    global_sigma = np.sqrt(np.maximum(global_var, args.sigma_floor_phys ** 2))

    for li, key in enumerate(lead_values):
        c = count[int(key)].astype(np.float64)      # [C,H,W]
        s = sum_err[int(key)]                      # [C,H,W]
        ss = sumsq_err[int(key)]                   # [C,H,W]

        lead_mean = lead_total_sum[int(key)] / np.maximum(lead_total_count[int(key)], 1)
        lead_var = lead_total_sumsq[int(key)] / np.maximum(lead_total_count[int(key)], 1) - lead_mean ** 2
        lead_std = np.sqrt(np.maximum(lead_var, args.sigma_floor_phys ** 2))

        lead_bias_fallback[:, li] = lead_mean.astype(np.float32)
        lead_sigma_fallback[:, li] = lead_std.astype(np.float32)

        for v in range(C):
            mean_v = np.full((H, W), lead_mean[v], dtype=np.float64)
            valid_mean = c[v] > 0
            mean_v[valid_mean] = s[v][valid_mean] / c[v][valid_mean]

            var_v = np.full((H, W), lead_std[v] ** 2, dtype=np.float64)
            valid_var = c[v] >= args.min_count_for_gridlead_sigma
            var_v[valid_var] = ss[v][valid_var] / c[v][valid_var] - mean_v[valid_var] ** 2
            var_v = np.maximum(var_v, args.sigma_floor_phys ** 2)

            bias[v, li] = mean_v.astype(np.float32)
            sigma[v, li] = np.sqrt(var_v).astype(np.float32)
            counts[v, li] = count[int(key)][v].astype(np.int32)

        print(f"  lead={int(key):>4d} summary:")
        for v, name in enumerate(var_names):
            print(
                f"    {name:>4s}: "
                f"mean_bias={float(np.mean(bias[v, li])):.4f}, "
                f"mean_sigma={float(np.mean(sigma[v, li])):.4f}, "
                f"count_range=[{int(counts[v, li].min())},{int(counts[v, li].max())}]"
            )

    print("\n[train] All-variable grid-lead bias summary")
    print(f"  unique leads: {lead_values.tolist()}")
    for v, name in enumerate(var_names):
        print(
            f"  {name:>4s}: "
            f"global mean error={global_mean[v]:.4f}, "
            f"global sigma error={global_sigma[v]:.4f}, "
            f"mean |bias(i,j,lead)|={float(np.mean(np.abs(bias[v]))):.4f}, "
            f"mean sigma(i,j,lead)={float(np.mean(sigma[v])):.4f}"
        )

    return {
        "lead_values": lead_values,
        "bias": bias,
        "sigma": sigma,
        "counts": counts,
        "lead_bias_fallback": lead_bias_fallback,
        "lead_sigma_fallback": lead_sigma_fallback,
        "global_bias_fallback": global_mean.astype(np.float32),
        "global_sigma_fallback": global_sigma.astype(np.float32),
    }


# =========================================================
# Evaluation
# =========================================================
def get_lead_slice(params, lead_value):
    """Return bias/sigma slice for lead. If lead is unseen, use nearest lead."""
    lead_values = params["lead_values"]
    key = lead_to_key(lead_value)

    matches = np.where(lead_values == key)[0]
    if len(matches) > 0:
        return int(matches[0]), False

    nearest = int(np.argmin(np.abs(lead_values.astype(np.float64) - float(key))))
    return nearest, True


def evaluate_allvars_gridlead_bias(params, dataset, stats, args, var_names, split_name="test"):
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    C = args.num_surface_vars
    H, W = args.height, args.width

    se_sum = np.zeros(C, dtype=np.float64)
    raw_se_sum = np.zeros(C, dtype=np.float64)
    crps_sum = np.zeros(C, dtype=np.float64)
    nll_sum = np.zeros(C, dtype=np.float64)
    spread_sum = np.zeros(C, dtype=np.float64)
    count = 0

    cover90_sum = np.zeros(C, dtype=np.int64)
    cover95_sum = np.zeros(C, dtype=np.int64)
    width90_sum = np.zeros(C, dtype=np.float64)
    width95_sum = np.zeros(C, dtype=np.float64)
    unseen_lead_count = 0

    z90 = 1.6448536269514722
    z95 = 1.959963984540054

    for fc, err, lead, _, _ in loader:
        fc_np = fc.numpy()
        err_np = err.numpy()[:, :, :H, :W].astype(np.float32)
        lead_np = lead.numpy()

        surface_fc_norm = get_surface_forecast_norm(fc_np, stats, args)
        surface_fc_phys = (
            surface_fc_norm
            * stats["surface_fc_std"][None, :C, None, None]
            + stats["surface_fc_mean"][None, :C, None, None]
        )

        err_phys = (
            err_np[:, :C]
            * stats["err_std"][None, :C, None, None]
            + stats["err_mean"][None, :C, None, None]
        )

        obs = surface_fc_phys + err_phys
        raw = surface_fc_phys

        B = raw.shape[0]
        for b in range(B):
            li, unseen = get_lead_slice(params, lead_np[b])
            unseen_lead_count += int(unseen)

            bias = params["bias"][:, li]                       # [C,H,W]
            sigma = np.maximum(params["sigma"][:, li], args.sigma_floor_phys)

            mu = raw[b] + bias                                  # [C,H,W]
            y = obs[b]

            err_corr = mu - y
            err_raw = raw[b] - y
            abs_err = np.abs(y - mu)

            se_sum += np.sum(err_corr ** 2, axis=(1, 2))
            raw_se_sum += np.sum(err_raw ** 2, axis=(1, 2))
            crps_sum += np.sum(crps_gaussian(mu, sigma, y), axis=(1, 2))
            nll_sum += np.sum(nll_gaussian(mu, sigma, y), axis=(1, 2))
            spread_sum += np.sum(sigma, axis=(1, 2))

            cover90_sum += np.sum(abs_err <= z90 * sigma, axis=(1, 2)).astype(np.int64)
            cover95_sum += np.sum(abs_err <= z95 * sigma, axis=(1, 2)).astype(np.int64)
            width90_sum += np.sum(2.0 * z90 * sigma, axis=(1, 2))
            width95_sum += np.sum(2.0 * z95 * sigma, axis=(1, 2))

            count += H * W

    rmse = np.sqrt(se_sum / max(1, count))
    raw_rmse = np.sqrt(raw_se_sum / max(1, count))
    crps = crps_sum / max(1, count)
    nll = nll_sum / max(1, count)
    spread = spread_sum / max(1, count)
    spread_rmse = spread / np.maximum(rmse, 1e-12)
    coverage90 = cover90_sum / max(1, count)
    coverage95 = cover95_sum / max(1, count)
    width90 = width90_sum / max(1, count)
    width95 = width95_sum / max(1, count)

    print(f"\n[{split_name}] All-variable GridLeadBias metrics")
    for i, name in enumerate(var_names):
        print(
            f"  {name:>4s}: "
            f"RMSE={rmse[i]:.4f}, "
            f"CRPS={crps[i]:.4f}, "
            f"NLL={nll[i]:.4f}, "
            f"Spread/RMSE={spread_rmse[i]:.4f}, "
            f"RawRMSE={raw_rmse[i]:.4f}, "
            f"Cov90={coverage90[i]:.4f}, "
            f"Cov95={coverage95[i]:.4f}, "
            f"Width90={width90[i]:.4f}, "
            f"Width95={width95[i]:.4f}"
        )
    if unseen_lead_count > 0:
        print(f"  [warn] unseen lead values encountered: {unseen_lead_count}; nearest lead was used.")

    return {
        "rmse": rmse.astype(np.float64),
        "raw_rmse": raw_rmse.astype(np.float64),
        "crps": crps.astype(np.float64),
        "nll": nll.astype(np.float64),
        "spread": spread.astype(np.float64),
        "spread_rmse": spread_rmse.astype(np.float64),
        "coverage90": coverage90.astype(np.float64),
        "coverage95": coverage95.astype(np.float64),
        "width90": width90.astype(np.float64),
        "width95": width95.astype(np.float64),
        "var_names": np.array(var_names, dtype=object),
    }


# =========================================================
# Main
# =========================================================
def main():
    args = build_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    var_names = parse_var_names(args)

    all_filepaths = sorted(glob.glob(args.data_root_glob))
    if len(all_filepaths) <= args.train_count + args.valid_count:
        raise RuntimeError(
            f"Not enough files: {len(all_filepaths)} for train_count={args.train_count}, "
            f"valid_count={args.valid_count}"
        )

    train_files = all_filepaths[: args.train_count]
    valid_files = all_filepaths[
        args.train_count : args.train_count + args.valid_count
    ]
    test_files = all_filepaths[args.train_count + args.valid_count :]

    stats = load_stats(args)

    normalizer_forecast = DataNormalizer_fc.load(stats["forecast_scaler_path"])
    normalizer_err = DataNormalizer_err.load(stats["error_scaler_path"])

    train_dataset = ForecastDataset(train_files, normalizer_forecast, normalizer_err)
    valid_dataset = ForecastDataset(valid_files, normalizer_forecast, normalizer_err)
    test_dataset = ForecastDataset(test_files, normalizer_forecast, normalizer_err)

    print("======================================")
    print("All-variable GridLeadBias traditional baseline")
    print(f"Train days: {len(train_files)}")
    print(f"Valid days: {len(valid_files)}")
    print(f"Test days : {len(test_files)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples : {len(test_dataset)}")
    print(f"Grid: {args.height} x {args.width}")
    print(f"Variables: {var_names}")
    print("Prediction: y_hat_v = forecast_v + mean_train_error(v,i,j,lead)")
    print("Uncertainty: sigma_v = std_train_residual(v,i,j,lead)")
    print(f"sigma_floor_phys={args.sigma_floor_phys}")
    print("Saved params layout: bias/sigma/counts = [C,L,H,W]")
    print("======================================")

    params = fit_allvars_gridlead_bias(train_dataset, stats, args, var_names)

    np.savez(
        os.path.join(args.save_dir, "allvars_gridlead_bias_params.npz"),
        lead_values=params["lead_values"],
        bias=params["bias"],
        sigma=params["sigma"],
        counts=params["counts"],
        lead_bias_fallback=params["lead_bias_fallback"],
        lead_sigma_fallback=params["lead_sigma_fallback"],
        global_bias_fallback=params["global_bias_fallback"],
        global_sigma_fallback=params["global_sigma_fallback"],
        var_names=np.array(var_names, dtype=object),
        surface_idx=stats["surface_idx"],
        args=np.array([str(vars(args))], dtype=object),
    )

    if args.use_validation:
        valid_metrics = evaluate_allvars_gridlead_bias(
            params, valid_dataset, stats, args, var_names, split_name="valid"
        )
        np.savez(
            os.path.join(args.save_dir, "allvars_gridlead_bias_valid_metrics.npz"),
            **valid_metrics,
        )

    test_metrics = evaluate_allvars_gridlead_bias(
        params, test_dataset, stats, args, var_names, split_name="test"
    )
    np.savez(
        os.path.join(args.save_dir, "allvars_gridlead_bias_test_metrics.npz"),
        **test_metrics,
    )

    print(f"\nSaved results to: {args.save_dir}")


if __name__ == "__main__":
    main()

