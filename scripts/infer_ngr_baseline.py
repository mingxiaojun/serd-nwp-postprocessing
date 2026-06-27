# -*- coding: utf-8 -*-
"""
Inference script for NGR-CovSigma.

Corresponds to train_traditional_ngr_error_covsigma.py.

Training formulation:
    error = analysis - forecast

For each variable v independently:
    error_v | x, z ~ Normal(mu_e_v, sigma_e_v^2)

Mean model:
    mu_e_v = X_mu @ beta_v

Variance model:
    log_sigma_e_v = gamma0 + gamma1*lead_norm + gamma2*|x_v_norm| + gamma3*topo_norm
                    + gamma4*|mu_e_v| + gamma5*sin(day_of_year) + gamma6*cos(day_of_year)

Inference output:
    sampled_error = mu_error + sigma_error * random_noise
    y_member = forecast_surface + sampled_error

Save format:
    output_root/init_time/init_time_lead.npy

Example:
    output_root/2024-12-24-09/2024-12-24-09_66.npy

Saved array:
    [K, 5, H, W], target-variable ensemble forecast in physical units.

Variable order:
    [q2m, u10, v10, sp, t2m]
"""

import os
import glob
import argparse
from datetime import datetime, timedelta

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from serd.data.normalizer_forecast import DataNormalizer as DataNormalizer_fc


VAR_NAMES = ["q2m", "u10", "v10", "sp", "t2m"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inference for NGR-CovSigma error-space NGR baseline"
    )

    parser.add_argument("--data_dir", type=str, default="./data")

    parser.add_argument(
        "--params_path",
        type=str,
        default="./outputs/checkpoints/serd_v1/ngr_baseline/traditional_ngr_error_covsigma_params.npz",
        help="Path to fitted NGR-CovSigma npz parameter file.",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="./outputs/predictions/serd_v1/ngr_baseline",
        help="Root directory for saving inference samples.",
    )

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

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "all", "custom"],
        help="Which date directories to run inference on.",
    )

    parser.add_argument("--custom_start_index", type=int, default=None)
    parser.add_argument("--custom_end_index", type=int, default=None)

    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--num_surface_vars", type=int, default=5)
    parser.add_argument("--num_levels", type=int, default=9)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--ensemble_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2024)

    # Must match the training script.
    parser.add_argument("--sigma_floor_norm", type=float, default=0.03)
    parser.add_argument("--log_sigma_min", type=float, default=-5.0)
    parser.add_argument("--log_sigma_max", type=float, default=1.0)

    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Dtype used to save ensemble npy files.",
    )

    parser.add_argument(
        "--save_distribution",
        action="store_true",
        help="Also save target distribution fields as *_mu.npy and *_sigma.npy.",
    )

    parser.add_argument(
        "--save_error_distribution",
        action="store_true",
        help="Also save error distribution fields as *_err_mu.npy and *_err_sigma.npy.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy outputs.",
    )

    return parser


def parse_forecast_time(date_str: str, fhour: int):
    """
    date_str: 'YYYY_MM_DD'
    fhour: lead hour, 3..72
    """
    init_dt = datetime.strptime(date_str, "%Y_%m_%d").replace(hour=9)
    valid_dt = init_dt + timedelta(hours=int(fhour))
    return valid_dt.strftime("%Y-%m-%d-%H"), init_dt.strftime("%Y-%m-%d-%H")


class ForecastOnlyDataset(Dataset):
    """
    Forecast-only inference dataset.

    Directory example:
        data_root/20241224/2024_12_24_66.npy

    Returns:
        fc_norm:    [45,200,200]
        lead_label: 0..23
        lead_hour:  3..72
        valid_time: 'YYYY-MM-DD-HH'
        init_time:  'YYYY-MM-DD-HH'
    """

    def __init__(self, file_paths, normalizer_forecast):
        self.samples = []
        self.normalizer_forecast = normalizer_forecast

        for file_path in file_paths:
            date_name = os.path.basename(file_path)
            date_str = datetime.strptime(str(date_name), "%Y%m%d").strftime("%Y_%m_%d")

            for lead_hour in np.arange(3, 73, 3):
                lead_hour = int(lead_hour)
                fc_path = os.path.join(file_path, f"{date_str}_{lead_hour}.npy")

                if not os.path.exists(fc_path):
                    continue

                valid_time, init_time = parse_forecast_time(date_str, lead_hour)
                lead_label = lead_hour // 3 - 1
                self.samples.append((fc_path, lead_label, lead_hour, valid_time, init_time))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fc_path, lead_label, lead_hour, valid_time, init_time = self.samples[idx]

        fc = np.load(fc_path).reshape(1, 45, 200, 200)
        fc = self.normalizer_forecast.transform(fc)
        fc = fc.reshape(45, 200, 200)

        return (
            torch.tensor(fc, dtype=torch.float32),
            torch.tensor(lead_label, dtype=torch.long),
            torch.tensor(lead_hour, dtype=torch.long),
            valid_time,
            init_time,
        )


def encode_time_ymdh(time_str):
    dt = datetime.strptime(time_str, "%Y-%m-%d-%H")
    hour = dt.hour
    doy = dt.timetuple().tm_yday

    theta_d = 2.0 * np.pi * hour / 24.0
    theta_y = 2.0 * np.pi * (doy - 1) / 365.2425

    return np.array(
        [np.sin(theta_d), np.cos(theta_d), np.sin(theta_y), np.cos(theta_y)],
        dtype=np.float32,
    )


def build_static_features(args):
    """
    Static feature channels [C_static,H,W]:
        normalized topo + x/y + sin/cos lon + sin/cos lat
    """
    topo = np.load(args.topo_path).astype(np.float32)
    if topo.ndim != 4:
        raise ValueError(f"Expected topo [1,C,H,W], got {topo.shape}")

    topo = topo[0, :, :args.height, :args.width]

    H, W = args.height, args.width
    yy = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :]

    xmap = np.broadcast_to(xx, (H, W))[None]
    ymap = np.broadcast_to(yy, (H, W))[None]

    lon_min, lon_max = 114.01, 119.74
    lat_min, lat_max = 29.019999, 34.749999

    lons = np.linspace(lon_min, lon_max, W, dtype=np.float32)[None, :]
    lats = np.linspace(lat_min, lat_max, H, dtype=np.float32)[:, None]

    lon2d = np.broadcast_to(lons, (H, W))
    lat2d = np.broadcast_to(lats, (H, W))

    lonr = np.deg2rad(lon2d).astype(np.float32)
    latr = np.deg2rad(lat2d).astype(np.float32)

    geo = np.stack(
        [
            xmap[0],
            ymap[0],
            np.sin(lonr),
            np.cos(lonr),
            np.sin(latr),
            np.cos(latr),
        ],
        axis=0,
    ).astype(np.float32)

    return np.concatenate([topo, geo], axis=0)


def load_stats(args):
    forecast_scaler_path = os.path.join(
        args.data_dir,
        "scalers_forecast_zscore_two_step_unet_train.pkl",
    )
    error_scaler_path = os.path.join(
        args.data_dir,
        "scalers_ana_zscore_two_step_unet_train.pkl",
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
    fc_norm = fc_norm[:, :, :args.height, :args.width].astype(np.float32)
    return fc_norm[:, stats["surface_idx"], :, :]


def build_mean_feature_matrix_for_points(
    surface_fc_norm,
    static_features,
    lead,
    valid_time,
    init_time,
    points_y,
    points_x,
    variable_index,
    args,
):
    rows = []
    B = surface_fc_norm.shape[0]

    for b in range(B):
        yy = points_y[b]
        xx = points_x[b]

        same_var_fc = surface_fc_norm[b, variable_index, yy, xx][:, None]
        static = static_features[:, yy, xx].T

        lead_norm = np.full(
            (len(yy), 1),
            float(lead[b]) / 23.0,
            dtype=np.float32,
        )

        valid_feat = encode_time_ymdh(valid_time[b])[None, :]
        init_feat = encode_time_ymdh(init_time[b])[None, :]

        valid_map = np.repeat(valid_feat, len(yy), axis=0)
        init_map = np.repeat(init_feat, len(yy), axis=0)

        intercept = np.ones((len(yy), 1), dtype=np.float32)

        Xb = np.concatenate(
            [
                intercept,
                same_var_fc,
                static,
                lead_norm,
                valid_map,
                init_map,
            ],
            axis=1,
        ).astype(np.float32)

        rows.append(Xb)

    return np.concatenate(rows, axis=0)


def build_sigma_base_features_for_points(
    surface_fc_norm,
    static_features,
    lead,
    valid_time,
    points_y,
    points_x,
    variable_index,
):
    """
    Base sigma features without |mu_error|.

    Base columns:
        [intercept, lead_norm, abs_fc_norm, topo_norm, sin_season, cos_season]

    During prediction, |mu_error| is inserted:
        [intercept, lead_norm, abs_fc_norm, topo_norm, abs_mu, sin_season, cos_season]
    """
    rows = []
    B = surface_fc_norm.shape[0]

    for b in range(B):
        yy = points_y[b]
        xx = points_x[b]
        n = len(yy)

        intercept = np.ones((n, 1), dtype=np.float32)
        lead_norm = np.full((n, 1), float(lead[b]) / 23.0, dtype=np.float32)

        abs_fc = np.abs(surface_fc_norm[b, variable_index, yy, xx])[:, None].astype(np.float32)
        topo = static_features[0, yy, xx][:, None].astype(np.float32)

        valid_feat = encode_time_ymdh(valid_time[b])
        sin_season = np.full((n, 1), valid_feat[2], dtype=np.float32)
        cos_season = np.full((n, 1), valid_feat[3], dtype=np.float32)

        Xb = np.concatenate(
            [intercept, lead_norm, abs_fc, topo, sin_season, cos_season],
            axis=1,
        ).astype(np.float32)

        rows.append(Xb)

    return np.concatenate(rows, axis=0)


def build_sigma_design_from_base(X_sigma_base, mu):
    """
    Insert |mu_error| into sigma features.

    X_sigma_base:
        [intercept, lead, abs_fc, topo, sin_season, cos_season]

    Output:
        [intercept, lead, abs_fc, topo, abs_mu, sin_season, cos_season]
    """
    abs_mu = np.abs(mu).reshape(-1, 1).astype(np.float64)
    return np.concatenate([X_sigma_base[:, 0:4], abs_mu, X_sigma_base[:, 4:6]], axis=1)


def predict_error_params(params, X_mu, X_sigma_base, args):
    """
    Predict normalized error mean and normalized error sigma.
    This script is only for NGR-CovSigma, so gamma length must be 7.
    """
    P_mu = X_mu.shape[1]

    beta = params[:P_mu]
    gamma = params[P_mu:]

    if gamma.shape[0] != 7:
        raise ValueError(
            f"NGR-CovSigma expects 7 variance parameters, got {gamma.shape[0]}. "
            "Check that params_path points to traditional_ngr_error_covsigma_params.npz."
        )

    mu = X_mu @ beta
    X_sigma = build_sigma_design_from_base(X_sigma_base, mu)

    log_sigma = np.clip(X_sigma @ gamma, args.log_sigma_min, args.log_sigma_max)
    sigma = np.exp(log_sigma) + args.sigma_floor_norm

    return mu.astype(np.float32), sigma.astype(np.float32)


def infer_one_sample(
    fc_norm,
    lead_label,
    valid_time,
    init_time,
    params_by_var,
    static_features,
    stats,
    args,
    rng,
):
    """
    Return:
        ensemble      [K,5,H,W], target forecast in physical units
        mu_y_all      [5,H,W]
        sigma_y_all   [5,H,W]
        mu_err_all    [5,H,W]
        sigma_err_all [5,H,W]
    """
    fc_norm = fc_norm[:, :, :args.height, :args.width].astype(np.float32)
    surface_fc_norm = get_surface_forecast_norm(fc_norm, stats, args)

    B, C, H, W = surface_fc_norm.shape
    if B != 1:
        raise ValueError("infer_one_sample expects B=1")

    surface_fc_phys = (
        surface_fc_norm * stats["surface_fc_std"][None, :, None, None]
        + stats["surface_fc_mean"][None, :, None, None]
    )

    yy_grid, xx_grid = np.meshgrid(
        np.arange(H, dtype=np.int64),
        np.arange(W, dtype=np.int64),
        indexing="ij",
    )
    yy_flat = yy_grid.ravel()
    xx_flat = xx_grid.ravel()

    K = args.ensemble_size
    ensemble = np.empty((K, C, H, W), dtype=np.float32)

    mu_y_all = np.empty((C, H, W), dtype=np.float32)
    sigma_y_all = np.empty((C, H, W), dtype=np.float32)
    mu_err_all = np.empty((C, H, W), dtype=np.float32)
    sigma_err_all = np.empty((C, H, W), dtype=np.float32)

    lead_arr = np.array([int(lead_label)], dtype=np.int64)

    for v in range(C):
        X_mu = build_mean_feature_matrix_for_points(
            surface_fc_norm=surface_fc_norm,
            static_features=static_features,
            lead=lead_arr,
            valid_time=[valid_time],
            init_time=[init_time],
            points_y=[yy_flat],
            points_x=[xx_flat],
            variable_index=v,
            args=args,
        )

        X_sigma_base = build_sigma_base_features_for_points(
            surface_fc_norm=surface_fc_norm,
            static_features=static_features,
            lead=lead_arr,
            valid_time=[valid_time],
            points_y=[yy_flat],
            points_x=[xx_flat],
            variable_index=v,
        )

        params = params_by_var[v]
        P_mu = X_mu.shape[1]
        expected_len = P_mu + 7
        if params.shape[0] != expected_len:
            raise ValueError(
                f"Parameter length mismatch for variable {v}: got {params.shape[0]}, "
                f"expected {expected_len}. This inference script is for NGR-CovSigma only."
            )

        mu_err_norm, sigma_err_norm = predict_error_params(params, X_mu, X_sigma_base, args)

        mu_err_norm_2d = mu_err_norm.reshape(H, W)
        sigma_err_norm_2d = sigma_err_norm.reshape(H, W)

        mu_err_phys = mu_err_norm_2d * stats["err_std"][v] + stats["err_mean"][v]
        sigma_err_phys = sigma_err_norm_2d * stats["err_std"][v]

        mu_y_phys = surface_fc_phys[0, v] + mu_err_phys
        sigma_y_phys = sigma_err_phys

        mu_y_all[v] = mu_y_phys.astype(np.float32)
        sigma_y_all[v] = sigma_y_phys.astype(np.float32)
        mu_err_all[v] = mu_err_phys.astype(np.float32)
        sigma_err_all[v] = sigma_err_phys.astype(np.float32)

        noise = rng.standard_normal(size=(K, H, W)).astype(np.float32)
        ensemble[:, v, :, :] = mu_y_phys[None, :, :] + sigma_y_phys[None, :, :] * noise

    return ensemble, mu_y_all, sigma_y_all, mu_err_all, sigma_err_all


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    all_filepaths = sorted(glob.glob(args.data_root_glob))

    if args.split == "test":
        start = args.train_count + args.valid_count
        file_paths = all_filepaths[start:]
    elif args.split == "all":
        file_paths = all_filepaths
    else:
        if args.custom_start_index is None:
            raise ValueError("--custom_start_index is required when --split custom")
        file_paths = all_filepaths[args.custom_start_index : args.custom_end_index]

    if len(file_paths) == 0:
        raise RuntimeError("No input date directories selected for inference.")

    stats = load_stats(args)
    normalizer_forecast = DataNormalizer_fc.load(stats["forecast_scaler_path"])

    dataset = ForecastOnlyDataset(file_paths, normalizer_forecast)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    if not os.path.exists(args.params_path):
        raise FileNotFoundError(args.params_path)

    params_npz = np.load(args.params_path, allow_pickle=True)
    params_by_var = params_npz["params"].astype(np.float64)

    if params_by_var.shape[0] != args.num_surface_vars:
        raise ValueError(
            f"Expected {args.num_surface_vars} variable parameter sets, got {params_by_var.shape[0]}"
        )

    static_features = build_static_features(args)

    print("======================================")
    print("NGR-CovSigma inference")
    print(f"Input date dirs : {len(file_paths)}")
    print(f"Forecast samples: {len(dataset)}")
    print(f"Params path     : {args.params_path}")
    print(f"Output root     : {args.output_root}")
    print(f"Ensemble size   : {args.ensemble_size}")
    print(f"Save dtype      : {args.save_dtype}")
    print("Saving format   : output_root/init_time/init_time_lead.npy")
    print("Saved shape     : [K, 5, H, W]")
    print("Saved field     : target-variable ensemble forecast in physical units")
    print("Variance model  : covariate-dependent sigma, 7 gamma parameters")
    print("======================================")

    rng = np.random.default_rng(args.seed)
    saved = 0
    skipped = 0

    for fc, lead_label, lead_hour, valid_time, init_time in loader:
        B = fc.shape[0]

        for b in range(B):
            fc_np = fc[b : b + 1].numpy()

            lead_label_b = int(lead_label[b].item())
            lead_hour_b = int(lead_hour[b].item())

            valid_time_b = valid_time[b]
            init_time_b = init_time[b]

            out_dir = os.path.join(args.output_root, init_time_b)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, f"{init_time_b}_{lead_hour_b}.npy")

            if os.path.exists(out_path) and not args.overwrite:
                skipped += 1
                continue

            ensemble, mu_y, sigma_y, mu_err, sigma_err = infer_one_sample(
                fc_norm=fc_np,
                lead_label=lead_label_b,
                valid_time=valid_time_b,
                init_time=init_time_b,
                params_by_var=params_by_var,
                static_features=static_features,
                stats=stats,
                args=args,
                rng=rng,
            )

            if args.save_dtype == "float16":
                ensemble_to_save = ensemble.astype(np.float16)
            else:
                ensemble_to_save = ensemble.astype(np.float32)

            np.save(out_path, ensemble_to_save)

            if args.save_distribution:
                np.save(out_path.replace(".npy", "_mu.npy"), mu_y.astype(np.float32))
                np.save(out_path.replace(".npy", "_sigma.npy"), sigma_y.astype(np.float32))

            if args.save_error_distribution:
                np.save(out_path.replace(".npy", "_err_mu.npy"), mu_err.astype(np.float32))
                np.save(out_path.replace(".npy", "_err_sigma.npy"), sigma_err.astype(np.float32))

            saved += 1

            if saved % 100 == 0:
                print(f"Saved {saved} files, skipped {skipped}. Last: {out_path}")

    print("======================================")
    print(f"Inference finished. Saved={saved}, skipped={skipped}")
    print(f"Output root: {args.output_root}")
    print("======================================")


if __name__ == "__main__":
    main()



