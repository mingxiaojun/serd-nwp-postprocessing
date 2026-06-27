# -*- coding: utf-8 -*-
"""
Traditional error-space NGR / EMOS baseline with covariates.

This version is intended as a fairer classical statistical baseline for SERD.

Problem formulation:
    error = analysis - forecast

For each variable v independently:
    error_v | x, z ~ Normal(mu_e_v, sigma_e_v^2)

Mean model:
    mu_e_v = X_mu @ beta_v

Variance model:
    log_sigma_e_v =
        gamma0
      + gamma1 * lead_norm
      + gamma2 * |x_v_norm|
      + gamma3 * topo_norm
      + gamma4 * |mu_e_v|
      + gamma5 * sin(day_of_year)
      + gamma6 * cos(day_of_year)

where:
    |x_v_norm| is the magnitude of the standardized raw forecast anomaly.
    topo_norm is your pre-normalized terrain-height field.
    |mu_e_v| is the magnitude of the predicted conditional error in normalized error space.

Important design:
    1. Mean model uses richer covariates.
    2. Variance model is stronger than lead-only NGR, but still controlled.
    3. gamma parameters have explicit L-BFGS-B bounds to avoid variance explosion.
    4. Training target is normalized error, consistent with your ForecastDataset.
    5. Evaluation is done in physical target-variable space:
            y_hat = forecast_surface + predicted_error

Input features for mean:
    - intercept
    - same-variable normalized surface forecast
    - normalized topo + geo features
    - normalized lead time
    - valid-time cyclic encoding
    - init-time cyclic encoding

Input features for variance:
    - intercept
    - lead_norm
    - abs(same-variable normalized surface forecast)
    - normalized terrain height
    - abs(predicted normalized error mean)
    - valid-time seasonal sin
    - valid-time seasonal cos

Outputs:
    - traditional_ngr_error_params.npz
    - traditional_ngr_error_test_metrics.npz
    - optionally validation metrics
"""

import os
import glob
import time
import argparse
from datetime import datetime

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import norm

import torch
from torch.utils.data import DataLoader

from serd.data.forecast_analysis_dataset import ForecastDataset
from serd.data.normalizer_forecast import DataNormalizer as DataNormalizer_fc
from serd.data.normalizer_analysis import DataNormalizer as DataNormalizer_err


# =========================================================
# Args
# =========================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="Traditional error-space NGR/EMOS baseline with controlled variance covariates"
    )

    # Paths
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./outputs/checkpoints/serd_v1/ngr_baseline")
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

    # Data split, consistent with your paper
    parser.add_argument("--train_count", type=int, default=1292)
    parser.add_argument("--valid_count", type=int, default=92)
    parser.add_argument("--use_validation", action="store_true")

    # Shape
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--num_surface_vars", type=int, default=5)
    parser.add_argument("--num_levels", type=int, default=9)

    # Data loading
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)

    # Fitting samples
    parser.add_argument("--fit_samples_per_variable", type=int, default=1000000)
    parser.add_argument("--samples_per_field", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    # Optimizer
    parser.add_argument("--maxiter", type=int, default=800)
    parser.add_argument("--l2_beta", type=float, default=1e-5)
    parser.add_argument("--l2_gamma", type=float, default=1e-2)

    # Standardized-error sigma floor
    parser.add_argument("--sigma_floor_norm", type=float, default=0.03)

    # log_sigma clipping used in both training and prediction
    parser.add_argument("--log_sigma_min", type=float, default=-5.0)
    parser.add_argument("--log_sigma_max", type=float, default=1.0)

    # gamma bounds for variance model:
    # [intercept, lead, |forecast|, topo, |mu_error|, sin_season, cos_season]
    parser.add_argument("--gamma0_min", type=float, default=-5.0)
    parser.add_argument("--gamma0_max", type=float, default=1.0)

    parser.add_argument("--gamma_lead_min", type=float, default=-1.5)
    parser.add_argument("--gamma_lead_max", type=float, default=1.5)

    parser.add_argument("--gamma_absfc_min", type=float, default=-1.0)
    parser.add_argument("--gamma_absfc_max", type=float, default=1.0)

    parser.add_argument("--gamma_topo_min", type=float, default=-1.0)
    parser.add_argument("--gamma_topo_max", type=float, default=1.0)

    parser.add_argument("--gamma_absmu_min", type=float, default=-1.0)
    parser.add_argument("--gamma_absmu_max", type=float, default=1.0)

    parser.add_argument("--gamma_season_min", type=float, default=-1.0)
    parser.add_argument("--gamma_season_max", type=float, default=1.0)

    return parser


# =========================================================
# Time / static features
# =========================================================
def encode_time_ymdh(time_str):
    """Convert 'YYYY-MM-DD-HH' into [sin(hour), cos(hour), sin(doy), cos(doy)]."""
    dt = datetime.strptime(time_str, "%Y-%m-%d-%H")
    hour = dt.hour
    doy = dt.timetuple().tm_yday

    theta_d = 2.0 * np.pi * hour / 24.0
    theta_y = 2.0 * np.pi * (doy - 1) / 365.2425

    return np.array(
        [
            np.sin(theta_d),
            np.cos(theta_d),
            np.sin(theta_y),
            np.cos(theta_y),
        ],
        dtype=np.float32,
    )


def build_static_features(args):
    """
    Return static features [C_static,H,W].

    The topo file is assumed to be already normalized:
        topo_data_Normalization.npy

    Static channels:
        topo channel(s)
        x, y
        sin(lon), cos(lon)
        sin(lat), cos(lat)
    """
    topo = np.load(args.topo_path).astype(np.float32)
    if topo.ndim != 4:
        raise ValueError(f"Expected topo [1,C,H,W], got {topo.shape}")

    topo = topo[0, :, :args.height, :args.width]  # [Ctopo,H,W]

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

    static = np.concatenate([topo, geo], axis=0)
    return static


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
    """Extract normalized surface forecast channels [B,5,H,W] from [B,45,H,W]."""
    fc_norm = fc_norm[:, :, :args.height, :args.width].astype(np.float32)
    return fc_norm[:, stats["surface_idx"], :, :]


# =========================================================
# Feature construction
# =========================================================
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
    """
    Build X_mu for selected grid points.

    X_mu columns:
        0: intercept
        1: same-variable normalized surface forecast
        2...: static topo/geo features
        lead_norm
        valid-time four cyclic features
        init-time four cyclic features
    """
    B = surface_fc_norm.shape[0]
    rows = []

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
    Build sigma base features without |mu_error|.

    Returned columns:
        0: intercept
        1: lead_norm
        2: abs(same-variable normalized surface forecast)
        3: normalized topo height, using static_features[0]
        4: sin(season) from valid time
        5: cos(season) from valid time

    During objective/prediction, |mu_error| is inserted between topo and season:
        [intercept, lead, abs_fc, topo, abs_mu, sin_season, cos_season]
    """
    B = surface_fc_norm.shape[0]
    rows = []

    for b in range(B):
        yy = points_y[b]
        xx = points_x[b]
        n = len(yy)

        intercept = np.ones((n, 1), dtype=np.float32)

        lead_norm = np.full(
            (n, 1),
            float(lead[b]) / 23.0,
            dtype=np.float32,
        )

        abs_fc = np.abs(
            surface_fc_norm[b, variable_index, yy, xx]
        )[:, None].astype(np.float32)

        # Your topo is pre-normalized. We use the first static channel as topo.
        topo = static_features[0, yy, xx][:, None].astype(np.float32)

        valid_feat = encode_time_ymdh(valid_time[b])
        sin_season = np.full((n, 1), valid_feat[2], dtype=np.float32)
        cos_season = np.full((n, 1), valid_feat[3], dtype=np.float32)

        Xb = np.concatenate(
            [
                intercept,
                lead_norm,
                abs_fc,
                topo,
                sin_season,
                cos_season,
            ],
            axis=1,
        ).astype(np.float32)

        rows.append(Xb)

    return np.concatenate(rows, axis=0)


def build_sigma_design_from_base(X_sigma_base, mu):
    """
    Insert |mu| into sigma features.

    X_sigma_base columns:
        [intercept, lead, abs_fc, topo, sin_season, cos_season]

    Output X_sigma columns:
        [intercept, lead, abs_fc, topo, abs_mu, sin_season, cos_season]
    """
    abs_mu = np.abs(mu).reshape(-1, 1).astype(np.float64)

    return np.concatenate(
        [
            X_sigma_base[:, 0:4],
            abs_mu,
            X_sigma_base[:, 4:6],
        ],
        axis=1,
    )


# =========================================================
# NGR objective and fitting
# =========================================================
def gaussian_nll_error_params(
    params,
    X_mu,
    X_sigma_base,
    y_err_norm,
    l2_beta=1e-5,
    l2_gamma=1e-2,
    sigma_floor=0.03,
    log_sigma_min=-5.0,
    log_sigma_max=1.0,
):
    """
    Error-space Gaussian NLL.

    params = [beta, gamma]

    mu = X_mu @ beta

    X_sigma =
        [1, lead, |x_v_norm|, topo, |mu|, sin_season, cos_season]

    sigma = exp(clip(X_sigma @ gamma, log_sigma_min, log_sigma_max)) + sigma_floor
    """
    P_mu = X_mu.shape[1]

    beta = params[:P_mu]
    gamma = params[P_mu:]

    mu = X_mu @ beta

    X_sigma = build_sigma_design_from_base(X_sigma_base, mu)

    log_sigma_raw = X_sigma @ gamma
    log_sigma = np.clip(log_sigma_raw, log_sigma_min, log_sigma_max)
    sigma = np.exp(log_sigma) + sigma_floor

    z = (y_err_norm - mu) / sigma
    nll = np.log(sigma) + 0.5 * z ** 2 + 0.5 * np.log(2.0 * np.pi)

    penalty = (
        l2_beta * np.sum(beta[1:] ** 2)
        + l2_gamma * np.sum(gamma[1:] ** 2)
    )

    return float(np.mean(nll) + penalty)


def fit_one_variable(X_mu, X_sigma_base, y, args):
    """
    Fit one variable by L-BFGS-B.

    Critical choices:
        - beta initialized by least squares.
        - gamma initialized from residual spread.
        - gamma has explicit bounds to avoid variance explosion.
    """
    P_mu = X_mu.shape[1]
    P_gamma = 7

    # Initialize beta by least squares for the mean.
    beta0, *_ = np.linalg.lstsq(X_mu, y, rcond=None)

    residual = y - X_mu @ beta0
    sigma0 = max(float(np.std(residual)), args.sigma_floor_norm)

    gamma0 = np.zeros(P_gamma, dtype=np.float64)
    gamma0[0] = np.log(max(sigma0 - args.sigma_floor_norm, 1e-3))
    gamma0[0] = np.clip(gamma0[0], args.gamma0_min, args.gamma0_max)

    init = np.concatenate([beta0.astype(np.float64), gamma0], axis=0)

    gamma_bounds = [
        (args.gamma0_min, args.gamma0_max),              # intercept
        (args.gamma_lead_min, args.gamma_lead_max),      # lead
        (args.gamma_absfc_min, args.gamma_absfc_max),    # |forecast|
        (args.gamma_topo_min, args.gamma_topo_max),      # topo
        (args.gamma_absmu_min, args.gamma_absmu_max),    # |mu_error|
        (args.gamma_season_min, args.gamma_season_max),  # sin_season
        (args.gamma_season_min, args.gamma_season_max),  # cos_season
    ]

    bounds = [(None, None)] * P_mu + gamma_bounds

    result = minimize(
        gaussian_nll_error_params,
        init,
        args=(
            X_mu.astype(np.float64),
            X_sigma_base.astype(np.float64),
            y.astype(np.float64),
            args.l2_beta,
            args.l2_gamma,
            args.sigma_floor_norm,
            args.log_sigma_min,
            args.log_sigma_max,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": args.maxiter,
            "ftol": 1e-8,
            "gtol": 1e-5,
            "maxls": 50,
        },
    )

    if not result.success:
        print(f"  [warn] optimizer did not fully converge: {result.message}")

    return result.x.astype(np.float64), result.fun


def predict_error_params(params, X_mu, X_sigma_base, args):
    """
    Predict normalized error mean and normalized error sigma.
    """
    P_mu = X_mu.shape[1]

    beta = params[:P_mu]
    gamma = params[P_mu:]

    mu = X_mu @ beta

    X_sigma = build_sigma_design_from_base(X_sigma_base, mu)

    log_sigma = np.clip(
        X_sigma @ gamma,
        args.log_sigma_min,
        args.log_sigma_max,
    )

    sigma = np.exp(log_sigma) + args.sigma_floor_norm

    return mu.astype(np.float32), sigma.astype(np.float32)


# =========================================================
# Training sample collection
# =========================================================
def collect_fit_samples(dataset, static_features, stats, args, variable_index):
    """
    Randomly collect grid-point samples from training set.

    Returns:
        X_mu:          [N, P_mu]
        X_sigma_base:  [N, 6]
        y:             [N]
    """
    rng = np.random.default_rng(args.seed + 2000 + variable_index)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    X_mu_parts = []
    X_sigma_base_parts = []
    y_parts = []
    total = 0

    for fc, err, lead, valid_time, init_time in loader:
        fc_np = fc.numpy()
        err_np = err.numpy()[:, :, :args.height, :args.width].astype(np.float32)
        lead_np = lead.numpy()

        surface_fc_norm = get_surface_forecast_norm(fc_np, stats, args)

        B = surface_fc_norm.shape[0]
        M = args.samples_per_field

        points_y = []
        points_x = []

        for _ in range(B):
            yy = rng.integers(0, args.height, size=M, endpoint=False)
            xx = rng.integers(0, args.width, size=M, endpoint=False)
            points_y.append(yy)
            points_x.append(xx)

        X_mu = build_mean_feature_matrix_for_points(
            surface_fc_norm=surface_fc_norm,
            static_features=static_features,
            lead=lead_np,
            valid_time=valid_time,
            init_time=init_time,
            points_y=points_y,
            points_x=points_x,
            variable_index=variable_index,
            args=args,
        )

        X_sigma_base = build_sigma_base_features_for_points(
            surface_fc_norm=surface_fc_norm,
            static_features=static_features,
            lead=lead_np,
            valid_time=valid_time,
            points_y=points_y,
            points_x=points_x,
            variable_index=variable_index,
        )

        y_list = []
        for b in range(B):
            y_list.append(err_np[b, variable_index, points_y[b], points_x[b]])

        y = np.concatenate(y_list, axis=0).astype(np.float32)

        X_mu_parts.append(X_mu)
        X_sigma_base_parts.append(X_sigma_base)
        y_parts.append(y)

        total += y.shape[0]
        if total >= args.fit_samples_per_variable:
            break

    X_mu_all = np.concatenate(X_mu_parts, axis=0)[: args.fit_samples_per_variable]
    X_sigma_base_all = np.concatenate(X_sigma_base_parts, axis=0)[: args.fit_samples_per_variable]
    y_all = np.concatenate(y_parts, axis=0)[: args.fit_samples_per_variable]

    return (
        X_mu_all.astype(np.float32),
        X_sigma_base_all.astype(np.float32),
        y_all.astype(np.float32),
    )


# =========================================================
# Metrics
# =========================================================
def crps_gaussian(mu, sigma, y):
    """
    Analytic CRPS for Gaussian predictive distribution.
    """
    sigma = np.maximum(sigma, 1e-8)
    z = (y - mu) / sigma

    return sigma * (
        z * (2.0 * ndtr(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )


def evaluate_error_ngr(params_by_var, dataset, static_features, stats, args, split_name="test"):
    """
    Evaluate full grid over a dataset.

    Predict error distribution, then convert to target-variable distribution:
        mu_y = forecast_surface + mu_error
        sigma_y = sigma_error
    """
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

    se_sum = np.zeros(C, dtype=np.float64)
    crps_sum = np.zeros(C, dtype=np.float64)
    spread_sum = np.zeros(C, dtype=np.float64)
    count = 0

    yy_grid, xx_grid = np.meshgrid(
        np.arange(args.height, dtype=np.int64),
        np.arange(args.width, dtype=np.int64),
        indexing="ij",
    )
    yy_flat = yy_grid.ravel()
    xx_flat = xx_grid.ravel()

    for fc, err, lead, valid_time, init_time in loader:
        fc_np = fc.numpy()
        err_np = err.numpy()[:, :, :args.height, :args.width].astype(np.float32)
        lead_np = lead.numpy()

        surface_fc_norm = get_surface_forecast_norm(fc_np, stats, args)

        B, _, H, W = surface_fc_norm.shape

        surface_fc_phys = (
            surface_fc_norm * stats["surface_fc_std"][None, :, None, None]
            + stats["surface_fc_mean"][None, :, None, None]
        )

        err_phys = (
            err_np * stats["err_std"][None, :, None, None]
            + stats["err_mean"][None, :, None, None]
        )

        obs_phys = surface_fc_phys + err_phys

        for b in range(B):
            for v in range(C):
                X_mu = build_mean_feature_matrix_for_points(
                    surface_fc_norm=surface_fc_norm[b:b+1],
                    static_features=static_features,
                    lead=lead_np[b:b+1],
                    valid_time=[valid_time[b]],
                    init_time=[init_time[b]],
                    points_y=[yy_flat],
                    points_x=[xx_flat],
                    variable_index=v,
                    args=args,
                )

                X_sigma_base = build_sigma_base_features_for_points(
                    surface_fc_norm=surface_fc_norm[b:b+1],
                    static_features=static_features,
                    lead=lead_np[b:b+1],
                    valid_time=[valid_time[b]],
                    points_y=[yy_flat],
                    points_x=[xx_flat],
                    variable_index=v,
                )

                mu_err_norm, sigma_err_norm = predict_error_params(
                    params_by_var[v],
                    X_mu,
                    X_sigma_base,
                    args,
                )

                mu_err_norm_2d = mu_err_norm.reshape(H, W)
                sigma_err_norm_2d = sigma_err_norm.reshape(H, W)

                mu_err_phys = (
                    mu_err_norm_2d * stats["err_std"][v]
                    + stats["err_mean"][v]
                )
                sigma_err_phys = sigma_err_norm_2d * stats["err_std"][v]

                mu_y_phys = surface_fc_phys[b, v] + mu_err_phys
                sigma_y_phys = sigma_err_phys

                se_sum[v] += np.sum((mu_y_phys - obs_phys[b, v]) ** 2)
                crps_sum[v] += np.sum(
                    crps_gaussian(mu_y_phys, sigma_y_phys, obs_phys[b, v])
                )
                spread_sum[v] += np.sum(sigma_y_phys)

            count += H * W

    rmse = np.sqrt(se_sum / max(1, count))
    crps = crps_sum / max(1, count)
    spread = spread_sum / max(1, count)
    spread_rmse = spread / np.maximum(rmse, 1e-12)

    print(f"\n[{split_name}] Error-space Traditional NGR metrics")

    var_names = ["q2m", "u10", "v10", "sp", "t2m"]
    for i, name in enumerate(var_names):
        print(
            f"  {name:>4s}: "
            f"RMSE={rmse[i]:.4f}, "
            f"CRPS={crps[i]:.4f}, "
            f"Spread/RMSE={spread_rmse[i]:.4f}"
        )

    return {
        "rmse": rmse,
        "crps": crps,
        "spread": spread,
        "spread_rmse": spread_rmse,
    }


# =========================================================
# Main
# =========================================================
def main():
    args = build_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    np.random.seed(args.seed)

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

    static_features = build_static_features(args)

    print("======================================")
    print("Traditional error-space NGR / EMOS with variance covariates")
    print(f"Train days: {len(train_files)}")
    print(f"Valid days: {len(valid_files)}")
    print(f"Test days : {len(test_files)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples : {len(test_dataset)}")
    print(f"Static channels: {static_features.shape[0]}")
    print(f"Fit samples per variable: {args.fit_samples_per_variable}")
    print(f"Samples per field: {args.samples_per_field}")
    print("Mean model: rich covariates")
    print("Variance model:")
    print("  log_sigma = gamma0 + gamma1*lead + gamma2*|x_norm|")
    print("            + gamma3*topo + gamma4*|mu_error|")
    print("            + gamma5*sin(season) + gamma6*cos(season)")
    print(
        f"log_sigma clip: [{args.log_sigma_min}, {args.log_sigma_max}], "
        f"sigma_floor_norm={args.sigma_floor_norm}"
    )
    print("======================================")

    var_names = ["q2m", "u10", "v10", "sp", "t2m"]

    params_by_var = []
    fit_losses = []

    for v, name in enumerate(var_names):
        start = time.time()
        print(f"\nFitting variable {v}: {name}")

        X_mu, X_sigma_base, y = collect_fit_samples(
            train_dataset,
            static_features,
            stats,
            args,
            variable_index=v,
        )

        print(
            f"  Collected X_mu={X_mu.shape}, "
            f"X_sigma_base={X_sigma_base.shape}, y={y.shape}"
        )

        params, fit_loss = fit_one_variable(X_mu, X_sigma_base, y, args)

        params_by_var.append(params)
        fit_losses.append(fit_loss)

        P_mu = X_mu.shape[1]
        beta = params[:P_mu]
        gamma = params[P_mu:]

        print(f"  beta range : [{beta.min():.4f}, {beta.max():.4f}]")
        print(f"  gamma value: {gamma}")

        if np.isclose(gamma[0], args.gamma0_max, atol=1e-4):
            print("  [note] gamma0 is at upper bound; spread may still be large.")
        if np.isclose(gamma[0], args.gamma0_min, atol=1e-4):
            print("  [note] gamma0 is at lower bound; spread may be very small.")

        print(
            f"  Finished {name}: NLL={fit_loss:.6f}, "
            f"time={time.time() - start:.1f}s"
        )

    params_array = np.stack(params_by_var, axis=0)

    np.savez(
        os.path.join(args.save_dir, "traditional_ngr_error_covsigma_params.npz"),
        params=params_array,
        fit_losses=np.array(fit_losses),
        err_mean=stats["err_mean"],
        err_std=stats["err_std"],
        surface_idx=stats["surface_idx"],
        args=np.array([str(vars(args))], dtype=object),
    )

    if args.use_validation:
        valid_metrics = evaluate_error_ngr(
            params_by_var,
            valid_dataset,
            static_features,
            stats,
            args,
            split_name="valid",
        )
        np.savez(
            os.path.join(args.save_dir, "traditional_ngr_error_covsigma_valid_metrics.npz"),
            **valid_metrics,
        )

    test_metrics = evaluate_error_ngr(
        params_by_var,
        test_dataset,
        static_features,
        stats,
        args,
        split_name="test",
    )

    np.savez(
        os.path.join(args.save_dir, "traditional_ngr_error_covsigma_test_metrics.npz"),
        **test_metrics,
    )

    print(f"\nSaved results to: {args.save_dir}")


if __name__ == "__main__":
    main()


