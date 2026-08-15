from __future__ import annotations

import numpy as np


def empirical_crps(ensemble: np.ndarray, observation: np.ndarray, member_axis: int = 0) -> np.ndarray:
    """Ordinary empirical-ensemble CRPS used by Table 3 and Figures 4--8."""
    ens = np.asarray(ensemble, dtype=np.float64)
    obs = np.asarray(observation, dtype=np.float64)
    ens = np.moveaxis(ens, member_axis, 0)
    if ens.shape[0] < 1:
        raise ValueError("ensemble must contain at least one member")
    term1 = np.mean(np.abs(ens - obs), axis=0)
    term2 = 0.5 * np.mean(np.abs(ens[:, None] - ens[None, :]), axis=(0, 1))
    return term1 - term2


def ensemble_spread(ensemble: np.ndarray, member_axis: int = 0) -> np.ndarray:
    ens = np.asarray(ensemble, dtype=np.float64)
    if ens.shape[member_axis] < 2:
        raise ValueError("sample spread with ddof=1 requires at least two members")
    return np.std(ens, axis=member_axis, ddof=1)


def coverage(ensemble: np.ndarray, observation: np.ndarray, nominal: float, member_axis: int = 0) -> float:
    if not 0.0 < nominal < 1.0:
        raise ValueError("nominal coverage must lie in (0,1)")
    ens = np.moveaxis(np.asarray(ensemble), member_axis, 0)
    obs = np.asarray(observation)
    alpha = 1.0 - nominal
    lower = np.quantile(ens, alpha / 2.0, axis=0)
    upper = np.quantile(ens, 1.0 - alpha / 2.0, axis=0)
    return float(np.mean((obs >= lower) & (obs <= upper)))


def absolute_coverage_error(actual: float, nominal: float) -> float:
    return abs(float(actual) - float(nominal))


def rank_histogram(ensemble: np.ndarray, observation: np.ndarray, member_axis: int = 0) -> np.ndarray:
    ens = np.moveaxis(np.asarray(ensemble), member_axis, 0)
    obs = np.asarray(observation)
    ranks = np.sum(ens < obs, axis=0)
    return np.bincount(ranks.ravel(), minlength=ens.shape[0] + 1)
