from __future__ import annotations

import numpy as np


def surface_forecast(forecast_45: np.ndarray) -> np.ndarray:
    from .spec import SURFACE_CHANNEL_INDICES, validate_forecast_shape

    validate_forecast_shape(tuple(forecast_45.shape))
    return np.asarray(forecast_45)[list(SURFACE_CHANNEL_INDICES)]


def reconstruct_total_error(raw_forecast_45: np.ndarray, error_ensemble: np.ndarray) -> np.ndarray:
    """Direct diffusion and NGR: y = raw surface forecast + sampled total error."""
    return error_ensemble + surface_forecast(raw_forecast_45)[None]


def reconstruct_two_stage(raw_forecast_45: np.ndarray, systematic_error: np.ndarray, residual_ensemble: np.ndarray) -> np.ndarray:
    """SERD/ablation: y = raw surface + stage-1 error + stage-2 residual."""
    corrected = surface_forecast(raw_forecast_45) + systematic_error
    return residual_ensemble + corrected[None]


def reconstruct_corrdiff(deterministic_analysis: np.ndarray, residual_ensemble: np.ndarray) -> np.ndarray:
    """CorrDiff: y = deterministic analysis estimate + sampled target residual."""
    return residual_ensemble + deterministic_analysis[None]
