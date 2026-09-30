"""Calibrate prediction intervals with held-out residuals."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ess_predictor.config import CONFORMAL_ALPHA


@dataclass
class ConformalCalibration:
    q_hat: float
    alpha: float
    n_calibration: int
    method: str = "absolute_residual"


@dataclass
class QuantileConformalCalibration:
    """Calibration for conformalized, input-dependent quantile intervals."""
    q_hat: float
    alpha: float
    n_calibration: int
    lower_quantile: float
    upper_quantile: float
    method: str = "conformalized_quantile_regression"


def fit_conformal(oof_true: np.ndarray, oof_pred: np.ndarray,
                   alpha: float = CONFORMAL_ALPHA) -> ConformalCalibration:
    """
    Fit split-conformal calibration from held-out (true, predicted) pairs.
    The estimator that produced `oof_pred` must not have trained on those
    calibration rows. Calibration can be from a dedicated holdout or from
    an inner cross-validation procedure.
    """
    mask = ~np.isnan(oof_pred) & ~np.isnan(oof_true)
    resid = np.abs(oof_true[mask] - oof_pred[mask])
    n = len(resid)
    if n == 0:
        raise ValueError("No valid out-of-fold predictions to calibrate conformal intervals.")

    # Split-conformal order statistic. If the requested finite-sample rank is
    # beyond the available calibration residuals, the valid interval is unbounded.
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        q_hat = float("inf")
    else:
        q_hat = float(np.partition(resid, rank - 1)[rank - 1])
    return ConformalCalibration(q_hat=q_hat, alpha=alpha, n_calibration=n)


def fit_conformalized_quantiles(y_true: np.ndarray, lower_pred: np.ndarray,
                                upper_pred: np.ndarray,
                                alpha: float = CONFORMAL_ALPHA,
                                lower_quantile: Optional[float] = None,
                                upper_quantile: Optional[float] = None
                                ) -> QuantileConformalCalibration:
    """Calibrate lower/upper quantile predictions with split conformal scores.

    The score is ``max(lower - y, y - upper)``. The conformal correction is
    constrained to be nonnegative, so it only expands the QRF interval and
    cannot turn it into an inverted interval.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower_pred = np.asarray(lower_pred, dtype=float)
    upper_pred = np.asarray(upper_pred, dtype=float)
    if not (y_true.shape == lower_pred.shape == upper_pred.shape):
        raise ValueError("Truth and lower/upper quantile predictions must have matching shapes.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    mask = np.isfinite(y_true) & np.isfinite(lower_pred) & np.isfinite(upper_pred)
    y_true = y_true[mask]
    lower_pred = lower_pred[mask]
    upper_pred = upper_pred[mask]
    if np.any(lower_pred > upper_pred):
        raise ValueError("Lower quantile predictions must not exceed upper predictions.")

    scores = np.maximum(lower_pred - y_true, y_true - upper_pred)
    n = len(scores)
    if n == 0:
        raise ValueError("No valid quantile calibration rows were provided.")

    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        q_hat = float("inf")
    else:
        q_hat = float(np.partition(scores, rank - 1)[rank - 1])
    q_hat = max(q_hat, 0.0)

    return QuantileConformalCalibration(
        q_hat=q_hat,
        alpha=alpha,
        n_calibration=n,
        lower_quantile=lower_quantile if lower_quantile is not None else alpha / 2,
        upper_quantile=upper_quantile if upper_quantile is not None else 1 - alpha / 2,
    )


def predict_interval(point_pred: float, calibration: ConformalCalibration):
    """Return (lower, upper) for a single point prediction."""
    if isinstance(calibration, QuantileConformalCalibration):
        raise ValueError("QRF intervals require native lower/upper quantiles; use predict_quantile_intervals.")
    lo = point_pred - calibration.q_hat
    hi = point_pred + calibration.q_hat
    return lo, hi


def predict_intervals(point_preds: np.ndarray, calibration: ConformalCalibration):
    if isinstance(calibration, QuantileConformalCalibration):
        raise ValueError("QRF intervals require native lower/upper quantiles; use predict_quantile_intervals.")
    lo = point_preds - calibration.q_hat
    hi = point_preds + calibration.q_hat
    return lo, hi


def predict_quantile_intervals(lower_preds: np.ndarray, upper_preds: np.ndarray,
                               calibration: QuantileConformalCalibration):
    """Expand predicted quantile bounds by their calibrated correction."""
    lower_preds = np.asarray(lower_preds, dtype=float)
    upper_preds = np.asarray(upper_preds, dtype=float)
    return lower_preds - calibration.q_hat, upper_preds + calibration.q_hat


def interval_width(calibration: ConformalCalibration,
                   lower_pred: Optional[float] = None,
                   upper_pred: Optional[float] = None) -> float:
    if isinstance(calibration, QuantileConformalCalibration):
        if lower_pred is None or upper_pred is None:
            raise ValueError("QRF interval width requires predicted lower and upper quantiles.")
        return (upper_pred - lower_pred) + 2 * calibration.q_hat
    return 2 * calibration.q_hat


def low_confidence_flag(point_pred: float, calibration: ConformalCalibration,
                         relative_width_threshold: float = 0.5) -> bool:
    """
    Flag "not enough confidence to decide reliably" when the interval half
    width is large relative to the magnitude of the prediction itself.
    Used to route borderline / high-uncertainty cases to REVIEW rather
    than auto-deciding SAFE (spec section 10).
    """
    denom = max(abs(point_pred), 1e-9)
    return (calibration.q_hat / denom) > relative_width_threshold


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = rng.normal(15, 3, 200)
    y_pred = y_true + rng.normal(0, 1, 200)
    cal = fit_conformal(y_true, y_pred)
    print(cal)
    print(predict_interval(16.7, cal))
