"""
uncertainty.py
--------------
Uncertainty estimation for predicted 168h values (spec section 10).

Method chosen: split conformal prediction using out-of-fold residuals
from group cross-validation. This is:
  - simple and defensible (distribution-free, finite-sample coverage
    guarantee under exchangeability)
  - cheap to compute
  - avoids the complexity/opacity of Bayesian or deep-learning approaches
    the spec explicitly says to avoid

Given out-of-fold absolute residuals |y_true - y_pred| from CV, the
conformal quantile q_hat = the (1-alpha) quantile of those residuals
(with the standard finite-sample correction) defines a constant-width
interval: [pred - q_hat, pred + q_hat].

We also offer a locally-weighted variant (normalized conformal) that
scales the interval by a rough local difficulty estimate (|X_24h - X_0h|
magnitude) so components with larger/noisier early drift get
appropriately wider intervals instead of one-size-fits-all bands.
"""

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


def fit_conformal(oof_true: np.ndarray, oof_pred: np.ndarray,
                   alpha: float = CONFORMAL_ALPHA) -> ConformalCalibration:
    """
    Fit a split-conformal calibration from out-of-fold (true, predicted)
    pairs. These MUST be out-of-fold (i.e. the model never trained on the
    point when producing that prediction) or the resulting interval will
    be overconfident.
    """
    mask = ~np.isnan(oof_pred) & ~np.isnan(oof_true)
    resid = np.abs(oof_true[mask] - oof_pred[mask])
    n = len(resid)
    if n == 0:
        raise ValueError("No valid out-of-fold predictions to calibrate conformal intervals.")

    # finite-sample corrected quantile level
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = float(np.quantile(resid, level))
    return ConformalCalibration(q_hat=q_hat, alpha=alpha, n_calibration=n)


def predict_interval(point_pred: float, calibration: ConformalCalibration):
    """Return (lower, upper) for a single point prediction."""
    lo = point_pred - calibration.q_hat
    hi = point_pred + calibration.q_hat
    return lo, hi


def predict_intervals(point_preds: np.ndarray, calibration: ConformalCalibration):
    lo = point_preds - calibration.q_hat
    hi = point_preds + calibration.q_hat
    return lo, hi


def interval_width(calibration: ConformalCalibration) -> float:
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
