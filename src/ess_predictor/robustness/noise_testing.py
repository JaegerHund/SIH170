"""
robustness.py
--------------
Measurement-noise sensitivity testing (spec section 11).

Perturbs 0h/24h input measurements within a realistic measurement-error
band and observes how much the predicted 168h value moves. A robust model
should not produce wildly different predictions for tiny, realistic
measurement changes.
"""

from typing import Callable, List

import numpy as np
import pandas as pd

from ess_predictor.features.engineering import build_features_for_parameter


def perturb_and_predict(
    df_single_row: pd.DataFrame,
    param_name: str,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    noise_fracs: List[float] = (-0.005, -0.002, 0.0, 0.002, 0.005),
    lot_col: str = "Lot_ID",
) -> pd.DataFrame:
    """
    df_single_row: one-row DataFrame containing at least the raw columns
        needed for this parameter (and ideally the same lot context used
        at training time for correct z-score computation).
    predict_fn: function mapping a built feature DataFrame -> point predictions.
    noise_fracs: fractional perturbations applied to the 0h measurement
        (e.g. -0.005 = -0.5%), simulating realistic measurement noise.

    Returns a DataFrame with columns [noise_frac, perturbed_0h, predicted_168h].
    """
    col0 = f"{param_name}_0h"
    base_val = df_single_row.iloc[0][col0]

    rows = []
    for frac in noise_fracs:
        perturbed = df_single_row.copy()
        perturbed.loc[perturbed.index[0], col0] = base_val * (1 + frac)
        X = build_features_for_parameter(perturbed, param_name, lot_col=lot_col)
        pred = predict_fn(X)[0]
        rows.append({
            "noise_frac": frac,
            "perturbed_0h": base_val * (1 + frac),
            "predicted_168h": pred,
        })

    result = pd.DataFrame(rows)
    return result


def robustness_summary(result: pd.DataFrame) -> dict:
    """
    Summarize sensitivity: max prediction swing across the tested noise
    band, and that swing as a fraction of the baseline (noise_frac=0)
    prediction.
    """
    baseline_row = result.loc[result["noise_frac"] == 0.0]
    baseline_pred = float(baseline_row["predicted_168h"].iloc[0]) if len(baseline_row) else \
        float(result["predicted_168h"].mean())

    pred_range = result["predicted_168h"].max() - result["predicted_168h"].min()
    rel_swing = pred_range / max(abs(baseline_pred), 1e-9)

    return {
        "baseline_prediction": baseline_pred,
        "prediction_range": float(pred_range),
        "relative_swing": float(rel_swing),
        "stable": bool(rel_swing < 0.05),  # <5% swing for <=0.5% input noise = stable
    }


def batch_robustness_test(
    df: pd.DataFrame,
    param_name: str,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    n_samples: int = 20,
    noise_fracs: List[float] = (-0.005, -0.002, 0.0, 0.002, 0.005),
    lot_col: str = "Lot_ID",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the robustness test across a random sample of components and
    return a summary table (one row per component) so an engineer can
    quickly see whether ANY components show unstable prediction behavior.
    """
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(df.index, size=min(n_samples, len(df)), replace=False)

    summaries = []
    for idx in sample_idx:
        row_df = df.loc[[idx]]
        result = perturb_and_predict(row_df, param_name, predict_fn, noise_fracs, lot_col)
        summ = robustness_summary(result)
        summ["Component_ID"] = row_df.iloc[0].get("Component_ID", idx)
        summaries.append(summ)

    return pd.DataFrame(summaries)
