"""Build per-parameter features from early readouts."""

import numpy as np
import pandas as pd

from ess_predictor.config import MIN_LOT_SIZE_FOR_STATS, EARLY_TIMEPOINTS


EPS = 1e-9  # safety epsilon for division-by-zero guards


def _safe_relative_drift(x24: pd.Series, x0: pd.Series) -> pd.Series:
    denom = x0.abs().where(x0.abs() > EPS, EPS)
    return (x24 - x0) / denom


def _lot_stats(df: pd.DataFrame, value_col: str, lot_col: str = "Lot_ID"):
    """
    Compute per-lot mean/std for `value_col`. Lots smaller than
    MIN_LOT_SIZE_FOR_STATS fall back to the global mean/std so z-scores
    don't become unstable/meaningless for tiny lots.
    """
    global_mean = df[value_col].mean()
    global_std = df[value_col].std(ddof=0)
    global_std = global_std if global_std > EPS else EPS

    lot_sizes = df.groupby(lot_col)[value_col].transform("count")
    lot_mean = df.groupby(lot_col)[value_col].transform("mean")
    lot_std = df.groupby(lot_col)[value_col].transform(lambda s: s.std(ddof=0))
    lot_std = lot_std.where(lot_std > EPS, EPS)

    use_lot = lot_sizes >= MIN_LOT_SIZE_FOR_STATS
    mean_used = lot_mean.where(use_lot, global_mean)
    std_used = lot_std.where(use_lot, global_std)
    return mean_used, std_used


def build_features_for_parameter(
    df: pd.DataFrame,
    param_name: str,
    lot_col: str = "Lot_ID",
    include_96h: bool = False,
    include_lot_stats: bool = True,
) -> pd.DataFrame:
    """
    Given the raw dataframe and a parameter name (e.g. "Iddq"), return a
    new DataFrame of derived features for that parameter, indexed the same
    as `df`. Column names are prefixed with the parameter name so multiple
    parameters' feature sets can be safely concatenated.

    Does NOT include the target (168h) column -- callers pull that
    separately to avoid any risk of leaking it into X.
    """
    out = pd.DataFrame(index=df.index)
    p = param_name

    col0 = f"{p}_0h"
    col24 = f"{p}_24h"
    col96 = f"{p}_96h"

    if col0 not in df.columns or col24 not in df.columns:
        raise ValueError(f"Missing required early columns for parameter '{p}'.")

    x0 = df[col0].astype(float)
    x24 = df[col24].astype(float)

    # --- Absolute measurements ---
    out[f"{p}_0h"] = x0
    out[f"{p}_24h"] = x24
    has_96 = include_96h and (col96 in df.columns) and df[col96].notna().any()
    if has_96:
        out[f"{p}_96h"] = df[col96].astype(float)

    # --- Early drift ---
    out[f"{p}_dX_0_24"] = x24 - x0

    # --- Relative drift (safe) ---
    out[f"{p}_relative_drift_0_24"] = _safe_relative_drift(x24, x0)

    # --- Drift rate (per hour) ---
    out[f"{p}_drift_rate_0_24"] = (x24 - x0) / 24.0

    if include_lot_stats and lot_col in df.columns:
        mean0, std0 = _lot_stats(df.assign(**{col0: x0}), col0, lot_col)
        mean24, std24 = _lot_stats(df.assign(**{col24: x24}), col24, lot_col)
        out[f"{p}_z_0h"] = (x0 - mean0) / std0
        out[f"{p}_z_24h"] = (x24 - mean24) / std24

        dcol = f"__{p}_dX_tmp"
        tmp = df.assign(**{dcol: x24 - x0})
        mean_d, std_d = _lot_stats(tmp, dcol, lot_col)
        out[f"{p}_z_dX_0_24"] = ((x24 - x0) - mean_d) / std_d
    elif include_lot_stats:
        # No lot column available: use global early-measurement statistics.
        out[f"{p}_z_0h"] = (x0 - x0.mean()) / max(x0.std(ddof=0), EPS)
        out[f"{p}_z_24h"] = (x24 - x24.mean()) / max(x24.std(ddof=0), EPS)
        d = x24 - x0
        out[f"{p}_z_dX_0_24"] = (d - d.mean()) / max(d.std(ddof=0), EPS)

    if has_96:
        x96 = df[col96].astype(float)
        out[f"{p}_dX_24_96"] = x96 - x24
        out[f"{p}_relative_drift_24_96"] = _safe_relative_drift(x96, x24)
        out[f"{p}_drift_rate_24_96"] = (x96 - x24) / (96.0 - 24.0)

    return out


def build_feature_matrix(
    df: pd.DataFrame,
    param_name: str,
    lot_col: str = "Lot_ID",
    include_96h: bool = False,
    include_lot_stats: bool = False,
):
    """
    Convenience wrapper: returns (X, y, groups) for a given
    parameter, where y is the 168h target (NaN rows dropped) and X is
    aligned to y's remaining index. Rows without a valid target are
    excluded here -- this is the ONLY place target-based filtering happens,
    keeping feature engineering itself target-agnostic and leak-free.
    """
    if include_96h:
        raise ValueError(
            "Prediction feature matrices are restricted to 0h/24h inputs; "
            "96h values are future information for the 168h prediction task."
        )
    X_all = build_features_for_parameter(
        df,
        param_name,
        lot_col=lot_col,
        include_96h=False,
        include_lot_stats=include_lot_stats,
    )
    target_col = f"{param_name}_168h"
    if target_col not in df.columns:
        return X_all, None, list(X_all.columns)

    y = df[target_col].astype(float)
    valid = y.notna() & X_all.notna().all(axis=1)
    X = X_all.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    groups = df.loc[valid, "Component_ID"].reset_index(drop=True) if "Component_ID" in df.columns else None
    return X, y, groups


if __name__ == "__main__":
    from ess_predictor.data.synthetic_data import generate_synthetic_dataset
    df = generate_synthetic_dataset(50)
    X, y, groups = build_feature_matrix(df, "Iddq")
    print(X.shape, y.shape)
    print(X.head())
    print(X.columns.tolist())
