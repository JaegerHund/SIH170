"""Grouped cross-validation and split-conformal validation helpers."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, GroupShuffleSplit
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ess_predictor.config import CONFORMAL_ALPHA, N_GROUP_FOLDS
from ess_predictor.models.zoo import QRF_MODEL_NAME, predict_qrf_quantiles


@dataclass
class CVResult:
    model_name: str
    fold_mae: List[float] = field(default_factory=list)
    fold_rmse: List[float] = field(default_factory=list)
    fold_r2: List[float] = field(default_factory=list)
    fold_nmae: List[float] = field(default_factory=list)
    fold_nrmse: List[float] = field(default_factory=list)
    fold_target_scale: List[float] = field(default_factory=list)
    fold_group_labels: List[Optional[str]] = field(default_factory=list)
    fold_test_sizes: List[int] = field(default_factory=list)
    oof_predictions: Optional[np.ndarray] = None
    oof_true: Optional[np.ndarray] = None
    oof_index: Optional[np.ndarray] = None
    oof_quantile_lower: Optional[np.ndarray] = None
    oof_quantile_upper: Optional[np.ndarray] = None

    @property
    def mae(self):
        return float(np.mean(self.fold_mae))

    @property
    def rmse(self):
        return float(np.mean(self.fold_rmse))

    @property
    def r2(self):
        return float(np.mean(self.fold_r2))

    @property
    def mae_std(self):
        return float(np.std(self.fold_mae))

    @property
    def nmae(self):
        return float(np.mean(self.fold_nmae))

    @property
    def nrmse(self):
        return float(np.mean(self.fold_nrmse))

    @property
    def mae_pct_typical(self):
        return 100.0 * self.nmae


def choose_splitter(n_groups: int, n_folds: int = N_GROUP_FOLDS):
    """Use GroupKFold, or leave one group out for small datasets."""
    if n_groups < 2:
        raise ValueError("Need at least 2 distinct components/groups to cross-validate.")
    if n_groups <= n_folds:
        return LeaveOneGroupOut(), n_groups
    return GroupKFold(n_splits=n_folds), n_folds


def group_cross_validate(pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                          model_name: str, n_folds: int = N_GROUP_FOLDS) -> CVResult:
    """Score one model by group and keep its out-of-fold predictions."""
    n_groups = groups.nunique()
    splitter, k = choose_splitter(n_groups, n_folds)

    result = CVResult(model_name=model_name)
    oof_pred = np.full(len(y), np.nan)
    oof_quantile_lower = np.full(len(y), np.nan) if model_name == QRF_MODEL_NAME else None
    oof_quantile_upper = np.full(len(y), np.nan) if model_name == QRF_MODEL_NAME else None

    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)
    g_arr = groups.reset_index(drop=True)

    for train_idx, test_idx in splitter.split(X_arr, y_arr, groups=g_arr):
        # Keep each component entirely in either the train or test fold.
        train_groups = set(g_arr.iloc[train_idx])
        test_groups = set(g_arr.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups), (
            "Data leakage detected: a Component_ID appears in both train "
            "and test folds."
        )

        model = clone(pipeline)
        model.fit(X_arr.iloc[train_idx], y_arr.iloc[train_idx])
        preds = model.predict(X_arr.iloc[test_idx])
        test_group_values = pd.unique(g_arr.iloc[test_idx])
        result.fold_group_labels.append(
            str(test_group_values[0]) if len(test_group_values) == 1 else None
        )
        result.fold_test_sizes.append(int(len(test_idx)))

        if model_name == QRF_MODEL_NAME:
            quantiles = predict_qrf_quantiles(
                model, X_arr.iloc[test_idx], alpha=CONFORMAL_ALPHA
            )
            oof_quantile_lower[test_idx] = quantiles[:, 0]
            oof_quantile_upper[test_idx] = quantiles[:, 1]

        oof_pred[test_idx] = preds

        result.fold_mae.append(mean_absolute_error(y_arr.iloc[test_idx], preds))
        fold_rmse = np.sqrt(mean_squared_error(y_arr.iloc[test_idx], preds))
        result.fold_rmse.append(fold_rmse)
        # Scale by the training fold's median absolute target.
        scale = float(np.median(np.abs(y_arr.iloc[train_idx].values)))
        scale = max(scale, 1e-9)
        result.fold_target_scale.append(scale)
        result.fold_nmae.append(result.fold_mae[-1] / scale)
        result.fold_nrmse.append(fold_rmse / scale)
        # R² is undefined for a one-row test fold.
        if len(test_idx) >= 2:
            result.fold_r2.append(r2_score(y_arr.iloc[test_idx], preds))

    result.oof_predictions = oof_pred
    result.oof_true = y_arr.values
    # Retain input indices so callers can map predictions back to rows.
    result.oof_index = X.index.to_numpy()
    result.oof_quantile_lower = oof_quantile_lower
    result.oof_quantile_upper = oof_quantile_upper
    return result


def compare_models(model_zoo: dict, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                    n_folds: int = N_GROUP_FOLDS) -> Dict[str, CVResult]:
    """Run group_cross_validate for every model in `model_zoo`."""
    results = {}
    for name, pipeline in model_zoo.items():
        results[name] = group_cross_validate(pipeline, X, y, groups, name, n_folds)
    return results


def split_conformal_oof(pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                        model_name: str, n_folds: int = N_GROUP_FOLDS,
                        alpha: float = 0.10, random_state: int = 42,
                        lot_ids: Optional[pd.Series] = None,
                        early_24: Optional[np.ndarray] = None,
                        compare_qrf_intervals: bool = False):
    """Return component-disjoint OOF predictions and intervals.

    Each outer test fold is scored by a model trained on a proper-training
    subset. A separate group-disjoint calibration subset inside the outer
    training fold supplies its conformal residual quantile. This avoids using
    an observation to both calibrate and assess its own interval.
    """
    from ess_predictor.uncertainty.conformal import (
        fit_conformal,
        fit_conformalized_quantiles,
        predict_intervals,
        predict_quantile_intervals,
    )

    if compare_qrf_intervals and model_name != QRF_MODEL_NAME:
        raise ValueError("QRF interval comparison requires the QuantileRandomForest model.")

    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)
    g_arr = groups.reset_index(drop=True)
    splitter, _ = choose_splitter(g_arr.nunique(), n_folds)
    pred = np.full(len(y_arr), np.nan)
    low = np.full(len(y_arr), np.nan)
    high = np.full(len(y_arr), np.nan)
    symmetric_low = np.full(len(y_arr), np.nan) if compare_qrf_intervals else None
    symmetric_high = np.full(len(y_arr), np.nan) if compare_qrf_intervals else None
    lot_z = np.full(len(y_arr), np.nan)
    if lot_ids is not None:
        if early_24 is None or len(lot_ids) != len(y_arr) or len(early_24) != len(y_arr):
            raise ValueError("lot_ids and aligned early_24 values are both required for lot z-scores")
        lot_arr = pd.Series(lot_ids).reset_index(drop=True).to_numpy()
        early_arr = np.asarray(early_24, dtype=float)
    else:
        lot_arr = early_arr = None

    for fold_index, (outer_train, outer_test) in enumerate(
        splitter.split(X_arr, y_arr, groups=g_arr)
    ):
        outer_groups = g_arr.iloc[outer_train]
        if outer_groups.nunique() < 2:
            raise ValueError("At least three groups are required for cross-calibrated OOF intervals.")
        inner_split = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=random_state + fold_index
        )
        proper_rel, cal_rel = next(
            inner_split.split(X_arr.iloc[outer_train], y_arr.iloc[outer_train], groups=outer_groups)
        )
        proper = outer_train[proper_rel]
        calibration = outer_train[cal_rel]
        assert set(g_arr.iloc[proper]).isdisjoint(set(g_arr.iloc[calibration]))
        assert set(g_arr.iloc[outer_train]).isdisjoint(set(g_arr.iloc[outer_test]))

        fitted = clone(pipeline)
        fitted.fit(X_arr.iloc[proper], y_arr.iloc[proper])
        cal_pred = fitted.predict(X_arr.iloc[calibration])
        test_pred = fitted.predict(X_arr.iloc[outer_test])
        pred[outer_test] = test_pred

        if model_name == QRF_MODEL_NAME:
            cal_quantiles = predict_qrf_quantiles(
                fitted, X_arr.iloc[calibration], alpha=alpha
            )
            quantile_calibration = fit_conformalized_quantiles(
                y_arr.iloc[calibration].to_numpy(),
                cal_quantiles[:, 0],
                cal_quantiles[:, 1],
                alpha=alpha,
            )
            test_quantiles = predict_qrf_quantiles(
                fitted, X_arr.iloc[outer_test], alpha=alpha
            )
            test_low, test_high = predict_quantile_intervals(
                test_quantiles[:, 0], test_quantiles[:, 1], quantile_calibration
            )
            low[outer_test] = test_low
            high[outer_test] = test_high

            if compare_qrf_intervals:
                symmetric_calibration = fit_conformal(
                    y_arr.iloc[calibration].to_numpy(),
                    np.asarray(cal_pred),
                    alpha=alpha,
                )
                base_low, base_high = predict_intervals(
                    np.asarray(test_pred), symmetric_calibration
                )
                symmetric_low[outer_test] = base_low
                symmetric_high[outer_test] = base_high
        else:
            calibration_fit = fit_conformal(
                y_arr.iloc[calibration].to_numpy(), np.asarray(cal_pred), alpha=alpha
            )
            test_low, test_high = predict_intervals(np.asarray(test_pred), calibration_fit)
            low[outer_test] = test_low
            high[outer_test] = test_high

        if lot_arr is not None:
            train_values = early_arr[outer_train]
            global_mean = float(np.mean(train_values))
            global_std = max(float(np.std(train_values)), 1e-9)
            for index in outer_test:
                same_lot = outer_train[lot_arr[outer_train] == lot_arr[index]]
                reference = early_arr[same_lot] if len(same_lot) >= 5 else train_values
                mean = float(np.mean(reference))
                std = max(float(np.std(reference)), 1e-9)
                if len(same_lot) < 5:
                    mean, std = global_mean, global_std
                lot_z[index] = (early_arr[index] - mean) / std

    if compare_qrf_intervals:
        comparison = {"symmetric_low": symmetric_low, "symmetric_high": symmetric_high}
        return pred, low, high, lot_z, comparison
    return pred, low, high, lot_z


def select_best_model(results: Dict[str, CVResult]) -> str:
    """
    Select the model with lowest mean CV MAE (spec: MAE is the primary
    regression metric because it's interpretable in original electrical
    units). Ties broken by RMSE.
    """
    return min(results, key=lambda k: (results[k].mae, results[k].rmse))


def condition_generalization_check(pipeline, X: pd.DataFrame, y: pd.Series,
                                    condition: pd.Series, model_name: str) -> CVResult:
    """
    Additional validation split by an arbitrary condition column
    (Lot_ID / Device_Type / Temperature), using LeaveOneGroupOut over the
    distinct values of `condition`. Tests whether the model generalizes to
    an entirely unseen lot/device type/temperature, not just an unseen
    component within seen conditions.
    """
    return group_cross_validate(pipeline, X, y, condition, model_name,
                                 n_folds=condition.nunique())


if __name__ == "__main__":
    from ess_predictor.data.synthetic_data import generate_synthetic_dataset
    from ess_predictor.features.engineering import build_feature_matrix
    from ess_predictor.models.zoo import build_model_zoo

    df = generate_synthetic_dataset(150)
    X, y, groups = build_feature_matrix(df, "Iddq")
    zoo = build_model_zoo()
    results = compare_models(zoo, X, y, groups)
    for name, r in results.items():
        print(f"{name:28s} MAE={r.mae:.4f} (+/-{r.mae_std:.4f})  RMSE={r.rmse:.4f}  R2={r.r2:.4f}")
    print("Best:", select_best_model(results))
