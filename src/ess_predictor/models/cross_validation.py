"""
validation.py
--------------
Component-level / group-level cross-validation (spec section 6-7).

CRITICAL: measurements from the same Component_ID must never appear in
both the training and test fold for the SAME evaluation. We use
GroupKFold keyed on Component_ID (or LeaveOneGroupOut when the dataset is
small) so a component's own 0h/24h features are never used to help
predict that same component's 168h value during evaluation.

Also supports an additional generalization check by splitting on
Lot_ID / Device_Type / Temperature (spec section 7) to test whether a
model generalizes beyond conditions seen during training.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ess_predictor.config import N_GROUP_FOLDS


@dataclass
class CVResult:
    model_name: str
    fold_mae: List[float] = field(default_factory=list)
    fold_rmse: List[float] = field(default_factory=list)
    fold_r2: List[float] = field(default_factory=list)
    oof_predictions: Optional[np.ndarray] = None
    oof_true: Optional[np.ndarray] = None
    oof_index: Optional[np.ndarray] = None

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


def choose_splitter(n_groups: int, n_folds: int = N_GROUP_FOLDS):
    """
    GroupKFold by default. Falls back toward Leave-One-Group-Out
    (or as close to it as n_groups allows) when the dataset is small,
    per spec section 7.
    """
    if n_groups < 2:
        raise ValueError("Need at least 2 distinct components/groups to cross-validate.")
    if n_groups <= n_folds:
        return LeaveOneGroupOut(), n_groups
    return GroupKFold(n_splits=n_folds), n_folds


def group_cross_validate(pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                          model_name: str, n_folds: int = N_GROUP_FOLDS) -> CVResult:
    """
    Run component-level GroupKFold (or LOGO for small N) cross-validation
    for a single sklearn pipeline. Returns out-of-fold predictions too,
    which downstream code uses for safety-threshold calibration and
    conformal-prediction interval fitting -- always evaluated on data the
    model did not train on.
    """
    n_groups = groups.nunique()
    splitter, k = choose_splitter(n_groups, n_folds)

    result = CVResult(model_name=model_name)
    oof_pred = np.full(len(y), np.nan)

    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)
    g_arr = groups.reset_index(drop=True)

    for train_idx, test_idx in splitter.split(X_arr, y_arr, groups=g_arr):
        # Hard assertion: no group overlap between train/test folds.
        train_groups = set(g_arr.iloc[train_idx])
        test_groups = set(g_arr.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups), (
            "Data leakage detected: a Component_ID appears in both train "
            "and test folds."
        )

        model = clone(pipeline)
        model.fit(X_arr.iloc[train_idx], y_arr.iloc[train_idx])
        preds = model.predict(X_arr.iloc[test_idx])

        oof_pred[test_idx] = preds

        result.fold_mae.append(mean_absolute_error(y_arr.iloc[test_idx], preds))
        result.fold_rmse.append(np.sqrt(mean_squared_error(y_arr.iloc[test_idx], preds)))
        # R2 needs >= 2 points to be meaningful; guard tiny folds (LOGO edge case)
        if len(test_idx) >= 2:
            result.fold_r2.append(r2_score(y_arr.iloc[test_idx], preds))

    result.oof_predictions = oof_pred
    result.oof_true = y_arr.values
    result.oof_index = X_arr.index.values
    return result


def compare_models(model_zoo: dict, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                    n_folds: int = N_GROUP_FOLDS) -> Dict[str, CVResult]:
    """Run group_cross_validate for every model in `model_zoo`."""
    results = {}
    for name, pipeline in model_zoo.items():
        results[name] = group_cross_validate(pipeline, X, y, groups, name, n_folds)
    return results


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
