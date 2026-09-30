"""Train and apply one 168 h regression model per parameter."""

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ess_predictor.config import (
    PARAMETERS,
    ParameterConfig,
    N_GROUP_FOLDS,
    CONFORMAL_ALPHA,
    RANDOM_STATE,
    MIN_LOT_SIZE_FOR_STATS,
)
from ess_predictor.data.validation import validate_dataset, handle_missing_values
from ess_predictor.features.engineering import build_feature_matrix, build_features_for_parameter
from ess_predictor.models.zoo import (
    QRF_MODEL_NAME,
    build_model_zoo,
    predict_qrf_quantiles,
)
from ess_predictor.models.cross_validation import (
    compare_models,
    select_best_model,
    split_conformal_oof,
    CVResult,
)
from ess_predictor.uncertainty.conformal import (
    ConformalCalibration,
    QuantileConformalCalibration,
    fit_conformal,
    fit_conformalized_quantiles,
    predict_interval,
    predict_quantile_intervals,
)
from ess_predictor.safety.decision import evaluate_safety, physics_plausibility_flags, safety_confusion_metrics
from ess_predictor.explainability.explain import explain_prediction, global_feature_importance


@dataclass
class ParameterModelBundle:
    """Model and validation results for one parameter."""
    param_cfg: ParameterConfig
    best_model_name: str
    pipeline: object                 # fitted sklearn Pipeline
    cv_results: Dict[str, CVResult]
    conformal: Union[ConformalCalibration, QuantileConformalCalibration]
    feature_columns: list
    safety_metrics: dict
    interval_comparison: dict = field(default_factory=dict)
    explanation_reference: Optional[pd.Series] = None
    lot_cv_results: Dict[str, CVResult] = field(default_factory=dict)
    calibration_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class TrainedSystem:
    bundles: Dict[str, ParameterModelBundle] = field(default_factory=dict)
    lot_reference: Optional[pd.DataFrame] = None   # lot mean/std reference for inference-time z-scores


def train_all_parameters(df: pd.DataFrame, n_folds: int = N_GROUP_FOLDS,
                          verbose: bool = True) -> TrainedSystem:
    """Validate the data, select models, and fit intervals for each parameter."""
    report = validate_dataset(df)
    if verbose:
        print(report.summary())
        print()

    usable_names = [p.name for p in report.usable_parameters]
    df = handle_missing_values(df, usable_names)

    system = TrainedSystem()

    for p in report.usable_parameters:
        target_col = f"{p.name}_168h"
        if target_col not in df.columns or df[target_col].notna().sum() < 10:
            if verbose:
                print(f"Skipping {p.name}: insufficient labeled 168h data for training.")
            continue

        X, y, groups = build_feature_matrix(df, p.name)
        if groups is None or groups.nunique() < 2:
            if verbose:
                print(f"Skipping {p.name}: not enough distinct components to group-validate.")
            continue

        lot_ids = None
        if "Lot_ID" in df.columns:
            component_lots = (
                df.drop_duplicates("Component_ID")
                .set_index("Component_ID")["Lot_ID"]
            )
            lot_ids = groups.map(component_lots).reset_index(drop=True)

        # Hold out components for conformal calibration.
        if groups.nunique() >= 4:
            split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
            fit_idx, cal_idx = next(split.split(X, y, groups=groups))
            assert set(groups.iloc[fit_idx]).isdisjoint(set(groups.iloc[cal_idx]))
        else:
            # With too few components, calibrate from out-of-fold residuals.
            fit_idx = np.arange(len(X))
            cal_idx = np.array([], dtype=int)
        X_fit, y_fit, groups_fit = X.iloc[fit_idx], y.iloc[fit_idx], groups.iloc[fit_idx]
        zoo = build_model_zoo(include_baselines=True)
        cv_results = compare_models(zoo, X_fit, y_fit, groups_fit, n_folds=n_folds)
        lot_cv_results = {}
        if lot_ids is not None:
            fit_lots = lot_ids.iloc[fit_idx].reset_index(drop=True)
            if fit_lots.nunique() >= 2:
                # Hold out one lot at a time; these scores select the model.
                lot_cv_results = compare_models(
                    zoo,
                    X_fit,
                    y_fit,
                    fit_lots,
                    n_folds=int(fit_lots.nunique()),
                )
        best_name = select_best_model(lot_cv_results or cv_results)

        if verbose:
            print(f"[{p.name}] Component-grouped CV MAE ({p.unit}):")
            for name, r in cv_results.items():
                marker = "  <== best by component CV" if name == best_name and not lot_cv_results else ""
                print(
                    f"    {name:28s} MAE={r.mae:.4f}  RMSE={r.rmse:.4f}  "
                    f"R2={r.r2:.4f}  NMAE={r.nmae:.3f} "
                    f"(MAE={r.mae_pct_typical:.1f}% median |target|){marker}"
                )
            if lot_cv_results:
                print(f"    Leave-one-lot-out MAE ({p.unit}; used for model selection):")
                for name, r in lot_cv_results.items():
                    marker = "  <== selected" if name == best_name else ""
                    per_lot = ", ".join(
                        f"{lot}={mae:.4f}"
                        for lot, mae in zip(r.fold_group_labels, r.fold_mae)
                        if lot is not None
                    )
                    print(f"      {name:28s} mean={r.mae:.4f}  [{per_lot}]{marker}")
            print("    Baselines are evaluated alongside regressors; selection uses lowest MAE.")
            print()

        best_cv = cv_results[best_name]
        best_pipeline = zoo[best_name]
        best_pipeline.fit(X_fit, y_fit)
        calibration_predictions = pd.DataFrame()
        if len(cal_idx):
            calibration_point_predictions = np.asarray(
                best_pipeline.predict(X.iloc[cal_idx]), dtype=float
            )
            if best_name == QRF_MODEL_NAME:
                cal_quantiles = predict_qrf_quantiles(
                    best_pipeline, X.iloc[cal_idx], alpha=CONFORMAL_ALPHA
                )
                conformal = fit_conformalized_quantiles(
                    y.iloc[cal_idx].to_numpy(),
                    cal_quantiles[:, 0],
                    cal_quantiles[:, 1],
                    alpha=CONFORMAL_ALPHA,
                )
            else:
                conformal = fit_conformal(
                    y.iloc[cal_idx].to_numpy(),
                    calibration_point_predictions,
                    alpha=CONFORMAL_ALPHA,
                )
            calibration_predictions = pd.DataFrame({
                "Component_ID": groups.iloc[cal_idx].to_numpy(),
                "Lot_ID": (
                    lot_ids.iloc[cal_idx].to_numpy() if lot_ids is not None else None
                ),
                "Parameter": p.name,
                "Model": best_name,
                "Prediction_Basis": "Held-out component split (not used to fit/select point model)",
                "Actual_168h": y.iloc[cal_idx].to_numpy(),
                "Predicted_168h": calibration_point_predictions,
            })
        else:
            # Fallback for too few groups: cross-validated residual calibration
            # is approximate and should be treated as exploratory.
            if best_name == QRF_MODEL_NAME:
                conformal = fit_conformalized_quantiles(
                    best_cv.oof_true,
                    best_cv.oof_quantile_lower,
                    best_cv.oof_quantile_upper,
                    alpha=CONFORMAL_ALPHA,
                )
            else:
                conformal = fit_conformal(
                    best_cv.oof_true, best_cv.oof_predictions, alpha=CONFORMAL_ALPHA
                )

        value_0h = X[f"{p.name}_0h"].values
        value_24h = X[f"{p.name}_24h"].values
        qrf_selected = best_name == QRF_MODEL_NAME
        oof_result = split_conformal_oof(
            zoo[best_name], X, y, groups, best_name,
            n_folds=n_folds, alpha=CONFORMAL_ALPHA,
            lot_ids=lot_ids, early_24=value_24h if lot_ids is not None else None,
            compare_qrf_intervals=qrf_selected,
        )
        interval_comparison = {}
        if qrf_selected:
            (
                safety_pred,
                safety_low,
                safety_high,
                safety_lot_z,
                symmetric_intervals,
            ) = oof_result
        else:
            safety_pred, safety_low, safety_high, safety_lot_z = oof_result

        def decisions_for_interval(lower, upper):
            return [
                evaluate_safety(v0, v24, pred, lo, hi, p, lot_z_24h=z).decision
                for v0, v24, pred, lo, hi, z in zip(
                    value_0h, value_24h, safety_pred, lower, upper, safety_lot_z
                )
            ]

        oof_decisions = decisions_for_interval(safety_low, safety_high)

        safety_metrics = safety_confusion_metrics(
            y_true_168h=y.to_numpy(),
            value_0h=value_0h,
            decisions=oof_decisions,
            param_cfg=p,
        )

        if qrf_selected:
            symmetric_decisions = decisions_for_interval(
                symmetric_intervals["symmetric_low"],
                symmetric_intervals["symmetric_high"],
            )
            symmetric_metrics = safety_confusion_metrics(
                y_true_168h=y.to_numpy(),
                value_0h=value_0h,
                decisions=symmetric_decisions,
                param_cfg=p,
            )

            def summarize_intervals(lower, upper, metrics):
                lower = np.asarray(lower, dtype=float)
                upper = np.asarray(upper, dtype=float)
                covered = (y.to_numpy() >= lower) & (y.to_numpy() <= upper)
                return {
                    "coverage": float(np.mean(covered)),
                    "mean_width": float(np.mean(upper - lower)),
                    "review_count": metrics["review_count"],
                    "reject_count": metrics["reject_count"],
                    "flagged_count": metrics["review_count"] + metrics["reject_count"],
                    "recall": metrics["recall_sensitivity"],
                    "false_positive_rate": metrics["false_positive_rate"],
                    "specificity": metrics["specificity"],
                }

            interval_comparison = {
                "symmetric_conformal": summarize_intervals(
                    symmetric_intervals["symmetric_low"],
                    symmetric_intervals["symmetric_high"],
                    symmetric_metrics,
                ),
                "qrf_conformalized_quantiles": summarize_intervals(
                    safety_low, safety_high, safety_metrics
                ),
            }

        if verbose:
            cm = safety_metrics["confusion_matrix"]
            print(f"    Safety metrics: Recall={safety_metrics['recall_sensitivity']:.3f}  "
                  f"Precision={safety_metrics['precision']:.3f}  "
                  f"FNR={safety_metrics['false_negative_rate']:.3f}  "
                  f"FPR={safety_metrics['false_positive_rate']:.3f}  "
                  f"Specificity={safety_metrics['specificity']:.3f}  "
                  f"REVIEW={safety_metrics['review_count']}  "
                  f"REJECT={safety_metrics['reject_count']}  "
                  f"threshold={safety_metrics['relative_drift_threshold']:.3f}  "
                  f"CM={cm}")
            print()
            if interval_comparison:
                for method, metrics in interval_comparison.items():
                    print(
                        f"    {method}: coverage={metrics['coverage']:.3f}  "
                        f"mean_width={metrics['mean_width']:.4f}  "
                        f"REVIEW={metrics['review_count']}  "
                        f"REJECT={metrics['reject_count']}  "
                        f"Recall={metrics['recall']:.3f}  "
                        f"FPR={metrics['false_positive_rate']:.3f}"
                    )
                print()

        system.bundles[p.name] = ParameterModelBundle(
            param_cfg=p,
            best_model_name=best_name,
            pipeline=best_pipeline,
            cv_results=cv_results,
            conformal=conformal,
            feature_columns=list(X.columns),
            safety_metrics=safety_metrics,
            interval_comparison=interval_comparison,
            explanation_reference=X_fit.median(axis=0),
            lot_cv_results=lot_cv_results,
            calibration_predictions=calibration_predictions,
        )

    # Keep lot-level reference values for new components.
    if "Lot_ID" in df.columns:
        system.lot_reference = df[["Lot_ID"] + [
            c for p in system.bundles for c in [f"{p}_0h", f"{p}_24h"] if c in df.columns
        ]].copy()

    return system


def predict_component(system: TrainedSystem, component_row: pd.DataFrame,
                       lot_reference_df: Optional[pd.DataFrame] = None,
                       include_explanations: bool = True) -> Dict[str, dict]:
    """
    Score a single new component (one-row DataFrame with raw 0h/24h/lot
    columns) across every trained parameter. Returns a dict keyed by
    parameter name with prediction, interval, decision, explanation, etc.

    `lot_reference_df` should be a dataframe (e.g. the training set, or a
    dedicated lot-stats table) containing enough rows from the component's
    lot to compute meaningful z-scores; if omitted, falls back to
    system.lot_reference captured at training time.
    """
    ref = lot_reference_df if lot_reference_df is not None else system.lot_reference
    if (
        ref is not None
        and "Component_ID" in ref.columns
        and "Component_ID" in component_row.columns
    ):
        scored_id = component_row.iloc[0]["Component_ID"]
        ref = ref[ref["Component_ID"] != scored_id]
    results = {}

    for pname, bundle in system.bundles.items():
        cfg = bundle.param_cfg
        col0 = f"{pname}_0h"
        col24 = f"{pname}_24h"
        if col0 not in component_row.columns or col24 not in component_row.columns:
            continue

        X_full = build_features_for_parameter(
            component_row, pname, include_lot_stats=False
        )
        X_row = X_full[bundle.feature_columns]

        pred = float(bundle.pipeline.predict(X_row)[0])
        if isinstance(bundle.conformal, QuantileConformalCalibration):
            quantiles = predict_qrf_quantiles(
                bundle.pipeline, X_row, alpha=bundle.conformal.alpha
            )
            lower, upper = predict_quantile_intervals(
                quantiles[:, 0], quantiles[:, 1], bundle.conformal
            )
            lo, hi = float(lower[0]), float(upper[0])
        else:
            lo, hi = predict_interval(pred, bundle.conformal)

        v0 = float(component_row.iloc[0][col0])
        v24 = float(component_row.iloc[0][col24])
        z24 = np.nan
        if ref is not None and "Lot_ID" in component_row.columns and "Lot_ID" in ref.columns:
            lot_ref = ref[ref["Lot_ID"] == component_row.iloc[0]["Lot_ID"]]
            reference = lot_ref[col24] if len(lot_ref) >= MIN_LOT_SIZE_FOR_STATS else ref[col24]
            if len(reference):
                z24 = (v24 - float(reference.mean())) / max(float(reference.std(ddof=0)), 1e-9)
        decision = evaluate_safety(v0, v24, pred, lo, hi, cfg, lot_z_24h=z24)

        implausible = bool(physics_plausibility_flags(
            np.array([v0]), np.array([v24]), np.array([pred]), cfg
        )[0])

        contributions = (
            explain_prediction(
                bundle.best_model_name,
                bundle.pipeline,
                X_row,
                reference_features=getattr(bundle, "explanation_reference", None),
            )
            if include_explanations else None
        )

        results[pname] = {
            "value_0h": v0,
            "value_24h": v24,
            "lot_z_24h": z24,
            "prediction": pred,
            "interval": (lo, hi),
            "decision": decision,
            "physically_implausible": implausible,
            "contributions": contributions,
            "model_used": bundle.best_model_name,
        }

    return results


def save_system(system: TrainedSystem, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(system, f)


def load_system(path: str) -> TrainedSystem:
    with open(path, "rb") as f:
        return pickle.load(f)
