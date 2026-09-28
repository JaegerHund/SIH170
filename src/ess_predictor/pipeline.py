"""
pipeline.py
-----------
Orchestrates the full Module B pipeline (spec section 14):

RAW DATA -> validation -> feature engineering -> lot normalization
  -> early drift calc -> regression model -> predicted 168h
  -> uncertainty estimation -> safety-threshold/slope comparison
  -> SAFE/REVIEW/REJECT -> explainability report

Trains one INDEPENDENT model per parameter (spec section 15, Approach A
chosen as the initial implementation for interpretability/debuggability).
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ess_predictor.config import PARAMETERS, ParameterConfig, N_GROUP_FOLDS, CONFORMAL_ALPHA
from ess_predictor.data.validation import validate_dataset, handle_missing_values
from ess_predictor.features.engineering import build_feature_matrix, build_features_for_parameter
from ess_predictor.models.zoo import build_model_zoo
from ess_predictor.models.cross_validation import compare_models, select_best_model, CVResult
from ess_predictor.uncertainty.conformal import fit_conformal, predict_interval, ConformalCalibration, low_confidence_flag
from ess_predictor.safety.decision import evaluate_safety, physics_plausibility_flags, safety_confusion_metrics
from ess_predictor.explainability.explain import explain_prediction, global_feature_importance


@dataclass
class ParameterModelBundle:
    """Everything needed to score a new component for one parameter."""
    param_cfg: ParameterConfig
    best_model_name: str
    pipeline: object                 # fitted sklearn Pipeline
    cv_results: Dict[str, CVResult]
    conformal: ConformalCalibration
    feature_columns: list
    safety_metrics: dict


@dataclass
class TrainedSystem:
    bundles: Dict[str, ParameterModelBundle] = field(default_factory=dict)
    lot_reference: Optional[pd.DataFrame] = None   # lot mean/std reference for inference-time z-scores


def train_all_parameters(df: pd.DataFrame, n_folds: int = N_GROUP_FOLDS,
                          verbose: bool = True) -> TrainedSystem:
    """
    Run the full training pipeline for every usable parameter in the
    dataset: validate -> build features -> compare models -> select best
    -> calibrate conformal intervals -> compute safety metrics.
    """
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

        zoo = build_model_zoo()
        cv_results = compare_models(zoo, X, y, groups, n_folds=n_folds)
        best_name = select_best_model(cv_results)

        if verbose:
            print(f"[{p.name}] Model comparison (CV MAE {p.unit}):")
            for name, r in cv_results.items():
                marker = "  <== selected" if name == best_name else ""
                print(f"    {name:28s} MAE={r.mae:.4f}  RMSE={r.rmse:.4f}  R2={r.r2:.4f}{marker}")

        best_cv = cv_results[best_name]
        conformal = fit_conformal(best_cv.oof_true, best_cv.oof_predictions, alpha=CONFORMAL_ALPHA)

        # Refit the best model on the FULL dataset for deployment use,
        # while safety metrics / intervals are calibrated from the
        # honest out-of-fold predictions above (never in-sample).
        best_pipeline = zoo[best_name]
        best_pipeline.fit(X, y)

        value_0h = X[f"{p.name}_0h"].values
        oof_decisions = []
        for true_val, pred_val in zip(best_cv.oof_true, best_cv.oof_predictions):
            if np.isnan(pred_val):
                oof_decisions.append("REVIEW")
                continue
            lo, hi = predict_interval(pred_val, conformal)
            d = evaluate_safety(
                value_0h=0.0,  # placeholder, replaced per-row below
                value_24h=0.0,
                pred_168h=pred_val,
                pred_low=lo,
                pred_high=hi,
                param_cfg=p,
            )
            oof_decisions.append(d.decision)

        # Recompute decisions properly using each row's actual value_0h
        oof_decisions = []
        for v0, pred_val in zip(value_0h, best_cv.oof_predictions):
            if np.isnan(pred_val):
                oof_decisions.append("REVIEW")
                continue
            lo, hi = predict_interval(pred_val, conformal)
            d = evaluate_safety(v0, v0, pred_val, lo, hi, p)
            oof_decisions.append(d.decision)

        safety_metrics = safety_confusion_metrics(
            y_true_168h=best_cv.oof_true,
            value_0h=value_0h,
            decisions=oof_decisions,
            param_cfg=p,
        )

        if verbose:
            cm = safety_metrics["confusion_matrix"]
            print(f"    Safety metrics: Recall={safety_metrics['recall_sensitivity']:.3f}  "
                  f"Precision={safety_metrics['precision']:.3f}  "
                  f"FNR={safety_metrics['false_negative_rate']:.3f}  "
                  f"FPR={safety_metrics['false_positive_rate']:.3f}  "
                  f"CM={cm}")
            print()

        system.bundles[p.name] = ParameterModelBundle(
            param_cfg=p,
            best_model_name=best_name,
            pipeline=best_pipeline,
            cv_results=cv_results,
            conformal=conformal,
            feature_columns=list(X.columns),
            safety_metrics=safety_metrics,
        )

    # Store a lot-level reference table (mean/std per parameter per lot)
    # so inference on brand-new components can compute consistent z-scores.
    if "Lot_ID" in df.columns:
        system.lot_reference = df[["Lot_ID"] + [
            c for p in system.bundles for c in [f"{p}_0h", f"{p}_24h"] if c in df.columns
        ]].copy()

    return system


def predict_component(system: TrainedSystem, component_row: pd.DataFrame,
                       lot_reference_df: Optional[pd.DataFrame] = None) -> Dict[str, dict]:
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
    results = {}

    for pname, bundle in system.bundles.items():
        cfg = bundle.param_cfg
        col0 = f"{pname}_0h"
        col24 = f"{pname}_24h"
        if col0 not in component_row.columns or col24 not in component_row.columns:
            continue

        # Build features using lot context if available so z-scores are
        # computed against real peers rather than a single-row "lot" of 1.
        if ref is not None and "Lot_ID" in component_row.columns:
            lot_id = component_row.iloc[0]["Lot_ID"]
            context_cols = [c for c in [col0, col24, "Lot_ID"] if c in ref.columns]
            if context_cols and "Lot_ID" in context_cols:
                context = ref[ref["Lot_ID"] == lot_id]
                combined = pd.concat([context, component_row], ignore_index=True, sort=False)
                X_full = build_features_for_parameter(combined, pname)
                X_row = X_full.iloc[[-1]][bundle.feature_columns]
            else:
                X_row = build_features_for_parameter(component_row, pname)[bundle.feature_columns]
        else:
            X_row = build_features_for_parameter(component_row, pname)[bundle.feature_columns]

        pred = float(bundle.pipeline.predict(X_row)[0])
        lo, hi = predict_interval(pred, bundle.conformal)

        v0 = float(component_row.iloc[0][col0])
        v24 = float(component_row.iloc[0][col24])
        z24 = float(X_row.iloc[0].get(f"{pname}_z_24h", np.nan))

        decision = evaluate_safety(v0, v24, pred, lo, hi, cfg, lot_z_24h=z24)

        low_conf = low_confidence_flag(pred, bundle.conformal)
        if low_conf and decision.decision == "SAFE":
            decision.decision = "REVIEW"
            decision.reasons.append(
                "Downgraded from SAFE to REVIEW: prediction interval too wide "
                "relative to the predicted value for a confident automatic decision."
            )

        implausible = bool(physics_plausibility_flags(
            np.array([v0]), np.array([v24]), np.array([pred]), cfg
        )[0])

        contributions = explain_prediction(bundle.best_model_name, bundle.pipeline, X_row)

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
