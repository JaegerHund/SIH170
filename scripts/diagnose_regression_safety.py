"""Compare early-value baselines, target formulations, and safety outcomes."""

import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ess_predictor.config import PARAMETERS, N_GROUP_FOLDS, CONFORMAL_ALPHA
from ess_predictor.data.synthetic_data import generate_synthetic_dataset
from ess_predictor.features.engineering import build_feature_matrix
from ess_predictor.models.cross_validation import (
    choose_splitter,
    group_cross_validate,
    split_conformal_oof,
)
from ess_predictor.models.zoo import build_model_zoo
from ess_predictor.safety.decision import evaluate_safety, safety_confusion_metrics


FORMULATIONS = ("absolute", "absolute_drift", "relative_drift")


class EarlyValueBaseline(BaseEstimator, RegressorMixin):
    """Persistence or straight-line extrapolation expressed in target space."""

    def __init__(self, parameter, formulation="absolute", mode="persistence"):
        self.parameter = parameter
        self.formulation = formulation
        self.mode = mode

    def fit(self, X, y):
        return self

    def predict(self, X):
        x0 = np.asarray(X[f"{self.parameter}_0h"], dtype=float)
        x24 = np.asarray(X[f"{self.parameter}_24h"], dtype=float)
        if self.mode == "persistence":
            value168 = x24
        else:
            value168 = x24 + 6.0 * (x24 - x0)
        if self.formulation == "absolute":
            return value168
        drift = value168 - x0
        if self.formulation == "absolute_drift":
            return drift
        return drift / np.maximum(np.abs(x0), 1e-9)


def _target(formulation, y168, x0):
    if formulation == "absolute":
        return np.asarray(y168, dtype=float)
    drift = np.asarray(y168, dtype=float) - np.asarray(x0, dtype=float)
    if formulation == "absolute_drift":
        return drift
    return drift / np.maximum(np.abs(x0), 1e-9)


def _as_168h(formulation, target_prediction, x0):
    if formulation == "absolute":
        return np.asarray(target_prediction, dtype=float)
    if formulation == "absolute_drift":
        return np.asarray(x0, dtype=float) + np.asarray(target_prediction, dtype=float)
    return np.asarray(x0, dtype=float) + np.asarray(target_prediction, dtype=float) * np.abs(x0)


def _fold_metrics(y_true, y_pred, groups, n_folds):
    splitter, _ = choose_splitter(pd.Series(groups).nunique(), n_folds)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = pd.Series(groups).reset_index(drop=True)
    mae, rmse, r2, nmae, nrmse = [], [], [], [], []
    for train, test in splitter.split(np.zeros((len(y_true), 1)), y_true, groups=groups):
        residual = y_pred[test] - y_true[test]
        scale = max(float(np.median(np.abs(y_true[train]))), 1e-9)
        mae.append(float(np.mean(np.abs(residual))))
        rmse.append(float(np.sqrt(np.mean(residual**2))))
        nmae.append(mae[-1] / scale)
        nrmse.append(rmse[-1] / scale)
        if len(test) > 1:
            denom = float(np.sum((y_true[test] - np.mean(y_true[test])) ** 2))
            r2.append(1.0 - float(np.sum(residual**2)) / denom if denom > 0 else np.nan)
    return {
        "mae": float(np.mean(mae)),
        "rmse": float(np.mean(rmse)),
        "r2": float(np.nanmean(r2)) if r2 else np.nan,
        "nmae": float(np.mean(nmae)),
        "nrmse": float(np.mean(nrmse)),
        "mae_pct_typical": float(np.mean(nmae) * 100),
    }


def _decisions(v0, predictions, low, high, cfg, lot_z=None):
    if lot_z is None:
        lot_z = np.full(len(v0), np.nan)
    return [
        evaluate_safety(a, a, p, lo, hi, cfg, lot_z_24h=z).decision
        for a, p, lo, hi, z in zip(v0, predictions, low, high, lot_z)
    ]


def run_diagnostics(df, n_folds=N_GROUP_FOLDS):
    regression_rows = []
    safety_rows = []
    behavior_rows = []
    residual_rows = []
    prediction_rows = []
    threshold_rows = []
    lot_rows = []

    for cfg in PARAMETERS:
        pname = cfg.name
        X, y168, groups = build_feature_matrix(df, pname)
        if groups is None:
            raise ValueError("Component_ID is required for grouped validation")
        forbidden = [c for c in X.columns if c.endswith(("96h", "168h"))]
        if forbidden:
            raise AssertionError(f"Future measurements leaked into features: {forbidden}")

        x0 = X[f"{pname}_0h"].to_numpy(dtype=float)
        x24 = X[f"{pname}_24h"].to_numpy(dtype=float)
        candidates = {}

        for formulation in FORMULATIONS:
            target = _target(formulation, y168.to_numpy(), x0)
            zoo = build_model_zoo()
            zoo.update({
                "Baseline_Value24": EarlyValueBaseline(pname, formulation, "persistence"),
                "Baseline_LinearExtrapolation": EarlyValueBaseline(pname, formulation, "linear"),
            })
            for model_name, estimator in zoo.items():
                cv = group_cross_validate(
                    estimator, X, pd.Series(target), groups, model_name, n_folds=n_folds
                )
                pred168 = _as_168h(formulation, cv.oof_predictions, x0)
                metrics168 = _fold_metrics(y168.to_numpy(), pred168, groups, n_folds)
                row = {
                    "parameter": pname,
                    "target_formulation": formulation,
                    "model": model_name,
                    "target_mae": cv.mae,
                    "target_rmse": cv.rmse,
                    "target_r2": cv.r2,
                    "target_nmae": cv.nmae,
                    "target_nrmse": cv.nrmse,
                    "target_mae_pct_median_abs_train_target": cv.mae_pct_typical,
                    "predicted_168h_mae": metrics168["mae"],
                    "predicted_168h_rmse": metrics168["rmse"],
                    "predicted_168h_r2": metrics168["r2"],
                    "predicted_168h_nmae": metrics168["nmae"],
                    "predicted_168h_nrmse": metrics168["nrmse"],
                }
                regression_rows.append(row)
                candidates[(formulation, model_name)] = {
                    "estimator": estimator,
                    "cv": cv,
                    "target": target,
                    "pred168": pred168,
                    "metrics168": metrics168,
                }

            # Best candidate for each target representation by mean fold MAE
            # after conversion back to the engineering-unit 168h prediction.
            form_items = [(key, value) for key, value in candidates.items() if key[0] == formulation]
            selected_key, selected = min(
                form_items, key=lambda item: item[1]["metrics168"]["mae"]
            )
            lot_ids = (
                df.drop_duplicates("Component_ID").set_index("Component_ID")
                .reindex(groups.to_numpy())["Lot_ID"].reset_index(drop=True)
                if "Lot_ID" in df.columns else None
            )
            target_pred, target_low, target_high, safety_lot_z = split_conformal_oof(
                selected["estimator"], X, pd.Series(selected["target"]), groups,
                selected_key[1], n_folds=n_folds, alpha=CONFORMAL_ALPHA,
                lot_ids=lot_ids, early_24=x24 if lot_ids is not None else None,
            )
            pred168 = _as_168h(formulation, target_pred, x0)
            low168 = _as_168h(formulation, target_low, x0)
            high168 = _as_168h(formulation, target_high, x0)
            decisions = _decisions(x0, pred168, low168, high168, cfg, safety_lot_z)
            metrics = safety_confusion_metrics(y168.to_numpy(), x0, decisions, cfg)
            point_decisions = _decisions(
                x0, selected["pred168"], selected["pred168"], selected["pred168"],
                cfg, safety_lot_z,
            )
            point_metrics = safety_confusion_metrics(y168.to_numpy(), x0, point_decisions, cfg)
            safety_rows.append({
                "parameter": pname,
                "target_formulation": formulation,
                "selected_model": selected_key[1],
                "selection_basis": "lowest mean fold 168h MAE among candidates",
                "mean_interval_width_168h": float(np.mean(high168 - low168)),
                "median_interval_width_168h": float(np.median(high168 - low168)),
                "point_only_fpr": point_metrics["false_positive_rate"],
                "point_only_fnr": point_metrics["false_negative_rate"],
                "point_only_precision": point_metrics["precision"],
                "point_only_recall": point_metrics["recall_sensitivity"],
                "point_only_review_count": point_metrics["review_count"],
                **metrics,
            })

            if formulation == "absolute":
                behavior = groups.map(df.set_index("Component_ID")["True_Behavior"]).to_numpy()
                residual = selected["pred168"] - y168.to_numpy()
                rel_early = (x24 - x0) / np.maximum(np.abs(x0), 1e-9)
                rel_actual = (y168.to_numpy() - x0) / np.maximum(np.abs(x0), 1e-9)
                residual_rows.append({
                    "parameter": pname,
                    "model": selected_key[1],
                    "n": int(len(residual)),
                    "residual_mean_bias": float(np.mean(residual)),
                    "residual_std": float(np.std(residual, ddof=1)),
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "p90_absolute_error": float(np.quantile(np.abs(residual), 0.90)),
                    "r2_oof": float(selected["metrics168"]["r2"]),
                })
                id_to_lot = df.drop_duplicates("Component_ID").set_index("Component_ID")["Lot_ID"] if "Lot_ID" in df.columns else None
                for i, component_id in enumerate(groups.to_numpy()):
                    prediction_rows.append({
                        "Component_ID": component_id,
                        "Lot_ID": id_to_lot.get(component_id, None) if id_to_lot is not None else None,
                        "parameter": pname,
                        "model": selected_key[1],
                        "target_formulation": formulation,
                        "True_Behavior": behavior[i],
                        "value_0h": x0[i],
                        "value_24h": x24[i],
                        "actual_168h": y168.iloc[i],
                        "predicted_168h": selected["pred168"][i],
                        "residual": residual[i],
                        "absolute_error": abs(residual[i]),
                        "relative_residual": residual[i] / max(abs(float(y168.iloc[i])), 1e-9),
                        "actual_relative_drift_24h": rel_early[i],
                        "actual_relative_drift_168h": rel_actual[i],
                    })
                for label in sorted(set(behavior)):
                    mask = behavior == label
                    behavior_rows.append({
                        "parameter": pname,
                        "target_formulation": formulation,
                        "model": selected_key[1],
                        "True_Behavior": label,
                        "n": int(mask.sum()),
                        "residual_mean": float(np.mean(residual[mask])),
                        "residual_std": float(np.std(residual[mask], ddof=1)) if mask.sum() > 1 else np.nan,
                        "mae": float(np.mean(np.abs(residual[mask]))),
                        "mean_actual_relative_drift_24h": float(np.mean(rel_early[mask])),
                        "mean_actual_relative_drift_168h": float(np.mean(rel_actual[mask])),
                    })

                # Threshold sensitivity is descriptive only; engineering
                # thresholds are not changed or selected from these rows.
                for scale in (0.5, 0.75, 1.0, 1.25, 1.5):
                    varied = replace(cfg, relative_drift_threshold=cfg.relative_drift_threshold * scale)
                    varied_decisions = _decisions(x0, pred168, low168, high168, varied, safety_lot_z)
                    sens = safety_confusion_metrics(
                        y168.to_numpy(), x0, varied_decisions, varied, truth_param_cfg=cfg
                    )
                    threshold_rows.append({
                        "parameter": pname,
                        "model": selected_key[1],
                        "threshold_scale": scale,
                        **sens,
                    })

                lot_groups = lot_ids
                if lot_groups is not None and lot_groups.nunique() >= 2:
                    lot_cv = group_cross_validate(
                        selected["estimator"], X, pd.Series(selected["target"]),
                        lot_groups, selected_key[1], n_folds=min(n_folds, lot_groups.nunique()),
                    )
                    lot_rows.append({
                        "parameter": pname,
                        "target_formulation": formulation,
                        "model": selected_key[1],
                        "groups": "Lot_ID",
                        "n_lots": int(lot_groups.nunique()),
                        "target_mae": lot_cv.mae,
                        "target_rmse": lot_cv.rmse,
                        "target_r2": lot_cv.r2,
                        "target_nmae": lot_cv.nmae,
                    })

    return tuple(pd.DataFrame(rows) for rows in (
        regression_rows, safety_rows, behavior_rows, residual_rows,
        prediction_rows, threshold_rows, lot_rows
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-components", type=int, default=350)
    parser.add_argument("--n-lots", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--folds", type=int, default=N_GROUP_FOLDS)
    parser.add_argument("--output-dir", default=os.path.join(_ROOT, "outputs", "regression_safety_diagnostics"))
    args = parser.parse_args()

    df = generate_synthetic_dataset(args.n_components, args.n_lots, seed=args.seed)
    reports = run_diagnostics(df, n_folds=args.folds)
    names = (
        "regression_comparison.csv",
        "target_safety_metrics.csv",
        "behavior_error_analysis.csv",
        "residual_summary.csv",
        "selected_oof_predictions.csv",
        "threshold_sensitivity.csv",
        "lot_generalization.csv",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    for report, name in zip(reports, names):
        report.to_csv(os.path.join(args.output_dir, name), index=False)

    regression, safety, behavior, residuals, predictions, threshold, lot = reports
    print("Regression comparison (target and 168h engineering metrics):")
    print(regression.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSafety metrics for best 168h-MAE candidate per target formulation:")
    safety_cols = ["parameter", "target_formulation", "selected_model", "confusion_matrix",
                   "precision", "recall_sensitivity", "specificity", "false_positive_rate",
                   "false_negative_rate", "point_only_fpr", "point_only_fnr",
                   "mean_interval_width_168h", "reject_false_positive_rate",
                   "reject_recall", "review_count", "reject_count",
                   "relative_drift_threshold", "absolute_safety_limit"]
    print(safety[safety_cols].to_string(index=False))
    print("\nBehavior error analysis for the absolute-target winner:")
    print(behavior.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nOverall residual summary for the absolute-target winner:")
    print(residuals.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nThreshold sensitivity (descriptive; configured thresholds unchanged):")
    print(threshold.to_string(index=False))
    print("\nLot-held-out generalization check:")
    print(lot.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nCSV reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
