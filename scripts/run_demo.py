"""Generate data, train the drift models, and write demo reports."""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd

from ess_predictor.data.synthetic_data import generate_synthetic_dataset
from ess_predictor.pipeline import train_all_parameters, predict_component, save_system, load_system
from ess_predictor.features.engineering import build_feature_matrix
from ess_predictor.uncertainty.conformal import (
    QuantileConformalCalibration,
    predict_interval,
    predict_quantile_intervals,
)
from ess_predictor.safety.decision import (
    evaluate_safety,
    overall_component_decision,
    physics_plausibility_flags,
)
from ess_predictor.explainability.explain import global_feature_importance
from ess_predictor.robustness.noise_testing import batch_robustness_test
import ess_predictor.visualization.plots as viz

_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
PLOTS_DIR = os.path.join(_PROJECT_ROOT, "plots")
OUT_DIR = os.path.join(_PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models_store")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    print("#" * 70)
    print("# Module B: Time-Series Drift Predictor -- End-to-End Demo")
    print("# NOTE: Uses SYNTHETIC data for development/testing only.")
    print("#" * 70)
    print()

    # Generate synthetic data.
    df = generate_synthetic_dataset(n_components=350, n_lots=7, seed=7)
    df.to_csv(os.path.join(OUT_DIR, "synthetic_dataset.csv"), index=False)
    print(f"Generated synthetic dataset: {df.shape[0]} components, "
          f"{df['Lot_ID'].nunique()} lots.\n")

    # Validate the data, compare models, and calibrate prediction intervals.
    system = train_all_parameters(df, n_folds=5, verbose=True)

    # Save the per-lot validation scores.
    lot_validation_rows = []
    for pname, bundle in system.bundles.items():
        for model_name, cv in bundle.lot_cv_results.items():
            for lot_id, n_components, mae, rmse in zip(
                cv.fold_group_labels,
                cv.fold_test_sizes,
                cv.fold_mae,
                cv.fold_rmse,
            ):
                lot_validation_rows.append({
                    "Parameter": pname,
                    "Model": model_name,
                    "Held_Out_Lot": lot_id,
                    "Components": n_components,
                    "MAE": mae,
                    "RMSE": rmse,
                    "Selected_By_LOLO_MAE": model_name == bundle.best_model_name,
                })
    if lot_validation_rows:
        lot_validation_path = os.path.join(OUT_DIR, "leave_one_lot_out_mae.csv")
        pd.DataFrame(lot_validation_rows).to_csv(lot_validation_path, index=False)
        print(f"Per-lot held-out MAE table saved to {lot_validation_path}\n")

    # Save actual/predicted pairs for both validation sets.
    forecast_rows = []
    component_table = df.drop_duplicates("Component_ID").set_index("Component_ID")
    for pname, bundle in system.bundles.items():
        X_param, y_param, component_groups = build_feature_matrix(df, pname)
        selected_cv = bundle.lot_cv_results.get(bundle.best_model_name)
        prediction_basis = "LOLO CV (used for model selection)"
        if selected_cv is None:
            selected_cv = bundle.cv_results[bundle.best_model_name]
            prediction_basis = "Component-grouped CV (used for model selection)"

        for position, actual, predicted in zip(
            selected_cv.oof_index,
            selected_cv.oof_true,
            selected_cv.oof_predictions,
        ):
            component_id = component_groups.iloc[int(position)]
            raw = component_table.loc[component_id]
            value_0h = float(raw[f"{pname}_0h"])
            value_24h = float(raw[f"{pname}_24h"])
            prediction_error = float(predicted - actual)
            forecast_rows.append({
                "Component_ID": component_id,
                "Lot_ID": raw["Lot_ID"],
                "Parameter": pname,
                "Model": bundle.best_model_name,
                "Prediction_Basis": prediction_basis,
                "Value_0h": value_0h,
                "Value_24h": value_24h,
                "Actual_168h": float(actual),
                "Predicted_168h": float(predicted),
                "Baseline_LastValue24h": value_24h,
                "Baseline_LinearExtrapolation": value_24h + 6.0 * (value_24h - value_0h),
                "Prediction_Error": prediction_error,
                "Absolute_Error": abs(prediction_error),
                "Squared_Error": prediction_error**2,
            })

        if not bundle.calibration_predictions.empty:
            for record in bundle.calibration_predictions.to_dict("records"):
                component_id = record["Component_ID"]
                raw = component_table.loc[component_id]
                value_0h = float(raw[f"{pname}_0h"])
                value_24h = float(raw[f"{pname}_24h"])
                prediction_error = float(record["Predicted_168h"] - record["Actual_168h"])
                forecast_rows.append({
                    **record,
                    "Value_0h": value_0h,
                    "Value_24h": value_24h,
                    "Baseline_LastValue24h": value_24h,
                    "Baseline_LinearExtrapolation": value_24h + 6.0 * (value_24h - value_0h),
                    "Prediction_Error": prediction_error,
                    "Absolute_Error": abs(prediction_error),
                    "Squared_Error": prediction_error**2,
                })

    forecast_path = os.path.join(OUT_DIR, "forecast_predictions.csv")
    pd.DataFrame(forecast_rows).sort_values(
        ["Parameter", "Component_ID"]
    ).to_csv(forecast_path, index=False)
    print(f"Component-level actual vs predicted values saved to {forecast_path}\n")

    # Check sensitivity to measurement noise.
    print("Running measurement-noise robustness tests...")
    for pname, bundle in system.bundles.items():
        def predict_fn(X, _bundle=bundle):
            return _bundle.pipeline.predict(X[_bundle.feature_columns])

        robustness_df = batch_robustness_test(df, pname, predict_fn, n_samples=15)
        n_unstable = (~robustness_df["stable"]).sum()
        robustness_df.to_csv(os.path.join(OUT_DIR, f"robustness_{pname}.csv"), index=False)
        print(f"  {pname}: {n_unstable}/{len(robustness_df)} components showed "
              f">5% prediction swing under +/-0.5% input noise "
              f"(mean relative swing={robustness_df['relative_swing'].mean():.4f})")
    print()

    # Save plots.
    print("Generating visualizations...")
    for pname, bundle in system.bundles.items():
        cfg = bundle.param_cfg
        best_cv = bundle.cv_results[bundle.best_model_name]
        mask = ~np.isnan(best_cv.oof_predictions)
        y_true = best_cv.oof_true[mask]
        y_pred = best_cv.oof_predictions[mask]
        cv_positions = best_cv.oof_index[mask]

        X, y, groups = build_feature_matrix(df, pname)
        value_0h = X[f"{pname}_0h"].values[cv_positions]

        if isinstance(bundle.conformal, QuantileConformalCalibration):
            lower = best_cv.oof_quantile_lower[mask]
            upper = best_cv.oof_quantile_upper[mask]
            lo, hi = predict_quantile_intervals(lower, upper, bundle.conformal)
        else:
            lo, hi = predict_interval(y_pred, bundle.conformal)

        decisions = []
        for v0, pv, l, h in zip(value_0h, y_pred, lo, hi):
            d = evaluate_safety(v0, v0, pv, l, h, cfg)
            decisions.append(d.decision)

        viz.plot_actual_vs_predicted(y_true, y_pred, pname, cfg.unit, PLOTS_DIR)
        viz.plot_residuals(y_true, y_pred, pname, cfg.unit, PLOTS_DIR)
        viz.plot_error_distribution(y_true, y_pred, pname, cfg.unit, PLOTS_DIR)
        viz.plot_degradation_trajectories(df, pname, PLOTS_DIR)
        viz.plot_prediction_intervals(y_true, y_pred, lo, hi, pname, cfg.unit, PLOTS_DIR)
        viz.plot_decision_distribution(decisions, pname, PLOTS_DIR)
        viz.plot_model_comparison(bundle.cv_results, pname, PLOTS_DIR)

        imp_df = global_feature_importance(bundle.best_model_name, bundle.pipeline, X)
        if len(imp_df):
            viz.plot_feature_importance(imp_df, pname, PLOTS_DIR)

        _, n_fn = viz.plot_false_negative_examples(
            df.set_index("Component_ID").loc[groups.iloc[cv_positions].to_numpy()].reset_index(),
            y_true, y_pred, decisions, pname, cfg.unit, PLOTS_DIR,
            cfg.relative_drift_threshold, value_0h
        )
        print(f"  {pname}: plots saved (false negatives in OOF eval: {n_fn})")
    print()

    # Score each component and build the disposition lists.
    print("Building complete SAFE / REVIEW / REJECT component lists...")
    decision_counts = {"SAFE": 0, "REVIEW": 0, "REJECT": 0}
    disposition_lists = {"SAFE": [], "REVIEW": [], "REJECT": []}
    detail_rows = []
    scored_components = 0
    for row_index in range(len(df)):
        component_row = df.iloc[[row_index]]
        component_results = predict_component(
            system,
            component_row,
            lot_reference_df=df,
            include_explanations=False,
        )
        if not component_results:
            continue
        overall = overall_component_decision({
            pname: result["decision"]
            for pname, result in component_results.items()
        })
        decision_counts[overall] += 1
        scored_components += 1

        component_id = str(component_row.iloc[0]["Component_ID"])
        lot_id = str(component_row.iloc[0]["Lot_ID"])
        issues = []
        for pname, result in component_results.items():
            cfg = system.bundles[pname].param_cfg
            decision = result["decision"]
            risk_direction = 1.0 if cfg.degrades_upward else -1.0
            risk_drift_pct = risk_direction * decision.predicted_relative_drift * 100.0
            if decision.decision != "SAFE":
                issue = (
                    f"{pname}: original 0h={result['value_0h']:.4g} {cfg.unit}; "
                    f"24h={result['value_24h']:.4g} {cfg.unit}; "
                    f"predicted 168h={decision.predicted_value:.4g} "
                    f"[{decision.predicted_low:.4g}, {decision.predicted_high:.4g}] {cfg.unit}; "
                    f"predicted drift={risk_drift_pct:+.1f}% vs "
                    f"{cfg.relative_drift_threshold * 100:.1f}% limit. "
                    f"Reason: {'; '.join(decision.reasons)}"
                )
                issues.append(issue)

            detail_rows.append({
                "Component_ID": component_id,
                "Lot_ID": lot_id,
                "Overall_Decision": overall,
                "Parameter": pname,
                "Parameter_Decision": decision.decision,
                "Value_0h": result["value_0h"],
                "Value_24h": result["value_24h"],
                "Predicted_168h": decision.predicted_value,
                "Interval_Low": decision.predicted_low,
                "Interval_High": decision.predicted_high,
                "Predicted_Drift_Pct": risk_drift_pct,
                "Drift_Limit_Pct": cfg.relative_drift_threshold * 100.0,
                "Reasons": "; ".join(decision.reasons),
            })

        disposition_lists[overall].append({
            "Component_ID": component_id,
            "Lot_ID": lot_id,
            "Issues": issues,
        })

    def count_line(label, key):
        count = decision_counts[key]
        percent = 100.0 * count / scored_components if scored_components else 0.0
        return f"{label:<24} {count:>4} ({percent:5.1f}%)"

    summary_lines = [
        "END-OF-RUN COMPONENT SCREENING SUMMARY",
        "=" * 48,
        f"Components scored: {scored_components} / {len(df)}",
        count_line("PASS (SAFE):", "SAFE"),
        count_line("UNDER REVIEW:", "REVIEW"),
        count_line("FAIL (REJECT):", "REJECT"),
        "",
        "Overall result uses the most cautious parameter decision per component:",
        "REJECT takes precedence over REVIEW, which takes precedence over SAFE.",
        "These are model-assigned synthetic-demo outcomes, not real-device rates.",
        "=" * 48,
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text + "\n")
    with open(os.path.join(OUT_DIR, "screening_summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    disposition_lines = [
        "COMPONENT DISPOSITION LISTS",
        "=" * 72,
        "Flagged entries show the original 0h value, 24h reading, predicted 168h value/range, drift limit, and reason.",
        "All values and decisions below come from the synthetic demo run.",
    ]
    for status, heading in (
        ("REVIEW", "UNDER REVIEW"),
        ("REJECT", "REJECTED"),
        ("SAFE", "SAFE / PASS"),
    ):
        entries = disposition_lists[status]
        disposition_lines.extend([
            "",
            f"{heading} — {len(entries)} component(s)",
            "-" * 72,
        ])
        if not entries:
            disposition_lines.append("None")
            continue
        for entry in entries:
            disposition_lines.append(f"{entry['Component_ID']} — Lot {entry['Lot_ID']}")
            if entry["Issues"]:
                disposition_lines.extend(f"  - {issue}" for issue in entry["Issues"])
    disposition_lines.extend(["", summary_text])
    disposition_text = "\n".join(disposition_lines)
    disposition_path = os.path.join(OUT_DIR, "component_disposition_report.txt")
    with open(disposition_path, "w", encoding="utf-8") as f:
        f.write(disposition_text + "\n")
    pd.DataFrame(detail_rows).to_csv(
        os.path.join(OUT_DIR, "component_dispositions.csv"), index=False
    )
    # Point the old sample-report file to the full component lists.
    with open(os.path.join(OUT_DIR, "example_component_reports.txt"), "w", encoding="utf-8") as f:
        f.write("Individual example reports were replaced by the complete categorized lists in component_disposition_report.txt.\n")
    print(f"Complete component lists saved to {disposition_path}")

    # Save the trained system and check that it reloads.
    model_path = os.path.join(MODELS_DIR, "trained_system.pkl")
    save_system(system, model_path)
    reloaded = load_system(model_path)
    print(f"Saved trained system to {model_path} and verified reload "
          f"({len(reloaded.bundles)} parameter models).")

    print()
    print("Demo complete. See ./plots for figures, ./outputs for reports/CSVs, "
          "./models_store for the saved model.")


if __name__ == "__main__":
    main()
