"""
run_demo.py
-----------
End-to-end demonstration script for the ess_predictor package.

Run from the project root:
    python scripts/run_demo.py

This works whether or not the package has been `pip install -e`'d: it
adds ../src to sys.path automatically if the package isn't already
importable, so you don't need to fight with PYTHONPATH.
"""

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
from ess_predictor.models.cross_validation import compare_models, select_best_model
from ess_predictor.uncertainty.conformal import predict_interval
from ess_predictor.safety.decision import evaluate_safety, physics_plausibility_flags
from ess_predictor.explainability.explain import global_feature_importance
from ess_predictor.robustness.noise_testing import batch_robustness_test
from ess_predictor.reporting.report import build_parameter_block, format_component_report
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

    # 1. Generate synthetic data
    df = generate_synthetic_dataset(n_components=350, n_lots=7, seed=7)
    df.to_csv(os.path.join(OUT_DIR, "synthetic_dataset.csv"), index=False)
    print(f"Generated synthetic dataset: {df.shape[0]} components, "
          f"{df['Lot_ID'].nunique()} lots.\n")

    # 2-5. Train full system (validation + features + model comparison +
    #      selection + conformal calibration + safety metrics all happen
    #      inside train_all_parameters)
    system = train_all_parameters(df, n_folds=5, verbose=True)

    # 6. Robustness / noise sensitivity testing (spec section 11)
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

    # 7. Visualizations
    print("Generating visualizations...")
    for pname, bundle in system.bundles.items():
        cfg = bundle.param_cfg
        best_cv = bundle.cv_results[bundle.best_model_name]
        mask = ~np.isnan(best_cv.oof_predictions)
        y_true = best_cv.oof_true[mask]
        y_pred = best_cv.oof_predictions[mask]

        X, y, groups = build_feature_matrix(df, pname)
        value_0h = X[f"{pname}_0h"].values[mask]

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
            df, y_true, y_pred, decisions, pname, cfg.unit, PLOTS_DIR,
            cfg.relative_drift_threshold, value_0h
        )
        print(f"  {pname}: plots saved (false negatives in OOF eval: {n_fn})")
    print()

    # 8. Component-level screening reports
    print("Generating component screening reports...")
    example_ids = []

    # Pick one 'normal', one 'latent_defect', one 'static_limit_escape'
    # component from the synthetic labels purely to make the demo report
    # illustrative -- True_Behavior is NEVER used as a model feature.
    for behavior in ["normal", "latent_defect", "static_limit_escape", "high_initial_stable"]:
        matches = df[df["True_Behavior"] == behavior]
        if len(matches):
            example_ids.append(matches.iloc[0]["Component_ID"])

    report_texts = []
    for comp_id in example_ids:
        row = df[df["Component_ID"] == comp_id]
        results = predict_component(system, row, lot_reference_df=df)

        blocks = {}
        for pname, r in results.items():
            block = build_parameter_block(
                param_cfg=r["decision"].__dict__.get("param_cfg", system.bundles[pname].param_cfg)
                if False else system.bundles[pname].param_cfg,
                value_0h=r["value_0h"],
                value_24h=r["value_24h"],
                lot_z_24h=r["lot_z_24h"],
                decision=r["decision"],
                contributions=r["contributions"],
                physically_implausible=r["physically_implausible"],
            )
            blocks[pname] = block

        text = format_component_report(
            comp_id, row.iloc[0]["Lot_ID"], row.iloc[0]["Temperature"], blocks
        )
        report_texts.append(text)
        print(text)
        print()

    with open(os.path.join(OUT_DIR, "example_component_reports.txt"), "w") as f:
        f.write("\n\n".join(report_texts))

    # 9. Save / reload demonstration
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
