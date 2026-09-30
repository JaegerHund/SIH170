"""
tests/test_smoke.py
--------------------
A lightweight smoke test (no pytest dependency required) that exercises
the core pipeline on a small synthetic dataset. Run it whenever you
reorganize files or edit imports, to catch breakage immediately:

    python tests/test_smoke.py

It is intentionally NOT a rigorous statistical test suite -- it just
verifies that every stage of the pipeline runs, produces the expected
shapes/types, and that group cross-validation has no leaked components.
"""

import sys
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ess_predictor.data.synthetic_data import generate_synthetic_dataset
from ess_predictor.data.validation import validate_dataset
from ess_predictor.features.engineering import build_feature_matrix
from ess_predictor.pipeline import train_all_parameters, predict_component
from ess_predictor.config import (
    ACTIVE_PARAMETER_NAMES,
    PARAMETER_CATALOG,
    PARAMETERS,
    ParameterConfig,
    SYNTHETIC_DATA_PROFILES,
)
from ess_predictor.safety.decision import evaluate_safety, safety_confusion_metrics
from ess_predictor.uncertainty.conformal import fit_conformal


def test_synthetic_data_shape():
    df = generate_synthetic_dataset(n_components=60, n_lots=3)
    assert len(df) == 60
    assert "Data_Source" in df.columns
    assert (df["Data_Source"] == "SYNTHETIC").all()
    print("OK: synthetic data generation")


def test_synthetic_profiles():
    assert tuple(p.name for p in PARAMETERS) == ACTIVE_PARAMETER_NAMES
    assert ACTIVE_PARAMETER_NAMES == ("Iddq", "Leakage", "PropagationDelay")
    profile_names = tuple(SYNTHETIC_DATA_PROFILES)
    generated = {}
    threshold_exceed_rates = {}
    for profile in profile_names:
        first = generate_synthetic_dataset(
            n_components=1200, n_lots=6, seed=170, profile=profile
        )
        second = generate_synthetic_dataset(
            n_components=1200, n_lots=6, seed=170, profile=profile
        )
        assert first.equals(second), f"{profile} profile is not reproducible"
        assert (first["Synthetic_Profile"] == profile).all()
        assert (first["Data_Source"] == "SYNTHETIC").all()
        assert {"Component_ID", "Lot_ID", "True_Behavior", "Data_Source",
                "Synthetic_Profile"}.issubset(first.columns)
        for cfg in PARAMETERS:
            assert {f"{cfg.name}_{hour}h" for hour in (0, 24, 96, 168)}.issubset(
                first.columns
            )
        inactive_measurements = {
            f"{name}_{hour}h"
            for name in set(PARAMETER_CATALOG) - set(ACTIVE_PARAMETER_NAMES)
            for hour in (0, 24, 96, 168)
        }
        assert not inactive_measurements.intersection(first.columns)
        assert {"abrupt_degradation", "latent_defect", "static_limit_escape"}.issubset(
            set(first["True_Behavior"])
        ), f"{profile} profile lost one of the rare challenge classes"
        generated[profile] = first

        exceedances = []
        for cfg in PARAMETERS:
            value_0h = first[f"{cfg.name}_0h"].to_numpy()
            value_168h = first[f"{cfg.name}_168h"].to_numpy()
            direction = 1.0 if cfg.degrades_upward else -1.0
            relative_drift = (value_168h - value_0h) / np.maximum(np.abs(value_0h), 1e-9)
            exceedances.extend(
                (direction * relative_drift > cfg.relative_drift_threshold).tolist()
            )
        threshold_exceed_rates[profile] = float(np.mean(exceedances))

    default = generate_synthetic_dataset(n_components=1200, n_lots=6, seed=170)
    assert default.equals(generated["balanced"])
    try:
        generate_synthetic_dataset(n_components=1, profile="unknown")
        raise AssertionError("Unknown synthetic profiles must be rejected")
    except ValueError:
        pass

    normal_rates = {
        profile: (frame["True_Behavior"] == "normal").mean()
        for profile, frame in generated.items()
    }
    assert normal_rates["easy_demo"] > normal_rates["balanced"] > normal_rates["stress"]
    assert (
        threshold_exceed_rates["easy_demo"]
        < threshold_exceed_rates["balanced"]
        < threshold_exceed_rates["stress"]
    )
    print("OK: synthetic profiles, labels, schema, and reproducibility")


def test_validation():
    df = generate_synthetic_dataset(n_components=60, n_lots=3)
    report = validate_dataset(df)
    assert len(report.usable_parameters) == len(PARAMETERS)
    print("OK: data validation")


def test_feature_matrix_no_target_leak():
    df = generate_synthetic_dataset(n_components=60, n_lots=3)
    X, y, groups = build_feature_matrix(df, "Iddq")
    assert not any(c.endswith("168h") for c in X.columns), "Target column leaked into features!"
    assert not any(c.endswith("96h") for c in X.columns), "Future 96h measurement leaked into features!"
    assert not any("_z_" in c for c in X.columns), "Full-dataset lot statistics leaked into CV features!"
    assert {"Iddq_dX_0_24", "Iddq_relative_drift_0_24", "Iddq_drift_rate_0_24"}.issubset(X.columns)
    try:
        build_feature_matrix(generate_synthetic_dataset(n_components=10), "Iddq", include_96h=True)
        raise AssertionError("Explicit 96h prediction features must be rejected")
    except ValueError:
        pass
    assert len(X) == len(y) == len(groups)
    print("OK: feature matrix excludes target")


def test_direction_limits_and_conformal_calibration():
    downward = ParameterConfig(
        name="Down", display_name="Downward", unit="u", degrades_upward=False,
        relative_drift_threshold=0.10,
    )
    decision = evaluate_safety(100.0, 99.0, 80.0, 78.0, 82.0, downward)
    assert decision.decision == "REJECT"
    metrics = safety_confusion_metrics(
        np.array([80.0, 100.0]), np.array([100.0, 100.0]),
        ["REJECT", "SAFE"], downward,
    )
    assert metrics["confusion_matrix"] == {"TP": 1, "FN": 0, "TN": 1, "FP": 0}
    assert metrics["specificity"] == 1.0

    limited = ParameterConfig(
        name="Limited", display_name="Limited", unit="u", absolute_safety_limit=60.0,
        relative_drift_threshold=0.80,
    )
    assert evaluate_safety(50.0, 50.0, 57.0, 52.0, 63.0, limited).decision == "REVIEW"
    assert evaluate_safety(50.0, 50.0, 62.0, 61.0, 63.0, limited).decision == "REJECT"

    exact = fit_conformal(np.arange(1.0, 10.0), np.zeros(9), alpha=0.10)
    assert exact.q_hat == 9.0
    unbounded = fit_conformal(np.arange(1.0, 9.0), np.zeros(8), alpha=0.10)
    assert np.isinf(unbounded.q_hat)
    print("OK: direction-aware safety, absolute limits, and conformal rank")


def test_train_and_predict():
    df = generate_synthetic_dataset(n_components=120, n_lots=4, seed=1)
    system = train_all_parameters(df, n_folds=3, verbose=False)
    assert len(system.bundles) == len(PARAMETERS)
    for bundle in system.bundles.values():
        assert not any(c.endswith(("96h", "168h")) or "_z_" in c for c in bundle.feature_columns)
        assert "specificity" in bundle.safety_metrics
        assert "reject_confusion_matrix" in bundle.safety_metrics

    new_row = df.iloc[[0]].copy()
    results = predict_component(system, new_row, lot_reference_df=df)
    assert set(results.keys()) == set(system.bundles.keys())
    for pname, r in results.items():
        assert r["decision"].decision in ("SAFE", "REVIEW", "REJECT")
        lo, hi = r["interval"]
        assert lo <= r["prediction"] <= hi
    print("OK: train + predict_component end-to-end")


if __name__ == "__main__":
    test_synthetic_data_shape()
    test_synthetic_profiles()
    test_validation()
    test_feature_matrix_no_target_leak()
    test_direction_limits_and_conformal_calibration()
    test_train_and_predict()
    print("\nAll smoke tests passed.")
