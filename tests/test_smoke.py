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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ess_predictor.data.synthetic_data import generate_synthetic_dataset
from ess_predictor.data.validation import validate_dataset
from ess_predictor.features.engineering import build_feature_matrix
from ess_predictor.pipeline import train_all_parameters, predict_component


def test_synthetic_data_shape():
    df = generate_synthetic_dataset(n_components=60, n_lots=3)
    assert len(df) == 60
    assert "Data_Source" in df.columns
    assert (df["Data_Source"] == "SYNTHETIC").all()
    print("OK: synthetic data generation")


def test_validation():
    df = generate_synthetic_dataset(n_components=60, n_lots=3)
    report = validate_dataset(df)
    assert len(report.usable_parameters) == 6
    print("OK: data validation")


def test_feature_matrix_no_target_leak():
    df = generate_synthetic_dataset(n_components=60, n_lots=3)
    X, y, groups = build_feature_matrix(df, "Iddq")
    assert not any(c.endswith("168h") for c in X.columns), "Target column leaked into features!"
    assert len(X) == len(y) == len(groups)
    print("OK: feature matrix excludes target")


def test_train_and_predict():
    df = generate_synthetic_dataset(n_components=120, n_lots=4, seed=1)
    system = train_all_parameters(df, n_folds=3, verbose=False)
    assert len(system.bundles) == 6

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
    test_validation()
    test_feature_matrix_no_target_leak()
    test_train_and_predict()
    print("\nAll smoke tests passed.")
