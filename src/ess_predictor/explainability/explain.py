"""
explainability.py
------------------
Per-prediction explanations (spec section 13), tailored so a QA/reliability
engineer -- not only an ML researcher -- can understand each decision.

Two paths:
  - Tree models (RandomForest, XGBoost): SHAP values via shap.TreeExplainer,
    reduced to the top contributing features in plain language.
  - Linear/Polynomial models: standardized coefficients -> approximate
    per-feature contribution = coefficient * (feature value in std units).

Falls back gracefully (returns None / a warning string) if SHAP is
unavailable or fails for a given model type, rather than crashing the
whole report.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class FeatureContribution:
    feature: str
    value: float
    contribution: float   # signed, in target units (approx for linear models)


def explain_tree_model(pipeline, X_row: pd.DataFrame, top_k: int = 5) -> Optional[List[FeatureContribution]]:
    """
    SHAP explanation for a single row, for a tree-based model (RandomForest
    or XGBoost) inside an sklearn Pipeline whose final step is named "model".
    """
    if not HAS_SHAP:
        return None
    try:
        model = pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        vals = np.array(shap_values).flatten()

        contributions = [
            FeatureContribution(feature=col, value=float(X_row.iloc[0][col]), contribution=float(v))
            for col, v in zip(X_row.columns, vals)
        ]
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
        return contributions[:top_k]
    except Exception:
        return None


def explain_linear_model(pipeline, X_row: pd.DataFrame, top_k: int = 5) -> Optional[List[FeatureContribution]]:
    """
    Approximate per-feature contribution for LinearRegression pipelines:
    contribution_i = coef_i * scaled_feature_value_i
    (in standardized units, since the pipeline's first step is a scaler).
    Works for plain LinearRegression; for the Ridge-on-polynomial-features
    pipeline this explains contributions in the *expanded* polynomial
    feature space, which is still useful diagnostically but noted as such.
    """
    try:
        scaler = pipeline.named_steps.get("scaler")
        model = pipeline.named_steps["model"]

        if "poly" in pipeline.named_steps:
            poly = pipeline.named_steps["poly"]
            X_scaled = scaler.transform(X_row)
            X_poly = poly.transform(X_scaled)
            feature_names = poly.get_feature_names_out(X_row.columns)
            coefs = model.coef_
            contributions = coefs * X_poly.flatten()
            items = list(zip(feature_names, X_poly.flatten(), contributions))
        else:
            X_scaled = scaler.transform(X_row).flatten()
            coefs = model.coef_
            contributions = coefs * X_scaled
            items = list(zip(X_row.columns, X_scaled, contributions))

        out = [FeatureContribution(feature=f, value=float(v), contribution=float(c))
               for f, v, c in items]
        out.sort(key=lambda c: abs(c.contribution), reverse=True)
        return out[:top_k]
    except Exception:
        return None


def explain_prediction(model_name: str, pipeline, X_row: pd.DataFrame, top_k: int = 5):
    """Dispatch to the right explanation method based on model type."""
    if model_name in ("RandomForest", "XGBoost"):
        return explain_tree_model(pipeline, X_row, top_k)
    else:
        return explain_linear_model(pipeline, X_row, top_k)


def contributions_to_text(contributions: Optional[List[FeatureContribution]]) -> List[str]:
    """Turn a list of FeatureContribution into short human-readable lines."""
    if not contributions:
        return ["Feature-level explanation unavailable for this model/prediction."]
    lines = []
    for c in contributions:
        direction = "increased" if c.contribution > 0 else "decreased"
        lines.append(
            f"{c.feature} (value={c.value:.3g}) {direction} the predicted 168h "
            f"value by ~{abs(c.contribution):.3g} units"
        )
    return lines


def global_feature_importance(model_name: str, pipeline, X: pd.DataFrame, top_k: int = 10):
    """
    Global (dataset-level) feature importance for the model-comparison
    visualizations (spec section 18, "feature importance").
    """
    try:
        if model_name in ("RandomForest", "XGBoost"):
            model = pipeline.named_steps["model"]
            importances = model.feature_importances_
            names = X.columns
        else:
            model = pipeline.named_steps["model"]
            if "poly" in pipeline.named_steps:
                names = pipeline.named_steps["poly"].get_feature_names_out(X.columns)
            else:
                names = X.columns
            importances = np.abs(model.coef_)
        order = np.argsort(importances)[::-1][:top_k]
        return pd.DataFrame({
            "feature": np.array(names)[order],
            "importance": np.array(importances)[order],
        })
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])
