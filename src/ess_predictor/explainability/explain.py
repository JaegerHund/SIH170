"""Explain predictions with feature contributions or local sensitivity."""

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
    method: str = "model contribution"


def explain_tree_model(pipeline, X_row: pd.DataFrame, top_k: int = 5) -> Optional[List[FeatureContribution]]:
    """
    SHAP explanation for a single row, for a tree-based model (RandomForest,
    QuantileRandomForest, or XGBoost) inside an sklearn Pipeline whose final
    step is named "model".
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


def explain_model_agnostic(pipeline, X_row: pd.DataFrame,
                           top_k: int = 5,
                           reference_features: Optional[pd.Series] = None
                           ) -> Optional[List[FeatureContribution]]:
    """Explain a prediction by changing each feature to its training reference.

    This is a one-feature-at-a-time counterfactual sensitivity, not a causal
    effect or an additive decomposition. Prefer the saved training-set
    medians; a fitted StandardScaler mean is used as a backward-compatible
    fallback.
    """
    try:
        scaler = pipeline.named_steps.get("scaler")
        if reference_features is not None:
            reference = np.asarray(
                [reference_features[feature] for feature in X_row.columns], dtype=float
            )
        elif scaler is not None and hasattr(scaler, "mean_"):
            reference = np.asarray(scaler.mean_, dtype=float).reshape(-1)
        else:
            return None
        if len(reference) != X_row.shape[1] or not np.isfinite(reference).all():
            return None

        baseline_prediction = float(np.asarray(pipeline.predict(X_row)).reshape(-1)[0])
        items = []
        for index, feature in enumerate(X_row.columns):
            counterfactual = X_row.copy()
            counterfactual.iloc[0, index] = reference[index]
            counterfactual_prediction = float(
                np.asarray(pipeline.predict(counterfactual)).reshape(-1)[0]
            )
            items.append(FeatureContribution(
                feature=feature,
                value=float(X_row.iloc[0, index]),
                contribution=baseline_prediction - counterfactual_prediction,
                method="counterfactual",
            ))

        items.sort(key=lambda item: abs(item.contribution), reverse=True)
        return items[:top_k]
    except Exception:
        return None


def explain_prediction(model_name: str, pipeline, X_row: pd.DataFrame, top_k: int = 5,
                       reference_features: Optional[pd.Series] = None):
    """Dispatch to the right explanation method based on model type."""
    if model_name in ("RandomForest", "QuantileRandomForest", "XGBoost"):
        contributions = explain_tree_model(pipeline, X_row, top_k)
        if contributions is not None:
            return contributions
    elif model_name in ("LinearRegression", "PolynomialRegression_deg2"):
        contributions = explain_linear_model(pipeline, X_row, top_k)
        if contributions is not None:
            return contributions

    return explain_model_agnostic(
        pipeline, X_row, top_k, reference_features=reference_features
    )


def contributions_to_text(contributions: Optional[List[FeatureContribution]]) -> List[str]:
    """Turn a list of FeatureContribution into short human-readable lines."""
    if not contributions:
        return ["Feature-level explanation unavailable for this model/prediction."]
    lines = []
    for c in contributions:
        direction = "increased" if c.contribution > 0 else "decreased"
        if c.method == "counterfactual":
            lines.append(
                f"{c.feature} (value={c.value:.3g}), compared with its typical "
                f"training value, {direction} the model's 168h prediction by about "
                f"{abs(c.contribution):.3g} units"
            )
        else:
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
        if model_name in ("RandomForest", "QuantileRandomForest", "XGBoost"):
            model = pipeline.named_steps["model"]
            importances = model.feature_importances_
            names = X.columns
        else:
            model = pipeline.named_steps["model"]
            if "poly" in pipeline.named_steps:
                names = pipeline.named_steps["poly"].get_feature_names_out(X.columns)
                importances = np.abs(model.coef_)
            else:
                names = X.columns
                if hasattr(model, "coef_"):
                    importances = np.abs(model.coef_)
                else:
                    scaler = pipeline.named_steps.get("scaler")
                    if scaler is None or not hasattr(scaler, "mean_"):
                        return pd.DataFrame(columns=["feature", "importance"])
                    training_means = np.asarray(scaler.mean_, dtype=float)
                    baseline_prediction = np.asarray(pipeline.predict(X), dtype=float)
                    importances = []
                    for index in range(X.shape[1]):
                        counterfactual = X.copy()
                        counterfactual.iloc[:, index] = training_means[index]
                        changed_prediction = np.asarray(
                            pipeline.predict(counterfactual), dtype=float
                        )
                        importances.append(
                            float(np.mean(np.abs(baseline_prediction - changed_prediction)))
                        )
                    importances = np.asarray(importances, dtype=float)
        order = np.argsort(importances)[::-1][:top_k]
        return pd.DataFrame({
            "feature": np.array(names)[order],
            "importance": np.array(importances)[order],
        })
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])
