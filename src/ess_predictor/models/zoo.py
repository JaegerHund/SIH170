"""
models.py
---------
Defines the candidate regression models (spec section 5) and a
model-comparison routine that selects based on validation performance
(spec: "select the model based on validation performance rather than
assuming the most complicated model is best").

Models:
  1. Linear Regression              - transparent baseline
  2. Polynomial Regression (deg=2)  - captures mild curvature, regularized
                                       via Ridge to avoid overfitting on
                                       small component counts
  3. Random Forest Regression       - nonlinear benchmark
  4. XGBoost                        - main nonlinear benchmark

All models are wrapped in sklearn Pipelines with StandardScaler where it
matters (linear/poly) so coefficients and behavior are well-conditioned.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from ess_predictor.config import RANDOM_STATE


def build_model_zoo() -> Dict[str, Pipeline]:
    """Return {model_name: unfitted sklearn Pipeline}."""
    zoo = {}

    zoo["LinearRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    # Degree-2 polynomial, but Ridge-regularized (RidgeCV picks alpha
    # internally) -- spec explicitly warns against high-degree unregularized
    # polynomial fits overfitting on small component counts.
    zoo["PolynomialRegression_deg2"] = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ])

    zoo["RandomForest"] = Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    if HAS_XGB:
        zoo["XGBoost"] = Pipeline([
            ("model", XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

    return zoo


@dataclass
class FittedModel:
    name: str
    pipeline: Pipeline
    cv_mae: float
    cv_rmse: float
    cv_r2: float
