"""Regression models compared by the drift predictor."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from quantile_forest import RandomForestQuantileRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from ess_predictor.config import CONFORMAL_ALPHA, RANDOM_STATE

QRF_MODEL_NAME = "QuantileRandomForest"


class EarlyReadoutBaseline(RegressorMixin, BaseEstimator):
    """A no-fit reference forecast from one parameter's early readouts."""

    def __init__(self, method: str = "last_value"):
        self.method = method

    def fit(self, X, y):
        if self.method not in ("last_value", "linear_extrapolation"):
            raise ValueError(f"Unknown baseline method: {self.method}")
        if not hasattr(X, "columns"):
            raise TypeError("EarlyReadoutBaseline expects a pandas DataFrame.")
        value_0_cols = [col for col in X.columns if str(col).endswith("_0h")]
        value_24_cols = [col for col in X.columns if str(col).endswith("_24h")]
        if len(value_0_cols) != 1 or len(value_24_cols) != 1:
            raise ValueError("Baseline needs exactly one parameter's 0h and 24h columns.")
        self.value_0_col_ = value_0_cols[0]
        self.value_24_col_ = value_24_cols[0]
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        if not hasattr(self, "value_0_col_"):
            raise RuntimeError("Baseline must be fitted before prediction.")
        value_0 = np.asarray(X[self.value_0_col_], dtype=float)
        value_24 = np.asarray(X[self.value_24_col_], dtype=float)
        if self.method == "last_value":
            return value_24
        # Extend the 0-24 h slope to 168 h.
        return value_24 + ((168.0 - 24.0) / (24.0 - 0.0)) * (value_24 - value_0)


def build_model_zoo(include_baselines: bool = False) -> Dict[str, Pipeline]:
    """Return {model_name: unfitted sklearn Pipeline}."""
    zoo = {}

    if include_baselines:
        zoo["Baseline_LastValue24h"] = Pipeline([
            ("model", EarlyReadoutBaseline(method="last_value")),
        ])
        zoo["Baseline_LinearExtrapolation"] = Pipeline([
            ("model", EarlyReadoutBaseline(method="linear_extrapolation")),
        ])

    zoo["LinearRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    # Keep the polynomial model regularized for the small dataset.
    zoo["PolynomialRegression_deg2"] = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ])

    # SVR's RBF kernel uses scaled features.
    zoo["RBF_SVR"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
    ])

    # Model a smooth trend plus measurement noise.
    gpr_kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    )
    zoo["GaussianProcess"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianProcessRegressor(
            kernel=gpr_kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=RANDOM_STATE,
        )),
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

    # Use the conditional median as QRF's point prediction.
    zoo[QRF_MODEL_NAME] = Pipeline([
        ("model", RandomForestQuantileRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            default_quantiles=0.5,
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


def predict_qrf_quantiles(pipeline: Pipeline, X, alpha: float = CONFORMAL_ALPHA):
    """Return the QRF's lower/upper nominal prediction quantiles."""
    model = pipeline.named_steps["model"]
    quantiles = [alpha / 2, 1 - alpha / 2]
    predictions = np.asarray(model.predict(X, quantiles=quantiles), dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("Quantile Random Forest did not return two quantile columns.")
    return predictions


@dataclass
class FittedModel:
    name: str
    pipeline: Pipeline
    cv_mae: float
    cv_rmse: float
    cv_r2: float
