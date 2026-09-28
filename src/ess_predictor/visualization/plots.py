"""
visualization.py
-----------------
Engineering plots (spec section 18). Each function saves a PNG to the
given output directory and returns the file path. Kept as plain
matplotlib (no seaborn dependency) for portability.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_actual_vs_predicted(y_true, y_pred, param_name, unit, out_dir):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.6, s=25, edgecolor="k", linewidth=0.3)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel(f"Actual 168h {param_name} ({unit})")
    ax.set_ylabel(f"Predicted 168h {param_name} ({unit})")
    ax.set_title(f"Actual vs Predicted -- {param_name}")
    ax.legend()
    return _save(fig, out_dir, f"{param_name}_actual_vs_predicted.png")


def plot_residuals(y_true, y_pred, param_name, unit, out_dir):
    resid = np.array(y_pred) - np.array(y_true)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, resid, alpha=0.6, s=25, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Actual 168h {param_name} ({unit})")
    ax.set_ylabel(f"Residual (Pred - Actual) ({unit})")
    ax.set_title(f"Residual Plot -- {param_name}")
    return _save(fig, out_dir, f"{param_name}_residuals.png")


def plot_error_distribution(y_true, y_pred, param_name, unit, out_dir):
    resid = np.array(y_pred) - np.array(y_true)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(resid, bins=25, edgecolor="k", alpha=0.75)
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Prediction Error ({unit})")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution -- {param_name}")
    return _save(fig, out_dir, f"{param_name}_error_distribution.png")


def plot_degradation_trajectories(df, param_name, out_dir, n_sample=25, seed=0):
    """Plot 0h -> 24h -> (96h) -> 168h trajectories for a sample of components."""
    rng = np.random.default_rng(seed)
    sample = df.sample(min(n_sample, len(df)), random_state=seed)

    tps = [0, 24, 96, 168]
    cols = [f"{param_name}_{t}h" for t in tps]
    available_tps = [t for t, c in zip(tps, cols) if c in df.columns]
    available_cols = [f"{param_name}_{t}h" for t in available_tps]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, row in sample.iterrows():
        vals = row[available_cols].values.astype(float)
        ax.plot(available_tps, vals, marker="o", alpha=0.5, linewidth=1)
    ax.set_xlabel("Burn-in time (h)")
    ax.set_ylabel(param_name)
    ax.set_title(f"Component Degradation Trajectories -- {param_name} (n={len(sample)})")
    return _save(fig, out_dir, f"{param_name}_trajectories.png")


def plot_prediction_intervals(y_true, y_pred, y_low, y_high, param_name, unit, out_dir, n_show=40):
    idx = np.argsort(y_true)[:n_show]
    y_true_s = np.array(y_true)[idx]
    y_pred_s = np.array(y_pred)[idx]
    y_low_s = np.array(y_low)[idx]
    y_high_s = np.array(y_high)[idx]

    x = np.arange(len(idx))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(x, y_pred_s, yerr=[y_pred_s - y_low_s, y_high_s - y_pred_s],
                fmt="o", markersize=4, capsize=2, alpha=0.7, label="Predicted (+/- interval)")
    ax.scatter(x, y_true_s, color="red", marker="x", s=30, label="Actual")
    ax.set_xlabel("Component (sorted by actual value)")
    ax.set_ylabel(f"{param_name} 168h ({unit})")
    ax.set_title(f"Prediction Intervals -- {param_name}")
    ax.legend()
    return _save(fig, out_dir, f"{param_name}_prediction_intervals.png")


def plot_feature_importance(importance_df: pd.DataFrame, param_name, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    df_sorted = importance_df.sort_values("importance")
    ax.barh(df_sorted["feature"], df_sorted["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance -- {param_name}")
    return _save(fig, out_dir, f"{param_name}_feature_importance.png")


def plot_false_negative_examples(df, y_true, y_pred, decisions, param_name, unit, out_dir,
                                  relative_drift_threshold, value_0h):
    """Highlight components that are truly defective but were classified SAFE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    value_0h = np.array(value_0h)
    decisions = np.array(decisions)

    true_rel_drift = (y_true - value_0h) / np.maximum(np.abs(value_0h), 1e-9)
    true_defective = true_rel_drift > relative_drift_threshold
    fn_mask = true_defective & (decisions == "SAFE")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(y_true[~fn_mask], y_pred[~fn_mask], alpha=0.4, s=20, label="Other")
    if fn_mask.sum() > 0:
        ax.scatter(y_true[fn_mask], y_pred[fn_mask], color="red", s=60,
                   marker="X", label=f"False Negatives (n={fn_mask.sum()})")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel(f"Actual 168h {param_name} ({unit})")
    ax.set_ylabel(f"Predicted 168h {param_name} ({unit})")
    ax.set_title(f"False Negative Examples -- {param_name}")
    ax.legend()
    return _save(fig, out_dir, f"{param_name}_false_negatives.png"), int(fn_mask.sum())


def plot_decision_distribution(decisions, param_name, out_dir):
    decisions = pd.Series(decisions)
    counts = decisions.value_counts().reindex(["SAFE", "REVIEW", "REJECT"]).fillna(0)
    colors = {"SAFE": "#2e8b57", "REVIEW": "#daa520", "REJECT": "#b22222"}
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=[colors[c] for c in counts.index])
    ax.set_ylabel("Component count")
    ax.set_title(f"SAFE / REVIEW / REJECT Distribution -- {param_name}")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(int(v)), ha="center")
    return _save(fig, out_dir, f"{param_name}_decision_distribution.png")


def plot_model_comparison(results: dict, param_name, out_dir):
    """Bar chart comparing CV MAE across candidate models."""
    names = list(results.keys())
    maes = [results[n].mae for n in names]
    stds = [results[n].mae_std for n in names]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(names, maes, yerr=stds, capsize=4, color="#4682b4")
    ax.set_ylabel("Cross-validated MAE")
    ax.set_title(f"Model Comparison -- {param_name}")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, out_dir, f"{param_name}_model_comparison.png")
