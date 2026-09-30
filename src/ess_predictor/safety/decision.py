"""Route predictions to SAFE, REVIEW, or REJECT using drift and limits."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ess_predictor.config import LOT_OUTLIER_REVIEW_Z_THRESHOLD, ParameterConfig


@dataclass
class SafetyDecision:
    decision: str                     # "SAFE" | "REVIEW" | "REJECT"
    predicted_value: float
    predicted_low: float
    predicted_high: float
    predicted_drift: float
    predicted_relative_drift: float
    threshold: float
    reasons: list


def _relative_drift(pred, value_0h):
    denom = max(abs(value_0h), 1e-9)
    return (pred - value_0h) / denom


def evaluate_safety(
    value_0h: float,
    value_24h: float,
    pred_168h: float,
    pred_low: float,
    pred_high: float,
    param_cfg: ParameterConfig,
    lot_z_24h: Optional[float] = None,
    z_score_flag_threshold: float = LOT_OUTLIER_REVIEW_Z_THRESHOLD,
) -> SafetyDecision:
    """
    Produce a SAFE / REVIEW / REJECT decision for one parameter of one
    component, using the predicted value, its conformal interval, and the
    configured (not hard-coded) safety thresholds.
    """
    reasons = []

    rel_drift = _relative_drift(pred_168h, value_0h)
    rel_drift_low = _relative_drift(pred_low, value_0h)
    rel_drift_high = _relative_drift(pred_high, value_0h)
    abs_drift = pred_168h - value_0h

    T = param_cfg.relative_drift_threshold
    margin = param_cfg.review_margin_frac * T
    direction = 1.0 if param_cfg.degrades_upward else -1.0
    risk_rel = direction * rel_drift
    risk_bounds = sorted((direction * rel_drift_low, direction * rel_drift_high))
    best_case_risk, worst_case_risk = risk_bounds

    reaches_abs_limit = False
    confidently_over_abs_limit = False
    if param_cfg.absolute_safety_limit is not None:
        reaches_abs_limit = max(pred_low, pred_high) >= param_cfg.absolute_safety_limit
        confidently_over_abs_limit = min(pred_low, pred_high) >= param_cfg.absolute_safety_limit
        if reaches_abs_limit:
            reasons.append(
                f"Prediction interval reaches or exceeds absolute safety limit "
                f"({param_cfg.absolute_safety_limit} {param_cfg.unit})."
            )

    if worst_case_risk < (T - margin) and not reaches_abs_limit:
        decision = "SAFE"
        reasons.append(
            f"Predicted drift in the configured degradation direction "
            f"({risk_rel*100:.1f}%) and its full "
            f"prediction interval stay comfortably below the {T*100:.0f}% "
            f"safety threshold."
        )
    elif best_case_risk > (T + margin) or confidently_over_abs_limit:
        decision = "REJECT"
        if confidently_over_abs_limit:
            reasons.append(
                f"The full prediction interval is at or above the absolute "
                f"safety limit ({param_cfg.absolute_safety_limit} {param_cfg.unit})."
            )
        else:
            reasons.append(
                f"Predicted drift in the configured degradation direction "
                f"({risk_rel*100:.1f}%) exceeds the {T*100:.0f}% safety "
                f"threshold, including under the optimistic end of the interval."
            )
    else:
        decision = "REVIEW"
        reasons.append(
            f"Predicted drift in the configured degradation direction "
            f"({risk_rel*100:.1f}%) is close to the "
            f"{T*100:.0f}% safety threshold, or the prediction interval "
            f"[{risk_bounds[0]*100:.1f}%, {risk_bounds[1]*100:.1f}%] straddles "
            f"the threshold -- not enough confidence to decide automatically."
        )

    if lot_z_24h is not None and abs(lot_z_24h) >= z_score_flag_threshold:
        reasons.append(
            f"Component is a strong outlier relative to its lot at 24h "
            f"(z={lot_z_24h:.2f})."
        )
        if decision == "SAFE":
            decision = "REVIEW"
            reasons.append("Downgraded from SAFE to REVIEW due to lot-outlier status.")

    return SafetyDecision(
        decision=decision,
        predicted_value=pred_168h,
        predicted_low=pred_low,
        predicted_high=pred_high,
        predicted_drift=abs_drift,
        predicted_relative_drift=rel_drift,
        threshold=T,
        reasons=reasons,
    )


def overall_component_decision(param_decisions: dict) -> str:
    """
    Combine per-parameter SAFE/REVIEW/REJECT decisions into one overall
    component decision. Any REJECT dominates, then any REVIEW keeps the
    component out of SAFE so an unresolved parameter cannot silently pass.
    """
    decisions = set(d.decision for d in param_decisions.values())
    if "REJECT" in decisions:
        return "REJECT"
    if "REVIEW" in decisions:
        return "REVIEW"
    return "SAFE"


# Label defects from true 168 h drift and any configured absolute limit.

def safety_confusion_metrics(y_true_168h: np.ndarray, value_0h: np.ndarray,
                              decisions: list, param_cfg: ParameterConfig,
                              truth_param_cfg: Optional[ParameterConfig] = None):
    """
    y_true_168h, value_0h: arrays of true 168h values and 0h baselines.
    decisions: list of SafetyDecision.decision strings ("SAFE"/"REVIEW"/"REJECT")
    param_cfg: for the relative drift threshold defining ground-truth "defective".

    A component is truly defective if actual drift in the configured bad
    direction exceeds the relative threshold or it exceeds a configured
    absolute ceiling. The main confusion matrix treats REVIEW and REJECT as
    flagged; a second matrix reports confident REJECT separately.
    """
    y_true_168h = np.asarray(y_true_168h, dtype=float)
    value_0h = np.asarray(value_0h, dtype=float)
    true_rel_drift = (y_true_168h - value_0h) / np.maximum(np.abs(value_0h), 1e-9)
    truth_cfg = truth_param_cfg or param_cfg
    direction = 1.0 if truth_cfg.degrades_upward else -1.0
    true_defective = direction * true_rel_drift > truth_cfg.relative_drift_threshold
    if truth_cfg.absolute_safety_limit is not None:
        true_defective |= y_true_168h >= truth_cfg.absolute_safety_limit

    decisions = np.array(decisions)
    predicted_safe = decisions == "SAFE"
    predicted_flagged = ~predicted_safe  # REVIEW or REJECT

    tp = int(np.sum(true_defective & predicted_flagged))   # correctly flagged
    fn = int(np.sum(true_defective & predicted_safe))       # DANGEROUS: missed
    tn = int(np.sum(~true_defective & predicted_safe))      # correctly passed
    fp = int(np.sum(~true_defective & predicted_flagged))   # unnecessarily flagged

    n_pos = tp + fn
    n_neg = tn + fp

    recall = tp / n_pos if n_pos > 0 else np.nan          # sensitivity for defects
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    fnr = fn / n_pos if n_pos > 0 else np.nan
    fpr = fp / n_neg if n_neg > 0 else np.nan
    specificity = tn / n_neg if n_neg > 0 else np.nan

    # A review routes the part for human scrutiny; it is not equivalent to a
    # confident rejection. Report both operational definitions of a positive.
    predicted_reject = decisions == "REJECT"
    reject_tp = int(np.sum(true_defective & predicted_reject))
    reject_fn = int(np.sum(true_defective & ~predicted_reject))
    reject_tn = int(np.sum(~true_defective & ~predicted_reject))
    reject_fp = int(np.sum(~true_defective & predicted_reject))

    return {
        "confusion_matrix": {"TP": tp, "FN": fn, "TN": tn, "FP": fp},
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "reject_confusion_matrix": {
            "TP": reject_tp, "FN": reject_fn, "TN": reject_tn, "FP": reject_fp,
        },
        "reject_TP": reject_tp,
        "reject_FN": reject_fn,
        "reject_TN": reject_tn,
        "reject_FP": reject_fp,
        "recall_sensitivity": recall,
        "precision": precision,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "specificity": specificity,
        "reject_precision": reject_tp / (reject_tp + reject_fp) if (reject_tp + reject_fp) else np.nan,
        "reject_recall": reject_tp / n_pos if n_pos else np.nan,
        "reject_false_positive_rate": reject_fp / n_neg if n_neg else np.nan,
        "review_count": int(np.sum(decisions == "REVIEW")),
        "reject_count": int(np.sum(predicted_reject)),
        "relative_drift_threshold": param_cfg.relative_drift_threshold,
        "ground_truth_relative_drift_threshold": truth_cfg.relative_drift_threshold,
        "review_margin_frac": param_cfg.review_margin_frac,
        "safe_relative_boundary": param_cfg.relative_drift_threshold * (1 - param_cfg.review_margin_frac),
        "reject_relative_boundary": param_cfg.relative_drift_threshold * (1 + param_cfg.review_margin_frac),
        "absolute_safety_limit": param_cfg.absolute_safety_limit,
        "degradation_direction": "upward" if param_cfg.degrades_upward else "downward",
        "n_true_defective": int(n_pos),
        "n_true_ok": int(n_neg),
    }


# Basic plausibility checks for predicted drift.

def physics_plausibility_flags(value_0h: np.ndarray, value_24h: np.ndarray,
                                pred_168h: np.ndarray, param_cfg: ParameterConfig):
    """
    Flags predictions that are physically implausible GIVEN the observed
    early trend, for parameters where degradation is expected to be
    monotonically increasing (param_cfg.degrades_upward=True). Does NOT
    force monotonicity on the model -- purely an informational flag for the
    explainability report distinguishing "statistical output" from
    "physically expected direction."
    """
    value_0h = np.asarray(value_0h, dtype=float)
    value_24h = np.asarray(value_24h, dtype=float)
    pred_168h = np.asarray(pred_168h, dtype=float)

    if not param_cfg.degrades_upward:
        return np.zeros(len(value_0h), dtype=bool)

    rising_early = value_24h > value_0h
    falling_late = pred_168h < value_24h
    implausible = rising_early & falling_late
    return implausible
