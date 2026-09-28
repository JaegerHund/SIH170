"""
safety.py
---------
Safety-slope decision logic (spec section 8-9) and physics-plausibility
checks (spec section 12).

Decision logic overview
------------------------
For a component/parameter:
  1. Compute Predicted_Relative_Drift = (Pred_168h - Value_0h) / |Value_0h|
  2. Compare against ParameterConfig.relative_drift_threshold (T).
  3. Also check absolute_safety_limit if configured.
  4. Use the conformal prediction interval to decide confidence:
       - if the interval is wide relative to the prediction -> REVIEW
       - if the prediction clearly crosses T even in the best case -> REJECT
       - if the prediction clearly stays under T even in the worst case
         -> SAFE
       - otherwise (threshold falls inside the interval, i.e. genuinely
         ambiguous) -> REVIEW

This directly implements "SAFE / REVIEW / REJECT rather than forcing
every component into SAFE/REJECT" and "prioritize avoiding defective
components escaping screening" -- REJECT is only assigned when there is
positive evidence of exceeding threshold; genuine ambiguity always routes
to REVIEW rather than defaulting to SAFE.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ess_predictor.config import ParameterConfig, DEFAULT_REVIEW_MARGIN_FRAC


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
    z_score_flag_threshold: float = 3.0,
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

    # Does the whole interval clear the threshold on the safe side?
    worst_case_rel = max(rel_drift_low, rel_drift_high)  # "worst" = largest drift
    best_case_rel = min(rel_drift_low, rel_drift_high)

    crosses_abs_limit = False
    if param_cfg.absolute_safety_limit is not None:
        crosses_abs_limit = pred_high >= param_cfg.absolute_safety_limit
        if crosses_abs_limit:
            reasons.append(
                f"Prediction interval reaches or exceeds absolute safety limit "
                f"({param_cfg.absolute_safety_limit} {param_cfg.unit})."
            )

    if worst_case_rel < (T - margin) and not crosses_abs_limit:
        decision = "SAFE"
        reasons.append(
            f"Predicted relative drift ({rel_drift*100:.1f}%) and its full "
            f"prediction interval stay comfortably below the {T*100:.0f}% "
            f"safety threshold."
        )
    elif best_case_rel > (T + margin) or crosses_abs_limit:
        decision = "REJECT"
        reasons.append(
            f"Predicted relative drift ({rel_drift*100:.1f}%) exceeds the "
            f"{T*100:.0f}% safety threshold, including under the optimistic "
            f"end of the prediction interval."
        )
    else:
        decision = "REVIEW"
        reasons.append(
            f"Predicted relative drift ({rel_drift*100:.1f}%) is close to the "
            f"{T*100:.0f}% safety threshold, or the prediction interval "
            f"[{rel_drift_low*100:.1f}%, {rel_drift_high*100:.1f}%] straddles "
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
    component decision. Conservative w.r.t. false negatives (spec section
    21): REJECT dominates REVIEW dominates SAFE.
    """
    decisions = set(d.decision for d in param_decisions.values())
    if "REJECT" in decisions:
        return "REJECT"
    if "REVIEW" in decisions:
        return "REVIEW"
    return "SAFE"


# ---------------------------------------------------------------------------
# Safety-oriented evaluation metrics (spec section 8): treat "actually
# exceeds threshold" (based on the TRUE 168h value) as the positive
# ("defective") class, and evaluate the SAFE/REVIEW/REJECT classifier
# (collapsing REVIEW+REJECT into "flagged", since the whole point of
# REVIEW is that the component does NOT silently pass) as a detector.
# ---------------------------------------------------------------------------

def safety_confusion_metrics(y_true_168h: np.ndarray, value_0h: np.ndarray,
                              decisions: list, param_cfg: ParameterConfig):
    """
    y_true_168h, value_0h: arrays of true 168h values and 0h baselines.
    decisions: list of SafetyDecision.decision strings ("SAFE"/"REVIEW"/"REJECT")
    param_cfg: for the relative drift threshold defining ground-truth "defective".

    A component is truly "defective" if its ACTUAL relative drift exceeds
    the safety threshold. A false negative = truly defective component
    that the system called SAFE (REVIEW is NOT a false negative, because
    it does not clear the component for use without further scrutiny).
    """
    true_rel_drift = (y_true_168h - value_0h) / np.maximum(np.abs(value_0h), 1e-9)
    true_defective = true_rel_drift > param_cfg.relative_drift_threshold

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

    return {
        "confusion_matrix": {"TP": tp, "FN": fn, "TN": tn, "FP": fp},
        "recall_sensitivity": recall,
        "precision": precision,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "n_true_defective": int(n_pos),
        "n_true_ok": int(n_neg),
    }


# ---------------------------------------------------------------------------
# Physics/engineering plausibility check (spec section 12)
# ---------------------------------------------------------------------------

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
