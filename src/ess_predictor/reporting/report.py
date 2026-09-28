"""
report.py
---------
Generates the human-readable COMPONENT SCREENING REPORT (spec sections
13 and 21), covering every trained parameter for one component and an
overall conservative decision.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from ess_predictor.config import ParameterConfig
from ess_predictor.safety.decision import SafetyDecision, overall_component_decision
from ess_predictor.explainability.explain import FeatureContribution, contributions_to_text


@dataclass
class ParameterReportBlock:
    param_cfg: ParameterConfig
    value_0h: float
    value_24h: float
    early_drift: float
    relative_drift_pct: float
    lot_z_24h: Optional[float]
    decision: SafetyDecision
    explanation_lines: List[str]
    physically_implausible: bool


def build_parameter_block(param_cfg: ParameterConfig, value_0h, value_24h,
                           lot_z_24h, decision: SafetyDecision,
                           contributions: Optional[List[FeatureContribution]],
                           physically_implausible: bool) -> ParameterReportBlock:
    early_drift = value_24h - value_0h
    rel_drift_pct = (early_drift / max(abs(value_0h), 1e-9)) * 100
    return ParameterReportBlock(
        param_cfg=param_cfg,
        value_0h=value_0h,
        value_24h=value_24h,
        early_drift=early_drift,
        relative_drift_pct=rel_drift_pct,
        lot_z_24h=lot_z_24h,
        decision=decision,
        explanation_lines=contributions_to_text(contributions),
        physically_implausible=physically_implausible,
    )


def format_component_report(component_id: str, lot_id: str, temperature,
                             blocks: Dict[str, ParameterReportBlock]) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("COMPONENT SCREENING REPORT")
    lines.append("=" * 60)
    lines.append(f"Component ID: {component_id}")
    lines.append(f"Lot: {lot_id}")
    lines.append(f"Temperature: {temperature}\u00b0C")
    lines.append("")

    for pname, b in blocks.items():
        cfg = b.param_cfg
        lines.append("-" * 60)
        lines.append(f"PARAMETER: {cfg.display_name}")
        lines.append("")
        lines.append(f"0h:               {b.value_0h:.3f} {cfg.unit}")
        lines.append(f"24h:              {b.value_24h:.3f} {cfg.unit}")
        lines.append(f"Early Drift:      {b.early_drift:+.3f} {cfg.unit}")
        lines.append(f"Relative Drift:   {b.relative_drift_pct:+.1f}%")
        if b.lot_z_24h is not None:
            lines.append(f"Lot Z-score:      {b.lot_z_24h:+.2f}")
        lines.append("")
        lines.append(f"Predicted 168h:   {b.decision.predicted_value:.3f} {cfg.unit}")
        lines.append(f"Prediction range: [{b.decision.predicted_low:.3f}, "
                      f"{b.decision.predicted_high:.3f}] {cfg.unit}")
        lines.append("")
        if cfg.absolute_safety_limit is not None:
            lines.append(f"Absolute safety limit: {cfg.absolute_safety_limit} {cfg.unit}")
        lines.append(f"Relative drift safety threshold: {cfg.relative_drift_threshold*100:.0f}%")
        lines.append("")
        lines.append(f"Decision: {b.decision.decision}")
        lines.append("")
        lines.append("Reasons:")
        for i, r in enumerate(b.decision.reasons, 1):
            lines.append(f"  {i}. {r}")
        if b.physically_implausible:
            lines.append("  NOTE: Early trend rises but 168h prediction falls -- "
                          "flagged as physically atypical for this parameter; "
                          "recommend engineering review of the raw trace.")
        lines.append("")
        lines.append("Top contributing factors:")
        for l in b.explanation_lines:
            lines.append(f"  - {l}")
        lines.append("")

    overall = overall_component_decision({k: b.decision for k, b in blocks.items()})
    lines.append("=" * 60)
    lines.append(f"OVERALL COMPONENT DECISION: {overall}")
    lines.append("=" * 60)
    flagged_params = [k for k, b in blocks.items() if b.decision.decision != "SAFE"]
    if flagged_params:
        lines.append(f"Driven by: {', '.join(flagged_params)}")
    lines.append("(Overall decision is conservative: REJECT > REVIEW > SAFE "
                  "across all parameters.)")

    return "\n".join(lines)
