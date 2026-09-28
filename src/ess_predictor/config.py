"""
config.py
---------
Central configuration for Module B: Time-Series Drift Predictor.

This is the single place where:
  - the set of tracked electrical parameters is defined
  - measurement timepoints are defined
  - safety thresholds (configurable, NOT hard-coded in logic) live
  - review-band margins live

Adding a new parameter later = add one entry to PARAMETERS below.
Nothing else in the codebase needs to change (models, features, reports
all iterate over PARAMETERS generically).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParameterConfig:
    """Configuration for a single electrical reliability parameter."""
    name: str                      # internal key, must match column prefix, e.g. "Iddq"
    display_name: str              # human-readable name for reports
    unit: str                      # e.g. "uA", "ns", "V", "mOhm"
    # Direction of "bad" drift. Most leakage/current/delay/Rds(on) parameters
    # degrade upward. Threshold voltage can drift either way depending on
    # failure mode, so it is handled with abs() safety logic by default.
    degrades_upward: bool = True
    # Safety threshold on the *absolute* predicted 168h value. If None,
    # only the relative-drift threshold below is used.
    absolute_safety_limit: Optional[float] = None
    # Safety threshold on relative drift (Predicted_168h - Value_0h)/|Value_0h|
    relative_drift_threshold: float = 0.50  # 50% drift default, override per param
    # Review-band margin: fraction of the threshold distance that triggers
    # REVIEW instead of a hard SAFE/REJECT split.
    review_margin_frac: float = 0.15


# ---------------------------------------------------------------------------
# The "major reliability indicators" requested. Kept deliberately limited
# per spec section 2 ("Do NOT automatically use every possible parameter").
# Architecture supports appending more ParameterConfig entries later.
# ---------------------------------------------------------------------------
PARAMETERS = [
    ParameterConfig(
        name="Iddq",
        display_name="Standby Current (Iddq)",
        unit="uA",
        degrades_upward=True,
        relative_drift_threshold=0.50,
    ),
    ParameterConfig(
        name="Leakage",
        display_name="Leakage Current",
        unit="uA",
        degrades_upward=True,
        relative_drift_threshold=0.50,
    ),
    ParameterConfig(
        name="PropagationDelay",
        display_name="Propagation Delay",
        unit="ns",
        degrades_upward=True,
        relative_drift_threshold=0.25,
    ),
    ParameterConfig(
        name="Icc",
        display_name="Supply Current (ICC)",
        unit="mA",
        degrades_upward=True,
        relative_drift_threshold=0.30,
    ),
    ParameterConfig(
        name="Vth",
        display_name="Threshold Voltage (Vth)",
        unit="V",
        degrades_upward=True,   # magnitude-of-shift is what matters; see safety.py
        relative_drift_threshold=0.20,
    ),
    ParameterConfig(
        name="RdsOn",
        display_name="On-State Resistance (RDS(on))",
        unit="mOhm",
        degrades_upward=True,
        relative_drift_threshold=0.30,
    ),
]

PARAM_NAMES = [p.name for p in PARAMETERS]

# Measurement timepoints available in the raw dataset (hours of burn-in).
TIMEPOINTS = [0, 24, 96, 168]
EARLY_TIMEPOINTS = [0, 24]          # always available at deployment time
OPTIONAL_TIMEPOINTS = [96]          # usable if present, not required
TARGET_TIMEPOINT = 168              # prediction target

# Lot-normalization / z-score settings
MIN_LOT_SIZE_FOR_STATS = 5          # below this, fall back to global stats

# Cross-validation
N_GROUP_FOLDS = 5
RANDOM_STATE = 42

# Synthetic generator tuning. Relative effects are intentionally kept here so
# the data-generating assumptions are reviewable without changing safety logic.
SYNTHETIC_DATA_CONFIG = {
    "lot_baseline_std": 0.025,
    "component_baseline_std": 0.035,
    "lot_degradation_std": 0.18,
    "component_degradation_std": 0.25,
    # Shared latent condition adds weak early clues and scales later severity.
    # Independent parameter and future terms keep the relationship uncertain.
    "lot_health_std": 0.35,
    "component_health_std": 0.85,
    "behavior_health_logit_scale": 0.35,
    "future_health_sensitivity": 0.42,
    "future_shock_std": 0.45,
    "early_health_drift_frac": 0.012,
    "early_health_baseline_frac": 0.005,
    "latent_severity_probabilities": {
        "mild": 0.25,
        "moderate": 0.45,
        "severe": 0.30,
    },
    "latent_late_drift_ranges": {
        "mild": (0.10, 0.25),
        "moderate": (0.30, 0.65),
        "severe": (0.70, 1.10),
    },
    "abrupt_early_warning_probability": 0.40,
    "abrupt_early_warning_range": (0.008, 0.025),
    "abrupt_quiet_warning_range": (0.0, 0.004),
    "abrupt_future_warning_sensitivity": 0.20,
    "parameter_early_sensitivity": {
        "Iddq": 1.00,
        "Leakage": 1.20,
        "PropagationDelay": 0.95,
        "Icc": 0.85,
        "Vth": 0.55,
        "RdsOn": 1.10,
    },
    "parameter_baseline_sensitivity": {
        "Iddq": 0.70,
        "Leakage": 0.80,
        "PropagationDelay": 0.60,
        "Icc": 0.55,
        "Vth": 0.35,
        "RdsOn": 0.75,
    },
    "lot_process_std": 0.006,
    "component_process_std": 0.006,
    "temporal_process_std": 0.004,
    "temporal_process_rho": 0.65,
    "measurement_noise_frac": {
        "Iddq": 0.015,
        "Leakage": 0.020,
        "PropagationDelay": 0.010,
        "Icc": 0.015,
        "Vth": 0.008,
        "RdsOn": 0.015,
    },
    "parameter_sensitivity": {
        "Iddq": 1.15,
        "Leakage": 1.30,
        "PropagationDelay": 0.85,
        "Icc": 0.95,
        "Vth": 0.55,
        "RdsOn": 1.10,
    },
}

# Conformal prediction
CONFORMAL_ALPHA = 0.10              # -> 90% prediction interval

# Decision-band thresholds, expressed as multiples of the safety threshold
# distance used to separate SAFE / REVIEW / REJECT (see safety.py).
DEFAULT_REVIEW_MARGIN_FRAC = 0.15
