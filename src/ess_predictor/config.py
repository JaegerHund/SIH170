"""Parameter definitions, timepoints, model settings, and safety limits."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParameterConfig:
    """Configuration for a single electrical reliability parameter."""
    name: str                      # internal key, must match column prefix, e.g. "Iddq"
    display_name: str              # human-readable name for reports
    unit: str                      # e.g. "uA", "ns", "V", "mOhm"
    # True when degradation is in the upward direction.
    degrades_upward: bool = True
    # Optional limit on the predicted 168 h value.
    absolute_safety_limit: Optional[float] = None
    # Limit on relative drift from 0 h to predicted 168 h.
    relative_drift_threshold: float = 0.50  # 50% drift default, override per param
    # Fraction of the limit used as the review band. Provisional for the demo.
    review_margin_frac: float = 0.05


# Add catalog entries here; ACTIVE_PARAMETER_NAMES controls which ones run.
PARAMETER_CATALOG = {
    "Iddq": ParameterConfig(
        name="Iddq",
        display_name="Standby Current (Iddq)",
        unit="uA",
        degrades_upward=True,
        relative_drift_threshold=0.50,
    ),
    "Leakage": ParameterConfig(
        name="Leakage",
        display_name="Leakage Current",
        unit="uA",
        degrades_upward=True,
        relative_drift_threshold=0.50,
    ),
    "PropagationDelay": ParameterConfig(
        name="PropagationDelay",
        display_name="Propagation Delay",
        unit="ns",
        degrades_upward=True,
        relative_drift_threshold=0.25,
    ),
    "Icc": ParameterConfig(
        name="Icc",
        display_name="Supply Current (ICC)",
        unit="mA",
        degrades_upward=True,
        relative_drift_threshold=0.30,
    ),
    "Vth": ParameterConfig(
        name="Vth",
        display_name="Threshold Voltage (Vth)",
        unit="V",
        degrades_upward=True,   # magnitude-of-shift is what matters; see safety.py
        relative_drift_threshold=0.20,
    ),
    "RdsOn": ParameterConfig(
        name="RdsOn",
        display_name="On-State Resistance (RDS(on))",
        unit="mOhm",
        degrades_upward=True,
        relative_drift_threshold=0.30,
    ),
}

ACTIVE_PARAMETER_NAMES = ("Iddq", "Leakage", "PropagationDelay")
PARAMETERS = [PARAMETER_CATALOG[name] for name in ACTIVE_PARAMETER_NAMES]

PARAM_NAMES = [p.name for p in PARAMETERS]

# Measurement timepoints available in the raw dataset (hours of burn-in).
TIMEPOINTS = [0, 24, 96, 168]
EARLY_TIMEPOINTS = [0, 24]          # always available at deployment time
OPTIONAL_TIMEPOINTS = [96]          # usable if present, not required
TARGET_TIMEPOINT = 168              # prediction target

# Lot-normalization / z-score settings
MIN_LOT_SIZE_FOR_STATS = 5          # below this, fall back to global stats
# Provisional synthetic-demo cutoff for routing lot outliers to REVIEW.
# Recalibrate with measured lot distributions when engineered data is available.
LOT_OUTLIER_REVIEW_Z_THRESHOLD = 5.0

# Cross-validation
N_GROUP_FOLDS = 5
RANDOM_STATE = 42

# Synthetic data generator settings.
SYNTHETIC_DATA_CONFIG = {
    "lot_baseline_std": 0.025,
    "component_baseline_std": 0.035,
    "lot_degradation_std": 0.18,
    "component_degradation_std": 0.25,
    "degradation_scale": 1.0,
    "process_noise_scale": 1.0,
    "measurement_noise_scale": 1.0,
    # A shared health factor affects early readings and later drift; other
    # random effects keep the relationship noisy.
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
    # Parameter-specific skew in initial device values.
    "parameter_baseline_gamma_cv": {
        "Iddq": 0.045,
        "Leakage": 0.060,
        "PropagationDelay": 0.025,
        "Icc": 0.035,
        "Vth": 0.025,
        "RdsOn": 0.050,
    },
    "parameter_late_acceleration": {
        "Iddq": 1.05,
        "Leakage": 1.20,
        "PropagationDelay": 1.25,
        "Icc": 0.95,
        "Vth": 0.70,
        "RdsOn": 1.20,
    },
    "parameter_temporal_rho": {
        "Iddq": 0.60,
        "Leakage": 0.52,
        "PropagationDelay": 0.72,
        "Icc": 0.58,
        "Vth": 0.74,
        "RdsOn": 0.54,
    },
    "segment_slope_log_std": 0.28,
    "tester_readout_offset_std": 0.002,
    "tester_readout_rho": 0.60,
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

# Presets for synthetic development data. The balanced preset intentionally
# matches the historical behavior weights and base generator configuration.
# Easier/stress presets are scenarios, not estimates of real defect rates.
SYNTHETIC_DATA_PROFILES = {
    "balanced": {
        "behavior_weights": {
            "normal": 0.55,
            "gradual_degradation": 0.15,
            "abrupt_degradation": 0.08,
            "high_initial_stable": 0.10,
            "latent_defect": 0.07,
            "static_limit_escape": 0.05,
        },
        "config_overrides": {},
    },
    "easy_demo": {
        "behavior_weights": {
            "normal": 0.70,
            "gradual_degradation": 0.12,
            "abrupt_degradation": 0.05,
            "high_initial_stable": 0.10,
            "latent_defect": 0.02,
            "static_limit_escape": 0.01,
        },
        "config_overrides": {
            "lot_degradation_std": 0.12,
            "component_degradation_std": 0.18,
            "degradation_scale": 0.82,
            "future_health_sensitivity": 0.52,
            "future_shock_std": 0.28,
            "early_health_drift_frac": 0.018,
            "latent_severity_probabilities": {
                "mild": 0.55,
                "moderate": 0.35,
                "severe": 0.10,
            },
            "latent_late_drift_ranges": {
                "mild": (0.08, 0.20),
                "moderate": (0.25, 0.45),
                "severe": (0.55, 0.80),
            },
            "abrupt_early_warning_probability": 0.75,
            "abrupt_early_warning_range": (0.012, 0.030),
            "abrupt_quiet_warning_range": (0.0, 0.003),
            "abrupt_future_warning_sensitivity": 0.35,
            "process_noise_scale": 0.85,
            "measurement_noise_scale": 0.80,
        },
    },
    "stress": {
        "behavior_weights": {
            "normal": 0.35,
            "gradual_degradation": 0.20,
            "abrupt_degradation": 0.15,
            "high_initial_stable": 0.05,
            "latent_defect": 0.15,
            "static_limit_escape": 0.10,
        },
        "config_overrides": {
            "lot_degradation_std": 0.28,
            "component_degradation_std": 0.35,
            "degradation_scale": 1.15,
            "future_health_sensitivity": 0.30,
            "future_shock_std": 0.65,
            "early_health_drift_frac": 0.008,
            "latent_severity_probabilities": {
                "mild": 0.15,
                "moderate": 0.35,
                "severe": 0.50,
            },
            "latent_late_drift_ranges": {
                "mild": (0.15, 0.35),
                "moderate": (0.40, 0.80),
                "severe": (0.85, 1.30),
            },
            "abrupt_early_warning_probability": 0.20,
            "abrupt_early_warning_range": (0.005, 0.015),
            "abrupt_quiet_warning_range": (0.0, 0.002),
            "abrupt_future_warning_sensitivity": 0.10,
            "process_noise_scale": 1.25,
            "measurement_noise_scale": 1.25,
        },
    },
}

# Conformal prediction. Provisional 85% interval for the synthetic demo to
# reduce REVIEW volume; recalibrate this operating point with engineered data.
CONFORMAL_ALPHA = 0.15              # -> 85% prediction interval

# Decision-band thresholds, expressed as multiples of the safety threshold
# distance used to separate SAFE / REVIEW / REJECT (see safety.py).
DEFAULT_REVIEW_MARGIN_FRAC = 0.05
