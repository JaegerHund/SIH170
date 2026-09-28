"""Hierarchical longitudinal ESS data generator for development and testing.

Initial device spread is positive/skewed, lot and component effects are
separate, and each behavior draws random interval slopes at the four readouts.
Persistent AR process variation and smaller tester/readout plus measurement
effects are then added to the latent piecewise trajectory.
"""

import hashlib

import numpy as np
import pandas as pd

from ess_predictor.config import (
    PARAMETERS,
    RANDOM_STATE,
    SYNTHETIC_DATA_CONFIG,
    TIMEPOINTS,
)

BEHAVIOR_WEIGHTS = {
    "normal": 0.55,
    "gradual_degradation": 0.15,
    "abrupt_degradation": 0.08,
    "high_initial_stable": 0.10,
    "latent_defect": 0.07,
    "static_limit_escape": 0.05,
}

BASELINES = {
    "Iddq": 10.0,
    "Leakage": 8.0,
    "PropagationDelay": 12.0,
    "Icc": 25.0,
    "Vth": 2.0,
    "RdsOn": 45.0,
}

# More susceptible latent conditions modestly increase the odds of degradation
# classes. High-initial-stable is sampled separately so its baseline level does
# not imply a latent degradation tendency.
_BEHAVIOR_HEALTH_LOADINGS = {
    "normal": -0.20,
    "gradual_degradation": 0.45,
    "abrupt_degradation": 0.35,
    "high_initial_stable": 0.0,
    "latent_defect": 0.55,
    "static_limit_escape": 0.40,
}


def _lot_rng(seed, lot_id):
    """Make lot-specific randomness stable across processes and Python runs."""
    token = f"{int(seed)}:{lot_id}".encode("utf-8")
    lot_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return np.random.default_rng(lot_seed)


def _sample_behavior(rng, health, cfg):
    """Sample a behavior with weak dependence on latent health/susceptibility."""
    high_initial_weight = BEHAVIOR_WEIGHTS["high_initial_stable"]
    if rng.random() < high_initial_weight:
        return "high_initial_stable"

    candidates = [name for name in BEHAVIOR_WEIGHTS if name != "high_initial_stable"]
    base = np.asarray([BEHAVIOR_WEIGHTS[name] for name in candidates], dtype=float)
    loadings = np.asarray([_BEHAVIOR_HEALTH_LOADINGS[name] for name in candidates])
    logits = np.log(base) + cfg["behavior_health_logit_scale"] * health * loadings
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()
    return str(rng.choice(candidates, p=probabilities))


def _segment_increments(total, weights, rng, late_acceleration, log_slope_std):
    """Split total relative drift into random positive piecewise increments."""
    weights = np.asarray(weights, dtype=float)
    acceleration = np.ones(len(weights), dtype=float)
    if len(acceleration) > 1:
        acceleration[-1] = max(float(late_acceleration), 0.05)
    random_slopes = rng.lognormal(-0.5 * log_slope_std**2, log_slope_std, len(weights))
    weighted_slopes = weights * acceleration * random_slopes
    return float(total) * weighted_slopes / weighted_slopes.sum()


def _cumulative_profile(increments):
    """Convert the three readout-interval changes to four device readouts."""
    return np.concatenate(([0.0], np.cumsum(increments)))


def _drift_profile(
    behavior,
    rng,
    tendency,
    latent_severity_range=None,
    relative_drift_threshold=None,
    late_acceleration=1.0,
    slope_log_std=0.28,
):
    """Draw a component-specific piecewise relative drift path at readout times."""
    times = np.asarray(TIMEPOINTS, dtype=float)
    durations = np.diff(times)
    if behavior == "normal":
        end = rng.uniform(0.02, 0.06) * np.clip(tendency, 0.5, 1.7)
        increments = _segment_increments(
            end, durations / durations.sum(), rng, 1.0, slope_log_std
        )
        return _cumulative_profile(increments)
    if behavior == "gradual_degradation":
        end = rng.uniform(0.35, 0.70) * np.clip(tendency, 0.65, 1.45)
        early = end * rng.uniform(0.04, 0.15)
        later = _segment_increments(
            end - early, (0.44, 0.56), rng, late_acceleration, slope_log_std
        )
        return _cumulative_profile(np.concatenate(([early], later)))
    if behavior == "abrupt_degradation":
        onset = rng.uniform(times[1], times[-1] - 24.0)
        end = rng.uniform(0.50, 1.20) * np.clip(tendency, 0.65, 1.45)
        progress = np.clip((times - onset) / (times[-1] - onset), 0, 1)
        raw_profile = end * progress**rng.uniform(0.75, 1.25)
        raw_increments = np.diff(raw_profile)
        weights = raw_increments / max(raw_increments.sum(), np.finfo(float).tiny)
        increments = _segment_increments(
            end, weights, rng, late_acceleration, slope_log_std
        )
        return _cumulative_profile(increments)
    if behavior == "latent_defect":
        early = rng.uniform(0.005, 0.025) * np.clip(tendency, 0.7, 1.4)
        severity_range = latent_severity_range or (0.50, 1.05)
        late = rng.uniform(*severity_range) * np.clip(tendency, 0.7, 1.4)
        later = _segment_increments(
            late, (0.38, 0.62), rng, late_acceleration, slope_log_std
        )
        return _cumulative_profile(np.concatenate(([early], later)))
    if behavior == "static_limit_escape":
        end = relative_drift_threshold * rng.uniform(0.75, 1.35) * np.clip(
            tendency, 0.7, 1.35
        )
        early = min(rng.uniform(0.0, 0.025) * np.clip(tendency, 0.7, 1.35), end * 0.20)
        later = _segment_increments(
            end - early, (0.40, 0.60), rng, late_acceleration, slope_log_std
        )
        return _cumulative_profile(np.concatenate(([early], later)))
    if behavior == "high_initial_stable":
        return np.zeros(len(times), dtype=float)
    raise ValueError(behavior)


def generate_synthetic_dataset(
    n_components: int = 300,
    n_lots: int = 6,
    device_types=("DeviceA", "DeviceB"),
    temperatures=(125,),
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Return one row per component with the established wide measurement schema."""
    if n_components < 0 or n_lots < 1:
        raise ValueError("n_components must be nonnegative and n_lots must be positive")
    if n_components and (not device_types or not temperatures):
        raise ValueError("device_types and temperatures must be nonempty")

    rng = np.random.default_rng(seed)
    lot_ids = [f"LOT_{i:02d}" for i in range(1, n_lots + 1)]
    cfg = SYNTHETIC_DATA_CONFIG

    # Shared lot characteristics induce modest within-lot covariance.
    lot_effects = {}
    for lot_id in lot_ids:
        lot_rng = _lot_rng(seed, lot_id)
        lot_effects[lot_id] = {
            "baseline": lot_rng.normal(0, cfg["lot_baseline_std"]),
            "degradation": lot_rng.normal(0, cfg["lot_degradation_std"]),
            "health": lot_rng.normal(0, cfg["lot_health_std"]),
            "process": lot_rng.normal(0, cfg["lot_process_std"]),
            "defect_propensity": lot_rng.normal(0, 0.12),
        }

    # A small readout/tester calibration path is shared across devices for
    # each parameter, as a real test system can shift between measurement runs.
    tester_offsets = {}
    for param in PARAMETERS:
        offsets = np.empty(len(TIMEPOINTS), dtype=float)
        tester_rho = cfg["tester_readout_rho"]
        tester_std = cfg["tester_readout_offset_std"]
        offsets[0] = rng.normal(0, tester_std)
        for j in range(1, len(TIMEPOINTS)):
            offsets[j] = (
                tester_rho * offsets[j - 1]
                + rng.normal(0, tester_std * np.sqrt(1 - tester_rho**2))
            )
        tester_offsets[param.name] = offsets

    rows = []
    for i in range(n_components):
        lot_id = str(rng.choice(lot_ids))
        lot = lot_effects[lot_id]
        # A shared lot/component condition weakly affects class likelihood,
        # early observations, and later degradation. Independent draws below
        # prevent this hidden state from becoming a deterministic label.
        health = lot["health"] + rng.normal(0, cfg["component_health_std"])
        behavior = _sample_behavior(rng, health, cfg)
        latent_severity_range = None
        if behavior == "latent_defect":
            severity_names = list(cfg["latent_severity_probabilities"])
            severity_probabilities = list(cfg["latent_severity_probabilities"].values())
            latent_severity = str(rng.choice(severity_names, p=severity_probabilities))
            latent_severity_range = cfg["latent_late_drift_ranges"][latent_severity]

        abrupt_warning = 0.0
        if behavior == "abrupt_degradation":
            if rng.random() < cfg["abrupt_early_warning_probability"]:
                abrupt_warning = rng.uniform(*cfg["abrupt_early_warning_range"])
            else:
                abrupt_warning = rng.uniform(*cfg["abrupt_quiet_warning_range"])
        row = {
            "Component_ID": f"C{i + 1:04d}",
            "Lot_ID": lot_id,
            "Device_Type": rng.choice(device_types),
            "Temperature": rng.choice(temperatures),
            "True_Behavior": behavior,
            "Data_Source": "SYNTHETIC",
        }

        # Inter-device susceptibility and an independent random slope both
        # affect aging; neither is a deterministic encoding of the behavior.
        quality = rng.normal(0, 1)
        degradation = np.exp(
            cfg["component_degradation_std"] * quality + lot["degradation"]
        )
        component_baseline = rng.normal(0, cfg["component_baseline_std"])
        component_process = rng.normal(0, cfg["component_process_std"])
        rho = cfg["temporal_process_rho"]
        temporal_process = np.empty(len(TIMEPOINTS), dtype=float)
        temporal_process[0] = rng.normal(0, cfg["temporal_process_std"])
        for j in range(1, len(TIMEPOINTS)):
            temporal_process[j] = (
                rho * temporal_process[j - 1]
                + rng.normal(0, cfg["temporal_process_std"] * np.sqrt(1 - rho**2))
            )

        for param in PARAMETERS:
            pname = param.name
            baseline = BASELINES.get(pname, 1.0)
            param_quality = rng.normal(0, 0.65)
            baseline_cv = cfg["parameter_baseline_gamma_cv"].get(pname, 0.04)
            gamma_shape = 1.0 / baseline_cv**2
            gamma_scale = baseline_cv**2
            positive_skew = rng.gamma(gamma_shape, gamma_scale)
            param_health = 0.78 * health + rng.normal(0, 0.55)
            early_sensitivity = cfg["parameter_early_sensitivity"].get(pname, 1.0)
            baseline_sensitivity = cfg["parameter_baseline_sensitivity"].get(pname, 1.0)
            direction = 1.0 if param.degrades_upward else -1.0
            param_base = baseline * positive_skew * np.exp(
                lot["baseline"]
                + component_baseline
                + 0.025 * param_quality
                + direction
                * cfg["early_health_baseline_frac"]
                * baseline_sensitivity
                * param_health
            )
            tendency = degradation * np.exp(
                cfg["future_health_sensitivity"] * param_health
                + 0.15 * param_quality
                + lot["defect_propensity"]
                + rng.normal(0, cfg["future_shock_std"])
                + cfg["abrupt_future_warning_sensitivity"]
                * abrupt_warning
                / max(cfg["abrupt_early_warning_range"])
            )
            sensitivity = cfg["parameter_sensitivity"].get(pname, 1.0)
            late_acceleration = cfg["parameter_late_acceleration"].get(pname, 1.0)
            drift = _drift_profile(
                behavior,
                rng,
                tendency,
                latent_severity_range=latent_severity_range,
                relative_drift_threshold=param.relative_drift_threshold,
                late_acceleration=late_acceleration,
                slope_log_std=cfg["segment_slope_log_std"],
            ) * sensitivity
            # Susceptibility leaves a small early footprint which persists
            # somewhat over time; independent profile/noise terms mean that
            # early values remain only a noisy clue to the eventual severity.
            early_profile = np.asarray([0.0, 1.0, 1.20, 1.35])
            drift += (
                cfg["early_health_drift_frac"]
                * early_sensitivity
                * param_health
                * early_profile
            )
            if behavior == "abrupt_degradation":
                drift += abrupt_warning * early_sensitivity * early_profile
            if behavior == "high_initial_stable":
                param_base *= rng.uniform(1.35, 1.80)

            # Each parameter has an independent temporal process/noise component,
            # with parameter-specific persistence and a smaller shared component.
            param_temporal = np.empty(len(TIMEPOINTS), dtype=float)
            param_rho = cfg["parameter_temporal_rho"].get(pname, 0.55)
            param_temporal[0] = rng.normal(0, 0.003)
            for j in range(1, len(TIMEPOINTS)):
                param_temporal[j] = (
                    param_rho * param_temporal[j - 1]
                    + rng.normal(0, 0.0025 * np.sqrt(1 - param_rho**2))
                )
            process = (
                lot["process"]
                + component_process
                + temporal_process
                + param_temporal
                + tester_offsets[pname]
            )
            noise_frac = cfg["measurement_noise_frac"].get(pname, 0.015)
            measurement = rng.normal(0, noise_frac, len(TIMEPOINTS))
            values = param_base * (1 + direction * drift + process + measurement)

            # Static-limit escape is intended to show relative drift below an
            # explicitly configured absolute ceiling, when there is room to do so.
            limit = param.absolute_safety_limit
            if behavior == "static_limit_escape" and limit is not None and limit > param_base:
                max_value = float(np.max(values))
                # Preserve a feasible positive drift even when the limit is
                # only slightly above this component's initial value.
                ceiling = max(limit * 0.98, (param_base + limit) / 2)
                if max_value > ceiling:
                    values = param_base + (values - param_base) * (
                        (ceiling - param_base) / (max_value - param_base)
                    )

            for tp, value in zip(TIMEPOINTS, values):
                row[f"{pname}_{tp}h"] = round(float(max(value, np.finfo(float).tiny)), 4)
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    dataset = generate_synthetic_dataset(n_components=300)
    print(dataset.shape)
    print(dataset.head())
    print(dataset["True_Behavior"].value_counts())
