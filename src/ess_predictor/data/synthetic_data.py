"""Hierarchical synthetic ESS data generator for development and testing."""

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


def _lot_rng(seed, lot_id):
    """Make lot-specific randomness stable across processes and Python runs."""
    token = f"{int(seed)}:{lot_id}".encode("utf-8")
    lot_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return np.random.default_rng(lot_seed)


def _drift_profile(behavior, rng, tendency):
    """Return nonnegative relative drift at the four configured timepoints."""
    t = np.asarray(TIMEPOINTS, dtype=float) / max(TIMEPOINTS)
    if behavior == "normal":
        end = rng.uniform(0.02, 0.06) * np.clip(tendency, 0.5, 1.7)
        exponent = rng.uniform(0.85, 1.35)
        return end * t**exponent + rng.normal(0, 0.002, len(t)) * t
    if behavior == "gradual_degradation":
        end = rng.uniform(0.35, 0.70) * np.clip(tendency, 0.65, 1.45)
        return end * t**rng.uniform(1.15, 1.65)
    if behavior == "abrupt_degradation":
        onset = rng.uniform(45.0, 105.0)
        end = rng.uniform(0.50, 1.20) * np.clip(tendency, 0.65, 1.45)
        progress = np.clip((np.asarray(TIMEPOINTS, dtype=float) - onset) / (168.0 - onset), 0, 1)
        return end * progress**rng.uniform(0.75, 1.25)
    if behavior == "latent_defect":
        early = rng.uniform(0.025, 0.085) * np.clip(tendency, 0.7, 1.4)
        late = rng.uniform(0.50, 1.05) * np.clip(tendency, 0.7, 1.4)
        return early * t + late * t**rng.uniform(1.8, 2.5)
    if behavior == "static_limit_escape":
        end = rng.uniform(0.80, 1.40) * np.clip(tendency, 0.7, 1.35)
        return end * t**rng.uniform(0.9, 1.35)
    if behavior == "high_initial_stable":
        return np.zeros(len(t), dtype=float)
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
    behaviors = list(BEHAVIOR_WEIGHTS)
    weights = np.asarray(list(BEHAVIOR_WEIGHTS.values()), dtype=float)
    weights /= weights.sum()
    lot_ids = [f"LOT_{i:02d}" for i in range(1, n_lots + 1)]
    cfg = SYNTHETIC_DATA_CONFIG

    # Shared lot characteristics induce modest within-lot covariance.
    lot_effects = {}
    for lot_id in lot_ids:
        lot_rng = _lot_rng(seed, lot_id)
        lot_effects[lot_id] = {
            "baseline": lot_rng.normal(0, cfg["lot_baseline_std"]),
            "degradation": lot_rng.normal(0, cfg["lot_degradation_std"]),
            "process": lot_rng.normal(0, cfg["lot_process_std"]),
            "defect_propensity": lot_rng.normal(0, 0.12),
        }

    rows = []
    for i in range(n_components):
        lot_id = str(rng.choice(lot_ids))
        lot = lot_effects[lot_id]
        behavior = str(rng.choice(behaviors, p=weights))
        row = {
            "Component_ID": f"C{i + 1:04d}",
            "Lot_ID": lot_id,
            "Device_Type": rng.choice(device_types),
            "Temperature": rng.choice(temperatures),
            "True_Behavior": behavior,
            "Data_Source": "SYNTHETIC",
        }

        # Common component health affects parameters together, with independent
        # parameter terms preserving realistic imperfect correlation.
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
            param_base = baseline * np.exp(
                lot["baseline"] + component_baseline + 0.025 * param_quality
            )
            tendency = degradation * np.exp(0.15 * param_quality + lot["defect_propensity"])
            sensitivity = cfg["parameter_sensitivity"].get(pname, 1.0)
            drift = _drift_profile(behavior, rng, tendency) * sensitivity
            if behavior == "high_initial_stable":
                param_base *= rng.uniform(1.35, 1.80)

            # Each parameter has an independent temporal process/noise component,
            # while sharing a smaller common process deviation with its peers.
            param_temporal = np.empty(len(TIMEPOINTS), dtype=float)
            param_temporal[0] = rng.normal(0, 0.003)
            for j in range(1, len(TIMEPOINTS)):
                param_temporal[j] = 0.55 * param_temporal[j - 1] + rng.normal(0, 0.0025)
            process = lot["process"] + component_process + temporal_process + param_temporal
            noise_frac = cfg["measurement_noise_frac"].get(pname, 0.015)
            measurement = rng.normal(0, noise_frac, len(TIMEPOINTS))
            direction = 1.0 if param.degrades_upward else -1.0
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
