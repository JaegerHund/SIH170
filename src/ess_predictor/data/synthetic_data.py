"""
synthetic_data.py
------------------
Optional synthetic-data generator for development/testing ONLY.

IMPORTANT: This data is synthetic and must never be treated as a substitute
for real experimental validation data (see spec section 17 / 19). Every
DataFrame produced here is tagged with a `Data_Source = "SYNTHETIC"` column
so downstream code and reports can never silently confuse it with real
measurements.

Simulated behaviors per component (chosen per-component at random, weighted):
  - normal:               gentle, expected drift
  - gradual_degradation:   steady upward drift, crosses threshold late
  - abrupt_degradation:    sharp change appearing between 24h and 96h/168h
  - high_initial_stable:   starts high (but within spec) and stays flat
                           -> should NOT be falsely flagged
  - latent_defect:         normal-looking at 0h, accelerating drift after
                           -> the hard case Module B exists to catch
  - static_limit_escape:   ends up under the absolute datasheet limit but
                           with abnormal drift shape -> must be caught by
                           relative-drift / z-score logic, not the static
                           limit
"""

import numpy as np
import pandas as pd

from ess_predictor.config import PARAMETERS, TIMEPOINTS, RANDOM_STATE

BEHAVIOR_WEIGHTS = {
    "normal": 0.55,
    "gradual_degradation": 0.15,
    "abrupt_degradation": 0.08,
    "high_initial_stable": 0.10,
    "latent_defect": 0.07,
    "static_limit_escape": 0.05,
}


def _trajectory(base, behavior, rng, noise_frac=0.02):
    """Return dict {0:..,24:..,96:..,168:..} for one parameter of one part."""
    t = np.array(TIMEPOINTS, dtype=float)

    if behavior == "normal":
        # ~2-6% total drift over the run, smooth
        total_drift_frac = rng.uniform(0.02, 0.06)
        shape = (t / t.max()) ** 1.0
        vals = base * (1 + total_drift_frac * shape)

    elif behavior == "gradual_degradation":
        total_drift_frac = rng.uniform(0.35, 0.70)
        shape = (t / t.max()) ** 1.2
        vals = base * (1 + total_drift_frac * shape)

    elif behavior == "abrupt_degradation":
        # flat then a step between 24h and 96h
        step_frac = rng.uniform(0.5, 1.2)
        vals = np.array([base, base * 1.02,
                          base * (1 + step_frac * 0.8),
                          base * (1 + step_frac)])

    elif behavior == "high_initial_stable":
        # starts elevated (e.g. 1.5-2.5x a "typical" base) but flat afterward
        elevated = base * rng.uniform(1.5, 2.2)
        vals = np.full(4, elevated) * (1 + rng.uniform(-0.01, 0.03, size=4))

    elif behavior == "latent_defect":
        # innocuous at 0h/24h, accelerating after
        early_drift = rng.uniform(0.03, 0.10)
        late_accel = rng.uniform(0.6, 1.3)
        vals = np.array([
            base,
            base * (1 + early_drift),
            base * (1 + early_drift + late_accel * 0.55),
            base * (1 + early_drift + late_accel),
        ])

    elif behavior == "static_limit_escape":
        # large relative drift, but engineered to stay under a generous
        # absolute datasheet ceiling (handled at generation call site)
        total_drift_frac = rng.uniform(0.8, 1.4)
        shape = (t / t.max()) ** 1.1
        vals = base * (1 + total_drift_frac * shape)

    else:
        raise ValueError(behavior)

    noise = rng.normal(0, noise_frac * base, size=4)
    vals = vals + noise
    return {int(tp): float(v) for tp, v in zip(t, vals)}


def generate_synthetic_dataset(
    n_components: int = 300,
    n_lots: int = 6,
    device_types=("DeviceA", "DeviceB"),
    temperatures=(125,),
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate a synthetic ESS burn-in dataset with the same wide-column
    structure described in the spec (section 3), for all parameters in
    config.PARAMETERS.

    Returns a DataFrame with one row per Component_ID.
    """
    rng = np.random.default_rng(seed)
    behaviors = list(BEHAVIOR_WEIGHTS.keys())
    weights = np.array(list(BEHAVIOR_WEIGHTS.values()))
    weights = weights / weights.sum()

    lot_ids = [f"LOT_{i:02d}" for i in range(1, n_lots + 1)]

    # Baseline "typical" value per parameter (rough, illustrative magnitudes)
    baseline = {
        "Iddq": 10.0,          # uA
        "Leakage": 8.0,        # uA
        "PropagationDelay": 12.0,  # ns
        "Icc": 25.0,           # mA
        "Vth": 2.0,            # V
        "RdsOn": 45.0,         # mOhm
    }

    rows = []
    for i in range(n_components):
        comp_id = f"C{i+1:04d}"
        lot_id = rng.choice(lot_ids)
        device_type = rng.choice(device_types)
        temperature = rng.choice(temperatures)
        behavior = rng.choice(behaviors, p=weights)

        # Lot-level shift: each lot has a small systematic offset so that
        # lot-normalization (z-scores) actually matters.
        lot_seed = abs(hash(lot_id)) % (2**32)
        lot_rng = np.random.default_rng(lot_seed)
        lot_offset = lot_rng.normal(1.0, 0.03)

        row = {
            "Component_ID": comp_id,
            "Lot_ID": lot_id,
            "Device_Type": device_type,
            "Temperature": temperature,
            "True_Behavior": behavior,   # kept for eval/debug only, NOT a feature
            "Data_Source": "SYNTHETIC",
        }

        for p in PARAMETERS:
            base = baseline[p.name] * lot_offset * rng.uniform(0.9, 1.1)
            traj = _trajectory(base, behavior, rng)
            for tp in TIMEPOINTS:
                row[f"{p.name}_{tp}h"] = round(traj[tp], 4)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset(n_components=300)
    print(df.shape)
    print(df.head())
    print(df["True_Behavior"].value_counts())
