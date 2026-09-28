"""
ess_predictor
=============
Module B: Time-Series Drift Predictor for high-reliability electronics
Environmental Stress Screening (ESS).

Predicts 168h parametric drift from 0h/24h burn-in measurements and
flags components whose early behavior indicates abnormal future
degradation, even when every individual measurement is within static
datasheet limits.

Typical usage
-------------
    from ess_predictor.data.synthetic_data import generate_synthetic_dataset
    from ess_predictor.pipeline import train_all_parameters, predict_component

    df = generate_synthetic_dataset(300)
    system = train_all_parameters(df)
    results = predict_component(system, new_component_row, lot_reference_df=df)

See the top-level README.md for the full package layout and design notes.
"""

__version__ = "0.1.0"
