# SIH170 Module B — 168-hour drift predictor

This project demonstrates a regression pipeline for forecasting electrical measurements after extended burn-in. It uses the component's **0-hour and 24-hour readings** to predict its **168-hour reading**, then routes the prediction to `SAFE`, `REVIEW`, or `REJECT` using a prediction interval and configured drift limits.

The repository includes a reproducible synthetic data generator, model comparison and validation, reports, plots, a saved model, and a Streamlit dashboard.

For a longer walkthrough of the code and each file, see [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md).

## What the model predicts

The current active parameters are `Iddq`, `Leakage`, and `PropagationDelay`. Each is a separate regression problem:

```text
Iddq_0h + Iddq_24h                     -> Iddq_168h
Leakage_0h + Leakage_24h               -> Leakage_168h
PropagationDelay_0h + PropagationDelay_24h -> PropagationDelay_168h
```

Only that parameter's early readings are used for its forecast. The model does not use another parameter's readings, or any 96-hour or 168-hour value, as a feature. The generator still writes values at 96 h and 168 h so you can inspect the simulated trajectory and compare the forecast with its synthetic target.

The configuration catalog also defines `Icc`, `Vth`, and `RdsOn`; the current active parameter list is defined in `src/ess_predictor/config.py`.

## Quick start on Windows PowerShell

Run these commands from the repository root (the folder containing `pyproject.toml`):

The project metadata requires Python 3.9 or later. Check the selected interpreter with `python --version` before installing.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[app]"
```

This installs the package and its dependencies, including the optional Streamlit dashboard. If the `py` launcher is unavailable, use `python -m venv .venv` for the first command. If you already have a virtual environment, activate it and install with the same `python -m pip install -e ".[app]"` command.

Keep using the same activated environment for all commands below. This avoids installing packages in one Python environment and running the project with another.

## Run the end-to-end demo

```powershell
python scripts/run_demo.py
```

The demo explicitly generates **350 components**, **7 lots**, with seed `7` and the `balanced` profile. It validates the data, compares candidate models, fits one selected model per active parameter, calibrates prediction intervals, assigns component dispositions, saves reports and charts, and saves/reloads the trained system.

Generated files go into these folders at the repository root:

| Path | Contents |
| --- | --- |
| `outputs/synthetic_dataset.csv` | Generated measurements and synthetic annotations. |
| `outputs/forecast_predictions.csv` | Actual/predicted 168-hour values, reference forecasts, and prediction errors. Includes multiple validation bases; see below. |
| `outputs/leave_one_lot_out_mae.csv` | Per-model scores for each held-out lot. Used for model selection when available. |
| `outputs/robustness_*.csv` | Forecast sensitivity to small 0-hour input perturbations. |
| `outputs/screening_summary.txt` | Counts of components marked `SAFE`/`PASS`, `REVIEW`, and `REJECT`. |
| `outputs/component_disposition_report.txt` | Human-readable component lists and per-parameter reasons. |
| `outputs/component_dispositions.csv` | Machine-readable disposition details. |
| `outputs/example_component_reports.txt` | Note pointing to the complete disposition report. |
| `plots/` | Regression, residual, trajectory, interval, model, and decision charts. |
| `models_store/trained_system.pkl` | Serialized trained model bundles and calibration information. |

The demo writes generated artifacts into `outputs/`, `plots/`, and `models_store/` at the repository root.

### Understand the validation rows

Check the `Prediction_Basis` field in `forecast_predictions.csv`:

- `LOLO CV (used for model selection)` (or component-grouped CV when lot validation is unavailable) contains out-of-fold predictions from the validation process used to compare models.
- `Held-out component split (not used to fit/select point model)` contains components reserved from fitting and model selection. Their targets are used to calibrate prediction intervals.

## Open the CSV dashboard

After running the demo, start the dashboard:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local address to open in your browser. The dashboard has an overview, actual-versus-predicted plots, a component explorer, lot-validation scores, dispositions, and a CSV browser/download view.

The dashboard reads the demo's saved CSVs. Run the demo first to generate the tables and forecasts shown there.

## Try the synthetic profiles

The generator has three named scenarios:

| Profile | Intended use |
| --- | --- |
| `easy_demo` | Mostly nominal components, while retaining some noisy and rare challenge cases. |
| `balanced` | Default profile; preserves the historical generator mix as closely as possible. |
| `stress` | More degradation, latent/abrupt cases, future variation, and measurement/process noise. |

The same seed and profile produce the same dataset. To create a separate 120-component CSV with a chosen profile, run this after installation:

```powershell
python -c "from ess_predictor.data.synthetic_data import generate_synthetic_dataset; generate_synthetic_dataset(n_components=120, n_lots=6, seed=170, profile='stress').to_csv('outputs/stress_example.csv', index=False)"
```

Change `profile='stress'` to `profile='easy_demo'` or `profile='balanced'` as needed. Change the component count, lot count, or seed to get a different reproducible sample. Every generated row includes `Data_Source=SYNTHETIC` and a `Synthetic_Profile` label.

The original 19-column generator CSVs are in `sample_data/`. Each contains the active parameters at 0 h, 24 h, 96 h, and 168 h, together with component/lot identifiers and synthetic annotations. Use 0 h and 24 h values as the forecast inputs; the 168 h values are the targets. More details are in [`sample_data/README.md`](sample_data/README.md).

## Run the smoke checks

```powershell
python tests/test_smoke.py
```

The script checks synthetic profile/schema/reproducibility behavior, input validation, the no-future-feature guard, safety/conformal behavior, and an end-to-end train-and-predict path.

## Common setup and run issues

- **`ModuleNotFoundError` for a project dependency:** activate the project's `.venv`, then run `python -m pip install -e ".[app]"` in that same terminal. To confirm which Python is active, run `python -c "import sys; print(sys.executable)"`.
- **Streamlit is missing or its command is not recognized:** use `python -m streamlit run app.py`; if that module is missing, install with `python -m pip install -e ".[app]"`.
- **The dashboard says output CSVs are missing:** run `python scripts/run_demo.py` from the repository root, then refresh the dashboard.

## Run the separate regression/safety diagnostic

```powershell
python scripts/diagnose_regression_safety.py
```

By default, it generates a separate balanced synthetic dataset with 350 components, 7 lots, and seed 7. It compares absolute-value, absolute-drift, and relative-drift targets; reports regression and safety measures, behavior-specific errors, threshold sensitivity, and lot generalization; and writes CSVs under `outputs/regression_safety_diagnostics/`.

You can override its settings:

```powershell
python scripts/diagnose_regression_safety.py --n-components 500 --n-lots 10 --seed 21 --folds 5 --output-dir outputs/my_diagnostics
```

This diagnostic is separate from the main demo. Threshold sensitivity is reported for comparison and does not modify the configured safety limits.

## Input schema and data handling

For **training**, the dataset should have one unique `Component_ID` per component row and, for each parameter being trained, its early columns plus a 168-hour target. The expected pattern is `{Parameter}_{hour}h`, for example:

```text
Component_ID, Lot_ID, Device_Type, Temperature,
Iddq_0h, Iddq_24h, Iddq_168h,
Leakage_0h, Leakage_24h, Leakage_168h,
PropagationDelay_0h, PropagationDelay_24h, PropagationDelay_168h
```

`Lot_ID`, `Device_Type`, and `Temperature` can be provided as context fields. The generator also includes optional 96-hour readings and labels. For **inference**, `predict_component()` uses a one-row DataFrame containing `Component_ID`, each active parameter's 0-hour and 24-hour values, and `Lot_ID` for the lot-level outlier check. Model inputs remain only that parameter's early readings and features derived from them.

The validator reports missing columns, targets, and measurement warnings. Rows without the 168-hour target are excluded from training for that parameter; early readings can still be used for inference with a trained system.

## Use the package from Python

After installation, the main functions can be imported from another Python script:

```python
from ess_predictor.data.synthetic_data import generate_synthetic_dataset
from ess_predictor.data.validation import validate_dataset
from ess_predictor.pipeline import train_all_parameters, predict_component, save_system, load_system

# Generate example training data.
data = generate_synthetic_dataset(n_components=350, n_lots=7, seed=7, profile="balanced")
print(validate_dataset(data).summary())

system = train_all_parameters(data)
save_system(system, "models_store/trained_system.pkl")
system = load_system("models_store/trained_system.pkl")

# Create a separate synthetic row to demonstrate the inference input shape.
new_component = generate_synthetic_dataset(n_components=1, n_lots=7, seed=8)
new_component["Component_ID"] = "NEW-0001"
future_columns = [c for c in new_component if c.endswith(("96h", "168h"))]
new_component = new_component.drop(
    columns=future_columns + ["True_Behavior", "Data_Source", "Synthetic_Profile"]
)
forecasts = predict_component(system, new_component, lot_reference_df=data)
for parameter, result in forecasts.items():
    print(parameter, result["prediction"], result["interval"], result["decision"].decision)
```

`predict_component()` returns the forecast, interval, decision, selected model, and optional explanation for each active parameter.

## How models and decisions work

For each active parameter, feature engineering derives early change, relative drift, and slope from its 0-hour and 24-hour measurements. Candidate models include Linear Regression, degree-2 Polynomial Regression with Ridge regularization, RBF-SVR, Gaussian Process Regression, Random Forest, Quantile Random Forest, and XGBoost when available. Last-value and linear-extrapolation forecasts are included in the comparison and can also be selected if their validation MAE is lowest. Selection minimizes grouped-validation MAE, with RMSE used to break ties.

Validation keeps a component's rows together; when lot labels are available, leave-one-lot-out scores are also used for model selection. Held-out calibration errors produce forecast intervals. The configured conformal alpha is 0.15, corresponding to nominal 85% intervals.

The configured relative drift thresholds are 50% for Iddq, 50% for Leakage, and 25% for PropagationDelay, with a 5% review band. A 24-hour lot outlier at an absolute z-score of 5 or more can downgrade `SAFE` to `REVIEW`; at component level `REJECT` takes precedence over `REVIEW`, which takes precedence over `SAFE`.

Each parameter status is determined by comparing its predicted drift and interval against its configured threshold. A `REVIEW` result indicates that the forecast interval is near or crosses the decision boundary. At component level, `REJECT` takes precedence over `REVIEW`, which takes precedence over `SAFE`.

## Project layout

```text
app.py                              Read-only Streamlit dashboard
pyproject.toml                      Package metadata and dependencies
scripts/run_demo.py                 End-to-end synthetic demo and report generation
scripts/diagnose_regression_safety.py  Separate regression/safety diagnostic
tests/test_smoke.py                 Lightweight smoke checks
src/ess_predictor/config.py          Active parameters, thresholds, profiles, model settings
src/ess_predictor/pipeline.py        Training, model selection, calibration, inference, save/load
src/ess_predictor/data/              Synthetic generator and schema validation
src/ess_predictor/features/          Early-readout feature engineering
src/ess_predictor/models/            Regression model zoo and grouped validation
src/ess_predictor/uncertainty/       Conformal prediction intervals
src/ess_predictor/safety/            Safety decisions and confusion metrics
src/ess_predictor/explainability/    Local explanations and feature importance
src/ess_predictor/robustness/        Input-noise sensitivity analysis
src/ess_predictor/reporting/         Text report formatting
src/ess_predictor/visualization/     Plot generation
sample_data/                         Generator-format synthetic CSV fixtures
outputs/, plots/, models_store/      Generated results, figures, and model files
```
