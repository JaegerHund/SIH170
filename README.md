# ess-predictor — Module B: Time-Series Drift Predictor

Predicts 168h electrical parametric drift from 0h/24h ESS burn-in
measurements and flags components whose *early* behavior indicates
abnormal *future* degradation — even when every measurement individually
sits within the datasheet's static pass/fail limits.

This is a proper installable Python package now (not a folder of loose
scripts), so you can import it from anywhere — including a future
FastAPI/Flask backend for your website — without fighting `sys.path` or
`PYTHONPATH`.

## Project layout

```
ess-predictor/
├── pyproject.toml              # makes this pip-installable (`pip install -e .`)
├── src/
│   └── ess_predictor/          # the actual importable package
│       ├── config.py           # parameter list + thresholds (edit thresholds here)
│       ├── pipeline.py         # orchestration: train_all_parameters(), predict_component()
│       ├── data/
│       │   ├── synthetic_data.py   # dev/test data generator (tagged SYNTHETIC)
│       │   └── validation.py       # flexible schema validation, missing-value policy
│       ├── features/
│       │   └── engineering.py      # absolute / drift / drift-rate / lot z-score features
│       ├── models/
│       │   ├── zoo.py              # Linear, Poly-deg2-Ridge, RandomForest, XGBoost
│       │   └── cross_validation.py # component-level GroupKFold / LeaveOneGroupOut
│       ├── uncertainty/
│       │   └── conformal.py        # split conformal prediction intervals
│       ├── safety/
│       │   └── decision.py         # SAFE / REVIEW / REJECT logic + safety metrics
│       ├── robustness/
│       │   └── noise_testing.py    # measurement-noise sensitivity testing
│       ├── explainability/
│       │   └── explain.py          # SHAP (trees) / coefficients (linear) explanations
│       ├── reporting/
│       │   └── report.py           # COMPONENT SCREENING REPORT text generation
│       └── visualization/
│           └── plots.py            # all required engineering plots
├── scripts/
│   └── run_demo.py             # end-to-end demo you actually run
├── app.py                      # read-only Streamlit dashboard for saved CSVs
├── tests/
│   └── test_smoke.py           # fast sanity check — run after any edit
├── plots/                      # generated figures land here (gitignored)
├── outputs/                    # generated reports/CSVs land here (gitignored)
└── models_store/               # saved trained models land here (gitignored)
```

Every subpackage has a one-line docstring in its `__init__.py` explaining
what it holds, so `import ess_predictor.safety` alone tells you what's
inside without opening a file.

## Install

```bash
cd ess-predictor
pip install -e .
```

The `-e` (editable) install means changes you make to files under `src/`
take effect immediately — no reinstalling. `scripts/run_demo.py` and
`tests/test_smoke.py` also work **even without** running `pip install -e .`
first: they add `src/` to `sys.path` themselves at the top of the file.
This is exactly the fix for the `ModuleNotFoundError: No module named
'synthetic_data'` / `PYTHONPATH` problem from your earlier session — you
no longer need `$env:PYTHONPATH=...` at all.

## Run the demo

```bash
python scripts/run_demo.py
```

Outputs always land in the project root's `plots/`, `outputs/`, and
`models_store/` folders, regardless of which directory you run the
command from.

## Open the CSV dashboard

The dashboard displays the saved synthetic dataset, component-level actual
and predicted 168h values, error metrics, per-lot validation scores, and
SAFE / REVIEW / REJECT details. It does not train a model or make new
predictions. Generate or refresh the CSVs first, then install the optional
dashboard dependency and launch it:

```bash
python scripts/run_demo.py
pip install -e ".[app]"
streamlit run app.py
```

The default forecast view uses components held out from point-model fitting
and selection. The separate LOLO view is labeled as model-selection
validation because those lot scores are also used to select the model.

## Run the smoke test after editing anything

```bash
python tests/test_smoke.py
```

Catches import breakage, target leakage into features, and group
cross-validation problems in a few seconds, before you dig into a real
traceback.

## Everyday imports

```python
from ess_predictor.data.synthetic_data import generate_synthetic_dataset
from ess_predictor.data.validation import validate_dataset
from ess_predictor.pipeline import train_all_parameters, predict_component, save_system, load_system
from ess_predictor.reporting.report import build_parameter_block, format_component_report
import ess_predictor.visualization.plots as viz
```

Because this is a real installed package, these imports work identically
whether you're in a Jupyter notebook, a script anywhere on disk, or —
later — inside a web backend.

## Editing / extending

- **Add a 7th electrical parameter**: add one `ParameterConfig` entry to
  `src/ess_predictor/config.py`. Nothing else needs to change — feature
  engineering, models, safety logic, and reports all iterate over the
  parameter list generically.
- **Adjust safety thresholds**: also in `config.py` — `relative_drift_threshold`
  and `absolute_safety_limit` per parameter. These were deliberately kept
  out of the decision logic itself.
- **Swap in real data**: replace `generate_synthetic_dataset(...)` with your
  real DataFrame (same wide-column schema: `{Param}_{timepoint}h`,
  `Component_ID`, `Lot_ID`, `Device_Type`, `Temperature`) and run
  `validate_dataset()` on it first — it tells you exactly which parameters
  are usable without guessing.

## Wiring this into a website later

When you get to building the website:
- A backend (FastAPI/Flask) endpoint can simply do
  `from ess_predictor.pipeline import load_system, predict_component`
  and call `predict_component(...)` per request — the package doesn't
  care whether its caller is a CLI script or a web request handler.
- Keep `models_store/trained_system.pkl` as the artifact your backend
  loads once at startup (`load_system(path)`), rather than retraining per
  request.
- The plots in `visualization/plots.py` currently save PNGs to disk; for
  a web UI you'll likely want a variant that returns the same data (e.g.
  as JSON or base64) instead of writing files — that's a natural next
  module to add under `visualization/` (e.g. `visualization/api_charts.py`)
  without touching anything else.

## Caveats (unchanged from before)

- Uses **synthetic data only** for development/testing (`Data_Source =
  "SYNTHETIC"` is stamped on every generated row). Not validated against
  real NASA PCoE or fab data.
- Thresholds in `config.py` are placeholders — replace with real
  datasheet/reliability-engineering limits before any operational use.
