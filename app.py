"""Streamlit dashboard for the synthetic demo's CSV outputs."""

from pathlib import Path
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
CSV_FILES = {
    "Synthetic dataset": "synthetic_dataset.csv",
    "Forecast predictions": "forecast_predictions.csv",
    "Lot validation scores": "leave_one_lot_out_mae.csv",
    "Component dispositions": "component_dispositions.csv",
}
REQUIRED_FORECAST_COLUMNS = {
    "Component_ID", "Lot_ID", "Parameter", "Model", "Prediction_Basis",
    "Value_0h", "Value_24h", "Actual_168h", "Predicted_168h",
}


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_csv(key: str) -> Optional[pd.DataFrame]:
    path = OUTPUTS / CSV_FILES[key]
    if not path.exists():
        return None
    try:
        return read_csv(str(path))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read `{path.name}`: {exc}")
        return None


def metric_summary(frame: pd.DataFrame) -> dict[str, float]:
    actual = pd.to_numeric(frame["Actual_168h"], errors="coerce").to_numpy(float)
    predicted = pd.to_numeric(frame["Predicted_168h"], errors="coerce").to_numpy(float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if not len(actual):
        return {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "R²": np.nan}
    error = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 0 else np.nan
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "Bias": float(np.mean(error)),
        "R²": r2,
    }


def safe_float(value) -> float:
    return float(value) if pd.notna(value) else np.nan


st.set_page_config(page_title="SIH170 Drift Forecast Viewer", page_icon="📈", layout="wide")
st.title("SIH170 · 168 h Drift Forecasts")
st.caption("Browse saved measurements, forecasts, validation scores, and decisions.")
st.warning("Synthetic demo data only; not measured or production-validated.", icon="⚠️")

dataset = load_csv("Synthetic dataset")
forecasts = load_csv("Forecast predictions")
lot_scores = load_csv("Lot validation scores")
dispositions = load_csv("Component dispositions")

missing = [name for name, frame in (("synthetic_dataset.csv", dataset), ("forecast_predictions.csv", forecasts)) if frame is None]
if missing:
    st.info(
        "Missing output files: " + ", ".join(missing)
        + ". Run `python scripts/run_demo.py` from the project folder, then refresh."
    )
    st.stop()

if not REQUIRED_FORECAST_COLUMNS.issubset(forecasts.columns):
    absent = sorted(REQUIRED_FORECAST_COLUMNS - set(forecasts.columns))
    st.error("Forecast CSV is missing columns: " + ", ".join(absent))
    st.stop()

parameters = sorted(forecasts["Parameter"].dropna().astype(str).unique())
bases = forecasts["Prediction_Basis"].dropna().astype(str).unique().tolist()
held_out_label = "Held-out component split (not used to fit/select point model)"
default_basis = held_out_label if held_out_label in bases else (bases[0] if bases else "")

with st.sidebar:
    st.header("View controls")
    selected_parameter = st.selectbox("Parameter", parameters)
    selected_basis = st.selectbox(
        "Prediction set",
        bases,
        index=bases.index(default_basis) if default_basis in bases else 0,
        help="Held-out components were excluded from fitting and model selection. LOLO scores were used to choose the model.",
    )
    st.divider()
    st.caption(f"Data folder: `{OUTPUTS}`")

filtered = forecasts.loc[
    (forecasts["Parameter"].astype(str) == selected_parameter)
    & (forecasts["Prediction_Basis"].astype(str) == selected_basis)
].copy()

tabs = st.tabs(["Overview", "Actual vs predicted", "Component explorer", "Lot validation", "Dispositions", "CSV browser"])

with tabs[0]:
    st.subheader("Run overview")
    cols = st.columns(4)
    profile = "unknown"
    if "Synthetic_Profile" in dataset.columns and dataset["Synthetic_Profile"].notna().any():
        profiles = dataset["Synthetic_Profile"].dropna().astype(str).unique()
        profile = ", ".join(profiles)
    cols[0].metric("Synthetic profile", profile)
    cols[1].metric("Components", f"{dataset['Component_ID'].nunique():,}" if "Component_ID" in dataset else f"{len(dataset):,}")
    cols[2].metric("Lots", f"{dataset['Lot_ID'].nunique():,}" if "Lot_ID" in dataset else "—")
    cols[3].metric("Forecast rows", f"{len(filtered):,}")

    if not filtered.empty:
        summary = metric_summary(filtered)
        unit = ""
        unit_map = {"Iddq": "uA", "Leakage": "uA", "PropagationDelay": "ns", "Icc": "mA", "Vth": "V", "RdsOn": "mOhm"}
        unit = unit_map.get(selected_parameter, "")
        mcols = st.columns(4)
        mcols[0].metric("MAE", f"{summary['MAE']:.4g} {unit}".strip())
        mcols[1].metric("RMSE", f"{summary['RMSE']:.4g} {unit}".strip())
        mcols[2].metric("Mean signed error", f"{summary['Bias']:+.4g} {unit}".strip())
        mcols[3].metric("R²", f"{summary['R²']:.3f}" if np.isfinite(summary["R²"]) else "—")

        baseline_rows = []
        for label, column in (
            ("Last value (24 h)", "Baseline_LastValue24h"),
            ("Linear extrapolation", "Baseline_LinearExtrapolation"),
        ):
            if column in filtered:
                actual = pd.to_numeric(filtered["Actual_168h"], errors="coerce")
                estimate = pd.to_numeric(filtered[column], errors="coerce")
                valid = actual.notna() & estimate.notna()
                if valid.any():
                    baseline_rows.append({"Reference forecast": label, "MAE": float((actual[valid] - estimate[valid]).abs().mean())})
        if baseline_rows:
            baseline_rows.insert(0, {"Reference forecast": "Selected model", "MAE": summary["MAE"]})
            st.markdown("**MAE against simple reference forecasts**")
            st.dataframe(pd.DataFrame(baseline_rows).style.format({"MAE": "{:.4g}"}), hide_index=True, use_container_width=True)

        st.caption(f"Model: {', '.join(filtered['Model'].dropna().astype(str).unique()) or '—'} · Prediction set: {selected_basis}")
        if "Synthetic_Profile" in dataset.columns:
            st.caption("Profile is read from the synthetic dataset CSV.")
    else:
        st.info("No forecasts exist for this parameter and prediction set.")

    summary_path = OUTPUTS / "screening_summary.txt"
    if summary_path.exists():
        with st.expander("Overall SAFE / REVIEW / REJECT counts"):
            st.code(summary_path.read_text(encoding="utf-8"), language="text")

with tabs[1]:
    st.subheader(f"{selected_parameter}: actual and predicted at 168 h")
    if filtered.empty:
        st.info("No rows to chart for this selection.")
    else:
        chart = filtered[["Component_ID", "Lot_ID", "Actual_168h", "Predicted_168h"]].copy()
        chart[["Actual_168h", "Predicted_168h"]] = chart[
            ["Actual_168h", "Predicted_168h"]
        ].apply(pd.to_numeric, errors="coerce")
        chart = chart.dropna(subset=["Actual_168h", "Predicted_168h"])
        low = float(min(chart["Actual_168h"].min(), chart["Predicted_168h"].min()))
        high = float(max(chart["Actual_168h"].max(), chart["Predicted_168h"].max()))
        points = alt.Chart(chart).mark_circle(size=65, opacity=0.7).encode(
            x=alt.X("Actual_168h:Q", title="Actual 168 h"),
            y=alt.Y("Predicted_168h:Q", title="Predicted 168 h"),
            tooltip=[
                alt.Tooltip("Component_ID:N", title="Component"),
                alt.Tooltip("Lot_ID:N", title="Lot"),
                alt.Tooltip("Actual_168h:Q", title="Actual 168 h", format=".4g"),
                alt.Tooltip("Predicted_168h:Q", title="Predicted 168 h", format=".4g"),
            ],
        )
        ideal = alt.Chart(pd.DataFrame({"x": [low, high], "y": [low, high]})).mark_line(
            color="#888888", strokeDash=[6, 4]
        ).encode(x="x:Q", y="y:Q")
        st.altair_chart((points + ideal).interactive(), use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.markdown("**Error distribution**")
            errors = pd.to_numeric(filtered["Prediction_Error"], errors="coerce").dropna() if "Prediction_Error" in filtered else filtered["Predicted_168h"] - filtered["Actual_168h"]
            counts, edges = np.histogram(errors.to_numpy(float), bins=20)
            histogram = pd.DataFrame({
                "Error range": [f"{edges[i]:.3g} to {edges[i + 1]:.3g}" for i in range(len(counts))],
                "Components": counts,
            })
            st.bar_chart(histogram, x="Error range", y="Components", use_container_width=True)
        with right:
            st.markdown("**Largest absolute errors**")
            sort_col = "Absolute_Error" if "Absolute_Error" in filtered else None
            largest = filtered.sort_values(sort_col, ascending=False) if sort_col else filtered.assign(_abs=(filtered["Predicted_168h"] - filtered["Actual_168h"]).abs()).sort_values("_abs", ascending=False)
            display_cols = [c for c in ["Component_ID", "Lot_ID", "Actual_168h", "Predicted_168h", "Prediction_Error", "Absolute_Error"] if c in largest]
            st.dataframe(largest[display_cols].head(15), hide_index=True, use_container_width=True)
        st.caption("Each dot is a component; the dashed line marks a perfect prediction.")

with tabs[2]:
    st.subheader("Inspect one component")
    if filtered.empty:
        st.info("No component forecasts to inspect.")
    else:
        components = sorted(filtered["Component_ID"].dropna().astype(str).unique())
        component_id = st.selectbox("Component", components, key="component_explorer_id")
        forecast_row = filtered.loc[filtered["Component_ID"].astype(str) == component_id].iloc[0]
        source_rows = dataset.loc[dataset["Component_ID"].astype(str) == component_id] if "Component_ID" in dataset else pd.DataFrame()
        source = source_rows.iloc[0] if not source_rows.empty else pd.Series(dtype=object)
        details = {
            "Component": component_id,
            "Lot": forecast_row.get("Lot_ID", "—"),
            "Model": forecast_row.get("Model", "—"),
            "0 h reading": forecast_row.get("Value_0h", np.nan),
            "24 h reading": forecast_row.get("Value_24h", np.nan),
            "Actual 168 h": forecast_row.get("Actual_168h", np.nan),
            "Predicted 168 h": forecast_row.get("Predicted_168h", np.nan),
            "24 h carry-forward baseline": forecast_row.get("Baseline_LastValue24h", np.nan),
            "Linear extrapolation baseline": forecast_row.get("Baseline_LinearExtrapolation", np.nan),
            "Prediction error": forecast_row.get("Prediction_Error", safe_float(forecast_row.get("Predicted_168h")) - safe_float(forecast_row.get("Actual_168h"))),
        }
        st.dataframe(pd.DataFrame([details]), hide_index=True, use_container_width=True)

        timepoints = ["0h", "24h", "96h", "168h"]
        trajectory = {}
        for timepoint in timepoints:
            column = f"{selected_parameter}_{timepoint}"
            if column in source.index:
                trajectory[f"Actual · {timepoint}"] = safe_float(source[column])
        if trajectory:
            trajectory_frame = pd.DataFrame({"Actual": [trajectory.get(f"Actual · {t}", np.nan) for t in timepoints]}, index=timepoints)
            trajectory_frame["Predicted 168 h"] = [np.nan, np.nan, np.nan, safe_float(forecast_row["Predicted_168h"])]
            st.markdown("**Measured synthetic trajectory and 168 h forecast**")
            st.line_chart(trajectory_frame, use_container_width=True)
            st.caption("The 168 h actual value is the synthetic target; the second series is the forecast.")

        if "Prediction_Error" in forecast_row and pd.notna(forecast_row["Prediction_Error"]):
            st.metric("Absolute error", f"{abs(float(forecast_row['Prediction_Error'])):.5g}")

with tabs[3]:
    st.subheader("Leave-one-lot-out validation")
    if lot_scores is None:
        st.info("`leave_one_lot_out_mae.csv` is not available yet. Run the demo to create it.")
    else:
        lot_view = lot_scores.loc[lot_scores["Parameter"].astype(str) == selected_parameter].copy()
        if not lot_view.empty:
            st.caption("Each score comes from a held-out lot. These same scores select the model, so they are not an independent benchmark.")
            model_names = sorted(lot_view["Model"].dropna().astype(str).unique())
            chosen_models = st.multiselect("Models", model_names, default=model_names, key="lot_models")
            lot_view = lot_view.loc[lot_view["Model"].astype(str).isin(chosen_models)]
            st.dataframe(lot_view.sort_values(["Model", "Held_Out_Lot"]), hide_index=True, use_container_width=True)
            if "MAE" in lot_view:
                st.markdown("**MAE by held-out lot**")
                pivot = lot_view.pivot_table(index="Held_Out_Lot", columns="Model", values="MAE", aggfunc="first")
                st.line_chart(pivot, use_container_width=True)
        else:
            st.info("No lot-validation rows for this parameter.")

with tabs[4]:
    st.subheader("Component dispositions")
    if dispositions is None:
        st.info("`component_dispositions.csv` is not available yet. Run the demo to create it.")
    else:
        status_col = "Overall_Decision" if "Overall_Decision" in dispositions else "Parameter_Decision"
        statuses = [s for s in ["SAFE", "REVIEW", "REJECT"] if s in set(dispositions[status_col].dropna().astype(str))]
        selected_status = st.multiselect("Show overall status", statuses, default=statuses)
        listed = dispositions.loc[dispositions[status_col].astype(str).isin(selected_status)].copy()
        if "Parameter" in listed:
            listed = listed.loc[listed["Parameter"].astype(str) == selected_parameter]
        st.caption("There is one row per component and parameter. Overall status repeats across a component's rows.")
        visible_cols = [c for c in ["Component_ID", "Lot_ID", "Overall_Decision", "Parameter", "Parameter_Decision", "Value_0h", "Value_24h", "Predicted_168h", "Interval_Low", "Interval_High", "Predicted_Drift_Pct", "Drift_Limit_Pct", "Reasons"] if c in listed]
        st.dataframe(listed[visible_cols], hide_index=True, use_container_width=True)
        if "Overall_Decision" in dispositions:
            unique_components = dispositions.drop_duplicates("Component_ID")
            counts = unique_components["Overall_Decision"].value_counts()
            ccols = st.columns(3)
            for col, status in zip(ccols, ("SAFE", "REVIEW", "REJECT")):
                col.metric(status, f"{int(counts.get(status, 0)):,}")

with tabs[5]:
    st.subheader("Browse source CSVs")
    available = {name: load_csv(name) for name in CSV_FILES if (OUTPUTS / CSV_FILES[name]).exists()}
    csv_name = st.selectbox("CSV", list(available), key="raw_csv_choice")
    raw = available[csv_name]
    if raw is not None:
        st.caption(f"{len(raw):,} rows · {len(raw.columns):,} columns")
        if "Component_ID" in raw.columns:
            ids = sorted(raw["Component_ID"].dropna().astype(str).unique())
            component_filter = st.text_input("Filter by component ID (optional)", placeholder="e.g. C0016")
            if component_filter.strip():
                raw = raw.loc[raw["Component_ID"].astype(str).str.contains(component_filter.strip(), case=False, regex=False)]
        st.dataframe(raw.head(5000), hide_index=True, use_container_width=True)
        st.download_button(
            "Download displayed CSV",
            data=raw.to_csv(index=False).encode("utf-8"),
            file_name=CSV_FILES[csv_name],
            mime="text/csv",
        )
