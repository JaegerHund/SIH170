"""Check input columns and values before model training."""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from ess_predictor.config import PARAMETERS, EARLY_TIMEPOINTS, TARGET_TIMEPOINT, ParameterConfig

REQUIRED_ID_COLUMNS = ["Component_ID"]
RECOMMENDED_ID_COLUMNS = ["Lot_ID", "Device_Type", "Temperature"]


@dataclass
class ValidationReport:
    usable_parameters: List[ParameterConfig] = field(default_factory=list)
    missing_parameters: List[str] = field(default_factory=list)
    missing_id_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_rows_in: int = 0
    n_rows_after_dropna_target: Dict[str, int] = field(default_factory=dict)
    data_source: str = "unspecified"
    synthetic_profile: str = ""

    def summary(self) -> str:
        lines = ["DATASET VALIDATION SUMMARY", "=" * 72]
        lines.append(f"Rows received: {self.n_rows_in:,}")
        source = self.data_source
        if self.synthetic_profile:
            source += f" (profile: {self.synthetic_profile})"
        lines.append(f"Data source:   {source}")
        lines.append("")
        lines.append(
            f"Usable parameters: {len(self.usable_parameters)} / "
            f"{len(self.usable_parameters) + len(self.missing_parameters)}"
        )
        if self.usable_parameters:
            name_width = max(9, *(len(p.name) for p in self.usable_parameters))
            lines.append(
                f"  {'Parameter':<{name_width}}  {'Early inputs':<16}  "
                f"{'168h target rows':>16}  {'Training coverage':>17}"
            )
            lines.append(
                f"  {'-' * name_width}  {'-' * 16}  {'-' * 16}  {'-' * 17}"
            )
            for p in self.usable_parameters:
                n_target = self.n_rows_after_dropna_target.get(p.name, 0)
                coverage = f"{n_target / self.n_rows_in:.0%}" if self.n_rows_in else "—"
                target_rows = f"{n_target:,}" if n_target else "unavailable"
                lines.append(
                    f"  {p.name:<{name_width}}  {'0h + 24h':<16}  "
                    f"{target_rows:>16}  {coverage:>17}"
                )
        if self.missing_parameters:
            lines.append("")
            lines.append(
                "Skipped parameters (missing required 0h/24h columns): "
                + ", ".join(self.missing_parameters)
            )
        if self.missing_id_columns:
            lines.append("")
            lines.append(
                "Missing recommended ID columns: "
                + ", ".join(self.missing_id_columns)
                + " (lot normalization or group validation may be limited)"
            )
        if self.warnings:
            lines.extend(["", f"Warnings ({len(self.warnings)}):"])
            lines.extend(f"  {i}. {warning}" for i, warning in enumerate(self.warnings, 1))
        else:
            lines.extend(["", "Warnings: none"])
        lines.append("=" * 72)
        return "\n".join(lines)


def _param_columns(param_name: str, timepoints):
    return [f"{param_name}_{tp}h" for tp in timepoints]


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """
    Inspect `df` and determine which parameters have enough columns
    (0h, 24h required; 168h required as the training target) to be used.
    Does not mutate df. Does not raise on missing optional pieces --
    only raises on truly unusable input (no Component_ID, empty df).
    """
    report = ValidationReport(n_rows_in=len(df))
    if "Data_Source" in df.columns and df["Data_Source"].nunique(dropna=True) == 1:
        report.data_source = str(df["Data_Source"].dropna().iloc[0])
    elif "Data_Source" in df.columns:
        report.data_source = "mixed"
    if "Synthetic_Profile" in df.columns and df["Synthetic_Profile"].nunique(dropna=True) == 1:
        report.synthetic_profile = str(df["Synthetic_Profile"].dropna().iloc[0])
    elif "Synthetic_Profile" in df.columns:
        report.synthetic_profile = "mixed"

    if len(df) == 0:
        raise ValueError("Dataset is empty.")

    for col in REQUIRED_ID_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from dataset.")

    for col in RECOMMENDED_ID_COLUMNS:
        if col not in df.columns:
            report.missing_id_columns.append(col)

    if df["Component_ID"].duplicated().any():
        n_dupe = df["Component_ID"].duplicated().sum()
        report.warnings.append(
            f"{n_dupe} duplicated Component_ID values found; this will corrupt "
            f"group-level splitting. Deduplicate before training."
        )

    for p in PARAMETERS:
        early_cols = _param_columns(p.name, EARLY_TIMEPOINTS)
        target_col = f"{p.name}_{TARGET_TIMEPOINT}h"
        has_early = all(c in df.columns for c in early_cols)
        has_target = target_col in df.columns

        if not has_early:
            report.missing_parameters.append(p.name)
            continue

        report.usable_parameters.append(p)

        if has_target:
            n_valid_target = df[target_col].notna().sum()
            report.n_rows_after_dropna_target[p.name] = int(n_valid_target)
            if n_valid_target < len(df):
                report.warnings.append(
                    f"{p.name}: {len(df) - n_valid_target} rows missing "
                    f"{TARGET_TIMEPOINT}h target -- these rows are usable for "
                    f"inference but NOT for training this parameter's model."
                )
        else:
            report.warnings.append(
                f"{p.name}: no {TARGET_TIMEPOINT}h column found. This parameter "
                f"can still be used for INFERENCE (0h/24h available) but cannot "
                f"be trained/validated without a target column."
            )
            report.n_rows_after_dropna_target[p.name] = 0

        # Sanity checks on early values
        for c in early_cols:
            n_neg = (df[c] < 0).sum()
            if n_neg > 0:
                report.warnings.append(
                    f"{c}: {n_neg} negative values found -- check units/sign convention."
                )
            n_nan = df[c].isna().sum()
            if n_nan > 0:
                report.warnings.append(f"{c}: {n_nan} missing values.")

    if not report.usable_parameters:
        raise ValueError(
            "No parameter in the dataset has the minimum required 0h and 24h "
            "columns. Cannot proceed."
        )

    return report


def handle_missing_values(df: pd.DataFrame, usable_param_names: List[str]) -> pd.DataFrame:
    """
    Conservative missing-value handling:
      - Rows missing BOTH 0h and 24h for a given parameter are unusable for
        that parameter and are left as NaN (excluded downstream, not imputed
        -- imputing early measurements would be scientifically indefensible
        for a safety-critical decision).
      - Isolated missing 96h values are fine (96h is optional).
      - We do NOT silently impute the target (168h); rows without it are
        simply excluded from training for that parameter.
    This function mainly documents / enforces the "no silent imputation of
    safety-relevant inputs" policy and returns df unchanged aside from
    dtype coercion.
    """
    df = df.copy()
    numeric_like = [c for c in df.columns if any(
        c.startswith(p + "_") for p in usable_param_names
    )]
    for c in numeric_like:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


if __name__ == "__main__":
    from ess_predictor.data.synthetic_data import generate_synthetic_dataset
    df = generate_synthetic_dataset(200)
    rep = validate_dataset(df)
    print(rep.summary())
