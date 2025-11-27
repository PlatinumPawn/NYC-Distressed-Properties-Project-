"""
Feature engineering helpers for the NYC Real Estate Distress pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

from . import config


def pick_date_column(parquet_dir: Path, candidates: Iterable[str]) -> str:
    """
    Identify which date column exists in the Parquet batches by inspecting the schema.
    We expect ingestion to be consistent, so the first batch is sufficient.
    """
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {parquet_dir}")

    schema_fields = set(pq.ParquetFile(files[0]).schema.names)
    for candidate in candidates:
        if candidate in schema_fields:
            return candidate
    raise ValueError(f"No date column from {candidates} found in {parquet_dir}")


def aggregate_monthly_counts(
    parquet_dir: Path,
    candidate_date_columns: Iterable[str],
    class_filter: Optional[Tuple[str, str]] = None,
) -> pd.DataFrame:
    """
    Roll up batch Parquet files into monthly counts per BBL.
    Uses an in-memory counter to avoid holding the full raw dataset.
    """
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Directory not found: {parquet_dir}")

    date_col = pick_date_column(parquet_dir, candidate_date_columns)
    counter: Dict[Tuple[int, pd.Timestamp], int] = {}

    for batch_path in sorted(parquet_dir.glob("*.parquet")):
        batch_df = pd.read_parquet(
            batch_path,
            columns=["bbl", date_col]
            + ([class_filter[0]] if class_filter else []),
        )

        batch_df["bbl"] = pd.to_numeric(batch_df["bbl"], errors="coerce")
        batch_df[date_col] = pd.to_datetime(batch_df[date_col], errors="coerce")
        batch_df = batch_df.dropna(subset=["bbl", date_col])

        if class_filter:
            class_col, target_value = class_filter
            if class_col in batch_df.columns:
                batch_df[class_col] = batch_df[class_col].astype(str).str.upper()
                batch_df = batch_df[batch_df[class_col] == target_value.upper()]

        if batch_df.empty:
            continue

        batch_df["bbl"] = batch_df["bbl"].astype("int64")
        batch_df["month"] = batch_df[date_col].dt.to_period("M").dt.to_timestamp()

        grouped = batch_df.groupby(["bbl", "month"]).size()
        for (bbl, month), cnt in grouped.items():
            counter[(bbl, month)] = counter.get((bbl, month), 0) + int(cnt)

    if not counter:
        return pd.DataFrame(columns=["bbl", "month", "count"])

    rows = [{"bbl": k[0], "month": k[1], "count": v} for k, v in counter.items()]
    return pd.DataFrame(rows)


def load_pluto(pluto_path: Path) -> pd.DataFrame:
    """Load PLUTO and keep only BBL + residential units for deterministic joins."""
    cols = ["bbl", "borough", "unitsres", "unitstotal"]
    available_cols = pd.read_csv(pluto_path, nrows=0).columns
    df = pd.read_csv(pluto_path, usecols=[c for c in cols if c in available_cols])
    df["bbl"] = pd.to_numeric(df["bbl"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["bbl"])
    df = df.rename(columns={"unitsres": "units_res", "unitstotal": "units_total"})
    df["units_res"] = pd.to_numeric(df["units_res"], errors="coerce")
    df["units_total"] = pd.to_numeric(df["units_total"], errors="coerce")
    df = df[df["units_res"] > 0]
    df["bbl"] = df["bbl"].astype("int64")
    return df


def build_feature_table(
    complaints_monthly: pd.DataFrame,
    violations_monthly: pd.DataFrame,
    pluto_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build per-BBL monthly feature set with 6-month lookback density metrics and
    a next-month Class C violation target.
    """
    features = complaints_monthly.rename(columns={"count": "complaints"}).merge(
        violations_monthly.rename(columns={"count": "class_c_violations"}),
        on=["bbl", "month"],
        how="outer",
    )

    features = features.fillna(0)
    features = features.merge(pluto_df[["bbl", "units_res"]], on="bbl", how="inner")
    features = features[features["units_res"] > 0]

    features = features.sort_values(["bbl", "month"])

    grouped_complaints = features.groupby("bbl")["complaints"]
    grouped_class_c = features.groupby("bbl")["class_c_violations"]

    # Use strictly prior months for rolling windows to avoid leakage.
    features["complaints_6m"] = (
        grouped_complaints.shift(1)
        .rolling(window=6, min_periods=6)
        .sum()
        .reset_index(level=0, drop=True)
    )
    features["class_c_violations_6m"] = (
        grouped_class_c.shift(1)
        .rolling(window=6, min_periods=6)
        .sum()
        .reset_index(level=0, drop=True)
    )

    features["complaints_per_unit_6m"] = features["complaints_6m"] / features["units_res"]
    features["class_c_per_unit_6m"] = features["class_c_violations_6m"] / features["units_res"]

    # Target: at least one Class C violation in the following month (drop rows where we don't know).
    features["target_next_month"] = (
        features.groupby("bbl")["class_c_violations"].shift(-1).gt(0)
    )

    # Drop rows without a full 6-month history or without a future month to predict.
    features = features.dropna(
        subset=[
            "complaints_6m",
            "class_c_violations_6m",
            "complaints_per_unit_6m",
            "class_c_per_unit_6m",
            "target_next_month",
        ]
    )
    return features


def run_feature_build(pluto_path: Path) -> pd.DataFrame:
    """Aggregate raw Parquet batches and build the model-ready feature table."""
    complaints_monthly = aggregate_monthly_counts(
        parquet_dir=config.RAW_DIR / "311",
        candidate_date_columns=["created_date"],
    )

    violations_monthly = aggregate_monthly_counts(
        parquet_dir=config.RAW_DIR / "hpd",
        candidate_date_columns=[
            "approveddate",
            "novissueddate",
            "inspectiondate",
            "currentstatusdate",
        ],
        class_filter=("class", "C"),
    )

    pluto_df = load_pluto(pluto_path)

    features = build_feature_table(complaints_monthly, violations_monthly, pluto_df)

    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = config.FEATURE_DIR / "monthly_features.parquet"
    features.to_parquet(feature_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Persisted features to {feature_path}")

    return features
