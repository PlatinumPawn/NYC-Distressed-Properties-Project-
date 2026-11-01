"""
Build a dense BBL x month panel and the model-ready feature table.

Row (bbl, t): features use only data from months <= t; the target is whether the
BBL gets >= 1 Class C HPD violation (by approveddate) in month t+1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config

# The 311 extract on disk ends 2022-10-28, so 2022-09 is the last complete month.
# Using later months would silently zero-fill every 311 feature.
START = pd.Period("2012-11", "M")
END = pd.Period("2022-09", "M")
N_MONTHS = END.ordinal - START.ordinal + 1
MODEL_START = pd.Period("2019-09", "M")  # earliest row any backtest fold trains on
WINDOWS = (1, 3, 6, 12)

# HPD-enforced housing-maintenance complaint types (agency names changed over time).
HEAT_311 = {"HEAT/HOT WATER", "HEATING"}
HOUSING_311 = {
    "UNSANITARY CONDITION", "PLUMBING", "PAINT/PLASTER", "PAINT - PLASTER",
    "GENERAL CONSTRUCTION/PLUMBING", "DOOR/WINDOW", "WATER LEAK", "ELECTRIC",
    "GENERAL", "FLOORING/STAIRS", "GENERAL CONSTRUCTION", "ELEVATOR", "APPLIANCE",
    "SAFETY", "NONCONST", "LEAD", "OUTSIDE BUILDING", "MOLD", "STRUCTURAL",
}
SERIES = ["heat311", "housing311", "other311", "viol_a", "viol_b", "viol_c"]


def _month_index(col: pd.Series) -> pd.Series:
    # Raw dates are a mix of timestamps and ISO strings; the "YYYY-MM" prefix is all we need.
    p = pd.to_datetime(col.astype(str).str[:7], format="%Y-%m", errors="coerce")
    return (p.dt.year - START.year) * 12 + p.dt.month - START.month


def _read_events(parquet_dir: Path, date_col: str, kind_col: str) -> pd.DataFrame:
    """Return (bbl, m, kind) for every raw event, reading one batch file at a time."""
    parts = []
    for path in sorted(parquet_dir.glob("*.parquet")):
        df = pq.read_table(path, columns=["bbl", date_col, kind_col]).to_pandas()
        out = pd.DataFrame({
            "bbl": pd.to_numeric(df["bbl"], errors="coerce"),
            "m": _month_index(df[date_col]),
            "kind": df[kind_col].astype(str).str.upper().str.strip(),
        }).dropna(subset=["bbl", "m"])
        parts.append(out)
    events = pd.concat(parts, ignore_index=True)
    events["bbl"] = events["bbl"].astype("int64")
    events["m"] = events["m"].astype("int64")
    return events


def load_pluto(pluto_path: Path) -> pd.DataFrame:
    """Residential lots with the static attributes used as features."""
    cols = ["bbl", "borough", "unitsres", "yearbuilt", "numfloors", "ownertype"]
    df = pd.read_csv(pluto_path, usecols=cols, low_memory=False)
    df["bbl"] = pd.to_numeric(df["bbl"], errors="coerce")
    df = df.dropna(subset=["bbl"])
    df["bbl"] = df["bbl"].astype("int64")
    df = df[df["unitsres"] > 0].copy()
    df["borough"] = df["borough"].astype("category").cat.codes
    # ownertype is blank for private owners; any code (C/X/O/P/M) means public or exempt.
    df["public_owner"] = df["ownertype"].notna().astype("int8")
    df["yearbuilt"] = df["yearbuilt"].where(df["yearbuilt"] > 1800)
    return df.drop(columns="ownertype").rename(columns={"unitsres": "units_res"})


def build_feature_table(pluto_path: Path) -> pd.DataFrame:
    pluto = load_pluto(pluto_path)

    c311 = _read_events(config.RAW_DIR / "311", "created_date", "complaint_type")
    c311["kind"] = np.select(
        [c311["kind"].isin(HEAT_311), c311["kind"].isin(HOUSING_311)],
        ["heat311", "housing311"], "other311",
    )
    c311 = c311[c311["m"].between(0, N_MONTHS - 1)]

    hpd = _read_events(config.RAW_DIR / "hpd", "approveddate", "class")
    hpd = hpd[hpd["kind"].isin(["A", "B", "C"]) & hpd["m"].between(0, N_MONTHS)]  # +1 month for target
    hpd["kind"] = "viol_" + hpd["kind"].str.lower()

    # Universe: residential PLUTO lots with any HPD violation or housing 311 complaint.
    active = pd.concat([hpd["bbl"], c311.loc[c311["kind"] != "other311", "bbl"]]).unique()
    bbls = np.sort(pluto["bbl"][pluto["bbl"].isin(active)].to_numpy())

    # Dense count tensor [series, bbl, month]; one extra month holds the target.
    counts = np.zeros((len(SERIES), len(bbls), N_MONTHS + 1), dtype=np.int32)
    for ev in (c311, hpd):
        ev = ev[ev["bbl"].isin(bbls)]
        g = ev.groupby(["kind", "bbl", "m"]).size()
        s = np.array([SERIES.index(k) for k in g.index.get_level_values(0)])
        b = np.searchsorted(bbls, g.index.get_level_values(1))
        counts[s, b, g.index.get_level_values(2)] = g.to_numpy()

    hist = counts[:, :, :N_MONTHS]
    csum = np.concatenate([np.zeros_like(hist[:, :, :1]), hist.cumsum(axis=2, dtype=np.int32)], axis=2)

    # Months since the most recent Class C violation (capped at 120 = never seen).
    last_c = np.where(hist[SERIES.index("viol_c")] > 0, np.arange(N_MONTHS), -10_000)
    last_c = np.maximum.accumulate(last_c, axis=1)
    months_since_c = np.minimum(np.arange(N_MONTHS) - last_c, 120)

    # A BBL enters the panel only once it has any record, so membership is causal.
    any_activity = csum[:, :, 1:].sum(axis=0) > 0

    t0 = MODEL_START.ordinal - START.ordinal
    frames = []
    for t in range(t0, N_MONTHS):
        rows = np.flatnonzero(any_activity[:, t])
        f = {"bbl": bbls[rows], "month": (START + t).to_timestamp()}
        for i, name in enumerate(SERIES):
            for w in WINDOWS:
                lo = max(t + 1 - w, 0)
                f[f"{name}_{w}m"] = (csum[i, rows, t + 1] - csum[i, rows, lo]).astype(np.float32)
        f["months_since_c"] = months_since_c[rows, t].astype(np.float32)
        f["target"] = (counts[SERIES.index("viol_c"), rows, t + 1] > 0).astype(np.int8)
        frames.append(pd.DataFrame(f))

    features = pd.concat(frames, ignore_index=True).merge(pluto, on="bbl", how="left")
    features["all311_6m"] = features[["heat311_6m", "housing311_6m", "other311_6m"]].sum(axis=1)
    return features


def run_feature_build(pluto_path: Path) -> pd.DataFrame:
    features = build_feature_table(pluto_path)
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FEATURE_DIR / "panel_features.parquet"
    features.to_parquet(path, index=False)
    print(f"Persisted {len(features):,} rows x {features.shape[1]} cols to {path}")
    return features
