#!/usr/bin/env python3
"""
Run every model on the same rolling-origin backtest and write results to runs/.

Usage: .venv/bin/python scripts/backtest.py
Requires data/features/panel_features.parquet (scripts/distress_pipeline.py --stage features).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distress_pipeline import config  # noqa: E402
from distress_pipeline.modeling import (  # noqa: E402
    FEATURE_SETS, TEST_QUARTERS, TRAIN_MONTHS, backtest, calibration_table, evaluate,
    logreg, xgb,
)
from sklearn.metrics import average_precision_score  # noqa: E402
from xgboost import DMatrix  # noqa: E402

OUT = ROOT / "runs"


def main() -> None:
    df = pd.read_parquet(config.FEATURE_DIR / "panel_features.parquet")
    df["all311_6m_per_unit"] = df["all311_6m"] / df["units_res"]
    df["viol_c_6m_per_unit"] = df["viol_c_6m"] / df["units_res"]
    original = ["all311_6m", "viol_c_6m", "all311_6m_per_unit", "viol_c_6m_per_unit"]

    experiments = {
        "heuristic: C violations last month": (None, None, "viol_c_1m"),
        "heuristic: C violations last 12m": (None, None, "viol_c_12m"),
        "logreg (history + units)": (FEATURE_SETS["history"], logreg, None),
        "xgb original 4 features (orig. params)": (
            original, lambda: xgb(n_estimators=300, max_depth=4, learning_rate=0.1), None),
        "xgb history + units": (FEATURE_SETS["history"], xgb, None),
        "xgb history + PLUTO static": (FEATURE_SETS["full"], xgb, None),
        "xgb full + scale_pos_weight": (FEATURE_SETS["full"], lambda: xgb(scale_pos_weight=10), None),
    }

    rows, preds = [], {}
    for name, (cols, make, score_col) in experiments.items():
        preds[name] = backtest(df, cols, make, score_col)
        rows.append({"model": name} | evaluate(preds[name]))
        print(pd.DataFrame(rows[-1:]).round(4).to_string(index=False), flush=True)
    results = pd.DataFrame(rows).set_index("model")

    best = preds["xgb history + PLUTO static"]
    test_months = sorted(best["month"].unique())
    lines = [
        f"Backtest: {len(TEST_QUARTERS)} quarterly refits, train window {TRAIN_MONTHS} months, "
        f"test months {pd.Timestamp(test_months[0]):%Y-%m}..{pd.Timestamp(test_months[-1]):%Y-%m} "
        f"({len(test_months)} months, {len(best):,} building-months).",
        "Metrics pooled over all test rows; prec/recall@1% computed per month then averaged.",
        "", results.round(4).to_string(), "",
        "## Calibration (xgb history + PLUTO static, test deciles)", "",
        calibration_table(best).round(4).to_string(), "",
    ]

    # Feature attribution: mean |SHAP| from XGBoost's native TreeSHAP on the last fold.
    cols = FEATURE_SETS["full"]
    last = pd.Timestamp(TEST_QUARTERS[-1])
    train = df[(df["month"] < last) & (df["month"] >= last - pd.DateOffset(months=TRAIN_MONTHS))]
    model = xgb().fit(train[cols], train["target"])
    sample = df[df["month"] >= last].sample(50_000, random_state=0)
    contrib = model.get_booster().predict(
        DMatrix(sample[cols]), pred_contribs=True)[:, :-1]
    shap = pd.Series(np.abs(contrib).mean(axis=0), index=cols).sort_values(ascending=False)
    lines += ["## Mean |SHAP| (last fold, 50k test rows sample)", "",
              shap.head(15).round(4).to_frame("mean_abs_shap").to_string(), ""]

    # Error analysis: where does top-1% targeting work, and who is missed?
    best = best.assign(
        cold=np.where(best["viol_c_12m"] == 0, "no C in prior 12m", "C in prior 12m"),
        size=pd.cut(best["units_res"], [0, 2, 5, 19, 99, np.inf],
                    labels=["1-2", "3-5", "6-19", "20-99", "100+"]))
    for col in ["cold", "size", "borough"]:
        seg = best.groupby(col, observed=True).agg(
            rows=("target", "size"), pos_rate=("target", "mean"),
            share_of_positives=("target", "sum"))
        seg["share_of_positives"] /= best["target"].sum()
        seg["pr_auc"] = best.groupby(col, observed=True).apply(
            evaluate_ap, include_groups=False)
        lines += [f"## Segment: {col}", "", seg.round(4).to_string(), ""]
    flagged = best.groupby("month", group_keys=False).apply(
        lambda g: g.nlargest(max(1, round(len(g) * 0.01)), "score"))
    lines += [f"Top-1% flagged rows with no C in prior 12m: "
              f"{(flagged['viol_c_12m'] == 0).mean():.4f}", ""]

    (OUT / "backtest_results.md").write_text("\n".join(lines))
    results.to_csv(OUT / "backtest_results.csv")
    print("\n".join(lines))


def evaluate_ap(g: pd.DataFrame) -> float:
    return average_precision_score(g["target"], g["score"]) if g["target"].nunique() > 1 else np.nan


if __name__ == "__main__":
    main()
