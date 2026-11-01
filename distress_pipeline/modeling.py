"""
Rolling-origin backtest, baselines and metrics for the next-month Class C target.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from .features import SERIES, WINDOWS

HISTORY = [f"{s}_{w}m" for s in SERIES for w in WINDOWS] + ["months_since_c"]
STATIC = ["units_res", "yearbuilt", "numfloors", "borough", "public_owner"]
FEATURE_SETS = {
    # The original repo's four features, computed correctly on the dense panel.
    "original4": ["all311_6m", "viol_c_6m", "units_res"],
    "history": HISTORY + ["units_res"],
    "full": HISTORY + STATIC,
}

# 4 quarterly refits covering 2021-10 .. 2022-09. A fold trains on the 24 feature
# months ending the month before its first test month; that row's label is that
# month's violations, which are known by prediction time.
TEST_QUARTERS = ["2021-10", "2022-01", "2022-04", "2022-07"]
TRAIN_MONTHS = 24
TOP_FRAC = 0.01


def xgb(**kw) -> XGBClassifier:
    params = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=0)
    return XGBClassifier(**(params | kw))


def logreg() -> object:
    return make_pipeline(
        FunctionTransformer(lambda x: np.log1p(np.clip(np.nan_to_num(x), 0, None))),
        StandardScaler(),
        LogisticRegression(max_iter=500),
    )


def backtest(df: pd.DataFrame, cols: list[str] | None, make_model: Callable | None,
             score_col: str | None = None) -> pd.DataFrame:
    """Return out-of-sample scores for every test row. A model or a raw score column."""
    out = []
    for q in TEST_QUARTERS:
        start = pd.Timestamp(q)
        test = df[(df["month"] >= start) & (df["month"] < start + pd.DateOffset(months=3))]
        if score_col:
            score = test[score_col].to_numpy(dtype=float)
        else:
            train = df[(df["month"] < start)
                       & (df["month"] >= start - pd.DateOffset(months=TRAIN_MONTHS))]
            model = make_model().fit(train[cols], train["target"])
            score = model.predict_proba(test[cols])[:, 1]
        out.append(test[["bbl", "month", "target", "borough", "units_res", "viol_c_12m"]]
                   .assign(score=score))
    return pd.concat(out, ignore_index=True)


def top_k(g: pd.DataFrame, frac: float = TOP_FRAC) -> pd.Series:
    """Precision/recall when inspecting the top `frac` of buildings in a month."""
    k = max(1, int(round(len(g) * frac)))
    # Random tie-break so coarse scores (e.g. counts) are not flattered by row order.
    rng = np.random.default_rng(0)
    order = np.lexsort((rng.random(len(g)), -g["score"].to_numpy()))
    hits = g["target"].to_numpy()[order[:k]].sum()
    return pd.Series({"precision": hits / k, "recall": hits / max(g["target"].sum(), 1)})


def evaluate(pred: pd.DataFrame) -> dict:
    by_month = pred.groupby("month")
    tk = by_month.apply(top_k, include_groups=False)
    y, s = pred["target"], pred["score"]
    return {
        "pos_rate": y.mean(),
        "pr_auc": average_precision_score(y, s),
        "roc_auc": roc_auc_score(y, s),
        "pr_auc_month_sd": by_month.apply(
            lambda g: average_precision_score(g["target"], g["score"]), include_groups=False).std(),
        "prec@1%": tk["precision"].mean(),
        "recall@1%": tk["recall"].mean(),
        "brier": brier_score_loss(y, s) if s.between(0, 1).all() else np.nan,
    }


def calibration_table(pred: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pred["score"].rank(method="first"), bins, labels=False)
    return pred.groupby(q).agg(mean_pred=("score", "mean"), obs_rate=("target", "mean"),
                               n=("target", "size"))
