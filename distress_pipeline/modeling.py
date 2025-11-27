"""
Model training and scoring utilities for the NYC Real Estate Distress pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


def time_based_split(
    df: pd.DataFrame, date_col: str = "month", train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by month to preserve temporal order."""
    unique_months = sorted(df[date_col].unique())
    if not unique_months:
        raise ValueError("No months available for splitting.")
    split_idx = max(1, int(len(unique_months) * train_ratio))
    split_month = unique_months[split_idx - 1]
    train_df = df[df[date_col] <= split_month]
    test_df = df[df[date_col] > split_month]
    return train_df, test_df


def train_model(features: pd.DataFrame):
    """Train an XGBoost classifier and return the fitted model and eval metrics."""
    feature_cols = [
        "complaints_per_unit_6m",
        "class_c_per_unit_6m",
        "complaints_6m",
        "class_c_violations_6m",
    ]

    features = features.dropna(subset=feature_cols + ["target_next_month"])

    train_df, test_df = time_based_split(features)

    X_train, y_train = train_df[feature_cols], train_df["target_next_month"]
    X_test, y_test = test_df[feature_cols], test_df["target_next_month"]

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.1,
        max_depth=4,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X_train, y_train)

    test_preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_preds) if len(np.unique(y_test)) > 1 else np.nan
    pr_auc = average_precision_score(y_test, test_preds)
    hard_preds = (test_preds > 0.5).astype(int)
    report = classification_report(y_test, hard_preds, digits=3)
    acc = accuracy_score(y_test, hard_preds)

    metrics = {
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "classification_report": report,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_buildings": int(train_df["bbl"].nunique()),
        "test_buildings": int(test_df["bbl"].nunique()),
        "train_month_start": str(train_df["month"].min()),
        "train_month_end": str(train_df["month"].max()),
        "test_month_start": str(test_df["month"].min()),
        "test_month_end": str(test_df["month"].max()),
    }
    return model, metrics


def save_predictions(
    model: XGBClassifier,
    features: pd.DataFrame,
    output_path: Path,
    scoring_month: Optional[pd.Timestamp] = None,
) -> None:
    """
    Score the latest month (or provided month) and persist probabilities by BBL.
    """
    feature_cols = [
        "complaints_per_unit_6m",
        "class_c_per_unit_6m",
        "complaints_6m",
        "class_c_violations_6m",
    ]

    if scoring_month is None:
        scoring_month = features["month"].max()

    scoring_frame = features[features["month"] == scoring_month].copy()
    scoring_frame["distress_probability"] = model.predict_proba(scoring_frame[feature_cols])[:, 1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scoring_frame[["bbl", "month", "distress_probability"]].to_parquet(
        output_path, engine="pyarrow", compression="snappy", index=False
    )

    print(f"Wrote scoring data for {len(scoring_frame):,} buildings to {output_path}")


def rolling_monthly_cv(
    features: pd.DataFrame,
    min_train_months: int = 6,
    max_folds: Optional[int] = None,
    threshold: float = 0.5,
):
    """
    Expanding-window CV: for each target month (after an initial training window),
    train on all prior months and score only that month. Returns per-fold metrics
    plus overall means (ignoring NaN AUCs when class is single-valued).
    """
    feature_cols = [
        "complaints_per_unit_6m",
        "class_c_per_unit_6m",
        "complaints_6m",
        "class_c_violations_6m",
    ]

    features = features.dropna(subset=feature_cols + ["target_next_month"])

    months = sorted(features["month"].unique())
    if len(months) < min_train_months + 1:
        raise ValueError("Not enough months to run rolling CV with the given min_train_months.")

    target_months = months[min_train_months:]
    if max_folds:
        target_months = target_months[-max_folds:]

    fold_metrics = []
    for target_month in target_months:
        train_df = features[features["month"] < target_month]
        test_df = features[features["month"] == target_month]

        if train_df.empty or test_df.empty:
            continue

        pos = float(train_df["target_next_month"].sum())
        neg = float(len(train_df) - pos)
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        model = XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.1,
            max_depth=4,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            n_jobs=4,
            scale_pos_weight=scale_pos_weight,
        )

        X_train, y_train = train_df[feature_cols], train_df["target_next_month"]
        X_test, y_test = test_df[feature_cols], test_df["target_next_month"]

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan
        pr_auc = average_precision_score(y_test, probs)
        preds = (probs >= threshold).astype(int)
        precision = precision_score(y_test, preds, zero_division=0)
        prevalence = float(y_test.mean())

        fold_metrics.append(
            {
                "month": target_month,
                "auc": auc,
                "pr_auc": pr_auc,
                "precision": precision,
                "prevalence": prevalence,
            }
        )

        month_label = getattr(target_month, "strftime", lambda fmt: str(target_month))("%Y-%m-%d")
        print(
            f"[CV] Target month {month_label}: AUC={auc:.3f} PR-AUC={pr_auc:.3f} "
            f"Precision={precision:.3f} PosRate={prevalence:.3f} n={len(y_test)}"
        )

    auc_values = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]
    pr_values = [m["pr_auc"] for m in fold_metrics if not np.isnan(m["pr_auc"])]
    precision_values = [m["precision"] for m in fold_metrics]

    summary = {
        "folds": fold_metrics,
        "mean_auc": float(np.mean(auc_values)) if auc_values else np.nan,
        "mean_pr_auc": float(np.mean(pr_values)) if pr_values else np.nan,
        "mean_precision": float(np.mean(precision_values)) if precision_values else np.nan,
        "fold_count": len(fold_metrics),
    }

    print(
        f"[CV] Summary over {summary['fold_count']} folds: "
        f"mean AUC={summary['mean_auc']:.3f} mean PR-AUC={summary['mean_pr_auc']:.3f} "
        f"mean Precision={summary['mean_precision']:.3f}"
    )

    return summary
