#!/usr/bin/env python3
"""
NYC Real Estate Distress Prediction Engine CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from distress_pipeline import config, features, ingestion, modeling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NYC Real Estate Distress Prediction Engine")
    parser.add_argument(
        "--stage",
        choices=["ingest", "features", "train", "all"],
        default="all",
        help="Stage to run. 'all' executes ingestion -> features -> train.",
    )
    parser.add_argument(
        "--start-date",
        default=config.START_DATE_DEFAULT,
        help="ISO date string for lower bound on created/approved dates.",
    )
    parser.add_argument(
        "--app-token",
        dest="app_token",
        default=None,
        help="NYC Open Data App Token. Falls back to environment variable.",
    )
    parser.add_argument(
        "--pluto-path",
        dest="pluto_path",
        default=str(config.PLUTO_PATH_DEFAULT),
        help="Path to PLUTO CSV file.",
    )
    parser.add_argument(
        "--start-offset-311",
        dest="start_offset_311",
        type=int,
        default=0,
        help="Optional row offset for resuming 311 ingestion.",
    )
    parser.add_argument(
        "--start-offset-hpd",
        dest="start_offset_hpd",
        type=int,
        default=0,
        help="Optional row offset for resuming HPD ingestion.",
    )
    parser.add_argument(
        "--skip-311",
        action="store_true",
        help="Skip 311 ingestion.",
    )
    parser.add_argument(
        "--skip-hpd",
        action="store_true",
        help="Skip HPD ingestion.",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="Run cross-validation only (skip training/saving the full model).",
    )
    parser.add_argument(
        "--run-cv",
        action="store_true",
        help="Run expanding-window monthly cross-validation after training.",
    )
    parser.add_argument(
        "--cv-min-train-months",
        dest="cv_min_train_months",
        type=int,
        default=6,
        help="Minimum number of initial months to train on before starting rolling CV.",
    )
    parser.add_argument(
        "--cv-max-folds",
        dest="cv_max_folds",
        type=int,
        default=None,
        help="Limit the number of CV folds (evaluates the latest N target months).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pluto_path = Path(args.pluto_path)

    if args.stage in {"ingest", "all"}:
        token = ingestion.require_app_token(args.app_token)
        client = ingestion.build_client(token)
        try:
            ingestion.run_ingestion(
                client,
                args.start_date,
                start_offset_311=args.start_offset_311,
                start_offset_hpd=args.start_offset_hpd,
                skip_311=args.skip_311,
                skip_hpd=args.skip_hpd,
            )
        finally:
            client.close()

    if args.stage in {"features", "train", "all"}:
        feature_frame = features.run_feature_build(pluto_path)
    else:
        feature_frame = None

    if args.stage in {"train", "all"}:
        if feature_frame is None:
            feature_path = config.FEATURE_DIR / "monthly_features.parquet"
            if not feature_path.exists():
                raise FileNotFoundError("Feature file missing; run --stage features first.")
            feature_frame = pd.read_parquet(feature_path)

        if args.cv_only:
            modeling.rolling_monthly_cv(
                feature_frame,
                min_train_months=args.cv_min_train_months,
                max_folds=args.cv_max_folds,
            )
            return

        model, metrics = modeling.train_model(feature_frame)

        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = config.MODEL_DIR / "xgboost_distress.json"
        model.save_model(model_path)
        print(f"Saved model to {model_path}")
        print(f"ROC-AUC: {metrics['roc_auc']}")
        print(f"PR-AUC: {metrics.get('pr_auc')}")
        print(f"Accuracy: {metrics.get('accuracy')}")
        print(
            f"Train rows: {metrics.get('train_rows')} "
            f"(buildings: {metrics.get('train_buildings')}) "
            f"months: {metrics.get('train_month_start')} to {metrics.get('train_month_end')}"
        )
        print(
            f"Test rows: {metrics.get('test_rows')} "
            f"(buildings: {metrics.get('test_buildings')}) "
            f"months: {metrics.get('test_month_start')} to {metrics.get('test_month_end')}"
        )
        print(metrics["classification_report"])

        prediction_path = config.PROCESSED_DIR / "current_month_predictions.parquet"
        modeling.save_predictions(model, feature_frame, prediction_path)

        if args.run_cv:
            modeling.rolling_monthly_cv(
                feature_frame,
                min_train_months=args.cv_min_train_months,
                max_folds=args.cv_max_folds,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"Pipeline failed: {err}")
        sys.exit(1)
