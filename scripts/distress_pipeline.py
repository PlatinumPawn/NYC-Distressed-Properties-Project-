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

from distress_pipeline import config, features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NYC Real Estate Distress Prediction Engine")
    parser.add_argument(
        "--stage",
        choices=["ingest", "features", "all"],
        default="all",
        help="Stage to run. 'all' executes ingestion -> features. Evaluate with scripts/backtest.py.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pluto_path = Path(args.pluto_path)

    if args.stage in {"ingest", "all"}:
        from distress_pipeline import ingestion  # needs sodapy; only required for API pulls

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

    if args.stage in {"features", "all"}:
        features.run_feature_build(pluto_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"Pipeline failed: {err}")
        sys.exit(1)
