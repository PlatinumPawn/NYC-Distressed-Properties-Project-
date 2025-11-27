"""
Central configuration for the NYC Real Estate Distress pipeline.
"""

from pathlib import Path

# API configuration
DOMAIN = "data.cityofnewyork.us"
APP_TOKEN_ENV = "NYC_OPEN_DATA_APP_TOKEN"
BATCH_SIZE = 50_000
SLEEP_BETWEEN_REQUESTS = 1
START_DATE_DEFAULT = "2024-01-01"

# Dataset identifiers
DATASET_311 = "erm2-nwe9"
DATASET_HPD = "wvxf-dwi5"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
PLUTO_PATH_DEFAULT = PROJECT_ROOT / "pluto_25v3.csv"
