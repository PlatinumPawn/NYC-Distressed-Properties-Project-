"""
Data ingestion helpers for NYC Open Data sources.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

import pandas as pd
from sodapy import Socrata

from . import config


def require_app_token(explicit: Optional[str]) -> str:
    """Fail fast if the App Token is missing."""
    token = explicit or os.getenv(config.APP_TOKEN_ENV)
    if not token:
        raise RuntimeError(
            f"Missing NYC Open Data App Token. Set {config.APP_TOKEN_ENV} or pass --app-token."
        )
    return token


def build_client(app_token: str) -> Socrata:
    """Create an authenticated Socrata client."""
    return Socrata(config.DOMAIN, app_token, timeout=90)


def ingest_dataset(
    client: Socrata,
    dataset_id: str,
    dataset_name: str,
    date_field: str,
    start_date: str,
    start_offset: int = 0,
    select_fields: Optional[str] = None,
    extra_where: Optional[str] = None,
    max_batches: Optional[int] = None,
    sleep_seconds: Optional[float] = None,
) -> None:
    """
    Stream a SODA3 dataset in 50k batches, drop rows without numeric BBL,
    and write each batch immediately to Parquet to avoid large in-memory tables.
    """
    target_dir = config.RAW_DIR / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    def determine_initial_batch_index() -> int:
        existing_files = sorted(target_dir.glob(f"{dataset_name}_batch_*.parquet"))
        if not existing_files:
            return 0

        pattern = re.compile(rf"{re.escape(dataset_name)}_batch_(\d+)\.parquet$")
        max_idx = -1
        for file in existing_files:
            match = pattern.match(file.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))

        if max_idx < 0:
            return 0

        prompt = (
            f"Detected existing {dataset_name} batches (highest index {max_idx:05d}). "
            "Type 'y' to restart numbering from 0 and delete existing batches, "
            "or press Enter to append after the last batch: "
        )
        choice = input(prompt).strip().lower()
        if choice in {"y", "yes"}:
            for file in existing_files:
                try:
                    file.unlink()
                except OSError as err:
                    print(f"Failed to remove {file}: {err}")
            print(f"[{dataset_name}] Cleared existing batches. Restarting at 00000.")
            return 0

        print(f"[{dataset_name}] Continuing from batch {max_idx + 1:05d}.")
        return max_idx + 1

    where_clause = f"{date_field} >= '{start_date}T00:00:00.000'"
    if extra_where:
        where_clause = f"{where_clause} AND {extra_where}"

    offset = max(0, start_offset)
    batch_idx = determine_initial_batch_index()
    total_rows = 0

    sleep = sleep_seconds if sleep_seconds is not None else config.SLEEP_BETWEEN_REQUESTS

    while True:
        if max_batches is not None and batch_idx >= max_batches:
            print(f"[{dataset_name}] Reached max_batches={max_batches}. Stopping early.")
            break

        print(
            f"[{dataset_name}] Requesting batch {batch_idx:05d} "
            f"(offset {offset:,}, limit {config.BATCH_SIZE:,})."
        )
        try:
            results = client.get(
                dataset_id,
                select=select_fields,
                where=where_clause,
                order=f"{date_field} ASC",
                limit=config.BATCH_SIZE,
                offset=offset,
            )
        except Exception as exc:
            retry_wait = sleep * 2
            print(
                f"[{dataset_name}] Request failed at offset {offset}: {exc}. "
                f"Retrying after {retry_wait}s."
            )
            time.sleep(retry_wait)
            continue

        if not results:
            print(f"[{dataset_name}] No more rows after offset {offset}.")
            break

        batch_df = pd.DataFrame.from_records(results)

        # Drop rows with missing BBL up front to keep only joinable records.
        batch_df["bbl"] = pd.to_numeric(batch_df.get("bbl"), errors="coerce")
        batch_df[date_field] = pd.to_datetime(batch_df.get(date_field), errors="coerce")
        batch_df = batch_df.dropna(subset=["bbl", date_field])

        if batch_df.empty:
            offset += config.BATCH_SIZE
            batch_idx += 1
            print(f"[{dataset_name}] Batch {batch_idx:05d} had no usable BBL rows.")
            print(f"[{dataset_name}] Sleeping {sleep}s before next request...")
            time.sleep(sleep)
            continue

        batch_df["bbl"] = batch_df["bbl"].astype("int64")

        batch_path = target_dir / f"{dataset_name}_batch_{batch_idx:05d}.parquet"
        batch_df.to_parquet(batch_path, engine="pyarrow", compression="snappy", index=False)

        rows = len(batch_df)
        total_rows += rows

        print(
            f"[{dataset_name}] Saved batch {batch_idx:05d} with {rows:,} rows "
            f"(offset {offset:,})."
        )

        batch_idx += 1
        offset += config.BATCH_SIZE

        print(f"[{dataset_name}] Sleeping {sleep}s before next request...")
        time.sleep(sleep)

    print(f"[{dataset_name}] Finished. Persisted {total_rows:,} rows to {target_dir}.")


def run_ingestion(
    client: Socrata,
    start_date: str,
    max_batches: Optional[int] = None,
    start_offset_311: int = 0,
    start_offset_hpd: int = 0,
    skip_311: bool = False,
    skip_hpd: bool = False,
) -> None:
    """Orchestrate ingestion for 311 and HPD datasets."""
    if not skip_311:
        ingest_dataset(
            client=client,
            dataset_id=config.DATASET_311,
            dataset_name="311",
            date_field="created_date",
            start_date=start_date,
            start_offset=start_offset_311,
            select_fields="""
                unique_key, created_date, closed_date, complaint_type, descriptor,
                incident_zip, incident_address, city, status, bbl, borough,
                latitude, longitude, resolution_action_updated_date
            """,
            max_batches=max_batches,
            sleep_seconds=config.SLEEP_BETWEEN_REQUESTS,
        )

    if not skip_hpd:
        ingest_dataset(
            client=client,
            dataset_id=config.DATASET_HPD,
            dataset_name="hpd",
            date_field="approveddate",
            start_date=start_date,
            start_offset=start_offset_hpd,
            select_fields="""
                violationid, approveddate, novissueddate, inspectiondate,
                currentstatusdate, class, bbl, boroid, housenumber, streetname, zip
            """,
            max_batches=max_batches,
            sleep_seconds=config.SLEEP_BETWEEN_REQUESTS,
        )
