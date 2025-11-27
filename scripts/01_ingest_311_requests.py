#!/usr/bin/env python3
"""
311 Service Requests Data Ingestion Pipeline
Fetches NYC 311 data from SODA API with pagination and rate limiting

FIXED VERSION:
- Max retry limits (no infinite loops)
- Incremental Parquet writing (no memory overflow)
- Checkpoint/resume capability
- API response validation
- Proper exception handling
- Automatic deduplication
"""

import os
import sys
import time
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from sodapy import Socrata
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / '.env')

# Configuration
APP_TOKEN = os.getenv('NYC_OPEN_DATA_APP_TOKEN')
DATASET_ID = 'erm2-nwe9'
BATCH_SIZE = 50000
RATE_LIMIT_DELAY = 10
START_DATE = '2023-01-01'
MAX_RETRIES = 3

# Output configuration
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'raw'
OUTPUT_FILE = OUTPUT_DIR / '311_service_requests.parquet'
CHECKPOINT_FILE = OUTPUT_DIR / '311_checkpoint.json'
TEMP_FILE = OUTPUT_DIR / '311_service_requests_temp.parquet'

# Expected columns for validation
REQUIRED_COLUMNS = ['unique_key', 'created_date', 'bbl']

def validate_config():
    """Validate configuration parameters"""
    assert BATCH_SIZE > 0, "BATCH_SIZE must be positive"
    assert RATE_LIMIT_DELAY >= 0, "RATE_LIMIT_DELAY must be non-negative"
    assert START_DATE, "START_DATE cannot be empty"
    try:
        datetime.strptime(START_DATE, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"START_DATE must be YYYY-MM-DD format, got: {START_DATE}")

def load_checkpoint():
    """Load checkpoint if exists"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(offset, batch_num, total_fetched):
    """Save checkpoint to allow resume"""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'offset': offset,
        'batch_num': batch_num,
        'total_fetched': total_fetched,
        'timestamp': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def clear_checkpoint():
    """Remove checkpoint file after successful completion"""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

def get_total_count(client, dataset_id, start_date):
    """Get total count of records matching our criteria"""
    print(f"Querying total count of records since {start_date}...")
    try:
        result = client.get(
            dataset_id,
            select="COUNT(*) as count",
            where=f"created_date >= '{start_date}T00:00:00.000'"
        )
        total = int(result[0]['count'])
        print(f"✓ Found {total:,} records to fetch")
        return total
    except Exception as e:
        print(f"⚠ Warning: Could not get exact count: {e}")
        return None

def validate_batch(batch_df, batch_num):
    """Validate batch has required columns"""
    missing = [col for col in REQUIRED_COLUMNS if col not in batch_df.columns]
    if missing:
        raise ValueError(f"Batch {batch_num} missing required columns: {missing}")
    return True

def fetch_311_data(app_token=None, resume=False):
    """
    Fetch 311 Service Requests with incremental saving

    Args:
        app_token: NYC Open Data App Token
        resume: If True, resume from checkpoint
    """
    print("="*80)
    print("311 SERVICE REQUESTS DATA INGESTION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset ID: {DATASET_ID}")
    print(f"  Start Date: {START_DATE}")
    print(f"  Batch Size: {BATCH_SIZE:,}")
    print(f"  Rate Limit Delay: {RATE_LIMIT_DELAY}s")
    print(f"  Max Retries: {MAX_RETRIES}")
    print(f"  App Token: {'✓ Configured' if app_token else '✗ Not configured'}")
    print(f"  Output: {OUTPUT_FILE}")

    # Check for checkpoint
    checkpoint = load_checkpoint() if resume else None
    if checkpoint:
        print(f"\n✓ Resuming from checkpoint:")
        print(f"  Batch: {checkpoint['batch_num']}")
        print(f"  Records fetched: {checkpoint['total_fetched']:,}")
        offset = checkpoint['offset']
        batch_num = checkpoint['batch_num']
        total_fetched = checkpoint['total_fetched']
    else:
        print("\n  Starting fresh ingestion")
        offset = 0
        batch_num = 1
        total_fetched = 0
        # Clean up temp file if exists
        if TEMP_FILE.exists():
            TEMP_FILE.unlink()

    print()

    # Initialize Socrata client
    client = Socrata("data.cityofnewyork.us", app_token, timeout=60)

    # Parquet writer for incremental writing
    parquet_writer = None
    schema = None

    try:
        # Get total count
        total_records = get_total_count(client, DATASET_ID, START_DATE)

        while True:
            print(f"\n[Batch {batch_num}] Fetching records {offset:,} to {offset+BATCH_SIZE:,}...")

            retry_count = 0
            batch_success = False

            while retry_count < MAX_RETRIES and not batch_success:
                try:
                    # Fetch batch
                    results = client.get(
                        DATASET_ID,
                        select="""
                            unique_key,
                            created_date,
                            closed_date,
                            agency,
                            agency_name,
                            complaint_type,
                            descriptor,
                            location_type,
                            incident_zip,
                            incident_address,
                            street_name,
                            address_type,
                            city,
                            status,
                            resolution_description,
                            resolution_action_updated_date,
                            bbl,
                            borough,
                            latitude,
                            longitude,
                            location
                        """,
                        where=f"created_date >= '{START_DATE}T00:00:00.000'",
                        order="created_date ASC",
                        limit=BATCH_SIZE,
                        offset=offset
                    )

                    # Check if empty
                    if not results or len(results) == 0:
                        print(f"✓ No more records found. Completed at batch {batch_num}")
                        batch_success = True
                        break

                    # Convert to DataFrame
                    batch_df = pd.DataFrame.from_records(results)
                    records_fetched = len(batch_df)

                    # Validate batch structure
                    validate_batch(batch_df, batch_num)

                    # Convert date columns
                    date_cols = ['created_date', 'closed_date', 'resolution_action_updated_date']
                    for col in date_cols:
                        if col in batch_df.columns:
                            batch_df[col] = pd.to_datetime(batch_df[col], errors='coerce')

                    # Initialize parquet writer with schema from first batch
                    if parquet_writer is None:
                        schema = pa.Schema.from_pandas(batch_df)
                        TEMP_FILE.parent.mkdir(parents=True, exist_ok=True)
                        parquet_writer = pq.ParquetWriter(TEMP_FILE, schema, compression='snappy')

                    # Write batch incrementally
                    table = pa.Table.from_pandas(batch_df, schema=schema)
                    parquet_writer.write_table(table)

                    total_fetched += records_fetched
                    print(f"  ✓ Fetched {records_fetched:,} records (total: {total_fetched:,})")

                    # Save checkpoint
                    save_checkpoint(offset + BATCH_SIZE, batch_num + 1, total_fetched)

                    batch_success = True

                    # Check if last batch
                    if records_fetched < BATCH_SIZE:
                        print(f"✓ Last batch received. Completed at batch {batch_num}")
                        break

                    # Increment for next batch
                    offset += BATCH_SIZE
                    batch_num += 1

                    # Rate limiting
                    print(f"  ⏳ Sleeping {RATE_LIMIT_DELAY}s to respect rate limits...")
                    time.sleep(RATE_LIMIT_DELAY)

                except KeyboardInterrupt:
                    print("\n\n⚠ Interrupted by user. Progress saved to checkpoint.")
                    print(f"  Resume by running script with --resume flag")
                    raise

                except (ConnectionError, TimeoutError, Exception) as e:
                    retry_count += 1
                    if retry_count < MAX_RETRIES:
                        wait_time = RATE_LIMIT_DELAY * (2 ** retry_count)
                        print(f"✗ Error fetching batch {batch_num} (attempt {retry_count}/{MAX_RETRIES}): {e}")
                        print(f"  Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"✗ Failed to fetch batch {batch_num} after {MAX_RETRIES} attempts")
                        print(f"  Error: {e}")
                        print(f"  Progress saved. Resume by running script again.")
                        raise

            # Exit loop if no more data
            if not results or len(results) == 0:
                break

        # Close parquet writer
        if parquet_writer:
            parquet_writer.close()

        print(f"\n{'='*80}")
        print(f"✓ Data fetching complete!")
        print(f"  Total batches: {batch_num}")
        print(f"  Total records: {total_fetched:,}")

        return total_fetched

    finally:
        client.close()

def deduplicate_and_finalize():
    """Deduplicate data and move to final location"""
    print(f"\nFinalizing data...")

    if not TEMP_FILE.exists():
        print("✗ No temporary file found!")
        return False

    # Read temp file
    print(f"  Reading temporary file...")
    df = pd.read_parquet(TEMP_FILE)

    initial_count = len(df)
    print(f"  Initial records: {initial_count:,}")

    # Deduplicate on unique_key
    if 'unique_key' in df.columns:
        df = df.drop_duplicates(subset=['unique_key'], keep='first')
        final_count = len(df)
        duplicates = initial_count - final_count
        if duplicates > 0:
            print(f"  ⚠ Removed {duplicates:,} duplicate records ({duplicates/initial_count*100:.1f}%)")
        else:
            print(f"  ✓ No duplicates found")

    # Check for empty BBLs
    if 'bbl' in df.columns:
        valid_bbl = df['bbl'].notna() & (df['bbl'] != '')
        bbl_pct = valid_bbl.sum() / len(df) * 100
        print(f"  Records with valid BBL: {valid_bbl.sum():,} ({bbl_pct:.1f}%)")

    # Save final file
    print(f"  Saving to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, engine='pyarrow', compression='snappy', index=False)

    file_size = OUTPUT_FILE.stat().st_size / 1024**2
    print(f"  ✓ Saved {len(df):,} records")
    print(f"  ✓ File size: {file_size:.1f} MB")

    # Save metadata
    try:
        metadata = {
            'records': len(df),
            'columns': len(df.columns),
            'date_range': f"{df['created_date'].min()} to {df['created_date'].max()}" if 'created_date' in df.columns else 'N/A',
            'ingestion_timestamp': datetime.now().isoformat(),
            'unique_complaints': df['unique_key'].nunique() if 'unique_key' in df.columns else 'N/A',
            'boroughs': df['borough'].nunique() if 'borough' in df.columns else 'N/A',
            'complaint_types': df['complaint_type'].nunique() if 'complaint_type' in df.columns else 'N/A',
            'valid_bbl_pct': f"{bbl_pct:.1f}%" if 'bbl' in df.columns else 'N/A'
        }

        metadata_file = OUTPUT_FILE.parent / '311_metadata.txt'
        with open(metadata_file, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"  ✓ Metadata saved to {metadata_file}")
    except Exception as e:
        print(f"  ⚠ Warning: Could not save metadata: {e}")

    # Clean up temp file
    TEMP_FILE.unlink()
    print(f"  ✓ Cleaned up temporary file")

    return df

def main():
    """Main execution"""
    start_time = time.time()

    # Validate configuration
    try:
        validate_config()
    except (AssertionError, ValueError) as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)

    # Check for App Token
    app_token = APP_TOKEN
    if not app_token:
        print("\n⚠ WARNING: NYC_OPEN_DATA_APP_TOKEN not set!")
        print("  You may be throttled without an App Token.")
        print("  Get one at: https://data.cityofnewyork.us/profile/edit/developer_settings")
        response = input("\n  Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(1)

    # Check for resume
    resume = '--resume' in sys.argv or load_checkpoint() is not None

    try:
        # Fetch data
        total_fetched = fetch_311_data(app_token, resume=resume)

        if total_fetched > 0:
            # Deduplicate and finalize
            df = deduplicate_and_finalize()

            if df is not None:
                # Summary statistics
                print(f"\n{'='*80}")
                print("DATA SUMMARY")
                print(f"{'='*80}")

                if 'complaint_type' in df.columns:
                    print(f"\nTop 10 Complaint Types:")
                    print(df['complaint_type'].value_counts().head(10).to_string())

                if 'borough' in df.columns:
                    print(f"\nRecords by Borough:")
                    print(df['borough'].value_counts().to_string())

                # Clear checkpoint on success
                clear_checkpoint()

                elapsed = time.time() - start_time
                print(f"\n✓ Pipeline completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
                print(f"✓ Data saved to: {OUTPUT_FILE}")
            else:
                print("\n✗ Finalization failed")
                sys.exit(1)
        else:
            print("\n✗ No data was fetched")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
        print(f"  Checkpoint saved. Resume with: python3 {__file__} --resume")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        print(f"  Checkpoint saved. Fix issue and resume with: python3 {__file__} --resume")
        sys.exit(1)

if __name__ == "__main__":
    main()
