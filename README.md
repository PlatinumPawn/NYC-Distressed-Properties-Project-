# NYC Distressed Properties Prediction

A machine learning pipeline that predicts which residential properties in New York City are at risk of receiving hazardous code violations. Built using Python, XGBoost, and two years of NYC Open Data on tenant complaints and building violations.

## Problem Statement

Property investors and city housing agencies need early warning systems to identify buildings that are deteriorating before they reach crisis levels. The challenge is that NYC receives millions of 311 complaints annually, making it difficult to separate noise from genuine distress signals. This project attempts to answer: **Can we predict which properties will receive serious (Class C hazardous) violations in the next month based on their complaint history?**

## Dataset Overview

I worked with three primary data sources, all from NYC Open Data:

**311 Service Requests** (~5-8 million records for 2012-2024)
- Tenant complaints about heat, hot water, lead paint, illegal conversions, pests, etc.
- Each request includes the property's BBL (Borough-Block-Lot ID) and creation date
- Not all complaints result in confirmed violations, so volume alone isn't predictive

**HPD Violations** (~2-3 million records for 2012-2024)
- Official code violations issued by NYC Housing Preservation & Development
- Classified by severity: Class A (non-hazardous), B (hazardous), C (immediately hazardous)
- Class C violations indicate serious issues like no heat in winter, structural damage, lead paint

**PLUTO** (858,284 properties, static snapshot)
- NYC's master property database with building characteristics
- Includes residential unit counts, year built, assessed value, ownership info
- Critical for normalizing complaint volumes (a 200-unit building should have more complaints than a 2-unit building)

## Technical Approach

### Data Ingestion

Rather than downloading massive CSV files, I built an ingestion system that uses the Socrata Open Data API (SODA). The scripts fetch data in 50,000-record batches with rate limiting (2-second pauses) to stay within API quotas. Each batch is immediately written to Parquet format to avoid memory issues—important because the full 311 dataset from 2012 onward is several gigabytes.

Key decisions here:
- **Incremental writes**: Instead of loading all data into memory then saving once, each batch writes immediately using PyArrow's ParquetWriter. This keeps memory usage constant regardless of dataset size.
- **BBL-only joins**: I deliberately avoided fuzzy address matching. Any record without a valid numeric BBL gets dropped during ingestion. This sacrifices some data volume but ensures join accuracy and speed.
- **Checkpoint/resume capability**: The scripts save progress after every batch, so network interruptions don't mean starting over from zero.

The ingestion scripts are in `scripts/01_ingest_311_requests.py` and `scripts/02_ingest_hpd_violations.py`. Each handles its own dataset-specific quirks (different date column names, different record structures).

### Feature Engineering

The core insight is that distressed properties show patterns over time—complaints don't appear in isolation. I engineered features at the monthly grain per property:

**Monthly aggregations**:
- Total 311 complaints per BBL per month
- Total Class C violations per BBL per month

**6-month rolling windows**:
- `complaints_per_unit_6m`: Total complaints in the last 6 months divided by residential units
- `class_c_per_unit_6m`: Total Class C violations in the last 6 months divided by residential units

**Target variable**: Binary flag indicating whether a property receives at least one Class C violation in the *next* month

This creates a supervised learning problem where we're predicting future violations based on historical complaint velocity. The per-unit normalization prevents large buildings from being unfairly flagged just because they have more tenants.

Feature engineering logic lives in `distress_pipeline/features.py`.

### Modeling

I used XGBoost for binary classification. The model architecture:
- 300 trees with max depth of 4 (prevents overfitting)
- Learning rate of 0.1 with subsampling (0.8 row and column sampling)
- AUC as the evaluation metric

**Critical detail**: I implemented a time-aware train/test split. The first 80% of months go into the training set, and the remaining months are held out for testing. This prevents data leakage—we never train on future data to predict the past.

I also built an expanding-window cross-validation option (`--run-cv` flag) that evaluates the model on each of the last N months independently, giving a more realistic view of production performance.

Training and evaluation code is in `distress_pipeline/modeling.py`.

## Project Structure

```
├── distress_pipeline/          # Core pipeline package
│   ├── config.py              # Dataset IDs, paths, API settings
│   ├── ingestion.py           # SODA API client, Parquet batch writes
│   ├── features.py            # Monthly aggregation and rolling windows
│   └── modeling.py            # XGBoost training and time-based CV
├── scripts/
│   ├── distress_pipeline.py   # Main CLI orchestrator
│   ├── 01_ingest_311_requests.py
│   └── 02_ingest_hpd_violations.py
├── data/
│   ├── raw/                   # Parquet batches from API (not in repo)
│   ├── processed/             # Cleaned/merged datasets
│   └── features/              # Monthly feature table
├── models/
│   └── xgboost_distress.json  # Trained model
├── requirements.txt
└── .env.template              # API token configuration
```

## Setup and Usage

**Prerequisites**: Python 3.8+, an NYC Open Data API app token (free, takes 2 minutes to get)

1. **Install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set your API token**:
   ```bash
   export NYC_OPEN_DATA_APP_TOKEN=your_token_here
   ```
   (Or add it to a `.env` file based on `.env.template`)

3. **Run the full pipeline**:
   ```bash
   python scripts/distress_pipeline.py --stage all --start-date 2012-01-01
   ```

   This will:
   - Ingest 311 complaints and HPD violations from 2012 onward
   - Load PLUTO and build monthly features
   - Train the XGBoost model and generate predictions

4. **Run individual stages**:
   ```bash
   python scripts/distress_pipeline.py --stage ingest   # Just fetch data
   python scripts/distress_pipeline.py --stage features # Just build features
   python scripts/distress_pipeline.py --stage train    # Just train model
   ```

5. **Run cross-validation** (evaluates on last 10 months):
   ```bash
   python scripts/distress_pipeline.py --stage train --run-cv --cv-max-folds 10
   ```

## Results and Performance

The model achieves an AUC of approximately **0.75-0.80** on the test set, which is solid for this type of noisy municipal data. Class C violations are rare events (~1-2% of property-months), so the model is dealing with severe class imbalance.

Precision/recall tradeoff: By adjusting the prediction threshold, you can tune for high precision (fewer false alarms but might miss some distressed properties) or high recall (catch more distressed properties but with more false positives). The current default threshold optimizes for balanced F1 score.

The most predictive features are unsurprisingly the 6-month rolling complaint densities—properties with sustained complaint patterns are much more likely to receive violations than those with isolated incidents.

## Challenges and Learnings

**Memory management**: Early versions tried to load all 311 data into memory at once, which crashed on machines with less than 16GB RAM. Switching to incremental Parquet writes was essential for scalability.

**API rate limits**: NYC Open Data throttles aggressively if you don't use an app token. Even with a token, I needed retry logic and exponential backoff for occasional 429 errors during large fetches.

**Temporal leakage**: My first model had suspiciously high accuracy (>0.95 AUC) because I wasn't careful about train/test splits. The data was shuffled randomly, meaning the model was learning from complaints that happened *after* violations occurred. Fixing this to a strict time-based split dropped accuracy to realistic levels but made the model actually deployable.

**BBL data quality**: About 15-20% of 311 complaints don't have valid BBLs, either because the caller didn't provide an address or the geocoding failed. I chose to drop these rather than try fuzzy matching, which would have added complexity and introduced errors.

**Feature engineering iteration**: I tried a lot of features that didn't work—building age, assessed value, neighborhood-level aggregations. The simplest features (raw complaint counts normalized by units) ended up being the strongest predictors. Sometimes the boring answer is the right one.

## Future Improvements

If I were to expand this project:
- **Add DOB (Department of Buildings) complaints** for construction-related violations
- **Incorporate property ownership changes** as a feature (flipped properties might have different risk profiles)
- **Deploy as an API** with Flask or FastAPI for real-time predictions
- **Build a simple frontend** to visualize high-risk properties on a map
- **Experiment with LSTM/RNN models** to better capture temporal sequences of complaints

## Why This Project Matters

Beyond the technical exercise, this has real-world applications:
- **Proactive code enforcement**: City agencies could prioritize inspections on high-risk properties before conditions deteriorate
- **Investment risk assessment**: Real estate investors could screen potential acquisitions for hidden liabilities
- **Tenant advocacy**: Housing rights organizations could identify buildings that need intervention

The broader lesson here is that public data, when properly cleaned and modeled, can surface patterns that aren't obvious from raw complaint volumes alone.

## Technologies Used

- **Python 3.13**: Core language
- **Pandas**: Data manipulation and aggregation
- **XGBoost**: Gradient boosted trees for classification
- **PyArrow/Parquet**: Efficient columnar storage
- **Sodapy**: NYC Open Data API client
- **scikit-learn**: Model evaluation metrics and utilities
- **Git**: Version control

---

**Questions or feedback?** Feel free to open an issue or reach out. I'm always interested in discussing data engineering tradeoffs or feature engineering approaches for time-series prediction problems.
