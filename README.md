# NYC Distressed Properties: next-month Class C violation risk

Predict which NYC residential buildings (by BBL) will receive at least one Class C
("immediately hazardous") HPD violation next month, using their 311 complaint and violation
history. Use case: rank buildings so an inspection team with fixed capacity visits the
riskiest ones first. Every number here comes from a logged run; see `RESULTS.md`.

## Data (NYC Open Data, already on disk)

| Source | Rows | Coverage |
|---|---|---|
| 311 service requests (`erm2-nwe9`) | 20,530,586 | created 2012-11-01 .. 2022-10-28 |
| HPD housing violations (`wvxf-dwi5`) | 7,094,173 (1,630,388 Class C) | approved 2012-11-01 .. 2024-05-14 |
| PLUTO 25v3 | 858,284 lots (767,302 residential) | 2025 snapshot |

The 311 extract stops in October 2022, so the modeling panel ends at 2022-09, the last
complete month with both sources.

## Method

- **Panel.** Dense building-month grid for 278,285 residential BBLs that have at least one HPD
  violation or housing 311 complaint. A building enters the panel only once it has its first
  record, so membership never uses future data. The table has 10,183,798 rows (2019-09 .. 2022-09).
- **Target.** At least one Class C violation with `approveddate` in month t+1 (1.60% of rows).
- **Features (month t and earlier only).** Counts over 1, 3, 6 and 12 months for heat 311,
  other housing 311, non-housing 311, and Class A, B and C violations. Also months since the
  last Class C violation and residential units. Computed with a count tensor and cumulative
  sums, then checked against raw rows for 185 sampled building-months (0 mismatches).
- **Evaluation.** Rolling-origin backtest: 4 quarterly refits, each trained on the previous
  24 months and tested on the next 3 months, covering 2021-10 .. 2022-09 (3.33M
  building-months). Metrics: PR-AUC, ROC-AUC, Brier score, and precision/recall when
  inspecting the top 1% of buildings each month (about 2,780).
- **Model.** XGBoost (400 trees, depth 6, learning rate 0.05), unweighted, so its
  probabilities stay calibrated.

## Results (`.venv/bin/python scripts/backtest.py`)

| Model | PR-AUC | ROC-AUC | Precision@1% | Recall@1% |
|---|---|---|---|---|
| Heuristic: # Class C last month | 0.144 | 0.650 | 0.370 | 0.210 |
| Heuristic: # Class C last 12 months | 0.288 | 0.855 | 0.443 | 0.252 |
| Logistic regression | 0.370 | 0.941 | 0.500 | 0.284 |
| XGBoost, original 4 features | 0.331 | 0.915 | 0.467 | 0.265 |
| **XGBoost, final** | **0.395** | **0.947** | **0.517** | **0.294** |

Test positive rate is 1.76%. In the top 1% the final model is right 52% of the time, 29x the
base rate, against 44% for the best heuristic. Monthly PR-AUC varies by about 0.02 (SD).
ROC-AUC looks high mainly because most buildings are easy negatives; PR-AUC and
precision@1% are the metrics that matter here.

- **Calibration.** Predicted vs observed rate per test decile matches closely (top decile:
  0.157 predicted vs 0.150 observed; `runs/backtest_results.md`). `scale_pos_weight` did not
  improve ranking (within noise) and more than doubled the Brier score (0.013 to 0.030), so it
  was dropped.
- **What did not help.** PLUTO attributes (year built, floors, borough, public owner) raised
  PR-AUC by 0.001, which is within noise, and they come from a 2025 snapshot. They are excluded.
- **Drivers (mean |SHAP|).** Months since last Class C, residential units, 12-month housing
  and heat 311 counts, and 12-month Class B violations.
- **Errors.** 24% of positives are buildings with no Class C in the previous 12 months. PR-AUC
  on those is 0.09 vs 0.47 for buildings with recent Class C, and they make up under 1% of
  the top-1% list. Small buildings are hardest: PR-AUC 0.13 for 1-2 units vs 0.47-0.59 for 20+ units.

## What changed from the original version

The first version reported ROC-AUC 0.879 / PR-AUC 0.639 / accuracy 0.793. Those numbers
were invalid:
1. 311 data ends in October 2022 but violations run to May 2024. Missing complaints were
   zero-filled, and the 80/20 test split (2022-03 .. 2024-05) fell mostly in that gap.
2. The panel only had rows for months with activity, so "next month" really meant "next active
   month" (46% of gaps were longer than one month). That inflated the positive rate to 10.5%.
3. The 6-month rolling sums were not grouped by building. 35% of rows mixed in another
   BBL's history.
4. Accuracy at a 0.5 threshold, with no baselines.

## Limitations

- Data ends in 2022 and the test year is a single period (2021-10 .. 2022-09). Results may not
  carry over to later years or to HPD policy changes.
- A violation is only recorded when an inspection happens, so the model partly predicts
  inspector attention, not just building condition. Complaints trigger inspections.
- The universe and units come from PLUTO 25v3. Lots that were merged or demolished before
  2025 are missing (survivorship).
- Universe membership is restricted to buildings that show up in HPD or housing 311 records
  at some point. Rows are only created after a building's first record, but the universe
  list itself uses the whole period.

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install pandas pyarrow numpy xgboost scikit-learn
.venv/bin/python scripts/distress_pipeline.py --stage features   # writes data/features/panel_features.parquet
.venv/bin/python scripts/backtest.py                             # ~18 min, writes runs/backtest_results.*
```

`--stage ingest` re-pulls raw data from the Socrata API (needs `sodapy` and an app token).
`models/xgboost_distress.json` and `data/features/monthly_features.parquet` are artifacts of
the original pipeline and are not used.

Code: `distress_pipeline/features.py` (panel and features), `distress_pipeline/modeling.py`
(backtest, models, metrics), `scripts/backtest.py` (experiments and report).
