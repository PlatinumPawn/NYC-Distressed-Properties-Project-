# Results and the commands that produced them

All runs on the data in `data/raw/` and `pluto_25v3.csv`, offline. Raw logs are in `runs/`.

| Output | Command | Log |
|---|---|---|
| Original pipeline (unmodified code, commit e3ac040) | `.venv-orig/bin/python3 scripts/distress_pipeline.py --stage train` | `runs/00_original_pipeline.txt` |
| Dense panel build | `.venv/bin/python scripts/distress_pipeline.py --stage features` (log came from an equivalent call to `features.run_feature_build`) | `runs/01_features.txt` |
| Panel spot check vs raw rows | inline script, recorded in `log.md` | `runs/01b_feature_spotcheck.txt` |
| Backtest table, calibration, SHAP, segments | `.venv/bin/python scripts/backtest.py` (18 min, 8 cores) | `runs/02_backtest.txt`, `runs/backtest_results.md`, `runs/backtest_results.csv` |

## Data counts (computed with pyarrow over all batch files)

- 311 service requests: 20,530,586 rows, created 2012-11-01 .. 2022-10-28.
- HPD violations: 7,094,173 rows, approved 2012-11-01 .. 2024-05-14; 1,630,388 Class C.
- PLUTO 25v3: 858,284 lots; 767,302 with residential units.
- Panel: 278,285 residential BBLs, 10,183,798 building-months (2019-09 .. 2022-09), positive rate 1.60%.
- Test set: 3,334,508 building-months (2021-10 .. 2022-09), positive rate 1.76%.

## Backtest (4 quarterly refits, 24-month train window, test 2021-10 .. 2022-09)

prec/recall@1% = inspect the top 1% of scored buildings each month (about 2,780), then average over the 12 months.

| Model | PR-AUC | ROC-AUC | Prec@1% | Recall@1% | Brier |
|---|---|---|---|---|---|
| Heuristic: # Class C last month | 0.144 | 0.650 | 0.370 | 0.210 | n/a |
| Heuristic: # Class C last 12 months | 0.288 | 0.855 | 0.443 | 0.252 | n/a |
| Logistic regression (history + units) | 0.370 | 0.941 | 0.500 | 0.284 | 0.0135 |
| XGBoost, original 4 features and params | 0.331 | 0.915 | 0.467 | 0.265 | 0.0139 |
| **XGBoost, history + units (final)** | **0.395** | **0.947** | **0.517** | **0.294** | **0.0131** |
| XGBoost, + PLUTO static | 0.396 | 0.948 | 0.518 | 0.295 | 0.0131 |
| XGBoost, + PLUTO static + scale_pos_weight=10 | 0.398 | 0.948 | 0.520 | 0.296 | 0.0299 |

The SD of monthly PR-AUC is about 0.02, so the last three rows are indistinguishable.

## Original pipeline (for reference, not comparable)

ROC-AUC 0.879, PR-AUC 0.639, accuracy 0.793, test 2022-03 .. 2024-05 with a 22.3% positive rate.
The sparse panel, cross-building rolling windows and zero-filled 311 data make these numbers invalid (see README).
