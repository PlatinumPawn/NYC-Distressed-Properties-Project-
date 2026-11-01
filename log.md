# log.md

## 2026-11-01: audit and rebuild of evaluation (portfolio prep)

### Prompt
Get the pipeline running from data on disk, record true baseline, audit leakage/target/split/
metrics, add honest backtest + baselines + features, rewrite README with real numbers.
Constraints: work only inside this folder; fully offline (no API pulls, no pip after the
initial setup; see Environment).

### Environment
- `.venv` (Python 3.15) was created and populated with `pip install pandas pyarrow numpy
  xgboost scikit-learn` BEFORE the offline rule arrived. No other downloads. Versions: xgboost
  3.4.1, scikit-learn 1.9.1, pandas 3.1.0.
- `.venv` has no sodapy, so the original pipeline (which imports ingestion eagerly) was run
  once under `.venv-orig` (Linus's original env) to get the untouched baseline.
- tabulate and shap are not installed; results use `to_string`, SHAP uses XGBoost's native
  `pred_contribs` (TreeSHAP).

### Data on disk (computed)
- 311: 20,530,586 rows (516 batch files), created 2012-11-01 .. 2022-10-28. Every row has a BBL.
- HPD violations: 7,094,173 rows (142 files), approved 2012-11-01 .. 2024-05-14.
  Class B 3,126,412 / A 1,672,706 / C 1,630,388 / I 664,667.
- PLUTO 25v3: 858,284 lots, 767,302 with unitsres > 0.

### Original pipeline baseline (runs/00_original_pipeline.txt)
`.venv-orig/bin/python3 scripts/distress_pipeline.py --stage train` (original code, commit e3ac040):
ROC-AUC 0.879, PR-AUC 0.639, accuracy 0.793 at threshold 0.5, 3,863,109 train rows /
325,354 train buildings, test 2022-03..2024-05, test positive rate 22.3%.
The old resume's "79.3% accuracy" and "325,000+ properties" trace to this run. Both come from
the flawed setup below and should not be used. The "25M service requests" claim is not
supported: 20.5M rows on disk.

### Problems found in original code
1. 311 data stops 2022-10-28 but HPD runs to 2024-05. The outer join + fillna(0) gives zero
   complaints for every building after Oct 2022. The 80/20 month split put the whole test
   set (2022-03..2024-05) mostly in that gap.
2. Sparse panel: rows exist only for (bbl, month) with any event. `shift(-1)` therefore
   predicts the next *active* month (46% of consecutive rows are > 1 month apart), and
   "6m" windows are 6 active months. Positive rate is inflated to 10.5% (true 1.6%), and
   the model only ever scores buildings that already had activity that month.
3. Rolling window was not grouped by BBL: `groupby().shift(1).rolling(6)` rolls across
   BBL boundaries. 35% of the kept rows had a window that included another building's
   history.
4. Accuracy at 0.5 with scale_pos_weight is not meaningful for a 1.6% base rate. No
   baselines, no ranking metric aligned with inspection targeting.
5. PLUTO 25v3 is a 2025 snapshot applied to 2012-2022 (static-attribute leakage /
   survivorship: lots that were merged or demolished drop out). Only near-static fields
   are used (units, year built, floors, borough, public owner flag); limitation documented.

### Decisions
- Dense panel per BBL from first observed record through 2022-09 (last full 311 month).
  Universe: residential PLUTO lots with >= 1 HPD violation or housing-type 311 complaint;
  a row exists only after the building's first record (membership is causal).
- Target: >= 1 Class C violation with approveddate in month t+1. approveddate is used for
  both features and target (median lag from inspection 1 day), so features only use
  violations already recorded.
- Rolling-origin backtest: 4 quarterly refits, 24-month train window, test months
  2021-10..2022-09. Training rows end at t-1 for first test month t, whose label (month t
  violations) is known at prediction time.
- Metrics: PR-AUC, ROC-AUC, precision/recall when inspecting the top 1% of buildings each
  month (~2,780 buildings), Brier, decile calibration.
- Count tensor + cumsum (numpy) instead of pandas rolling: simpler and impossible to roll
  across BBLs.

### Verification
- runs/01_features.txt: panel 10,183,798 rows, 278,285 BBLs, 2019-09..2022-09, pos rate 1.597%.
- runs/01b_feature_spotcheck.txt: 185 rows for 5 random BBLs recomputed from raw data,
  0 mismatches (viol_c_1m, viol_c_12m, heat311_3m, target).

### Backtest results (runs/02_backtest.txt, 18 min)
Test 2021-10..2022-09, 3,334,508 rows, pos rate 1.76%. PR-AUC / prec@1%:
last-month-C heuristic 0.144 / 0.370; 12m-C heuristic 0.288 / 0.443; logreg 0.370 / 0.500;
xgb original 4 features 0.331 / 0.467; xgb history+units 0.395 / 0.517 (FINAL);
+PLUTO static 0.396 / 0.518 (no gain, dropped: within the 0.02 monthly SD and uses a 2025
snapshot); +scale_pos_weight 0.398 / 0.520 but Brier 0.030 vs 0.013 (dropped).
Calibration and SHAP in runs/backtest_results.md were computed for the +PLUTO variant, which
has nearly identical metrics. Rerun if exact figures for the final model are needed.

### Not done / next steps
- The CLI path `scripts/distress_pipeline.py --stage features` was not re-run end to end after
  the edit. The panel was built by calling `features.run_feature_build` directly (same function).
- No paired significance test between models (predictions not persisted). The monthly PR-AUC
  SD is the only spread estimate.
- Old artifacts left in place: models/xgboost_distress.json, data/features/monthly_features.parquet,
  data/processed/current_month_predictions.parquet (original pipeline). CODEX.md untracked, untouched.
- Ideas, not tried: DOB complaints, the HPD registration/owner dataset, extending 311 past 2022-10
  (needs network).
