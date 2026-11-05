# 🏚️ NYC Hazardous Housing Forecaster

**Which NYC buildings will get an "immediately hazardous" violation next month?**

NYC's housing department (HPD) can only inspect so many buildings. This model ranks every residential
building by its risk of a Class C violation (think no heat or hot water, or lead paint) in the
coming month, so the riskiest ones get looked at first.

> 🎯 **Top 1% of buildings flagged each month: 52% actually get a Class C violation.**
> The base rate is 1.8%, so that's about **29x better than picking at random**.

---

## 📊 Results

One year of out-of-sample months (Oct 2021 to Sep 2022), retrained every quarter, 3.3M building-months.

| Model | PR-AUC | Precision @ top 1% |
|---|---|---|
| 🧮 "How many Class C violations last month?" | 0.144 | 37% |
| 🧮 "How many Class C violations last 12 months?" | 0.288 | 44% |
| 📈 Logistic regression | 0.370 | 50% |
| 🌲 **XGBoost (final)** | **0.395** | **52%** |

Why PR-AUC and not accuracy? Only ~1.8% of building-months have a Class C violation, so a model that
always says "no" is 98% accurate and completely useless. 🙃 ROC-AUC is 0.947, but it looks good mostly
because the easy negatives pad it.

**Calibrated, too:** in the riskiest 10% of buildings it predicts 15.7% and the real rate is 15.0%
(measured on the near-identical + PLUTO variant, see `RESULTS.md`).

## 🗽 Data

All public, from [NYC Open Data](https://opendata.cityofnewyork.us/):

| Source | Rows |
|---|---|
| ☎️ 311 service requests (`erm2-nwe9`) | 20.5M (2012 to 2022) |
| 📋 HPD housing violations (`wvxf-dwi5`) | 7.1M, 1.6M of them Class C |
| 🏢 PLUTO 25v3 (lot data) | 858K lots |

The 311 pull stops in Oct 2022, so the panel ends Sep 2022.

## 🔧 How it works

1. **Building-month panel:** one row per building per month, 278K buildings from Sep 2019 to Sep 2022,
   **10.2M rows**. A building only gets rows once it first shows up in the records.
2. **Features (only from the past):** Class A/B/C violation counts and heat, housing and other 311
   complaints over the last 1, 3, 6 and 12 months, months since the last Class C, and number of units.
3. **Target:** at least one Class C violation next month.
4. **Backtest:** 4 quarterly refits, each trained on the previous 24 months and tested on the next 3.
   Rolling forward like this is how it would actually be used.
5. **Model:** XGBoost, 400 trees, depth 6. No class reweighting: it barely changed the ranking and
   more than doubled the Brier score, so the probabilities stay honest.

**What drives it (SHAP, + PLUTO variant):** how recently the building had a Class C, how big it is, and
its recent heat and housing complaints.

## 🐛 v1 was wrong (and that's the interesting part)

The first version reported PR-AUC 0.64 and 79% accuracy. Looked great. It wasn't real:

- 🕳️ **Missing data posing as zeros.** 311 data ends Oct 2022 but violations run to 2024. The test set
  sat in that gap, with complaints filled in as zero.
- 📅 **"Next month" wasn't next month.** Rows only existed for active months, so "next month" often
  meant "next month something happened". That inflated the positive rate from 1.6% to 10.5%.
- 🔀 **Buildings bleeding into each other.** Rolling sums weren't grouped by building, so 35% of rows
  mixed in a neighbor's history.
- 🎲 **Accuracy at a 0.5 cutoff, no baselines.** No way to tell whether the model beat a simple rule.

Fixing all four cut PR-AUC from 0.64 to 0.40. Lower, but now it's true, and it still beats every baseline.

## ⚠️ Limitations

- 📆 One test year, and the data ends in 2022. HPD policy changes since then could shift things.
- 👀 A violation only exists if an inspector shows up, and complaints trigger inspections. So the
  model partly predicts *where inspectors go*, not only which buildings are in bad shape.
- 🏚️ Building info comes from a 2025 snapshot, so lots merged or demolished before then are missing.
- 🏠 Small buildings are the hardest: PR-AUC 0.13 for 1-2 units vs about 0.5 for 20+ units.
- 🆕 About a quarter of positives are buildings with no Class C in the past year. Those are hard to
  catch (PR-AUC 0.09).
- 🧾 Which buildings are in the panel at all is decided using the whole data period (rows still
  only start after a building's first record).

## 🚀 Run it

```bash
python3 -m venv .venv && .venv/bin/pip install pandas pyarrow numpy xgboost scikit-learn
.venv/bin/python scripts/distress_pipeline.py --stage features   # builds the panel
.venv/bin/python scripts/backtest.py                             # ~18 min, writes runs/backtest_results.*
```

Re-pulling the raw data (`--stage ingest`) needs `sodapy` and a free NYC Open Data app token in `.env`
(see `.env.template`).

📁 **Where things are:** `distress_pipeline/features.py` (panel and features),
`distress_pipeline/modeling.py` (backtest and metrics), `scripts/backtest.py` (experiments),
`RESULTS.md` and `runs/` (every number above, with the command that produced it).
