# Maternal Mortality Risk Drivers & District Priority Classification (XAI)

A reproducible research pipeline and Streamlit decision-support dashboard that finds Indian districts where
**institutional-birth coverage is high but reported maternal risk is still unexpectedly high** (discordance), classifies
every district into four operational tiers, and explains each district's classification with TreeSHAP mapped onto
LaQshya, PMSMA, JSY and POSHAN Abhiyaan levers.

> **Research prototype, not a clinical tool.** Outputs are statistical associations from facility-reported HMIS data and
> NFHS-5 survey indicators. Validate every finding with MDSR, CRS/verbal-autopsy and LaQshya facility audits before acting.

## Architecture

| Layer | Implementation |
| --- | --- |
| 1. Ingestion | `src/data/nfhs_extract.py` (NFHS-5 district factsheets, long → wide, 707 districts); `src/data/hmis_extract.py` (streaming parser for the official HMIS C2 item-wise archives, ~2.5 GB of HTML per year, counts pooled across available years) |
| 2. Harmonisation & features | `src/data/merge_nfhs_hmis.py` (within-state aliases → exact → `rapidfuzz` ≥ 90 auto / 75–89 review, HMIS split-district aggregation, full audit); `src/data/preprocess_impute.py` (HMIS-derived rates, C-section deficit, log OOP, missingness / anchor-proxy / VIF selection, state → zone → national KNN imputation fitted only on training folds) |
| 3. Stage 1 target | `src/target/ols_priority_classifier.py` — EB-smoothed facility MMR (mortality-only by default; morbidity weight configurable), OLS on institutional births, internally studentized residuals, 4 quadrants, anti-leakage firewall |
| 4. Stage 2 model | `src/models/train_evaluate.py` — **discordance task**: Priority vs Best Performer among high-access districts (configurable back to four classes); 80/20 stratified holdout, 5-fold stratified CV, balanced weights × 1.25 for class 1, Optuna (50 trials) for RF, GradientBoosting, XGBoost, LightGBM; class-1 decision threshold tuned on out-of-fold predictions to reach recall ≥ 0.82; isotonic calibration |
| 4. Stage 3 XAI | `src/explain/shap_profiler.py` — exact TreeSHAP with additivity check, five-domain roll-ups and national domain shares, top-5 local drivers → scheme mapping, master priority table |
| 5. Dashboard | `app/streamlit_app.py` — national choropleth & OLS quadrant plot, district scorecard (gauge, SHAP waterfall, deprivation radar vs state and best performers, policy card), what-if simulator with live TreeSHAP, priority table with CSV/Excel/PDF briefs, methods & audit tab |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # macOS: XGBoost/LightGBM also need `brew install libomp`

# Official data (expects the files listed in config/data_sources.yaml under data/raw/)
python -m src.run_pipeline               # ~10-20 min with 50 Optuna trials per model
python -m src.run_pipeline --trials 5    # quick run
python -m src.run_pipeline --rebuild-sources   # re-extract NFHS/HMIS interim tables

# Or a fully SYNTHETIC demo (overwrites the generated artefacts; the app labels it as synthetic)
python -m src.run_pipeline --demo

streamlit run app/streamlit_app.py
pytest -q
```

## Data sources

See `config/data_sources.yaml`. Place in `data/raw/`:

- `nfhs5_districts_source.csv` — NFHS-5 district factsheets (long format with `State, District, Indicator, NFHS 5`).
- `hmis_2017_18_official.zip`, `hmis_2018_19_official.zip`, `hmis_2019_20_official.zip` — HMIS *C2 Data Itemwise
  Monthly – All States and Districts Across Months*, public on the HMIS portal (URL pattern in `config/data_sources.yaml`).
  Counts are pooled across the three pre-COVID years; 2020-21 onward are login-only in this format.
- `india_districts_source.geojson` — district boundaries (simplified to ~1.1 MB at build time; `mapshaper` is used if it is
  on `PATH`, otherwise a built-in Douglas-Peucker simplifier).

## Key research choices and known limitations

- **Anchor exclusion.** `institutional_births_pct`, its public-facility sub-share, all HMIS outcome counts/rates and Stage-1
  outputs are forbidden as Stage-2 features (`FORBIDDEN_FEATURES`). Features correlating |r| > 0.9 with the anchor
  (e.g. skilled birth attendance) are dropped as proxies. All decisions are in `data/processed/feature_selection_audit.csv`.
- **Morbidity component.** HMIS publishes a single "obstetric complications attended" total, not separate PPH/eclampsia/
  sepsis/ruptured-uterus counts, so the morbidity rate uses that total.
- **HMIS window.** Maternal deaths and live births are pooled over 2017-18 to 2019-20 (median 58 deaths per modelled
  district), which overlaps the start of NFHS-5 fieldwork (2019-21) and avoids the COVID-disrupted 2020-21 year.
- **Indicators not in district factsheets.** JSY financial assistance, SC/ST and rural population share, male literacy
  (gender gap), FRU and blood-storage-unit counts are not published at district level in the sources used. HMIS proxies
  are used where they exist (blood units issued per 1,000 deliveries, JSSK free transport, severe anaemia Hb < 7).
- **Mirror data quality.** The community NFHS mirror stores suppressed (`*`) values as `0`; for indicators where zero is
  implausible those are treated as missing. "Third or higher order births" is excluded because its mirror values do not
  match the indicator's scale.
- **Unmatched districts.** Districts that NFHS-5 splits but HMIS 2019-20 does not (six new Assam districts,
  Mumbai/Mumbai Suburban) cannot be split without population weights and are excluded; districts with < 500 pooled live
  births are excluded as too noisy. ~70 newer districts have no polygon in the boundary file and are absent from the map
  only.
- **Why a binary Stage 2 (`modelling.task: discordance_binary`).** Priority and Best Performer districts both have
  above-threshold institutional births by construction. Training only on those districts stops coverage indicators
  (postnatal care, ANC) from acting as proxies for the excluded access anchor, so SHAP drivers describe *why high-access
  districts diverge*. Coverage Gap / Low Burden districts are still scored, but flagged `in_training_domain = False` and
  shown as an extrapolation ("quality-gap risk if access improved"). Set `task: four_class` for the original formulation.
- **Decision threshold.** The Priority cut-off is the highest probability threshold whose out-of-fold class-1 recall
  reaches `target_class_1_recall` (0.82); it is chosen on training folds only and stored with the model. Holdout metrics
  are reported both at argmax and at this threshold. Expect lower precision: this is a screening list for audits.
- **Validation (`python -m src.models.validate`, also run by the pipeline).** Compares random vs **state-grouped**
  5-fold CV with the selected model (thresholds re-chosen inside each training fold) and re-runs Stage 1 under six
  alternative settings. Findings that shaped the defaults: (1) the HMIS morbidity item is not comparable across states
  (Tamil Nadu median 56% of deliveries vs 5.4% nationally), so the target is **mortality-only**; (2) pooling three HMIS
  years cut label noise. Held-out-state ROC-AUC rose 0.54 → 0.61 → **0.71** (random CV 0.78). See
  `models/validation_report.json`.
- **External consistency check.** Our pooled facility MMR reproduces the published district estimates of Goli et al.
  (2022, *PLOS Global Public Health*, S3 Table; same HMIS window, state-calibrated to SRS): within-state Spearman median
  0.97 over 26 states; Priority districts sit at a median 85th within-state percentile of the published MMR
  (`reports/external_mmr_comparison.csv`). This verifies the extraction, not the clinical validity of the labels.
- **GradientBoosting.** sklearn's `GradientBoostingClassifier` is TreeSHAP-compatible only for binary problems; in
  four-class mode it is benchmarked but cannot be selected.
- **Associative, not causal.** What-if results are model sensitivity, not programme impact estimates.

## Outputs

| File | Content |
| --- | --- |
| `data/processed/master_district_dataset.csv` | Harmonised district dataset with Stage-1 risk, residuals and classes |
| `data/processed/priority_action_table.csv` | Ranked districts, risk score, residual, top-5 SHAP drivers, schemes, CRS-audit flag |
| `data/processed/shap_matrix_class1.parquet` | Pre-computed class-1 SHAP values and domain roll-ups for instant UI loading |
| `data/processed/domain_shares.csv` | National five-domain attribution |
| `data/processed/match_audit.csv`, `geo_match_audit.csv`, `feature_selection_audit.csv` | Audit trails |
| `data/processed/india_districts_simplified.geojson` | Simplified boundaries keyed by `state|district` |
| `models/benchmark_metrics.json` | CV / out-of-fold / holdout metrics for all four models, calibration, best params |
| `reports/figures/*.png` | EDA and SHAP publication figures |
| `reports/methodology_report.html` | Full methodology and results write-up (figures in `reports/report_assets/`) |
| `reports/external_mmr_comparison.csv` | District comparison with published HMIS-based MMR (Goli et al. 2022) |

## Deployment (Hugging Face Spaces)

Generated artefacts are deliberately not committed to Git, so deployment uploads them from a local build:

```bash
python -m src.run_pipeline
pip install huggingface_hub
HF_TOKEN=hf_xxx python scripts/deploy_hf.py --space <username>/maternal-mortality-xai
```

The Space uses the Docker SDK (`Dockerfile`, port 7860). `.github/workflows/ci.yml` runs the test suite (including an
end-to-end synthetic pipeline run) on every push.

## Repository layout

```
config/      pipeline, feature-domain, alias, scheme and source configuration
src/         data, target, models, explain, geo and reports packages + run_pipeline.py
app/         Streamlit app, tab components and CSS
notebooks/   01-04 walkthroughs of each pipeline stage
tests/       leakage, Stage-1, preprocessing, merge/extract, SHAP additivity and pipeline smoke tests
scripts/     Hugging Face deployment helper
```
