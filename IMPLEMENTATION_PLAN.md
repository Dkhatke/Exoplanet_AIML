# IMPLEMENTATION PLAN — AIML_Exoplanet
## Phase-by-Phase Tasks, Subphases, Checklists, and Acceptance Criteria

> **Last Updated:** Phase 0 Planning
> **Current Phase:** NOT STARTED — Awaiting Approval
> **Total Estimated Effort:** 21 working days across 8 phases

---

## HOW TO USE THIS FILE

- Each Phase has numbered Subphases
- Each Subphase has a Task List with checkboxes
- Mark tasks [x] as you complete them
- Each Phase has Acceptance Criteria that must ALL pass before moving to next phase
- Do NOT skip phases or subphases
- Do NOT start Phase N+1 before Phase N acceptance criteria are met

---

## PHASE 0: Audit, Scaffolding, Configuration & Offline Test Harness
**Estimated Effort:** 2 working days
**Goal:** Clean, reproducible environment with pinned deps, expanded config, and offline test harness.
**Status:** [ ] NOT STARTED

### Dependencies
- None (this is the first phase)

### Phase 0.1 — Dependency Pinning and Environment Setup

**Objective:** Pin all package versions; add missing packages; verify installation.

Tasks:
- [ ] 0.1.1 Rewrite `requirements.txt` with pinned versions (see REQUIREMENTS_SPEC.md Part B)
    - numpy>=1.24.0,<2.0.0
    - pandas>=2.0.0,<3.0.0
    - astropy>=5.3.0,<7.0.0
    - lightkurve>=2.4.2,<3.0.0
    - wotan>=1.10,<2.0.0
    - pyarrow>=13.0.0,<16.0.0
    - scipy>=1.11.0,<2.0.0
    - scikit-learn>=1.3.0,<2.0.0
    - xgboost>=1.7.6,<3.0.0
    - joblib>=1.3.0,<2.0.0
    - astroquery>=0.4.6,<1.0.0
    - matplotlib>=3.7.0,<4.0.0
    - plotly>=5.16.0,<6.0.0
    - streamlit>=1.28.0,<2.0.0
    - pytest>=7.4.0,<9.0.0
    - pytest-cov>=4.1.0,<6.0.0
- [ ] 0.1.2 Create virtual environment: `python3.11 -m venv venv`
- [ ] 0.1.3 Install: `pip install -r requirements.txt`
- [ ] 0.1.4 Verify: `python -c "import lightkurve, wotan, astroquery; print('OK')"`

### Phase 0.2 — Configuration Expansion

**Objective:** Expand `config.py` from single-star constants to project-wide typed configuration.

Tasks:
- [ ] 0.2.1 Rename sections in config.py: INGESTION, PREPROCESSING, DETECTION, ML, TRIAGE, PATHS, LIMITS
- [ ] 0.2.2 Add INGESTION constants:
    - PREFERRED_AUTHOR = "SPOC"
    - FALLBACK_AUTHOR = "QLP"
    - MAX_SECTORS = 6
- [ ] 0.2.3 Add DETECTION constants:
    - PERIOD_MIN_DAYS = 0.5
    - PERIOD_MAX_DAYS = 15.0
    - BLS_OVERSAMPLE = 10
    - MAX_BLS_FREQUENCY_BINS = 50000
    - TRANSIT_DURATION_MIN = 0.01
    - TRANSIT_DURATION_MAX = 0.25
- [ ] 0.2.4 Add LIMITS constants:
    - MAX_PARALLEL_WORKERS = 2
    - MAX_SECONDS_PER_TARGET = 120
    - PILOT_SIZE = 5
    - INITIAL_DATASET_SIZE = 100
- [ ] 0.2.5 Add TRIAGE constants:
    - CLASS_LABELS = ["PLANET", "EB", "VARIABLE", "NOISE"]
    - REVIEW_PROB_THRESHOLD = 0.60
    - CLASS_CONFLICT_PRIORITY = {"PLANET": 4, "EB": 3, "VARIABLE": 2, "NOISE": 1}
- [ ] 0.2.6 Add PATHS constants:
    - DATA_DIR = "data/"
    - CATALOG_DIR = "data/catalogs/"
    - CACHE_DIR = "data/cache/"
    - OUTPUT_DIR = "outputs/"
    - MODEL_PATH = "models/triage_model.joblib"
    - DATASET_PATH = "data/dataset_100.parquet"
    - TARGETS_MANIFEST = "data/catalogs/targets_100.csv"
- [ ] 0.2.7 Remove hardcoded HD 209458 stellar constants (STAR_RADIUS_SOLAR, etc.) from config.py
      NOTE: Keep STAR_NAME = "HD 209458" as a demo default only

### Phase 0.3 — Project Directory Scaffolding

**Objective:** Create all new directories and empty placeholder `__init__.py` files.

Tasks:
- [ ] 0.3.1 Create directories:
    - `src/ingestion/`, `src/preprocessing/`, `src/detection/`
    - `src/features/`, `src/ml/`, `src/recycler/`, `src/utils/`
    - `ui/components/`
    - `scripts/`
    - `tests/`
    - `docs/`
    - `data/catalogs/`, `data/cache/`
    - `models/`, `outputs/`
- [ ] 0.3.2 Add `__init__.py` to each new `src/` subdirectory
- [ ] 0.3.3 Add `__init__.py` to `tests/`
- [ ] 0.3.4 Move existing src files temporarily:
    - `src/fetcher.py` --> `src/ingestion/fetcher.py` (refactor in Phase 1)
    - `src/preprocessor.py` --> `src/preprocessing/preprocessor.py` (refactor in Phase 1)
    - `src/detector.py` --> `src/detection/detector.py` (refactor in Phase 2)
    - `src/analyzer.py` --> keep as-is for now (refactor in Phase 2/5)
    - `src/ml_model.py` --> keep as-is for now (replace in Phase 4)
    - NOTE: Update imports in app.py after moving files

### Phase 0.4 — Structured Logger

**Objective:** Centralized logging module for all pipeline components.

Tasks:
- [ ] 0.4.1 Create `src/utils/logger.py`:
    - Function: `get_logger(name: str) -> logging.Logger`
    - Format: `[timestamp] [LEVEL] [module] message`
    - Log to console AND to `outputs/pipeline.log`
    - Include TIC ID in log records where applicable

### Phase 0.5 — Offline Test Harness

**Objective:** Build synthetic offline fixtures so all unit tests run with zero internet.

Tasks:
- [ ] 0.5.1 Create `tests/conftest.py` with pytest fixtures:
    - `synthetic_planet_lc`: 1000 cadences, period=3.524d, depth=0.015, flat baseline
    - `synthetic_eb_lc`: 1000 cadences, period=2.0d, primary depth=0.05, secondary depth=0.02
    - `synthetic_variable_lc`: 1000 cadences, sinusoidal period=0.8d, amplitude=0.02
    - `synthetic_noise_lc`: 1000 cadences, Gaussian noise std=500ppm, no periodic signal
    - All fixtures return `lightkurve.LightCurve` objects (not astropy Quantities)
- [ ] 0.5.2 Create `tests/test_harness.py`:
    - Test that all 4 fixtures load correctly
    - Test that planet_lc has a detectable dip at phase 0
    - Test that noise_lc has no peak > 3-sigma in BLS
    - All tests must complete in < 5 seconds total

### Phase 0.6 — Update .gitignore

**Objective:** Ensure new directories are properly gitignored.

Tasks:
- [ ] 0.6.1 Add to .gitignore:
    - `data/cache/`
    - `data/catalogs/*.parquet`
    - `data/dataset_100.parquet`
    - `models/`
    - `outputs/`
    - `venv/`
    - `.pytest_cache/`
    - `__pycache__/`

### Phase 0 Acceptance Criteria

- [ ] AC-0.1: `pip install -r requirements.txt` completes without errors on Python 3.11
- [ ] AC-0.2: `python -c "import lightkurve, wotan, astroquery, xgboost, pyarrow"` succeeds
- [ ] AC-0.3: `pytest tests/test_harness.py -v` passes all tests offline in < 10 seconds
- [ ] AC-0.4: `config.py` has no hardcoded stellar constants; all paths use constants
- [ ] AC-0.5: All new directories exist with `__init__.py` files
- [ ] AC-0.6: `streamlit run app.py` still starts without import errors

---

## PHASE 1: Multi-Sector Ingestion and Preprocessing
**Estimated Effort:** 2 working days
**Goal:** Download, stitch, and detrend multiple TESS sectors; cache in Parquet; preserve transit depth.
**Status:** [ ] NOT STARTED

### Dependencies
- Phase 0 complete (all acceptance criteria passed)

### Phase 1.1 — Multi-Sector Fetcher

**Objective:** Replace single-sector pickle fetcher with multi-sector Parquet fetcher.

File: `src/ingestion/fetcher.py` (REFACTOR of old `src/fetcher.py`)

Tasks:
- [ ] 1.1.1 Function `search_tess_sectors(tic_id, author="SPOC")`:
    - Returns lightkurve SearchResult with all available sectors
    - Filters to preferred_author first, falls back to QLP
    - Logs how many sectors found
- [ ] 1.1.2 Function `download_sectors(tic_id, max_sectors=6, author="SPOC")`:
    - Downloads up to max_sectors sectors
    - Returns list of LightCurve objects (one per sector)
    - Imposes 120-second timeout per download
    - Calls cache_manager.save() after each sector download
    - Calls cache_manager.load() first if cache exists
- [ ] 1.1.3 Error handling:
    - No data found: raise ValueError with helpful message
    - Timeout: log warning, continue with sectors already downloaded
    - Cache corrupted: delete and re-download

### Phase 1.2 — Parquet Cache Manager

**Objective:** Replace pickle cache with Parquet format with schema validation.

File: `src/ingestion/cache_manager.py` (NEW)

Tasks:
- [ ] 1.2.1 Function `save_lightcurve(lc, tic_id, sector, pipeline, cache_dir)`:
    - Converts LightCurve to DataFrame: time, flux, flux_err, quality, sector
    - Adds metadata columns: tic_id, pipeline, lightkurve_version, download_timestamp
    - Writes atomically: write to temp path, rename to final path
    - File naming: `{cache_dir}/tic_{tic_id}_s{sector:04d}_{pipeline}.parquet`
- [ ] 1.2.2 Function `load_lightcurve(tic_id, sector, pipeline, cache_dir)`:
    - Reads Parquet file; validates schema
    - Checks lightkurve_version -- invalidate if version changed
    - Returns LightCurve object or None if not cached
- [ ] 1.2.3 Function `list_cached_sectors(tic_id, cache_dir)`:
    - Returns list of (sector, pipeline) tuples available in cache

### Phase 1.3 — Sector Stitcher

**Objective:** Normalize individual sectors and stitch into a continuous baseline.

File: `src/ingestion/stitcher.py` (NEW)

Tasks:
- [ ] 1.3.1 Function `normalize_sector(lc)`:
    - Normalize flux to median = 1.0 within each sector
    - Remove quality-flagged cadences (quality != 0)
    - Returns normalized LightCurve
- [ ] 1.3.2 Function `stitch_sectors(lc_list)`:
    - Calls normalize_sector on each
    - Stitches using lightkurve LightCurveCollection.stitch()
    - Verifies no time discontinuities > 14 days are bridged incorrectly
    - Returns single stitched LightCurve sorted by time
- [ ] 1.3.3 Unit test: Two synthetic sectors stitch without step discontinuities > 0.005 in flux

### Phase 1.4 — Stellar Parameter Resolver

**Objective:** Dynamically retrieve stellar parameters (R*, M*, Teff) from TIC.

File: `src/ingestion/stellar_resolver.py` (NEW)

Tasks:
- [ ] 1.4.1 Function `resolve_stellar_params(tic_id)`:
    - Query TIC via `astroquery.mast.Catalogs.query_criteria(catalog="Tic", ID=tic_id)`
    - Extract: rad (stellar radius in R_sun), mass (M_sun), Teff (K), logg
    - Return dict: {"radius_solar": float, "mass_solar": float, "teff_k": float, "logg": float}
    - If column is None/missing: return default solar values with warning logged
- [ ] 1.4.2 Function `get_stellar_params_or_default(tic_id)`:
    - Wrapper that catches exceptions; always returns valid dict
    - Logs WARNING if defaults used

### Phase 1.5 — Transit-Masked Detrending Preprocessor

**Objective:** Detrend stitched light curve while preserving transit depth.

File: `src/preprocessing/preprocessor.py` (REFACTOR)

Tasks:
- [ ] 1.5.1 Two-pass detrending function `detrend_lc(lc, window_length_days=0.5)`:
    - PASS 1: Quick BLS scan to detect candidate period and mask windows
    - Mask cadences within +/- 1.5x transit_duration of each predicted transit
    - PASS 2: Run wotan.flatten() on masked light curve (biweight window)
    - Apply detrend trend to full unmasked light curve
    - Returns: lc_raw_norm (normalized, not detrended), lc_detrended (flat + transit preserved)
- [ ] 1.5.2 Asymmetric flare/outlier clipping:
    - Clip upward outliers > +3 sigma (flares)
    - Clip downward outliers > +7 sigma (instrument spikes)
    - Do NOT clip real transit depths
- [ ] 1.5.3 Unit test: Transit depth in lc_detrended is within 5% of injected depth in synthetic_planet_lc

### Phase 1.6 — Integration: Update app.py imports

Tasks:
- [ ] 1.6.1 Update `app.py` to import from new module paths:
    - `from src.ingestion.fetcher import download_sectors`
    - `from src.ingestion.stitcher import stitch_sectors`
    - `from src.preprocessing.preprocessor import detrend_lc`
- [ ] 1.6.2 Verify `streamlit run app.py` starts without errors

### Phase 1 Acceptance Criteria

- [ ] AC-1.1: `pytest tests/test_ingestion.py tests/test_preprocessor.py -v` all pass offline
- [ ] AC-1.2: Multi-sector stitch produces a continuous light curve without baseline jumps > 0.005
- [ ] AC-1.3: Transit depth after two-pass detrending is within 5% of input depth
- [ ] AC-1.4: All light curves cached as valid Parquet files with required schema columns
- [ ] AC-1.5: Stellar parameters returned as dict; Solar defaults used with logged WARNING when TIC data missing
- [ ] AC-1.6: `streamlit run app.py` still starts without errors

---

## PHASE 2: Detection and Vetting Feature Extraction
**Estimated Effort:** 3 working days
**Goal:** Upgrade BLS to frequency-linear bounded grid; extract 20+ physics-informed diagnostic features.
**Status:** [ ] NOT STARTED

### Dependencies
- Phase 1 complete (all acceptance criteria passed)

### Phase 2.1 — Frequency-Linear BLS Detector

**Objective:** Replace linear period grid with mathematically correct frequency-linear grid.

File: `src/detection/detector.py` (REFACTOR)

Tasks:
- [ ] 2.1.1 Function `compute_frequency_grid(t_baseline, period_min=0.5, oversample=10)`:
    - f_max = 1.0 / period_min
    - f_min = 3.0 / t_baseline  (require at least 3 transits)
    - delta_f = 1.0 / (oversample * t_baseline)
    - n_freqs = min(int((f_max - f_min) / delta_f), MAX_BLS_FREQUENCY_BINS)
    - Returns np.linspace(f_min, f_max, n_freqs) in frequency space
    - Converts back to periods array
    - Logs actual number of bins used
- [ ] 2.1.2 Function `run_bls(lc_detrended, stellar_params=None)`:
    - Computes frequency grid from light curve baseline
    - Tests transit durations from 0.01 to 0.25 days (5 duration values, log-spaced)
    - Runs BoxLeastSquares for each duration; keeps best peak
    - Returns: best_period, t0, duration, bls_power, full_result, top3_candidates
- [ ] 2.1.3 Harmonic check function `check_harmonics(best_period, bls_result)`:
    - Tests P/2 and 2P for higher power (EB period doubling)
    - Returns: corrected_period, harmonic_flag (True if harmonic is stronger)
- [ ] 2.1.4 Add execution timeout:
    - BLS must complete within 60 seconds
    - If timeout: return best partial result with timeout_flag=True

### Phase 2.2 — Transit Metric Features

**Objective:** Compute astrophysical features that discriminate planets from EBs.

File: `src/features/transit_metrics.py` (NEW)

Tasks:
- [ ] 2.2.1 Function `compute_odd_even_depths(lc_detrended, period, t0, duration)`:
    - Separate odd and even transit events
    - Compute: delta_odd, delta_even, delta_odd_even_significance (z-score)
    - Returns dict with: odd_depth, even_depth, odd_even_diff, odd_even_z
- [ ] 2.2.2 Function `compute_secondary_eclipse(lc_detrended, period, t0, duration)`:
    - Fold at period, check phase 0.5 for dip
    - Returns: secondary_depth, secondary_snr, secondary_detected (bool)
- [ ] 2.2.3 Function `compute_transit_shape(lc_folded, duration, period)`:
    - Compute ingress fraction: T_12 / T_14 (ratio of ingress to full transit duration)
    - V-shape metric: flux curvature at transit center
    - Returns: ingress_fraction, v_shape_metric, shape_quality
- [ ] 2.2.4 Function `compute_centroid_flux_ratio(lc_detrended, period, t0, duration)`:
    - In-transit vs out-of-transit flux ratio
    - Returns: flux_ratio, mean_flux_in, mean_flux_out

### Phase 2.3 — Variability Metric Features

**Objective:** Compute features that discriminate periodic variables from transiting signals.

File: `src/features/variability_metrics.py` (NEW)

Tasks:
- [ ] 2.3.1 Function `compute_out_of_transit_variability(lc_detrended, period, t0, duration)`:
    - Mask transit windows; compute variability of out-of-transit flux
    - Returns: oot_mad, oot_rms, oot_p2p (peak-to-peak), oot_skew
- [ ] 2.3.2 Function `compute_lomb_scargle_peak(lc_detrended, period_range=(0.1, 50.0))`:
    - Compute Lomb-Scargle periodogram
    - Return peak frequency, peak power, and False Alarm Probability (FAP)
    - Returns: ls_peak_period, ls_peak_power, ls_fap
- [ ] 2.3.3 Function `compute_flux_percentile_ratio(lc_detrended)`:
    - p5 to p95 flux percentile range (captures variability amplitude)
    - Returns: p5_p95_ratio, p10_p90_ratio, median_flux

### Phase 2.4 — Unified Feature Extractor

**Objective:** Single entry-point that takes a light curve and returns the complete feature dictionary.

File: `src/features/feature_extractor.py` (NEW)

Tasks:
- [ ] 2.4.1 Function `extract_features(lc_detrended, stellar_params=None)`:
    - Calls run_bls -> get period, t0, duration
    - Calls compute_odd_even_depths
    - Calls compute_secondary_eclipse
    - Calls compute_transit_shape
    - Calls compute_out_of_transit_variability
    - Calls compute_lomb_scargle_peak
    - Calls compute_flux_percentile_ratio
    - Computes classic features: depth, duration, snr, symmetry, sharpness, n_transits
    - Returns: flat dict of all ~25 features; NaN replaced by 0.0 with warning
    - Returns: feature_dict, bls_output, extraction_metadata
- [ ] 2.4.2 FEATURE_NAMES constant: ordered list of all 25 feature names
    Must include: depth, duration, snr, symmetry, sharpness, n_transits,
    odd_depth, even_depth, odd_even_diff, odd_even_z,
    secondary_depth, secondary_snr, secondary_detected,
    ingress_fraction, v_shape_metric,
    mean_flux_in, mean_flux_out, flux_ratio,
    oot_mad, oot_rms, oot_p2p, oot_skew,
    ls_peak_power, ls_fap, p5_p95_ratio

### Phase 2.5 — Unit Tests for Features

File: `tests/test_features.py`

Tasks:
- [ ] 2.5.1 Test on synthetic_planet_lc:
    - odd_even_z < 2 (similar odd/even depths -- planet-like)
    - secondary_detected == False
    - v_shape_metric < 0.3 (U-shaped transit)
- [ ] 2.5.2 Test on synthetic_eb_lc:
    - odd_even_z > 3 (very different depths -- EB-like)
    - secondary_detected == True
- [ ] 2.5.3 Test on synthetic_variable_lc:
    - ls_peak_power > 0.5 (strong LS periodicity)
    - secondary_detected == False
- [ ] 2.5.4 Test all: feature_dict has no NaN values

### Phase 2 Acceptance Criteria

- [ ] AC-2.1: BLS frequency grid uses <= 50,000 bins and logs actual count
- [ ] AC-2.2: BLS completes in < 60 seconds on a 3-sector stitched light curve
- [ ] AC-2.3: Feature extractor returns all 25 features with no NaN values
- [ ] AC-2.4: `pytest tests/test_features.py -v` all pass offline in < 30 seconds
- [ ] AC-2.5: Odd/even test distinguishes synthetic_planet from synthetic_eb correctly

---

## PHASE 3: Independently Labeled Dataset Construction
**Estimated Effort:** 3 working days
**Goal:** Build the 100-star ground-truth catalog manifest and extract features for all 100 stars.
**Status:** [ ] NOT STARTED

### Dependencies
- Phase 2 complete (all acceptance criteria passed)

### Phase 3.1 — Catalog Acquisition

**Objective:** Download and save the 4 catalog CSV files locally.

File: `scripts/build_dataset.py` (part 1)

Tasks:
- [ ] 3.1.1 Download NASA Exoplanet Archive TOI catalog:
    - URL: https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+toi,tic_id,tfopwg_disp,toi_depth_snr+from+toi+where+tfopwg_disp+in+('CP','KP')&format=csv
    - Save to: data/catalogs/toi_confirmed.csv
    - Log record count
- [ ] 3.1.2 Download Villanova TESS-EB catalog:
    - URL: see REQUIREMENTS_SPEC.md Source 2
    - Save to: data/catalogs/villanova_eb.csv
    - Log record count
- [ ] 3.1.3 Query Gaia DR3 variable stars:
    - Use astroquery.gaia.Gaia.launch_job() with ADQL
    - Filter: best_class_name IN ('DSCT_SXPHE','RR','CEP','ROT','LPV')
    - Save to: data/catalogs/gaia_dr3_variables.csv
- [ ] 3.1.4 Verify all 3 catalogs saved with correct column names

### Phase 3.2 — Target Manifest Construction

**Objective:** Build the master 100-star manifest (targets_100.csv).

Tasks:
- [ ] 3.2.1 Select 25 PLANET targets from toi_confirmed.csv:
    - Filter by SNR > 7
    - Sort by SNR descending, take top 30 as candidates
    - Verify SPOC availability for each via lightkurve.search_lightcurve
    - Take 25 with confirmed SPOC availability
- [ ] 3.2.2 Select 25 EB targets from villanova_eb.csv:
    - Filter morph < 0.5, period between 0.5 and 15 days
    - Randomly sample 30 candidates
    - Verify SPOC/QLP availability for each
    - Take 25 with confirmed availability
- [ ] 3.2.3 Select 25 VARIABLE targets from gaia_dr3_variables.csv:
    - Cross-match to TIC via position (2 arcsec radius)
    - Verify availability; exclude any already in PLANET or EB lists
    - Take 25 with SPOC or QLP availability
- [ ] 3.2.4 Select 25 NOISE targets from TIC:
    - Query TIC for G/K stars, Tmag 8-12
    - Verify NOT in TOI, EB, or Gaia variable lists
    - Download and compute MAD; keep MAD < 500 ppm
    - Take 25 qualifying stars
- [ ] 3.2.5 Write data/catalogs/targets_100.csv with required schema
- [ ] 3.2.6 Write data/catalogs/label_conflicts.csv (document any conflicts found)
- [ ] 3.2.7 Verify: len(unique(tic_id)) == 100 across all classes

### Phase 3.3 — Pilot Run (5 Stars)

**STOP: Run pilot before full batch. Validate timing and output quality.**

Tasks:
- [ ] 3.3.1 Run pipeline on 5-star pilot set (see PROJECT_CONTEXT.md Section 4.5)
- [ ] 3.3.2 Verify per-star runtime < 120 seconds
- [ ] 3.3.3 Verify feature_dict has no NaN for all 5 pilot stars
- [ ] 3.3.4 Verify total pilot runtime < 2 minutes with 1 worker
- [ ] 3.3.5 Log detected period vs catalog period for each pilot star (audit only, not model input)
- [ ] 3.3.6 DECISION POINT: If pilot fails timing or quality, fix before proceeding to full batch

### Phase 3.4 — Full Batch Feature Extraction (100 Stars)

**NOTE: Only run after pilot (Phase 3.3) succeeds.**
**NOTE: Estimated total time: ~25 minutes with 2 workers. Do NOT exceed 30 min without approval.**

Tasks:
- [ ] 3.4.1 Run build_dataset.py for all 100 targets:
    - --workers 2 (never > 2)
    - --timeout 120 (per target)
    - --resume (skip already-processed TIC IDs)
- [ ] 3.4.2 Save feature table to data/dataset_100.parquet
- [ ] 3.4.3 Log failed targets; retry once; report final failure count
- [ ] 3.4.4 If > 10 targets fail: stop and report before continuing

### Phase 3.5 — Dataset Integrity Validation

File: `tests/test_dataset_integrity.py`

Tasks:
- [ ] 3.5.1 Verify len(dataset) == 100 (exact)
- [ ] 3.5.2 Verify len(unique(tic_id)) == 100 (no duplicates)
- [ ] 3.5.3 Verify class distribution: 25 each for PLANET, EB, VARIABLE, NOISE
- [ ] 3.5.4 Verify all 25 FEATURE_NAMES columns are present
- [ ] 3.5.5 Verify zero NaN values in feature columns
- [ ] 3.5.6 Verify provenance: each row has tic_id, class_label, source_catalog

### Phase 3 Acceptance Criteria

- [ ] AC-3.1: All 4 catalog CSV files saved in data/catalogs/
- [ ] AC-3.2: targets_100.csv has exactly 100 rows, 25 per class
- [ ] AC-3.3: Pilot run (5 stars) completes in < 2 minutes with no NaN features
- [ ] AC-3.4: dataset_100.parquet exists with 100 rows and 25 feature columns
- [ ] AC-3.5: `pytest tests/test_dataset_integrity.py -v` all pass
- [ ] AC-3.6: All feature extraction was ephemeris-blind (verified in build_dataset.py logs)

---

## PHASE 4: Four-Class Triage Classifier and Evaluation
**Estimated Effort:** 3 working days
**Goal:** Train a calibrated 4-class classifier; evaluate honestly with GroupKFold; implement REVIEW routing.
**Status:** [ ] NOT STARTED

### Dependencies
- Phase 3 complete (all acceptance criteria passed)

### Phase 4.1 — Classifier Implementation

File: `src/ml/classifier.py` (NEW)

Tasks:
- [ ] 4.1.1 Build classifier pipeline:
    - StandardScaler
    - XGBClassifier(max_depth=4, n_estimators=200, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                   reg_lambda=1.0, use_label_encoder=False, eval_metric='mlogloss',
                   random_state=42)
    - Wrap with CalibratedClassifierCV(cv=5, method='sigmoid')
- [ ] 4.1.2 Function `train_triage_model(X, y, groups)`:
    - Encodes labels: PLANET=0, EB=1, VARIABLE=2, NOISE=3
    - Fits calibrated pipeline
    - Returns fitted model
- [ ] 4.1.3 Function `predict_triage(model, feature_dict)`:
    - Returns: predicted_class (str), class_probabilities (dict), review_flag (bool)
    - review_flag = True if max(class_probabilities.values()) < REVIEW_PROB_THRESHOLD
    - If review_flag: predicted_class = "REVIEW"

### Phase 4.2 — Evaluation Module

File: `src/ml/evaluator.py` (NEW)

Tasks:
- [ ] 4.2.1 Function `evaluate_with_group_kfold(X, y, groups, n_splits=5)`:
    - Uses StratifiedGroupKFold(n_splits=5)
    - For each fold: train on 80 stars, test on 20 stars
    - Compute: precision, recall, F1 per class; macro F1; multiclass log-loss
    - Verify no TIC ID in both train and test fold (assert)
    - Returns: metrics_dict, fold_results
- [ ] 4.2.2 Function `plot_confusion_matrix(y_true, y_pred, labels)`:
    - Saves confusion matrix to outputs/confusion_matrix.png
- [ ] 4.2.3 Function `plot_calibration_curve(model, X, y)`:
    - Saves calibration plot to outputs/calibration_curve.png
- [ ] 4.2.4 Function `plot_feature_importance(model, feature_names)`:
    - Saves feature importance to outputs/feature_importance.png

### Phase 4.3 — Training CLI Script

File: `scripts/train_classifier.py` (NEW)

Tasks:
- [ ] 4.3.1 CLI arguments:
    - --dataset data/dataset_100.parquet
    - --output models/triage_model.joblib
    - --evaluate (run GroupKFold cross-validation before saving)
- [ ] 4.3.2 Load dataset, encode labels, split features
- [ ] 4.3.3 Run GroupKFold evaluation; print metrics; save confusion matrix
- [ ] 4.3.4 Train final model on all 100 stars; save to models/triage_model.joblib
- [ ] 4.3.5 Save metrics to outputs/model_metrics.json

### Phase 4.4 — Deprecate Old ml_model.py

Tasks:
- [ ] 4.4.1 Mark `src/ml_model.py` as DEPRECATED in file header comment
- [ ] 4.4.2 Remove build_dataset() call from app.py
- [ ] 4.4.3 Remove train_model() call from app.py
- [ ] 4.4.4 Update app.py to load pre-trained model from MODEL_PATH

### Phase 4.5 — Unit Tests

File: `tests/test_classifier.py`

Tasks:
- [ ] 4.5.1 Test that predict_triage returns all 4 classes + REVIEW
- [ ] 4.5.2 Test that probabilities sum to 1.0
- [ ] 4.5.3 Test that GroupKFold has no TIC ID overlap between folds
- [ ] 4.5.4 Test that model loads correctly from joblib file

### Phase 4 Acceptance Criteria

- [ ] AC-4.1: GroupKFold evaluation shows macro F1 > 0.65 (accept lower for N=100; honesty required)
- [ ] AC-4.2: No TIC ID appears in both train and test fold (verified by assertion in test)
- [ ] AC-4.3: predict_triage returns REVIEW when max probability < 0.60
- [ ] AC-4.4: Class probabilities always sum to 1.0
- [ ] AC-4.5: `pytest tests/test_classifier.py -v` all pass
- [ ] AC-4.6: models/triage_model.joblib exists and loads without error

---

## PHASE 5: Specialized False Positive Recycler Modules
**Estimated Effort:** 3 working days
**Goal:** Dedicated analysis modules for each of the 4 triage outputs.
**Status:** [x] COMPLETE

### Dependencies
- Phase 4 complete (all acceptance criteria passed)

### Phase 5.1 — Planet Characterization Module

File: `src/recycler/planet_module.py` (NEW)

Tasks:
- [x] 5.1.1 Function `analyze_planet_candidate(lc_detrended, bls_output, stellar_params)`:
    - Compute Rp/Rs from transit depth (depth = (Rp/Rs)^2)
    - Compute Rp in Earth radii using stellar_params["radius_solar"]
    - Compute semi-major axis via Kepler's 3rd Law using stellar_params["mass_solar"]
    - Compute equilibrium temperature using stellar_params["teff_k"]
    - Estimate transit duration consistency: compare predicted vs measured duration
    - Returns: planet_params dict (period, Rp_Rearth, a_AU, T_eq_K, duration_consistency)
- [x] 5.1.2 NOTE: All outputs labeled as CANDIDATE, not confirmed planet
- [x] 5.1.3 NOTE: Uses dynamic stellar_params -- NOT hardcoded constants

### Phase 5.2 — Eclipsing Binary Diagnostic Module

File: `src/recycler/eb_module.py` (NEW)

Tasks:
- [x] 5.2.1 Function `analyze_eb(lc_detrended, bls_output, feature_dict)`:
    - Primary eclipse depth and SNR
    - Secondary eclipse depth and phase offset from phase 0.5
    - Odd/even depth ratio and significance
    - Morphology estimate: contact (morph~1) vs detached (morph~0) based on light curve shape
    - Period doubling test: check if P/2 gives deeper primary eclipse
    - Returns: eb_params dict with all diagnostics
- [x] 5.2.2 Classify EB subtype: DETACHED, CONTACT, SEMI-DETACHED

### Phase 5.3 — Variable Star Diagnostic Module

File: `src/recycler/variable_module.py` (NEW)

Tasks:
- [x] 5.3.1 Function `analyze_variable(lc_raw_norm, lc_detrended)`:
    - Compute Lomb-Scargle periodogram
    - Identify dominant period and harmonics
    - Compute variability amplitude (min-max) and mean-to-peak
    - Classify variability mode: PULSATION, ROTATION, QUASI-PERIODIC
    - Returns: variable_params dict
- [x] 5.3.2 Cross-check Gaia DR3 variability class if TIC ID in Gaia catalog
- [x] 5.3.3 Return possible stellar variability type labels

### Phase 5.4 — Noise / Instrumental Diagnostic Module

File: `src/recycler/noise_module.py` (NEW)

Tasks:
- [x] 5.4.1 Function `analyze_noise(lc_detrended, feature_dict)`:
    - Compute CDPP (Combined Differential Photometric Precision) at multiple timescales
    - Identify high-frequency noise vs low-frequency instrumental systematics
    - Check for correlation with quality flags
    - Check for periodic instrumental signal at 13.5 days (TESS orbital period)
    - Returns: noise_params dict (cdpp_1hr, cdpp_6hr, systematic_flag, instrument_period_flag)

### Phase 5.5 — Recycler Router

File: `src/recycler/__init__.py` (updated)

Tasks:
- [x] 5.5.1 Function `route_to_recycler(predicted_class, lc_detrended, lc_raw_norm, bls_output, feature_dict, stellar_params)`:
    - Routes to appropriate module based on predicted_class
    - Returns: recycler_output dict

### Phase 5.6 — Unit Tests

File: `tests/test_recycler_modules.py`

Tasks:
- [x] 5.6.1 Test planet_module on synthetic_planet_lc: returns valid Rp and a_AU
- [x] 5.6.2 Test eb_module on synthetic_eb_lc: detects secondary eclipse
- [x] 5.6.3 Test variable_module on synthetic_variable_lc: detects dominant period
- [x] 5.6.4 Test noise_module on synthetic_noise_lc: returns noise diagnostics dict

### Phase 5 Acceptance Criteria

- [x] AC-5.1: planet_module uses dynamic stellar_params (not config hardcoded constants)
- [x] AC-5.2: eb_module correctly identifies secondary eclipse in synthetic_eb_lc
- [x] AC-5.3: variable_module detects period in synthetic_variable_lc within 5% of injected
- [x] AC-5.4: `pytest tests/test_recycler_modules.py -v` all pass offline
- [x] AC-5.5: route_to_recycler returns appropriate output for all 5 class labels including REVIEW

---

## PHASE 6: Streamlit UX Overhaul
**Estimated Effort:** 3 working days
**Goal:** Modern multi-tab dashboard replacing app.py; fast inference from pre-trained model.
**Status:** [x] COMPLETE

### Dependencies
- Phase 5 complete (all acceptance criteria passed)

### Phase 6.1 — UI Architecture

Tasks:
- [x] 6.1.1 Restructure app.py into 5 tabs:
    - Tab 1: Multi-Sector Light Curves (raw + detrended, sector selector)
    - Tab 2: Signal Detection (BLS periodogram, top candidates, harmonic flags)
    - Tab 3: Triage Classification (4-class probability gauge, REVIEW indicator)
    - Tab 4: Recycler Diagnostics (contextual panel based on predicted class)
    - Tab 5: Provenance & Report (star metadata, audit log, download report)
- [x] 6.1.2 Sidebar: TIC ID input, star name input, sector selection, run button
- [x] 6.1.3 Session state: cache results so re-running tabs does not re-trigger analysis
- [x] 6.1.4 NO model training in app.py -- only model.predict() from pre-trained model

### Phase 6.2 — Light Curve Component

File: `ui/components/lightcurve_view.py` (NEW)

Tasks:
- [x] 6.2.1 Interactive Plotly chart: raw normalized vs detrended flux
- [x] 6.2.2 Sector highlighting: different color per sector
- [x] 6.2.3 Phase-folded transit view (raw data + binned median)
- [x] 6.2.4 Transit window annotation on phase-folded plot

### Phase 6.3 — Triage Gauge Component

File: `ui/components/triage_gauge.py` (NEW)

Tasks:
- [x] 6.3.1 Bar chart or radar chart showing probability for each of 4 classes
- [x] 6.3.2 Highlight REVIEW status with distinct visual indicator
- [x] 6.3.3 Color scheme: PLANET=blue, EB=red, VARIABLE=green, NOISE=gray, REVIEW=orange

### Phase 6.4 — Recycler View Component

File: `ui/components/recycler_view.py` (NEW)

Tasks:
- [x] 6.4.1 Dynamically render recycler output based on predicted_class
- [x] 6.4.2 Planet view: physical parameter table, Rp vs stellar radius chart
- [x] 6.4.3 EB view: primary/secondary eclipse depth comparison, morphology label
- [x] 6.4.4 Variable view: Lomb-Scargle periodogram, variability type
- [x] 6.4.5 Noise view: CDPP table, instrumental flag indicators
- [x] 6.4.6 REVIEW view: uncertainty explanation, top-2 competing classes displayed

### Phase 6.5 — Report Exporter

File: `ui/components/report_exporter.py` (NEW)

Tasks:
- [x] 6.5.1 Generate JSON report: all features, triage result, recycler output, metadata
- [x] 6.5.2 Generate HTML report: formatted summary with plots embedded
- [x] 6.5.3 Streamlit download buttons for both formats

### Phase 6.6 — Styling

File: `ui/styles.css` (NEW)

Tasks:
- [x] 6.6.1 Dark astronomy theme: deep navy background, star-white text
- [x] 6.6.2 Glassmorphism cards for metric displays
- [x] 6.6.3 Animated spinner during analysis (Streamlit native)
- [x] 6.6.4 Responsive column layout

### Phase 6 Acceptance Criteria

- [x] AC-6.1: App starts with `streamlit run app.py` and renders all 5 tabs without errors
- [x] AC-6.2: Analysis of HD 209458 completes and displays triage result within 60 seconds
- [x] AC-6.3: No model training occurs during app runtime
- [x] AC-6.4: JSON report downloads correctly with all required fields
- [x] AC-6.5: REVIEW state renders with distinct visual indicator when uncertainty threshold triggered

---

## PHASE 7: Batch Processing CLI, CI/CD, Documentation, and Final Polish
**Estimated Effort:** 2 working days
**Goal:** Headless batch scanner, automated CI, complete scientific documentation.
**Status:** [ ] NOT STARTED

### Dependencies
- Phase 6 complete (all acceptance criteria passed)

### Phase 7.1 — Batch Scanner CLI

File: `batch_scan.py` (NEW)

Tasks:
- [ ] 7.1.1 CLI arguments:
    - --input: CSV file with column tic_id (required)
    - --output: output CSV/JSON path (required)
    - --workers: number of parallel workers (default=2, max=2)
    - --timeout: per-target timeout seconds (default=120)
    - --resume: skip already-processed TIC IDs
- [ ] 7.1.2 Process targets in parallel with concurrent.futures.ThreadPoolExecutor
- [ ] 7.1.3 Write results incrementally (not at end) so progress is preserved on interruption
- [ ] 7.1.4 Summary report: total processed, failures, class distribution

### Phase 7.2 — End-to-End Test

File: `tests/test_end_to_end.py`

Tasks:
- [ ] 7.2.1 Test batch_scan.py on 3-star pilot input
- [ ] 7.2.2 Verify output CSV has required columns
- [ ] 7.2.3 Verify all predictions are one of: PLANET, EB, VARIABLE, NOISE, REVIEW
- [ ] 7.2.4 NOTE: This test requires internet and is skipped in CI

### Phase 7.3 — GitHub Actions CI

File: `.github/workflows/ci.yml` (NEW)

Tasks:
- [ ] 7.3.1 Trigger: push to main; pull request to main
- [ ] 7.3.2 Steps:
    - Set up Python 3.11
    - Install requirements.txt
    - Run pytest tests/ -v --ignore=tests/test_end_to_end.py (offline only)
    - Generate coverage report
- [ ] 7.3.3 All offline tests must pass in < 60 seconds in CI

### Phase 7.4 — Scientific Documentation

File: `docs/scientific_pipeline.md` (NEW)

Tasks:
- [ ] 7.4.1 Section: TESS Mission Background
- [ ] 7.4.2 Section: BLS Algorithm -- mathematical formulation; period grid selection
- [ ] 7.4.3 Section: Feature Definitions -- each of 25 features with formula and scientific rationale
- [ ] 7.4.4 Section: 4-Class Triage -- model training, GroupKFold, calibration, limitations
- [ ] 7.4.5 Section: False Positive Recycler -- each module and its diagnostic approach
- [ ] 7.4.6 Section: Limitations and Caveats -- what the pipeline can and cannot do
- [ ] 7.4.7 Section: Reproducibility -- how to reproduce results from scratch

### Phase 7.5 — Update README.md

Tasks:
- [ ] 7.5.1 Rewrite README to describe the multi-sector 4-class architecture
- [ ] 7.5.2 Add: Installation, Environment Setup, Running Dashboard, Batch CLI, Running Tests
- [ ] 7.5.3 Add: Dataset strategy and catalog sources
- [ ] 7.5.4 Add: Honest evaluation metrics from GroupKFold (not inflated synthetic metrics)
- [ ] 7.5.5 Remove: Claimed 96.79% confidence (was circular training artifact)

### Phase 7 Acceptance Criteria

- [ ] AC-7.1: `pytest tests/ -v --ignore=tests/test_end_to_end.py` all pass in < 60 seconds
- [ ] AC-7.2: CI workflow file exists and is syntactically valid
- [ ] AC-7.3: batch_scan.py processes 3 targets and writes valid output CSV
- [ ] AC-7.4: docs/scientific_pipeline.md covers all 25 features and the triage logic
- [ ] AC-7.5: README.md updated with correct architecture and honest metrics

---

## MILESTONES

| Milestone | After Phase | Deliverable |
|---|---|---|
| M1: Minimum Viable Pipeline | Phase 2 | Multi-sector ingestion + 25 features extracted offline |
| M2: Minimum Viable Product | Phase 4 | 4-class triage working on 100 real stars |
| M3: Full System Demo | Phase 6 | Interactive dashboard with recycler diagnostics |
| M4: Stable Semester Submission | Phase 7 | Complete, tested, documented system |

---

## RISKS AND DECISIONS REQUIRING APPROVAL

| ID | Risk | Trigger Condition | Action Required |
|---|---|---|---|
| RSK-01 | MAST API throttling | >20% of targets fail in Phase 3.4 | Stop and report; implement exponential backoff |
| RSK-02 | Gaia cross-match failure | Cannot resolve >10 VARIABLE stars to TIC | Use alternative: TESS Asteroseismic catalog or VSX |
| RSK-03 | BLS timeout | >10% of targets timeout in Phase 3.4 | Reduce MAX_BLS_FREQUENCY_BINS; rerun pilot |
| RSK-04 | Low model accuracy | Macro F1 < 0.50 after Phase 4 | Add more features; review class label quality; do not inflate |
| RSK-05 | Full batch runtime >30 min | Batch estimate exceeds 30 minutes | STOP; get approval before running |

---

## PHASE STATUS TRACKER

Update these as you work:

```
Phase 0: [ ] NOT STARTED  / [ ] IN PROGRESS  / [ ] COMPLETE
Phase 1: [ ] NOT STARTED  / [ ] IN PROGRESS  / [ ] COMPLETE
Phase 2: [ ] NOT STARTED  / [ ] IN PROGRESS  / [ ] COMPLETE
Phase 3: [ ] NOT STARTED  / [ ] IN PROGRESS  / [ ] COMPLETE
Phase 4: [ ] NOT STARTED  / [ ] IN PROGRESS  / [ ] COMPLETE
Phase 5: [x] COMPLETE
Phase 6: [x] COMPLETE
Phase 7: [ ] NOT STARTED  / [ ] IN PROGRESS  / [ ] COMPLETE
```

---

*Update checkboxes and phase status as implementation progresses.*
*Never mark a phase COMPLETE unless all its Acceptance Criteria are checked off.*
