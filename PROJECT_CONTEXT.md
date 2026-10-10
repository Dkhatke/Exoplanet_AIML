# PROJECT CONTEXT — AIML_Exoplanet
## Multi-Sector Exoplanet Pipeline + False Positive Recycler

> **Last Updated:** Phase 5 Complete (Recycler Modules implemented)
> **Status:** Phases 0–5 DONE | Phase 6 (Streamlit UX) NEXT
>
> **Remote:** `https://github.com/Dkhatke/AIML-Project.git` (feature-branch workflow; see .clinerules)

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Project Name** | AIML_Exoplanet — Multi-Sector Exoplanet Pipeline + False Positive Recycler |
| **Repository Root** | `Exoplanet_Detection/` |
| **Entry Point** | `app.py` (Streamlit web dashboard) |
| **Python Version** | 3.10+ required (3.11 recommended via Homebrew) |
| **Git Remote** | `github.com/mahi-mahalle/Exoplanet_Detection` (branch: `main`) |

---

## 2. Central Scientific Objective

Detect periodic transit signals in multi-sector NASA TESS photometric light curves,
classify candidates into one of four evidence-based classes, route false positives to
specialized diagnostic modules, and produce interpretable results via a Streamlit dashboard.

### Four Output Classes

```
PLANET   ->  Transit shape consistent with a sub-stellar occulter (planet candidate)
EB       ->  Eclipsing Binary: odd/even depth mismatch, secondary eclipse, V-shape
VARIABLE ->  Stellar pulsation or rotational modulation; sinusoidal/quasi-periodic flux
NOISE    ->  No significant periodic signal; instrumental or statistical scatter
REVIEW   ->  Classifier uncertainty too high (max prob < 0.60); needs human check
```

### Central Workflow

```
Target Input (Star Name / TIC ID)
        |
[Phase 1] Multi-Sector TESS Ingestion (SPOC / QLP) + Parquet Cache
        |
[Phase 1] Sector Stitching + Transit-Masked Detrending
        |
[Phase 2] Frequency-Linear BLS Grid Search (Bounded Period Grid)
        |
[Phase 2] Harmonic Vetting + 20+ Physics-Informed Feature Extraction
        |
[Phase 4] 4-Class Calibrated Triage Classifier (XGBoost + CalibratedClassifierCV)
        |
  PLANET | EB | VARIABLE | NOISE | REVIEW
        |
[Phase 5] Specialized Recycler Modules
        |
[Phase 6] Streamlit Multi-Tab Dashboard + JSON/PDF Export
```

---

## 3. Current Repository State (Audit Summary)

### What Exists

| File / Module | Lines | Status | Notes |
|---|---|---|---|
| `app.py` | 206 | REFACTOR | Functional layout; hardcoded to HD 209458 |
| `config.py` | 39 | EXPAND | Hardcoded stellar constants; needs dynamic support |
| `requirements.txt` | 11 | FIX | No version pins; missing wotan, astroquery, pytest |
| `src/fetcher.py` | 79 | REFACTOR | Single-sector only; pickle cache |
| `src/preprocessor.py` | 42 | REFACTOR | No transit masking; flattens transits |
| `src/detector.py` | 68 | REFACTOR | Linear period grid; fixed durations |
| `src/analyzer.py` | 111 | REFACTOR | 9 features; hardcoded stellar constants |
| `src/ml_model.py` | 197 | CRITICAL REWRITE | Circular training; synthetic-only data; binary only |
| `src/images/` | -- | KEEP | Static screenshots for README |

### Critical Defects to Fix (in order)

1. **AUD-01 CRITICAL:** Circular ML training -- synthetic perturbations of one star
2. **AUD-02 CRITICAL:** Hardcoded stellar constants for HD 209458 used for all stars
3. **AUD-03 HIGH:** Single-sector ingestion only
4. **AUD-04 HIGH:** Destructive preprocessing -- transits eroded by flattening
5. **AUD-06 HIGH:** Binary classifier; no EB/Variable/Noise/Review triage

---

## 4. CHOSEN DATASET STRATEGY (Priority #1 Decision)

### 4.1 Decision Summary

**100-Star Curated Ground-Truth Dataset: 25 stars per class**

Sources selected based on:
- 100% TESS light curve availability via MAST
- Independently verified class labels (not just pipeline flags)
- Covers all 4 required classes
- Laptop-feasible: total < 1 GB, < 30 min processing

---

### 4.2 The Four Catalog Sources

#### CLASS 1 — PLANET (25 stars)
**Source:** NASA Exoplanet Archive TOI Catalog

| Property | Details |
|---|---|
| Official Name | NASA Exoplanet Archive -- TESS Objects of Interest (TOI) |
| API URL | https://exoplanetarchive.ipac.caltech.edu/TAP/sync |
| Query Table | toi |
| Key Column | tfopwg_disp |
| Values Used | CP (Confirmed Planet), KP (Known Planet) |
| TIC ID Column | tic_id |
| File Format | CSV via HTTP or astroquery |
| Records Available | ~500+ CP+KP TOIs with TESS light curves |
| Download Size | ~3.5 MB (full catalog CSV) |
| Licensing | NASA Open Data -- public domain |

**Label Rule:** tfopwg_disp = CP or KP --> class PLANET

---

#### CLASS 2 — EB (25 stars)
**Source:** Villanova TESS Eclipsing Binary (TESS-EBs) Catalog

| Property | Details |
|---|---|
| Official Name | TESS Eclipsing Binary Catalog (Villanova / MAST HLSP) |
| Portal URL | https://tessebs.villanova.edu/ |
| Direct CSV | https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tess-ebs/tess_lcf-ffi_s0001-s0026/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv |
| TIC ID Column | TIC |
| Morphology Column | morph (0.0 = detached, 1.0 = contact) |
| Records Available | ~4,500 verified EBs (Sectors 1-26) |
| Download Size | ~2.1 MB |

**Selection:** Prefer morph < 0.5 (detached); periods 0.5-15 days  
**Label Rule:** In Villanova TESS-EB catalog --> class EB

---

#### CLASS 3 — VARIABLE (25 stars)
**Source:** Gaia DR3 Variable Star Catalog (positional cross-match to TESS/TIC)

| Property | Details |
|---|---|
| Official Name | Gaia DR3 Variability Classification (vari_classifier_result) |
| Archive URL | https://gea.esac.esa.int/archive/ |
| Query Table | gaiadr3.vari_classifier_result |
| Key Column | best_class_name |
| Values Used | DSCT_SXPHE (d Scuti), RR (RR Lyrae), CEP (Cepheid), ROT (Rotational), LPV |
| Records Available | ~10.5 million variable sources |
| Download Size | ~5 MB filtered subset |

**Cross-Match Protocol:**
1. Query Gaia DR3 for variability classes
2. Cross-match RA/Dec to TIC via MAST or CDS X-Match (2 arcsec radius)
3. Resolve to TIC ID for lightkurve download
4. Exclude any TIC already in PLANET or EB lists

**Label Rule:** Gaia DR3 variable (non-EB class) --> class VARIABLE

---

#### CLASS 4 — NOISE / QUIET (25 stars)
**Source:** TESS Input Catalog (TIC v8.2) -- Custom Quiet Star Selection

| Property | Details |
|---|---|
| Official Name | TESS Input Catalog v8.2 (TIC) |
| Access Method | astroquery.mast.Catalogs.query_criteria(catalog="Tic") |
| MAST Portal | https://mast.stsci.edu/ |
| Selection | objType=STAR, Tmag 8-12, NOT in TOI, NOT in EB catalog, NOT in Gaia variable list |
| Variability Check | Detrended flux MAD < 500 ppm |

**Label Rule:** TIC star + no classification + MAD < 500 ppm --> class NOISE

---

### 4.3 Conflict Resolution

If a star appears in multiple catalogs:

```
Priority:  PLANET > EB > VARIABLE > NOISE
Conflicts: Logged in data/catalogs/label_conflicts.csv
```

### 4.4 Ephemeris-Blind Rule (NON-NEGOTIABLE)

- Catalog periods and T0 values are NEVER fed into the BLS search
- Catalog ephemerides used ONLY for post-hoc validation audit
- Prevents classifier from learning "catalog period = planet" shortcuts

### 4.5 Pilot Set (5 Stars -- Run First to Validate Pipeline)

| # | Star / TIC | Class | Reason |
|---|---|---|---|
| 1 | HD 209458 (TIC 420814525) | PLANET | Well-studied hot Jupiter; strong deep transit |
| 2 | L 98-59 b (TOI-175, TIC 307210830) | PLANET | Shallow rocky planet; sensitivity test |
| 3 | TIC 272272670 | EB | Detached binary; clear primary+secondary eclipses |
| 4 | TIC 38846515 | VARIABLE | Delta Scuti pulsator; strong sinusoidal signal |
| 5 | TIC 25155310 | NOISE | Quiet solar-type star; flat light curve |

### 4.6 Storage Estimate

| Data Type | Size | Format | Gitignored |
|---|---|---|---|
| Raw FITS cache (~100 stars x 2 sectors) | ~400-500 MB | FITS/Parquet | Yes |
| Cleaned time series Parquet | ~70-100 MB | Parquet | Yes |
| Feature table | < 1 MB | CSV/Parquet | Yes |
| Catalog CSVs (TOI, EB, Gaia) | ~12 MB | CSV | Yes |
| Trained model weights | < 5 MB | joblib | Yes |
| **Total workspace footprint** | **< 1.0 GB** | -- | -- |

---

## 5. Architecture Modules Map

```
Exoplanet_Detection/
+-- app.py                            <- Streamlit entry point (refactor Phase 6)
+-- config.py                         <- All constants, paths, thresholds (Phase 0)
+-- requirements.txt                  <- Pinned dependencies (Phase 0)
+-- batch_scan.py                     <- [NEW Phase 7] Headless CLI batch analyzer
|
+-- src/
|   +-- ingestion/
|   |   +-- fetcher.py                <- [REFACTOR Phase 1] Multi-sector MAST download
|   |   +-- stitcher.py               <- [NEW Phase 1] Sector normalization + stitching
|   |   +-- cache_manager.py          <- [NEW Phase 1] Parquet read/write + metadata
|   |   +-- stellar_resolver.py       <- [NEW Phase 1] TIC -> stellar parameters
|   |
|   +-- preprocessing/
|   |   +-- preprocessor.py           <- [REFACTOR Phase 1] Transit-masked detrending
|   |
|   +-- detection/
|   |   +-- detector.py               <- [REFACTOR Phase 2] BLS + harmonic check
|   |
|   +-- features/
|   |   +-- transit_metrics.py        <- [NEW Phase 2] Odd/even depth, secondary eclipse
|   |   +-- variability_metrics.py    <- [NEW Phase 2] LS power, MAD, percentile ratios
|   |   +-- feature_extractor.py      <- [NEW Phase 2] Unified feature pipeline
|   |
|   +-- ml/
|   |   +-- classifier.py             <- [NEW Phase 4] 4-class XGBoost + calibration
|   |   +-- evaluator.py              <- [NEW Phase 4] GroupKFold CV + metrics
|   |
|   +-- recycler/
|   |   +-- planet_module.py          <- [NEW Phase 5] Planet physical characterization
|   |   +-- eb_module.py              <- [NEW Phase 5] EB diagnostic analyzer
|   |   +-- variable_module.py        <- [NEW Phase 5] Variable star mode classifier
|   |   +-- noise_module.py           <- [NEW Phase 5] Noise floor diagnostics
|   |
|   +-- utils/
|       +-- logger.py                 <- [NEW Phase 0] Structured logging
|
+-- ui/
|   +-- components/
|   |   +-- lightcurve_view.py        <- [NEW Phase 6] Multi-sector + folded plots
|   |   +-- triage_gauge.py           <- [NEW Phase 6] 4-class probability gauge
|   |   +-- recycler_view.py          <- [NEW Phase 6] Contextual diagnostic panel
|   |   +-- report_exporter.py        <- [NEW Phase 6] JSON + HTML export
|   +-- styles.css                    <- [NEW Phase 6] Dark astronomy theme
|
+-- scripts/
|   +-- build_dataset.py              <- [NEW Phase 3] Batch catalog + feature builder
|   +-- train_classifier.py           <- [NEW Phase 4] Offline model training CLI
|
+-- data/
|   +-- catalogs/                     <- Catalog CSVs (TOI, EB, Gaia, TIC quiet)
|   |   +-- targets_100.csv           <- Master 100-star manifest
|   +-- cache/                        <- Raw FITS + Parquet light curves (gitignored)
|   +-- dataset_100.parquet           <- Extracted feature table (gitignored)
|
+-- models/
|   +-- triage_model.joblib           <- Trained 4-class model (gitignored)
|
+-- tests/
|   +-- conftest.py                   <- [NEW Phase 0] Offline synthetic fixtures
|   +-- test_harness.py               <- [NEW Phase 0] Validate offline scaffold
|   +-- test_ingestion.py             <- [NEW Phase 1] Fetcher + stitcher tests
|   +-- test_preprocessor.py          <- [NEW Phase 1] Detrending fidelity tests
|   +-- test_features.py              <- [NEW Phase 2] Feature extractor tests
|   +-- test_dataset_integrity.py     <- [NEW Phase 3] Dataset schema validation
|   +-- test_classifier.py            <- [NEW Phase 4] Model + leakage check
|   +-- test_recycler_modules.py      <- [NEW Phase 5] Module output tests
|   +-- test_end_to_end.py            <- [NEW Phase 7] Full pipeline E2E test
|
+-- docs/
|   +-- scientific_pipeline.md        <- [NEW Phase 7] Scientific documentation
|
+-- PROJECT_CONTEXT.md               <- THIS FILE (update each phase)
+-- REQUIREMENTS_SPEC.md             <- Dataset + dependency specification
+-- IMPLEMENTATION_PLAN.md           <- Phase-by-phase tasks and checklists
```

---

## 6. Key Technical Decisions (Locked)

| Decision | Choice | Rationale |
|---|---|---|
| Primary Data Source | NASA MAST via Lightkurve | Official archive; programmatic access |
| PLANET Labels | NASA Exoplanet Archive TOI (CP+KP) | Independently confirmed |
| EB Labels | Villanova TESS-EBs Catalog | Expert-verified; period + morphology metadata |
| VARIABLE Labels | Gaia DR3 vari_classifier_result | Gold standard; 25 variable classes |
| NOISE Labels | TIC + MAD < 500 ppm post-download | No pre-existing quiet flag |
| Feature Extraction | Ephemeris-blind BLS + 20+ physics features | Prevents data leakage |
| ML Architecture | XGBoost + CalibratedClassifierCV | Best for N=100; calibrated probabilities |
| CV Strategy | 5-fold Stratified GroupKFold on tic_id | Prevents star-level data leakage |
| Detrending Engine | Wotan (biweight) + SG fallback | Transit-preserving |
| Cache Format | Parquet (Apache Arrow) | Schema enforcement; smaller than pickle |
| Max Workers | 2 parallel (never > 3) | Laptop RAM safety; MAST rate-limit |
| Max per-target time | 120 seconds | Enforced by watchdog timer |
| Python Version | 3.10+ (3.11 recommended) | Lightkurve 2.4+ compatibility |

---

## 7. Resource Limits (Non-Negotiable)

```
MAX_PARALLEL_WORKERS      = 2        # Never exceed 3
MAX_SECONDS_PER_TARGET    = 120      # Enforced per batch job
MAX_BLS_FREQUENCY_BINS    = 50000    # Bounded period grid
PILOT_SIZE                = 5        # Always run pilot first
INITIAL_DATASET_SIZE      = 100      # 25 per class
REVIEW_PROB_THRESHOLD     = 0.60     # Route to REVIEW if max(P_class) < this
```

---

## 8. Phase Summary

| Phase | Name | Key Deliverable |
|---|---|---|
| 0 | Audit + Scaffolding | Pinned deps, config, offline test harness |
| 1 | Multi-Sector Ingestion | Parquet cache + transit-masked detrend |
| 2 | Detection + Features | Bounded BLS + 20+ diagnostic features |
| 3 | Dataset Build | 100-star ground-truth feature table |
| 4 | 4-Class Classifier | Calibrated triage model + GroupKFold eval |
| 5 | Recycler Modules | Planet/EB/Variable/Noise diagnostics |
| 6 | Streamlit UX | Modern multi-tab dashboard + export |
| 7 | Batch + CI + Docs | CLI scanner, CI pipeline, scientific docs |

---

## 9. Glossary

| Term | Definition |
|---|---|
| BLS | Box Least Squares -- detects periodic box-shaped dips (transits) |
| TLS | Transit Least Squares -- improved BLS using realistic transit shapes |
| TIC | TESS Input Catalog -- master list of all TESS target stars |
| TOI | TESS Object of Interest -- TESS planet candidate with alert |
| SPOC | Science Processing Operations Center -- official TESS 2-min pipeline |
| QLP | Quick-Look Pipeline -- TESS 10/30-min cadence FFI products |
| EB | Eclipsing Binary -- two stars orbiting each other, producing eclipses |
| MAST | Mikulski Archive for Space Telescopes -- NASA data archive |
| MAD | Median Absolute Deviation -- robust measure of flux scatter |
| GroupKFold | Cross-validation keeping all star observations in one fold |
| Wotan | Python detrending library; biweight/spline/SG with transit masking |
| SNR | Signal-to-Noise Ratio -- transit depth / out-of-transit scatter |

---

*Update this file at the start of each new phase.*
