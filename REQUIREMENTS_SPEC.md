# REQUIREMENTS SPECIFICATION — AIML_Exoplanet
## Dataset, Dependencies, and Environment

> **Last Updated:** Phase 0 Planning
> **Status:** Final — Approved for Phase 0

---

## PART A: DATASET REQUIREMENTS

### A1. Dataset Decision Summary

We use a **4-source, 100-star curated ground-truth dataset**.
Each class (PLANET, EB, VARIABLE, NOISE) has exactly 25 independently labeled stars.
All light curves are downloaded from NASA MAST via Lightkurve.

### A2. Source 1 — PLANET Class (25 stars)

| Field | Value |
|---|---|
| Catalog | NASA Exoplanet Archive TOI |
| Access URL | https://exoplanetarchive.ipac.caltech.edu/TAP/sync |
| Query | SELECT toi, tic_id, tfopwg_disp FROM toi WHERE tfopwg_disp IN ('CP','KP') |
| Format | CSV |
| Python Access | astroquery.ipac.nexsci.nasa_exoplanet_archive.NasaExoplanetArchive |
| Label Column | tfopwg_disp |
| Valid Values | CP = Confirmed Planet, KP = Known Planet |
| Records Pool | ~500+ CP+KP TOIs |
| Download Size | ~3.5 MB catalog |
| Light Curves | Via lightkurve.search_lightcurve("TIC {tic_id}", author="SPOC") |

**Selection filters applied:**
- tfopwg_disp IN ('CP', 'KP')
- SNR > 7 (from TOI table column toi_depth_snr or similar)
- Prefer single-planet systems for pilot
- At least 1 SPOC sector available

### A3. Source 2 — EB Class (25 stars)

| Field | Value |
|---|---|
| Catalog | Villanova TESS Eclipsing Binary Catalog (MAST HLSP) |
| Portal | https://tessebs.villanova.edu/ |
| CSV Download | https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tess-ebs/tess_lcf-ffi_s0001-s0026/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv |
| TIC ID Column | TIC |
| Period Column | period |
| Morphology Column | morph (0.0=detached, 1.0=contact) |
| Records Pool | ~4,500 EBs |
| Download Size | ~2.1 MB catalog |
| Light Curves | Via lightkurve.search_lightcurve("TIC {TIC}", author="SPOC") |

**Selection filters applied:**
- morph < 0.5 (prefer detached binaries with clear secondary eclipses)
- period between 0.5 and 15.0 days
- At least 1 SPOC sector available
- Not already in PLANET list

### A4. Source 3 — VARIABLE Class (25 stars)

| Field | Value |
|---|---|
| Catalog | Gaia Data Release 3 -- Variable Star Classification |
| Archive | https://gea.esac.esa.int/archive/ |
| Query Table | gaiadr3.vari_classifier_result |
| Key Column | best_class_name |
| Valid Types | DSCT_SXPHE, RR, CEP, ROT, LPV |
| Python Access | astroquery.gaia.Gaia.launch_job() |
| Records Pool | ~10.5 million (filtered ~50,000 TESS-observable) |
| Download Size | ~5 MB filtered subset |
| Light Curves | Positional cross-match to TIC, then lightkurve |

**Cross-match protocol:**
1. Query gaiadr3.vari_classifier_result for chosen variability classes
2. Filter to stars with TESS magnitude Tmag between 7 and 13
3. Cross-match RA/Dec to TIC using CDS X-Match or astroquery.mast
   (search radius: 2 arcsec)
4. Resolve TIC ID
5. Exclude any TIC already in PLANET or EB lists
6. Download SPOC or QLP light curve; confirm periodic variability visually

### A5. Source 4 — NOISE / QUIET Class (25 stars)

| Field | Value |
|---|---|
| Catalog | TESS Input Catalog v8.2 (TIC) |
| Access | astroquery.mast.Catalogs.query_criteria(catalog="Tic", ...) |
| Portal | https://mast.stsci.edu/ |
| Filter Criteria | objType=STAR, Tmag between 8 and 12, spectral type G or K |
| Exclusion | NOT in TOI catalog, NOT in Villanova EB catalog, NOT in Gaia variable list |
| Variability Check | Post-download: detrended flux MAD < 500 ppm |
| Records Pool | Millions -- filtered to ~few thousand quiet stars |
| Light Curves | lightkurve.search_lightcurve("TIC {id}", author="SPOC") |

**Selection steps:**
1. Query TIC for G/K type stars, Tmag 8-12, with SPOC observations
2. Download detrended light curve
3. Compute MAD of residual flux
4. Keep only stars with MAD < 500 ppm
5. Exclude any TIC already in PLANET, EB, or VARIABLE lists

### A6. Master Manifest File

File: data/catalogs/targets_100.csv

Required columns:
```
tic_id        -- integer TIC identifier
class_label   -- PLANET / EB / VARIABLE / NOISE
source_catalog -- NASA_TOI / Villanova_EB / Gaia_DR3 / TIC_Quiet
catalog_period -- float days (from catalog; NOT used in BLS search)
catalog_source_id -- original ID in source catalog (TOI number, Gaia source_id, etc.)
sectors_available -- comma-separated list of TESS sectors
pipeline_preference -- SPOC or QLP
notes          -- any conflict or edge case notes
```

### A7. Conflict Resolution Log

File: data/catalogs/label_conflicts.csv

Columns: tic_id, conflict_type, catalog_1, catalog_2, resolved_label, resolution_reason

### A8. Dataset Integrity Rules

1. No TIC ID appears in more than one class
2. All 100 TIC IDs must have at least 1 downloadable SPOC or QLP light curve
3. Feature extraction runs blind (no catalog period input to BLS)
4. Star-level GroupKFold splits: no TIC ID in both train and test fold
5. Final dataset_100.parquet must have zero NaN values in feature columns

---

## PART B: DEPENDENCY REQUIREMENTS

### B1. Pinned Python Dependencies

```
# requirements.txt (pinned for Python 3.10+ / 3.11)
# Astronomy & Data
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
astropy>=5.3.0,<7.0.0
lightkurve>=2.4.2,<3.0.0
wotan>=1.10,<2.0.0
pyarrow>=13.0.0,<16.0.0

# Signal Detection & ML
scipy>=1.11.0,<2.0.0
scikit-learn>=1.3.0,<2.0.0
xgboost>=1.7.6,<3.0.0
joblib>=1.3.0,<2.0.0

# Catalog & Archive Access
astroquery>=0.4.6,<1.0.0

# Visualization
matplotlib>=3.7.0,<4.0.0
plotly>=5.16.0,<6.0.0

# Web Dashboard
streamlit>=1.28.0,<2.0.0

# Testing
pytest>=7.4.0,<9.0.0
pytest-cov>=4.1.0,<6.0.0
```

### B2. New Dependencies Explanation

| Package | Why Added | Phase Added |
|---|---|---|
| wotan | Transit-preserving detrending (biweight/SG) | Phase 1 |
| pyarrow | Parquet file format (Pandas backend) | Phase 1 |
| astroquery | TIC query, Gaia DR3 cross-match, ExoFOP access | Phase 3 |
| pytest | Offline unit and integration testing | Phase 0 |
| pytest-cov | Test coverage reports | Phase 0 |

### B3. Removed / Replaced

| Original Usage | Issue | Replacement |
|---|---|---|
| pickle for caching | Version-fragile; security risk | Parquet via pyarrow |
| On-the-fly synthetic ML training | Circular; no real data | Offline training from real labeled dataset |
| Hardcoded stellar constants in config | Wrong for all non-HD209458 stars | Dynamic TIC query in stellar_resolver.py |

---

## PART C: ENVIRONMENT REQUIREMENTS

### C1. Python Environment Setup

```bash
# Step 1: Verify Python version
python3.11 --version  # must be 3.10+

# Step 2: Create virtual environment in repository root
python3.11 -m venv venv

# Step 3: Activate
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Step 4: Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Step 5: Verify key packages
python -c "import lightkurve; print(lightkurve.__version__)"
python -c "import wotan; print('wotan ok')"
python -c "import astroquery; print('astroquery ok')"
```

### C2. Directory Setup

The following directories are gitignored and must be created locally:

```bash
mkdir -p data/catalogs data/cache models outputs tests docs scripts ui/components
```

### C3. Running the Application

```bash
# Run Streamlit dashboard
streamlit run app.py

# Run offline tests (no internet required)
pytest tests/ -v

# Run batch analysis CLI (Phase 7)
python batch_scan.py --input data/catalogs/targets_100.csv --output results.csv --workers 2

# Train the classifier (Phase 4, offline)
python scripts/train_classifier.py --dataset data/dataset_100.parquet --output models/triage_model.joblib
```

### C4. System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB free | 5 GB free |
| Internet | Required for data download | Not required for tests |
| OS | macOS / Linux / Windows | macOS arm64 (M-series) |
| CPU | Any 4-core | 8-core (Apple M-series or Intel i7) |
| GPU | Not required | Not required |

---

## PART D: TESTING REQUIREMENTS

### D1. Offline Test Fixtures (conftest.py)

All unit tests must run without internet. Fixtures to create:

```
synthetic_planet_lc   -- Flat flux with periodic box dips; period ~3.5 days; depth ~1.5%
synthetic_eb_lc       -- Unequal primary/secondary dips; period ~2.0 days
synthetic_variable_lc -- Sinusoidal variability; period ~0.8 days; amplitude ~2%
synthetic_noise_lc    -- Random Gaussian scatter; no periodic signal
```

### D2. Test Coverage Requirements

| Test File | Covers | Must Pass Offline |
|---|---|---|
| test_harness.py | Fixture generation and schema | Yes |
| test_ingestion.py | Fetcher + stitcher logic | Yes (with mocked MAST) |
| test_preprocessor.py | Transit depth preserved after detrend | Yes |
| test_features.py | Feature extraction completeness, no NaN | Yes |
| test_dataset_integrity.py | 100-star manifest schema, no duplicates | Yes |
| test_classifier.py | Model loads, predicts 4 classes, no leakage | Yes |
| test_recycler_modules.py | Each recycler produces output dict | Yes |
| test_end_to_end.py | Full pipeline on pilot set | Requires network |

### D3. Acceptance Criteria for Tests

- All offline tests pass in < 30 seconds on a standard laptop
- No test imports a real MAST/Gaia API without mocking
- Zero circular dependencies between test modules

---

## PART E: DATA PIPELINE CONSTRAINTS

### E1. Feature Extraction Must Be Ephemeris-Blind

The feature extractor (src/features/feature_extractor.py) must:
- NOT accept catalog_period or catalog_t0 as inputs
- Run BLS independently on each target
- Only compare detected period to catalog period in audit log (never model input)

### E2. Detrending Must Preserve Transit Depth

The preprocessor (src/preprocessing/preprocessor.py) must:
- Implement two-pass detrending: preliminary BLS -> mask transits -> re-detrend
- Verify transit depth is not attenuated by more than 5% (tested in test_preprocessor.py)
- Return both raw_normalized_lc and detrended_lc

### E3. Caching Must Be Atomic and Versioned

The cache manager (src/ingestion/cache_manager.py) must:
- Write Parquet files atomically (write to temp, rename to final)
- Include metadata: tic_id, sectors, pipeline, download_timestamp, lightkurve_version
- Detect stale cache (lightkurve version mismatch) and invalidate

### E4. Period Grid Must Be Bounded

The detector (src/detection/detector.py) must:
- Use frequency-linear spacing: delta_f = 1 / (n_oversample * T_baseline)
- Cap maximum period at T_baseline / 3 (requires at least 3 transits)
- Cap minimum period at 0.5 days
- Never exceed MAX_BLS_FREQUENCY_BINS = 50,000
- Log actual number of frequency bins used

---

*This file specifies all data, dependency, and environment requirements for implementation.*
*Update when requirements change.*
