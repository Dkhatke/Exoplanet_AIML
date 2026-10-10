# Exoplanet_AIML

Machine-learning pipeline for detecting exoplanet transit signals in NASA TESS
photometric light curves, with a false-positive "recycler" that diagnoses
non-planet signals.

## Overview

The pipeline ingests multi-sector TESS light curves, removes stellar variability
while preserving transit depth, searches for periodic transits with the Box
Least Squares (BLS) algorithm, extracts physics-informed features, and
classifies each target into one of four triage classes.

```
TIC ID / star name
  -> Ingestion        multi-sector SPOC/QLP download + Parquet cache
  -> Stitching        per-sector normalisation + stitch
  -> Preprocessing    transit-masked two-pass detrending (Wotan)
  -> Detection        frequency-linear bounded BLS + harmonic check
  -> Features         25 ephemeris-blind physics features
  -> Classification   4-class XGBoost triage (PLANET/EB/VARIABLE/NOISE)
  -> Recycler         per-class diagnostics for false positives
```

## Triage classes

| Class | Meaning |
|---|---|
| PLANET | transit consistent with a sub-stellar occulter |
| EB | eclipsing binary (secondary eclipse, odd/even mismatch, V-shape) |
| VARIABLE | stellar pulsation or rotational modulation |
| NOISE | no significant periodic signal |
| REVIEW | classifier confidence below threshold; needs human review |

## Tech stack

- Python 3.10+ (3.11 recommended)
- lightkurve, astropy, wotan, astroquery (TESS data + detrending)
- numpy, pandas, scipy, scikit-learn, xgboost (features + ML)
- pytest (offline test suite)

## Install

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the tests (offline, no internet required)

```bash
pytest tests/ -q
```

All unit tests use synthetic light-curve fixtures and never hit the MAST/Gaia
APIs.
