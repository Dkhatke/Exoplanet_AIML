# config.py — AIML_Exoplanet Multi-Sector Pipeline + False Positive Recycler
# ============================================================
# Phase 0: Full project-wide configuration
# NOTE: Hardcoded HD 209458 stellar constants REMOVED.
#       Stellar parameters are now queried dynamically from TIC.
#       STAR_NAME is kept as a dashboard demo default ONLY.
# ============================================================

# ── DEMO DEFAULT ─────────────────────────────────────────────
STAR_NAME         = "HD 209458"   # Default for Streamlit sidebar input only
MISSION           = "TESS"

# ── INGESTION ────────────────────────────────────────────────
PREFERRED_AUTHOR  = "SPOC"        # Science Processing Operations Center (2-min cadence)
FALLBACK_AUTHOR   = "QLP"         # Quick-Look Pipeline (10/30-min cadence FFIs)
MAX_SECTORS       = 6             # Maximum sectors to stitch per target

# ── PREPROCESSING ────────────────────────────────────────────
# Sigma clipping: asymmetric — clip flares (upward) hard, transits (downward) gently
SIGMA_UPPER       = 3.0           # Clip upward outliers (flares)
SIGMA_LOWER       = 7.0           # Clip downward outliers (instrument spikes, NOT transits)
DETREND_WINDOW_DAYS = 0.5         # Wotan biweight window width in days
# Legacy compat (used in old preprocessor.py — preserved until Phase 1 refactor)
WINDOW_LENGTH     = 201
SIGMA_CLIP_VAL    = 5.0

# ── DETECTION (BLS) ──────────────────────────────────────────
PERIOD_MIN_DAYS   = 0.5           # Minimum period to search (days)
PERIOD_MAX_DAYS   = 15.0          # Maximum period (capped further by baseline/3 rule)
BLS_OVERSAMPLE    = 10            # Frequency grid oversampling factor
MAX_BLS_FREQUENCY_BINS = 50_000   # Hard cap on BLS frequency grid size
TRANSIT_DURATION_MIN   = 0.01     # Minimum transit duration to test (days)
TRANSIT_DURATION_MAX   = 0.25     # Maximum transit duration to test (days)
N_DURATION_STEPS       = 5        # Number of duration values to test (log-spaced)
BLS_TIMEOUT_SECONDS    = 60       # Per-target BLS timeout
# Legacy compat
PERIOD_MIN        = 0.5
PERIOD_MAX        = 15.0
PERIOD_STEPS      = 5000
TRANSIT_DURATIONS = [0.01, 0.05, 0.10, 0.15, 0.25]
PHASE_WIDTH       = 0.02          # Phase fold window (legacy; now computed dynamically)

# ── TRIAGE CLASSIFIER ────────────────────────────────────────
CLASS_LABELS      = ["PLANET", "EB", "VARIABLE", "NOISE"]
CLASS_LABEL_MAP   = {"PLANET": 0, "EB": 1, "VARIABLE": 2, "NOISE": 3}
CLASS_LABEL_INV   = {0: "PLANET", 1: "EB", 2: "VARIABLE", 3: "NOISE"}
REVIEW_PROB_THRESHOLD  = 0.60     # Route to REVIEW if max(P_class) < this
# Class conflict resolution priority (higher = wins)
CLASS_CONFLICT_PRIORITY = {"PLANET": 4, "EB": 3, "VARIABLE": 2, "NOISE": 1}

# ── ML / TRAINING ────────────────────────────────────────────
RANDOM_STATE      = 42
TEST_SIZE         = 0.2
N_CV_FOLDS        = 5             # GroupKFold folds for cross-validation
# Legacy compat (kept until ml_model.py is deprecated in Phase 4)
N_AUGMENTATIONS   = 40

# ── FULL 25-FEATURE VECTOR ───────────────────────────────────
# All features extracted by src/features/feature_extractor.py (Phase 2)
# Ordered list — order MUST match model training input
FEATURE_NAMES = [
    # Core transit geometry
    "depth",              # Transit depth (fraction of stellar flux)
    "duration",           # Transit duration (days)
    "snr",                # Transit signal-to-noise ratio
    "symmetry",           # Transit shape symmetry score (0–1)
    "sharpness",          # Depth / duration ratio
    "n_transits",         # Number of observed transits in baseline
    # Odd/even depth diagnostics (EB discriminator)
    "odd_depth",          # Median depth of odd-numbered transits
    "even_depth",         # Median depth of even-numbered transits
    "odd_even_diff",      # |odd_depth - even_depth|
    "odd_even_z",         # Z-score significance of odd/even difference
    # Secondary eclipse (EB discriminator)
    "secondary_depth",    # Depth at phase 0.5 (secondary eclipse)
    "secondary_snr",      # SNR of secondary eclipse
    "secondary_detected", # Binary flag: 1 if secondary eclipse SNR > 3
    # Transit shape (planet vs EB discriminator)
    "ingress_fraction",   # Ingress duration / total transit duration
    "v_shape_metric",     # V-shape vs U-shape score (higher = more V-shaped = EB)
    # Flux ratio
    "mean_flux_in",       # Mean flux during transit
    "mean_flux_out",      # Mean flux out of transit
    "flux_ratio",         # mean_flux_in / mean_flux_out
    # Out-of-transit variability (variable star discriminator)
    "oot_mad",            # Median absolute deviation of OOT flux
    "oot_rms",            # RMS of OOT flux
    "oot_p2p",            # Peak-to-peak amplitude of OOT flux
    "oot_skew",           # Skewness of OOT flux distribution
    # Lomb-Scargle periodicity (variable star discriminator)
    "ls_peak_power",      # Peak Lomb-Scargle power
    "ls_fap",             # Lomb-Scargle False Alarm Probability
    # Flux percentile variability
    "p5_p95_ratio",       # 5th–95th percentile flux range
]

# Legacy 9-feature list (used by old analyzer.py until Phase 2 refactor)
FEATURE_NAMES_LEGACY = [
    "depth", "duration", "mean_flux_in",
    "std_flux_in", "std_flux_out", "snr",
    "symmetry", "sharpness", "n_transits"
]

# ── STELLAR PARAMETER DEFAULTS ───────────────────────────────
# Used ONLY when TIC lookup fails — must log a WARNING
# These are NOT hardcoded to HD 209458; they are generic Solar defaults
DEFAULT_STAR_RADIUS_SOLAR = 1.0   # R_sun
DEFAULT_STAR_MASS_SOLAR   = 1.0   # M_sun
DEFAULT_STAR_TEFF         = 5778  # K (Solar effective temperature)
DEFAULT_STAR_LOGG         = 4.44  # log(g) cm/s^2 (Solar surface gravity)

# ── RESOURCE LIMITS ──────────────────────────────────────────
MAX_PARALLEL_WORKERS      = 2     # Never exceed 3 — MAST rate-limit & RAM safety
MAX_SECONDS_PER_TARGET    = 120   # Enforced per batch job target
PILOT_SIZE                = 5     # Always run pilot first before full batch
INITIAL_DATASET_SIZE      = 100   # Total stars: 25 per class

# ── PATHS ────────────────────────────────────────────────────
import os
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "data")
CATALOG_DIR       = os.path.join(BASE_DIR, "data", "catalogs")
CACHE_DIR         = os.path.join(BASE_DIR, "data", "cache")
OUTPUT_DIR        = os.path.join(BASE_DIR, "outputs")
MODEL_DIR         = os.path.join(BASE_DIR, "models")          # Directory for saved models
MODEL_PATH        = os.path.join(BASE_DIR, "models", "triage_model.joblib")
DATASET_PATH      = os.path.join(BASE_DIR, "data", "dataset_100.parquet")
TARGETS_MANIFEST  = os.path.join(BASE_DIR, "data", "catalogs", "targets_manifest.csv")
LOG_FILE          = os.path.join(BASE_DIR, "outputs", "pipeline.log")
# Aliases for backwards compatibility
MODEL_PATH_LEGACY = "models/xgb_model.pkl"
REVIEW_THRESHOLD  = REVIEW_PROB_THRESHOLD  # Alias used by classifier module


# ── DATASET CATALOG URLs ─────────────────────────────────────
TOI_API_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+toi,tic_id,tfopwg_disp,toi_depth_snr"
    "+from+toi+where+tfopwg_disp+in+('CP','KP')&format=csv"
)
VILLANOVA_EB_URL = (
    "https://mast.stsci.edu/api/v0.1/Download/file"
    "?uri=mast:HLSP/tess-ebs/tess_lcf-ffi_s0001-s0026"
    "/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv"
)