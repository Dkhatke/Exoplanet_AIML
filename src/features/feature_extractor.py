# src/features/feature_extractor.py
# ============================================================
# Phase 2: Unified feature extractor entry point.
# Takes a detrended LightCurve; returns the complete 25-feature dict.
#
# CRITICAL: This function is EPHEMERIS-BLIND.
# It does NOT accept catalog period or t0 as inputs.
# BLS runs independently on every target.
# ============================================================

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import numpy as np
from lightkurve import LightCurve

from config import FEATURE_NAMES
from src.detection.detector import run_bls
from src.features.transit_metrics import (
    compute_odd_even_depths,
    compute_secondary_eclipse,
    compute_transit_shape,
)
from src.features.variability_metrics import (
    compute_out_of_transit_variability,
    compute_lomb_scargle_peak,
    compute_flux_percentile_ratio,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def extract_features(
    lc_detrended: LightCurve,
    stellar_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    """
    Extract the full 25-feature vector from a detrended light curve.

    This is the ONLY function the ML pipeline calls.
    It is completely ephemeris-blind — no catalog period input.

    Parameters
    ----------
    lc_detrended   : Detrended LightCurve (output of preprocessor.detrend_lc).
    stellar_params : Optional stellar params dict (for physical feature scaling,
                     currently not used inside feature computation itself).

    Returns
    -------
    (feature_dict, bls_output, metadata)
        feature_dict : Ordered dict with exactly 25 features; NaN → 0.0.
        bls_output   : Full BLS result dict for plotting/diagnostics.
        metadata     : Extraction metadata (runtime, warnings, etc.)
    """
    import time as _time
    t_start = _time.monotonic()
    warnings_list = []

    # ── Step 1: BLS detection ────────────────────────────────
    log.info("Running BLS on %d cadences…", len(lc_detrended))
    bls_output = run_bls(lc_detrended, stellar_params=stellar_params)

    period   = bls_output["corrected_period"]
    t0       = bls_output["t0"]
    duration = bls_output["duration"]
    bls_power = bls_output["bls_power"]

    log.info("BLS: P=%.4fd | power=%.4f | harmonic=%s", period, bls_power, bls_output["harmonic_flag"])

    # ── Step 2: Core transit geometry ────────────────────────
    t = lc_detrended.time.value
    f = lc_detrended.flux.value if hasattr(lc_detrended.flux, "value") else np.array(lc_detrended.flux)

    phase = ((t - t0) % period)
    phase[phase > period / 2] -= period
    in_transit = np.abs(phase) < 0.5 * duration
    oot        = np.abs(phase) > 0.5 * duration

    n_transits = max(1, int(round((t[-1] - t[0]) / period))) if period > 0 else 1

    if np.sum(in_transit) >= 3 and np.sum(oot) >= 5:
        depth     = float(np.median(f[oot])) - float(np.median(f[in_transit]))
        depth     = max(0.0, depth)
        std_oot   = float(np.std(f[oot])) + 1e-9
        snr       = depth / (std_oot / np.sqrt(np.sum(in_transit)))

        # Symmetry: compare left vs right half of transit
        left  = phase[in_transit & (phase < 0)]
        right = phase[in_transit & (phase > 0)]
        if len(left) >= 2 and len(right) >= 2:
            f_left  = float(np.mean(f[in_transit & (phase < 0)]))
            f_right = float(np.mean(f[in_transit & (phase > 0)]))
            symmetry = 1.0 - abs(f_left - f_right) / (depth + 1e-9)
        else:
            symmetry = 0.5

        sharpness = depth / (duration + 1e-9)
    else:
        depth = 0.0; snr = 0.0; symmetry = 0.5; sharpness = 0.0
        warnings_list.append("Too few in-transit cadences for depth/SNR computation")

    # ── Step 3: Odd/even depths ──────────────────────────────
    oe = compute_odd_even_depths(lc_detrended, period, t0, duration)

    # ── Step 4: Secondary eclipse ────────────────────────────
    sec = compute_secondary_eclipse(lc_detrended, period, t0, duration)

    # ── Step 5: Transit shape ────────────────────────────────
    shape = compute_transit_shape(lc_detrended, period, t0, duration)

    # ── Step 6: Out-of-transit variability ───────────────────
    oot_var = compute_out_of_transit_variability(lc_detrended, period, t0, duration)

    # ── Step 7: Lomb-Scargle periodogram ─────────────────────
    ls = compute_lomb_scargle_peak(lc_detrended)

    # ── Step 8: Flux percentile ratio ────────────────────────
    pct = compute_flux_percentile_ratio(lc_detrended)

    # ── Step 9: Assemble 25-feature dict ─────────────────────
    raw = {
        # Core transit geometry
        "depth":            depth,
        "duration":         duration,
        "snr":              snr,
        "symmetry":         symmetry,
        "sharpness":        sharpness,
        "n_transits":       float(n_transits),
        # Odd/even
        "odd_depth":        oe["odd_depth"],
        "even_depth":       oe["even_depth"],
        "odd_even_diff":    oe["odd_even_diff"],
        "odd_even_z":       oe["odd_even_z"],
        # Secondary eclipse
        "secondary_depth":    sec["secondary_depth"],
        "secondary_snr":      sec["secondary_snr"],
        "secondary_detected": float(sec["secondary_detected"]),
        # Transit shape
        "ingress_fraction": shape["ingress_fraction"],
        "v_shape_metric":   shape["v_shape_metric"],
        # Flux ratio
        "mean_flux_in":     shape["mean_flux_in"],
        "mean_flux_out":    shape["mean_flux_out"],
        "flux_ratio":       shape["flux_ratio"],
        # OOT variability
        "oot_mad":          oot_var["oot_mad"],
        "oot_rms":          oot_var["oot_rms"],
        "oot_p2p":          oot_var["oot_p2p"],
        "oot_skew":         oot_var["oot_skew"],
        # Lomb-Scargle
        "ls_peak_power":    ls["ls_peak_power"],
        "ls_fap":           ls["ls_fap"],
        # Percentile ratio
        "p5_p95_ratio":     pct["p5_p95_ratio"],
    }

    # ── Step 10: Replace NaN/Inf → 0.0 with warning ──────────
    feature_dict: Dict[str, float] = {}
    for name in FEATURE_NAMES:
        val = raw.get(name, 0.0)
        if not np.isfinite(val):
            log.warning("Feature '%s' is non-finite (%s) — replacing with 0.0", name, val)
            warnings_list.append(f"Non-finite feature: {name}={val}")
            val = 0.0
        feature_dict[name] = float(val)

    runtime = _time.monotonic() - t_start
    log.info("Feature extraction complete: %d features | %.1fs", len(feature_dict), runtime)

    metadata = {
        "bls_period":       period,
        "bls_t0":           t0,
        "bls_duration":     duration,
        "bls_power":        bls_power,
        "bls_runtime_s":    bls_output["bls_runtime_s"],
        "harmonic_flag":    bls_output["harmonic_flag"],
        "timeout_flag":     bls_output["timeout_flag"],
        "n_cadences":       len(lc_detrended),
        "extraction_time_s": runtime,
        "warnings":         warnings_list,
    }

    return feature_dict, bls_output, metadata


def features_to_array(feature_dict: Dict[str, float]) -> np.ndarray:
    """
    Convert feature dict to ordered numpy array matching FEATURE_NAMES order.
    Used as input to the XGBoost classifier.
    """
    return np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float32)
