# src/detection/detector.py
# ============================================================
# Phase 2: Frequency-linear BLS detector with bounded grid.
#
# Fixes AUD-05: Original used a linear PERIOD grid (wrong — undersamples
# short periods). This uses a linear FREQUENCY grid (standard practice)
# capped at MAX_BLS_FREQUENCY_BINS and bounded by the baseline.
#
# Also adds:
#   - Dynamic transit duration range (log-spaced, not fixed)
#   - Harmonic period check (P/2, 2P)
#   - Execution timeout
# ============================================================

from __future__ import annotations

import time as _time
from typing import Dict, Any, Optional, Tuple

import numpy as np
from astropy.timeseries import BoxLeastSquares
from lightkurve import LightCurve

from config import (
    PERIOD_MIN_DAYS, PERIOD_MAX_DAYS,
    BLS_OVERSAMPLE, MAX_BLS_FREQUENCY_BINS,
    TRANSIT_DURATION_MIN, TRANSIT_DURATION_MAX,
    N_DURATION_STEPS, BLS_TIMEOUT_SECONDS,
)
from src.ingestion.stitcher import get_baseline_days
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Grid construction ─────────────────────────────────────────

def compute_frequency_grid(
    t_baseline: float,
    period_min: float = PERIOD_MIN_DAYS,
    oversample: int = BLS_OVERSAMPLE,
    max_bins: int = MAX_BLS_FREQUENCY_BINS,
) -> np.ndarray:
    """
    Build a frequency-linear period grid bounded by the light curve baseline.

    Physics:
    - Minimum detectable period: period_min (default 0.5 days)
    - Maximum useful period: t_baseline / 3 (need ≥3 transits to be confident)
    - Grid spacing: 1 / (oversample × t_baseline)  [frequency-linear]
    - Hard cap at max_bins to prevent runaway computation

    Parameters
    ----------
    t_baseline : Total light curve duration in days.
    period_min : Minimum period to search (days).
    oversample : Oversampling factor (higher → finer grid).
    max_bins   : Hard cap on number of frequency bins.

    Returns
    -------
    np.ndarray — periods array in days.
    """
    # At least 3 transits required
    period_max = min(PERIOD_MAX_DAYS, t_baseline / 3.0)

    if period_max <= period_min:
        log.warning(
            "Baseline %.1fd too short for 3 transits at P>%.1fd — using P_max=%.1fd",
            t_baseline, period_min, t_baseline / 2.0,
        )
        period_max = max(period_min + 0.1, t_baseline / 2.0)

    f_min = 1.0 / period_max
    f_max = 1.0 / period_min
    delta_f = 1.0 / (oversample * t_baseline)

    n_freqs = int((f_max - f_min) / delta_f)
    n_freqs = min(n_freqs, max_bins)   # Hard cap
    n_freqs = max(n_freqs, 500)        # Minimum for any useful search

    frequencies = np.linspace(f_min, f_max, n_freqs)
    periods = 1.0 / frequencies[::-1]  # Convert back to periods (ascending)

    log.info(
        "BLS grid: %.2f–%.2f days | %d bins | oversample=%d",
        periods[0], periods[-1], n_freqs, oversample,
    )
    return periods


def _transit_durations(
    n_steps: int = N_DURATION_STEPS,
    dur_min: float = TRANSIT_DURATION_MIN,
    dur_max: float = TRANSIT_DURATION_MAX,
) -> np.ndarray:
    """Log-spaced transit durations to test."""
    return np.logspace(np.log10(dur_min), np.log10(dur_max), n_steps)


# ── BLS runner ────────────────────────────────────────────────

def run_bls(
    lc_detrended: LightCurve,
    stellar_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the BLS periodogram on a detrended light curve.

    Parameters
    ----------
    lc_detrended   : Flattened LightCurve (transit depth preserved).
    stellar_params : Optional dict from stellar_resolver (unused in BLS itself
                     but passed through for downstream modules).

    Returns
    -------
    dict with keys:
        best_period      : float — best BLS period (days)
        t0               : float — transit epoch (BTJD)
        duration         : float — transit duration (days)
        bls_power        : float — BLS power at best period
        bls_result       : astropy BLS result object (for plotting)
        periods          : np.ndarray — full period grid searched
        top3_periods     : np.ndarray — top 3 candidate periods
        top3_powers      : np.ndarray — BLS power at top 3 periods
        harmonic_flag    : bool — True if a harmonic (P/2 or 2P) is dominant
        corrected_period : float — harmonic-corrected period (same as best_period if no flag)
        n_bins_searched  : int — actual number of frequency bins used
        bls_runtime_s    : float — elapsed time in seconds
        timeout_flag     : bool — True if search was interrupted by timeout
    """
    t = lc_detrended.time.value
    f = lc_detrended.flux.value

    t_baseline = get_baseline_days(lc_detrended)
    periods = compute_frequency_grid(t_baseline)
    durations = _transit_durations()

    best_power  = -np.inf
    best_result = None
    timeout_flag = False
    t_start = _time.monotonic()

    log.info("BLS search: %d periods × %d durations", len(periods), len(durations))

    for dur in durations:
        elapsed = _time.monotonic() - t_start
        if elapsed > BLS_TIMEOUT_SECONDS:
            log.warning("BLS timeout (%.0fs) — returning partial result", elapsed)
            timeout_flag = True
            break

        try:
            bls = BoxLeastSquares(t, f)
            result = bls.power(periods, dur)
            peak = float(np.max(result.power))
            if peak > best_power:
                best_power  = peak
                best_result = result
        except Exception as exc:
            log.warning("BLS error at duration %.3fd: %s", dur, exc)
            continue

    bls_runtime = _time.monotonic() - t_start

    if best_result is None:
        log.error("BLS produced no result — returning default output")
        return _default_bls_output(periods)

    # ── Extract best parameters ───────────────────────────────
    idx          = int(np.argmax(best_result.power))
    best_period  = float(best_result.period[idx])
    t0           = float(best_result.transit_time[idx])
    duration     = float(best_result.duration[idx])

    # Top 3 candidates
    top3_idx     = np.argsort(best_result.power)[-3:][::-1]
    top3_periods = np.array(best_result.period)[top3_idx]
    top3_powers  = np.array(best_result.power)[top3_idx]

    log.info(
        "BLS best: P=%.4fd | t0=%.3f | dur=%.4fd | power=%.4f | %.1fs",
        best_period, t0, duration, best_power, bls_runtime,
    )

    # ── Harmonic check ────────────────────────────────────────
    corrected_period, harmonic_flag = check_harmonics(best_period, best_result, periods)
    if harmonic_flag:
        log.info(
            "Harmonic flag: %.4fd → corrected to %.4fd",
            best_period, corrected_period,
        )

    return {
        "best_period":      best_period,
        "t0":               t0,
        "duration":         duration,
        "bls_power":        best_power,
        "bls_result":       best_result,
        "periods":          periods,
        "top3_periods":     top3_periods,
        "top3_powers":      top3_powers,
        "harmonic_flag":    harmonic_flag,
        "corrected_period": corrected_period,
        "n_bins_searched":  len(periods),
        "bls_runtime_s":    bls_runtime,
        "timeout_flag":     timeout_flag,
    }


def check_harmonics(
    best_period: float,
    bls_result: Any,
    periods: np.ndarray,
) -> Tuple[float, bool]:
    """
    Check if P/2 or 2P gives a higher BLS power than best_period.

    This catches the common EB alias where an eclipsing binary's true
    period (P_EB) is detected at P/2 because both eclipses look similar.

    Returns
    -------
    (corrected_period, harmonic_flag)
    """
    period_arr = np.array(bls_result.period)
    power_arr  = np.array(bls_result.power)
    best_power = float(np.max(power_arr))

    def _power_at(p: float) -> float:
        """Interpolate BLS power at a specific period."""
        if p < periods[0] or p > periods[-1]:
            return -np.inf
        idx = np.argmin(np.abs(period_arr - p))
        return float(power_arr[idx])

    candidates = {
        "P/2": best_period / 2.0,
        "2P":  best_period * 2.0,
    }

    for label, p_cand in candidates.items():
        p_power = _power_at(p_cand)
        if p_power > best_power * 0.90:   # Within 10% of best
            log.info("Harmonic %s (P=%.4fd) has power %.4f vs best %.4f", label, p_cand, p_power, best_power)
            # Return the candidate period that makes more physical sense
            # For 2P: likely means the true EB period is 2× detected period
            if label == "2P":
                return p_cand, True
            # For P/2: detected period may already be correct; flag for inspection
            return p_cand, True

    return best_period, False


def _default_bls_output(periods: np.ndarray) -> Dict[str, Any]:
    """Return a safe default dict when BLS produces no result."""
    return {
        "best_period":      1.0,
        "t0":               0.0,
        "duration":         0.1,
        "bls_power":        0.0,
        "bls_result":       None,
        "periods":          periods,
        "top3_periods":     np.array([1.0, 2.0, 3.0]),
        "top3_powers":      np.array([0.0, 0.0, 0.0]),
        "harmonic_flag":    False,
        "corrected_period": 1.0,
        "n_bins_searched":  len(periods),
        "bls_runtime_s":    0.0,
        "timeout_flag":     True,
    }
