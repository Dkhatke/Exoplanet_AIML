# src/preprocessing/preprocessor.py
# ============================================================
# Phase 1: Transit-preserving two-pass detrending preprocessor.
#
# Critical fix for AUD-04: the original preprocessor used a
# fixed Savitzky-Golay window without masking transits, which
# eroded transit depth. This version:
#   1. Does a quick BLS pass to identify candidate transit times
#   2. Masks those windows before the detrending spline/filter
#   3. Applies the trend to the FULL (unmasked) light curve
#   → Transits are preserved; stellar variability is removed.
# ============================================================

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from lightkurve import LightCurve

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Asymmetric Outlier Clipping ───────────────────────────────

def clip_outliers(
    lc: LightCurve,
    sigma_upper: float = 3.0,
    sigma_lower: float = 7.0,
) -> Tuple[LightCurve, int]:
    """
    Asymmetric sigma-clipping:
    - Clips upward outliers aggressively (stellar flares, cosmic rays).
    - Clips downward outliers very conservatively (deep transits should NOT be clipped).

    Parameters
    ----------
    sigma_upper : Clip upward outliers beyond this many sigma.
    sigma_lower : Clip downward outliers beyond this many sigma.

    Returns
    -------
    (clipped_lc, n_removed)
    """
    # Safely extract flux values whether Astropy Quantity or plain ndarray
    raw_flux = lc.flux
    flux = raw_flux.value if hasattr(raw_flux, "value") else np.asarray(raw_flux)

    median = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - median))
    sigma = 1.4826 * mad   # MAD → standard deviation equivalent

    if sigma == 0 or not np.isfinite(sigma):
        return lc, 0

    upper_bound = median + sigma_upper * sigma
    lower_bound = median - sigma_lower * sigma
    good = (flux <= upper_bound) & (flux >= lower_bound)
    n_removed = int(np.sum(~good))

    if n_removed > 0:
        log.debug(
            "Clipped %d outliers (upper >%.1fσ, lower <%.1fσ)",
            n_removed, sigma_upper, sigma_lower,
        )

    return lc[good], n_removed


# ── Quick BLS for transit masking ────────────────────────────

def _quick_bls_period(lc: LightCurve) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Fast BLS scan to find the most likely transit period, t0, and duration.
    Used only for building a transit mask — NOT the final detection.

    Returns (period, t0, duration) or (None, None, None) if no signal found.
    """
    try:
        from astropy.timeseries import BoxLeastSquares
        t = lc.time.value
        f = lc.flux.value

        baseline = t[-1] - t[0]
        period_max = min(15.0, baseline / 3.0)   # Need at least 3 transits
        if period_max < 0.5:
            return None, None, None

        periods = np.linspace(0.5, period_max, 2000)
        durations = [0.05, 0.10, 0.15]

        best_power = -np.inf
        best_result = None
        for dur in durations:
            bls = BoxLeastSquares(t, f)
            result = bls.power(periods, dur)
            peak = float(np.max(result.power))
            if peak > best_power:
                best_power = peak
                best_result = result

        if best_result is None:
            return None, None, None

        idx = int(np.argmax(best_result.power))
        period  = float(best_result.period[idx])
        t0      = float(best_result.transit_time[idx])
        duration = float(best_result.duration[idx])
        return period, t0, duration

    except Exception as e:
        log.debug("Quick BLS failed: %s", e)
        return None, None, None


def _build_transit_mask(
    time: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    padding_factor: float = 1.5,
) -> np.ndarray:
    """
    Return a boolean mask — True where a cadence is IN transit (to be masked from detrending).

    Parameters
    ----------
    padding_factor : Extra width around transit edges to ensure full masking.
    """
    half_width = 0.5 * duration * padding_factor
    phase = ((time - t0) % period)
    phase[phase > period / 2] -= period
    in_transit = np.abs(phase) < half_width
    return in_transit


# ── Main Detrending Function ──────────────────────────────────

def detrend_lc(
    lc: LightCurve,
    window_length_days: float = 0.5,
    sigma_upper: float = 3.0,
    sigma_lower: float = 7.0,
    use_wotan: bool = True,
) -> Tuple[LightCurve, LightCurve]:
    """
    Two-pass transit-preserving detrending pipeline.

    Pass 1: Quick BLS to locate candidate transit windows.
    Mask: Cadences inside transit windows are excluded from detrending.
    Pass 2: Detrend the light curve using masked data only.
    Apply: Apply the detrending to the FULL light curve (including transits).

    Parameters
    ----------
    lc                 : Input stitched LightCurve (normalised to ~1.0).
    window_length_days : Smoothing window in days for Wotan/SG filter.
    sigma_upper        : Upward sigma clip threshold.
    sigma_lower        : Downward sigma clip threshold.
    use_wotan          : Use Wotan biweight filter (True) or LK SG fallback (False).

    Returns
    -------
    (lc_raw_norm, lc_detrended)
        lc_raw_norm  : Normalised + outlier-clipped; stellar trend still present.
        lc_detrended : Flattened; transit depth is preserved.
    """
    # ── Step 0: Normalise and clip outliers ──────────────────
    lc_norm = lc.normalize()
    lc_clipped, n_removed = clip_outliers(lc_norm, sigma_upper=sigma_upper, sigma_lower=sigma_lower)
    log.info("Preprocessing: %d cadences after clipping (%d removed)", len(lc_clipped), n_removed)

    # Store as raw_norm output (has stellar variability but no outliers)
    lc_raw_norm = lc_clipped

    if len(lc_clipped) < 100:
        log.warning("Too few cadences (%d) after clipping — returning raw_norm as detrended", len(lc_clipped))
        return lc_raw_norm, lc_raw_norm

    # ── Step 1: Quick BLS scan for transit mask ───────────────
    period, t0, duration = _quick_bls_period(lc_clipped)

    time_arr = lc_clipped.time.value
    flux_arr = lc_clipped.flux.value

    if period is not None:
        # Sanity check: reject unphysical quick-BLS result
        # If duration/period > 0.3, the BLS likely fit noise on a short baseline
        duration_ratio = duration / period
        if duration_ratio > 0.3:
            log.info(
                "Quick BLS duration/period ratio %.2f is unphysical — skipping transit mask",
                duration_ratio,
            )
            period = None   # Discard bad detection; proceed without masking

    if period is not None:
        transit_mask = _build_transit_mask(time_arr, period, t0, duration)
        n_masked = int(np.sum(transit_mask))
        log.info(
            "Transit mask: P=%.4fd, t0=%.4f, dur=%.4fd → %d cadences masked (%.1f%%)",
            period, t0, duration, n_masked, 100 * n_masked / len(time_arr),
        )
        # Cadences to use for fitting the baseline (outside transits)
        fit_mask = ~transit_mask
    else:
        log.info("No significant transit signal found — detrending without mask")
        fit_mask = np.ones(len(time_arr), dtype=bool)

    # ── Step 2: Detrend ──────────────────────────────────────
    cadence_days = float(np.nanmedian(np.diff(time_arr)))
    window_length_cadences = max(11, int(window_length_days / cadence_days))
    # Ensure odd window length for Savitzky-Golay
    if window_length_cadences % 2 == 0:
        window_length_cadences += 1

    try:
        if use_wotan:
            from wotan import flatten
            # Flatten only using out-of-transit cadences; apply to all cadences
            flat_flux, trend = flatten(
                time_arr,
                flux_arr,
                method="biweight",
                window_length=window_length_days,
                mask=transit_mask if period is not None else None,
                return_trend=True,
            )
        else:
            raise ImportError("Wotan disabled")

    except (ImportError, Exception) as e:
        log.warning("Wotan detrending failed (%s) — using Lightkurve SG fallback", e)
        # Fallback: use LightKurve's built-in flatten on masked light curve
        lc_for_fit = lc_clipped[fit_mask] if period is not None else lc_clipped
        lc_flat_fit, trend_fit = lc_for_fit.flatten(
            window_length=window_length_cadences,
            return_trend=True,
        )
        # Interpolate trend to full time grid
        from scipy.interpolate import interp1d
        interp_fn = interp1d(
            lc_for_fit.time.value,
            trend_fit.flux.value,
            kind="linear",
            fill_value="extrapolate",
        )
        flat_flux = flux_arr / interp_fn(time_arr)

    # ── Step 3: Build output LightCurve ──────────────────────
    import astropy.units as u
    from astropy.time import Time

    lc_detrended = LightCurve(
        time=lc_clipped.time,
        flux=flat_flux * u.dimensionless_unscaled,
        flux_err=lc_clipped.flux_err,
    )

    log.info(
        "Detrending complete: baseline=%.1fd | OOT std=%.0fppm",
        time_arr[-1] - time_arr[0],
        float(np.std(flat_flux) * 1e6),
    )

    return lc_raw_norm, lc_detrended
