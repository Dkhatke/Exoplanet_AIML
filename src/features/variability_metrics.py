# src/features/variability_metrics.py
# ============================================================
# Phase 2: Out-of-transit variability and Lomb-Scargle features.
# These discriminate VARIABLE stars from PLANET/EB candidates.
# ============================================================

from __future__ import annotations
from typing import Dict
import numpy as np
from lightkurve import LightCurve
from src.utils.logger import get_logger

log = get_logger(__name__)


def compute_out_of_transit_variability(
    lc: LightCurve,
    period: float,
    t0: float,
    duration: float,
) -> Dict[str, float]:
    """
    Measure variability of out-of-transit flux.

    A variable star shows high OOT variability even when transits are masked.
    A quiet planet host has flat OOT flux.

    Returns
    -------
    dict: oot_mad, oot_rms, oot_p2p, oot_skew
    """
    from scipy.stats import skew as scipy_skew

    t = lc.time.value
    f = lc.flux.value if hasattr(lc.flux, "value") else np.array(lc.flux)

    # Build OOT mask: exclude primary transit window (and secondary at phase 0.5)
    phase = ((t - t0) % period)
    phase[phase > period / 2] -= period
    half_dur = 0.5 * duration

    phase_sec = ((t - (t0 + 0.5 * period)) % period)
    phase_sec[phase_sec > period / 2] -= phase_sec[phase_sec > period / 2]

    oot = np.abs(phase) > half_dur * 1.5

    if np.sum(oot) < 20:
        return {"oot_mad": 0.0, "oot_rms": 0.0, "oot_p2p": 0.0, "oot_skew": 0.0}

    f_oot = f[oot]
    median = np.nanmedian(f_oot)
    mad = float(np.nanmedian(np.abs(f_oot - median)))
    rms = float(np.sqrt(np.nanmean((f_oot - median) ** 2)))
    p2p = float(np.nanmax(f_oot) - np.nanmin(f_oot))
    skewness = float(scipy_skew(f_oot)) if len(f_oot) >= 8 else 0.0

    return {
        "oot_mad":  mad,
        "oot_rms":  rms,
        "oot_p2p":  p2p,
        "oot_skew": skewness,
    }


def compute_lomb_scargle_peak(
    lc: LightCurve,
    period_range: tuple[float, float] = (0.1, 50.0),
) -> Dict[str, float]:
    """
    Compute the Lomb-Scargle periodogram and return the dominant peak.

    A strongly periodic variable has high LS power and low FAP.
    A quiet star or planet host has low LS power everywhere.

    Returns
    -------
    dict: ls_peak_period, ls_peak_power, ls_fap
    """
    from astropy.timeseries import LombScargle

    t = lc.time.value
    f = lc.flux.value if hasattr(lc.flux, "value") else np.array(lc.flux)

    # Subtract median to centre around zero
    f_centred = f - np.nanmedian(f)

    try:
        ls = LombScargle(t, f_centred)
        freq, power = ls.autopower(
            minimum_frequency=1.0 / period_range[1],
            maximum_frequency=1.0 / period_range[0],
            samples_per_peak=5,
        )

        if len(power) == 0:
            raise ValueError("Empty LS result")

        peak_idx   = int(np.argmax(power))
        peak_power = float(power[peak_idx])
        peak_freq  = float(freq[peak_idx])
        peak_period = 1.0 / peak_freq

        # False Alarm Probability at the peak power level
        fap = float(ls.false_alarm_probability(peak_power, method="baluev"))

        return {
            "ls_peak_period": peak_period,
            "ls_peak_power":  peak_power,
            "ls_fap":         fap,
        }

    except Exception as e:
        log.debug("Lomb-Scargle failed: %s", e)
        return {"ls_peak_period": 1.0, "ls_peak_power": 0.0, "ls_fap": 1.0}


def compute_flux_percentile_ratio(lc: LightCurve) -> Dict[str, float]:
    """
    Compute flux percentile range metrics.

    Captures the amplitude of variability without being sensitive to
    individual outliers.

    Returns
    -------
    dict: p5_p95_ratio, p10_p90_ratio, median_flux
    """
    f = lc.flux.value if hasattr(lc.flux, "value") else np.array(lc.flux)
    f = f[np.isfinite(f)]

    if len(f) < 10:
        return {"p5_p95_ratio": 0.0, "p10_p90_ratio": 0.0, "median_flux": 1.0}

    p5,  p95 = np.percentile(f, [5,  95])
    p10, p90 = np.percentile(f, [10, 90])
    median   = float(np.median(f))

    return {
        "p5_p95_ratio":  float(p95 - p5),
        "p10_p90_ratio": float(p90 - p10),
        "median_flux":   median,
    }
