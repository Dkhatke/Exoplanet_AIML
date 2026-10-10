# src/recycler/noise_module.py
# ============================================================
# Phase 5.4 - Noise / Instrumental Diagnostic Module
#
# Runs when the triage classifier routes a target to "NOISE".
# Quantifies the photometric noise floor and separates:
#   * high-frequency white noise (photon + read noise), and
#   * low-frequency instrumental systematics (thermal drift,
#     pointing jitter, stray light), and
#   * a possible periodic instrumental signal at the TESS
#     orbital period (~13.7 days) or its harmonics.
#
# CDPP (Combined Differential Photometric Precision) is the
# standard TESS/Kepler noise metric: the rms scatter expected
# on a timescale tau, in ppm. We estimate it by binning.
# ============================================================

from __future__ import annotations

from typing import Dict, Any

import numpy as np
from lightkurve import LightCurve

from src.utils.logger import get_logger

log = get_logger(__name__)

# CDPP reference timescales (hours). 1 hr and 6 hr are the two
# canonical TESS CDPP products used for transit detectability.
CDPP_SCALES_HOURS = (1.0, 6.0)

# TESS orbital period (days) and its first two harmonics. A peak in
# the LS periodogram near these is a red flag for an instrumental,
# not astrophysical, signal.
TESS_ORBIT_DAYS = 13.7
ORBIT_HARMONICS = (TESS_ORBIT_DAYS, TESS_ORBIT_DAYS / 2.0, TESS_ORBIT_DAYS * 2.0)
ORBIT_MATCH_TOL = 0.15    # fractional tolerance for "near" the orbit period

# Systematics heuristic: low-frequency rms / high-frequency rms.
SYSTEMATIC_RATIO_THRESHOLD = 2.0


def _flux_time(lc: LightCurve):
    t = lc.time.value if hasattr(lc.time, "value") else np.asarray(lc.time)
    f = lc.flux.value if hasattr(lc.flux, "value") else np.asarray(lc.flux)
    return t, f


def _estimate_cdpp_ppm(t: np.ndarray, f: np.ndarray, tau_hours: float) -> float:
    """
    Estimate CDPP (ppm) on a timescale tau by binning the (already
    near-flat) detrended flux and taking the rms of the bin medians,
    scaled back to per-cadence precision.

    Returns 0.0 if there is not enough baseline to form >= 3 bins.
    """
    if len(t) < 10:
        return 0.0
    cadence_days = float(np.median(np.diff(np.sort(t))))
    if cadence_days <= 0:
        return 0.0
    bin_days = tau_hours / 24.0
    n_per_bin = max(1, int(round(bin_days / cadence_days)))

    order = np.argsort(t)
    t_s, f_s = t[order], f[order]
    n_bins = len(f_s) // n_per_bin
    if n_bins < 3:
        # Not enough baseline; fall back to per-cadence rms.
        return float(np.std(f_s) * 1e6)

    trimmed = f_s[: n_bins * n_per_bin].reshape(n_bins, n_per_bin)
    bin_means = np.mean(trimmed, axis=1)
    # CDPP = rms scatter of the flux binned at this cadence (ppm).
    # This is the standard TESS/Kepler definition and naturally decreases
    # as the bin timescale grows (more averaging).
    cdpp = float(np.std(bin_means) * 1e6)
    return cdpp


def _high_vs_low_freq(f: np.ndarray) -> Dict[str, float]:
    """
    Split variance into a high-frequency (white) component from the
    median absolute successive difference and a total component, to
    flag low-frequency systematics.
    """
    if len(f) < 5:
        return {"hf_rms": 0.0, "total_rms": 0.0, "ratio": 1.0}
    diff = np.diff(f)
    # MAD of successive differences / sqrt(2) ~ robust white-noise sigma.
    hf = 1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2.0)
    total = float(np.std(f))
    ratio = float(total / hf) if hf > 0 else 1.0
    return {"hf_rms": float(hf), "total_rms": total, "ratio": ratio}


def _orbit_periodicity(t: np.ndarray, f: np.ndarray) -> Dict[str, Any]:
    """Look for excess LS power near the TESS orbital period / harmonics."""
    if len(t) < 30 or (t[-1] - t[0]) < TESS_ORBIT_DAYS:
        return {"checked": False, "reason": "baseline shorter than TESS orbit"}
    try:
        from astropy.timeseries import LombScargle
        f_c = f - np.nanmedian(f)
        ls = LombScargle(t, f_c)
        freq, power = ls.autopower(
            minimum_frequency=1.0 / (TESS_ORBIT_DAYS * 3.0),
            maximum_frequency=1.0 / (TESS_ORBIT_DAYS / 3.0),
            samples_per_peak=10,
        )
        if len(power) == 0:
            return {"checked": False, "reason": "empty LS near orbit band"}
        peak_period = float(1.0 / freq[int(np.argmax(power))])
        peak_power = float(np.max(power))
        # Compare to overall median power as a significance proxy.
        med = float(np.median(power)) + 1e-12
        near = any(
            abs(peak_period - h) / h < ORBIT_MATCH_TOL for h in ORBIT_HARMONICS
        )
        return {
            "checked": True,
            "near_orbit_period": near,
            "peak_period_days": peak_period,
            "peak_over_median": peak_power / med,
            "instrument_period_flag": bool(near and peak_power / med > 5.0),
        }
    except Exception as exc:
        return {"checked": False, "reason": f"orbit LS failed: {exc}"}


def _quality_correlation(lc: LightCurve) -> Dict[str, Any]:
    """
    If a quality array is present, compare scatter in flagged vs clean
    cadences. Requires the LightCurve to carry a .quality attribute.
    """
    if not hasattr(lc, "quality") or lc.quality is None:
        return {"available": False}
    try:
        q = lc.quality.value if hasattr(lc.quality, "value") else np.asarray(lc.quality)
        f = lc.flux.value if hasattr(lc.flux, "value") else np.asarray(lc.flux)
        good = q == 0
        bad = ~good
        if np.sum(bad) < 3 or np.sum(good) < 10:
            return {"available": True, "n_flagged": int(np.sum(bad)),
                    "reason": "too few flagged cadences"}
        return {
            "available": True,
            "n_flagged": int(np.sum(bad)),
            "rms_clean": float(np.std(f[good])),
            "rms_flagged": float(np.std(f[bad])),
            "systematic_flag": bool(np.std(f[bad]) > 2.0 * np.std(f[good])),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}



def analyze_noise(
    lc_detrended: LightCurve,
    feature_dict: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Characterize the noise floor of a target routed to the NOISE class.

    Parameters
    ----------
    lc_detrended : Detrended LightCurve (near-flat if truly noise).
    feature_dict : Optional 25-feature dict; oot_* features are echoed
                   for provenance.

    Returns
    -------
    noise_params : dict with CDPP at 1 hr / 6 hr, high- vs low-frequency
                   split, quality-flag correlation, and a TESS-orbit
                   periodicity check.
    """
    feature_dict = feature_dict or {}
    t, f = _flux_time(lc_detrended)
    finite = np.isfinite(t) & np.isfinite(f)
    t, f = t[finite], f[finite]

    if len(t) < 10:
        return {
            "classification": "NOISE",
            "notes": [f"too few cadences: {len(t)}"],
            "cdpp_1hr_ppm": 0.0,
            "cdpp_6hr_ppm": 0.0,
            "systematic_flag": False,
            "instrument_period_flag": False,
        }

    # -- CDPP at reference timescales -----------------------------
    cdpp = {f"cdpp_{int(h)}hr_ppm": _estimate_cdpp_ppm(t, f, h)
            for h in CDPP_SCALES_HOURS}

    # -- High- vs low-frequency split -----------------------------
    hf = _high_vs_low_freq(f)
    systematic_flag = bool(hf["ratio"] > SYSTEMATIC_RATIO_THRESHOLD)

    # -- Quality-flag correlation --------------------------------
    qual = _quality_correlation(lc_detrended)

    # -- TESS-orbit periodicity ----------------------------------
    orbit = _orbit_periodicity(t, f)
    instrument_period_flag = bool(orbit.get("instrument_period_flag", False))

    baseline = float(t[-1] - t[0])
    log.info(
        "noise_module: cdpp1hr=%.1f cdpp6hr=%.1f ppm | lf/hf=%.2f "
        "systematic=%s orbit_periodic=%s",
        cdpp["cdpp_1hr_ppm"], cdpp["cdpp_6hr_ppm"], hf["ratio"],
        systematic_flag, instrument_period_flag,
    )

    notes = []
    if systematic_flag:
        notes.append(
            f"Low-frequency rms {hf['ratio']:.1f}x white noise - possible "
            "instrumental systematics."
        )
    if instrument_period_flag:
        notes.append("Excess power near TESS orbital period - likely instrumental.")
    if qual.get("systematic_flag"):
        notes.append("Quality-flagged cadences are >2x noisier than clean ones.")

    return {
        "classification": "NOISE",
        "n_cadences": int(len(t)),
        "baseline_days": baseline,
        **cdpp,
        "hf_rms": hf["hf_rms"],
        "total_rms": hf["total_rms"],
        "lf_over_hf_ratio": hf["ratio"],
        "systematic_flag": systematic_flag,
        "quality_correlation": qual,
        "orbit_periodicity": orbit,
        "instrument_period_flag": instrument_period_flag,
        "phase2_oot_echo": {
            k: feature_dict.get(k)
            for k in ("oot_mad", "oot_rms", "oot_p2p")
        },
        "notes": notes,
    }
