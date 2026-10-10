# src/recycler/variable_module.py
# ============================================================
# Phase 5.3 - Variable Star Diagnostic Module
#
# Runs when the triage classifier routes a target to "VARIABLE".
# A pulsating or rotating star produces smooth, quasi-periodic
# flux modulation - NOT a box-shaped transit. This module
# characterizes that modulation so an astronomer can tell a
# delta-Scuti pulsator from a spotted rotator, etc.
#
# Steps:
#   1. Lomb-Scargle periodogram -> dominant period + harmonics.
#   2. Amplitude metrics (min-max, mean-to-peak).
#   3. Mode classification: PULSATION / ROTATION / QUASI_PERIODIC.
#   4. Optional Gaia DR3 variability-class cross-check.
# ============================================================

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
from lightkurve import LightCurve

from src.utils.logger import get_logger

log = get_logger(__name__)

# Harmonics are considered "present" if their LS power is within this
# fraction of the dominant peak's power.
HARMONIC_POWER_FRAC = 0.5

# Mode-classification heuristics.
# High FAP => not convincingly periodic => quasi-periodic.
FAP_QUASI_THRESHOLD = 0.01
# A strong 2P or 0.5P harmonic relative to the fundamental hints at a
# non-sinusoidal (often rotational / spotted) waveform.
HARMONIC_ROTATION_FRAC = 0.3


def _flux_time(lc: LightCurve):
    t = lc.time.value if hasattr(lc.time, "value") else np.asarray(lc.time)
    f = lc.flux.value if hasattr(lc.flux, "value") else np.asarray(lc.flux)
    return t, f


def _lomb_scargle(t: np.ndarray, f: np.ndarray,
                  period_range=(0.1, 50.0)) -> Dict[str, Any]:
    """Full Lomb-Scargle periodogram with dominant peak + FAP."""
    from astropy.timeseries import LombScargle

    f_c = f - np.nanmedian(f)
    ls = LombScargle(t, f_c)
    freq, power = ls.autopower(
        minimum_frequency=1.0 / period_range[1],
        maximum_frequency=1.0 / period_range[0],
        samples_per_peak=10,
    )
    if len(power) == 0:
        raise ValueError("empty LS result")

    peak_idx = int(np.argmax(power))
    peak_power = float(power[peak_idx])
    peak_period = float(1.0 / freq[peak_idx])
    try:
        fap = float(ls.false_alarm_probability(peak_power, method="baluev"))
    except Exception:
        fap = 1.0

    # Power at harmonics (2P and P/2), for waveform shape diagnostics.
    def _power_at_period(p):
        if p < period_range[0] or p > period_range[1]:
            return 0.0
        target_f = 1.0 / p
        i = int(np.argmin(np.abs(freq - target_f)))
        return float(power[i])

    return {
        "freq": freq,
        "power": power,
        "dominant_period": peak_period,
        "peak_power": peak_power,
        "fap": fap,
        "power_at_2P": _power_at_period(peak_period * 2.0),
        "power_at_halfP": _power_at_period(peak_period * 0.5),
    }


def _classify_mode(ls: Dict[str, Any]) -> str:
    """Heuristic variability-mode label from the periodogram shape."""
    peak = ls["peak_power"] + 1e-12
    harmonic_2p = ls["power_at_2P"] / peak
    harmonic_half = ls["power_at_halfP"] / peak

    if ls["fap"] > FAP_QUASI_THRESHOLD:
        return "QUASI_PERIODIC"
    if max(harmonic_2p, harmonic_half) > HARMONIC_ROTATION_FRAC:
        # Strong harmonic content => non-sinusoidal, consistent with
        # rotational modulation by spots (or a sawtooth-like pulsator).
        return "ROTATION"
    return "PULSATION"


def _gaia_crosscheck(tic_id: Optional[int],
                     gaia_csv: Optional[str] = None) -> Dict[str, Any]:
    """
    Best-effort cross-check against a local Gaia DR3 variability table.
    Never raises; returns an empty dict with a reason if unavailable.
    The heavy Gaia query itself lives in Phase 3 (build_dataset); here we
    only consult a CSV if one is provided/present.
    """
    if tic_id is None:
        return {"checked": False, "reason": "no tic_id provided"}
    if gaia_csv is None:
        try:
            from config import CATALOG_DIR
            import os
            gaia_csv = os.path.join(CATALOG_DIR, "gaia_dr3_variables.csv")
        except Exception:
            return {"checked": False, "reason": "gaia catalog path unavailable"}

    import os
    if not os.path.exists(gaia_csv):
        return {"checked": False, "reason": f"no Gaia catalog at {gaia_csv}"}

    try:
        import pandas as pd
        df = pd.read_csv(gaia_csv)
        # Try common join columns
        for col in ("tic_id", "tic", "TIC"):
            if col in df.columns:
                match = df[df[col].astype("Int64") == int(tic_id)]
                if len(match):
                    row = match.iloc[0]
                    cls_col = next((c for c in
                                    ("best_class_name", "best_class", "vari_class")
                                    if c in df.columns), None)
                    return {
                        "checked": True,
                        "matched": True,
                        "gaia_class": str(row[cls_col]) if cls_col else "unknown",
                    }
                return {"checked": True, "matched": False}
        return {"checked": False, "reason": "no tic join column in Gaia catalog"}
    except Exception as exc:
        return {"checked": False, "reason": f"gaia crosscheck failed: {exc}"}



def analyze_variable(
    lc_raw_norm: LightCurve,
    lc_detrended: LightCurve,
    tic_id: Optional[int] = None,
    period_range: tuple = (0.1, 50.0),
) -> Dict[str, Any]:
    """
    Characterize a target routed to the VARIABLE class.

    Parameters
    ----------
    lc_raw_norm  : Normalized (but not detrended) LightCurve. Variability
                   is measured here so genuine stellar signal is not
                   removed by detrending.
    lc_detrended : Detrended LightCurve (kept for API symmetry / future use).
    tic_id       : Optional TIC ID for a Gaia DR3 cross-check.
    period_range : (min, max) period search window in days.

    Returns
    -------
    variable_params : dict with dominant period, harmonics, amplitude,
                      mode classification, and Gaia cross-check result.
    """
    # Measure variability on the RAW normalized curve (signal preserved).
    t, f = _flux_time(lc_raw_norm)
    finite = np.isfinite(t) & np.isfinite(f)
    t, f = t[finite], f[finite]

    if len(t) < 20:
        log.warning("variable_module: too few cadences (%d) for analysis", len(t))
        return {
            "classification": "VARIABLE",
            "mode": "UNKNOWN",
            "dominant_period_days": 0.0,
            "notes": [f"too few cadences: {len(t)}"],
        }

    ls = _lomb_scargle(t, f, period_range=period_range)
    mode = _classify_mode(ls)

    # -- Amplitude metrics ----------------------------------------
    f_min = float(np.min(f))
    f_max = float(np.max(f))
    f_mean = float(np.mean(f))
    amp_minmax = f_max - f_min
    amp_mean_to_peak = f_max - f_mean
    # Robust amplitude from percentiles (outlier-resistant).
    p5, p50, p95 = np.percentile(f, [5, 50, 95])
    amp_p5_p95 = float(p95 - p5)

    # -- Harmonic presence flags ----------------------------------
    peak = ls["peak_power"] + 1e-12
    harmonics_present = bool(
        max(ls["power_at_2P"], ls["power_at_halfP"]) / peak > HARMONIC_POWER_FRAC * 0.5
    )

    # -- Possible physical type labels ----------------------------
    type_labels = _possible_types(mode, ls["dominant_period"])

    # -- Gaia cross-check -----------------------------------------
    gaia = _gaia_crosscheck(tic_id)

    log.info(
        "variable_module: P=%.4fd power=%.3f fap=%.2e mode=%s amp_p5p95=%.4f",
        ls["dominant_period"], ls["peak_power"], ls["fap"], mode, amp_p5_p95,
    )

    return {
        "classification": "VARIABLE",
        "mode": mode,
        "dominant_period_days": ls["dominant_period"],
        "ls_peak_power": ls["peak_power"],
        "ls_fap": ls["fap"],
        "power_at_2P": ls["power_at_2P"],
        "power_at_halfP": ls["power_at_halfP"],
        "harmonics_present": harmonics_present,
        "amplitude_minmax": float(amp_minmax),
        "amplitude_mean_to_peak": float(amp_mean_to_peak),
        "amplitude_p5_p95": float(amp_p5_p95),
        "flux_min": f_min,
        "flux_max": f_max,
        "flux_mean": f_mean,
        "possible_types": type_labels,
        "gaia_crosscheck": gaia,
        "n_cadences": int(len(t)),
    }


def _possible_types(mode: str, period: float):
    """Suggest candidate variability classes from mode + period (heuristic)."""
    labels = []
    if mode == "PULSATION":
        if period < 0.2:
            labels += ["DSCT", "SXPHE", "roAp"]
        elif period < 1.0:
            labels += ["DSCT", "GDOR", "SXPHE"]
        else:
            labels += ["RR", "CEP", "long-period pulsator"]
    elif mode == "ROTATION":
        labels += ["ROT (spotted rotator)", "BY_Dra", "FKCom"]
    elif mode == "QUASI_PERIODIC":
        labels += ["quasi-periodic (e.g. solar-like)", "irregular / multi-periodic"]
    else:
        labels = ["unknown"]
    return labels
