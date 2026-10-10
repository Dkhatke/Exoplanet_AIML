# src/recycler/eb_module.py
# ============================================================
# Phase 5.2 - Eclipsing Binary (EB) Diagnostic Module
#
# Runs when the triage classifier routes a target to "EB".
# A single stellar transit (planet) and an eclipsing binary can
# look similar in a BLS periodogram. This module applies the
# classic EB discriminators to justify / refine the call:
#
#   1. Secondary eclipse presence (EB almost always has one;
#      a planet's secondary is negligible).
#   2. Odd/even transit-depth mismatch (two different bodies
#      eclipse on alternating cycles).
#   3. Light-curve morphology: V-shape + rounded = CONTACT;
#      flat-bottomed U-shape = DETACHED.
#   4. Period-doubling test: if P/2 produces an equally deep
#      primary, the true period may be 2x the detected one.
#
# Returns an eb_params dict with a subtype classification.
# ============================================================

from __future__ import annotations

from typing import Dict, Any

import numpy as np
from lightkurve import LightCurve

from src.utils.logger import get_logger

log = get_logger(__name__)

# Thresholds
SECONDARY_SNR_THRESHOLD = 3.0     # > this => secondary eclipse "detected"
ODD_EVEN_Z_THRESHOLD = 3.0        # > this => significant odd/even depth mismatch
PERIOD_DOUBLING_TOL = 0.95        # P/2 depth >= 0.95 * P depth => doubling candidate
# Morphology boundaries on the V-shape metric in [0, 1].
# Higher = more V-shaped (pointed) = more likely CONTACT.
CONTACT_V_SHAPE = 0.65
DETACHED_V_SHAPE = 0.40


def _flux_time(lc: LightCurve):
    t = lc.time.value if hasattr(lc.time, "value") else np.asarray(lc.time)
    f = lc.flux.value if hasattr(lc.flux, "value") else np.asarray(lc.flux)
    return t, f


def _depth_at(lc: LightCurve, period: float, t0: float, half_width: float):
    """Measure eclipse depth near a reference epoch t0 for a given period."""
    t, f = _flux_time(lc)
    phase = ((t - t0) % period)
    phase[phase > period / 2] -= period
    in_e = np.abs(phase) < half_width
    oot = ~in_e
    if np.sum(in_e) < 3 or np.sum(oot) < 5:
        return 0.0, 0.0, int(np.sum(in_e))
    oot_med = float(np.median(f[oot]))
    in_med = float(np.median(f[in_e]))
    depth = max(0.0, oot_med - in_med)
    oot_std = float(np.std(f[oot])) + 1e-9
    snr = depth / (oot_std / np.sqrt(np.sum(in_e)))
    return depth, float(snr), int(np.sum(in_e))


def _secondary_phase_offset(lc: LightCurve, period: float, t0: float):
    """Find the phase of the deepest point near phase 0.5 (secondary eclipse)."""
    t, f = _flux_time(lc)
    phase = ((t - t0) % period) / period       # in [0, 1)
    # Look only in the half-orbit window around 0.5
    window = (phase > 0.35) & (phase < 0.65)
    if np.sum(window) < 5:
        return 0.5, False
    # Bin to find the deepest phase
    edges = np.linspace(0.35, 0.65, 31)
    idx = np.digitize(phase[window], edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    f_win = f[window]
    best_bin, best_flux = -1, np.inf
    for b in range(len(edges) - 1):
        m = idx == b
        if np.sum(m) >= 2:
            med = float(np.median(f_win[m]))
            if med < best_flux:
                best_flux, best_bin = med, b
    if best_bin < 0:
        return 0.5, False
    sec_phase = 0.5 * (edges[best_bin] + edges[best_bin + 1])
    return float(sec_phase), True


def _odd_even(lc: LightCurve, period: float, t0: float, half_width: float) -> Dict[str, float]:
    """Odd vs even eclipse depths and a simple z-score of their difference."""
    t, f = _flux_time(lc)
    phase = ((t - t0) % period)
    phase[phase > period / 2] -= period
    in_e = np.abs(phase) < half_width
    cycle = np.floor((t - t0) / period).astype(int)

    odd_d, even_d = [], []
    for cyc in np.unique(cycle[in_e]):
        m = in_e & (cycle == cyc)
        if np.sum(m) < 3:
            continue
        d = 1.0 - float(np.median(f[m]))
        (even_d if cyc % 2 == 0 else odd_d).append(d)

    if not odd_d or not even_d:
        return {"odd_depth": 0.0, "even_depth": 0.0, "ratio": 1.0, "z": 0.0,
                "significant": False}

    odd_mean = float(np.mean(odd_d))
    even_mean = float(np.mean(even_d))
    diff = abs(odd_mean - even_mean)
    s_odd = float(np.std(odd_d)) if len(odd_d) > 1 else 1e-6
    s_even = float(np.std(even_d)) if len(even_d) > 1 else 1e-6
    pooled = np.sqrt(s_odd ** 2 + s_even ** 2) + 1e-9
    z = float(diff / pooled)
    ratio = float(odd_mean / even_mean) if even_mean > 1e-9 else 1.0
    return {"odd_depth": odd_mean, "even_depth": even_mean, "ratio": ratio,
            "z": z, "significant": bool(z > ODD_EVEN_Z_THRESHOLD)}


def _v_shape(lc: LightCurve, period: float, t0: float, duration: float) -> float:
    """
    V-shape metric in [0, 1]: how pointed the eclipse is.
    ~0 => flat-bottomed U (detached); ~1 => pointed V (contact/rounded).
    """
    t, f = _flux_time(lc)
    phase = ((t - t0) % period)
    phase[phase > period / 2] -= period
    half = 0.5 * duration
    in_e = np.abs(phase) < half
    if np.sum(in_e) < 10:
        return 0.5
    centre = np.abs(phase) < 0.3 * half
    wing = (np.abs(phase) > 0.6 * half) & (np.abs(phase) < half)
    oot = np.abs(phase) > half
    if np.sum(centre) < 2 or np.sum(wing) < 2 or np.sum(oot) < 5:
        return 0.5
    f_out = float(np.median(f[oot]))
    f_c = float(np.median(f[centre]))
    f_w = float(np.median(f[wing]))
    total = f_out - f_c + 1e-9
    wing_depth = f_out - f_w
    v = wing_depth / total            # ~1 => wings as deep as centre (flat) -> U
    # Invert so higher = more V-shaped / pointed.
    return float(np.clip(1.0 - v, 0.0, 1.0))



def _classify_subtype(v_shape: float, secondary_detected: bool) -> str:
    """
    Map morphology + secondary-eclipse evidence to an EB subtype.
    This is a heuristic label for triage, not a rigorous morphological fit.
    """
    if v_shape >= CONTACT_V_SHAPE:
        # Pointed, continuously varying eclipses => components share an envelope.
        return "CONTACT"
    if v_shape <= DETACHED_V_SHAPE:
        return "DETACHED"
    return "SEMI_DETACHED"


def _has_secondary(lc: LightCurve, period: float, t0: float, half: float) -> tuple:
    """Return (secondary_snr, sec_phase, sec_depth) at a given trial period."""
    sec_phase, _ = _secondary_phase_offset(lc, period, t0)
    sec_depth, sec_snr, _ = _depth_at(lc, period, t0 + sec_phase * period, half)
    return sec_snr, sec_phase, sec_depth


def _refine_eb_period(lc: LightCurve, period: float, t0: float, half: float):
    """
    Correct the classic EB period-halving alias.

    BLS frequently reports P/2 for an eclipsing binary, because folding at
    half the true period stacks the primary and secondary eclipses into one
    deep feature. The tell-tale is that the reported period shows NO
    secondary at phase 0.5, while 2x that period DOES show a significant
    secondary (and a clean primary at phase 0). When that pattern holds we
    adopt the doubled period so downstream diagnostics see both eclipses.

    Returns (refined_period, doubled: bool).
    """
    sec_snr_half, _, _ = _has_secondary(lc, period, t0, half)
    sec_snr_full, _, _ = _has_secondary(lc, period * 2.0, t0, half)
    prim_depth_full, prim_snr_full, _ = _depth_at(lc, period * 2.0, t0, half)

    doubled = (
        sec_snr_half <= SECONDARY_SNR_THRESHOLD
        and sec_snr_full > SECONDARY_SNR_THRESHOLD
        and prim_snr_full > SECONDARY_SNR_THRESHOLD
    )
    return (period * 2.0 if doubled else period), doubled


def analyze_eb(
    lc_detrended: LightCurve,
    bls_output: Dict[str, Any],
    feature_dict: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Run the full EB diagnostic suite on a target routed to the EB class.

    Parameters
    ----------
    lc_detrended : Detrended LightCurve.
    bls_output   : BLS result dict (needs best_period / t0 / duration).
    feature_dict : Optional 25-feature dict; used to echo secondary /
                   odd-even features already computed in Phase 2.

    Returns
    -------
    eb_params : dict with primary/secondary eclipse diagnostics,
                odd/even analysis, morphology, subtype, and a
                period-doubling test.
    """
    feature_dict = feature_dict or {}
    period = float(bls_output.get("corrected_period") or bls_output.get("best_period", 1.0))
    t0 = float(bls_output.get("t0", 0.0))
    duration = float(bls_output.get("duration", 0.1))
    if period <= 0:
        period = 1.0
    if duration <= 0:
        duration = 0.01
    half = 0.5 * duration

    # -- Correct the EB period-halving alias before diagnosing ----
    # BLS often locks onto P/2; refine to the true period so the
    # secondary eclipse becomes visible.
    reported_period = period
    period, period_doubled = _refine_eb_period(lc_detrended, period, t0, half)
    if period_doubled:
        log.info(
            "eb_module: period-halving corrected %.4fd -> %.4fd", reported_period, period
        )

    # -- Primary eclipse ------------------------------------------
    prim_depth, prim_snr, prim_n = _depth_at(lc_detrended, period, t0, half)

    # -- Secondary eclipse at phase ~0.5 --------------------------
    sec_phase, found = _secondary_phase_offset(lc_detrended, period, t0)
    sec_depth, sec_snr, sec_n = _depth_at(
        lc_detrended, period, t0 + sec_phase * period, half
    )
    secondary_detected = bool(sec_snr > SECONDARY_SNR_THRESHOLD)
    sec_offset_from_half = float(sec_phase - 0.5)

    # -- Odd / even depth mismatch --------------------------------
    oe = _odd_even(lc_detrended, period, t0, half)

    # -- Morphology (V vs U) --------------------------------------
    v_shape = _v_shape(lc_detrended, period, t0, duration)
    subtype = _classify_subtype(v_shape, secondary_detected)

    # -- Period-doubling test -------------------------------------
    # (Period already refined above via _refine_eb_period; report the
    #  independent P/2 cross-check here for provenance.)
    half_depth, half_snr, _ = _depth_at(lc_detrended, period / 2.0, t0, half)
    doubling_candidate = bool(
        prim_depth > 0 and half_depth >= PERIOD_DOUBLING_TOL * prim_depth
    )
    suggested_period = period if period_doubled else (period * 2.0 if doubling_candidate else period)

    # -- Echo any pre-computed Phase-2 features for provenance ----
    eb_evidence = {
        "secondary_detected": secondary_detected,
        "odd_even_significant": oe["significant"],
        "v_shape_high": bool(v_shape >= CONTACT_V_SHAPE),
    }
    n_evidence = sum(eb_evidence.values())

    log.info(
        "eb_module: P=%.4fd prim_depth=%.4f sec_depth=%.4f(sec?=%s) "
        "oe_z=%.2f v_shape=%.2f subtype=%s doubling=%s",
        period, prim_depth, sec_depth, secondary_detected, oe["z"],
        v_shape, subtype, doubling_candidate,
    )

    return {
        "classification": "EB",
        "period_days": period,
        "reported_period_days": reported_period,
        "period_doubled": period_doubled,
        "epoch_btjd": t0,
        "duration_days": duration,
        "primary_depth": prim_depth,
        "primary_snr": prim_snr,
        "primary_n_cadences": prim_n,
        "secondary_depth": sec_depth,
        "secondary_snr": sec_snr,
        "secondary_detected": secondary_detected,
        "secondary_phase": sec_phase,
        "secondary_offset_from_half": sec_offset_from_half,
        "odd_depth": oe["odd_depth"],
        "even_depth": oe["even_depth"],
        "odd_even_ratio": oe["ratio"],
        "odd_even_z": oe["z"],
        "odd_even_significant": oe["significant"],
        "v_shape_metric": v_shape,
        "eb_subtype": subtype,
        "period_doubling_candidate": doubling_candidate,
        "suggested_period_days": suggested_period,
        "eb_evidence_count": n_evidence,
        "phase2_features_echo": {
            k: feature_dict.get(k)
            for k in ("secondary_depth", "secondary_snr", "secondary_detected",
                      "odd_even_diff", "odd_even_z", "v_shape_metric")
        },
    }
