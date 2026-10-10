# src/recycler/planet_module.py
# ============================================================
# Phase 5.1 - Planet Characterization Module
#
# Given a detrended light curve, its BLS detection, and the
# DYNAMIC stellar parameters resolved from the TIC (never the
# hardcoded HD 209458 constants), compute the physical
# properties of a transit candidate.
#
# IMPORTANT: every output is labeled a CANDIDATE. This module
# does not confirm a planet - it only characterizes a signal
# that the triage classifier routed here as "PLANET".
#
# Physics (all standard):
#   depth      = (Rp / R*)^2                      -> Rp/R*
#   Rp         = (Rp/R*) * R*                     -> Earth radii
#   a          = (G M* P^2 / 4 pi^2)^(1/3)        (Kepler's 3rd law)
#   T_eq       = T* (1-A)^0.25 sqrt(R* / 2a)      (equilibrium temp)
#   T_dur,pred = (P / pi) arcsin((R*+Rp)/a)       (central transit)
# ============================================================

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import astropy.constants as const
from lightkurve import LightCurve

from config import (
    DEFAULT_STAR_RADIUS_SOLAR,
    DEFAULT_STAR_MASS_SOLAR,
    DEFAULT_STAR_TEFF,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# Assumed Bond albedo for the equilibrium-temperature estimate.
# 0.3 is the canonical Earth-like / typical hot-Jupiter assumption
# used throughout the exoplanet literature (e.g. Seager 2010).
DEFAULT_ALBEDO = 0.3

# A central-transit duration prediction can only be an upper bound
# (impact parameter b>=0 shortens the real transit). We accept a
# measured/predicted ratio in this window as geometrically plausible.
DURATION_RATIO_MIN = 0.2
DURATION_RATIO_MAX = 1.3


def _flux_time(lc: LightCurve):
    """Return (time, flux) as plain numpy arrays from a LightCurve."""
    t = lc.time.value if hasattr(lc.time, "value") else np.asarray(lc.time)
    f = lc.flux.value if hasattr(lc.flux, "value") else np.asarray(lc.flux)
    return t, f


def _resolve_stellar(stellar_params: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Pull radius/mass/teff out of the dynamic stellar_params dict,
    falling back to generic Solar defaults (with a warning) only when
    a value is missing or non-physical. This satisfies AC-5.1: the
    primary source is ALWAYS the passed-in dynamic stellar_params.
    """
    sp = stellar_params or {}
    radius = sp.get("radius_solar")
    mass = sp.get("mass_solar")
    teff = sp.get("teff_k")

    used_defaults = []
    if radius is None or not np.isfinite(radius) or radius <= 0:
        radius = DEFAULT_STAR_RADIUS_SOLAR
        used_defaults.append("radius_solar")
    if mass is None or not np.isfinite(mass) or mass <= 0:
        mass = DEFAULT_STAR_MASS_SOLAR
        used_defaults.append("mass_solar")
    if teff is None or not np.isfinite(teff) or teff <= 0:
        teff = DEFAULT_STAR_TEFF
        used_defaults.append("teff_k")

    if used_defaults:
        log.warning(
            "planet_module: missing/invalid stellar fields %s - using Solar "
            "defaults for those fields", used_defaults,
        )

    return {
        "radius_solar": float(radius),
        "mass_solar": float(mass),
        "teff_k": float(teff),
        "source": sp.get("source", "default"),
        "used_defaults": used_defaults,
    }


def _measure_depth(lc: LightCurve, period: float, t0: float, duration: float):
    """
    Re-measure the transit depth directly from the detrended curve at the
    BLS ephemeris. Returns (depth, n_in, n_oot, oot_std).
    """
    t, f = _flux_time(lc)
    phase = ((t - t0) % period)
    phase[phase > period / 2] -= period
    half = 0.5 * duration
    in_t = np.abs(phase) < half
    oot = ~in_t

    n_in = int(np.sum(in_t))
    n_oot = int(np.sum(oot))
    if n_in < 3 or n_oot < 5:
        return 0.0, n_in, n_oot, 0.0

    depth = float(np.median(f[oot]) - np.median(f[in_t]))
    depth = max(0.0, depth)
    oot_std = float(np.std(f[oot]))
    return depth, n_in, n_oot, oot_std



def analyze_planet_candidate(
    lc_detrended: LightCurve,
    bls_output: Dict[str, Any],
    stellar_params: Optional[Dict[str, Any]] = None,
    albedo: float = DEFAULT_ALBEDO,
) -> Dict[str, Any]:
    """
    Characterize a transit candidate routed to the PLANET class.

    Parameters
    ----------
    lc_detrended : Detrended LightCurve (transit depth preserved).
    bls_output   : BLS result dict from src.detection.detector.run_bls.
    stellar_params : DYNAMIC stellar params (radius_solar, mass_solar, teff_k)
                     from src.ingestion.stellar_resolver. NOT hardcoded.
    albedo       : Assumed Bond albedo for T_eq (default 0.3).

    Returns
    -------
    planet_params : dict - all values describe a CANDIDATE, not a
                    confirmed planet.
    """
    star = _resolve_stellar(stellar_params)
    notes = []

    period = float(bls_output.get("corrected_period") or bls_output.get("best_period", 1.0))
    t0 = float(bls_output.get("t0", 0.0))
    duration = float(bls_output.get("duration", 0.1))
    if period <= 0:
        period = 1.0
    if duration <= 0:
        duration = 0.01

    # -- Transit depth measured at the BLS ephemeris ---------------
    depth, n_in, n_oot, oot_std = _measure_depth(lc_detrended, period, t0, duration)
    if depth <= 0 or n_in < 3:
        notes.append("No measurable transit depth at the detected ephemeris.")

    # -- Planet-to-star radius ratio and radius in Earth radii -----
    rp_over_rs = float(np.sqrt(depth))
    r_star_m = star["radius_solar"] * const.R_sun.value
    rp_m = rp_over_rs * r_star_m
    rp_earth = float(rp_m / const.R_earth.value)

    # -- Semi-major axis via Kepler's 3rd law ----------------------
    p_sec = period * 86400.0
    m_star_kg = star["mass_solar"] * const.M_sun.value
    a_m = (const.G.value * m_star_kg * p_sec ** 2 / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
    a_au = float(a_m / const.au.value)

    # -- Equilibrium temperature -----------------------------------
    if a_m > 0:
        t_eq = float(
            star["teff_k"]
            * (1.0 - albedo) ** 0.25
            * np.sqrt(r_star_m / (2.0 * a_m))
        )
    else:
        t_eq = float("nan")

    # -- Transit-duration consistency (central-transit upper bound)
    geom_ratio = (r_star_m + rp_m) / a_m if a_m > 0 else 1.0
    if geom_ratio >= 1.0:
        # Degenerate (grazing/overflow) - predicted duration saturates.
        t_dur_pred_days = period
        notes.append("(R*+Rp)/a >= 1 - duration prediction degenerate.")
    else:
        t_dur_pred_days = float((period / np.pi) * np.arcsin(np.clip(geom_ratio, -1.0, 1.0)))

    if t_dur_pred_days > 0:
        duration_ratio = float(duration / t_dur_pred_days)
    else:
        duration_ratio = float("nan")

    duration_consistent = bool(
        np.isfinite(duration_ratio)
        and DURATION_RATIO_MIN <= duration_ratio <= DURATION_RATIO_MAX
    )
    if not duration_consistent and np.isfinite(duration_ratio):
        notes.append(
            f"Measured/predicted duration ratio {duration_ratio:.2f} outside "
            f"[{DURATION_RATIO_MIN}, {DURATION_RATIO_MAX}] - check ephemeris or impact parameter."
        )

    snr = float(depth / (oot_std / np.sqrt(n_in))) if (oot_std > 0 and n_in > 0) else 0.0

    log.info(
        "planet_module: P=%.4fd depth=%.4f Rp=%.2f Rearth a=%.4f AU Teq=%.0f K "
        "(stellar source=%s)",
        period, depth, rp_earth, a_au, t_eq, star["source"],
    )

    return {
        "classification": "PLANET_CANDIDATE",
        "period_days": period,
        "epoch_btjd": t0,
        "measured_duration_days": duration,
        "transit_depth": depth,
        "transit_snr": snr,
        "n_in_transit": n_in,
        "n_out_transit": n_oot,
        "rp_over_rs": rp_over_rs,
        "rp_earth_radii": rp_earth,
        "semi_major_axis_au": a_au,
        "eq_temperature_k": t_eq,
        "albedo_assumed": albedo,
        "predicted_duration_days": t_dur_pred_days,
        "duration_ratio": duration_ratio,
        "duration_consistent": duration_consistent,
        "stellar_params_used": star,
        "notes": notes,
    }
