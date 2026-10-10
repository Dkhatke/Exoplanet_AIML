# src/ingestion/stellar_resolver.py
# ============================================================
# Phase 1: Dynamically resolve stellar parameters from the
# TESS Input Catalog (TIC) via astroquery.mast.
#
# Replaces the hardcoded HD 209458 constants in old config.py.
# Always returns a valid dict with Solar defaults on failure.
# ============================================================

from __future__ import annotations

from typing import Dict, Any

from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Default (Solar) fallback values ──────────────────────────
_SOLAR_DEFAULTS: Dict[str, float] = {
    "radius_solar": 1.0,    # R_sun
    "mass_solar":   1.0,    # M_sun
    "teff_k":       5778.0, # K
    "logg":         4.44,   # log(g) cm/s^2
}

_TIC_COLUMN_MAP = {
    "radius_solar": "rad",    # TIC column for stellar radius (R_sun)
    "mass_solar":   "mass",   # TIC column for stellar mass (M_sun)
    "teff_k":       "Teff",   # TIC column for effective temperature (K)
    "logg":         "logg",   # TIC column for surface gravity
}


def resolve_stellar_params(tic_id: int) -> Dict[str, Any]:
    """
    Query the TESS Input Catalog (TIC) for stellar parameters.

    Parameters
    ----------
    tic_id : int — TIC identifier.

    Returns
    -------
    dict with keys:
        radius_solar  : float  — Stellar radius in solar radii
        mass_solar    : float  — Stellar mass in solar masses
        teff_k        : float  — Effective temperature in Kelvin
        logg          : float  — log10(surface gravity) in cm/s^2
        tic_id        : int    — TIC ID (for reference)
        source        : str    — "TIC" or "Solar_default"
    """
    try:
        from astroquery.mast import Catalogs
        result = Catalogs.query_criteria(catalog="Tic", ID=tic_id)

        if result is None or len(result) == 0:
            log.warning("TIC %d: no result from TIC query — using Solar defaults", tic_id)
            return _build_result(tic_id, _SOLAR_DEFAULTS.copy(), source="Solar_default")

        row = result[0]
        params: Dict[str, float] = {}
        missing_cols = []

        for param_key, tic_col in _TIC_COLUMN_MAP.items():
            try:
                val = float(row[tic_col])
                if val != val or val <= 0:    # NaN or non-positive
                    raise ValueError(f"Invalid value {val}")
                params[param_key] = val
            except (KeyError, TypeError, ValueError):
                params[param_key] = _SOLAR_DEFAULTS[param_key]
                missing_cols.append(tic_col)

        if missing_cols:
            log.warning(
                "TIC %d: missing/invalid TIC columns %s — Solar defaults used for those fields",
                tic_id, missing_cols,
            )
            source = "TIC_partial"
        else:
            source = "TIC"
            log.info(
                "TIC %d: R*=%.2f Rsun, M*=%.2f Msun, Teff=%dK, logg=%.2f",
                tic_id,
                params["radius_solar"],
                params["mass_solar"],
                int(params["teff_k"]),
                params["logg"],
            )

        return _build_result(tic_id, params, source=source)

    except ImportError:
        log.error("astroquery not installed — cannot query TIC. Using Solar defaults.")
        return _build_result(tic_id, _SOLAR_DEFAULTS.copy(), source="Solar_default")

    except Exception as exc:
        log.warning(
            "TIC %d: TIC query failed (%s) — using Solar defaults", tic_id, exc
        )
        return _build_result(tic_id, _SOLAR_DEFAULTS.copy(), source="Solar_default")


def get_stellar_params_or_default(tic_id: int) -> Dict[str, Any]:
    """
    Safe wrapper around resolve_stellar_params.
    Guaranteed to return a valid dict; never raises.
    """
    try:
        return resolve_stellar_params(tic_id)
    except Exception as exc:
        log.error("TIC %d: unexpected error in resolve_stellar_params: %s", tic_id, exc)
        return _build_result(tic_id, _SOLAR_DEFAULTS.copy(), source="Solar_default")


def _build_result(
    tic_id: int,
    params: Dict[str, float],
    source: str,
) -> Dict[str, Any]:
    """Attach tic_id and source to a stellar params dict."""
    return {
        **params,
        "tic_id": tic_id,
        "source": source,
    }
