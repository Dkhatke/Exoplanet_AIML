# src/recycler/__init__.py
# ============================================================
# Phase 5.5 - Recycler Router
#
# The triage classifier (Phase 4) assigns each target one of:
#   PLANET | EB | VARIABLE | NOISE  (or REVIEW on low confidence)
#
# This router dispatches the target to the matching specialized
# diagnostic module and returns a single normalized dict that the
# dashboard (Phase 6) can render without knowing which module ran.
#
# Public API:
#   route_to_recycler(...)   -> dict
# ============================================================

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
from lightkurve import LightCurve

from config import CLASS_LABELS, REVIEW_PROB_THRESHOLD
from src.utils.logger import get_logger

log = get_logger(__name__)

# The four concrete classes each map to a module. REVIEW (and any
# unexpected label) is handled explicitly.
_VALID_CLASSES = set(CLASS_LABELS) | {"REVIEW"}


def route_to_recycler(
    predicted_class: str,
    lc_detrended: LightCurve,
    lc_raw_norm: Optional[LightCurve],
    bls_output: Dict[str, Any],
    feature_dict: Dict[str, Any],
    stellar_params: Optional[Dict[str, Any]] = None,
    tic_id: Optional[int] = None,
    class_probabilities: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Route a classified target to its specialized recycler module.

    Parameters
    ----------
    predicted_class    : One of PLANET / EB / VARIABLE / NOISE / REVIEW.
    lc_detrended       : Detrended LightCurve (transit depth preserved).
    lc_raw_norm        : Normalized (un-detrended) LightCurve. Used by the
                         variable module so real stellar signal survives.
                         If None, lc_detrended is substituted.
    bls_output         : BLS result dict from the detector.
    feature_dict       : 25-feature dict from the feature extractor.
    stellar_params     : Dynamic stellar params (for the planet module).
    tic_id             : Optional TIC ID (for the variable Gaia cross-check).
    class_probabilities: Optional {class: prob} dict (for REVIEW context).

    Returns
    -------
    recycler_output : dict, always containing:
        module          : name of the module that ran
        predicted_class : the input class
        details         : module-specific diagnostics dict
    """
    predicted_class = str(predicted_class).upper()
    if lc_raw_norm is None:
        lc_raw_norm = lc_detrended

    if predicted_class not in _VALID_CLASSES:
        log.warning("Unknown predicted_class '%s' - routing to REVIEW", predicted_class)
        predicted_class = "REVIEW"

    base = {
        "predicted_class": predicted_class,
        "class_probabilities": class_probabilities or {},
    }

    try:
        if predicted_class == "PLANET":
            from src.recycler.planet_module import analyze_planet_candidate
            details = analyze_planet_candidate(lc_detrended, bls_output, stellar_params)
            module = "planet_module"

        elif predicted_class == "EB":
            from src.recycler.eb_module import analyze_eb
            details = analyze_eb(lc_detrended, bls_output, feature_dict)
            module = "eb_module"

        elif predicted_class == "VARIABLE":
            from src.recycler.variable_module import analyze_variable
            details = analyze_variable(lc_raw_norm, lc_detrended, tic_id=tic_id)
            module = "variable_module"

        elif predicted_class == "NOISE":
            from src.recycler.noise_module import analyze_noise
            details = analyze_noise(lc_detrended, feature_dict)
            module = "noise_module"

        else:  # REVIEW
            details = _review_diagnostics(bls_output, feature_dict, class_probabilities)
            module = "review"

    except Exception as exc:  # never let a module crash the pipeline
        log.error("Recycler module for %s failed: %s", predicted_class, exc)
        return {
            **base,
            "module": "error",
            "error": str(exc),
            "details": {},
        }

    return {**base, "module": module, "details": details}


def _review_diagnostics(
    bls_output: Dict[str, Any],
    feature_dict: Dict[str, Any],
    class_probabilities: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    """
    No specialized module runs for REVIEW. Instead we surface the raw
    signal strength and the class-probability spread so a human can make
    the final call.
    """
    probs = class_probabilities or {}
    max_prob = float(max(probs.values())) if probs else 0.0
    runner_up = 0.0
    if len(probs) >= 2:
        top2 = sorted(probs.values(), reverse=True)[:2]
        runner_up = float(top2[1])
    return {
        "reason": (
            f"Classifier confidence {max_prob:.2f} below review threshold "
            f"{REVIEW_PROB_THRESHOLD:.2f} - routed for human inspection."
        ),
        "bls_power": float(bls_output.get("bls_power", 0.0)) if bls_output else 0.0,
        "bls_period": float(
            bls_output.get("corrected_period") or bls_output.get("best_period", 0.0)
        ) if bls_output else 0.0,
        "transit_snr": float(feature_dict.get("snr", 0.0)),
        "max_probability": max_prob,
        "runner_up_probability": runner_up,
        "probability_margin": max_prob - runner_up,
        "probabilities": probs,
    }


__all__ = ["route_to_recycler"]

