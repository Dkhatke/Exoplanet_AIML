# src/features/transit_metrics.py
# ============================================================
# Phase 2: Transit-specific diagnostic features.
# These discriminate PLANET from EB primarily.
# ============================================================

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from lightkurve import LightCurve
from src.utils.logger import get_logger

log = get_logger(__name__)


def _fold_lc(
    lc: LightCurve,
    period: float,
    t0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-fold a light curve. Returns (phase, flux) arrays centred at 0."""
    t = lc.time.value
    f = lc.flux.value if hasattr(lc.flux, "value") else np.array(lc.flux)
    phase = ((t - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    order = np.argsort(phase)
    return phase[order], f[order]


def compute_odd_even_depths(
    lc: LightCurve,
    period: float,
    t0: float,
    duration: float,
) -> Dict[str, float]:
    """
    Compute odd vs even transit depths.

    A planet produces nearly EQUAL odd/even depths (same body transiting).
    An eclipsing binary produces UNEQUAL depths (two different bodies).

    Returns
    -------
    dict: odd_depth, even_depth, odd_even_diff, odd_even_z
    """
    t = lc.time.value
    f = lc.flux.value if hasattr(lc.flux, "value") else np.array(lc.flux)

    half_dur = 0.5 * duration
    phase = ((t - t0) % period)
    phase[phase > period / 2] -= period
    in_transit = np.abs(phase) < half_dur

    # Assign cycle numbers
    cycle = np.floor((t - t0) / period).astype(int)

    odd_depths, even_depths = [], []

    for cyc in np.unique(cycle[in_transit]):
        mask = in_transit & (cycle == cyc)
        if np.sum(mask) < 3:
            continue
        # Depth = 1 - median_in_transit (assuming OOT flux ≈ 1.0)
        depth = 1.0 - float(np.median(f[mask]))
        if cyc % 2 == 0:
            even_depths.append(depth)
        else:
            odd_depths.append(depth)

    if not odd_depths or not even_depths:
        return {"odd_depth": 0.0, "even_depth": 0.0, "odd_even_diff": 0.0, "odd_even_z": 0.0}

    odd_d  = float(np.mean(odd_depths))
    even_d = float(np.mean(even_depths))
    diff   = abs(odd_d - even_d)

    # Z-score: diff / pooled uncertainty
    std_odd  = float(np.std(odd_depths))  if len(odd_depths)  > 1 else 1e-6
    std_even = float(np.std(even_depths)) if len(even_depths) > 1 else 1e-6
    pooled_err = np.sqrt(std_odd**2 + std_even**2) + 1e-9
    z_score = diff / pooled_err

    return {
        "odd_depth":      odd_d,
        "even_depth":     even_d,
        "odd_even_diff":  diff,
        "odd_even_z":     float(z_score),
    }


def compute_secondary_eclipse(
    lc: LightCurve,
    period: float,
    t0: float,
    duration: float,
    phase_offset: float = 0.5,
    snr_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Check for a secondary eclipse at phase ~0.5.

    A real planet rarely has a detectable secondary (reflected light is tiny).
    An EB almost always has a secondary eclipse.

    Returns
    -------
    dict: secondary_depth, secondary_snr, secondary_detected (bool as 0/1)
    """
    t = lc.time.value
    f = lc.flux.value if hasattr(lc.flux, "value") else np.array(lc.flux)

    # Epoch of secondary eclipse
    t0_sec = t0 + phase_offset * period
    phase = ((t - t0_sec) % period)
    phase[phase > period / 2] -= period
    in_secondary = np.abs(phase) < 0.5 * duration

    oot_mask = np.abs(((t - t0) % period - period / 2 * 0)) > 2 * duration
    # Simple OOT: not in primary and not in secondary
    phase_primary = ((t - t0) % period)
    phase_primary[phase_primary > period / 2] -= period
    oot = (np.abs(phase_primary) > duration) & (np.abs(phase) > duration)

    if np.sum(in_secondary) < 3 or np.sum(oot) < 10:
        return {"secondary_depth": 0.0, "secondary_snr": 0.0, "secondary_detected": 0}

    oot_median = float(np.median(f[oot]))
    oot_std    = float(np.std(f[oot])) + 1e-9
    sec_median = float(np.median(f[in_secondary]))
    sec_depth  = oot_median - sec_median
    sec_snr    = sec_depth / (oot_std / np.sqrt(np.sum(in_secondary)))

    detected = 1 if sec_snr > snr_threshold else 0
    return {
        "secondary_depth":    max(0.0, sec_depth),
        "secondary_snr":      float(sec_snr),
        "secondary_detected": detected,
    }


def compute_transit_shape(
    lc: LightCurve,
    period: float,
    t0: float,
    duration: float,
) -> Dict[str, float]:
    """
    Compute transit shape metrics to distinguish U-shaped (planet) vs V-shaped (EB) transits.

    Returns
    -------
    dict: ingress_fraction, v_shape_metric, mean_flux_in, mean_flux_out, flux_ratio
    """
    t = lc.time.value
    f = lc.flux.value if hasattr(lc.flux, "value") else np.array(lc.flux)

    phase = ((t - t0) % period)
    phase[phase > period / 2] -= period
    half_dur = 0.5 * duration

    in_transit = np.abs(phase) < half_dur
    oot        = np.abs(phase) > half_dur

    if np.sum(in_transit) < 5 or np.sum(oot) < 10:
        return {
            "ingress_fraction": 0.5,
            "v_shape_metric":   0.5,
            "mean_flux_in":     1.0,
            "mean_flux_out":    1.0,
            "flux_ratio":       1.0,
        }

    mean_in  = float(np.mean(f[in_transit]))
    mean_out = float(np.mean(f[oot]))
    flux_ratio = mean_in / (mean_out + 1e-9)

    # V-shape: compare flux at transit centre vs wings
    centre_mask = np.abs(phase) < half_dur * 0.3
    wing_mask   = (np.abs(phase) > half_dur * 0.5) & (np.abs(phase) < half_dur)

    if np.sum(centre_mask) >= 2 and np.sum(wing_mask) >= 2:
        centre_flux = float(np.mean(f[centre_mask]))
        wing_flux   = float(np.mean(f[wing_mask]))
        depth_total = mean_out - mean_in + 1e-9
        depth_wing  = mean_out - wing_flux
        # V-shape metric: if wings are almost as deep as centre → V-shape (EB)
        v_shape = depth_wing / depth_total
        # Ingress fraction: rough estimate from wing width
        ingress_fraction = 0.5 * half_dur / (duration + 1e-9)
    else:
        v_shape = 0.5
        ingress_fraction = 0.25

    return {
        "ingress_fraction": float(np.clip(ingress_fraction, 0, 1)),
        "v_shape_metric":   float(np.clip(v_shape, 0, 1)),
        "mean_flux_in":     mean_in,
        "mean_flux_out":    mean_out,
        "flux_ratio":       float(flux_ratio),
    }
