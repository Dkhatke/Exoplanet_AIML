# src/ingestion/stitcher.py
# ============================================================
# Phase 1: Sector normalisation and stitching module.
# Each sector is normalised to median=1.0 before stitching
# to eliminate sector-to-sector flux level offsets.
# ============================================================

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import lightkurve as lk
from lightkurve import LightCurve, LightCurveCollection

from src.utils.logger import get_logger

log = get_logger(__name__)


def normalize_sector(lc: LightCurve, remove_bad_quality: bool = True) -> LightCurve:
    """
    Prepare a single TESS sector for stitching.

    Steps:
    1. Remove cadences flagged with non-zero quality bits (instrument artifacts).
    2. Remove NaN values.
    3. Normalise flux to median = 1.0.

    Parameters
    ----------
    lc                 : Raw LightCurve for one sector.
    remove_bad_quality : If True, removes cadences where quality != 0.

    Returns
    -------
    LightCurve — cleaned and normalised.
    """
    # Remove quality-flagged cadences if quality column exists
    if remove_bad_quality and hasattr(lc, "quality") and lc.quality is not None:
        try:
            quality_vals = (
                lc.quality.value
                if hasattr(lc.quality, "value")
                else np.array(lc.quality)
            )
            good_mask = quality_vals == 0
            n_removed = int(np.sum(~good_mask))
            if n_removed > 0:
                lc = lc[good_mask]
                log.debug("Removed %d quality-flagged cadences", n_removed)
        except Exception as e:
            log.debug("Could not apply quality filter: %s", e)

    # Remove NaNs
    lc = lc.remove_nans()

    # Normalise to median = 1.0
    lc = lc.normalize()

    return lc


def stitch_sectors(
    lc_list: List[LightCurve],
    normalize: bool = True,
) -> LightCurve:
    """
    Stitch multiple TESS sectors into a single continuous light curve.

    Each sector is normalised independently (median=1.0) before stitching
    to remove inter-sector flux offsets. The result is sorted by time.

    Parameters
    ----------
    lc_list   : List of LightCurve objects (one per sector).
    normalize : If True, apply normalize_sector to each before stitching.

    Returns
    -------
    LightCurve — stitched, time-sorted, and normalised.

    Raises
    ------
    ValueError : If lc_list is empty.
    """
    if not lc_list:
        raise ValueError("stitch_sectors received an empty list — nothing to stitch.")

    if len(lc_list) == 1:
        log.info("Only one sector — no stitching needed")
        lc = lc_list[0]
        return normalize_sector(lc) if normalize else lc

    prepared = []
    for i, lc in enumerate(lc_list):
        n_before = len(lc)
        lc_clean = normalize_sector(lc) if normalize else lc
        log.info(
            "Sector %d/%d: %d → %d cadences after cleaning",
            i + 1, len(lc_list), n_before, len(lc_clean),
        )
        prepared.append(lc_clean)

    # Stitch using lightkurve LightCurveCollection
    collection = LightCurveCollection(prepared)
    stitched = collection.stitch()

    # Sort by time (guarantee monotonic time axis)
    order = np.argsort(stitched.time.value)
    stitched = stitched[order]

    # Sanity: check for large gaps (> 14 days between adjacent cadences)
    time_arr = stitched.time.value
    gaps = np.diff(time_arr)
    large_gap_idx = np.where(gaps > 14.0)[0]
    if len(large_gap_idx) > 0:
        largest = gaps[large_gap_idx].max()
        log.info(
            "Stitched light curve has %d gap(s) > 14 days; largest = %.1f days "
            "(expected for multi-sector TESS data).",
            len(large_gap_idx), largest,
        )

    log.info(
        "Stitched %d sectors → %d total cadences | baseline = %.1f days",
        len(lc_list),
        len(stitched),
        time_arr[-1] - time_arr[0],
    )
    return stitched


def get_baseline_days(lc: LightCurve) -> float:
    """Return the total time baseline of a light curve in days."""
    t = lc.time.value
    return float(t[-1] - t[0])


def split_by_gap(
    lc: LightCurve,
    gap_threshold_days: float = 5.0,
) -> List[LightCurve]:
    """
    Split a stitched light curve back into contiguous segments by gap size.

    Useful for diagnostics — e.g., inspecting individual sectors after stitching.

    Parameters
    ----------
    lc                  : Stitched LightCurve.
    gap_threshold_days  : Gaps larger than this split the curve.

    Returns
    -------
    List[LightCurve] — one per contiguous segment.
    """
    time = lc.time.value
    gaps = np.diff(time)
    split_at = np.where(gaps > gap_threshold_days)[0] + 1   # indices after gap

    segments = []
    prev = 0
    for idx in split_at:
        segments.append(lc[prev:idx])
        prev = idx
    segments.append(lc[prev:])   # last segment

    log.debug("Split into %d segment(s) at %.1f-day gap threshold", len(segments), gap_threshold_days)
    return segments
