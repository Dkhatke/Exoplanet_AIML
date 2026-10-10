# src/ingestion/fetcher.py
# ============================================================
# Phase 1: Multi-sector TESS light curve downloader.
# Replaces single-sector pickle-based src/fetcher.py.
#
# Key improvements over original:
#   - Downloads ALL available sectors (up to MAX_SECTORS)
#   - Prefers SPOC (2-min); falls back to QLP (30-min)
#   - Caches each sector individually as Parquet (not pickle)
#   - Imposes per-download timeout
#   - Returns list of LightCurve objects (one per sector)
# ============================================================

from __future__ import annotations

import signal
from contextlib import contextmanager
from typing import List, Optional

import lightkurve as lk
from lightkurve import LightCurve

from config import (
    PREFERRED_AUTHOR, FALLBACK_AUTHOR,
    MAX_SECTORS, MAX_SECONDS_PER_TARGET,
    CACHE_DIR,
)
from src.ingestion.cache_manager import (
    save_lightcurve, load_lightcurve, list_cached_sectors,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Timeout context manager ───────────────────────────────────

class DownloadTimeout(Exception):
    pass


@contextmanager
def _timeout(seconds: int):
    """Raise DownloadTimeout if block takes longer than `seconds`."""
    def _handler(signum, frame):
        raise DownloadTimeout(f"Download timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ── Public API ───────────────────────────────────────────────

def search_tess_sectors(
    tic_id: int,
    author: str = PREFERRED_AUTHOR,
) -> lk.SearchResult:
    """
    Search MAST for all available TESS light curves for a TIC ID.

    Parameters
    ----------
    tic_id  : int   — TIC identifier.
    author  : str   — Pipeline author; "SPOC" or "QLP".

    Returns
    -------
    lightkurve.SearchResult — may be empty if no data found.
    """
    target = f"TIC {tic_id}"
    log.info("Searching MAST for %s (author=%s)", target, author)
    result = lk.search_lightcurve(target, mission="TESS", author=author)
    log.info("Found %d sector(s) for %s [%s]", len(result), target, author)
    return result


def download_sectors(
    tic_id: int,
    max_sectors: int = MAX_SECTORS,
    preferred_author: str = PREFERRED_AUTHOR,
    fallback_author: str = FALLBACK_AUTHOR,
    cache_dir: str = CACHE_DIR,
    timeout_seconds: int = 30,
) -> List[LightCurve]:
    """
    Download up to `max_sectors` TESS light curves for a TIC ID.

    Strategy:
    1. Check Parquet cache for already-downloaded sectors.
    2. Search MAST for SPOC data; fall back to QLP if none found.
    3. Download missing sectors (skip already cached).
    4. Enforce per-sector download timeout.

    Parameters
    ----------
    tic_id            : TIC identifier (integer).
    max_sectors       : Maximum number of sectors to download.
    preferred_author  : "SPOC" (default).
    fallback_author   : "QLP" (fallback if SPOC unavailable).
    cache_dir         : Directory to read/write Parquet cache files.
    timeout_seconds   : Per-sector download timeout in seconds.

    Returns
    -------
    List[LightCurve]  — One LightCurve per sector, sorted by sector number.
                        Returns empty list if no data found at all.
    """
    # ── 1. Check what is already cached ──────────────────────
    cached = list_cached_sectors(tic_id, cache_dir)
    cached_sectors = {s for s, _ in cached}
    lc_list: List[LightCurve] = []

    for sector, pipeline in cached:
        lc = load_lightcurve(tic_id, sector, pipeline, cache_dir)
        if lc is not None:
            lc_list.append(lc)

    if len(lc_list) >= max_sectors:
        log.info("TIC %d: all %d sector(s) loaded from cache", tic_id, len(lc_list))
        return lc_list

    # ── 2. Search MAST ────────────────────────────────────────
    author = preferred_author
    search_result = search_tess_sectors(tic_id, author=author)

    if len(search_result) == 0:
        log.warning("TIC %d: no %s data found — trying %s", tic_id, author, fallback_author)
        author = fallback_author
        search_result = search_tess_sectors(tic_id, author=author)

    if len(search_result) == 0:
        raise ValueError(
            f"TIC {tic_id}: no TESS light curves found in either "
            f"{preferred_author} or {fallback_author}."
        )

    # ── 3. Download missing sectors ───────────────────────────
    remaining_slots = max_sectors - len(lc_list)
    downloaded = 0

    for i, item in enumerate(search_result):
        if downloaded >= remaining_slots:
            break

        # Extract sector number from search result
        try:
            sector = int(item.mission[0].split()[-1])   # e.g. "TESS Sector 1" → 1
        except (IndexError, ValueError, AttributeError):
            sector = i + 1   # Fallback numbering

        if sector in cached_sectors:
            log.debug("TIC %d sector %d already cached — skipping", tic_id, sector)
            continue

        log.info("TIC %d: downloading sector %d (%s)…", tic_id, sector, author)
        try:
            with _timeout(timeout_seconds):
                raw_lc = item.download()

            if raw_lc is None:
                log.warning("TIC %d sector %d: download returned None", tic_id, sector)
                continue

            raw_lc = raw_lc.remove_nans()

            # Save to Parquet cache
            try:
                save_lightcurve(raw_lc, tic_id, sector, author, cache_dir)
            except Exception as cache_err:
                log.warning("TIC %d sector %d: cache save failed: %s", tic_id, sector, cache_err)

            lc_list.append(raw_lc)
            downloaded += 1

        except DownloadTimeout:
            log.warning(
                "TIC %d sector %d: download timed out after %ds — skipping",
                tic_id, sector, timeout_seconds,
            )
            continue
        except Exception as exc:
            log.warning("TIC %d sector %d: download error: %s", tic_id, sector, exc)
            continue

    if not lc_list:
        raise ValueError(
            f"TIC {tic_id}: download attempted but no sectors retrieved. "
            "Check MAST availability."
        )

    log.info("TIC %d: %d sector(s) ready", tic_id, len(lc_list))
    return lc_list


# ── Legacy shim (keeps old app.py import working) ────────────

def fetch_lightcurve(star_name: str) -> LightCurve:
    """
    Legacy single-sector fetcher shim.
    Used by app.py until Phase 6 Streamlit overhaul.
    Resolves star name → TIC ID and returns sector 0 light curve.
    """
    log.warning(
        "fetch_lightcurve('%s') is a legacy shim. "
        "Use download_sectors(tic_id) in new code.", star_name
    )
    search = lk.search_lightcurve(star_name, mission="TESS")
    if len(search) == 0:
        raise ValueError(f"No TESS data found for '{star_name}'")

    lc = search[0].download().remove_nans()
    return lc
