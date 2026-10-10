# src/ingestion/cache_manager.py
# ============================================================
# Phase 1: Atomic Parquet-based cache manager for TESS light curves.
# Replaces the original pickle-based cache in src/fetcher.py.
# ============================================================

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lightkurve import LightCurve
from astropy.time import Time
import astropy.units as u

from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Schema ───────────────────────────────────────────────────
_SCHEMA = pa.schema([
    pa.field("time",               pa.float64()),
    pa.field("flux",               pa.float64()),
    pa.field("flux_err",           pa.float64()),
    pa.field("quality",            pa.int32()),
    pa.field("sector",             pa.int32()),
    pa.field("tic_id",             pa.int64()),
    pa.field("pipeline",           pa.string()),
    pa.field("lk_version",         pa.string()),
    pa.field("download_timestamp", pa.string()),
])

_REQUIRED_COLS = {"time", "flux", "flux_err", "sector", "tic_id"}


def _cache_path(cache_dir: str, tic_id: int, sector: int, pipeline: str) -> str:
    """Return the canonical path for a cached sector file."""
    fname = f"tic_{tic_id}_s{sector:04d}_{pipeline.lower()}.parquet"
    return os.path.join(cache_dir, fname)


def save_lightcurve(
    lc: LightCurve,
    tic_id: int,
    sector: int,
    pipeline: str,
    cache_dir: str,
) -> str:
    """
    Save a LightCurve to a Parquet file atomically.

    Writes to a temp file first, then renames to final path —
    so a partially written file is never left on disk.

    Returns
    -------
    str  — path of the saved Parquet file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    final_path = _cache_path(cache_dir, tic_id, sector, pipeline)

    # Build DataFrame
    flux_vals    = lc.flux.value if hasattr(lc.flux, "value") else np.array(lc.flux)
    flux_err_vals = lc.flux_err.value if hasattr(lc.flux_err, "value") else np.array(lc.flux_err)
    time_vals    = lc.time.value if hasattr(lc.time, "value") else np.array(lc.time)

    quality_arr = (
        lc.quality.value
        if hasattr(lc, "quality") and lc.quality is not None
        else np.zeros(len(lc), dtype=np.int32)
    )

    df = pd.DataFrame({
        "time":               time_vals.astype(np.float64),
        "flux":               flux_vals.astype(np.float64),
        "flux_err":           flux_err_vals.astype(np.float64),
        "quality":            quality_arr.astype(np.int32),
        "sector":             np.full(len(lc), sector, dtype=np.int32),
        "tic_id":             np.full(len(lc), tic_id, dtype=np.int64),
        "pipeline":           [pipeline] * len(lc),
        "lk_version":         [lk.__version__] * len(lc),
        "download_timestamp": [datetime.now(timezone.utc).isoformat()] * len(lc),
    })

    # Atomic write
    dir_ = os.path.dirname(final_path)
    with tempfile.NamedTemporaryFile(dir=dir_, suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
        pq.write_table(table, tmp_path, compression="snappy")
        os.replace(tmp_path, final_path)      # atomic rename
        log.info("Cached TIC %d sector %d → %s", tic_id, sector, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return final_path


def load_lightcurve(
    tic_id: int,
    sector: int,
    pipeline: str,
    cache_dir: str,
) -> LightCurve | None:
    """
    Load a cached LightCurve from Parquet.

    Returns None if:
    - File does not exist
    - Schema is invalid (missing required columns)
    - Lightkurve version mismatch (cache is stale)
    """
    path = _cache_path(cache_dir, tic_id, sector, pipeline)

    if not os.path.exists(path):
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        log.warning("TIC %d sector %d cache unreadable: %s — will re-download", tic_id, sector, e)
        return None

    # Validate schema
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        log.warning("TIC %d sector %d cache missing columns %s — invalidating", tic_id, sector, missing)
        os.remove(path)
        return None

    # Check lightkurve version
    cached_ver = df["lk_version"].iloc[0]
    if cached_ver != lk.__version__:
        log.warning(
            "TIC %d sector %d: cache was built with lightkurve %s (current %s) — invalidating",
            tic_id, sector, cached_ver, lk.__version__,
        )
        os.remove(path)
        return None

    log.info("Loaded TIC %d sector %d from cache (%d cadences)", tic_id, sector, len(df))

    lc = LightCurve(
        time=Time(df["time"].to_numpy(), format="btjd", scale="tdb"),
        flux=df["flux"].to_numpy() * u.dimensionless_unscaled,
        flux_err=df["flux_err"].to_numpy() * u.dimensionless_unscaled,
    )
    return lc


def list_cached_sectors(tic_id: int, cache_dir: str) -> list[tuple[int, str]]:
    """
    Return list of (sector, pipeline) tuples available in cache for this TIC ID.
    """
    prefix = f"tic_{tic_id}_s"
    results = []
    if not os.path.isdir(cache_dir):
        return results

    for fname in os.listdir(cache_dir):
        if not fname.startswith(prefix) or not fname.endswith(".parquet"):
            continue
        # filename: tic_{id}_s{sector:04d}_{pipeline}.parquet
        parts = fname.replace(".parquet", "").split("_")
        # parts: ["tic", "{id}", "s{sector}", "{pipeline}"]
        try:
            sector  = int(parts[2][1:])      # strip leading 's'
            pipeline = "_".join(parts[3:]).upper()
            results.append((sector, pipeline))
        except (IndexError, ValueError):
            continue

    return sorted(results)
