#!/usr/bin/env python3
# scripts/build_dataset.py
# ============================================================
# Phase 3: Build the 100-star ground-truth dataset.
#
# Usage:
#   python scripts/build_dataset.py --pilot          # 5-star pilot only
#   python scripts/build_dataset.py --full           # All 100 stars
#   python scripts/build_dataset.py --full --resume  # Skip already-done TIC IDs
#
# RESOURCE LIMITS (enforced):
#   - Max 2 parallel workers
#   - Max 120s per target
#   - Stop if >10 targets fail
# ============================================================

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CATALOG_DIR, CACHE_DIR, DATASET_PATH,
    TARGETS_MANIFEST, FEATURE_NAMES,
    MAX_PARALLEL_WORKERS, MAX_SECONDS_PER_TARGET,
    PILOT_SIZE, TOI_API_URL, VILLANOVA_EB_URL,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Catalog download functions ────────────────────────────────

def download_toi_catalog(save_path: str) -> pd.DataFrame:
    """Download NASA Exoplanet Archive TOI catalog (CP+KP confirmed planets)."""
    log.info("Downloading NASA TOI catalog via astroquery…")
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
        # Correct column names: tid=TIC ID, pl_trandep=transit depth ppm, pl_orbper=period
        table = NasaExoplanetArchive.query_criteria(
            table="toi",
            where="tfopwg_disp in ('CP','KP')",
            select="tid,toi,tfopwg_disp,pl_trandep,pl_orbper,pl_trandurh,st_tmag",
        )
        df = table.to_pandas()
        # Normalise: rename tid → tic_id for downstream compatibility
        if "tid" in df.columns:
            df = df.rename(columns={"tid": "tic_id"})
        df.to_csv(save_path, index=False)
        log.info("TOI catalog: %d records saved to %s", len(df), save_path)
        return df
    except Exception as e:
        log.warning("astroquery TOI failed (%s) — trying direct HTTP TAP", e)
        import requests
        url = (
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
            "?query=select+tid,toi,tfopwg_disp,pl_trandep,pl_orbper"
            "+from+toi+where+tfopwg_disp+in+(%27CP%27,%27KP%27)&format=csv"
        )
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        df = pd.read_csv(save_path)
        if "tid" in df.columns:
            df = df.rename(columns={"tid": "tic_id"})
        log.info("TOI catalog (HTTP): %d records saved to %s", len(df), save_path)
        return df



def download_villanova_eb_catalog(save_path: str) -> pd.DataFrame:
    """
    Download Prsa et al. 2022 TESS Eclipsing Binary catalog via VizieR.
    Catalog: J/ApJS/258/16  (replaces the defunct Villanova HTTP URLs)
    Columns returned: TIC, Per, Morph, Tmag
    """
    log.info("Downloading TESS-EB catalog from VizieR (J/ApJS/258/16)…")
    try:
        from astroquery.vizier import Vizier
        v = Vizier(columns=["TIC", "Per", "Morph", "Tmag"], row_limit=10000)
        result = v.query_constraints(catalog="J/ApJS/258/16")
        if not result or len(result) == 0:
            raise ValueError("VizieR returned no tables for J/ApJS/258/16")
        df = result[0].to_pandas()
        # Normalise column names for downstream compatibility
        rename = {"TIC": "tic_id", "Per": "period", "Morph": "morph", "Tmag": "tmag"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df.to_csv(save_path, index=False)
        log.info("TESS-EB catalog (VizieR): %d records saved to %s", len(df), save_path)
        return df
    except Exception as e:
        log.error("VizieR EB download failed: %s — returning empty DataFrame", e)
        return pd.DataFrame()



def query_gaia_variables(save_path: str, max_records: int = 5000) -> pd.DataFrame:
    """Query Gaia DR3 variable star catalog via astroquery."""
    from astroquery.gaia import Gaia
    log.info("Querying Gaia DR3 variable stars…")
    query = f"""
    SELECT TOP {max_records}
        v.source_id, v.best_class_name, v.best_class_score,
        g.ra, g.dec, g.phot_g_mean_mag
    FROM gaiadr3.vari_classifier_result AS v
    JOIN gaiadr3.gaia_source AS g ON v.source_id = g.source_id
    WHERE v.best_class_name IN ('DSCT_SXPHE', 'RR', 'CEP', 'ROT', 'LPV')
    AND g.phot_g_mean_mag BETWEEN 8 AND 14
    ORDER BY v.best_class_score DESC
    """
    try:
        job = Gaia.launch_job(query)
        result = job.get_results()
        df = result.to_pandas()
        df.to_csv(save_path, index=False)
        log.info("Gaia DR3 variables: %d records saved to %s", len(df), save_path)
        return df
    except Exception as e:
        log.error("Gaia query failed: %s", e)
        return pd.DataFrame()


# ── Manifest-building helpers ─────────────────────────────────

def select_planet_targets(toi_df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Select 25 PLANET targets from TOI catalog."""
    # Column name variations across TOI versions
    tic_col = next((c for c in toi_df.columns if "tic" in c.lower()), None)
    snr_col = next((c for c in toi_df.columns if "snr" in c.lower()), None)
    disp_col = next((c for c in toi_df.columns if "tfopwg" in c.lower() or "disp" in c.lower()), None)

    if not all([tic_col, disp_col]):
        log.error("TOI catalog missing required columns. Columns: %s", list(toi_df.columns))
        return pd.DataFrame()

    mask = toi_df[disp_col].isin(["CP", "KP"])
    filtered = toi_df[mask].copy()

    if snr_col and snr_col in filtered.columns:
        filtered = filtered.dropna(subset=[snr_col])
        filtered = filtered[filtered[snr_col].astype(float) > 7]
        filtered = filtered.sort_values(snr_col, ascending=False)

    # Deduplicate by TIC ID
    filtered = filtered.drop_duplicates(subset=[tic_col])
    selected = filtered.head(min(n * 2, len(filtered)))

    rows = []
    for _, row in selected.iterrows():
        tic_id = int(row[tic_col])
        rows.append({
            "tic_id": tic_id,
            "class_label": "PLANET",
            "source_catalog": "NASA_TOI",
            "catalog_period": float(row.get("pl_orbper", 0.0)) if "pl_orbper" in row else 0.0,
            "catalog_source_id": str(row.get("toi", tic_id)),
            "notes": f"tfopwg_disp={row.get(disp_col, 'unknown')}",
        })
        if len(rows) >= n:
            break

    return pd.DataFrame(rows)


def select_eb_targets(eb_df: pd.DataFrame, n: int = 25, exclude_tics: set = None) -> pd.DataFrame:
    """
    Select 25 EB targets from the VizieR TESS-EB catalog (Prsa+2022).
    Expected columns after normalisation: tic_id, period, morph, tmag.
    """
    exclude_tics = exclude_tics or set()

    if eb_df.empty:
        log.warning("EB catalog is empty — no EB targets can be selected")
        return pd.DataFrame()

    # Column name matching (VizieR normalised names)
    tic_col    = next((c for c in eb_df.columns if c.lower() in ("tic_id", "tic")), None)
    period_col = next((c for c in eb_df.columns if c.lower() in ("period", "per")), None)
    morph_col  = next((c for c in eb_df.columns if c.lower() in ("morph", "morphology")), None)

    if not tic_col:
        log.error("EB catalog missing TIC column. Columns: %s", list(eb_df.columns))
        return pd.DataFrame()

    filtered = eb_df.copy()
    filtered[tic_col] = pd.to_numeric(filtered[tic_col], errors="coerce")
    filtered = filtered.dropna(subset=[tic_col])

    # Prefer detached EBs (morph < 0.5) — cleaner secondary eclipse
    if morph_col and morph_col in filtered.columns:
        filtered[morph_col] = pd.to_numeric(filtered[morph_col], errors="coerce")
        detached = filtered[filtered[morph_col] < 0.5]
        filtered = detached if len(detached) >= n else filtered

    # Period filter 0.5–15 days
    if period_col and period_col in filtered.columns:
        filtered[period_col] = pd.to_numeric(filtered[period_col], errors="coerce")
        filtered = filtered[filtered[period_col].between(0.5, 15.0)]

    # Exclude TIC IDs already selected in another class
    filtered = filtered[~filtered[tic_col].astype(int).isin(exclude_tics)]
    filtered = filtered.drop_duplicates(subset=[tic_col])

    rows = []
    for _, row in filtered.iterrows():
        tic_id = int(row[tic_col])
        period = float(row[period_col]) if period_col and pd.notna(row.get(period_col)) else 0.0
        morph  = float(row[morph_col])  if morph_col  and pd.notna(row.get(morph_col))  else -1.0
        rows.append({
            "tic_id":            tic_id,
            "class_label":       "EB",
            "source_catalog":    "Villanova_EB",
            "catalog_period":    period,
            "catalog_source_id": str(tic_id),
            "notes":             f"morph={morph:.3f}" if morph >= 0 else "morph=unknown",
        })
        if len(rows) >= n:
            break

    log.info("Selected %d EB targets", len(rows))
    return pd.DataFrame(rows)



def select_variable_targets(gaia_df: pd.DataFrame, n: int = 25, exclude_tics: set = None) -> pd.DataFrame:
    """
    Select 25 VARIABLE targets by cross-matching Gaia DR3 variables to TIC.
    Requires internet for TIC cross-match.
    """
    exclude_tics = exclude_tics or set()

    if gaia_df.empty:
        log.warning("Gaia catalog is empty — cannot select variable targets")
        return pd.DataFrame()

    from astroquery.mast import Catalogs
    rows = []
    log.info("Cross-matching %d Gaia variables to TIC…", len(gaia_df))

    for _, row in gaia_df.iterrows():
        if len(rows) >= n:
            break
        try:
            ra, dec = float(row["ra"]), float(row["dec"])
            result = Catalogs.query_region(
                f"{ra} {dec}", radius="0.00056 deg",   # 2 arcsec
                catalog="TIC"
            )
            if result is None or len(result) == 0:
                continue
            tic_id = int(result["ID"][0])
            if tic_id in exclude_tics:
                continue
            rows.append({
                "tic_id": tic_id,
                "class_label": "VARIABLE",
                "source_catalog": "Gaia_DR3",
                "catalog_period": 0.0,
                "catalog_source_id": str(int(row["source_id"])),
                "notes": f"best_class={row.get('best_class_name', 'unknown')}",
            })
            exclude_tics.add(tic_id)
        except Exception as e:
            log.debug("Gaia cross-match failed for source %s: %s", row.get("source_id"), e)
            continue

    log.info("Selected %d VARIABLE targets", len(rows))
    return pd.DataFrame(rows)


def select_noise_targets(n: int = 25, exclude_tics: set = None) -> pd.DataFrame:
    """
    Select 25 NOISE targets from TIC (G/K stars, quiet: flux MAD < 500 ppm).
    This requires downloading light curves — expensive; done conservatively.
    """
    exclude_tics = exclude_tics or set()
    from astroquery.mast import Catalogs

    log.info("Querying TIC for quiet G/K stars…")
    try:
        result = Catalogs.query_criteria(
            catalog="Tic",
            objType="STAR",
            Tmag=[8.0, 12.0],
        )
        tic_df = result.to_pandas() if result is not None else pd.DataFrame()
    except Exception as e:
        log.error("TIC query failed: %s", e)
        return pd.DataFrame()

    if tic_df.empty:
        return pd.DataFrame()

    # Filter to G/K spectral types if column exists
    if "SPType" in tic_df.columns:
        tic_df = tic_df[tic_df["SPType"].str.startswith(("G", "K"), na=False)]

    tic_df = tic_df[~tic_df["ID"].astype(int).isin(exclude_tics)]

    rows = []
    for _, row in tic_df.iterrows():
        if len(rows) >= n:
            break
        tic_id = int(row["ID"])
        if tic_id in exclude_tics:
            continue
        rows.append({
            "tic_id": tic_id,
            "class_label": "NOISE",
            "source_catalog": "TIC_Quiet",
            "catalog_period": 0.0,
            "catalog_source_id": str(tic_id),
            "notes": "MAD verification pending (done during feature extraction)",
        })
        exclude_tics.add(tic_id)

    log.info("Selected %d candidate NOISE targets (MAD check during extraction)", len(rows))
    return pd.DataFrame(rows)


# ── Per-target pipeline ───────────────────────────────────────

def process_single_target(row: dict, timeout: int = MAX_SECONDS_PER_TARGET) -> dict:
    """
    Run the full pipeline on a single target (download → stitch → detrend → features).
    Returns a dict with tic_id, class_label, all 25 features, and metadata.
    """
    tic_id = int(row["tic_id"])
    label  = row["class_label"]
    t_start = time.monotonic()

    result = {
        "tic_id":       tic_id,
        "class_label":  label,
        "source_catalog": row.get("source_catalog", ""),
        "catalog_period": row.get("catalog_period", 0.0),
        "success":      False,
        "error":        "",
    }
    # Pre-fill features with 0
    for feat in FEATURE_NAMES:
        result[feat] = 0.0

    try:
        from src.ingestion.fetcher import download_sectors
        from src.ingestion.stitcher import stitch_sectors
        from src.preprocessing.preprocessor import detrend_lc
        from src.features.feature_extractor import extract_features

        log.info("[TIC %d] Downloading sectors…", tic_id)
        sectors = download_sectors(tic_id, cache_dir=CACHE_DIR)

        log.info("[TIC %d] Stitching %d sector(s)…", tic_id, len(sectors))
        stitched = stitch_sectors(sectors)

        log.info("[TIC %d] Detrending…", tic_id)
        _, lc_detrended = detrend_lc(stitched)

        log.info("[TIC %d] Extracting features…", tic_id)
        feat_dict, bls_out, meta = extract_features(lc_detrended)

        elapsed = time.monotonic() - t_start
        if elapsed > timeout:
            log.warning("[TIC %d] Timeout exceeded (%.0fs > %ds)", tic_id, elapsed, timeout)
            result["error"] = f"timeout_exceeded:{elapsed:.0f}s"

        result.update(feat_dict)
        result["bls_period"]    = meta["bls_period"]
        result["bls_power"]     = meta["bls_power"]
        result["runtime_s"]     = elapsed
        result["success"]       = True

        # Audit log: compare detected vs catalog period (NOT used as model input)
        cat_period = float(row.get("catalog_period", 0.0))
        if cat_period > 0:
            ratio = meta["bls_period"] / cat_period
            log.info(
                "[TIC %d] AUDIT: detected_P=%.4fd catalog_P=%.4fd ratio=%.3f",
                tic_id, meta["bls_period"], cat_period, ratio,
            )

        log.info("[TIC %d] Done in %.1fs", tic_id, elapsed)

    except Exception as exc:
        elapsed = time.monotonic() - t_start
        log.error("[TIC %d] FAILED after %.1fs: %s", tic_id, elapsed, exc)
        result["error"]     = str(exc)
        result["runtime_s"] = elapsed

    return result


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build AIML_Exoplanet ground-truth dataset")
    parser.add_argument("--pilot",   action="store_true", help="Run 5-star pilot only")
    parser.add_argument("--full",    action="store_true", help="Process all 100 targets")
    parser.add_argument("--resume",  action="store_true", help="Skip already-processed TIC IDs")
    parser.add_argument("--workers", type=int, default=MAX_PARALLEL_WORKERS,
                        help=f"Parallel workers (max {MAX_PARALLEL_WORKERS})")
    parser.add_argument("--timeout", type=int, default=MAX_SECONDS_PER_TARGET,
                        help="Per-target timeout (seconds)")
    parser.add_argument("--download-catalogs", action="store_true",
                        help="Force re-download of catalog CSVs")
    args = parser.parse_args()

    if not args.pilot and not args.full:
        parser.error("Specify --pilot or --full")

    workers = min(args.workers, MAX_PARALLEL_WORKERS)
    os.makedirs(CATALOG_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR,   exist_ok=True)
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    # ── Step 1: Download catalogs ──────────────────────────────
    toi_path      = os.path.join(CATALOG_DIR, "toi_confirmed.csv")
    villanova_path = os.path.join(CATALOG_DIR, "villanova_eb.csv")
    gaia_path     = os.path.join(CATALOG_DIR, "gaia_dr3_variables.csv")

    toi_df, eb_df, gaia_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if args.download_catalogs or not os.path.exists(toi_path):
        toi_df = download_toi_catalog(toi_path)
    else:
        toi_df = pd.read_csv(toi_path)
        log.info("Loaded cached TOI catalog: %d records", len(toi_df))

    if args.download_catalogs or not os.path.exists(villanova_path):
        eb_df = download_villanova_eb_catalog(villanova_path)
    else:
        eb_df = pd.read_csv(villanova_path)
        log.info("Loaded cached Villanova EB catalog: %d records", len(eb_df))

    if args.download_catalogs or not os.path.exists(gaia_path):
        gaia_df = query_gaia_variables(gaia_path)
    else:
        gaia_df = pd.read_csv(gaia_path)
        log.info("Loaded cached Gaia catalog: %d records", len(gaia_df))

    # ── Step 2: Build manifest ────────────────────────────────
    if os.path.exists(TARGETS_MANIFEST):
        manifest = pd.read_csv(TARGETS_MANIFEST)
        log.info("Loaded existing manifest: %d targets", len(manifest))
    else:
        log.info("Building target manifest from catalogs…")
        planet_df = select_planet_targets(toi_df, n=25)
        all_tics = set(planet_df["tic_id"].tolist()) if not planet_df.empty else set()

        eb_targets = select_eb_targets(eb_df, n=25, exclude_tics=all_tics)
        if not eb_targets.empty:
            all_tics.update(eb_targets["tic_id"].tolist())

        var_targets = select_variable_targets(gaia_df, n=25, exclude_tics=all_tics)
        if not var_targets.empty:
            all_tics.update(var_targets["tic_id"].tolist())


        noise_targets = select_noise_targets(n=25, exclude_tics=all_tics)

        manifest = pd.concat(
            [df for df in [planet_df, eb_targets, var_targets, noise_targets] if not df.empty],
            ignore_index=True,
        )
        manifest.to_csv(TARGETS_MANIFEST, index=False)
        log.info("Manifest saved: %d targets | %s", len(manifest), TARGETS_MANIFEST)

    # ── Step 3: Select targets to process ────────────────────
    if args.pilot:
        targets = manifest.groupby("class_label").head(2).head(PILOT_SIZE)
        log.info("PILOT RUN: processing %d targets", len(targets))
    else:
        targets = manifest.copy()
        log.info("FULL RUN: processing %d targets", len(targets))

    # Resume: skip already-processed TIC IDs
    results_path = DATASET_PATH.replace(".parquet", "_results.json")
    already_done = set()
    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            done_rows = json.load(f)
        already_done = {r["tic_id"] for r in done_rows if r.get("success")}
        log.info("Resuming: %d targets already done", len(already_done))
        targets = targets[~targets["tic_id"].isin(already_done)]

    if len(targets) == 0:
        log.info("All targets already processed — nothing to do")
        return

    # ── Step 4: Run pipeline ──────────────────────────────────
    log.info("Processing %d targets with %d worker(s), %ds timeout…",
             len(targets), workers, args.timeout)

    all_results = []
    fail_count  = 0
    FAIL_LIMIT  = 10

    rows_list = targets.to_dict("records")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_single_target, row, args.timeout): row
            for row in rows_list
        }

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            all_results.append(result)

            if not result["success"]:
                fail_count += 1
                log.error("Failure %d/%d: TIC %d — %s",
                           fail_count, FAIL_LIMIT, result["tic_id"], result["error"])
                if fail_count >= FAIL_LIMIT:
                    log.critical("Failure limit reached (%d). STOPPING — review errors before continuing.", FAIL_LIMIT)
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
            else:
                log.info("Progress: %d/%d done", i + 1, len(rows_list))

            # Incremental save
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)

    # ── Step 5: Save dataset ──────────────────────────────────
    success_results = [r for r in all_results if r.get("success")]
    log.info("Completed: %d/%d successful", len(success_results), len(all_results))

    if success_results:
        df = pd.DataFrame(success_results)
        df.to_parquet(DATASET_PATH, index=False)
        log.info("Dataset saved: %s (%d rows)", DATASET_PATH, len(df))
        print(f"\n✅ Dataset saved to {DATASET_PATH}")
        print(f"   Rows: {len(df)} | Failures: {fail_count}")
        if "class_label" in df.columns:
            print(f"   Class distribution:\n{df['class_label'].value_counts().to_string()}")
    else:
        log.error("No successful results — dataset not saved")
        sys.exit(1)


if __name__ == "__main__":
    main()
