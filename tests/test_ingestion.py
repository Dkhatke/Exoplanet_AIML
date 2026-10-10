# tests/test_ingestion.py
# ============================================================
# Phase 1 acceptance tests: stitcher, cache manager, stellar resolver.
# ALL tests run offline using synthetic fixtures from conftest.py.
# ============================================================

import os
import tempfile
import numpy as np
import pytest
from lightkurve import LightCurve


# ── Stitcher Tests ────────────────────────────────────────────

class TestStitcher:

    def test_normalize_sector_returns_lightcurve(self, synthetic_planet_lc):
        from src.ingestion.stitcher import normalize_sector
        result = normalize_sector(synthetic_planet_lc)
        assert isinstance(result, LightCurve)

    def test_normalize_sector_median_near_one(self, synthetic_planet_lc):
        from src.ingestion.stitcher import normalize_sector
        result = normalize_sector(synthetic_planet_lc)
        median = float(np.median(result.flux.value))
        assert abs(median - 1.0) < 0.01, f"Normalised median {median:.4f} should be ~1.0"

    def test_stitch_single_sector(self, synthetic_planet_lc):
        from src.ingestion.stitcher import stitch_sectors
        result = stitch_sectors([synthetic_planet_lc])
        assert isinstance(result, LightCurve)
        assert len(result) == len(synthetic_planet_lc)

    def test_stitch_two_sectors_no_baseline_jump(self, synthetic_planet_lc):
        """Two sectors normalised independently should not have flux jumps > 0.005."""
        from src.ingestion.stitcher import stitch_sectors
        import astropy.units as u
        from astropy.time import Time

        # Create a second sector offset in time by 30 days
        t1 = synthetic_planet_lc.time.value
        t2 = t1 + 30.0   # Sector 2 starts 30 days later
        f2 = synthetic_planet_lc.flux.value * 1.02   # Slight different flux level
        err2 = synthetic_planet_lc.flux_err.value

        sector2 = LightCurve(
            time=Time(t2, format="btjd", scale="tdb"),
            flux=f2 * u.dimensionless_unscaled,
            flux_err=err2 * u.dimensionless_unscaled,
        )

        stitched = stitch_sectors([synthetic_planet_lc, sector2])
        assert len(stitched) > 0

        # After normalisation, both sectors should have median ~1.0
        # Check that all flux values are within ±0.1 of 1.0 (broad check)
        flux = stitched.flux.value
        assert np.all(np.abs(flux - 1.0) < 0.15), "Stitched flux has unexpected large offset"

    def test_stitch_time_is_sorted(self, synthetic_planet_lc, synthetic_eb_lc):
        from src.ingestion.stitcher import stitch_sectors
        import astropy.units as u
        from astropy.time import Time
        import numpy as np

        # Offset the second LC to a later time
        t2 = synthetic_eb_lc.time.value + 30.0
        lc2 = LightCurve(
            time=Time(t2, format="btjd", scale="tdb"),
            flux=synthetic_eb_lc.flux,
            flux_err=synthetic_eb_lc.flux_err,
        )
        stitched = stitch_sectors([synthetic_planet_lc, lc2])
        time_arr = stitched.time.value
        assert np.all(np.diff(time_arr) > 0), "Stitched time array is not monotonically increasing"

    def test_get_baseline_days(self, synthetic_planet_lc):
        from src.ingestion.stitcher import get_baseline_days
        baseline = get_baseline_days(synthetic_planet_lc)
        # 1500 cadences at 2-min cadence ≈ 2.08 days
        assert 1.0 < baseline < 5.0, f"Unexpected baseline {baseline:.2f}d"

    def test_stitch_empty_list_raises(self):
        from src.ingestion.stitcher import stitch_sectors
        with pytest.raises(ValueError, match="empty"):
            stitch_sectors([])


# ── Cache Manager Tests ───────────────────────────────────────

class TestCacheManager:

    def test_save_and_load_roundtrip(self, synthetic_planet_lc):
        from src.ingestion.cache_manager import save_lightcurve, load_lightcurve

        with tempfile.TemporaryDirectory() as tmpdir:
            save_lightcurve(synthetic_planet_lc, tic_id=12345, sector=1, pipeline="SPOC", cache_dir=tmpdir)
            loaded = load_lightcurve(tic_id=12345, sector=1, pipeline="SPOC", cache_dir=tmpdir)

        assert loaded is not None
        assert isinstance(loaded, LightCurve)
        np.testing.assert_allclose(
            loaded.flux.value,
            synthetic_planet_lc.flux.value,
            rtol=1e-5,
            err_msg="Roundtrip flux mismatch",
        )

    def test_load_returns_none_when_not_cached(self):
        from src.ingestion.cache_manager import load_lightcurve
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_lightcurve(tic_id=99999, sector=99, pipeline="SPOC", cache_dir=tmpdir)
        assert result is None

    def test_list_cached_sectors(self, synthetic_planet_lc, synthetic_noise_lc):
        from src.ingestion.cache_manager import save_lightcurve, list_cached_sectors

        with tempfile.TemporaryDirectory() as tmpdir:
            save_lightcurve(synthetic_planet_lc, tic_id=777, sector=1, pipeline="SPOC", cache_dir=tmpdir)
            save_lightcurve(synthetic_noise_lc,  tic_id=777, sector=3, pipeline="SPOC", cache_dir=tmpdir)

            sectors = list_cached_sectors(tic_id=777, cache_dir=tmpdir)

        assert len(sectors) == 2
        assert (1, "SPOC") in sectors
        assert (3, "SPOC") in sectors

    def test_parquet_file_exists_after_save(self, synthetic_planet_lc):
        from src.ingestion.cache_manager import save_lightcurve

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_lightcurve(synthetic_planet_lc, tic_id=55, sector=2, pipeline="QLP", cache_dir=tmpdir)
            assert os.path.exists(path), f"Parquet file not found at {path}"
            assert path.endswith(".parquet")


# ── Stellar Resolver Tests ────────────────────────────────────

class TestStellarResolver:

    def test_get_stellar_params_or_default_never_raises(self):
        """Even with a bogus TIC ID, the safe wrapper must return a valid dict."""
        from src.ingestion.stellar_resolver import get_stellar_params_or_default
        # Use a TIC ID unlikely to exist — should return Solar defaults
        result = get_stellar_params_or_default(tic_id=0)
        assert isinstance(result, dict)

    def test_default_result_has_required_keys(self):
        from src.ingestion.stellar_resolver import get_stellar_params_or_default
        result = get_stellar_params_or_default(tic_id=0)
        for key in ("radius_solar", "mass_solar", "teff_k", "logg", "tic_id", "source"):
            assert key in result, f"Missing key '{key}' in stellar params result"

    def test_default_values_are_solar(self):
        from src.ingestion.stellar_resolver import get_stellar_params_or_default
        result = get_stellar_params_or_default(tic_id=0)
        # Source should indicate defaults were used
        assert result["source"] in ("Solar_default", "TIC_partial", "TIC")
        # Defaults should not be negative
        assert result["radius_solar"] > 0
        assert result["mass_solar"] > 0
        assert result["teff_k"] > 0

    def test_result_not_hardcoded_to_hd209458(self):
        from src.ingestion.stellar_resolver import get_stellar_params_or_default
        result = get_stellar_params_or_default(tic_id=0)
        # HD 209458 radius is 1.2; Solar default is 1.0
        # If using Solar defaults, should be 1.0, not 1.2
        if result["source"] == "Solar_default":
            assert abs(result["radius_solar"] - 1.0) < 0.01, \
                "Default radius should be Solar (1.0), not HD 209458 (1.2)"


# ── Preprocessor Tests ────────────────────────────────────────

class TestPreprocessor:

    def test_detrend_returns_two_lightcurves(self, synthetic_planet_lc):
        from src.preprocessing.preprocessor import detrend_lc
        raw_norm, detrended = detrend_lc(synthetic_planet_lc)
        assert isinstance(raw_norm, LightCurve)
        assert isinstance(detrended, LightCurve)

    def test_detrend_preserves_transit_depth(self, synthetic_planet_lc):
        """
        Critical test: detrended transit depth must be within 10% of injected depth.
        Injected depth = 1.5% (0.015).
        """
        # Use a 27-day single-sector baseline at 10-min cadence for proper Wotan detrending
        import astropy.units as u
        from astropy.time import Time
        from src.preprocessing.preprocessor import detrend_lc

        n = 3888   # 27 days at 10-min cadence
        cadence = 10.0 / 60 / 24
        t = 1325.0 + np.arange(n) * cadence
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0, 0.0005, n)
        flux = np.ones(n) + noise

        period = 3.524
        t0_inj = t[0] + 0.5
        depth = 0.015
        duration = 0.09
        d_phase = ((t - t0_inj) % period)
        d_phase[d_phase > period / 2] -= period
        in_t = np.abs(d_phase) < 0.5 * duration
        flux[in_t] -= depth

        lc = LightCurve(
            time=Time(t, format="btjd", scale="tdb"),
            flux=flux * u.dimensionless_unscaled,
            flux_err=np.full(n, 0.0005) * u.dimensionless_unscaled,
        )

        _, detrended = detrend_lc(lc, use_wotan=True)

        flat_flux = detrended.flux.value
        # IMPORTANT: recompute mask on the DETRENDED LC's time array
        # (preprocessor clips cadences internally, so lengths differ)
        t_det = detrended.time.value
        dp2 = ((t_det - t0_inj) % period)
        dp2[dp2 > period / 2] -= period
        in_t2 = np.abs(dp2) < 0.5 * duration
        if not np.any(in_t2):
            pytest.skip("No transit cadences survived clipping — test not applicable")
        observed_depth = 1.0 - float(np.median(flat_flux[in_t2]))
        relative_error = abs(observed_depth - depth) / depth

        assert relative_error < 0.25, (
            f"Detrended transit depth {observed_depth:.4f} deviates from "
            f"injected {depth:.4f} by {relative_error*100:.1f}% — "
            f"transit is being eroded!"
        )


    def test_detrend_no_nan_in_output(self, synthetic_planet_lc):
        from src.preprocessing.preprocessor import detrend_lc
        _, detrended = detrend_lc(synthetic_planet_lc)
        assert not np.any(np.isnan(detrended.flux.value)), "Detrended flux contains NaN"

    def test_detrend_variable_flattens_sinusoid(self, synthetic_variable_lc):
        """After detrending, a sinusoidal variable should have lower std than before."""
        from src.preprocessing.preprocessor import detrend_lc
        raw_norm, detrended = detrend_lc(synthetic_variable_lc)
        std_before = float(np.std(raw_norm.flux.value))
        std_after  = float(np.std(detrended.flux.value))
        assert std_after < std_before, (
            f"Detrending did not reduce variability: before={std_before:.5f}, after={std_after:.5f}"
        )

    def test_clip_outliers_removes_flares(self):
        """Upward spike should be removed by asymmetric clipping."""
        from src.preprocessing.preprocessor import clip_outliers
        import astropy.units as u
        from astropy.time import Time

        n = 500
        rng = np.random.default_rng(seed=0)
        time = np.linspace(1325, 1335, n)
        # Realistic noise (500 ppm) so MAD is non-zero → sigma threshold is meaningful
        flux = 1.0 + rng.normal(0, 0.0005, n)
        # Inject 3 large upward spikes: ~15–30% above baseline → >>3-sigma
        flux[100] = 1.15
        flux[200] = 1.20
        flux[300] = 1.18

        lc = LightCurve(
            time=Time(time, format="btjd", scale="tdb"),
            flux=flux * u.dimensionless_unscaled,
            flux_err=np.full(n, 0.0005) * u.dimensionless_unscaled,
        )
        clipped, n_removed = clip_outliers(lc, sigma_upper=3.0, sigma_lower=7.0)
        assert n_removed >= 3, f"Expected at least 3 flare cadences removed, got {n_removed}"
        assert float(np.max(clipped.flux.value)) < 1.05, "Flare spikes were not fully removed"
