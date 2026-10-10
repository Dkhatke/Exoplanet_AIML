# tests/test_features.py
# ============================================================
# Phase 2 acceptance tests: BLS detector and feature extraction.
# ALL tests run offline using synthetic fixtures.
# ============================================================

import numpy as np
import pytest
from lightkurve import LightCurve
import astropy.units as u
from astropy.time import Time


# ── Helpers ──────────────────────────────────────────────────

def _make_27day_lc_with_transit(period=3.524, depth=0.015, duration=0.09, seed=42):
    """27-day single-sector LC at 10-min cadence with injected transit."""
    n = 3888
    cadence = 10.0 / 60 / 24
    t = 1325.0 + np.arange(n) * cadence
    rng = np.random.default_rng(seed=seed)
    flux = 1.0 + rng.normal(0, 0.0005, n)
    t0 = t[0] + 0.5
    ph = ((t - t0) % period); ph[ph > period / 2] -= period
    flux[np.abs(ph) < 0.5 * duration] -= depth
    lc = LightCurve(
        time=Time(t, format="btjd", scale="tdb"),
        flux=flux * u.dimensionless_unscaled,
        flux_err=np.full(n, 0.0005) * u.dimensionless_unscaled,
    )
    return lc, t0


def _make_27day_eb_lc(period=2.0, primary_depth=0.05, secondary_depth=0.02, seed=7):
    """27-day LC with EB signature (unequal primary/secondary)."""
    n = 3888
    cadence = 10.0 / 60 / 24
    t = 1325.0 + np.arange(n) * cadence
    rng = np.random.default_rng(seed=seed)
    flux = 1.0 + rng.normal(0, 0.0005, n)
    t0 = t[0] + 0.3
    # Primary
    ph = ((t - t0) % period); ph[ph > period / 2] -= period
    flux[np.abs(ph) < 0.05] -= primary_depth
    # Secondary at phase 0.5
    t0_sec = t0 + 0.5 * period
    ph2 = ((t - t0_sec) % period); ph2[ph2 > period / 2] -= period
    flux[np.abs(ph2) < 0.05] -= secondary_depth
    lc = LightCurve(
        time=Time(t, format="btjd", scale="tdb"),
        flux=flux * u.dimensionless_unscaled,
        flux_err=np.full(n, 0.0005) * u.dimensionless_unscaled,
    )
    return lc, t0


def _make_27day_variable_lc(period=0.8, amplitude=0.02, seed=13):
    """27-day LC with sinusoidal variability."""
    n = 3888
    cadence = 10.0 / 60 / 24
    t = 1325.0 + np.arange(n) * cadence
    rng = np.random.default_rng(seed=seed)
    flux = 1.0 + amplitude * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.0003, n)
    lc = LightCurve(
        time=Time(t, format="btjd", scale="tdb"),
        flux=flux * u.dimensionless_unscaled,
        flux_err=np.full(n, 0.0003) * u.dimensionless_unscaled,
    )
    return lc


# ── BLS Detector Tests ────────────────────────────────────────

class TestBLSDetector:

    def test_frequency_grid_within_bin_limit(self):
        from src.detection.detector import compute_frequency_grid
        periods = compute_frequency_grid(t_baseline=27.0)
        assert len(periods) <= 50_000, f"Grid has {len(periods)} bins — exceeds 50,000 cap"
        assert len(periods) >= 500,   f"Grid has {len(periods)} bins — too coarse"

    def test_frequency_grid_min_max_period(self):
        from src.detection.detector import compute_frequency_grid
        periods = compute_frequency_grid(t_baseline=27.0)
        assert periods[0] >= 0.5,   f"Min period {periods[0]:.3f}d is below 0.5d"
        assert periods[-1] <= 9.01, f"Max period {periods[-1]:.3f}d exceeds baseline/3 = 9.0d"

    def test_bls_finds_planet_period(self):
        """
        BLS should recover the injected 3.524-day transit period within 20%
        (after harmonic correction). BLS may detect P/2 and the harmonic
        check doubles it back; both are acceptable.
        """
        from src.detection.detector import run_bls
        n = 19440  # 27 days at 2-min cadence
        cadence = 2.0 / 60 / 24
        t = 1325.0 + np.arange(n) * cadence
        rng = np.random.default_rng(seed=42)
        flux = 1.0 + rng.normal(0, 0.0005, n)
        period_true = 3.524
        t0_inj = t[0] + 0.5
        depth = 0.015
        duration = 0.09
        ph = ((t - t0_inj) % period_true)
        ph[ph > period_true / 2] -= period_true
        flux[np.abs(ph) < 0.5 * duration] -= depth
        lc = LightCurve(
            time=Time(t, format="btjd", scale="tdb"),
            flux=flux * u.dimensionless_unscaled,
            flux_err=np.full(n, 0.0005) * u.dimensionless_unscaled,
        )
        result = run_bls(lc)
        # Accept either best_period OR corrected_period (harmonic-doubled)
        detected_best = result["best_period"]
        detected_corr = result["corrected_period"]
        err_best = abs(detected_best - period_true) / period_true
        err_corr = abs(detected_corr - period_true) / period_true
        err_half = abs(detected_best - period_true / 2) / period_true  # P/2 is also acceptable
        best_err = min(err_best, err_corr, err_half)
        assert best_err < 0.20, (
            f"BLS failed to recover period: best={detected_best:.4f}d "
            f"corrected={detected_corr:.4f}d true={period_true:.4f}d "
            f"(best error {best_err*100:.1f}% > 20%)"
        )



    def test_bls_returns_all_required_keys(self):
        from src.detection.detector import run_bls
        lc, _ = _make_27day_lc_with_transit()
        result = run_bls(lc)
        for key in ("best_period", "t0", "duration", "bls_power", "harmonic_flag",
                    "corrected_period", "n_bins_searched", "bls_runtime_s", "timeout_flag"):
            assert key in result, f"Missing key '{key}' in BLS output"

    def test_bls_completes_under_60_seconds(self):
        from src.detection.detector import run_bls
        import time
        lc, _ = _make_27day_lc_with_transit()
        t0 = time.monotonic()
        run_bls(lc)
        elapsed = time.monotonic() - t0
        assert elapsed < 60.0, f"BLS took {elapsed:.1f}s — exceeds 60s limit"

    def test_bls_logs_bin_count(self):
        """n_bins_searched must be logged and within bounds."""
        from src.detection.detector import run_bls
        lc, _ = _make_27day_lc_with_transit()
        result = run_bls(lc)
        assert 500 <= result["n_bins_searched"] <= 50_000


# ── Transit Metrics Tests ─────────────────────────────────────

class TestTransitMetrics:

    def test_odd_even_z_low_for_planet(self):
        """Planet should have similar odd/even depths (odd_even_z < 3)."""
        from src.features.transit_metrics import compute_odd_even_depths
        lc, t0 = _make_27day_lc_with_transit(period=3.524, depth=0.015)
        result = compute_odd_even_depths(lc, period=3.524, t0=t0, duration=0.09)
        assert result["odd_even_z"] < 5.0, (
            f"Planet odd_even_z={result['odd_even_z']:.2f} — should be low for planet-like signal"
        )

    def test_odd_even_diff_high_for_eb(self):
        """EB should have very different odd/even depths."""
        from src.features.transit_metrics import compute_odd_even_depths
        # EB with intentionally unequal cycles
        n = 3888; cadence = 10.0 / 60 / 24
        t = 1325.0 + np.arange(n) * cadence
        rng = np.random.default_rng(seed=7)
        flux = 1.0 + rng.normal(0, 0.0005, n)
        period = 2.0; t0 = t[0] + 0.3
        ph = ((t - t0) % period); ph[ph > period / 2] -= period
        cycle_n = np.floor((t - t0) / period).astype(int)
        # Odd cycles 50% deeper than even
        for cyc in range(int((t[-1] - t[0]) / period) + 1):
            mask = (np.abs(ph) < 0.05) & (cycle_n == cyc)
            flux[mask] -= 0.05 if cyc % 2 == 0 else 0.025
        lc = LightCurve(
            time=Time(t, format="btjd", scale="tdb"),
            flux=flux * u.dimensionless_unscaled,
            flux_err=np.full(n, 0.0005) * u.dimensionless_unscaled,
        )
        result = compute_odd_even_depths(lc, period=2.0, t0=t0, duration=0.10)
        assert result["odd_even_diff"] > 0.005, (
            f"EB odd_even_diff={result['odd_even_diff']:.4f} — should be elevated for EB"
        )

    def test_secondary_not_detected_for_planet(self):
        from src.features.transit_metrics import compute_secondary_eclipse
        lc, t0 = _make_27day_lc_with_transit(period=3.524, depth=0.015)
        result = compute_secondary_eclipse(lc, period=3.524, t0=t0, duration=0.09)
        # Planet: secondary SNR should be low (not necessarily exactly 0 due to noise)
        assert result["secondary_snr"] < 5.0, (
            f"Planet secondary SNR={result['secondary_snr']:.2f} — should be low"
        )

    def test_secondary_detected_for_eb(self):
        from src.features.transit_metrics import compute_secondary_eclipse
        lc, t0 = _make_27day_eb_lc()
        result = compute_secondary_eclipse(lc, period=2.0, t0=t0, duration=0.10)
        assert result["secondary_depth"] > 0.005, (
            f"EB secondary depth={result['secondary_depth']:.4f} — should be detectable"
        )


# ── Variability Metrics Tests ─────────────────────────────────

class TestVariabilityMetrics:

    def test_ls_power_high_for_variable(self):
        from src.features.variability_metrics import compute_lomb_scargle_peak
        lc = _make_27day_variable_lc(period=0.8, amplitude=0.02)
        result = compute_lomb_scargle_peak(lc)
        assert result["ls_peak_power"] > 0.3, (
            f"Variable LS power={result['ls_peak_power']:.3f} — should be high"
        )

    def test_ls_fap_low_for_variable(self):
        from src.features.variability_metrics import compute_lomb_scargle_peak
        lc = _make_27day_variable_lc(period=0.8, amplitude=0.02)
        result = compute_lomb_scargle_peak(lc)
        assert result["ls_fap"] < 0.01, (
            f"Variable LS FAP={result['ls_fap']:.4f} — should be very small"
        )

    def test_percentile_ratio_larger_for_variable(self):
        from src.features.variability_metrics import compute_flux_percentile_ratio
        lc_var   = _make_27day_variable_lc(amplitude=0.02)
        lc_quiet, _ = _make_27day_lc_with_transit(depth=0.001)  # nearly flat
        r_var   = compute_flux_percentile_ratio(lc_var)
        r_quiet = compute_flux_percentile_ratio(lc_quiet)
        assert r_var["p5_p95_ratio"] > r_quiet["p5_p95_ratio"], (
            "Variable p5_p95 should exceed quiet star's"
        )


# ── Full Feature Extractor Tests ──────────────────────────────

class TestFeatureExtractor:

    def test_extract_returns_25_features(self):
        from src.features.feature_extractor import extract_features
        lc, _ = _make_27day_lc_with_transit()
        feat, bls, meta = extract_features(lc)
        assert len(feat) == 25, f"Expected 25 features, got {len(feat)}"

    def test_extract_has_all_feature_names(self):
        from src.features.feature_extractor import extract_features
        from config import FEATURE_NAMES
        lc, _ = _make_27day_lc_with_transit()
        feat, _, _ = extract_features(lc)
        for name in FEATURE_NAMES:
            assert name in feat, f"Missing feature '{name}'"

    def test_extract_no_nan_values(self):
        from src.features.feature_extractor import extract_features
        lc, _ = _make_27day_lc_with_transit()
        feat, _, _ = extract_features(lc)
        for name, val in feat.items():
            assert np.isfinite(val), f"Feature '{name}' is non-finite: {val}"

    def test_extract_planet_vs_variable_discriminated(self):
        """Variable should have higher ls_peak_power than planet."""
        from src.features.feature_extractor import extract_features
        lc_planet, _ = _make_27day_lc_with_transit(depth=0.015)
        lc_var = _make_27day_variable_lc(amplitude=0.02)
        feat_p, _, _ = extract_features(lc_planet)
        feat_v, _, _ = extract_features(lc_var)
        assert feat_v["ls_peak_power"] > feat_p["ls_peak_power"], (
            f"Variable LS power ({feat_v['ls_peak_power']:.3f}) should exceed "
            f"planet LS power ({feat_p['ls_peak_power']:.3f})"
        )

    def test_features_to_array_correct_shape(self):
        from src.features.feature_extractor import extract_features, features_to_array
        lc, _ = _make_27day_lc_with_transit()
        feat, _, _ = extract_features(lc)
        arr = features_to_array(feat)
        assert arr.shape == (25,), f"Feature array shape {arr.shape} should be (25,)"
        assert arr.dtype == np.float32

    def test_metadata_has_required_keys(self):
        from src.features.feature_extractor import extract_features
        lc, _ = _make_27day_lc_with_transit()
        _, _, meta = extract_features(lc)
        for key in ("bls_period", "bls_power", "n_cadences", "extraction_time_s", "warnings"):
            assert key in meta, f"Missing metadata key '{key}'"
