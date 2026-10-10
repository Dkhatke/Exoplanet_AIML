# tests/test_harness.py
# ============================================================
# Phase 0 Acceptance Test: Validates that the offline test harness
# works correctly. ALL tests must pass without internet access.
# Run: pytest tests/test_harness.py -v
# ============================================================

import numpy as np
import pytest
import lightkurve as lk
from lightkurve import LightCurve


class TestFixtureSchema:
    """Verify each fixture returns a valid LightCurve with expected shape."""

    def test_planet_lc_is_lightcurve(self, synthetic_planet_lc):
        assert isinstance(synthetic_planet_lc, LightCurve)

    def test_eb_lc_is_lightcurve(self, synthetic_eb_lc):
        assert isinstance(synthetic_eb_lc, LightCurve)

    def test_variable_lc_is_lightcurve(self, synthetic_variable_lc):
        assert isinstance(synthetic_variable_lc, LightCurve)

    def test_noise_lc_is_lightcurve(self, synthetic_noise_lc):
        assert isinstance(synthetic_noise_lc, LightCurve)

    def test_all_fixtures_have_1500_cadences(self, all_synthetic_lcs):
        for name, lc in all_synthetic_lcs.items():
            assert len(lc) == 1500, f"{name} LightCurve should have 1500 cadences"

    def test_all_fixtures_have_no_nan(self, all_synthetic_lcs):
        for name, lc in all_synthetic_lcs.items():
            assert not np.any(np.isnan(lc.flux.value)), f"{name} flux contains NaN"
            assert not np.any(np.isnan(lc.time.value)), f"{name} time contains NaN"

    def test_all_fixtures_time_is_increasing(self, all_synthetic_lcs):
        for name, lc in all_synthetic_lcs.items():
            diffs = np.diff(lc.time.value)
            assert np.all(diffs > 0), f"{name} time array is not strictly increasing"


class TestPlanetFixtureProperties:
    """Verify the planet fixture has physically meaningful properties."""

    def test_planet_mean_flux_near_unity(self, synthetic_planet_lc):
        mean_flux = np.mean(synthetic_planet_lc.flux.value)
        assert 0.99 < mean_flux < 1.01, f"Planet mean flux {mean_flux:.4f} should be near 1.0"

    def test_planet_has_detectable_dip(self, synthetic_planet_lc):
        """The planet light curve should have a minimum flux well below 1.0."""
        min_flux = np.min(synthetic_planet_lc.flux.value)
        # Injected depth = 1.5%, so minimum flux should be < 0.99
        assert min_flux < 0.990, f"Planet min flux {min_flux:.4f} should show transit dip"

    def test_planet_baseline_is_flat(self, synthetic_planet_lc):
        """Out-of-transit scatter should be < 0.2% RMS (no stellar variability)."""
        flux = synthetic_planet_lc.flux.value
        # Rough OOT: use values above 0.99 (above transit depth region)
        oot = flux[flux > 0.990]
        rms = np.std(oot)
        assert rms < 0.002, f"Planet OOT RMS {rms:.5f} too high — baseline not flat"


class TestEBFixtureProperties:
    """Verify the EB fixture has both primary and secondary eclipses."""

    def test_eb_has_deep_primary(self, synthetic_eb_lc):
        min_flux = np.min(synthetic_eb_lc.flux.value)
        # Primary depth is 5%; flux should drop below 0.96
        assert min_flux < 0.96, f"EB primary eclipse not deep enough: min={min_flux:.4f}"

    def test_eb_flux_variance_higher_than_noise(self, synthetic_eb_lc, synthetic_noise_lc):
        """EB should have much higher flux variance than the noise floor."""
        eb_var = np.var(synthetic_eb_lc.flux.value)
        noise_var = np.var(synthetic_noise_lc.flux.value)
        assert eb_var > noise_var * 10, (
            f"EB variance ({eb_var:.6f}) should be >> noise variance ({noise_var:.6f})"
        )


class TestVariableFixtureProperties:
    """Verify the variable fixture has sinusoidal variability."""

    def test_variable_has_large_amplitude(self, synthetic_variable_lc):
        flux = synthetic_variable_lc.flux.value
        p2p = np.max(flux) - np.min(flux)
        # Injected amplitude ~2%, so peak-to-peak should be ~ 4%
        assert p2p > 0.03, f"Variable p2p {p2p:.4f} should be > 0.03 for 2% amplitude"

    def test_variable_is_more_variable_than_planet(self, synthetic_variable_lc, synthetic_planet_lc):
        """Variable should have higher overall std than the planet fixture."""
        var_std = np.std(synthetic_variable_lc.flux.value)
        planet_std = np.std(synthetic_planet_lc.flux.value)
        assert var_std > planet_std, (
            f"Variable std ({var_std:.5f}) should exceed planet std ({planet_std:.5f})"
        )


class TestNoiseFixtureProperties:
    """Verify the noise fixture has no significant periodic signal."""

    def test_noise_mean_near_unity(self, synthetic_noise_lc):
        mean = np.mean(synthetic_noise_lc.flux.value)
        assert 0.999 < mean < 1.001, f"Noise mean flux {mean:.5f} should be near 1.0"

    def test_noise_has_smallest_variance(self, all_synthetic_lcs):
        """Noise fixture should have the smallest flux variance of all 4 classes."""
        variances = {name: np.var(lc.flux.value) for name, lc in all_synthetic_lcs.items()}
        noise_var = variances["NOISE"]
        for name, var in variances.items():
            if name != "NOISE":
                assert var > noise_var, (
                    f"{name} variance ({var:.7f}) should exceed NOISE variance ({noise_var:.7f})"
                )


class TestConfigImport:
    """Verify config.py imports cleanly and has required new constants."""

    def test_config_imports(self):
        import config  # noqa: F401

    def test_config_has_class_labels(self):
        from config import CLASS_LABELS
        assert set(CLASS_LABELS) == {"PLANET", "EB", "VARIABLE", "NOISE"}

    def test_config_has_feature_names_25(self):
        from config import FEATURE_NAMES
        assert len(FEATURE_NAMES) == 25, f"FEATURE_NAMES should have 25 entries, got {len(FEATURE_NAMES)}"

    def test_config_has_no_hardcoded_hd209458_radius(self):
        import config
        # Old hardcoded values should NOT exist as top-level constants
        assert not hasattr(config, "STAR_RADIUS_SOLAR"), \
            "STAR_RADIUS_SOLAR should not be in config (use DEFAULT_STAR_RADIUS_SOLAR)"
        assert not hasattr(config, "PLANET_RADIUS_JUP"), \
            "PLANET_RADIUS_JUP (HD 209458 specific) should be removed from config"

    def test_config_resource_limits(self):
        from config import MAX_PARALLEL_WORKERS, MAX_SECONDS_PER_TARGET, MAX_BLS_FREQUENCY_BINS
        assert MAX_PARALLEL_WORKERS <= 3, "Workers must never exceed 3"
        assert MAX_SECONDS_PER_TARGET == 120
        assert MAX_BLS_FREQUENCY_BINS == 50_000

    def test_config_review_threshold(self):
        from config import REVIEW_PROB_THRESHOLD
        assert 0.5 < REVIEW_PROB_THRESHOLD <= 1.0


class TestLoggerModule:
    """Verify the logger module initialises without errors."""

    def test_get_logger_returns_logger(self):
        from src.utils.logger import get_logger
        import logging
        log = get_logger("test.module", log_file=False)
        assert isinstance(log, logging.Logger)

    def test_logger_does_not_duplicate_handlers(self):
        from src.utils.logger import get_logger
        log1 = get_logger("test.dedup", log_file=False)
        log2 = get_logger("test.dedup", log_file=False)
        # Should not add handlers on second call
        assert len(log1.handlers) == len(log2.handlers)
        assert log1 is log2
