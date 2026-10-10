# tests/test_recycler_modules.py
# ============================================================
# Phase 5 acceptance tests: specialized recycler modules.
# ALL tests run offline using synthetic light curves built inline
# (27-day baselines so BLS can recover the injected periods).
#
# Covers:
#   AC-5.1  planet_module uses dynamic stellar_params
#   AC-5.2  eb_module identifies the secondary eclipse
#   AC-5.3  variable_module detects the period within 5%
#   AC-5.4  noise_module returns a diagnostics dict
#   AC-5.5  route_to_recycler handles all 5 labels incl. REVIEW
# ============================================================

import numpy as np
import pytest
from lightkurve import LightCurve
import astropy.units as u
from astropy.time import Time


# -- Synthetic light-curve builders (27-day baseline) -----------

def _base_time(n=3888):
    cadence = 10.0 / 60 / 24
    return 1325.0 + np.arange(n) * cadence


def _make_lc(t, flux, err):
    return LightCurve(
        time=Time(t, format="btjd", scale="tdb"),
        flux=flux * u.dimensionless_unscaled,
        flux_err=err * u.dimensionless_unscaled,
    )


def make_planet_lc(period=3.524, depth=0.015, duration=0.09, seed=42):
    t = _base_time()
    rng = np.random.default_rng(seed)
    flux = 1.0 + rng.normal(0, 0.0005, len(t))
    t0 = t[0] + 0.5
    ph = ((t - t0) % period); ph[ph > period / 2] -= period
    flux[np.abs(ph) < 0.5 * duration] -= depth
    return _make_lc(t, flux, np.full(len(t), 0.0005)), t0, period, depth, duration


def make_eb_lc(period=2.0, primary_depth=0.05, secondary_depth=0.02, seed=7):
    t = _base_time()
    rng = np.random.default_rng(seed)
    flux = 1.0 + rng.normal(0, 0.0005, len(t))
    t0 = t[0] + 0.3
    ph = ((t - t0) % period); ph[ph > period / 2] -= period
    flux[np.abs(ph) < 0.05] -= primary_depth
    t0_sec = t0 + 0.5 * period
    ph2 = ((t - t0_sec) % period); ph2[ph2 > period / 2] -= period
    flux[np.abs(ph2) < 0.05] -= secondary_depth
    return _make_lc(t, flux, np.full(len(t), 0.0005)), t0, period


def make_variable_lc(period=0.8, amplitude=0.02, seed=13):
    t = _base_time()
    rng = np.random.default_rng(seed)
    flux = 1.0 + amplitude * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.0003, len(t))
    return _make_lc(t, flux, np.full(len(t), 0.0003)), period


def make_noise_lc(seed=99):
    t = _base_time()
    rng = np.random.default_rng(seed)
    flux = 1.0 + rng.normal(0, 0.0005, len(t))
    return _make_lc(t, flux, np.full(len(t), 0.0005))


def _run_bls(lc):
    from src.detection.detector import run_bls
    return run_bls(lc)



# -- Planet Module (AC-5.1) ------------------------------------

class TestPlanetModule:

    def test_returns_valid_rp_and_axis(self):
        from src.recycler.planet_module import analyze_planet_candidate
        lc, t0, period, depth, dur = make_planet_lc()
        bls = _run_bls(lc)
        star = {"radius_solar": 1.2, "mass_solar": 1.1, "teff_k": 6000, "source": "TIC"}
        out = analyze_planet_candidate(lc, bls, star)

        assert out["rp_earth_radii"] > 0, "Rp must be positive"
        assert out["semi_major_axis_au"] > 0, "Semi-major axis must be positive"
        assert np.isfinite(out["eq_temperature_k"])
        assert out["eq_temperature_k"] > 0

    def test_uses_dynamic_stellar_params_not_hardcoded(self):
        """AC-5.1: two different stars -> different Rp for the same light curve."""
        from src.recycler.planet_module import analyze_planet_candidate
        lc, *_ = make_planet_lc()
        bls = _run_bls(lc)

        small_star = {"radius_solar": 0.5, "mass_solar": 0.5, "teff_k": 4000}
        big_star = {"radius_solar": 2.0, "mass_solar": 2.0, "teff_k": 8000}

        rp_small = analyze_planet_candidate(lc, bls, small_star)["rp_earth_radii"]
        rp_big = analyze_planet_candidate(lc, bls, big_star)["rp_earth_radii"]

        # Same depth => Rp scales linearly with stellar radius.
        assert abs(rp_big - 4.0 * rp_small) < 1e-6, (
            "Rp should scale linearly with R* (2.0/0.5 = 4x)"
        )
        assert analyze_planet_candidate(lc, bls, big_star)["stellar_params_used"]["radius_solar"] == 2.0

    def test_teff_used_from_stellar_params(self):
        from src.recycler.planet_module import analyze_planet_candidate
        lc, *_ = make_planet_lc()
        bls = _run_bls(lc)
        hot = analyze_planet_candidate(lc, bls, {"radius_solar": 1.0, "mass_solar": 1.0, "teff_k": 9000})
        cool = analyze_planet_candidate(lc, bls, {"radius_solar": 1.0, "mass_solar": 1.0, "teff_k": 3500})
        assert hot["eq_temperature_k"] > cool["eq_temperature_k"]

    def test_falls_back_to_defaults_when_missing(self):
        from src.recycler.planet_module import analyze_planet_candidate
        lc, *_ = make_planet_lc()
        bls = _run_bls(lc)
        out = analyze_planet_candidate(lc, bls, {})  # empty -> defaults
        assert out["stellar_params_used"]["used_defaults"], "should flag defaults used"
        assert out["rp_earth_radii"] > 0  # still produces a value


# -- EB Module (AC-5.2) ----------------------------------------

class TestEBModule:

    def test_detects_secondary_eclipse(self):
        from src.recycler.eb_module import analyze_eb
        lc, t0, period = make_eb_lc(secondary_depth=0.03)
        bls = _run_bls(lc)
        out = analyze_eb(lc, bls)
        assert out["secondary_depth"] > 0, "should measure a secondary eclipse"
        assert out["secondary_snr"] > 3.0, "secondary eclipse SNR should be significant"
        assert out["secondary_detected"] is True

    def test_classifies_subtype(self):
        from src.recycler.eb_module import analyze_eb
        lc, t0, period = make_eb_lc()
        bls = _run_bls(lc)
        out = analyze_eb(lc, bls)
        assert out["eb_subtype"] in ("DETACHED", "CONTACT", "SEMI_DETACHED")

    def test_period_doubling_test_present(self):
        from src.recycler.eb_module import analyze_eb


# -- Variable Module (AC-5.3) ----------------------------------

class TestVariableModule:

    def test_detects_dominant_period_within_5pct(self):
        from src.recycler.variable_module import analyze_variable
        injected = 0.8
        lc, period = make_variable_lc(period=injected)
        out = analyze_variable(lc, lc)
        detected = out["dominant_period_days"]
        rel_err = abs(detected - injected) / injected
        assert rel_err < 0.05, (
            f"detected period {detected:.4f}d off by {rel_err*100:.1f}% (injected {injected}d)"
        )

    def test_amplitude_metrics(self):
        from src.recycler.variable_module import analyze_variable
        lc, period = make_variable_lc(amplitude=0.03)
        out = analyze_variable(lc, lc)
        assert out["amplitude_p5_p95"] > 0.01
        assert out["amplitude_minmax"] >= out["amplitude_p5_p95"]

    def test_mode_classified(self):
        from src.recycler.variable_module import analyze_variable
        lc, period = make_variable_lc()
        out = analyze_variable(lc, lc)
        assert out["mode"] in ("PULSATION", "ROTATION", "QUASI_PERIODIC", "UNKNOWN")

    def test_gaia_crosscheck_graceful_without_tic(self):
        from src.recycler.variable_module import analyze_variable
        lc, period = make_variable_lc()
        out = analyze_variable(lc, lc, tic_id=None)
        assert out["gaia_crosscheck"]["checked"] is False


# -- Noise Module (AC-5.4) -------------------------------------

class TestNoiseModule:

    def test_returns_cdpp_and_flags(self):
        from src.recycler.noise_module import analyze_noise
        lc = make_noise_lc()
        out = analyze_noise(lc)
        assert "cdpp_1hr_ppm" in out
        assert "cdpp_6hr_ppm" in out
        assert out["cdpp_1hr_ppm"] > 0
        assert out["cdpp_6hr_ppm"] > 0
        assert isinstance(out["systematic_flag"], bool)
        assert isinstance(out["instrument_period_flag"], bool)

    def test_white_noise_not_flagged_systematic(self):
        from src.recycler.noise_module import analyze_noise
        lc = make_noise_lc()
        out = analyze_noise(lc)
        assert out["systematic_flag"] is False

    def test_cdpp_scales_down_with_timescale(self):
        from src.recycler.noise_module import analyze_noise
        lc = make_noise_lc()
        out = analyze_noise(lc)
        assert out["cdpp_6hr_ppm"] < out["cdpp_1hr_ppm"]


# -- Router (AC-5.5) -------------------------------------------

class TestRecyclerRouter:

    def test_routes_all_five_labels(self):
        import config
        from src.recycler import route_to_recycler

        planet_lc, *_ = make_planet_lc()
        bls = _run_bls(planet_lc)
        feats = {k: 0.0 for k in config.FEATURE_NAMES}
        star = {"radius_solar": 1.0, "mass_solar": 1.0, "teff_k": 5778}

        expected_modules = {
            "PLANET": "planet_module",
            "EB": "eb_module",
            "VARIABLE": "variable_module",
            "NOISE": "noise_module",
            "REVIEW": "review",
        }
        for cls, mod in expected_modules.items():
            out = route_to_recycler(
                cls, planet_lc, planet_lc, bls, feats,
                stellar_params=star, class_probabilities={cls: 0.8},
            )
            assert out["predicted_class"] == cls
            assert out["module"] == mod, f"{cls} routed to {out['module']}, expected {mod}"
            assert "details" in out

    def test_unknown_class_routes_to_review(self):
        from src.recycler import route_to_recycler
        lc, *_ = make_planet_lc()
        bls = _run_bls(lc)
        out = route_to_recycler("BOGUS", lc, lc, bls, {})
        assert out["predicted_class"] == "REVIEW"
        assert out["module"] == "review"

    def test_router_never_raises_on_module_error(self):
        """Router must degrade gracefully, never propagate an exception."""
        from src.recycler import route_to_recycler
        lc, *_ = make_planet_lc()
        # Pass a deliberately broken bls_output (None) to force an error path.
        out = route_to_recycler("PLANET", lc, lc, None, {})
        assert out["predicted_class"] == "PLANET"
        # Either it succeeded with defaults or returned a structured error.
        assert out["module"] in ("planet_module", "error")


    def test_planet_lc_has_no_secondary(self):
        """A planet LC should NOT trigger a significant secondary eclipse."""
        from src.recycler.eb_module import analyze_eb
        lc, *_ = make_planet_lc()
        bls = _run_bls(lc)
        out = analyze_eb(lc, bls)
        assert out["secondary_detected"] is False, (
            "planet-like curve should not show a significant secondary eclipse"
        )
