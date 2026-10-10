# tests/conftest.py
# ============================================================
# Offline pytest fixtures for AIML_Exoplanet pipeline.
# ALL fixtures generate synthetic light curves locally.
# NO internet access is required for any test using these fixtures.
# ============================================================

import numpy as np
import pytest
import lightkurve as lk
from lightkurve import LightCurve
from astropy.time import Time
import astropy.units as u


# ── Helpers ──────────────────────────────────────────────────

def _make_time_array(n_cadences: int = 1000, cadence_days: float = 2.0 / 60 / 24) -> np.ndarray:
    """Return a uniformly spaced time array (BTJD days)."""
    t_start = 1325.0  # Representative TESS BTJD start
    return np.linspace(t_start, t_start + n_cadences * cadence_days, n_cadences)


def _inject_box_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    depth: float,
    duration: float,
) -> np.ndarray:
    """Inject periodic box-shaped transit dips into a flux array."""
    flux = flux.copy()
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0          # Centre phase at 0
    half_width = 0.5 * duration / period
    in_transit = np.abs(phase) < half_width
    flux[in_transit] -= depth
    return flux


def _make_lc(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> LightCurve:
    """Wrap numpy arrays into a LightCurve object (no Astropy Quantity wrapping)."""
    return LightCurve(
        time=Time(time, format="btjd", scale="tdb"),
        flux=flux * u.dimensionless_unscaled,
        flux_err=flux_err * u.dimensionless_unscaled,
    )


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def synthetic_planet_lc() -> LightCurve:
    """
    Synthetic PLANET-class light curve.
    - Flat baseline (no stellar variability)
    - Periodic box-shaped transit: period=3.524 d, depth=1.5%, duration=0.09 d
    - Equal odd/even depths (planet-like)
    - No secondary eclipse at phase 0.5
    """
    n = 1500
    time = _make_time_array(n_cadences=n)
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 0.0005, n)          # 500 ppm white noise
    flux = np.ones(n) + noise

    flux = _inject_box_transit(
        time, flux,
        period=3.524,
        t0=time[0] + 0.5,
        depth=0.015,
        duration=0.09,
    )
    flux_err = np.full(n, 0.0005)
    return _make_lc(time, flux, flux_err)


@pytest.fixture
def synthetic_eb_lc() -> LightCurve:
    """
    Synthetic EB-class light curve.
    - Primary eclipse: depth=5%, duration=0.1 d, period=2.0 d
    - Secondary eclipse: depth=2% at phase 0.5 (unequal depths = EB signature)
    - Odd transits deeper than even (another EB discriminator)
    """
    n = 1500
    time = _make_time_array(n_cadences=n)
    rng = np.random.default_rng(seed=7)
    noise = rng.normal(0, 0.0008, n)
    flux = np.ones(n) + noise

    # Primary eclipse
    flux = _inject_box_transit(time, flux, period=2.0, t0=time[0] + 0.3,
                               depth=0.05, duration=0.10)
    # Secondary eclipse at phase ~0.5
    flux = _inject_box_transit(time, flux, period=2.0, t0=time[0] + 0.3 + 1.0,
                               depth=0.02, duration=0.10)

    # Make odd transits slightly deeper than even (EB odd/even signature)
    phase = ((time - (time[0] + 0.3)) % 2.0) / 2.0
    phase[phase > 0.5] -= 1.0
    half_w = 0.5 * 0.10 / 2.0
    in_primary = np.abs(phase) < half_w
    # Cycle indices
    cycle_num = ((time - (time[0] + 0.3)) // 2.0).astype(int)
    odd_mask = in_primary & (cycle_num % 2 == 1)
    flux[odd_mask] -= 0.005                   # Odd transits 0.5% deeper

    flux_err = np.full(n, 0.0008)
    return _make_lc(time, flux, flux_err)


@pytest.fixture
def synthetic_variable_lc() -> LightCurve:
    """
    Synthetic VARIABLE-class light curve.
    - Sinusoidal stellar pulsation: period=0.8 d, amplitude=2%
    - No box-transit signal
    """
    n = 1500
    time = _make_time_array(n_cadences=n)
    rng = np.random.default_rng(seed=13)
    noise = rng.normal(0, 0.0003, n)
    period = 0.8
    flux = 1.0 + 0.02 * np.sin(2 * np.pi * time / period) + noise
    flux_err = np.full(n, 0.0003)
    return _make_lc(time, flux, flux_err)


@pytest.fixture
def synthetic_noise_lc() -> LightCurve:
    """
    Synthetic NOISE-class light curve.
    - No periodic signal
    - Gaussian white noise at ~500 ppm
    - Flat baseline
    """
    n = 1500
    time = _make_time_array(n_cadences=n)
    rng = np.random.default_rng(seed=99)
    noise = rng.normal(0, 0.0005, n)
    flux = np.ones(n) + noise
    flux_err = np.full(n, 0.0005)
    return _make_lc(time, flux, flux_err)


@pytest.fixture
def all_synthetic_lcs(
    synthetic_planet_lc,
    synthetic_eb_lc,
    synthetic_variable_lc,
    synthetic_noise_lc,
) -> dict:
    """Convenience bundle of all 4 synthetic light curves keyed by class name."""
    return {
        "PLANET": synthetic_planet_lc,
        "EB": synthetic_eb_lc,
        "VARIABLE": synthetic_variable_lc,
        "NOISE": synthetic_noise_lc,
    }
