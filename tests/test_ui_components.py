# tests/test_ui_components.py
# ============================================================
# Phase 6 acceptance tests: Streamlit UI component logic.
# ALL tests run offline. We test the PURE functions (report builders,
# figure builders, verdict logic, colour scheme) — never the st.*
# rendering, which needs a browser runtime.
#
# Covers:
#   AC-6.4  JSON report contains all required fields + is serialisable
#   AC-6.5  REVIEW state has a distinct colour/indicator
# ============================================================

import json

import numpy as np
import pytest
from lightkurve import LightCurve
import astropy.units as u
from astropy.time import Time


def _make_lc(n=1500, period=3.524, depth=0.015, duration=0.09, seed=42):
    cadence = 10.0 / 60 / 24
    t = 1325.0 + np.arange(n) * cadence
    rng = np.random.default_rng(seed)
    flux = 1.0 + rng.normal(0, 0.0005, n)
    t0 = t[0] + 0.5
    ph = ((t - t0) % period); ph[ph > period / 2] -= period
    flux[np.abs(ph) < 0.5 * duration] -= depth
    return LightCurve(
        time=Time(t, format="btjd", scale="tdb"),
        flux=flux * u.dimensionless_unscaled,
        flux_err=np.full(n, 0.0005) * u.dimensionless_unscaled,
    )


class TestReportExporter:

    def test_json_report_has_required_sections(self):
        from ui.components.report_exporter import build_json_report
        rep = build_json_report(
            tic_id=123, star_name="HD 209458",
            feature_dict={"depth": 0.015, "snr": 12.0},
            metadata={"bls_period": 3.5, "bls_power": 0.9, "n_cadences": 1500,
                      "warnings": ["x"]},
            prediction={"predicted_class": "PLANET", "max_probability": 0.9,
                        "probabilities": {"PLANET": 0.9}, "review_flag": False},
            recycler_output={"module": "planet_module", "details": {"rp_earth_radii": 2.8}},
            stellar_params={"radius_solar": 1.1},
        )
        for key in ("target", "detection", "features", "triage", "recycler", "provenance"):
            assert key in rep, f"missing section: {key}"
        assert rep["target"]["tic_id"] == 123
        assert rep["features"]["depth"] == 0.015
        assert rep["triage"]["predicted_class"] == "PLANET"

    def test_json_report_is_serialisable(self):
        from ui.components.report_exporter import build_json_report
        rep = build_json_report(
            tic_id=1, star_name="x",
            feature_dict={"depth": float("nan"), "snr": np.float64(3.0)},
            metadata={"bls_period": np.float32(2.0), "n_cadences": np.int64(100)},
            prediction=None, recycler_output=None, stellar_params={},
        )
        s = json.dumps(rep)  # must not raise
        assert isinstance(s, str)
        assert rep["features"]["depth"] is None  # NaN -> null

    def test_html_report_contains_verdict(self):
        from ui.components.report_exporter import build_html_report
        html = build_html_report(
            tic_id=42, star_name="Test",
            prediction={"predicted_class": "EB", "max_probability": 0.8,
                        "probabilities": {"EB": 0.8}, "review_flag": False},
            recycler_output={"details": {"eb_subtype": "DETACHED"}},
        )
        assert "Verdict: EB" in html
        assert "TIC 42" in html


class TestTriageGauge:

    def test_verdict_review_when_flag_set(self):
        from ui.components.triage_gauge import verdict_from_prediction
        assert verdict_from_prediction({"review_flag": True, "predicted_class": "PLANET"}) == "REVIEW"

    def test_verdict_class_when_confident(self):
        from ui.components.triage_gauge import verdict_from_prediction
        assert verdict_from_prediction({"review_flag": False, "predicted_class": "EB"}) == "EB"

    def test_verdict_review_when_none(self):
        from ui.components.triage_gauge import verdict_from_prediction
        assert verdict_from_prediction(None) == "REVIEW"

    def test_probability_figure_is_plotly(self):
        from ui.components.triage_gauge import build_probability_figure
        import plotly.graph_objects as go
        fig = build_probability_figure({"PLANET": 0.7, "EB": 0.1, "VARIABLE": 0.1, "NOISE": 0.1})
        assert isinstance(fig, go.Figure)


class TestColorScheme:

    def test_all_five_classes_have_colors(self):
        from config import CLASS_COLORS, CLASS_LABELS
        for cls in CLASS_LABELS:
            assert cls in CLASS_COLORS, f"{cls} missing colour"
        assert "REVIEW" in CLASS_COLORS, "REVIEW must have a distinct colour"

    def test_review_color_distinct(self):
        from config import CLASS_COLORS
        others = {CLASS_COLORS[c] for c in ("PLANET", "EB", "VARIABLE", "NOISE")}
        assert CLASS_COLORS["REVIEW"] not in others, "REVIEW colour must be distinct"


class TestLightcurveView:

    def test_lightcurve_figure_builds(self):
        from ui.components.lightcurve_view import build_lightcurve_figure
        import plotly.graph_objects as go
        lc = _make_lc()
        fig = build_lightcurve_figure(lc, lc)
        assert isinstance(fig, go.Figure)

    def test_phase_fold_figure_builds(self):
        from ui.components.lightcurve_view import build_phase_fold_figure
        import plotly.graph_objects as go
        lc = _make_lc()
        fig = build_phase_fold_figure(lc, period=3.524, t0=float(lc.time.value[0]) + 0.5,
                                      duration=0.09)
        assert isinstance(fig, go.Figure)


class TestRecyclerView:

    def test_eb_depth_bar_builds(self):
        from ui.components.recycler_view import build_eb_depth_bar
        import plotly.graph_objects as go
        fig = build_eb_depth_bar({"primary_depth": 0.05, "secondary_depth": 0.02})
        assert isinstance(fig, go.Figure)

    def test_ls_periodogram_builds(self):
        from ui.components.recycler_view import build_ls_periodogram
        import plotly.graph_objects as go
        lc = _make_lc()
        fig = build_ls_periodogram(lc)
        assert isinstance(fig, go.Figure)

