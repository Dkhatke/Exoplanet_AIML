# ui/components/recycler_view.py
# ============================================================
# Phase 6.4 — Contextual recycler diagnostics panel.
#
# Renders a different panel depending on the predicted class:
#   PLANET   -> physical parameter table
#   EB       -> primary/secondary depth comparison + morphology
#   VARIABLE -> Lomb-Scargle periodogram + variability type
#   NOISE    -> CDPP table + instrumental flags
#   REVIEW   -> uncertainty explanation + top-2 competing classes
#
# Figure builders are pure (offline-testable); render() wraps Streamlit.
# ============================================================

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import plotly.graph_objects as go


def build_ls_periodogram(lc, period_range=(0.1, 50.0)):
    """Lomb-Scargle periodogram of a light curve (for the VARIABLE panel)."""
    t = lc.time.value if hasattr(lc.time, "value") else np.asarray(lc.time)
    f = lc.flux.value if hasattr(lc.flux, "value") else np.asarray(lc.flux)
    from astropy.timeseries import LombScargle
    ls = LombScargle(t, f - np.nanmedian(f))
    freq, power = ls.autopower(
        minimum_frequency=1.0 / period_range[1],
        maximum_frequency=1.0 / period_range[0],
        samples_per_peak=10,
    )
    fig = go.Figure(go.Scatter(x=1.0 / freq, y=power, mode="lines",
                               line=dict(color="#22c55e")))
    fig.update_layout(
        title="Lomb-Scargle periodogram",
        xaxis_title="Period (days)", yaxis_title="Power",
        template="plotly_dark", height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_eb_depth_bar(eb: Dict[str, Any]):
    """Primary vs secondary eclipse depth bar chart (EB panel)."""
    fig = go.Figure(go.Bar(
        x=["Primary", "Secondary"],
        y=[eb.get("primary_depth", 0.0), eb.get("secondary_depth", 0.0)],
        marker=dict(color=["#ef4444", "#f97316"]),
        text=[f"{eb.get('primary_depth', 0):.4f}", f"{eb.get('secondary_depth', 0):.4f}"],
        textposition="auto",
    ))
    fig.update_layout(
        title="Eclipse depths", yaxis_title="Depth (fraction)",
        template="plotly_dark", height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig



def render(recycler_output: Dict[str, Any] | None, lc_raw_norm=None):
    """Render Tab 4: contextual diagnostics for the predicted class."""
    import streamlit as st

    st.subheader("Recycler diagnostics")

    if not recycler_output:
        st.info("No recycler output available.")
        return

    module = recycler_output.get("module")
    details = recycler_output.get("details", {}) or {}
    predicted = recycler_output.get("predicted_class", "")

    if module == "error":
        st.error(f"Recycler module failed: {recycler_output.get('error')}")
        return

    # -- PLANET -------------------------------------------------
    if predicted == "PLANET":
        st.markdown("#### Planet candidate")
        st.dataframe({
            "Parameter": ["Period (d)", "Rp/R*", "Rp (Earth radii)",
                          "Semi-major axis (AU)", "Equilibrium T (K)",
                          "Measured duration (d)", "Duration consistent?"],
            "Value": [
                f"{details.get('period_days', 0):.5f}",
                f"{details.get('rp_over_rs', 0):.4f}",
                f"{details.get('rp_earth_radii', 0):.2f}",
                f"{details.get('semi_major_axis_au', 0):.4f}",
                f"{details.get('eq_temperature_k', 0):.0f}",
                f"{details.get('measured_duration_days', 0):.4f}",
                "yes" if details.get("duration_consistent") else "no",
            ],
        }, use_container_width=True, hide_index=True)
        for n in details.get("notes", []):
            st.caption(f"- {n}")

    # -- EB -----------------------------------------------------
    elif predicted == "EB":
        st.markdown("#### Eclipsing binary")
        st.plotly_chart(build_eb_depth_bar(details), use_container_width=True)
        st.dataframe({
            "Diagnostic": ["Subtype", "Secondary detected?", "Secondary phase",
                           "Odd/even ratio", "Odd/even significant?",
                           "V-shape metric", "Period doubling?",
                           "Reported period (d)"],
            "Value": [
                details.get("eb_subtype", "?"),
                "yes" if details.get("secondary_detected") else "no",
                f"{details.get('secondary_phase', 0):.3f}",
                f"{details.get('odd_even_ratio', 0):.3f}",
                "yes" if details.get("odd_even_significant") else "no",
                f"{details.get('v_shape_metric', 0):.3f}",
                "yes" if details.get("period_doubling_candidate") else "no",
                f"{details.get('reported_period_days', details.get('period_days', 0)):.4f}",
            ],
        }, use_container_width=True, hide_index=True)

    # -- VARIABLE -----------------------------------------------
    elif predicted == "VARIABLE":
        st.markdown("#### Variable star")
        if lc_raw_norm is not None:
            st.plotly_chart(build_ls_periodogram(lc_raw_norm), use_container_width=True)
        st.dataframe({
            "Property": ["Mode", "Dominant period (d)", "LS peak power",
                         "LS FAP", "Amplitude (p5-p95)", "Possible types"],
            "Value": [
                details.get("mode", "?"),
                f"{details.get('dominant_period_days', 0):.4f}",
                f"{details.get('ls_peak_power', 0):.3f}",
                f"{details.get('ls_fap', 0):.2e}",
                f"{details.get('amplitude_p5_p95', 0):.4f}",
                ", ".join(details.get("possible_types", [])),
            ],
        }, use_container_width=True, hide_index=True)

    # -- NOISE --------------------------------------------------
    elif predicted == "NOISE":
        st.markdown("#### Noise / instrumental")
        st.dataframe({
            "Metric": ["CDPP 1hr (ppm)", "CDPP 6hr (ppm)", "LF/HF ratio",
                       "Systematic flag", "Instrument-period flag", "Baseline (d)"],
            "Value": [
                f"{details.get('cdpp_1hr_ppm', 0):.1f}",
                f"{details.get('cdpp_6hr_ppm', 0):.1f}",
                f"{details.get('lf_over_hf_ratio', 0):.2f}",
                "YES" if details.get("systematic_flag") else "no",
                "YES" if details.get("instrument_period_flag") else "no",
                f"{details.get('baseline_days', 0):.1f}",
            ],
        }, use_container_width=True, hide_index=True)

    # -- REVIEW -------------------------------------------------
    elif predicted == "REVIEW":
        st.markdown("#### Needs human review")
        st.info(details.get("reason", "Classifier confidence below threshold."))
        c1, c2, c3 = st.columns(3)
        c1.metric("Max probability", f"{details.get('max_probability', 0):.1%}")
        c2.metric("Runner-up", f"{details.get('runner_up_probability', 0):.1%}")
        c3.metric("Margin", f"{details.get('probability_margin', 0):.3f}")
        probs = details.get("probabilities", {})
        if probs:
            top2 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:2]
            st.markdown("**Top-2 competing classes:** " +
                        ", ".join(f"{k} ({v:.1%})" for k, v in top2))

    else:
        st.json(details)
