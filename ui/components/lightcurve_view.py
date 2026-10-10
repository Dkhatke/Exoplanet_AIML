# ui/components/lightcurve_view.py
# ============================================================
# Phase 6.2 — Light curve visualisation component.
#
# Provides:
#   build_lightcurve_figure()  raw-normalised vs detrended flux (Plotly)
#   build_phase_fold_figure()  phase-folded transit + binned median
#   render()                   Streamlit wrapper for Tab 1
#
# The figure builders are pure (no Streamlit) so they can be unit
# tested offline.
# ============================================================

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _flux(lc) -> np.ndarray:
    return lc.flux.value if hasattr(lc.flux, "value") else np.asarray(lc.flux)


def _time(lc) -> np.ndarray:
    return lc.time.value if hasattr(lc.time, "value") else np.asarray(lc.time)


def _sector_of(lc) -> Optional[np.ndarray]:
    """Best-effort per-cadence sector array if the LightCurve carries one."""
    if hasattr(lc, "sector") and lc.sector is not None:
        s = lc.sector.value if hasattr(lc.sector, "value") else np.asarray(lc.sector)
        if len(s) == len(lc):
            return s
    return None


def build_lightcurve_figure(lc_raw_norm, lc_detrended, title: str = "Light curve"):
    """
    Overlay raw-normalised flux (with stellar trend) and the detrended flux.

    Returns a plotly Figure. If a per-cadence sector array is available the
    raw points are coloured by sector.
    """
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    t_raw, f_raw = _time(lc_raw_norm), _flux(lc_raw_norm)
    t_det, f_det = _time(lc_detrended), _flux(lc_detrended)

    sectors = _sector_of(lc_raw_norm)
    if sectors is not None and len(np.unique(sectors)) > 1:
        for sec in np.unique(sectors):
            m = sectors == sec
            fig.add_trace(go.Scattergl(
                x=t_raw[m], y=f_raw[m], mode="markers",
                marker=dict(size=2, opacity=0.5),
                name=f"raw (sector {int(sec)})",
            ))
    else:
        fig.add_trace(go.Scattergl(
            x=t_raw, y=f_raw, mode="markers",
            marker=dict(size=2, color="#9aa4c0", opacity=0.5),
            name="raw normalised",
        ))

    fig.add_trace(go.Scattergl(
        x=t_det, y=f_det, mode="markers",
        marker=dict(size=2, color="#6ea8fe"),
        name="detrended",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (BTJD days)",
        yaxis_title="Normalised flux",
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_phase_fold_figure(
    lc_detrended,
    period: float,
    t0: float,
    duration: float,
    n_bins: int = 100,
    title: Optional[str] = None,
):
    """
    Phase-fold the detrended curve at (period, t0); overlay scatter,
    binned median, and a shaded transit window of +/- duration/2.
    """
    t, f = _time(lc_detrended), _flux(lc_detrended)
    phase = ((t - t0) % period)
    phase[phase > period / 2] -= period
    phase = phase / period  # normalise to [-0.5, 0.5]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=phase, y=f, mode="markers",
        marker=dict(size=2, color="#9aa4c0", opacity=0.35),
        name="folded data",
    ))

    # Binned median
    bins = np.linspace(-0.5, 0.5, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.clip(np.digitize(phase, bins) - 1, 0, n_bins - 1)
    medians = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if np.any(m):
            medians[b] = np.median(f[m])
    fig.add_trace(go.Scatter(
        x=centers, y=medians, mode="lines",
        line=dict(color="#ef4444", width=2),
        name="binned median",
    ))

    # Transit window (+/- duration/2 in phase units)
    half_phase = 0.5 * duration / period
    fig.add_vrect(
        x0=-half_phase, x1=half_phase,
        fillcolor="#22d3ee", opacity=0.12, line_width=0,
        annotation_text="transit", annotation_position="top left",
    )

    fig.update_layout(
        title=title or f"Phase-folded transit (P = {period:.5f} d)",
        xaxis_title="Orbital phase",
        yaxis_title="Normalised flux",
        template="plotly_dark",
        height=420,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ── Streamlit render (Tab 1) ─────────────────────────────────

def render(lc_raw_norm, lc_detrended, bls_output: dict):
    """Render Tab 1: raw vs detrended + phase-folded transit."""
    import streamlit as st

    st.subheader("Multi-sector light curve")
    st.caption("Raw normalised flux (stellar trend present) vs transit-preserving detrended flux.")
    fig_lc = build_lightcurve_figure(lc_raw_norm, lc_detrended)
    st.plotly_chart(fig_lc, use_container_width=True)

    period = float(bls_output.get("corrected_period") or bls_output.get("best_period", 1.0))
    t0 = float(bls_output.get("t0", 0.0))
    duration = float(bls_output.get("duration", 0.1))
    st.subheader("Phase-folded transit")
    fig_pf = build_phase_fold_figure(lc_detrended, period, t0, duration)
    st.plotly_chart(fig_pf, use_container_width=True)
