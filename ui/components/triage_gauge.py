# ui/components/triage_gauge.py
# ============================================================
# Phase 6.3 — Triage classification gauge component.
#
#   build_probability_figure()  horizontal bar of 4-class probabilities
#   render()                    Streamlit wrapper for Tab 3 (verdict + gauge)
#
# Colour scheme (config.CLASS_COLORS):
#   PLANET=blue  EB=red  VARIABLE=green  NOISE=gray  REVIEW=orange
# ============================================================

from __future__ import annotations

from typing import Dict

import numpy as np
import plotly.graph_objects as go

from config import CLASS_LABELS, CLASS_COLORS, CLASS_BLURB, REVIEW_PROB_THRESHOLD


def build_probability_figure(probabilities: Dict[str, float]):
    """
    Horizontal bar chart of class probabilities, coloured per class.
    REVIEW is appended as a marker row when the verdict is REVIEW.
    """
    labels = list(CLASS_LABELS)
    values = [float(probabilities.get(l, 0.0)) for l in labels]
    colors = [CLASS_COLORS.get(l, "#9ca3af") for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1%}" for v in values],
        textposition="auto",
    ))
    # Threshold line
    fig.add_vline(
        x=REVIEW_PROB_THRESHOLD, line=dict(color="#f97316", dash="dash"),
        annotation_text=f"review threshold {REVIEW_PROB_THRESHOLD:.2f}",
        annotation_position="top",
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1], title="Probability"),
        yaxis=dict(title=""),
        template="plotly_dark",
        height=300,
        margin=dict(l=80, r=20, t=30, b=40),
    )
    return fig


def verdict_from_prediction(prediction: Dict) -> str:
    """Return the effective class label, honouring the review flag."""
    if prediction is None:
        return "REVIEW"
    if prediction.get("review_flag"):
        return "REVIEW"
    return str(prediction.get("predicted_class", "REVIEW"))


def render(prediction: Dict | None, model_available: bool = True):
    """Render Tab 3: verdict banner + probability gauge + REVIEW indicator."""
    import streamlit as st

    st.subheader("Triage classification")

    if prediction is None:
        st.warning(
            "No trained model found at `models/`. Run "
            "`python scripts/train_classifier.py` first to enable triage. "
            "Detection and recycler diagnostics below do not require the model."
        )
        return

    verdict = verdict_from_prediction(prediction)
    color = CLASS_COLORS.get(verdict, "#9ca3af")
    blurb = CLASS_BLURB.get(verdict, "")

    # Verdict banner
    st.markdown(
        f"<div class='verdict' style='background:{color}22;border-color:{color};"
        f"color:{color}'>Verdict: {verdict}</div>",
        unsafe_allow_html=True,
    )
    st.caption(blurb)

    if verdict == "REVIEW":
        st.info(
            "⚠️ Classifier uncertainty is high — this target is routed for human "
            "review. Inspect the detection and recycler tabs before deciding."
        )

    # Metric cards
    probs = prediction.get("probabilities", {})
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predicted class", verdict)
    with c2:
        st.metric("Max probability", f"{prediction.get('max_probability', 0.0):.1%}")

    # Probability gauge
    fig = build_probability_figure(probs)
    st.plotly_chart(fig, use_container_width=True)

    # Full probability table
    st.markdown("**Class probabilities**")
    st.dataframe(
        {l: [f"{probs.get(l, 0.0):.3f}"] for l in CLASS_LABELS},
        use_container_width=True,
    )
