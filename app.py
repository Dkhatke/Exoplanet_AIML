# app.py — Exoplanet_AIML multi-tab triage dashboard (Phase 6)
# ============================================================
# Replaces the legacy single-star demo. Orchestrates the modular
# pipeline and renders 5 tabs:
#   1. Multi-sector light curves
#   2. Signal detection (BLS)
#   3. Triage classification (4-class gauge)
#   4. Recycler diagnostics (contextual)
#   5. Provenance & report (JSON/HTML export)
#
# IMPORTANT (AC-6.3): this app NEVER trains a model. It only loads a
# pre-trained XGBoost model (models/xgb_model.json) and calls predict().
# ============================================================

import os

import numpy as np
import streamlit as st

from config import (
    STAR_NAME, MAX_SECTORS, CACHE_DIR, MODEL_DIR, CLASS_COLORS,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Exoplanet_AIML — Transit Triage",
    page_icon="🪐",
    layout="wide",
)

# ── Inject dark theme CSS ─────────────────────────────────────
_CSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "styles.css")
try:
    with open(_CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception as e:  # non-fatal
    log.warning("Could not load styles.css: %s", e)

st.title("🪐 Exoplanet_AIML — Transit Triage Pipeline")
st.markdown(
    "Detects and classifies periodic transit signals in NASA TESS light curves "
    "using **BLS + XGBoost**, with a false-positive recycler."
)


def resolve_tic_id(tic_input, star_name):
    """Return a TIC id from the numeric input, else resolve a star name."""
    if tic_input is not None and str(tic_input).strip() != "":
        return int(tic_input)
    if star_name and star_name.strip():
        import lightkurve as lk
        sr = lk.search_lightcurve(star_name.strip(), mission="TESS")
        if len(sr) == 0:
            raise ValueError(f"Could not resolve '{star_name}' to a TESS target.")
        for col in ("target_name", "targetid"):
            if col in sr.table.colnames:
                raw = str(sr.table[col][0])
                digits = "".join(ch for ch in raw if ch.isdigit())
                if digits:
                    return int(digits)
        raise ValueError(
            f"Resolved '{star_name}' but could not read a TIC id from the "
            "search result. Please enter the TIC ID directly."
        )
    raise ValueError("Provide a TIC ID or a star name.")


def model_exists():
    return os.path.exists(os.path.join(MODEL_DIR, "xgb_model.json"))


def run_pipeline(tic_id, max_sectors):
    """Full pipeline: download -> stitch -> detrend -> features."""
    from src.ingestion.fetcher import download_sectors
    from src.ingestion.stitcher import stitch_sectors
    from src.preprocessing.preprocessor import detrend_lc
    from src.features.feature_extractor import extract_features
    from src.ingestion.stellar_resolver import get_stellar_params_or_default

    sectors = download_sectors(tic_id, max_sectors=max_sectors, cache_dir=CACHE_DIR)
    stitched = stitch_sectors(sectors)
    lc_raw_norm, lc_detrended = detrend_lc(stitched)
    stellar = get_stellar_params_or_default(tic_id)
    feature_dict, bls_output, metadata = extract_features(lc_detrended, stellar_params=stellar)

    return {
        "tic_id": tic_id,
        "lc_raw_norm": lc_raw_norm,
        "lc_detrended": lc_detrended,
        "feature_dict": feature_dict,
        "bls_output": bls_output,
        "metadata": metadata,
        "stellar_params": stellar,
        "n_sectors": len(sectors),
    }


def run_triage(result):
    """Load pre-trained model + predict + route to recycler. No training."""
    from src.classifier.xgb_classifier import load_model, predict_single
    from src.recycler import route_to_recycler

    prediction = None
    recycler_output = None
    if model_exists():
        model = load_model()
        prediction = predict_single(result["feature_dict"], model=model)
        predicted_class = "REVIEW" if prediction["review_flag"] else prediction["predicted_class"]
        recycler_output = route_to_recycler(
            predicted_class,
            result["lc_detrended"], result["lc_raw_norm"], result["bls_output"],
            result["feature_dict"], stellar_params=result["stellar_params"],
            tic_id=result["tic_id"], class_probabilities=prediction["probabilities"],
        )
    return prediction, recycler_output


def _render_detection(bls_output: dict):
    """Tab 2: BLS periodogram + top candidate periods + harmonic flag."""
    import plotly.graph_objects as go

    st.subheader("BLS periodogram")
    periods = bls_output.get("periods")
    bls_result = bls_output.get("bls_result")
    if periods is not None and bls_result is not None:
        power = np.asarray(bls_result.power)
        fig = go.Figure(go.Scatter(x=np.asarray(periods), y=power, mode="lines",
                                   line=dict(color="#6ea8fe")))
        best = bls_output.get("corrected_period") or bls_output.get("best_period")
        fig.add_vline(x=best, line=dict(color="#ef4444", dash="dash"),
                      annotation_text=f"best {best:.4f} d")
        fig.update_layout(
            xaxis_title="Period (days)", yaxis_title="BLS power",
            template="plotly_dark", height=400,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No BLS periodogram available.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Best period (d)", f"{bls_output.get('corrected_period', 0):.5f}")
    c2.metric("Transit duration (d)", f"{bls_output.get('duration', 0):.4f}")
    c3.metric("BLS power", f"{bls_output.get('bls_power', 0):.4f}")

    if bls_output.get("harmonic_flag"):
        st.warning("⚠️ Harmonic flag raised — the reported period may be an alias "
                   "(P/2 or 2P). Inspect the phase-folded transit.")

    st.markdown("**Top candidate periods**")
    top_periods = bls_output.get("top3_periods")
    top_powers = bls_output.get("top3_powers")
    if top_periods is not None:
        st.dataframe({
            "Rank": [1, 2, 3],
            "Period (days)": [f"{p:.5f}" for p in list(top_periods)[:3]],
            "BLS power": [f"{p:.4f}" for p in list(top_powers)[:3]],
        }, use_container_width=True, hide_index=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Target")
    tic_input = st.text_input("TIC ID", value="", placeholder="e.g. 420814525")
    star_input = st.text_input("Star name (optional)", value=STAR_NAME)
    max_sectors = st.slider("Max sectors", 1, MAX_SECTORS, min(3, MAX_SECTORS))
    run_btn = st.button("🚀 Analyze", use_container_width=True)

    st.markdown("---")
    if model_exists():
        st.success("Pre-trained model: found")
    else:
        st.warning("No trained model — triage disabled. "
                   "Run `python scripts/train_classifier.py`.")
    st.caption("Data: NASA TESS via MAST (lightkurve)")
    st.caption("This app never trains a model at runtime.")


# ── Run analysis (cached in session state) ────────────────────
if run_btn:
    try:
        with st.spinner("📡 Resolving target and downloading TESS data…"):
            tic_id = resolve_tic_id(tic_input, star_input)

        with st.spinner(f"🔬 Running pipeline on TIC {tic_id} ({max_sectors} sector(s))…"):
            result = run_pipeline(tic_id, max_sectors)
            prediction, recycler_output = run_triage(result)

        st.session_state["result"] = result
        st.session_state["prediction"] = prediction
        st.session_state["recycler"] = recycler_output
        st.session_state["star_name"] = star_input
        st.success("✅ Analysis complete.")
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        log.exception("Pipeline failed")


# ── Tabs ──────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.info("👈 Enter a TIC ID (or star name) in the sidebar and click **Analyze**.")
else:
    result = st.session_state["result"]
    prediction = st.session_state.get("prediction")
    recycler_output = st.session_state.get("recycler")

    from ui.components import lightcurve_view, triage_gauge, recycler_view, report_exporter

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Light Curves",
        "🔍 Signal Detection",
        "🧭 Triage",
        "♻️ Recycler",
        "📄 Report",
    ])

    with tab1:
        lightcurve_view.render(result["lc_raw_norm"], result["lc_detrended"], result["bls_output"])

    with tab2:
        _render_detection(result["bls_output"])

    with tab3:
        triage_gauge.render(prediction, model_available=model_exists())

    with tab4:
        recycler_view.render(recycler_output, lc_raw_norm=result["lc_raw_norm"])

    with tab5:
        report_exporter.render_downloads(
            result["tic_id"], st.session_state.get("star_name"),
            result["feature_dict"], result["metadata"],
            prediction, recycler_output, result["stellar_params"],
        )


