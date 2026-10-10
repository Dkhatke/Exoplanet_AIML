# ui/components/report_exporter.py
# ============================================================
# Phase 6.5 — Report export (JSON + HTML).
#
#   build_json_report()   -> dict (JSON-serialisable)
#   build_html_report()   -> str (self-contained HTML summary)
#   render_downloads()    -> Streamlit download buttons (Tab 5)
#
# The builders are pure functions with no Streamlit dependency, so they
# are unit-tested offline.
# ============================================================

from __future__ import annotations

import json
from typing import Dict, Any, Optional


def _json_safe(obj):
    """Recursively convert numpy / non-finite values to JSON-safe types."""
    import math
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def build_json_report(
    tic_id: Optional[int],
    star_name: Optional[str],
    feature_dict: Dict[str, float],
    metadata: Dict[str, Any],
    prediction: Optional[Dict[str, Any]],
    recycler_output: Optional[Dict[str, Any]],
    stellar_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the full JSON report dict (JSON-serialisable)."""
    report = {
        "target": {
            "tic_id": tic_id,
            "star_name": star_name,
            "stellar_params": stellar_params or {},
        },
        "detection": {
            "bls_period": metadata.get("bls_period"),
            "bls_t0": metadata.get("bls_t0"),
            "bls_duration": metadata.get("bls_duration"),
            "bls_power": metadata.get("bls_power"),
            "harmonic_flag": metadata.get("harmonic_flag"),
            "n_cadences": metadata.get("n_cadences"),
        },
        "features": feature_dict,
        "triage": prediction,
        "recycler": recycler_output,
        "provenance": {
            "extraction_time_s": metadata.get("extraction_time_s"),
            "warnings": metadata.get("warnings", []),
        },
    }
    return _json_safe(report)


def build_html_report(
    tic_id: Optional[int],
    star_name: Optional[str],
    prediction: Optional[Dict[str, Any]],
    recycler_output: Optional[Dict[str, Any]],
) -> str:
    """Return a small self-contained HTML summary string."""
    from config import CLASS_COLORS
    verdict = "REVIEW"
    if prediction:
        verdict = "REVIEW" if prediction.get("review_flag") else str(
            prediction.get("predicted_class", "REVIEW"))
    color = CLASS_COLORS.get(verdict, "#9ca3af")
    maxp = f"{prediction.get('max_probability', 0):.1%}" if prediction else "n/a"

    probs_rows = ""
    if prediction and prediction.get("probabilities"):
        for cls, p in prediction["probabilities"].items():
            probs_rows += f"<tr><td>{cls}</td><td>{p:.3f}</td></tr>"

    recycler_summary = ""
    if recycler_output and recycler_output.get("details"):
        recycler_summary = "<pre style='background:#131a2e;padding:10px;border-radius:8px;" \
                           "overflow:auto'>" + json.dumps(
            recycler_output["details"], indent=2, default=str)[:4000] + "</pre>"

    return f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>Exoplanet_AIML report</title></head>
<body style="background:#0b1020;color:#e8ecf8;font-family:system-ui,sans-serif;padding:24px">
  <h1>Exoplanet_AIML — triage report</h1>
  <p class="dim">TIC {tic_id} &middot; {star_name or 'name not provided'}</p>
  <div style="border:1px solid {color};background:{color}22;color:{color};
              padding:14px;border-radius:12px;font-weight:700;font-size:1.2rem">
      Verdict: {verdict} &nbsp;(max probability {maxp})
  </div>
  <h3>Class probabilities</h3>
  <table border="0" cellpadding="6" style="border-collapse:collapse">
    <tr style="color:#9aa4c0"><th align="left">Class</th><th align="left">Probability</th></tr>
    {probs_rows}
  </table>
  <h3>Recycler diagnostics</h3>
  {recycler_summary or '<p class="dim">none</p>'}
</body></html>
"""


def render_downloads(
    tic_id, star_name, feature_dict, metadata, prediction, recycler_output, stellar_params
):
    """Streamlit Tab 5 download buttons for JSON + HTML reports."""
    import streamlit as st

    st.subheader("Provenance & report")

    json_report = build_json_report(
        tic_id, star_name, feature_dict, metadata, prediction, recycler_output, stellar_params
    )
    html_report = build_html_report(tic_id, star_name, prediction, recycler_output)

    st.download_button(
        "Download JSON report",
        data=json.dumps(json_report, indent=2),
        file_name=f"tic_{tic_id}_report.json",
        mime="application/json",
    )
    st.download_button(
        "Download HTML report",
        data=html_report,
        file_name=f"tic_{tic_id}_report.html",
        mime="text/html",
    )

    st.markdown("**Provenance**")
    st.json({
        "tic_id": tic_id,
        "star_name": star_name,
        "bls_period": metadata.get("bls_period"),
        "bls_power": metadata.get("bls_power"),
        "n_cadences": metadata.get("n_cadences"),
        "extraction_time_s": metadata.get("extraction_time_s"),
        "warnings": metadata.get("warnings", []),
    })
