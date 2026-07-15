"""
Agrospheric AI — Streamlit demo UI.

Run with: streamlit run app.py

Folder layout this expects (matches your repo structure):
  /vision_pipeline/inference.py
  /tabular_analytics/{inference.py, schema.py}
  /agent_workflow/{translation_agent.py, audio_agent.py, language_config.py, ...}
  /frontend/{app.py, pipeline.py, visual_helpers.py}   <- this file lives here

If vision_pipeline / tabular_analytics / agent_workflow aren't importable (e.g.
you're running this file standalone to test the UI shell), pipeline.py falls
back to Day 1 canonical demo data automatically — the UI will never crash, it
will just clearly label itself as running on fallback data via `used_fallback`.
"""

import sys
import os
import json
import time

import streamlit as st
from PIL import Image

# --- Make sibling repo folders importable ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for subfolder in ["", "agent_workflow", "vision_pipeline", "tabular_analytics"]:
    p = os.path.join(REPO_ROOT, subfolder) if subfolder else REPO_ROOT
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline import build_graph
from visual_helpers import render_attention_overlay, render_shap_bar_chart, render_risk_gauge

try:
    from language_config import language_choices
except Exception:
    def language_choices():
        return ["Hindi", "Marathi", "Tamil", "Telugu", "Bengali", "Gujarati",
                "Kannada", "Punjabi", "Malayalam", "Urdu", "English"]

st.set_page_config(page_title="Agrospheric AI", layout="wide")

NODE_LABELS = {
    "vision": "🔍 Vision Node — classifying leaf image",
    "analytical": "🌡️ Analytical Node — computing 14-day risk",
    "pathologist_agent": "🩺 Pathologist Agent — interpreting diagnosis",
    "climate_agent": "🌦️ Climate Agent — interpreting risk factors",
    "ambiguous_review": "⚠️ HITL Gate — routing to human review",
    "proceed_to_translation": "✅ HITL Gate — cleared, proceeding",
    "translation_agent": "🌐 Translation Agent — localizing report",
    "audio_agent": "🔊 Audio Agent — synthesizing speech",
}

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
st.sidebar.title("🌾 Agrospheric AI")
st.sidebar.caption("Multi-Agent Crop Intelligence Pipeline")

st.sidebar.subheader("1. Leaf Image")
uploaded_image = st.sidebar.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])

st.sidebar.subheader("2. Environmental Vector")
temperature_c = st.sidebar.slider("Temperature (°C)", 5.0, 45.0, 27.5)
humidity_pct = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 88.0)
rainfall_mm_14d = st.sidebar.slider("Rainfall, 14d (mm)", 0.0, 150.0, 95.0)
soil_n = st.sidebar.number_input("Soil Nitrogen (kg/ha)", 0.0, 200.0, 70.0)
soil_p = st.sidebar.number_input("Soil Phosphorus (kg/ha)", 0.0, 100.0, 30.0)
soil_k = st.sidebar.number_input("Soil Potassium (kg/ha)", 0.0, 150.0, 45.0)
soil_moisture_pct = st.sidebar.slider("Soil Moisture (%)", 5.0, 95.0, 62.0)

st.sidebar.subheader("3. Report Language")
target_language = st.sidebar.selectbox("Translate report to", language_choices(), index=0)

run_clicked = st.sidebar.button("▶ Run Pipeline", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Precision Phytopathology & Risk Mitigation")
st.caption("Fusing computer vision leaf diagnostics with environmental risk analytics — explainable, localized, actionable.")

if not run_clicked:
    st.info("Upload a leaf image, set the environmental readings in the sidebar, and click **Run Pipeline**.")
    st.stop()

# --- Build initial state ---
pil_image = None
if uploaded_image is not None:
    pil_image = Image.open(uploaded_image).convert("RGB")

initial_state = {
    "image": pil_image,
    "tabular_input": {
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "rainfall_mm_14d": rainfall_mm_14d,
        "soil_n": soil_n,
        "soil_p": soil_p,
        "soil_k": soil_k,
        "soil_moisture_pct": soil_moisture_pct,
    },
    "target_language": target_language,
}

graph = build_graph()

# ---------------------------------------------------------------------------
# Live agent-log panel — streams node-by-node via graph.stream()
# ---------------------------------------------------------------------------
st.subheader("🖥️ Live Agent Log")
log_container = st.container(border=True)
log_lines = []
log_placeholder = log_container.empty()

final_state = {}
with st.spinner("Running multi-agent pipeline..."):
    for step in graph.stream(initial_state):
        for node_name, node_output in step.items():
            label = NODE_LABELS.get(node_name, node_name)
            log_lines.append(f"`{time.strftime('%H:%M:%S')}` {label} — done")
            log_placeholder.markdown("\n\n".join(log_lines))
            final_state.update(node_output)

st.success("Pipeline complete.")

if final_state.get("used_fallback"):
    st.warning(
        "⚠️ Real vision/tabular inference modules weren't available in this session — "
        "showing Day 1 canonical demo data instead. Wire in `vision_pipeline`/`tabular_analytics` "
        "for live inference."
    )

# ---------------------------------------------------------------------------
# HITL banner
# ---------------------------------------------------------------------------
if final_state.get("route") == "ambiguous":
    st.error(
        "🛑 **HITL Gate triggered — Ambiguous Case.** "
        "Vision Node confidence was below the 75% threshold. This case has been flagged "
        "for human agronomist review. No automated treatment recommendation is provided."
    )

# ---------------------------------------------------------------------------
# Side-by-side judge hook: attention heatmap + SHAP chart
# ---------------------------------------------------------------------------
st.subheader("🔬 Explainability — Vision + Environmental Attribution")
col1, col2 = st.columns(2)

vision_output = final_state.get("vision_output", {})
analytical_output = final_state.get("analytical_output", {})

with col1:
    st.markdown(f"**Classification:** `{vision_output.get('disease_class', 'N/A')}`  \n"
                f"**Confidence:** {vision_output.get('confidence_score', 0)*100:.1f}%")
    if pil_image is not None and vision_output.get("attention_matrix"):
        fig = render_attention_overlay(pil_image, vision_output["attention_matrix"])
        st.pyplot(fig)
    else:
        st.caption("Upload an image to see the attention rollout overlay.")

with col2:
    if analytical_output:
        st.markdown(f"**14-day risk:** {analytical_output.get('14_day_risk_pct', 0):.1f}% "
                    f"({analytical_output.get('risk_band', 'N/A')})")
        st.pyplot(render_risk_gauge(
            analytical_output.get("14_day_risk_pct", 0),
            analytical_output.get("risk_band", "Low"),
        ))
        if analytical_output.get("top_3_environmental_stressors"):
            st.pyplot(render_shap_bar_chart(analytical_output["top_3_environmental_stressors"]))

# ---------------------------------------------------------------------------
# Interpretations + localized report + audio
# ---------------------------------------------------------------------------
st.subheader("📋 Agent Interpretations")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Pathologist Agent**")
    st.write(final_state.get("pathologist_interpretation", "—"))
with c2:
    st.markdown("**Climate Agent**")
    st.write(final_state.get("climate_interpretation", "—"))

st.subheader(f"🌐 Localized Report — {target_language}")
st.write(final_state.get("translated_text", "—"))

audio_output = final_state.get("audio_output", {})
if audio_output.get("tts_status") == "success" and audio_output.get("audio_file_path"):
    st.audio(audio_output["audio_file_path"])
elif audio_output.get("tts_status") == "skipped_unsupported_language":
    st.info(f"🔇 Audio not available for {target_language}: {audio_output.get('skip_reason')}")
elif audio_output.get("tts_status") == "failed":
    st.warning(f"Audio synthesis failed: {audio_output.get('skip_reason')}")

with st.expander("Raw pipeline state (debug)"):
    st.json({k: v for k, v in final_state.items() if k != "image"})
