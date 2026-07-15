# Day 3 — Streamlit Frontend: Build & Test Notes

**Files delivered:** `app.py`, `pipeline.py`, `visual_helpers.py`, `requirements.txt`
**Location in your repo:** all four go in `/frontend`

---

## What each file does

- **`app.py`** — the actual Streamlit app. Sidebar: image upload, environmental-vector
  sliders/inputs, language dropdown, Run button. Main area: live agent-log panel (streams
  node-by-node via `graph.stream()`), HITL banner, side-by-side attention-heatmap + SHAP
  chart (your 30-second judge hook), agent interpretations, localized report text, audio
  player, and a "raw state" debug expander.
- **`pipeline.py`** — adapts your LangGraph graph to take real UI input (uploaded image +
  form values) instead of the `mock_vision_case`/`mock_tabular_case` selectors used for
  testing. Tries real inference first (`vision_pipeline`, `tabular_analytics`,
  `agent_workflow`); if any of those aren't importable, it falls back to Day 1 canonical
  demo data automatically — **the UI will never crash from a missing module**, it just
  clearly flags `used_fallback: true` and keeps demoing.
- **`visual_helpers.py`** — the attention-overlay (3-panel: original/heatmap/overlay) and
  SHAP bar chart renderers, plus a small risk gauge. Pure matplotlib, returns `Figure`
  objects for `st.pyplot()`.

---

## What's verified vs. not

**Verified in sandbox:**
- `visual_helpers.py` — all 3 chart functions run correctly on mock data (attention overlay, SHAP bar chart, risk gauge)
- `pipeline.py` — full graph runs end-to-end via `build_graph()`, tested twice:
  1. With no `agent_workflow` package importable → correctly falls back to demo data + fallback translation/audio messages, no crash
  2. With `agent_workflow` importable → correctly uses your real `translation_agent_node`/`audio_agent_node` (confirmed via mock-mode output showing up, not fallback text)
- `app.py` — boots cleanly under `streamlit run` with no import/syntax errors in server logs (checked via headless server + health endpoint)

**Not verified — needs your Colab/local session with the full repo assembled:**
- The actual "Run Pipeline" button click path through a real browser (Streamlit runs via
  websocket; I can't simulate a real click from this sandbox, only test that every function
  the click handler calls works standalone — which I did)
- Real vision/tabular inference (needs your ViT + XGBoost artifacts alongside this)
- Real NLLB translation + gTTS audio (needs Hugging Face + Google API network access, which this sandbox doesn't have but Colab/your machine does)

---

## How to run it

**Locally (recommended for the live demo, not Colab — Streamlit + Colab don't mix well):**

1. Assemble your full repo:
   ```
   /vision_pipeline/inference.py
   /tabular_analytics/{inference.py, schema.py, model.json, explainer.pkl, risk_heuristic.py}
   /agent_workflow/{translation_agent.py, audio_agent.py, language_config.py, translation_schema.py, audio_schema.py, __init__.py}
   /frontend/{app.py, pipeline.py, visual_helpers.py, requirements.txt}
   ```
   `agent_workflow` needs an `__init__.py` (even an empty one) so `pipeline.py`'s
   `from agent_workflow.translation_agent import ...` resolves correctly.

2. Set `MOCK_MODE = False` in `translation_agent.py` and `audio_agent.py` (per the earlier
   handoff docs) once you're ready for real NLLB/gTTS output.

3. Set your `GROQ_API_KEY` environment variable (`export GROQ_API_KEY=...` or a `.env` file)
   so `pipeline.py`'s Pathologist/Climate agents use real Groq calls instead of their
   built-in canned fallback text.

4. From the repo root:
   ```bash
   pip install -r frontend/requirements.txt
   # plus: pip install transformers torch xgboost shap scikit-learn pandas sentencepiece gTTS groq
   cd frontend
   streamlit run app.py
   ```

5. Open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## One thing to decide before the demo

Right now, if a real module is missing OR real inference throws an exception, `pipeline.py`
silently substitutes fallback data and just shows a small warning banner
(`used_fallback: true`). This is deliberately safe for live judging — nothing crashes on
stage. But double-check before the pitch that you *want* silent fallback rather than a hard
error, in case a judge asks "wait, is this using my uploaded image or not?" — the debug
expander at the bottom of the page (`Raw pipeline state`) will always tell you the truth,
so glance at that during rehearsal.

---

*Next: side-by-side render is already built into this app (judge hook is live). Remaining
Day 3 items: edge-case testing (bad image, missing fields), backup demo video, pitch prep.*
