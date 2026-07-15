# Agrospheric AI — Build Log: Translation Agent → Audio Agent → Streamlit Frontend

This README covers everything built after the Tabular/Analytical Node and initial
LangGraph orchestration — i.e. the localization stack (Translation + Audio) and the
demo-facing UI. For earlier work (Vision Node, Tabular Node, base LangGraph wiring),
see `day1_vision_node_results.md`, `day1_tabular_node_results.md`, and
`Tasks_of_day_2_completed_updated_.txt`.

---

## 1. Graph shape as of this build

```
vision -> analytical -> pathologist_agent -> climate_agent -> HITL router
                                                                  |
                                        +-------------------------+-------------------------+
                                        |                                                    |
                                   "ambiguous"                                          "proceed"
                                        v                                                    v
                              ambiguous_review                                  proceed_to_translation
                                        |                                                    |
                                        +-------------------------+-------------------------+
                                                                  v
                                                          translation_agent
                                                                  v
                                                             audio_agent
                                                                  v
                                                                 END
```

**Key design decision:** both HITL branches now flow through `translation_agent` and
`audio_agent`, not just the "proceed" path. A farmer whose photo was too ambiguous to
classify still gets a translated, spoken explanation of *why* — not silence. This directly
serves the "Accessibility Barrier" point in the problem statement, applied to every exit
of the graph, not just the happy path.

---

## 2. Translation Agent

**Files:** `agent_workflow/language_config.py`, `agent_workflow/translation_schema.py`, `agent_workflow/translation_agent.py`

**What it does:** translates the Pathologist/Climate agents' combined English output (or
the ambiguous-case review notice) into a farmer-selected regional language, using
**NLLB-200-distilled-600M** — chosen because it runs on CPU or a shared T4, per the
resource-constrained-optimization requirement in the problem statement.

**Dynamic language support:** 12 languages configured (11 Indian languages + English
pass-through), keyed by human-readable name so the UI dropdown never touches raw
FLORES-200 codes directly:

Hindi, Marathi, Tamil, Telugu, Bengali, Gujarati, Kannada, Punjabi, Malayalam, Odia,
Urdu, English.

**Node contract (`translation_agent_node`):**
- Reads: `text_to_translate`, `content_type` (`"action_plan"` or `"review_notice"`), `target_language`
- Writes: `translated_text`, `translation_output` (full `TranslationOutput` packet)
- Falls back to Hindi if `target_language` is missing or unrecognized — the graph never breaks on a missing UI selection

**Mock mode:** `MOCK_MODE = True` by default (this sandbox has no network route to
huggingface.co, so the real ~2.4GB model can't be downloaded/tested here). Flip to
`False` in Colab or your local machine — the real model loads lazily on first call and
stays cached for the session.

**Verified (mock mode, 4 cases):** routing unaffected by adding translation; Hindi/Tamil/Marathi
language codes resolve correctly; missing `target_language` correctly falls back to Hindi.
**Not verified here:** actual NLLB output quality — confirm the translated script looks
correct (not garbled) once you run it for real.

---

## 3. Audio Agent

**Files:** `agent_workflow/audio_schema.py`, `agent_workflow/audio_agent.py` (plus the gTTS map added to `language_config.py`)

**What it does:** synthesizes `translated_text` into an `.mp3` using **gTTS** (Google
Text-to-Speech), per the vernacular-audio-broadcast requirement in the problem statement.

**Real finding worth knowing:** gTTS does **not** support every language NLLB translates
to. Checked directly against `gtts.lang.tts_langs()` — **Odia is translatable but not
speakable.** Rather than crash or silently drop the language, the agent degrades
gracefully: translated *text* still reaches the report, audio is simply omitted with a
clear, structured reason (`tts_status: "skipped_unsupported_language"`). This is a legitimate
scoping statement for the pitch, not a bug to hide.

**Node contract (`audio_agent_node`):**
- Reads: `translated_text`, `target_language`, `content_type`
- Writes: `audio_output` (`AudioOutput` packet — `tts_status` is one of `success` /
  `skipped_unsupported_language` / `failed`, with `audio_file_path` or `skip_reason` set accordingly)
- Output filenames are deterministic: `{content_type}_{gtts_code}.mp3`, so downstream PDF
  assembly can predict the path without extra plumbing

**Mock mode:** `MOCK_MODE = True` by default (gTTS calls Google's TTS endpoint live over
the network at call time — also unavailable in this sandbox, unlike NLLB's one-time
download this needs network on *every* call). Flip to `False` in Colab/local, where gTTS
has real internet access.

**Verified (mock mode, full graph, 3 cases):**

| Case | Language | Result |
|---|---|---|
| Confident + Moderate | Hindi | `tts_status: success`, file generated |
| Ambiguous + High | Tamil | `tts_status: success`, file generated |
| Confident + Low | Odia | `tts_status: skipped_unsupported_language`, no crash, translated text preserved |

**Not verified here:** real audio quality/pronunciation, and gTTS live rate-limit behavior
under repeated testing (add `time.sleep()` between calls if you hit `429`s while testing).

---

## 4. Streamlit Frontend

**Files:** `frontend/app.py`, `frontend/pipeline.py`, `frontend/visual_helpers.py`, `frontend/requirements.txt`

**What it does:** the full demo UI — this is what a judge interacts with.

- **Sidebar:** leaf image upload, environmental-vector sliders/inputs (temperature,
  humidity, rainfall, N/P/K, soil moisture), target-language dropdown, Run button
- **Live Agent Log:** streams node-by-node via `graph.stream()` as the pipeline runs —
  satisfies the "intermediate logs terminal" requirement cheaply, without custom websocket work
- **HITL banner:** a clear red alert if the case routes to `ambiguous`
- **Judge hook:** side-by-side attention-rollout heatmap (3-panel: original/heatmap/overlay)
  next to a SHAP bar chart of the top 3 environmental stressors — built to visually lock in
  attention within the first 30 seconds, per the evaluation checklist
- **Report section:** Pathologist/Climate interpretations, the translated report text, and
  an inline audio player (or a clear "not available for this language" note)
- **Debug expander:** raw pipeline state, useful during rehearsal to confirm whether a run
  used real inference or fallback data

**`pipeline.py`'s key design decision:** it tries real inference (`vision_pipeline`,
`tabular_analytics`, `agent_workflow`) first, and **silently substitutes Day 1 canonical
demo data** if a module is missing or throws an exception — flagged via `used_fallback:
true` and a small warning banner, never a hard crash. This matters for **live judging**:
a broken import or a flaky API call mid-demo degrades to known-good numbers instead of
an on-stage stack trace.

**Verified:**
- All 3 chart functions in `visual_helpers.py` run correctly on mock data
- `pipeline.py`'s full graph runs end-to-end in **both** modes: with `agent_workflow`
  unavailable (correctly falls back) and with it available (correctly uses the real
  translation/audio nodes — confirmed by mock-mode output appearing instead of fallback text)
- `app.py` boots cleanly under `streamlit run` with zero import/syntax errors (checked via
  headless server + health-endpoint check)

**Not verified:** an actual button click through a real browser session (Streamlit runs
over websocket; not simulable from this sandbox) — mitigated by testing every function the
click handler calls, standalone, which all passed.

---

## 5. How to assemble and run all of this together

```
/vision_pipeline/inference.py
/tabular_analytics/{inference.py, schema.py, model.json, explainer.pkl, risk_heuristic.py}
/agent_workflow/{translation_agent.py, audio_agent.py, language_config.py,
                  translation_schema.py, audio_schema.py, __init__.py}
/frontend/{app.py, pipeline.py, visual_helpers.py, requirements.txt}
```

`agent_workflow` needs an `__init__.py` (can be empty) so `pipeline.py`'s
`from agent_workflow.translation_agent import ...` resolves.

1. Flip `MOCK_MODE = False` in `translation_agent.py` and `audio_agent.py` once you're
   ready for real NLLB/gTTS output.
2. Set `GROQ_API_KEY` in your environment so the Pathologist/Climate agents in
   `pipeline.py` use real Groq calls instead of canned fallback text.
3. From the repo root:
   ```bash
   pip install -r frontend/requirements.txt
   pip install transformers torch xgboost shap scikit-learn pandas sentencepiece gTTS groq
   cd frontend
   streamlit run app.py
   ```
4. Open the local URL Streamlit prints.

Run locally for the live demo, not from Colab — Streamlit and Colab don't mix well for a
polished presentation.

---

## 6. What's still open

- **PDF report reconciliation** — the PDF generator built in a separate session needs to
  pull in `translation_output` and `audio_output` from this pipeline (translated text +
  mp3 path) alongside the heatmap and SHAP chart, so the final deliverable is one
  consistent report instead of disconnected pieces.
- **Real end-to-end mock→real flip** — running the complete pipeline (Vision + Tabular +
  Groq + NLLB + gTTS) together, for real, in one live session has not yet been done —
  still the highest-risk remaining integration step.
- Day 3 leftovers: edge-case testing (bad image, missing fields, low-confidence path),
  backup demo video, pitch prep.
