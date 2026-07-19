# 🌾 Agrospheric AI

**A Multimodal Multi-Agent Pipeline for Precision Phytopathology and Risk Mitigation**

Agrospheric AI fuses computer vision leaf diagnostics with environmental risk analytics
through a deterministic multi-agent LangGraph pipeline — turning a leaf photo and a
weather/soil reading into a localized, explainable, actionable report for farmers, in
their own language, with audio narration and a downloadable PDF.

Built for Global AI Hackathon by Pratyush Sharma (Agentic AI engineer)





## Why this exists

Most crop-disease tools are single-purpose: a vision classifier *or* a weather alert,
never both, and rarely explainable or localized. Agrospheric AI is built around three
problems those tools don't solve:

* **The Modality Gap** — vision-only tools detect disease after damage is visible;
weather-only tools don't know the crop's health history. Fusing both enables proactive,
not reactive, guidance.
* **The Black-Box Dilemma** — "your crop has blight" without showing *why* or *where*
doesn't build farmer trust. Every prediction here comes with an explanation.
* **The Accessibility Barrier** — a PDF in English is useless to a farmer who doesn't
read English. Every report is translated and narrated in the farmer's own language —
including when the system can't make a diagnosis at all.

\---

## Architecture



```

&#x20;               \[Leaf Image]                    \[Weather / Soil Vector]

&#x20;                    |                                    |

&#x20;                    v                                    v

&#x20;             ┌─────────────┐                    ┌─────────────────┐

&#x20;             │ Vision Node │                    │ Analytical Node │

&#x20;             │  ViT + XAI  │                    │ XGBoost + SHAP  │

&#x20;             └──────┬──────┘                    └────────┬────────┘

&#x20;                    |                                     |

&#x20;                    v                                     v

&#x20;             ┌─────────────────┐                ┌────────────────┐

&#x20;             │ Pathologist Agent│                │  Climate Agent │

&#x20;             │   (Groq LLM)     │                │   (Groq LLM)   │

&#x20;             └────────┬─────────┘                └───────┬────────┘

&#x20;                       \\\\                                /

&#x20;                        v                               v

&#x20;                         ┌───────────────────────────┐

&#x20;                         │        HITL Gate           │

&#x20;                         │ confidence < 75% OR         │

&#x20;                         │ non-diagnostic class         │

&#x20;                         │  → route to human review     │

&#x20;                         └──────────────┬────────────────┘

&#x20;                         "ambiguous"   /   \\\\   "proceed"

&#x20;                                      v     v

&#x20;                         ┌────────────┐   ┌──────────────────────┐

&#x20;                         │  Ambiguous  │   │ Proceed to Translation│

&#x20;                         │  Review     │   └───────────┬────────────┘

&#x20;                         └──────┬──────┘               │

&#x20;                                └───────────┬───────────┘

&#x20;                                            v

&#x20;                                 ┌────────────────────┐

&#x20;                                 │ Translation Agent   │

&#x20;                                 │   NLLB-200-distilled │

&#x20;                                 └──────────┬───────────┘

&#x20;                                            v

&#x20;                                 ┌────────────────────┐

&#x20;                                 │   Audio Agent       │

&#x20;                                 │      gTTS            │

&#x20;                                 └──────────┬───────────┘

&#x20;                                            v

&#x20;                                 ┌────────────────────┐

&#x20;                                 │  PDF Report Node    │

&#x20;                                 │ reportlab + shaped   │

&#x20;                                 │ text (PIL + raqm)     │

&#x20;                                 └──────────┬───────────┘

&#x20;                                            v

&#x20;                                \[PDF + Localized Text + Audio]

```



Both HITL branches — not just the "cleared" path — flow through translation, audio, and
the PDF report. A farmer whose case gets flagged for human review still gets a translated,
narrated explanation of why, in their own language, not silence.

\---

## Features

* **Vision Node** — fine-tuned Vision Transformer (`wambugu71/crop\_leaf\_diseases\_vit`)
with attention-rollout heatmaps for explainability, not just a bare classification.
* **Analytical Node** — XGBoost regressor trained on a documented agronomic heuristic
(humidity/rainfall interaction, temperature/moisture stress curves, NPK deficiency),
with SHAP for per-feature attribution — a smooth, explainable surrogate rather than a
black-box classifier or a raw hardcoded rule engine.
* **Multi-agent orchestration** — a deterministic LangGraph state machine, not a fragile
prompt chain, with strict Pydantic schemas at every node boundary.
* **Human-in-the-loop safety gate** — routes to mandatory human review when vision
confidence is below 75% **or** the model predicts a non-diagnostic class (e.g.
out-of-distribution images correctly flagged "Invalid" by the model itself — verified
via real edge-case testing with a non-leaf photo).
* **Vernacular localization** — 11 languages via NLLB-200-distilled-600M, with properly
shaped complex-script rendering (Devanagari conjuncts, Tamil/Telugu/etc. vowel
reordering, Urdu RTL + letter-joining) in the final PDF — verified against
ReportLab's native text layer, which does not shape complex scripts correctly on its
own.
* **Audio narration** — gTTS speech synthesis of the localized report, with graceful
degradation for languages gTTS doesn't support (e.g. Odia: translated text still
delivered, audio section clearly explains why narration isn't available).
* **PDF report generation** — technical section (exact classification, exact SHAP
values) preserved in English for judges/agronomists, plus a separate localized section
for the farmer — not a straight replacement of one with the other.
* **Streamlit UI** — image upload, environmental-vector sliders, language picker, a live
agent-log panel streaming node-by-node progress, side-by-side heatmap + SHAP chart,
HITL banner, and a PDF download button.

\---

## Repository structure

```
vision\_pipeline/
  inference.py            # ViT + attention rollout + HITL confidence/class check
tabular\_analytics/
  schema.py                # Pydantic I/O contract
  risk\_heuristic.py         # documented agronomic risk formula
  generate\_dataset.py        # synthetic training data generator
  train\_model.py              # XGBoost + SHAP training
  inference.py                  # real-time prediction + SHAP attribution
  model.json, explainer.pkl       # trained artifacts
agent\_workflow/
  language\_config.py        # 11-language FLORES/gTTS code mapping
  translation\_schema.py, translation\_agent.py   # NLLB-200 translation node
  audio\_schema.py, audio\_agent.py               # gTTS narration node
reporting/
  font\_config.py             # Noto Sans font registration for non-Latin scripts
  report\_node.py               # PDF assembly (ReportLab + PIL/raqm shaped text)
pipeline.py                  # LangGraph state graph wiring every node together
visual\_helpers.py             # heatmap / SHAP chart rendering for the UI
app.py                          # Streamlit frontend
requirements.txt
```

\---

## Setup

### Option A — Google Colab (two notebooks, recommended for training/GPU work)

1. **`01\_agrospheric\_backend.ipynb`** — builds every backend module, trains the Tabular
Node, downloads fonts, and runs a full real end-to-end test. Ends by packaging
everything into `agrospheric\_backend\_package.zip`.
2. **`02\_agrospheric\_streamlit.ipynb`** — upload the zip from Notebook 1, installs the UI
layer, sanity-checks every real import, and launches Streamlit behind a tunnel for
live testing.

You'll need a free [Groq API key](https://console.groq.com) for the Pathologist/Climate
agents — add it via Colab's Secrets panel (key icon in the left sidebar) as
`GROQ\_API\_KEY`, rather than hardcoding it in a cell.

### Option B — Local

```bash
git clone <this-repo>
cd agrospheric-ai
pip install -r requirements.txt
export GROQ\_API\_KEY="your-key-here"
streamlit run app.py
```

Local is recommended for the actual live demo — Streamlit tunneled out of Colab (via
`localtunnel` or similar) is noticeably less reliable for a live audience than running
directly on your machine.

\---

## Known limitations

* **gTTS doesn't support every language NLLB translates to** — confirmed Odia is
translatable but not speakable via gTTS. The report degrades gracefully: translated
text is always present, audio is simply omitted with a clear explanation when
unsupported.
* **Free tunnel services (localtunnel) can drop individual static asset requests**,
occasionally breaking a Streamlit widget's lazy-loaded JS chunk (shows as `Failed to fetch dynamically imported module` in the browser console). A hard refresh usually
fixes it; Cloudflare's tunnel is a more reliable alternative if it recurs.
* **The HITL gate depends on knowing every non-diagnostic label** your specific vision
checkpoint can output (e.g. `"Invalid"` for this model). If you swap in a different
vision model, check its `id2label` and update `NON\_DIAGNOSTIC\_LABELS` in
`vision\_pipeline/inference.py` accordingly.

\---

## Tech stack

Vision Transformer (HuggingFace Transformers) · XGBoost + SHAP · LangGraph ·
Groq (Llama 3.3 70B) · NLLB-200-distilled-600M · gTTS · ReportLab + PIL/raqm ·
Streamlit · Pydantic

\---

## License

\[Add your license here]

