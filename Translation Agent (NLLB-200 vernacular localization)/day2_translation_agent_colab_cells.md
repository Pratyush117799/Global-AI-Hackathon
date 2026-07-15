# Day 2 — Adding the Translation Agent to your LangGraph notebook

Add these cells to your existing `day2_orchestration_langgraph.ipynb`, after the Groq
Pathologist/Climate agent cells. This wires in NLLB-200-distilled-600M for vernacular
localization, as the next stage after the HITL gate.

**Design decision made here (flag this in your pitch):** both graph branches now get
translated — not just the "proceed" action plan, but also the "ambiguous/needs review"
notice. A farmer whose photo couldn't be confidently classified still deserves to know
why, in their own language, not just silence. This directly serves the "Accessibility
Barrier" point in your problem statement.

---

## New cell — install deps (add near your other `!pip install` cells)

```python
!pip install -q transformers sentencepiece
```

## New cell — upload the 3 new files

Upload `language_config.py`, `translation_schema.py`, `translation_agent.py` via the
Colab file pane (same as you did for `tabular_analytics`). Then:

```python
import sys
sys.path.append("/content")
```

## New cell — flip MOCK_MODE for real inference

Open `translation_agent.py` in Colab's file editor (double-click it in the file pane)
and change:
```python
MOCK_MODE = True
```
to:
```python
MOCK_MODE = False
```
The real NLLB-200-distilled-600M model (~2.4GB) will download from Hugging Face and
load lazily on the first translation call, then stay cached for the rest of the session.
This runs fine on CPU runtime — no GPU required, per your original plan.

## Update `GraphState` — add 4 new fields

```python
class GraphState(TypedDict, total=False):
    image_path: str
    tabular_input: dict
    mock_vision_case: str
    mock_tabular_case: str
    target_language: str          # NEW — e.g. "Hindi", "Tamil", set by your UI dropdown

    vision_output: dict
    analytical_output: dict
    pathologist_interpretation: str
    climate_interpretation: str

    text_to_translate: str        # NEW
    content_type: str             # NEW
    translated_text: str          # NEW
    translation_output: dict      # NEW

    route: str
    status: str
```

## Update `ambiguous_review_node` and `proceed_to_translation_node`

Replace your existing versions with these (same function names, now they also prep
the text for translation):

```python
def ambiguous_review_node(state: GraphState) -> dict:
    review_text = (
        "Your leaf photo could not be classified with enough confidence for an automatic "
        "diagnosis. An agronomist will review this case. No treatment action is recommended "
        "until then."
    )
    return {
        "status": "AMBIGUOUS_REVIEW_REQUIRED",
        "route": "ambiguous",
        "text_to_translate": review_text,
        "content_type": "review_notice",
    }


def proceed_to_translation_node(state: GraphState) -> dict:
    action_plan = (
        f"{state['pathologist_interpretation']} "
        f"{state['climate_interpretation']} "
        f"Please monitor the field closely and consider preventive measures."
    )
    return {
        "status": "READY_FOR_TRANSLATION_AND_REPORTING",
        "route": "proceed",
        "text_to_translate": action_plan,
        "content_type": "action_plan",
    }
```

## New cell — import the translation node

```python
from translation_agent import translation_agent_node
```

## Update graph wiring — add the translation_agent node and rewire both branches through it

```python
builder = StateGraph(GraphState)

builder.add_node("vision", vision_node)
builder.add_node("analytical", analytical_node)
builder.add_node("pathologist_agent", pathologist_agent)
builder.add_node("climate_agent", climate_agent)
builder.add_node("ambiguous_review", ambiguous_review_node)
builder.add_node("proceed_to_translation", proceed_to_translation_node)
builder.add_node("translation_agent", translation_agent_node)   # NEW

builder.set_entry_point("vision")
builder.add_edge("vision", "analytical")
builder.add_edge("analytical", "pathologist_agent")
builder.add_edge("pathologist_agent", "climate_agent")

builder.add_conditional_edges(
    "climate_agent",
    hitl_router,
    {"ambiguous": "ambiguous_review", "proceed": "proceed_to_translation"},
)

# NEW — both branches now flow through translation_agent before ending
builder.add_edge("ambiguous_review", "translation_agent")
builder.add_edge("proceed_to_translation", "translation_agent")
builder.add_edge("translation_agent", END)

graph = builder.compile()
print("Graph compiled OK")
```

## Update your test harness — pass a target_language and print the translation

```python
test_runs = [
    {"mock_vision_case": "confident", "mock_tabular_case": "moderate", "target_language": "Hindi"},
    {"mock_vision_case": "ambiguous", "mock_tabular_case": "high", "target_language": "Tamil"},
    {"mock_vision_case": "confident", "mock_tabular_case": "low", "target_language": "Marathi"},
]

for run in test_runs:
    result = graph.invoke(run)
    print(json.dumps({
        "route": result["route"],
        "status": result["status"],
        "content_type": result["content_type"],
        "target_language": run.get("target_language"),
        "translated_text": result["translated_text"],
    }, indent=2, ensure_ascii=False))
```

Use `ensure_ascii=False` in the test harness — otherwise Devanagari/Tamil/etc. script
will print as escaped unicode instead of the actual characters.

---

## Verified in sandbox (mock mode, before handing off)

All 4 test combinations pass — routing logic unaffected by adding translation, language
resolution correct for Hindi/Tamil/Marathi codes, and missing `target_language` correctly
falls back to Hindi rather than crashing the graph. What is **not** yet verified: the real
NLLB-200 model download and actual translation quality — that needs to run in your Colab
session since this sandbox has no Hugging Face network access. Run the cells above with
`MOCK_MODE = False` and sanity-check that the Hindi/Tamil/Marathi output actually looks
like real translated text (not garbled) before treating this as done.

---

*Next: gTTS audio synthesis on `translated_text`, then PDF report generation (already
built in your other session) needs to pull in the translated text alongside the heatmap
and SHAP chart.*
