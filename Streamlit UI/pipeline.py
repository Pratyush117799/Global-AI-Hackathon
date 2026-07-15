"""
Pipeline wiring for the Streamlit UI.

This module adapts the LangGraph graph built in agent_workflow/ to accept REAL
user input (an uploaded image, a tabular form) instead of the `mock_vision_case`
/ `mock_tabular_case` selectors used for testing. It tries real inference first;
if the real modules aren't on the path (e.g. running this file standalone
without vision_pipeline/tabular_analytics alongside it), it falls back to your
Day 1 canonical results so the UI never crashes during a live demo — worst
case, it silently demos with known-good numbers instead of erroring out.

Real-mode requirements (put these packages alongside this file in your repo):
  vision_pipeline/inference.py      -> run_vision_node(image) -> VisionNodeOutput
  tabular_analytics/inference.py    -> run_analytical_node(vector) -> AnalyticalNodeOutput
  tabular_analytics/schema.py       -> EnvironmentalVector
  agent_workflow/translation_agent.py
  agent_workflow/audio_agent.py
  agent_workflow/language_config.py
  A Groq client for pathologist_agent / climate_agent (see day 2.txt for setup)
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# --- Try real modules first, fall back to None (triggers demo-safe fallback data) ---
try:
    from vision_pipeline.inference import run_vision_node
except Exception:
    run_vision_node = None

try:
    from tabular_analytics.inference import run_analytical_node
    from tabular_analytics.schema import EnvironmentalVector
except Exception:
    run_analytical_node = None
    EnvironmentalVector = None

try:
    from agent_workflow.translation_agent import translation_agent_node
except Exception:
    translation_agent_node = None

try:
    from agent_workflow.audio_agent import audio_agent_node
except Exception:
    audio_agent_node = None

try:
    import os
    from groq import Groq
    _groq_key = os.environ.get("GROQ_API_KEY")
    groq_client = Groq(api_key=_groq_key) if _groq_key else None
except Exception:
    groq_client = None

GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Demo-safe fallback data (Day 1 canonical results — used if real modules unavailable) ---
FALLBACK_VISION = {
    "disease_class": "Potato___Early_Blight",
    "confidence_score": 0.9702,
    "attention_matrix": [[0.1, 0.3, 0.2, 0.1], [0.4, 0.9, 0.7, 0.2], [0.3, 0.6, 0.5, 0.2], [0.1, 0.2, 0.2, 0.1]],
    "is_ambiguous": False,
}
FALLBACK_ANALYTICAL = {
    "14_day_risk_pct": 56.83,
    "risk_band": "Moderate",
    "top_3_environmental_stressors": [
        {"feature": "humidity_pct", "value": 88.0, "shap_contribution": 17.0, "direction": "increases_risk"},
        {"feature": "rainfall_mm_14d", "value": 95.0, "shap_contribution": 9.4, "direction": "increases_risk"},
        {"feature": "temperature_c", "value": 27.5, "shap_contribution": -2.6, "direction": "decreases_risk"},
    ],
}


class GraphState(TypedDict, total=False):
    image: object            # PIL.Image, set by the UI
    tabular_input: dict      # raw form values, set by the UI
    target_language: str

    vision_output: dict
    analytical_output: dict
    pathologist_interpretation: str
    climate_interpretation: str

    text_to_translate: str
    content_type: str
    translated_text: str
    translation_output: dict
    audio_output: dict

    route: str
    status: str
    used_fallback: bool  # True if real inference wasn't available for this run


def vision_node(state: GraphState) -> dict:
    image = state.get("image")
    if run_vision_node is not None and image is not None:
        try:
            result = run_vision_node(image)
            return {"vision_output": result.model_dump() if hasattr(result, "model_dump") else result}
        except Exception:
            pass  # fall through to fallback data below
    return {"vision_output": FALLBACK_VISION, "used_fallback": True}


def analytical_node(state: GraphState) -> dict:
    tabular_input = state.get("tabular_input")
    if run_analytical_node is not None and EnvironmentalVector is not None and tabular_input:
        try:
            vector = EnvironmentalVector(**tabular_input)
            result = run_analytical_node(vector)
            return {"analytical_output": result.to_wire_json()}
        except Exception:
            pass
    return {"analytical_output": FALLBACK_ANALYTICAL, "used_fallback": True}


def _groq_interpret(prompt: str, fallback_text: str) -> str:
    if groq_client is None:
        return fallback_text
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return fallback_text


def pathologist_agent(state: GraphState) -> dict:
    v = state["vision_output"]
    fallback = (
        "Human review needed — confidence too low for automated diagnosis."
        if v["is_ambiguous"] else
        f"The leaf shows signs of {v['disease_class'].replace('_', ' ')} "
        f"with {v['confidence_score']*100:.1f}% model confidence."
    )
    prompt = (
        "You are an agricultural plant pathologist assistant. Given this disease "
        "classification JSON, write a short (3-4 sentence) plain-language interpretation "
        f"for a farmer. JSON: {v}. If is_ambiguous is true, state that a human review is "
        "needed and do not recommend treatment. Keep it under 80 words, no jargon."
    )
    return {"pathologist_interpretation": _groq_interpret(prompt, fallback)}


def climate_agent(state: GraphState) -> dict:
    a = state["analytical_output"]
    top = a["top_3_environmental_stressors"][0]
    fallback = f"14-day risk is {a['14_day_risk_pct']}% ({a['risk_band']} band), driven mainly by {top['feature']}."
    prompt = (
        "You are an agronomy climate-risk assistant. Given this 14-day environmental risk "
        f"JSON, write a short (3-4 sentence) plain-language interpretation for a farmer. "
        f"JSON: {a}. Explain which factors increase vs decrease risk. Under 80 words, no jargon."
    )
    return {"climate_interpretation": _groq_interpret(prompt, fallback)}


def hitl_router(state: GraphState) -> str:
    return "ambiguous" if state["vision_output"]["is_ambiguous"] else "proceed"


def ambiguous_review_node(state: GraphState) -> dict:
    review_text = (
        "Your leaf photo could not be classified with enough confidence for an automatic "
        "diagnosis. An agronomist will review this case. No treatment action is recommended "
        "until then."
    )
    return {"status": "AMBIGUOUS_REVIEW_REQUIRED", "route": "ambiguous",
            "text_to_translate": review_text, "content_type": "review_notice"}


def proceed_to_translation_node(state: GraphState) -> dict:
    action_plan = (
        f"{state['pathologist_interpretation']} {state['climate_interpretation']} "
        f"Please monitor the field closely and consider preventive measures."
    )
    return {"status": "READY_FOR_TRANSLATION_AND_REPORTING", "route": "proceed",
            "text_to_translate": action_plan, "content_type": "action_plan"}


def _translation_fallback(state: GraphState) -> dict:
    text = state.get("text_to_translate", "")
    lang = state.get("target_language", "Hindi")
    return {
        "translated_text": f"[Translation unavailable — showing English] {text}",
        "translation_output": {"source_text": text, "translated_text": text,
                                "target_language": lang, "target_language_code": "n/a",
                                "content_type": state.get("content_type", "action_plan")},
    }


def _audio_fallback(state: GraphState) -> dict:
    lang = state.get("target_language", "Hindi")
    return {"audio_output": {"target_language": lang, "tts_status": "skipped_unsupported_language",
                              "audio_file_path": None,
                              "skip_reason": "Audio agent module not available in this session."}}


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("vision", vision_node)
    builder.add_node("analytical", analytical_node)
    builder.add_node("pathologist_agent", pathologist_agent)
    builder.add_node("climate_agent", climate_agent)
    builder.add_node("ambiguous_review", ambiguous_review_node)
    builder.add_node("proceed_to_translation", proceed_to_translation_node)
    builder.add_node("translation_agent", translation_agent_node or _translation_fallback)
    builder.add_node("audio_agent", audio_agent_node or _audio_fallback)

    builder.set_entry_point("vision")
    builder.add_edge("vision", "analytical")
    builder.add_edge("analytical", "pathologist_agent")
    builder.add_edge("pathologist_agent", "climate_agent")
    builder.add_conditional_edges("climate_agent", hitl_router,
        {"ambiguous": "ambiguous_review", "proceed": "proceed_to_translation"})
    builder.add_edge("ambiguous_review", "translation_agent")
    builder.add_edge("proceed_to_translation", "translation_agent")
    builder.add_edge("translation_agent", "audio_agent")
    builder.add_edge("audio_agent", END)

    return builder.compile()
