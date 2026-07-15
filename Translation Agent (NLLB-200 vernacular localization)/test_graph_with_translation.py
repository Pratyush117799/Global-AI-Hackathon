"""
Updated Day 2 graph nodes — adds Translation Agent after both HITL branches.

New graph shape:
  vision -> analytical -> pathologist_agent -> climate_agent -> HITL router ->
    {ambiguous_review -> translation_agent -> END,
     proceed_to_translation -> translation_agent -> END}

Design decision (stated explicitly, since it wasn't in the original spec):
Both branches now get translated, not just the "proceed" action plan. The
ambiguous case gets a short translated review notice instead of a full
treatment plan — a farmer whose photo was too ambiguous to classify still
deserves to know *why*, in their own language, not just get silence. This
matches the accessibility goal in the problem statement ("Accessibility
Barrier" section) applying to every pipeline exit, not just the happy path.
"""

import json
from typing import TypedDict
from langgraph.graph import StateGraph, END

from translation_agent import translation_agent_node, MOCK_MODE


class GraphState(TypedDict, total=False):
    image_path: str
    tabular_input: dict
    mock_vision_case: str
    mock_tabular_case: str
    target_language: str  # NEW — set by UI, e.g. "Hindi", "Tamil"

    vision_output: dict
    analytical_output: dict
    pathologist_interpretation: str
    climate_interpretation: str

    text_to_translate: str      # NEW
    content_type: str           # NEW
    translated_text: str        # NEW
    translation_output: dict    # NEW

    route: str
    status: str


# --- Mock data (Day 1 canonical results, unchanged) ---
MOCK_VISION_CASES = {
    "confident": {
        "disease_class": "Potato___Early_Blight",
        "confidence_score": 0.9702,
        "attention_matrix": [[0.1, 0.2], [0.3, 0.4]],
        "is_ambiguous": False,
    },
    "ambiguous": {
        "disease_class": "Corn___Common_Rust",
        "confidence_score": 0.494,
        "attention_matrix": [[0.1, 0.1], [0.1, 0.1]],
        "is_ambiguous": True,
    },
}

MOCK_TABULAR_CASES = {
    "moderate": {"14_day_risk_pct": 56.83, "risk_band": "Moderate",
                 "top_3_environmental_stressors": [{"feature": "humidity_pct", "direction": "increases_risk"}]},
    "low": {"14_day_risk_pct": 3.05, "risk_band": "Low",
            "top_3_environmental_stressors": [{"feature": "humidity_pct", "direction": "decreases_risk"}]},
    "high": {"14_day_risk_pct": 75.57, "risk_band": "High",
             "top_3_environmental_stressors": [{"feature": "humidity_pct", "direction": "increases_risk"}]},
}


def vision_node(state: GraphState) -> dict:
    case = state.get("mock_vision_case", "confident")
    return {"vision_output": MOCK_VISION_CASES[case]}


def analytical_node(state: GraphState) -> dict:
    case = state.get("mock_tabular_case", "moderate")
    return {"analytical_output": MOCK_TABULAR_CASES[case]}


def pathologist_agent(state: GraphState) -> dict:
    v = state["vision_output"]
    if v["is_ambiguous"]:
        text = "The model is not confident about this diagnosis. Human review is needed before any treatment is recommended."
    else:
        text = f"The leaf appears to have {v['disease_class'].replace('_', ' ')} with {v['confidence_score']*100:.1f}% confidence."
    return {"pathologist_interpretation": text}


def climate_agent(state: GraphState) -> dict:
    a = state["analytical_output"]
    text = f"14-day risk is {a['14_day_risk_pct']}% ({a['risk_band']} band), driven mainly by {a['top_3_environmental_stressors'][0]['feature']}."
    return {"climate_interpretation": text}


def hitl_router(state: GraphState) -> str:
    return "ambiguous" if state["vision_output"]["is_ambiguous"] else "proceed"


def ambiguous_review_node(state: GraphState) -> dict:
    review_text = (
        "Your leaf photo could not be classified with enough confidence for an automatic diagnosis. "
        "An agronomist will review this case. No treatment action is recommended until then."
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


# --- Build graph ---
builder = StateGraph(GraphState)
builder.add_node("vision", vision_node)
builder.add_node("analytical", analytical_node)
builder.add_node("pathologist_agent", pathologist_agent)
builder.add_node("climate_agent", climate_agent)
builder.add_node("ambiguous_review", ambiguous_review_node)
builder.add_node("proceed_to_translation", proceed_to_translation_node)
builder.add_node("translation_agent", translation_agent_node)  # NEW

builder.set_entry_point("vision")
builder.add_edge("vision", "analytical")
builder.add_edge("analytical", "pathologist_agent")
builder.add_edge("pathologist_agent", "climate_agent")

builder.add_conditional_edges(
    "climate_agent", hitl_router,
    {"ambiguous": "ambiguous_review", "proceed": "proceed_to_translation"},
)

# NEW — both branches now flow through translation_agent before ending
builder.add_edge("ambiguous_review", "translation_agent")
builder.add_edge("proceed_to_translation", "translation_agent")
builder.add_edge("translation_agent", END)

graph = builder.compile()
print(f"Graph compiled OK (MOCK_MODE={MOCK_MODE})")


if __name__ == "__main__":
    test_runs = [
        {"name": "Confident + Moderate + Hindi", "mock_vision_case": "confident", "mock_tabular_case": "moderate", "target_language": "Hindi"},
        {"name": "Ambiguous + High + Tamil", "mock_vision_case": "ambiguous", "mock_tabular_case": "high", "target_language": "Tamil"},
        {"name": "Confident + Low + Marathi", "mock_vision_case": "confident", "mock_tabular_case": "low", "target_language": "Marathi"},
        {"name": "Confident + Moderate + no language specified", "mock_vision_case": "confident", "mock_tabular_case": "moderate"},
    ]

    for run in test_runs:
        print(f"\n=== {run['name']} ===")
        result = graph.invoke({k: v for k, v in run.items() if k != "name"})
        print(json.dumps({
            "route": result["route"],
            "status": result["status"],
            "content_type": result["content_type"],
            "translated_text": result["translated_text"],
        }, indent=2))
