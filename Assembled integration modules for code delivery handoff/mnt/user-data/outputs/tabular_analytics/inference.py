"""
Analytical Node — inference entrypoint.

Usage:
    from inference import run_analytical_node
    result = run_analytical_node(EnvironmentalVector(...))
    print(result.to_wire_json())
"""

import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from schema import EnvironmentalVector, AnalyticalNodeOutput, StressorContribution
from risk_heuristic import risk_band as compute_risk_band

FEATURE_COLUMNS = [
    "temperature_c", "humidity_pct", "rainfall_mm_14d",
    "soil_n", "soil_p", "soil_k", "soil_moisture_pct",
]

# Resolve artifact paths relative to THIS FILE, not the caller's working directory.
# Without this, model.json/explainer.pkl only load correctly if the graph happens
# to be invoked with tabular_analytics/ as the cwd — silently breaks otherwise
# (found during real-integration testing: pipeline.py invokes this from /frontend).
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_MODULE_DIR, "model.json")
EXPLAINER_PATH = os.path.join(_MODULE_DIR, "explainer.pkl")

_model = None
_explainer = None


def _load_artifacts():
    global _model, _explainer
    if _model is None:
        _model = xgb.XGBRegressor()
        _model.load_model(MODEL_PATH)
    if _explainer is None:
        with open(EXPLAINER_PATH, "rb") as f:
            _explainer = pickle.load(f)
    return _model, _explainer


def run_analytical_node(vector: EnvironmentalVector) -> AnalyticalNodeOutput:
    model, explainer = _load_artifacts()

    row = pd.DataFrame([[getattr(vector, col) for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    risk_pct = float(np.clip(model.predict(row)[0], 0, 100))

    shap_values = explainer(row)
    contributions = shap_values.values[0]  # one SHAP value per feature, for this single row

    ranked_idx = np.argsort(-np.abs(contributions))[:3]

    top_stressors = []
    for idx in ranked_idx:
        feat_name = FEATURE_COLUMNS[idx]
        shap_val = float(contributions[idx])
        top_stressors.append(
            StressorContribution(
                feature=feat_name,
                value=float(row.iloc[0][feat_name]),
                shap_contribution=shap_val,
                direction="increases_risk" if shap_val > 0 else "decreases_risk",
            )
        )

    return AnalyticalNodeOutput(
        **{"14_day_risk_pct": round(risk_pct, 2)},
        top_3_environmental_stressors=top_stressors,
        risk_band=compute_risk_band(risk_pct),
    )


if __name__ == "__main__":
    import json

    # Quick smoke test — a hot, humid, wet scenario (should read High risk)
    high_risk_case = EnvironmentalVector(
        temperature_c=27.5,
        humidity_pct=88.0,
        rainfall_mm_14d=95.0,
        soil_n=70.0,
        soil_p=30.0,
        soil_k=45.0,
        soil_moisture_pct=62.0,
    )
    result = run_analytical_node(high_risk_case)
    print("=== High-risk scenario ===")
    print(json.dumps(result.to_wire_json(), indent=2))

    # A dry, mild, well-fed scenario (should read Low risk)
    low_risk_case = EnvironmentalVector(
        temperature_c=23.0,
        humidity_pct=45.0,
        rainfall_mm_14d=8.0,
        soil_n=95.0,
        soil_p=42.0,
        soil_k=65.0,
        soil_moisture_pct=48.0,
    )
    result2 = run_analytical_node(low_risk_case)
    print("\n=== Low-risk scenario ===")
    print(json.dumps(result2.to_wire_json(), indent=2))
