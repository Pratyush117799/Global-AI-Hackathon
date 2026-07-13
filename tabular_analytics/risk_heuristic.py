"""
Domain heuristic for 14-day disease-conducive-conditions risk.

WHY THIS EXISTS (for the pitch, when a judge asks "isn't this circular?"):
XGBoost is not "discovering" plant pathology from nothing. It is trained as a
smooth, differentiable surrogate for a hand-authored agronomic heuristic. The
benefit over just shipping the heuristic directly: (1) it interpolates
continuously between thresholds instead of hard if/else cliffs, (2) it lets us
attach calibrated SHAP attributions to a model instead of hand-waving about
which threshold fired, and (3) it can be re-trained on real field-trial data
later without changing the downstream contract at all.

Agronomic logic encoded below:
  - Humidity: fungal/bacterial sporulation and spread rises sharply above ~70%,
    plateaus near saturation.
  - Rainfall (14d cumulative): excess moisture promotes blight/rot; risk ramps
    up past ~40mm/14d.
  - Humidity x Rainfall interaction: wet + humid together is worse than either
    alone (classic late-blight/early-blight conducive-conditions logic).
  - Temperature: stress U-curve around an ideal 20-28C band for most row crops;
    both cold stress and heat stress reduce plant vigor and immune response.
  - Soil moisture: U-curve — drought stress and waterlogging/root-rot risk both
    increase disease susceptibility; ideal band ~40-60%.
  - N/P/K deficiency: below-recommended levels weaken structural and chemical
    plant defenses (esp. K, which is tied to disease resistance); risk rises
    as nutrients fall below agronomic sufficiency thresholds.
"""

import numpy as np

# Agronomic reference thresholds (kg/ha for NPK, % for humidity/moisture, mm for rainfall)
N_SUFFICIENT = 90.0
P_SUFFICIENT = 40.0
K_SUFFICIENT = 60.0
TEMP_IDEAL_LOW, TEMP_IDEAL_HIGH = 20.0, 28.0
MOISTURE_IDEAL_LOW, MOISTURE_IDEAL_HIGH = 40.0, 60.0
HUMIDITY_RISK_ONSET = 70.0
RAINFALL_RISK_ONSET = 40.0


def _sigmoid(x: float, midpoint: float, steepness: float) -> float:
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


def _deficiency_score(value: float, sufficient_level: float) -> float:
    """0 if at/above sufficiency, ramps to 1 as value -> 0."""
    return max(0.0, 1.0 - (value / sufficient_level)) if sufficient_level > 0 else 0.0


def _u_curve_stress(value: float, ideal_low: float, ideal_high: float, span: float) -> float:
    """0 inside [ideal_low, ideal_high], ramps to 1 as value moves `span` units outside the band."""
    if ideal_low <= value <= ideal_high:
        return 0.0
    dist = ideal_low - value if value < ideal_low else value - ideal_high
    return min(1.0, dist / span)


def compute_risk_components(
    temperature_c: float,
    humidity_pct: float,
    rainfall_mm_14d: float,
    soil_n: float,
    soil_p: float,
    soil_k: float,
    soil_moisture_pct: float,
) -> dict:
    """Returns each weighted sub-component (0-1 scale) plus the final blended risk_pct (0-100)."""

    humidity_component = _sigmoid(humidity_pct, HUMIDITY_RISK_ONSET, 0.15)
    rainfall_component = _sigmoid(rainfall_mm_14d, RAINFALL_RISK_ONSET, 0.06)
    wet_interaction = humidity_component * rainfall_component  # compounding wet+humid effect

    temp_stress = _u_curve_stress(temperature_c, TEMP_IDEAL_LOW, TEMP_IDEAL_HIGH, span=12.0)
    moisture_stress = _u_curve_stress(soil_moisture_pct, MOISTURE_IDEAL_LOW, MOISTURE_IDEAL_HIGH, span=30.0)

    n_deficiency = _deficiency_score(soil_n, N_SUFFICIENT)
    p_deficiency = _deficiency_score(soil_p, P_SUFFICIENT)
    k_deficiency = _deficiency_score(soil_k, K_SUFFICIENT)

    # Weighted blend — weights reflect relative agronomic importance for disease conduciveness
    weighted_sum = (
        0.24 * humidity_component +
        0.16 * rainfall_component +
        0.15 * wet_interaction +
        0.13 * temp_stress +
        0.12 * moisture_stress +
        0.08 * n_deficiency +
        0.06 * p_deficiency +
        0.06 * k_deficiency
    )

    risk_pct = float(np.clip(weighted_sum * 100.0, 0.0, 100.0))

    return {
        "humidity_component": humidity_component,
        "rainfall_component": rainfall_component,
        "wet_interaction": wet_interaction,
        "temp_stress": temp_stress,
        "moisture_stress": moisture_stress,
        "n_deficiency": n_deficiency,
        "p_deficiency": p_deficiency,
        "k_deficiency": k_deficiency,
        "risk_pct": risk_pct,
    }


def risk_band(risk_pct: float) -> str:
    if risk_pct < 30:
        return "Low"
    elif risk_pct < 60:
        return "Moderate"
    return "High"
