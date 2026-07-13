# Day 1 — Tabular/Analytical Node: Build & Test Results

**Approach chosen:** A — synthetic risk labels from a domain-expert heuristic (humidity/rainfall/temperature/moisture/NPK thresholds → 14-day risk %), with XGBoost trained as a smooth surrogate over the heuristic and SHAP for local attribution.
**Model:** `XGBRegressor` (300 trees, depth 4, lr 0.05) + `shap.TreeExplainer`
**Environment:** local sandbox (portable to Colab as-is)
**Output contract:** `14_day_risk_pct`, `top_3_environmental_stressors` (+ `risk_band` convenience field)

---

## Why Approach A over B

The Kaggle crop-recommendation dataset (Approach B) targets "best crop for this soil/climate," not disease risk — repurposing it still requires a hand-built heuristic to convert suitability into risk, so it inherits the "synthetic label" question without the benefit of a heuristic actually designed for phytopathology. Approach A's heuristic encodes real agronomic logic (fungal pressure from humidity, compounding wet+humid interaction, NPK deficiency weakening plant defenses, U-curve stress on temperature/moisture) so SHAP attributions are directly defensible in the pitch.

---

## Training results

| Metric | Value |
|---|---|
| Test MAE | 2.53 percentage points |
| Test R² | 0.9642 |
| Training rows | 4,800 (6,000 total, 80/20 split) |

**Feature importance ranking:** `humidity_pct` (0.56) > `rainfall_mm_14d` (0.34) > `temperature_c` (0.03) > `soil_moisture_pct` (0.03) > `soil_n`/`soil_k`/`soil_p` (~0.01 each) — this tracks the heuristic's designed weighting, confirming the model learned the intended structure rather than spurious correlations.

---

## Test 1 — High-risk scenario (hot, humid, wet)

Input: 27.5°C, 88% humidity, 95mm rainfall/14d, moderate NPK, 62% soil moisture.

| Field | Value |
|---|---|
| `14_day_risk_pct` | 56.83 |
| `risk_band` | Moderate |
| Top stressors | `humidity_pct` (+17.0, increases_risk), `rainfall_mm_14d` (+9.4, increases_risk), `temperature_c` (−2.6, decreases_risk) |

**Result:** Correctly identifies humidity and rainfall as the dominant risk drivers. Interesting nuance: temperature at 27.5°C sits near the ideal band, so it slightly *reduces* predicted risk — a good talking point for judges on why SHAP beats a flat "hot = bad" assumption.

---

## Test 2 — Low-risk scenario (mild, dry, well-fed)

Input: 23°C, 45% humidity, 8mm rainfall/14d, good NPK, 48% soil moisture.

| Field | Value |
|---|---|
| `14_day_risk_pct` | 3.05 |
| `risk_band` | Low |
| Top stressors | `humidity_pct` (−13.9, decreases_risk), `rainfall_mm_14d` (−8.2, decreases_risk), `temperature_c` (−2.5, decreases_risk) |

**Result:** Correctly reads as low risk with all top features pushing risk *down* — good mirror case to Test 1.

---

## Test 3 — Canonical High-band demo case (waterlogged, nutrient-poor, hot+humid)

Input: 31°C, 94% humidity, 130mm rainfall/14d, poor NPK, 85% soil moisture.

| Field | Value |
|---|---|
| `14_day_risk_pct` | 75.57 |
| `risk_band` | High |
| Top stressors | `humidity_pct` (+17.4), `rainfall_mm_14d` (+9.9), `soil_moisture_pct` (+7.2) — all increase risk |

**Result:** This is your canonical **High-band** demo case — High risk was the rarest class in training (333/6000, ~5.5%), so worth confirming it predicts sensibly, and it does. Note SHAP correctly promotes `soil_moisture_pct` (waterlogging) into the top 3 here instead of temperature, showing the model picks up the interaction rather than always returning the same 3 features.

---

## Summary / go-forward decisions

1. **Node is functioning end-to-end**: synthetic data generation, XGBoost training, SHAP attribution, and Pydantic-validated JSON output all work as specified.
2. **Output contract is frozen**: `14_day_risk_pct` (float, alias-mapped since Python identifiers can't start with a digit), `top_3_environmental_stressors` (list of 3, each with feature/value/shap_contribution/direction), `risk_band` (Low/Moderate/High convenience field).
3. **Keep all three test cases** — Test 1 (Moderate, humidity/rainfall-driven), Test 2 (Low, dry/mild), Test 3 (High, waterlogged) — as your canonical demo set, same role as Vision Node's Test 1/Test 2.
4. **Retraining path**: if you get real field-trial data before Day 3, swap `generate_dataset.py`'s heuristic labels for real observed outcomes — `train_model.py` and `inference.py` don't need to change since they only depend on the CSV schema, not the label source.

---

*Next: LangGraph orchestration wiring Vision Node + Analytical Node → Pathologist Agent + Climate Agent → HITL Gate.*
