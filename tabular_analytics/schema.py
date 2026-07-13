"""
Tabular / Analytical Node — I/O Schema
----------------------------------------
This is the FROZEN contract for Day 2 (LangGraph Climate Agent consumes this
output directly, same as Vision Node's JSON is consumed by the Pathologist Agent).

Do not change field names/types after Day 1 without updating the orchestrator.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List


class EnvironmentalVector(BaseModel):
    """Input: structured weather/soil vector for a single field/plot."""

    temperature_c: float = Field(..., description="Mean ambient temperature, Celsius")
    humidity_pct: float = Field(..., ge=0, le=100, description="Relative humidity, %")
    rainfall_mm_14d: float = Field(..., ge=0, description="Cumulative rainfall over last 14 days, mm")
    soil_n: float = Field(..., ge=0, description="Soil Nitrogen, kg/ha")
    soil_p: float = Field(..., ge=0, description="Soil Phosphorus, kg/ha")
    soil_k: float = Field(..., ge=0, description="Soil Potassium, kg/ha")
    soil_moisture_pct: float = Field(..., ge=0, le=100, description="Soil moisture, %")


class StressorContribution(BaseModel):
    """One entry in top_3_environmental_stressors — a SHAP-attributed driver of risk."""

    feature: str = Field(..., description="Name of the environmental/soil feature")
    value: float = Field(..., description="The raw observed value of this feature")
    shap_contribution: float = Field(
        ..., description="Signed SHAP value: positive = pushed risk up, negative = pushed risk down"
    )
    direction: Literal["increases_risk", "decreases_risk"] = Field(
        ..., description="Human-readable direction of this feature's effect on risk"
    )


class AnalyticalNodeOutput(BaseModel):
    """
    Output contract — matches the spec table exactly on the two required keys,
    plus one extra convenience field (risk_band) in the same spirit as Vision
    Node's extra `is_ambiguous` field.
    """

    model_config = ConfigDict(populate_by_name=True)

    risk_pct_14day: float = Field(
        ..., alias="14_day_risk_pct", ge=0, le=100,
        description="Predicted probability (%) of disease-conducive stress conditions over next 14 days",
    )
    top_3_environmental_stressors: List[StressorContribution] = Field(
        ..., min_length=3, max_length=3,
        description="Top 3 features driving the risk score, ranked by |SHAP value|",
    )
    risk_band: Literal["Low", "Moderate", "High"] = Field(
        ..., description="Convenience bucket for downstream routing/UI, derived from risk_pct_14day"
    )

    def to_wire_json(self) -> dict:
        """Serialize using the exact wire field name '14_day_risk_pct' for the orchestrator."""
        return self.model_dump(by_alias=True)
