from pydantic import BaseModel, Field
from typing import Literal


class TranslationOutput(BaseModel):
    """Output contract for the Translation Agent — feeds gTTS next."""

    source_text: str = Field(..., description="Original English text (action plan or review message)")
    translated_text: str = Field(..., description="Text translated into the target language")
    target_language: str = Field(..., description="Human-readable language name, e.g. 'Hindi'")
    target_language_code: str = Field(..., description="NLLB FLORES-200 code, e.g. 'hin_Deva'")
    content_type: Literal["action_plan", "review_notice"] = Field(
        ..., description="Whether this was a full treatment action plan (proceed path) "
                          "or a short ambiguous-case review notice (HITL path)"
    )
