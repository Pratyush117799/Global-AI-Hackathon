from pydantic import BaseModel, Field
from typing import Literal, Optional


class AudioOutput(BaseModel):
    """Output contract for the Audio Agent — final artifact before PDF assembly."""

    target_language: str = Field(..., description="Human-readable language name")
    tts_status: Literal["success", "skipped_unsupported_language", "failed"] = Field(
        ..., description="success = mp3 generated, skipped_... = gTTS doesn't support this "
                          "language (translation still succeeded), failed = gTTS call errored"
    )
    audio_file_path: Optional[str] = Field(
        None, description="Path to the generated .mp3 file, only set when tts_status == 'success'"
    )
    skip_reason: Optional[str] = Field(
        None, description="Human-readable explanation, set when tts_status != 'success'"
    )
