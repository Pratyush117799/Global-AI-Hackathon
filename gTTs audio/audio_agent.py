"""
Audio Agent — synthesizes translated_text into speech using gTTS.

MOCK_MODE note: gTTS calls Google's translate-TTS endpoint over the network.
This sandbox has no route to that domain, so MOCK_MODE = True writes a fake
placeholder file so the surrounding graph/state logic is still testable. In
Colab (which does have normal internet access), set MOCK_MODE = False.

Graceful degradation: gTTS does not support every language NLLB translates to
(confirmed: Odia is not supported as of this check). When that happens, this
agent does NOT fail the whole pipeline — it returns tts_status =
'skipped_unsupported_language', keeps the translated TEXT available for the
PDF report, and simply omits the audio file. A farmer in an unsupported
language still gets a translated written report; they just don't get an
audio file for that specific language yet.
"""

import os

from language_config import resolve_gtts_code
from audio_schema import AudioOutput

MOCK_MODE = True  # flip to False in Colab, where gTTS has real internet access

AUDIO_OUTPUT_DIR = "audio_output"


def _synthesize_real(text: str, gtts_code: str, output_path: str) -> None:
    from gtts import gTTS
    tts = gTTS(text=text, lang=gtts_code)
    tts.save(output_path)


def _synthesize_mock(text: str, gtts_code: str, output_path: str) -> None:
    """Writes a placeholder file (not real audio) so state/file-path logic is testable."""
    with open(output_path, "w") as f:
        f.write(f"[MOCK AUDIO PLACEHOLDER] lang={gtts_code} chars={len(text)}\n{text}")


def audio_agent_node(state: dict) -> dict:
    """
    LangGraph node. Reads `translated_text`, `target_language`, and `content_type`
    (all set by translation_agent_node) and produces an mp3 file path, or a
    documented skip if the language isn't supported by gTTS.
    """
    translated_text = state.get("translated_text", "")
    target_language = state.get("target_language", "Hindi")
    content_type = state.get("content_type", "action_plan")

    if not translated_text:
        output = AudioOutput(
            target_language=target_language,
            tts_status="skipped_unsupported_language",
            skip_reason="No translated_text present in state — translation step may have failed.",
        )
        return {"audio_output": output.model_dump()}

    gtts_code = resolve_gtts_code(target_language)

    if gtts_code is None:
        output = AudioOutput(
            target_language=target_language,
            tts_status="skipped_unsupported_language",
            skip_reason=f"gTTS does not currently support {target_language}. "
                        f"Translated text is still available for the PDF report.",
        )
        return {"audio_output": output.model_dump()}

    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    filename = f"{content_type}_{gtts_code}.mp3"
    output_path = os.path.join(AUDIO_OUTPUT_DIR, filename)

    try:
        if MOCK_MODE:
            _synthesize_mock(translated_text, gtts_code, output_path)
        else:
            _synthesize_real(translated_text, gtts_code, output_path)

        output = AudioOutput(
            target_language=target_language,
            tts_status="success",
            audio_file_path=output_path,
        )
    except Exception as e:
        output = AudioOutput(
            target_language=target_language,
            tts_status="failed",
            skip_reason=f"gTTS call raised an exception: {e}",
        )

    return {"audio_output": output.model_dump()}
