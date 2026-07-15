"""
Language configuration for the Translation Agent.

NLLB-200 uses FLORES-200 language codes (e.g. "hin_Deva" not "hi"). This module
maps human-readable language names (for a dropdown in the Streamlit/Gradio UI)
to the exact codes NLLB expects, so the rest of the pipeline never has to know
the FLORES code format.

To add a language: just add a line here. Nothing else in the pipeline needs to change.
"""

SUPPORTED_LANGUAGES = {
    "Hindi": "hin_Deva",
    "Marathi": "mar_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Bengali": "ben_Beng",
    "Gujarati": "guj_Gujr",
    "Kannada": "kan_Knda",
    "Punjabi": "pan_Guru",
    "Malayalam": "mal_Mlym",
    "Odia": "ory_Orya",
    "Urdu": "urd_Arab",
    "English": "eng_Latn",  # pass-through option, useful for judges/testing
}

SOURCE_LANG_CODE = "eng_Latn"  # Pathologist/Climate agent output is always English
DEFAULT_TARGET_LANGUAGE = "Hindi"  # used only if state doesn't specify one


def resolve_language_code(language_name: str) -> str:
    """Look up a FLORES code by display name. Falls back to default if unrecognized."""
    if language_name in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[language_name]
    return SUPPORTED_LANGUAGES[DEFAULT_TARGET_LANGUAGE]


def language_choices() -> list:
    """For wiring into a Gradio/Streamlit dropdown directly."""
    return list(SUPPORTED_LANGUAGES.keys())


# --- gTTS support ---
# gTTS (Google Text-to-Speech) uses ISO 639-1 codes and does NOT cover every
# language NLLB translates to. Verified directly against gtts.lang.tts_langs():
# Odia is translatable (NLLB) but NOT speakable (gTTS) as of this check. Any
# language present in SUPPORTED_LANGUAGES but absent here must degrade
# gracefully in the Audio Agent (translation still succeeds, audio is skipped).
GTTS_LANGUAGE_MAP = {
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Punjabi": "pa",
    "Malayalam": "ml",
    "Urdu": "ur",
    "English": "en",
    # "Odia" intentionally omitted — not supported by gTTS. Re-check with
    # gtts.lang.tts_langs() if the gTTS/Google TTS backend changes.
}


def resolve_gtts_code(language_name: str):
    """Returns the gTTS code, or None if this language isn't speakable (caller must handle gracefully)."""
    return GTTS_LANGUAGE_MAP.get(language_name)
