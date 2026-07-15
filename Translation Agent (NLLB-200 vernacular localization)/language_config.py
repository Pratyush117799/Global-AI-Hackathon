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
