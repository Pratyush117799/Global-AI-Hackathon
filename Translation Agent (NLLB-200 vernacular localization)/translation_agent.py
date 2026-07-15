"""
Translation Agent — vernacular localization using NLLB-200-distilled-600M.

MOCK_MODE note: This module can't download facebook/nllb-200-distilled-600M in
environments without huggingface.co network access (e.g. this sandbox). Set
MOCK_MODE = True to test the surrounding logic (text building, state merging,
graph wiring) with a fake translator. In Colab, set MOCK_MODE = False and the
real model loads once (lazily, on first call) and is cached for the session.

Model choice: NLLB-200-distilled-600M was picked (per your Day 2 plan) because
it runs comfortably on CPU or a shared T4, unlike the full 3.3B NLLB variant.
"""

from language_config import resolve_language_code, SOURCE_LANG_CODE, DEFAULT_TARGET_LANGUAGE
from translation_schema import TranslationOutput

MOCK_MODE = True  # flip to False in Colab once transformers+torch+model download are available

NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

_tokenizer = None
_model = None


def _load_nllb():
    """Lazy-load NLLB tokenizer + model once per session. Real inference only (MOCK_MODE=False)."""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    _tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME, src_lang=SOURCE_LANG_CODE)
    _model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
    _model.eval()
    return _tokenizer, _model


def _forced_bos_token_id(tokenizer, target_code: str) -> int:
    """
    Handles both modern and older transformers NLLB tokenizer APIs:
    - modern (>=4.32-ish): tokenizer.convert_tokens_to_ids(target_code)
    - older: tokenizer.lang_code_to_id[target_code]
    """
    try:
        token_id = tokenizer.convert_tokens_to_ids(target_code)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return token_id
    except Exception:
        pass
    # fallback for older transformers versions
    return tokenizer.lang_code_to_id[target_code]


def _translate_real(text: str, target_code: str) -> str:
    tokenizer, model = _load_nllb()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    forced_bos = _forced_bos_token_id(tokenizer, target_code)
    generated = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=512)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def _translate_mock(text: str, target_code: str) -> str:
    """Deterministic fake translation so graph wiring is testable without the real model."""
    return f"[MOCK translation to {target_code}] {text}"


def translate_text(text: str, target_language: str) -> TranslationOutput:
    """Not used directly by the graph — see translation_agent_node below. Exposed for standalone testing."""
    target_code = resolve_language_code(target_language)
    translated = _translate_mock(text, target_code) if MOCK_MODE else _translate_real(text, target_code)
    return translated


def translation_agent_node(state: dict) -> dict:
    """
    LangGraph node. Reads `text_to_translate`, `content_type`, and `target_language`
    (set by proceed_to_translation_node or ambiguous_review_node) and produces
    `translated_text` + a full TranslationOutput packet in `translation_output`.

    If `target_language` isn't set in state, falls back to DEFAULT_TARGET_LANGUAGE
    so the graph never breaks on a missing UI selection.
    """
    text = state.get("text_to_translate", "")
    content_type = state.get("content_type", "action_plan")
    target_language = state.get("target_language") or DEFAULT_TARGET_LANGUAGE
    target_code = resolve_language_code(target_language)

    if not text:
        return {
            "translated_text": "",
            "translation_output": None,
            "status": state.get("status", "UNKNOWN") + "_NO_TEXT_TO_TRANSLATE",
        }

    translated = _translate_mock(text, target_code) if MOCK_MODE else _translate_real(text, target_code)

    output = TranslationOutput(
        source_text=text,
        translated_text=translated,
        target_language=target_language,
        target_language_code=target_code,
        content_type=content_type,
    )

    return {
        "translated_text": translated,
        "translation_output": output.model_dump(),
    }
