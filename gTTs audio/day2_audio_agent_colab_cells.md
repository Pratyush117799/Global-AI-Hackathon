# Day 2 — Adding the Audio Agent (gTTS) to your LangGraph notebook

Add these cells after your Translation Agent cells. This is the last stage before
PDF report assembly.

**Important — different from the Translation Agent's model download:** gTTS calls
Google's translate-TTS endpoint live, over the network, every time you synthesize audio.
It needs real internet access at *call time* (not just once, like NLLB's one-time model
download). Colab has this by default, so no special setup — just don't run it from an
offline/restricted environment.

**Graceful degradation, tested and verified:** gTTS does not support every language NLLB
translates to. Confirmed directly against `gtts.lang.tts_langs()`: **Odia is translatable
but not speakable** (as of this check — Google could add it later, easy to re-verify with
one line). When this happens, the Audio Agent does NOT crash the graph — it returns
`tts_status = "skipped_unsupported_language"`, keeps the translated *text* available for
the PDF report, and simply has no audio file for that language. If you demo Odia, be ready
to say "written report is fully localized; audio synthesis is a to-be-expanded feature for
this specific language" — it's a legitimate scoping statement, not a bug.

---

## New cell — install deps

```python
!pip install -q gTTS
```

## New cell — upload the 2 new files

Upload `audio_schema.py` and `audio_agent.py` via the Colab file pane (same folder as
`translation_agent.py`, `language_config.py`).

## New cell — flip MOCK_MODE

Open `audio_agent.py` in Colab's file editor and change:
```python
MOCK_MODE = True
```
to:
```python
MOCK_MODE = False
```

## Update `GraphState` — add 1 new field

```python
class GraphState(TypedDict, total=False):
    # ...(all your existing fields)...
    audio_output: dict   # NEW
```

## New cell — import the audio node

```python
from audio_agent import audio_agent_node
```

## Update graph wiring — insert audio_agent between translation_agent and END

```python
builder.add_node("audio_agent", audio_agent_node)   # NEW

# ...(all your existing add_node / add_conditional_edges calls stay the same)...

builder.add_edge("ambiguous_review", "translation_agent")
builder.add_edge("proceed_to_translation", "translation_agent")
builder.add_edge("translation_agent", "audio_agent")   # was -> END, now -> audio_agent
builder.add_edge("audio_agent", END)                    # NEW

graph = builder.compile()
print("Graph compiled OK")
```

## Update your test harness — print the audio result too

```python
for run in test_runs:
    result = graph.invoke(run)
    print(json.dumps({
        "route": result["route"],
        "content_type": result["content_type"],
        "translated_text": result["translated_text"],
        "audio_output": result["audio_output"],
    }, indent=2, ensure_ascii=False))
```

Play the generated file directly in Colab to sanity-check the audio sounds right:
```python
from IPython.display import Audio
Audio(result["audio_output"]["audio_file_path"])
```

---

## Verified in sandbox (mock mode, before handing off)

3 cases run end-to-end through the full graph (vision → analytical → pathologist → climate
→ HITL → translation → audio):

| Case | Language | Result |
|---|---|---|
| Confident + Moderate | Hindi | `tts_status: success`, file at `audio_output/action_plan_hi.mp3` |
| Ambiguous + High | Tamil | `tts_status: success`, file at `audio_output/review_notice_ta.mp3` |
| Confident + Low | Odia | `tts_status: skipped_unsupported_language`, translated text preserved, audio correctly omitted with a clear reason instead of crashing |

Filenames are deterministic (`{content_type}_{gtts_code}.mp3`), so the PDF assembly step
can predict the audio path without needing it passed around separately.

**Not verified here** (needs your Colab session, which has real internet access): the
actual audio quality/pronunciation for each language, and that gTTS's live API doesn't
rate-limit you during repeated testing — if you hit `429` errors during heavy testing,
add a short `time.sleep()` between calls or cache repeated inputs.

---

*Next: reconcile with the PDF report generator from your other session — it needs to pull
in `translation_output` (translated text) and `audio_output` (mp3 path, or the skip
message for unsupported languages) alongside the heatmap and SHAP chart.*
