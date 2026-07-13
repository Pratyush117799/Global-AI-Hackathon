# Day 1 — Vision Node: Test Results & Analysis

**Model:** `wambugu71/crop_leaf_diseases_vit` (pretrained, open-weight, no fine-tuning applied)
**Environment:** Google Colab, T4 GPU
**Threshold:** `CONFIDENCE_THRESHOLD = 0.75` (HITL gate)

---

## Test 1 — Potato/tomato leaf with visible lesions (`1.jpg`)

| Field | Value |
|---|---|
| `disease_class` | `Potato___Early_Blight` |
| `confidence_score` | 0.9702 (97.0%) |
| `is_ambiguous` | `False` |

**Result:** High-confidence, non-ambiguous classification. Correctly passes the HITL gate and would proceed to the Analytical Node + Orchestrator without human review.

**Attention rollout observations:** The heatmap is only loosely aligned with the visible necrotic lesions on the leaf. A meaningful share of attention mass concentrates in the image corners and along the frame edges rather than tightly around the brown/tan spots that a plant pathologist would flag. This is a known behavior of full-depth attention rollout on ViT — averaging attention across *every* layer (including early layers, which tend to attend broadly to low-level texture/edges rather than semantic content) diffuses the map. The photo also contains a printed catalog number (`5487924`) in the bottom-right corner, which is high-contrast against the background and is a plausible driver of the bottom-right hotspot — worth being aware of as a confound, since real farmer-submitted photos won't have this artifact.

**Action item before the demo:** consider computing rollout over only the last 4–6 layers (rather than all layers), or switching head-fusion from `mean` to `max`, both of which typically sharpen localization around the discriminative region instead of diffusing it. Not a blocker, but worth 20–30 minutes if time allows — a tighter heatmap will land better in the judge-facing side-by-side.

---

## Test 2 — Dried/damaged foliage, different specimen (`2.jpg`)

| Field | Value |
|---|---|
| `disease_class` | `Corn___Common_Rust` |
| `confidence_score` | 0.494 (49.4%) |
| `is_ambiguous` | `True` (below 0.75 threshold) |

**Result:** Low-confidence prediction, correctly flagged as ambiguous. This is exactly the case the HITL gate exists for: the model's top guess (corn common rust) doesn't visually match what's actually a wilted, necrotic potato/tomato-type leaf, and the model itself signals that uncertainty through a sub-50% confidence score rather than forcing a confident-looking wrong answer.

**Why this matters for your pitch:** this is a real, unstaged example of the safety gate doing its job — the model encountering an input outside its confident zone and deferring to a human reviewer instead of silently guessing. Recommend using this exact input as your **live "ambiguous case" demo** on Day 3 instead of a synthetic one — it's more credible to judges than a manufactured low-confidence example, and it's genuine evidence the HITL gate isn't decorative.

---

## Summary / Go-forward decisions

1. **Node is functioning correctly end-to-end**: inference, confidence scoring, attention rollout, and the HITL threshold check all work as specified.
2. **Attention localization is the one soft spot** — usable for the demo as-is, but a quick tweak (fewer layers in rollout, or max-fusion) would visibly tighten it. Low-risk, optional polish.
3. **Keep both test images** — Test 1 (high-confidence, correct-looking) and Test 2 (low-confidence, correctly gated) are your two canonical demo cases going forward. Don't lose them.
4. **No fine-tuning needed** for now — the pretrained model is producing sensible, differentiated outputs on real (if imperfect) test photos. Revisit only if your actual demo crops fall outside its label set.

---

*Next: Tabular Node (Approach B — environmental stress index) + XGBoost + SHAP.*
