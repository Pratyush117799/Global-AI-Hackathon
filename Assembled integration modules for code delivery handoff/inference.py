"""
Vision Node — real inference module.

This packages the exact tested logic from groq.ipynb (ViT load, attention
rollout, confidence scoring, HITL threshold) into an importable module. Until
this file existed, pipeline.py's `from vision_pipeline.inference import
run_vision_node` was silently failing and every run — including anything you
thought was "real" — was actually using fallback demo data. This file closes
that gap.

Contract: run_vision_node(image: PIL.Image) -> VisionNodeOutput
matching exactly what pipeline.py expects (it calls .model_dump() on the result).
"""

import torch
from typing import List
from pydantic import BaseModel
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

MODEL_NAME = "wambugu71/crop_leaf_diseases_vit"
CONFIDENCE_THRESHOLD = 0.75

device = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None


class VisionNodeOutput(BaseModel):
    disease_class: str
    confidence_score: float
    attention_matrix: List[List[float]]
    is_ambiguous: bool


def _load_model():
    """Lazy-load once per session, cached — same pattern as translation_agent.py/audio_agent.py."""
    global _processor, _model
    if _model is not None:
        return _processor, _model

    _processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    _model = ViTForImageClassification.from_pretrained(MODEL_NAME, output_attentions=True)
    _model.eval()
    _model.to(device)
    return _processor, _model


def _attention_rollout(attentions):
    tokens = attentions[0].size(-1)
    result = torch.eye(tokens, device=attentions[0].device).unsqueeze(0)
    for attn in attentions:
        attn_avg = attn.mean(dim=1)
        attn_avg = attn_avg + torch.eye(tokens, device=attn_avg.device)
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        result = torch.matmul(attn_avg, result)
    return result


def run_vision_node(image: Image.Image) -> VisionNodeOutput:
    processor, model = _load_model()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_idx = int(torch.argmax(probs))
    confidence = float(probs[pred_idx])
    label = model.config.id2label[pred_idx]

    rolled = _attention_rollout(outputs.attentions)
    cls_to_patches = rolled[0, 0, 1:]
    grid_size = int(len(cls_to_patches) ** 0.5)
    attn_grid = cls_to_patches.reshape(grid_size, grid_size).cpu().numpy()
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)

    return VisionNodeOutput(
        disease_class=label,
        confidence_score=round(confidence, 4),
        attention_matrix=attn_grid.tolist(),
        is_ambiguous=confidence < CONFIDENCE_THRESHOLD,
    )
