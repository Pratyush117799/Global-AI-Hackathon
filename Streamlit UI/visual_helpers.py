"""
Visual helpers for the Streamlit UI: attention heatmap overlay + SHAP bar chart.
Both return matplotlib Figure objects so st.pyplot() can render them directly.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def render_attention_overlay(image: Image.Image, attention_matrix: list) -> plt.Figure:
    """3-panel: original / heatmap / overlay — matches groq.ipynb's visualize_attention."""
    attn = np.array(attention_matrix, dtype=float)
    attn_img = Image.fromarray((attn * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)
    attn_resized = np.array(attn_img) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original Leaf", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(attn_resized, cmap="jet")
    axes[1].set_title("Attention Rollout", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(attn_resized, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis("off")

    fig.tight_layout()
    return fig


def render_shap_bar_chart(top_stressors: list) -> plt.Figure:
    """
    top_stressors: list of dicts with keys 'feature', 'shap_contribution', 'direction'
    (matches AnalyticalNodeOutput.top_3_environmental_stressors from schema.py)
    """
    features = [s["feature"] for s in top_stressors]
    values = [s.get("shap_contribution", 0) for s in top_stressors]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution (+ increases risk, − decreases risk)")
    ax.set_title("Top Environmental Stressors", fontsize=11)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def render_risk_gauge(risk_pct: float, risk_band: str) -> plt.Figure:
    """Simple horizontal gauge bar for the 14-day risk percentage."""
    band_colors = {"Low": "#2ca02c", "Moderate": "#ff7f0e", "High": "#d62728"}
    color = band_colors.get(risk_band, "#888888")

    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh([0], [100], color="#eeeeee", height=0.5)
    ax.barh([0], [risk_pct], color=color, height=0.5)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("14-day risk %")
    ax.text(risk_pct + 2, 0, f"{risk_pct:.1f}% ({risk_band})", va="center", fontsize=10)
    fig.tight_layout()
    return fig
