"""STEAR — Spatio-Temporal Evidence-Augmented Retracing."""

from stear.core.visual_retracing import VisualRetracingHook, apply_visual_retracing
from stear.core.contrastive_decoding import TemporalContrastiveHook, apply_temporal_contrastive

__all__ = [
    "VisualRetracingHook",
    "apply_visual_retracing",
    "TemporalContrastiveHook",
    "apply_temporal_contrastive",
]
