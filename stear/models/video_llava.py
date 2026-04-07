"""
STEAR: Spatio-Temporal Evidence-Augmented Retracing
=====================================================

Model adapter: Video-LLaVA

Provides a clean interface for loading Video-LLaVA-7B and attaching
the STEAR hook. Handles model/tokenizer loading, video preprocessing,
and inference with automatic visual feature capture.

Video-LLaVA specifics:
  - Backbone: LLaMA-2-7B with LanguageBind video/image encoders
  - Video tower: LanguageBind-Video-FT, outputs [1, T, 257, 1024]
  - mm_projector: Linear(1024 → 4096), outputs [1, T, 257, 4096]
  - Default: 8 frames, 257 tokens/frame (7×7 patches + CLS)
"""

import os
import sys
from typing import List, Optional, Union

import torch

# Video-LLaVA uses a local package; add its path if needed
_VIDEOLLAVA_CODE = os.path.join(os.path.dirname(__file__), "..", "..", "videollava_code")
if os.path.isdir(_VIDEOLLAVA_CODE) and _VIDEOLLAVA_CODE not in sys.path:
    sys.path.insert(0, _VIDEOLLAVA_CODE)

from stear.core.contrastive_decoding import TemporalContrastiveHook, apply_temporal_contrastive
from stear.core.visual_retracing import VisualRetracingHook, apply_visual_retracing


# Default STEAR configuration (tuned for Video-LLaVA-7B on EventHallusion)
DEFAULT_CONFIG = {
    "starting_layer": 5,
    "ending_layer": 16,
    "injection_ratio": 0.2,
    "uncertainty_threshold": 0.75,
    "topk_frames": 3,
    "tokens_per_frame": 257,
    "ref_layers": list(range(10, 21)),
    "contrastive_alpha": 0.5,
    "topk_frame_ratio": 0.2,
    "homogenize_gamma": 0.5,
    "contrastive_mode": "both",
    "injection_mode": "frame_based",
}


class VideoLLaVAWithSTEAR:
    """
    Video-LLaVA-7B with STEAR hook attached.

    Usage:
        wrapper = VideoLLaVAWithSTEAR.from_pretrained(
            "LanguageBind/Video-LLaVA-7B", device="cuda"
        )
        answer = wrapper.generate(video_path, question)
        wrapper.reset()  # call between videos
    """

    def __init__(self, model, tokenizer, video_processor, hook, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.hook = hook
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        use_contrastive: bool = True,
        stear_config: Optional[dict] = None,
    ) -> "VideoLLaVAWithSTEAR":
        """
        Load Video-LLaVA-7B and attach STEAR.

        Args:
            model_path: HuggingFace model ID or local path.
            device: "cuda" or "cpu".
            use_contrastive: If True, attach full STEAR (retracing + contrastive).
                             If False, attach retracing-only (base hook).
            stear_config: Override default STEAR hyperparameters.

        Returns:
            VideoLLaVAWithSTEAR instance.
        """
        from videollava.model.builder import load_pretrained_model
        from videollava.mm_utils import get_model_name_from_path

        cfg = {**DEFAULT_CONFIG, **(stear_config or {})}
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, processor, _ = load_pretrained_model(
            model_path, None, model_name, device_map=device
        )
        model.eval()

        video_processor = processor["video"] if isinstance(processor, dict) else processor

        if use_contrastive:
            hook = apply_temporal_contrastive(
                model,
                starting_layer=cfg["starting_layer"],
                ending_layer=cfg["ending_layer"],
                injection_ratio=cfg["injection_ratio"],
                uncertainty_threshold=cfg["uncertainty_threshold"],
                topk_frames=cfg["topk_frames"],
                tokens_per_frame=cfg["tokens_per_frame"],
                ref_layers=cfg["ref_layers"],
                contrastive_alpha=cfg["contrastive_alpha"],
                topk_frame_ratio=cfg["topk_frame_ratio"],
                homogenize_gamma=cfg["homogenize_gamma"],
                contrastive_mode=cfg["contrastive_mode"],
                injection_mode=cfg["injection_mode"],
            )
        else:
            hook = apply_visual_retracing(
                model,
                starting_layer=cfg["starting_layer"],
                ending_layer=cfg["ending_layer"],
                injection_ratio=cfg["injection_ratio"],
                uncertainty_threshold=cfg["uncertainty_threshold"],
                topk_frames=cfg["topk_frames"],
                tokens_per_frame=cfg["tokens_per_frame"],
                ref_layers=cfg["ref_layers"],
            )

        hook.enable_auto_capture()
        return cls(model, tokenizer, video_processor, hook, device)

    def generate(
        self,
        video_path: str,
        question: str,
        num_frames: int = 8,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """
        Run inference on a single video-question pair.

        Args:
            video_path: Path to the video file.
            question: Natural language question.
            num_frames: Number of frames to sample.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).

        Returns:
            Generated answer string.
        """
        from videollava.conversation import conv_templates, SeparatorStyle
        from videollava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

        video_tensor = _load_video(video_path, self.video_processor, num_frames, self.device)

        conv = conv_templates["llava_v1"].copy()
        inp = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        self.hook.enable_auto_capture()

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video_tensor,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        self.hook.clear_video_frames()

        output = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        if output.endswith(stop_str):
            output = output[: -len(stop_str)].strip()
        return output

    def reset(self):
        """Clear video state between inferences."""
        self.hook.clear_video_frames()
        self.hook.enable_auto_capture()


# ---------------------------------------------------------------------------
# Standalone inference helpers (for use without the wrapper class)
# ---------------------------------------------------------------------------

def load_model(
    model_path: str, device: str = "cuda"
):
    """
    Load Video-LLaVA model, tokenizer, and video processor.

    Returns:
        (model, tokenizer, video_processor)
    """
    from videollava.model.builder import load_pretrained_model
    from videollava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path, None, model_name, device_map=device
    )
    model.eval()
    video_processor = processor["video"] if isinstance(processor, dict) else processor
    return model, tokenizer, video_processor


def infer(
    model,
    tokenizer,
    video_processor,
    video_path: str,
    question: str,
    device: str = "cuda",
    num_frames: int = 8,
    max_new_tokens: int = 1024,
) -> str:
    """
    Single-sample inference without STEAR (baseline).

    Returns:
        Generated answer string.
    """
    from videollava.conversation import conv_templates, SeparatorStyle
    from videollava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
    from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    video_tensor = _load_video(video_path, video_processor, num_frames, device)

    conv = conv_templates["llava_v1"].copy()
    inp = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video_tensor,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    output = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    if output.endswith(stop_str):
        output = output[: -len(stop_str)].strip()
    return output


def _load_video(
    video_path: str, video_processor, num_frames: int, device: str
) -> torch.Tensor:
    """Load and preprocess a video file into a model-ready tensor."""
    from videollava.mm_utils import process_video

    video = process_video(video_path, video_processor, num_frames=num_frames)
    return video.to(device, dtype=torch.float16)
