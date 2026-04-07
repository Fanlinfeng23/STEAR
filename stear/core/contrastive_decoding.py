"""
STEAR: Spatio-Temporal Evidence-Augmented Retracing
=====================================================

Core module: Temporal Contrastive Decoding

Extends VisualRetracingHook with frame-level temporal contrastive decoding.
When visual retracing is triggered, a negative hidden state is constructed
by perturbing the most-attended video frames (temporal shuffle + homogenization).
The negative branch is propagated through the remaining layers and used to
sharpen the final token distribution:

    logits_final = (1 + alpha) * logits_pos - alpha * logits_neg

This amplifies predictions that are robust to temporal disruption and
suppresses those that rely on spurious temporal patterns.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .visual_retracing import VisualRetracingHook


# ---------------------------------------------------------------------------
# Frame-level temporal perturbation utilities
# ---------------------------------------------------------------------------

def select_topk_frame_indices(
    frames_3d: torch.Tensor,
    attn_weights: Optional[torch.Tensor],
    topk_ratio: float = 0.3,
) -> torch.Tensor:
    """
    Return indices of the top-k most-attended frames.

    Args:
        frames_3d: [T, P, D] video features.
        attn_weights: [T*P] or [T, P] attention weights, or None.
        topk_ratio: Fraction of frames to select.

    Returns:
        1-D LongTensor of selected frame indices (CPU).
    """
    T, P, D = frames_3d.shape
    if attn_weights is None:
        return torch.arange(T)

    w = attn_weights.reshape(T, P) if attn_weights.dim() == 1 else attn_weights
    frame_attn = w.mean(dim=1)
    k = max(1, int(T * topk_ratio))
    _, idx = torch.topk(frame_attn, k=k)
    return idx.cpu()


def temporal_shuffle(frames_3d: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
    """Randomly permute the selected frames along the time axis."""
    frames_neg = frames_3d.clone()
    if len(frame_indices) > 0:
        selected = frames_3d[frame_indices].clone()
        perm = torch.randperm(len(frame_indices), device=frames_3d.device)
        frames_neg[frame_indices] = selected[perm]
    return frames_neg


def temporal_homogenize(
    frames_3d: torch.Tensor, frame_indices: torch.Tensor, gamma: float = 0.4
) -> torch.Tensor:
    """
    Blend selected frames toward their temporal mean.

    output[i] = (1 - gamma) * frame[i] + gamma * mean(selected_frames)
    """
    frames_neg = frames_3d.clone()
    if len(frame_indices) > 0:
        selected = frames_3d[frame_indices]
        mean_frame = selected.mean(dim=0, keepdim=True)
        frames_neg[frame_indices] = (1 - gamma) * selected + gamma * mean_frame
    return frames_neg


def apply_temporal_perturbation(
    frames_3d: torch.Tensor,
    attn_weights: Optional[torch.Tensor],
    mode: str = "both",
    topk_ratio: float = 0.3,
    homo_gamma: float = 0.4,
) -> torch.Tensor:
    """
    Apply temporal perturbation to construct a negative video representation.

    Args:
        frames_3d: [T, P, D] video features.
        attn_weights: Attention weights for frame selection (None = use all frames).
        mode: "shuffle", "homo", or "both".
        topk_ratio: Fraction of frames to perturb.
        homo_gamma: Homogenization blend weight.

    Returns:
        [T, P, D] perturbed video features.
    """
    indices = select_topk_frame_indices(frames_3d, attn_weights, topk_ratio=topk_ratio)
    frames_neg = frames_3d.clone()
    if mode in ("homo", "both"):
        frames_neg = temporal_homogenize(frames_neg, indices, gamma=homo_gamma)
    if mode in ("shuffle", "both"):
        frames_neg = temporal_shuffle(frames_neg, indices)
    return frames_neg


def select_topk_patches(
    frames_3d: torch.Tensor,
    attn_weights: Optional[torch.Tensor],
    topk_ratio: float = 0.3,
) -> torch.Tensor:
    """
    Select top-k% patches by attention weight for patch-based injection.

    Returns:
        [k, D] selected patch features.
    """
    T, P, D = frames_3d.shape
    total = T * P
    if attn_weights is None:
        return frames_3d.reshape(total, D)
    w = attn_weights.reshape(-1) if attn_weights.dim() == 2 else attn_weights
    k = max(1, int(total * topk_ratio))
    _, idx = torch.topk(w, k=k)
    return frames_3d.reshape(total, D)[idx]


# ---------------------------------------------------------------------------
# Temporal Contrastive Decoding Hook
# ---------------------------------------------------------------------------

class TemporalContrastiveHook(VisualRetracingHook):
    """
    Extends VisualRetracingHook with frame-level temporal contrastive decoding.

    On each uncertainty-triggered injection:
      1. The base class injects visual memory into the positive branch (normal forward).
      2. This class additionally constructs a negative hidden state by replacing
         the video tokens with temporally perturbed features.
      3. At lm_head, the negative branch is propagated through remaining layers
         and the final logits are sharpened:
             logits = (1 + alpha) * logits_pos - alpha * logits_neg

    Injection mode:
      - "frame_based": select top-k frames for visual memory (default, same as base).
      - "patch_based": select top-k% patches for visual memory.
    """

    def __init__(
        self,
        model,
        starting_layer: int = 5,
        ending_layer: int = 16,
        injection_ratio: float = 0.2,
        uncertainty_threshold: float = 0.75,
        topk_frames: int = 3,
        tokens_per_frame: int = 257,
        ref_layers: Optional[List[int]] = None,
        # Contrastive decoding parameters
        contrastive_alpha: float = 0.5,
        topk_frame_ratio: float = 0.3,
        homogenize_gamma: float = 0.4,
        contrastive_mode: str = "both",
        injection_mode: str = "frame_based",
    ):
        """
        Args:
            contrastive_alpha: Alpha for logit sharpening (0 = no contrastive effect).
            topk_frame_ratio: Fraction of frames to perturb for negative branch.
            homogenize_gamma: Blend weight for temporal homogenization.
            contrastive_mode: "shuffle", "homo", or "both".
            injection_mode: "frame_based" or "patch_based" for visual memory selection.
            (remaining args: see VisualRetracingHook)
        """
        self.contrastive_alpha = contrastive_alpha
        self.topk_frame_ratio = topk_frame_ratio
        self.homogenize_gamma = homogenize_gamma
        self.contrastive_mode = contrastive_mode
        self.injection_mode = injection_mode

        # Contrastive state (per decode step)
        self._contrastive_pending: bool = False
        self._h_neg_at_trigger: Optional[torch.Tensor] = None
        self._trigger_layer_idx: Optional[int] = None
        self._layer_fwd_kwargs: Dict[int, Dict[str, Any]] = {}
        self._orig_layer_forwards: Dict[int, Any] = {}
        self.contrastive_count: int = 0

        super().__init__(
            model,
            starting_layer=starting_layer,
            ending_layer=ending_layer,
            injection_ratio=injection_ratio,
            uncertainty_threshold=uncertainty_threshold,
            topk_frames=topk_frames,
            tokens_per_frame=tokens_per_frame,
            ref_layers=ref_layers,
        )

        self._wrap_lm_head_for_contrastive()

    # ------------------------------------------------------------------
    # Override: layer wrapping to add contrastive state capture
    # ------------------------------------------------------------------

    def _wrap_all_layers_for_dynamic_selection(self):
        """
        Override base class to additionally:
          - Store per-layer forward kwargs for negative branch replay.
          - Cache original layer forwards for unhooked negative propagation.
          - Trigger negative hidden state construction on injection.
        """
        hook_self = self
        num_layers = len(self.model.model.layers)

        for layer_idx in range(self.starting_layer + 1, min(self.ending_layer, num_layers)):
            layer = self.model.model.layers[layer_idx]
            orig_fwd = layer.forward
            hook_self._orig_layer_forwards[layer_idx] = orig_fwd
            _idx = layer_idx

            def make_layer_hook(orig, idx):
                def hooked(hidden_states, *args, **kwargs):
                    hook_self._layer_fwd_kwargs[idx] = {"args": args, "kwargs": kwargs}
                    outputs = orig(hidden_states, *args, **kwargs)

                    if hook_self.video_frames is None or hook_self._visual_retracing_event:
                        return outputs

                    try:
                        layer_out = outputs[0] if isinstance(outputs, tuple) else outputs
                        u = hook_self._compute_token_uncertainty(layer_out)
                        hook_self.uncertainty_stats.append(u)

                        if u > hook_self.uncertainty_threshold:
                            next_idx = idx + 1
                            if next_idx < num_layers:
                                hook_self._visual_retracing_event = True
                                hook_self._pending_inject_layer = next_idx
                                hook_self.injection_count += 1
                                # Positive branch: visual memory injection
                                hook_self._activate_injection_with_mode(next_idx, layer_out)
                                # Negative branch: temporal perturbation
                                hook_self._prepare_negative_hidden(idx, layer_out)
                    except Exception:
                        pass

                    return outputs
                return hooked

            layer.forward = make_layer_hook(orig_fwd, layer_idx)

        # Cache original forwards for layers beyond ending_layer (used in neg branch)
        for layer_idx in range(self.ending_layer, num_layers):
            if layer_idx not in hook_self._orig_layer_forwards:
                hook_self._orig_layer_forwards[layer_idx] = (
                    self.model.model.layers[layer_idx].forward
                )

    # ------------------------------------------------------------------
    # Injection with mode support (frame_based / patch_based)
    # ------------------------------------------------------------------

    def _activate_injection_with_mode(self, layer_idx: int, trigger_hidden: torch.Tensor):
        """Injection supporting both frame_based and patch_based visual memory."""
        if self.video_frames is None:
            return

        try:
            vf = self.video_frames
            frames_3d = vf.squeeze(0) if vf.dim() == 4 else vf
            if frames_3d.dim() != 3:
                return
            num_frames, tpf, vision_dim = frames_3d.shape
        except Exception:
            return

        seq_len = trigger_hidden.shape[1]
        total_video_tokens = num_frames * tpf
        if seq_len > 1 and self._video_token_start is None:
            self._video_token_start = 1 if seq_len >= total_video_tokens + 1 else 0
            self._video_token_end = self._video_token_start + total_video_tokens
            self._num_frames = num_frames

        attn_weights = self._get_video_attn_weights(num_frames, tpf)

        if self.injection_mode == "patch_based":
            visual_memory = select_topk_patches(
                frames_3d, attn_weights, topk_ratio=self.topk_frame_ratio
            )
        else:
            selected = self._select_topk_frames(frames_3d, num_frames, tpf)
            visual_memory = selected.reshape(-1, vision_dim)

        device = trigger_hidden.device
        dtype = trigger_hidden.dtype
        visual_memory = visual_memory.to(device, dtype=dtype)

        if self._ref_attn_accum is not None and self._ref_attn_count > 0:
            self._ref_attn_prev = self._ref_attn_accum / self._ref_attn_count
        self._ref_attn_accum = None
        self._ref_attn_count = 0

        mlp = self.model.model.layers[layer_idx].mlp
        orig_mlp_fwd = mlp.forward

        try:
            up_scale = torch.mean(torch.abs(mlp.up_proj.weight)).clamp(min=1e-8)
            dn_scale = torch.mean(torch.abs(mlp.down_proj.weight)).clamp(min=1e-8)
            vm_scale = torch.mean(torch.abs(visual_memory)).clamp(min=1e-8)
            V = visual_memory.to(device=device, dtype=dtype)
            adpt_w1 = (up_scale / vm_scale) * V
            adpt_w2 = (dn_scale / vm_scale) * V.T
        except Exception:
            adpt_w1 = visual_memory.to(device=device, dtype=dtype)
            adpt_w2 = visual_memory.to(device=device, dtype=dtype).T

        _w1, _w2, _ratio = adpt_w1, adpt_w2, self.injection_ratio

        def one_shot_mlp(x):
            ffn_out = orig_mlp_fwd(x)
            mlp.forward = orig_mlp_fwd
            try:
                B, S, H = x.shape
                x_flat = x.reshape(B * S, H)
                adapter_out = torch.matmul(
                    torch.matmul(x_flat, _w1.T), _w2.T
                ).reshape(B, S, H)
                mean_ffn = torch.mean(torch.abs(ffn_out)).clamp(min=1e-8)
                mean_adp = torch.mean(torch.abs(adapter_out)).clamp(min=1e-8)
                norm_adapter = (mean_ffn / mean_adp) * adapter_out
                result = (1 - _ratio) * ffn_out + _ratio * norm_adapter
                if torch.isnan(result).any() or torch.isinf(result).any():
                    return ffn_out
                return result
            except Exception:
                return ffn_out

        mlp.forward = one_shot_mlp

    # ------------------------------------------------------------------
    # Negative hidden state construction
    # ------------------------------------------------------------------

    def _prepare_negative_hidden(self, trigger_layer_idx: int, trigger_hidden: torch.Tensor):
        """
        Build h_neg by replacing video tokens with temporally perturbed features.

        If video token positions are known, patches the token slice directly.
        Otherwise falls back to attention-weighted delta perturbation.
        """
        try:
            frames_3d = self._get_frames_3d()
            if frames_3d is None:
                return

            num_frames, tpf, vision_dim = frames_3d.shape
            device = trigger_hidden.device
            dtype = trigger_hidden.dtype

            attn_weights = self._get_video_attn_weights(num_frames, tpf)
            frames_neg = apply_temporal_perturbation(
                frames_3d.to(device, dtype),
                attn_weights.to(device, dtype) if attn_weights is not None else None,
                mode=self.contrastive_mode,
                topk_ratio=self.topk_frame_ratio,
                homo_gamma=self.homogenize_gamma,
            )

            h_neg = trigger_hidden.clone()
            B, S, D = h_neg.shape
            total_vid = num_frames * tpf

            if (
                self._video_token_start is not None
                and self._video_token_end is not None
                and self._video_token_end <= S
                and self._video_token_end - self._video_token_start == total_vid
            ):
                vid_s = self._video_token_start
                vid_e = self._video_token_end
                h_neg[0, vid_s:vid_e, :] = frames_neg.reshape(total_vid, vision_dim)
            else:
                h_neg = self._delta_perturbation(trigger_hidden, frames_3d, frames_neg, device, dtype)

            self._h_neg_at_trigger = h_neg
            self._trigger_layer_idx = trigger_layer_idx
            self._contrastive_pending = True
            self.contrastive_count += 1

        except Exception as e:
            if not getattr(self, "_neg_error_logged", False):
                import traceback
                print(f"[STEAR] _prepare_negative_hidden error: {e}")
                traceback.print_exc()
                self._neg_error_logged = True

    def _get_video_attn_weights(
        self, num_frames: int, tpf: int
    ) -> Optional[torch.Tensor]:
        """Extract video-token attention slice from accumulated reference attention."""
        if self._ref_attn_prev is None or self._video_token_start is None:
            return None
        vid_s = self._video_token_start
        vid_e = self._video_token_end
        if vid_e > self._ref_attn_prev.shape[0]:
            return None
        return self._ref_attn_prev[vid_s:vid_e].cpu()

    def _delta_perturbation(
        self,
        h_pos: torch.Tensor,
        frames_orig: torch.Tensor,
        frames_neg: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Fallback: apply attention-weighted perturbation delta to the last token.
        Used when video token positions in the sequence are unknown.
        """
        T, P, D = frames_orig.shape
        total_vid = T * P
        delta = (frames_neg - frames_orig).reshape(total_vid, D).to(device, dtype)

        attn = None
        if (
            self._ref_attn_prev is not None
            and self._video_token_start is not None
            and self._video_token_end is not None
        ):
            vid_s = self._video_token_start
            vid_e = self._video_token_end
            if vid_e <= self._ref_attn_prev.shape[0] and (vid_e - vid_s) == total_vid:
                attn = self._ref_attn_prev[vid_s:vid_e].to(device, dtype)

        if attn is None or attn.shape[0] != total_vid:
            attn = torch.ones(total_vid, device=device, dtype=dtype) / total_vid

        weighted_delta = (attn.unsqueeze(-1) * delta).sum(0)
        h_neg = h_pos.clone()
        h_neg[0, -1, :] = h_neg[0, -1, :] + weighted_delta
        return h_neg

    def _get_frames_3d(self) -> Optional[torch.Tensor]:
        """Return video features as [T, P, D]."""
        if self.video_frames is None:
            return None
        vf = self.video_frames
        if vf.dim() == 4:
            return vf.squeeze(0)
        if vf.dim() == 3:
            return vf
        if vf.dim() == 2:
            tpf = self.tokens_per_frame
            T = vf.shape[0] // tpf
            return vf.reshape(T, tpf, vf.shape[-1])
        return None

    # ------------------------------------------------------------------
    # lm_head wrapper: contrastive logit sharpening
    # ------------------------------------------------------------------

    def _wrap_lm_head_for_contrastive(self):
        """
        Wrap lm_head.forward to apply contrastive logit sharpening when
        a negative branch has been prepared.
        """
        orig_lm_head_fwd = self.model.lm_head.forward
        hook_self = self

        def lm_head_with_contrastive(hidden_states):
            logits_pos = orig_lm_head_fwd(hidden_states)

            if not hook_self._contrastive_pending:
                return logits_pos

            hook_self._contrastive_pending = False
            h_neg = hook_self._h_neg_at_trigger
            trigger_idx = hook_self._trigger_layer_idx
            hook_self._h_neg_at_trigger = None
            hook_self._trigger_layer_idx = None

            if h_neg is None or trigger_idx is None:
                return logits_pos

            try:
                logits_neg = hook_self._run_negative_branch(
                    h_neg, trigger_idx, hidden_states.device, hidden_states.dtype
                )
                if logits_neg is None:
                    return logits_pos

                alpha = hook_self.contrastive_alpha
                final = (1 + alpha) * logits_pos - alpha * logits_neg

                if torch.isnan(final).any() or torch.isinf(final).any():
                    return logits_pos
                return final

            except Exception as e:
                if not getattr(hook_self, "_lm_head_error_logged", False):
                    import traceback
                    print(f"[STEAR] lm_head contrastive error: {e}")
                    traceback.print_exc()
                    hook_self._lm_head_error_logged = True
                return logits_pos

        self.model.lm_head.forward = lm_head_with_contrastive

    def _run_negative_branch(
        self,
        h_neg: torch.Tensor,
        trigger_layer_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Propagate h_neg through layers [trigger_layer_idx+1, num_layers) using
        the original (unhooked) layer forwards, then apply final norm and lm_head.

        Returns logits_neg or None on failure.
        """
        num_layers = len(self.model.model.layers)
        h = h_neg.to(device, dtype)

        with torch.no_grad():
            for layer_idx in range(trigger_layer_idx + 1, num_layers):
                orig_fwd = self._orig_layer_forwards.get(layer_idx)
                if orig_fwd is None:
                    orig_fwd = self.model.model.layers[layer_idx].forward

                stored = self._layer_fwd_kwargs.get(layer_idx, {})
                args = stored.get("args", ())
                kwargs = dict(stored.get("kwargs", {}))
                kwargs["use_cache"] = False
                kwargs.pop("cache_position", None)

                try:
                    out = orig_fwd(h, *args, **kwargs)
                    h = out[0] if isinstance(out, tuple) else out
                except Exception:
                    try:
                        out = orig_fwd(h)
                        h = out[0] if isinstance(out, tuple) else out
                    except Exception:
                        return None

            h = self.model.model.norm(h)
            logits_neg = F.linear(h, self.model.lm_head.weight)
            if hasattr(self.model.lm_head, "bias") and self.model.lm_head.bias is not None:
                logits_neg = logits_neg + self.model.lm_head.bias

        return logits_neg

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_state(self):
        super()._reset_state()
        self._contrastive_pending = False
        self._h_neg_at_trigger = None
        self._trigger_layer_idx = None
        self._layer_fwd_kwargs.clear()
        self.contrastive_count = 0

    def clear_video_frames(self):
        if self.total_tokens > 0:
            print(
                f"[STEAR] contrastive_triggers={self.contrastive_count}/"
                f"{self.total_tokens} tokens"
            )
        super().clear_video_frames()


def apply_temporal_contrastive(
    model,
    starting_layer: int = 5,
    ending_layer: int = 16,
    injection_ratio: float = 0.2,
    uncertainty_threshold: float = 0.75,
    topk_frames: int = 3,
    tokens_per_frame: int = 257,
    ref_layers: Optional[List[int]] = None,
    contrastive_alpha: float = 0.5,
    topk_frame_ratio: float = 0.2,
    homogenize_gamma: float = 0.5,
    contrastive_mode: str = "both",
    injection_mode: str = "frame_based",
) -> TemporalContrastiveHook:
    """
    Attach a TemporalContrastiveHook to a Video-LLaVA model.

    This is the recommended entry point for STEAR's full pipeline
    (visual retracing + temporal contrastive decoding).

    Args:
        model: Video-LLaVA LlamaForCausalLM.
        starting_layer: Start of uncertainty monitoring range.
        ending_layer: End of uncertainty monitoring range (exclusive).
        injection_ratio: Visual memory blend weight alpha.
        uncertainty_threshold: Entropy threshold gamma.
        topk_frames: Top-k frames for visual memory (frame_based mode).
        tokens_per_frame: Visual tokens per frame (257 for LanguageBind).
        ref_layers: Reference layers for attention-based frame selection.
        contrastive_alpha: Logit sharpening weight.
        topk_frame_ratio: Fraction of frames to perturb for negative branch.
        homogenize_gamma: Temporal homogenization blend weight.
        contrastive_mode: "shuffle", "homo", or "both".
        injection_mode: "frame_based" or "patch_based".

    Returns:
        Attached TemporalContrastiveHook instance.
    """
    return TemporalContrastiveHook(
        model,
        starting_layer=starting_layer,
        ending_layer=ending_layer,
        injection_ratio=injection_ratio,
        uncertainty_threshold=uncertainty_threshold,
        topk_frames=topk_frames,
        tokens_per_frame=tokens_per_frame,
        ref_layers=ref_layers,
        contrastive_alpha=contrastive_alpha,
        topk_frame_ratio=topk_frame_ratio,
        homogenize_gamma=homogenize_gamma,
        contrastive_mode=contrastive_mode,
        injection_mode=injection_mode,
    )
