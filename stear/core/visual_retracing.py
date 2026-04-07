"""
STEAR: Spatio-Temporal Evidence-Augmented Retracing
=====================================================

Core module: Visual Retracing Hook

Implements uncertainty-triggered visual memory injection for Video-LLaVA.
At each decode step, the model monitors token-level uncertainty across a
configurable layer range. When uncertainty exceeds a threshold, the most
attended video frames are re-injected into the FFN of the next layer via
a one-shot visual memory adapter.

Key design choices:
  - Uncertainty metric: top-10 logits entropy in float32, normalized by log(10)
  - Frame selection: attention-weighted top-k, aggregated across reference layers
  - Injection: one-shot MLP adapter with magnitude normalization
  - Flash Attention compatibility: Q·K computed manually from q_proj/k_proj weights
"""

import math
from typing import List, Optional

import torch
import torch.nn.functional as F


class VisualRetracingHook:
    """
    Uncertainty-triggered visual memory injection hook for Video-LLaVA.

    Pipeline per decode step:
      1. Reference layers (ref_layers) accumulate attention weights over the
         last query token via hooked self_attn.forward calls.
      2. After each layer in [starting_layer, ending_layer], token uncertainty
         is computed from the top-10 logit entropy.
      3. On the first layer exceeding the threshold, the top-k most-attended
         video frames are selected and injected into the next layer's MLP as
         a one-shot visual memory adapter.
      4. The injection fires exactly once per decode step.

    Visual memory adapter formula (magnitude-normalized):
        adapter_out  = x @ V^T @ V
        norm_adapter = (mean_abs(ffn_out) / mean_abs(adapter_out)) * adapter_out
        output       = (1 - alpha) * ffn_out + alpha * norm_adapter

    where V = [N_v, hidden_dim] is the selected visual token matrix, scaled
    by the MLP weight magnitudes to align with the FFN output scale.
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
    ):
        """
        Args:
            model: Video-LLaVA LlamaForCausalLM instance.
            starting_layer: First layer to monitor for uncertainty (inclusive).
            ending_layer: Last layer to monitor for uncertainty (exclusive).
            injection_ratio: Alpha — blend weight for the visual memory adapter.
            uncertainty_threshold: Gamma — entropy threshold that triggers injection.
            topk_frames: Number of top-attended frames to include in visual memory.
            tokens_per_frame: Visual tokens per frame (257 for LanguageBind video tower).
            ref_layers: Layers whose attention weights are used for frame selection.
                        Defaults to layers 10–20.
        """
        self.model = model
        self.starting_layer = starting_layer
        self.ending_layer = ending_layer
        self.injection_ratio = injection_ratio
        self.uncertainty_threshold = uncertainty_threshold
        self.topk_frames = topk_frames
        self.tokens_per_frame = tokens_per_frame
        self.ref_layers = ref_layers if ref_layers is not None else list(range(10, 21))

        # Video feature storage (set via set_video_frames or auto-capture)
        self.video_frames: Optional[torch.Tensor] = None
        self.auto_capture_enabled: bool = False

        # Per-inference statistics
        self.injection_count: int = 0
        self.total_tokens: int = 0
        self.uncertainty_stats: List[float] = []

        # Reference-layer attention accumulator (averaged across ref_layers)
        self._ref_attn_prev: Optional[torch.Tensor] = None
        self._ref_attn_accum: Optional[torch.Tensor] = None
        self._ref_attn_count: int = 0

        # Video token position in the sequence (determined at prefill)
        self._video_token_start: Optional[int] = None
        self._video_token_end: Optional[int] = None
        self._num_frames: Optional[int] = None

        # Per-step injection state
        self._pending_inject_layer: Optional[int] = None
        self._visual_retracing_event: bool = False

        self._register_ref_layer_hooks()
        self._wrap_model_forward()
        self._wrap_vision_encoder()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_video_frames(self, frames: torch.Tensor):
        """Manually provide projected video features [T, P, D] or [1, T, P, D]."""
        self.video_frames = frames
        self.auto_capture_enabled = False
        self._reset_state()

    def enable_auto_capture(self):
        """Enable automatic capture of mm_projector output as video features."""
        self.auto_capture_enabled = True
        self.video_frames = None
        self._reset_state()

    def clear_video_frames(self):
        """Clear video features and reset per-inference state after generation."""
        self.video_frames = None
        self._video_token_start = None
        self._video_token_end = None
        self._num_frames = None
        self._ref_attn_prev = None
        self._ref_attn_accum = None
        self._ref_attn_count = 0
        self._pending_inject_layer = None
        self._visual_retracing_event = False
        if self.total_tokens > 0:
            avg_u = (
                sum(self.uncertainty_stats) / len(self.uncertainty_stats)
                if self.uncertainty_stats else 0.0
            )
            print(
                f"[STEAR] tokens={self.total_tokens}  "
                f"injections={self.injection_count}  "
                f"rate={self.injection_count / self.total_tokens * 100:.1f}%  "
                f"avg_uncertainty={avg_u:.4f}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self):
        self.injection_count = 0
        self.total_tokens = 0
        self.uncertainty_stats = []
        self._ref_attn_prev = None
        self._ref_attn_accum = None
        self._ref_attn_count = 0
        self._video_token_start = None
        self._video_token_end = None
        self._num_frames = None
        self._pending_inject_layer = None
        self._visual_retracing_event = False

    def _compute_token_uncertainty(self, hidden_states: torch.Tensor) -> float:
        """
        Compute token uncertainty as top-10 logit entropy, normalized by log(10).

        Follows the original paper exactly:
          1. Apply final RMSNorm to hidden_states.
          2. Project to vocabulary logits (float32).
          3. Take top-10 scores, compute softmax, compute entropy / log(10).

        Args:
            hidden_states: [B, S, D] layer output (before final norm).

        Returns:
            Scalar uncertainty in [0, 1].
        """
        try:
            final_norm = self.model.model.norm
            lm_head = self.model.lm_head
            device = hidden_states.device
            with torch.no_grad():
                norm_h = final_norm(hidden_states.float())
                last_h = norm_h[:, -1:, :].float()
                weight = lm_head.weight.to(device=device, dtype=torch.float32)
                logits = F.linear(last_h, weight)
            logits = logits[0, 0, :]
            top_k_scores, _ = torch.topk(logits, 10)
            probs = F.softmax(top_k_scores, dim=-1)
            entropy = torch.sum(-probs * torch.log(probs + 1e-10)) / math.log(10)
            return float(entropy.item())
        except Exception:
            return 0.5  # fallback: do not trigger injection

    # ------------------------------------------------------------------
    # Reference-layer attention hooks (Flash Attention compatible)
    # ------------------------------------------------------------------

    def _register_ref_layer_hooks(self):
        """
        Hook each reference layer's self_attn.forward to manually compute
        Q·K attention weights for the last query token.

        This bypasses Flash Attention's restriction on returning attention
        weights by recomputing them from q_proj and k_proj directly.
        """
        for layer_idx in self.ref_layers:
            if layer_idx >= len(self.model.model.layers):
                continue
            attn = self.model.model.layers[layer_idx].self_attn
            orig_fwd = attn.forward

            def make_hook(attn_module, orig_forward):
                def hooked_forward(hidden_states, *args, **kwargs):
                    outputs = orig_forward(hidden_states, *args, **kwargs)
                    if self.video_frames is not None:
                        try:
                            self._accumulate_ref_attn(attn_module, hidden_states, args, kwargs)
                        except Exception:
                            pass
                    return outputs
                return hooked_forward

            attn.forward = make_hook(attn, orig_fwd)

    def _accumulate_ref_attn(self, attn_module, hidden_states, args, kwargs):
        """
        Compute attention weights for the last query token and accumulate
        across reference layers.

        During decode (KV cache active), hidden_states has S_q=1. We
        reconstruct the full K by concatenating past_key_value with the
        current token's K projection.
        """
        with torch.no_grad():
            B, S_q, D = hidden_states.shape
            q = attn_module.q_proj(hidden_states[:, -1:, :])
            k_cur = attn_module.k_proj(hidden_states)

            past_kv = kwargs.get("past_key_value", None)
            if past_kv is None and len(args) >= 4:
                past_kv = args[3]

            if past_kv is not None:
                try:
                    # past_kv[0]: [B, num_kv_heads, S_past, head_dim]
                    past_k = past_kv[0]
                    num_kv_heads_past = past_k.shape[1]
                    head_dim_past = past_k.shape[-1]
                    # reshape to [B, S_past, num_kv_heads * head_dim] for concat
                    past_k_2d = past_k.permute(0, 2, 1, 3).reshape(
                        B, -1, num_kv_heads_past * head_dim_past
                    )
                    k_full = torch.cat([past_k_2d, k_cur], dim=1)
                except Exception:
                    k_full = k_cur
            else:
                k_full = k_cur

            S_k = k_full.shape[1]
            num_heads = attn_module.num_heads
            num_kv_heads = getattr(attn_module, "num_key_value_heads", num_heads)
            head_dim = D // num_heads

            q = q.reshape(B, 1, num_heads, head_dim).transpose(1, 2)
            d_kv = num_kv_heads * head_dim
            k_full = k_full[..., :d_kv].reshape(B, S_k, num_kv_heads, head_dim).transpose(1, 2)
            if num_kv_heads != num_heads:
                k_full = k_full.repeat_interleave(num_heads // num_kv_heads, dim=1)

            scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(head_dim)
            scores = torch.clamp(scores, min=-50, max=50)
            attn_w = F.softmax(scores, dim=-1)
            mean_attn = attn_w[0, :, 0, :].mean(dim=0)

            if self._ref_attn_accum is None or self._ref_attn_accum.shape[0] != S_k:
                self._ref_attn_accum = mean_attn.clone()
                self._ref_attn_count = 1
            else:
                self._ref_attn_accum = self._ref_attn_accum + mean_attn
                self._ref_attn_count += 1

    # ------------------------------------------------------------------
    # mm_projector capture hook
    # ------------------------------------------------------------------

    def _wrap_vision_encoder(self):
        """
        Hook mm_projector.forward to automatically capture projected video
        features (4096-dim LLM hidden space) when auto_capture is enabled.

        Video-LLaVA's LanguageBind video tower outputs [1, T, 257, 1024].
        After mm_projector, features are [1, T, 257, 4096] — this is what
        we capture, matching the original paper's visual memory construction.
        """
        try:
            mm_projector = self.model.model.mm_projector
        except AttributeError:
            print("[STEAR] Warning: mm_projector not found; auto-capture unavailable.")
            return

        orig_fwd = mm_projector.forward

        def forward_with_capture(*args, **kwargs):
            outputs = orig_fwd(*args, **kwargs)
            if self.auto_capture_enabled:
                features = outputs[0] if isinstance(outputs, tuple) else outputs
                if features is not None:
                    self.video_frames = features.detach()
            return outputs

        mm_projector.forward = forward_with_capture

    # ------------------------------------------------------------------
    # Dynamic layer selection and FFN injection
    # ------------------------------------------------------------------

    def _wrap_model_forward(self):
        """
        Wrap LlamaModel.forward to reset per-step state and count tokens,
        then wrap each layer in [starting_layer, ending_layer] to monitor
        uncertainty and trigger injection.
        """
        llm_model = self.model.model
        orig_fwd = llm_model.forward
        hook_self = self

        def forward_with_retracing(*args, **kwargs):
            hook_self._visual_retracing_event = False
            hook_self._pending_inject_layer = None
            hook_self.total_tokens += 1
            return orig_fwd(*args, **kwargs)

        llm_model.forward = forward_with_retracing
        self._wrap_all_layers_for_dynamic_selection()

    def _wrap_all_layers_for_dynamic_selection(self):
        """Wrap each layer in the monitoring range to check uncertainty post-forward."""
        hook_self = self
        num_layers = len(self.model.model.layers)

        for layer_idx in range(self.starting_layer + 1, min(self.ending_layer, num_layers)):
            layer = self.model.model.layers[layer_idx]
            orig_fwd = layer.forward
            _idx = layer_idx

            def make_layer_hook(orig, idx):
                def hooked(hidden_states, *args, **kwargs):
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
                                hook_self._activate_injection(next_idx, layer_out)
                    except Exception:
                        pass
                    return outputs
                return hooked

            layer.forward = make_layer_hook(orig_fwd, layer_idx)

    def _activate_injection(self, layer_idx: int, trigger_hidden: torch.Tensor):
        """
        Wrap layer_idx's MLP for a single one-shot visual memory injection.

        Selects top-k frames by reference-layer attention, builds the
        magnitude-normalized adapter, and restores the original MLP after
        the injection fires.
        """
        if self.video_frames is None:
            return

        try:
            vf = self.video_frames
            if vf.dim() == 4:
                frames_3d = vf.squeeze(0)
                num_frames, tpf, vision_dim = frames_3d.shape
            elif vf.dim() == 3:
                frames_3d = vf
                num_frames, tpf, vision_dim = frames_3d.shape
            else:
                return
        except Exception:
            return

        seq_len = trigger_hidden.shape[1]
        total_video_tokens = num_frames * tpf
        if seq_len > 1 and self._video_token_start is None:
            self._video_token_start = 1 if seq_len >= total_video_tokens + 1 else 0
            self._video_token_end = self._video_token_start + total_video_tokens
            self._num_frames = num_frames

        selected = self._select_topk_frames(frames_3d, num_frames, tpf)
        visual_memory = selected.reshape(-1, vision_dim)

        device = trigger_hidden.device
        dtype = trigger_hidden.dtype
        visual_memory = visual_memory.to(device, dtype=dtype)

        # Flush reference attention accumulator
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
            mlp.forward = orig_mlp_fwd  # restore immediately
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

    def _select_topk_frames(
        self, frames_3d: torch.Tensor, num_frames: int, tpf: int
    ) -> torch.Tensor:
        """
        Select top-k frames by aggregated reference-layer attention.

        Falls back to uniform sampling when attention weights are unavailable.

        Args:
            frames_3d: [T, P, D] projected video features.
            num_frames: Number of frames T.
            tpf: Tokens per frame P.

        Returns:
            [k, P, D] selected frames in temporal order.
        """
        k = min(self.topk_frames, num_frames)

        if self._ref_attn_prev is None or self._video_token_start is None:
            indices = torch.linspace(0, num_frames - 1, k).long()
            return frames_3d[indices]

        mean_attn = self._ref_attn_prev
        vid_s = self._video_token_start
        vid_e = self._video_token_end

        if vid_e > mean_attn.shape[0]:
            indices = torch.linspace(0, num_frames - 1, k).long()
            return frames_3d[indices]

        video_attn = mean_attn[vid_s:vid_e]
        frame_attn = video_attn.reshape(num_frames, tpf).mean(dim=1)
        _, topk_idx = torch.topk(frame_attn, k=k)
        topk_idx_sorted, _ = torch.sort(topk_idx)
        return frames_3d[topk_idx_sorted.cpu()]


def apply_visual_retracing(
    model,
    starting_layer: int = 5,
    ending_layer: int = 16,
    injection_ratio: float = 0.2,
    uncertainty_threshold: float = 0.75,
    topk_frames: int = 3,
    tokens_per_frame: int = 257,
    ref_layers: Optional[List[int]] = None,
) -> VisualRetracingHook:
    """
    Attach a VisualRetracingHook to a Video-LLaVA model.

    Args:
        model: Video-LLaVA LlamaForCausalLM.
        starting_layer: Start of uncertainty monitoring range.
        ending_layer: End of uncertainty monitoring range (exclusive).
        injection_ratio: Visual memory blend weight alpha (0.05–0.35 recommended).
        uncertainty_threshold: Entropy threshold gamma (default 0.75).
        topk_frames: Number of top-attended frames for visual memory.
        tokens_per_frame: Visual tokens per frame (257 for LanguageBind).
        ref_layers: Reference layers for attention-based frame selection.

    Returns:
        Attached VisualRetracingHook instance.
    """
    return VisualRetracingHook(
        model,
        starting_layer=starting_layer,
        ending_layer=ending_layer,
        injection_ratio=injection_ratio,
        uncertainty_threshold=uncertainty_threshold,
        topk_frames=topk_frames,
        tokens_per_frame=tokens_per_frame,
        ref_layers=ref_layers,
    )
