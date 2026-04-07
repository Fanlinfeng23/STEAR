# STEAR: Spatio-Temporal Evidence-Augmented Retracing

Official implementation of **STEAR**, a training-free method for reducing temporal hallucinations in video large language models (Video-LLMs).

STEAR addresses the tendency of Video-LLMs to ignore fine-grained temporal evidence by:
1. **Visual Retracing** — detecting uncertain tokens at inference time and re-injecting relevant frame evidence into the model's residual stream via a lightweight MLP adapter.
2. **Temporal Contrastive Decoding** — constructing a temporally-perturbed negative branch and sharpening the output distribution via contrastive logit adjustment.

This repository provides the Video-LLaVA-7B implementation only.

---

## Directory Structure

```
stear/
├── __init__.py
├── core/
│   ├── visual_retracing.py      # VisualRetracingHook — uncertainty-triggered frame injection
│   └── contrastive_decoding.py  # TemporalContrastiveHook — contrastive decoding extension
└── models/
    └── video_llava.py           # VideoLLaVAWithSTEAR — high-level model wrapper
eval/
└── eval_eventhallusion.py       # Evaluation script for EventHallusion benchmark
```

---

## Installation

```bash
# 1. Clone this repository
git clone [<repo_url>](https://github.com/Fanlinfeng23/STEAR)
cd stear

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Video-LLaVA (required)
pip install git+https://github.com/PKU-YuanGroup/Video-LLaVA.git
# or place the videollava_code/ directory alongside this repo
```

---

## Quick Start

```python
from stear.models.video_llava import VideoLLaVAWithSTEAR

model = VideoLLaVAWithSTEAR.from_pretrained(
    "LanguageBind/Video-LLaVA-7B",
    device="cuda",
    use_contrastive=True,   # enable temporal contrastive decoding
)

answer = model.generate(
    video_path="path/to/video.mp4",
    question="Did the person pick up the red ball before or after sitting down?",
)
print(answer)
```

### Low-level API

```python
from stear.models.video_llava import load_model, infer
from stear.core.contrastive_decoding import apply_temporal_contrastive

model, tokenizer, video_processor = load_model("LanguageBind/Video-LLaVA-7B", device="cuda")

hook = apply_temporal_contrastive(
    model,
    starting_layer=5,
    ending_layer=16,
    contrastive_alpha=0.5,
    contrastive_mode="both",   # "shuffle" | "homo" | "both"
)
hook.enable_auto_capture()

answer = infer(model, tokenizer, video_processor, "video.mp4", "What happened first?")
hook.clear_video_frames()
```

---

## Evaluation on EventHallusion

```bash
python eval/eval_eventhallusion.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --dataset_path /path/to/EventHallusion \
    --output results/stear_eventhallusion.json \
    --gpu 0
```

The script evaluates all three splits (`entire`, `subtle`, `mix`) and prints per-split and overall accuracy. Results are saved incrementally — interrupted runs can be resumed by re-running the same command.

### Expected dataset layout

```
EventHallusion/
├── annotations/
│   ├── entire.json
│   ├── subtle.json
│   └── mix.json
└── videos/
    ├── entire/
    ├── subtle/
    └── mix/
```

---

## Hyperparameter Reference

| Parameter | Default | Description |
|---|---|---|
| `starting_layer` | 5 | First transformer layer where injection is active |
| `ending_layer` | 16 | Last transformer layer where injection is active |
| `injection_ratio` | 0.2 | Blend weight α for MLP adapter injection |
| `uncertainty_threshold` | 0.75 | Normalized entropy threshold to trigger retracing |
| `topk_frames` | 3 | Number of top-attended frames to inject |
| `tokens_per_frame` | 257 | Visual tokens per frame (7×7 patches + CLS for LanguageBind) |
| `ref_layers` | 10–20 | Transformer layers used to compute frame attention weights |
| `contrastive_alpha` | 0.5 | Contrastive sharpening weight: `(1+α)·pos - α·neg` |
| `topk_frame_ratio` | 0.2 | Fraction of frames selected for temporal perturbation |
| `homogenize_gamma` | 0.5 | Blend weight for temporal homogenization perturbation |
| `contrastive_mode` | `"both"` | Perturbation type: `"shuffle"`, `"homo"`, or `"both"` |
| `injection_mode` | `"frame_based"` | Injection granularity: `"frame_based"` or `"patch_based"` |

---

## Algorithm Overview

### Visual Retracing (VisualRetracingHook)

At each transformer layer in `[starting_layer, ending_layer]`, STEAR monitors token uncertainty via the normalized entropy of the top-10 predicted tokens:

```
uncertainty = H(top10_probs) / log(10)
```

When uncertainty exceeds `uncertainty_threshold`, the hook selects the top-k most-attended frames (aggregated attention across `ref_layers`) and injects their mm_projector features into the current hidden state:

```
h_out = (1 - α) · FFN(h) + α · normalize(adapter(frames))
```

### Temporal Contrastive Decoding (TemporalContrastiveHook)

Extends visual retracing with a contrastive negative branch. A temporally-perturbed copy of the video (shuffled and/or homogenized frames) is propagated through the remaining layers to produce negative logits. The final output sharpens the positive distribution:

```
logits_final = (1 + α) · logits_pos - α · logits_neg
```

---

## Citation

If you use STEAR in your research, please cite:

```bibtex
@article{stear2025,
  title   = {STEAR: Spatio-Temporal Evidence-Augmented Retracing for Video Large Language Models},
  author  = {},
  year    = {2025},
}
```
