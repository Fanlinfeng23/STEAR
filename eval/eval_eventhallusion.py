#!/usr/bin/env python3
"""
Evaluate STEAR on the EventHallusion benchmark with Video-LLaVA-7B.

Usage:
    python eval/eval_eventhallusion.py \
        --model_path LanguageBind/Video-LLaVA-7B \
        --dataset_path /path/to/EventHallusion \
        --output results/stear_eventhallusion.json \
        --gpu 0

The script evaluates all three EventHallusion splits (entire, subtle, mix)
and reports per-split and overall accuracy.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch

# Add parent directory to path so `stear` package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add Video-LLaVA code to path
_VIDEOLLAVA_CODE = str(Path(__file__).parent.parent / "videollava_code")
if os.path.isdir(_VIDEOLLAVA_CODE):
    sys.path.insert(0, _VIDEOLLAVA_CODE)


# ---------------------------------------------------------------------------
# EventHallusion dataset loader
# ---------------------------------------------------------------------------

def load_eventhallusion(dataset_path: str):
    """
    Load EventHallusion annotations.

    Expected directory structure:
        dataset_path/
          annotations/
            entire.json
            subtle.json
            mix.json
          videos/
            entire/
            subtle/
            mix/

    Returns:
        List of dicts with keys: split, video_id, video_path, question, answer
    """
    samples = []
    ann_dir = os.path.join(dataset_path, "annotations")
    vid_dir = os.path.join(dataset_path, "videos")

    for split in ("entire", "subtle", "mix"):
        ann_file = os.path.join(ann_dir, f"{split}.json")
        if not os.path.isfile(ann_file):
            print(f"[WARN] Annotation file not found: {ann_file}")
            continue

        with open(ann_file) as f:
            annotations = json.load(f)

        for item in annotations:
            video_id = item.get("video_id") or item.get("id")
            video_path = os.path.join(vid_dir, split, f"{video_id}.mp4")
            if not os.path.isfile(video_path):
                continue

            questions = item.get("questions") or item.get("QA") or []
            if isinstance(questions, dict):
                questions = [questions]

            for qa in questions:
                q = qa.get("question") or qa.get("Q", "")
                a = qa.get("answer") or qa.get("A", "")
                if q and a:
                    samples.append({
                        "split": split,
                        "video_id": video_id,
                        "video_path": video_path,
                        "question": q,
                        "answer": a.strip(),
                    })

    return samples


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Extract Yes/No from model output."""
    t = text.strip().lower()
    if t.startswith("yes"):
        return "Yes."
    if t.startswith("no"):
        return "No."
    # Scan for first yes/no word
    for word in t.split():
        w = word.strip(".,!?;:")
        if w == "yes":
            return "Yes."
        if w == "no":
            return "No."
    return text.strip()


def is_correct(prediction: str, ground_truth: str) -> bool:
    pred = normalize_answer(prediction).lower().rstrip(".")
    gt = ground_truth.strip().lower().rstrip(".")
    return pred == gt


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[STEAR] Loading model from {args.model_path} on {device}")

    from stear.models.video_llava import load_model
    from stear.core.contrastive_decoding import apply_temporal_contrastive

    model, tokenizer, video_processor = load_model(args.model_path, device=device)

    stear_cfg = {
        "starting_layer": args.starting_layer,
        "ending_layer": args.ending_layer,
        "injection_ratio": args.injection_ratio,
        "uncertainty_threshold": args.uncertainty_threshold,
        "topk_frames": args.topk_frames,
        "tokens_per_frame": 257,
        "ref_layers": list(range(10, 21)),
        "contrastive_alpha": args.contrastive_alpha,
        "topk_frame_ratio": args.topk_frame_ratio,
        "homogenize_gamma": args.homogenize_gamma,
        "contrastive_mode": args.contrastive_mode,
        "injection_mode": args.injection_mode,
    }

    hook = apply_temporal_contrastive(model, **stear_cfg)
    hook.enable_auto_capture()

    print(f"[STEAR] Loading EventHallusion from {args.dataset_path}")
    samples = load_eventhallusion(args.dataset_path)
    print(f"[STEAR] {len(samples)} samples loaded")

    # Resume from checkpoint if output file exists
    results = []
    done_keys = set()
    if os.path.isfile(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        results = existing.get("results", [])
        done_keys = {(r["split"], r["video_id"], r["question"]) for r in results}
        print(f"[STEAR] Resuming: {len(done_keys)} samples already done")

    from videollava.conversation import conv_templates, SeparatorStyle
    from videollava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_video
    from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    for i, sample in enumerate(samples):
        key = (sample["split"], sample["video_id"], sample["question"])
        if key in done_keys:
            continue

        try:
            video_tensor = process_video(
                sample["video_path"], video_processor, num_frames=args.num_frames
            ).to(device, dtype=torch.float16)

            conv = conv_templates["llava_v1"].copy()
            inp = DEFAULT_IMAGE_TOKEN + "\n" + sample["question"]
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            hook.enable_auto_capture()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=video_tensor,
                    do_sample=False,
                    max_new_tokens=256,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            hook.clear_video_frames()

            raw_output = tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            if raw_output.endswith(stop_str):
                raw_output = raw_output[: -len(stop_str)].strip()

            prediction = normalize_answer(raw_output)
            correct = is_correct(prediction, sample["answer"])

            results.append({
                "split": sample["split"],
                "video_id": sample["video_id"],
                "question": sample["question"],
                "ground_truth": sample["answer"],
                "prediction": prediction,
                "raw_output": raw_output[:200],
                "correct": correct,
            })

        except Exception as e:
            hook.clear_video_frames()
            results.append({
                "split": sample["split"],
                "video_id": sample["video_id"],
                "question": sample["question"],
                "ground_truth": sample["answer"],
                "prediction": "ERROR",
                "raw_output": str(e)[:200],
                "correct": False,
            })
            if args.verbose:
                traceback.print_exc()

        if (i + 1) % args.checkpoint_interval == 0:
            _save_results(results, args.output, stear_cfg)
            print(f"[STEAR] {i + 1}/{len(samples)} done")

    _save_results(results, args.output, stear_cfg)
    _print_summary(results)


def _save_results(results, output_path, config):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    summary = _compute_summary(results)
    out = {"config": config, "summary": summary, "results": results}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def _compute_summary(results):
    from collections import defaultdict
    by_split = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        s = r["split"]
        by_split[s]["total"] += 1
        if r["correct"]:
            by_split[s]["correct"] += 1

    total = sum(v["total"] for v in by_split.values())
    correct = sum(v["correct"] for v in by_split.values())
    summary = {
        "overall_accuracy": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "by_split": {
            s: {
                "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for s, v in by_split.items()
        },
    }
    return summary


def _print_summary(results):
    summary = _compute_summary(results)
    print("\n" + "=" * 60)
    print("EventHallusion Results — STEAR (Video-LLaVA-7B)")
    print("=" * 60)
    for split, stats in sorted(summary["by_split"].items()):
        print(
            f"  {split:10s}: {stats['accuracy'] * 100:.2f}%  "
            f"({stats['correct']}/{stats['total']})"
        )
    print("-" * 60)
    print(
        f"  {'Overall':10s}: {summary['overall_accuracy'] * 100:.2f}%  "
        f"({summary['correct']}/{summary['total']})"
    )
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate STEAR on EventHallusion")
    p.add_argument("--model_path", default="LanguageBind/Video-LLaVA-7B")
    p.add_argument("--dataset_path", required=True, help="Path to EventHallusion dataset")
    p.add_argument("--output", default="results/stear_eventhallusion.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--checkpoint_interval", type=int, default=50)
    p.add_argument("--verbose", action="store_true")
    # STEAR hyperparameters
    p.add_argument("--starting_layer", type=int, default=5)
    p.add_argument("--ending_layer", type=int, default=16)
    p.add_argument("--injection_ratio", type=float, default=0.2)
    p.add_argument("--uncertainty_threshold", type=float, default=0.75)
    p.add_argument("--topk_frames", type=int, default=3)
    p.add_argument("--contrastive_alpha", type=float, default=0.5)
    p.add_argument("--topk_frame_ratio", type=float, default=0.2)
    p.add_argument("--homogenize_gamma", type=float, default=0.5)
    p.add_argument("--contrastive_mode", default="both", choices=["shuffle", "homo", "both"])
    p.add_argument("--injection_mode", default="frame_based", choices=["frame_based", "patch_based"])
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
