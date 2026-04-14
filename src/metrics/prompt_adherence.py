"""
Prompt Adherence.

Measures how well generated audio matches its text prompt using CLAP
(Contrastive Language-Audio Pre-training) embeddings.

For each (audio, prompt) pair the metric is the cosine similarity between
the CLAP text embedding and the CLAP audio embedding. The shared latent space
makes this a direct, prompt-grounded measure — unlike FAD, which only reflects
audio quality relative to a reference distribution.

Model: laion/larger_clap_music_and_speech
  • Trained on music + general audio + speech.
  • Input audio is expected at 48 kHz; we resample automatically.

Usage:
    # Score one condition from a sweep manifest
    python -m src.metrics.prompt_adherence \
        --manifest ./results/sweep/manifest.json \
        --condition  temp_1.0 \
        --output_file ./results/metrics/pa_temp_1.0.json

    # Score all conditions at once
    python -m src.metrics.prompt_adherence \
        --manifest ./results/sweep/manifest.json \
        --all_conditions \
        --output_dir ./results/metrics/
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd
from transformers import AutoProcessor, ClapModel

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
CLAP_SAMPLE_RATE = 48_000


# model loading (cached per process)

_clap_model: Optional[ClapModel] = None
_clap_processor: Optional[AutoProcessor] = None


def _get_clap(device: str = "cpu"):
    global _clap_model, _clap_processor
    if _clap_model is None:
        print(f"  Loading CLAP: {CLAP_MODEL_ID}")
        _clap_model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device).eval()
        _clap_processor = AutoProcessor.from_pretrained(CLAP_MODEL_ID)
    return _clap_model, _clap_processor


# audio loading

def _load_mono_48k(path: Path) -> np.ndarray:
    """Load audio as mono float32 at 48 kHz."""
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != CLAP_SAMPLE_RATE:
        g = gcd(sr, CLAP_SAMPLE_RATE)
        audio = resample_poly(audio, CLAP_SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio


# scoring

@torch.no_grad()
def compute_clap_similarity(
    audio_paths: list[Path],
    prompts: list[str],
    device: str = "cpu",
) -> list[float]:
    """
    Return per-item cosine similarity between CLAP audio and text embeddings.

    Values are in [-1, 1]; higher means better prompt match.

    Args:
        audio_paths: WAV files to evaluate.
        prompts: Matching text prompts (same order, same length).
        device: Torch device string ("cpu", "cuda", "mps").
    """
    assert len(audio_paths) == len(prompts), (
        f"Expected equal lengths, got {len(audio_paths)} paths and {len(prompts)} prompts"
    )

    model, processor = _get_clap(device)
    scores: list[float] = []

    for path, prompt in zip(audio_paths, prompts):
        audio = _load_mono_48k(Path(path))

        text_inputs = processor(
            text=[prompt], return_tensors="pt", padding=True
        ).to(device)
        audio_inputs = processor(
            audio=[audio], sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
        ).to(device)

        t_emb = model.get_text_features(**text_inputs)
        a_emb = model.get_audio_features(**audio_inputs)

        # Newer transformers may return a ModelOutput object instead of a tensor
        if not isinstance(t_emb, torch.Tensor):
            t_emb = t_emb.pooler_output
        if not isinstance(a_emb, torch.Tensor):
            a_emb = a_emb.pooler_output

        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-8)
        a_emb = a_emb / (a_emb.norm(dim=-1, keepdim=True) + 1e-8)
        score = (t_emb * a_emb).sum(dim=-1).item()
        scores.append(float(score))

    return scores


# condition-level scoring

def score_condition(
    manifest_path: Path,
    condition_name: str,
    output_file: Optional[Path] = None,
    device: str = "cpu",
) -> dict:
    """
    Compute prompt-adherence scores for every sample in one sweep condition.

    Reads the sweep manifest to pair each audio file with its prompt.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    records = [r for r in manifest["records"] if r["condition"] == condition_name]
    if not records:
        raise ValueError(f"No records for condition '{condition_name}' in {manifest_path}")

    audio_paths = [Path(r["audio_path"]) for r in records]
    prompts = [r["prompt"] for r in records]

    print(f"  [{condition_name}] scoring {len(records)} samples …")
    scores = compute_clap_similarity(audio_paths, prompts, device=device)

    per_item = [
        {
            "audio_path": r["audio_path"],
            "prompt": r["prompt"],
            "prompt_idx": r["prompt_idx"],
            "sample_idx": r["sample_idx"],
            "clap_similarity": s,
        }
        for r, s in zip(records, scores)
    ]

    result = {
        "condition": condition_name,
        "clap_model": CLAP_MODEL_ID,
        "n": len(scores),
        "mean_clap_similarity": float(np.mean(scores)),
        "std_clap_similarity": float(np.std(scores)),
        "per_item": per_item,
    }

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {output_file}")

    return result


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Compute CLAP-based prompt adherence for generated audio"
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to sweep manifest.json")
    parser.add_argument("--condition", type=str, default=None,
                        help="Single condition name to score")
    parser.add_argument("--all_conditions", action="store_true",
                        help="Score every condition in the manifest")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON (single condition mode)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (all-conditions mode)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)

    if args.all_conditions:
        with open(manifest_path) as f:
            manifest = json.load(f)
        out_dir = Path(args.output_dir or "./results/metrics")
        for cond in manifest["conditions"]:
            name = cond["name"]
            result = score_condition(
                manifest_path=manifest_path,
                condition_name=name,
                output_file=out_dir / f"pa_{name}.json",
                device=args.device,
            )
            print(f"  {name}: mean={result['mean_clap_similarity']:.4f}")
    elif args.condition:
        result = score_condition(
            manifest_path=manifest_path,
            condition_name=args.condition,
            output_file=Path(args.output_file) if args.output_file else None,
            device=args.device,
        )
        print(
            f"Condition: {args.condition}\n"
            f"  Mean CLAP similarity: {result['mean_clap_similarity']:.4f} "
            f"± {result['std_clap_similarity']:.4f}  (n={result['n']})"
        )
    else:
        parser.error("Supply --condition NAME or --all_conditions")


if __name__ == "__main__":
    main()
