"""
Experiment Runner.

Runs a clean, separated sweep over decoding strategies so that each
independent variable is studied in isolation:

  greedy   — use_sampling=False  (temperature is irrelevant)
  temp     — use_sampling=True, top_k=0, top_p=0, temperature ∈ [0.5 … 1.5]
  topk     — use_sampling=True, top_p=0, temperature=1.0, top_k ∈ [50 … 500]
  topp     — use_sampling=True, top_k=0, temperature=1.0, top_p ∈ [0.85 … 0.99]

This separation prevents the confounding that occurs when temperature is swept
while top_k/top_p are simultaneously non-zero.

Usage:
    # Temperature sweep only
    python -m src.run_experiments --group temp --num_samples 5

    # Full sweep across all conditions
    python -m src.run_experiments --group all --output_dir ./results/sweep

    # Supply custom prompts
    python -m src.run_experiments --group temp --prompts_file ./data/eval_prompts.jsonl
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv

load_dotenv()

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


@dataclass
class Condition:
    name: str
    use_sampling: bool
    temperature: float
    top_k: int
    top_p: float
    description: str


CONDITIONS: list[Condition] = [
    # greedy baseline
    Condition("greedy",    False, 1.0, 0,   0.00,
              "Greedy (argmax) decoding. Temperature parameter is irrelevant here."),

    # temperature sweep — no truncation (top_k=0, top_p=0)
    Condition("temp_0.5",  True,  0.5, 0,   0.00,
              "Sampling, T=0.50 — highly peaked, low diversity"),
    Condition("temp_0.7",  True,  0.7, 0,   0.00,
              "Sampling, T=0.70 — moderately peaked"),
    Condition("temp_0.9",  True,  0.9, 0,   0.00,
              "Sampling, T=0.90 — slightly below neutral"),
    Condition("temp_1.0",  True,  1.0, 0,   0.00,
              "Sampling, T=1.00 — unmodified distribution"),
    Condition("temp_1.2",  True,  1.2, 0,   0.00,
              "Sampling, T=1.20 — slightly flattened"),
    Condition("temp_1.5",  True,  1.5, 0,   0.00,
              "Sampling, T=1.50 — highly flattened, high diversity"),

    # top-k sweep (T=1.0, top_p=0)
    Condition("topk_50",   True,  1.0, 50,  0.00,
              "Sampling, T=1.0, top_k=50 — narrow truncation"),
    Condition("topk_100",  True,  1.0, 100, 0.00,
              "Sampling, T=1.0, top_k=100"),
    Condition("topk_250",  True,  1.0, 250, 0.00,
              "Sampling, T=1.0, top_k=250 — AudioCraft default"),
    Condition("topk_500",  True,  1.0, 500, 0.00,
              "Sampling, T=1.0, top_k=500 — wide truncation"),

    # top-p sweep (T=1.0, top_k=0)
    Condition("topp_0.85", True,  1.0, 0,   0.85,
              "Sampling, T=1.0, top_p=0.85 — narrow nucleus"),
    Condition("topp_0.90", True,  1.0, 0,   0.90,
              "Sampling, T=1.0, top_p=0.90"),
    Condition("topp_0.95", True,  1.0, 0,   0.95,
              "Sampling, T=1.0, top_p=0.95"),
    Condition("topp_0.99", True,  1.0, 0,   0.99,
              "Sampling, T=1.0, top_p=0.99 — wide nucleus"),
]

CONDITION_GROUPS: dict[str, list[str]] = {
    "greedy": ["greedy"],
    "temp":   ["greedy", "temp_0.5", "temp_0.7", "temp_0.9",
               "temp_1.0", "temp_1.2", "temp_1.5"],
    "topk":   ["greedy", "topk_50", "topk_100", "topk_250", "topk_500"],
    "topp":   ["greedy", "topp_0.85", "topp_0.90", "topp_0.95", "topp_0.99"],
    "all":    [c.name for c in CONDITIONS],
}

# fallback prompts when no prompts_file is given

DEFAULT_PROMPTS = [
    "upbeat electronic dance music with a driving beat",
    "melancholic jazz piano trio, late night mood",
    "aggressive heavy metal guitar riff with drums",
    "peaceful acoustic folk guitar, fingerpicking style",
    "cinematic orchestral music, rising tension",
]


def load_prompts(prompts_file: Optional[Path]) -> list[str]:
    if prompts_file is None or not Path(prompts_file).exists():
        return DEFAULT_PROMPTS
    prompts = []
    with open(prompts_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("prompt") or obj.get("genre", "")
            except json.JSONDecodeError:
                text = line
            if text:
                prompts.append(text)
    return prompts or DEFAULT_PROMPTS


def run_condition(
    model: MusicGen,
    condition: Condition,
    prompts: list[str],
    output_dir: Path,
    num_samples: int,
    duration: float,
) -> list[dict]:
    cond_dir = output_dir / condition.name
    cond_dir.mkdir(parents=True, exist_ok=True)

    model.set_generation_params(
        duration=duration,
        temperature=condition.temperature,
        use_sampling=condition.use_sampling,
        top_k=condition.top_k,
        top_p=condition.top_p,
    )

    records = []
    for p_idx, prompt in enumerate(prompts):
        for s_idx in range(num_samples):
            wav = model.generate([prompt])
            wav = wav[0].cpu()
            stem = str(cond_dir / f"gen_{p_idx:03d}_{s_idx:03d}")
            path = audio_write(stem, wav, model.sample_rate,
                               format="wav", add_suffix=True)
            records.append({
                "condition": condition.name,
                "prompt_idx": p_idx,
                "sample_idx": s_idx,
                "prompt": prompt,
                "audio_path": str(path),
                **asdict(condition),
            })
    return records


def run_experiments(
    output_dir: Path,
    prompts_file: Optional[Path] = None,
    model_name: str = "facebook/musicgen-small",
    checkpoint: Optional[Path] = None,
    duration: float = 30.0,
    num_samples: int = 5,
    group: str = "all",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_file)
    names = CONDITION_GROUPS.get(group, [c.name for c in CONDITIONS])
    active = [c for c in CONDITIONS if c.name in names]

    print(
        f"Sweep: group='{group}'  |  {len(active)} conditions  |  "
        f"{len(prompts)} prompts × {num_samples} samples  |  {duration}s each"
    )

    print(f"Loading model: {model_name}")
    model = MusicGen.get_pretrained(model_name)

    if checkpoint is not None:
        print(f"Loading fine-tuned LM weights from {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu")
        is_lora = any(k.startswith("base_model.model.") for k in state.keys())
        if is_lora:
            print("  detected lora checkpoint — applying adapter structure first")
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["out_proj", "linear1", "linear2"],
                lora_dropout=0.05,
                bias="none",
            )
            model.lm = get_peft_model(model.lm, lora_config)
        model.lm.load_state_dict(state)
        model.lm.float()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.lm.to(device)
    model.compression_model.to(device)

    all_records: list[dict] = []
    for cond in active:
        print(f"\n[{cond.name}]  {cond.description}")
        records = run_condition(model, cond, prompts, output_dir, num_samples, duration)
        all_records.extend(records)
        print(f"  → {len(records)} files  →  {output_dir / cond.name}/")

    manifest = {
        "model": model_name,
        "duration_s": duration,
        "num_prompts": len(prompts),
        "num_samples_per_prompt": num_samples,
        "group": group,
        "prompts": prompts,
        "conditions": [asdict(c) for c in active],
        "records": all_records,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}  ({len(all_records)} total files)")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled decoding sweep for MusicGen"
    )
    parser.add_argument("--output_dir", type=str, default="./results/sweep",
                        help="Root directory for generated audio and manifest")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="JSONL file with text prompts (optional)")
    parser.add_argument("--model", type=str, default="facebook/musicgen-small")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned LM checkpoint (.pt) to load over base weights")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Generation length per clip (seconds)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Samples per prompt per condition")
    parser.add_argument(
        "--group", type=str, default="all",
        choices=list(CONDITION_GROUPS.keys()),
        help=(
            "Condition group to run: "
            "'temp' (temperature sweep), "
            "'topk' (top-k sweep), "
            "'topp' (top-p sweep), "
            "'all' (everything)"
        ),
    )
    args = parser.parse_args()

    run_experiments(
        output_dir=Path(args.output_dir),
        prompts_file=Path(args.prompts_file) if args.prompts_file else None,
        model_name=args.model,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        duration=args.duration,
        num_samples=args.num_samples,
        group=args.group,
    )


if __name__ == "__main__":
    main()
