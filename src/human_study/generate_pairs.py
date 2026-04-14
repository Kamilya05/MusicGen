"""
Human Study — Pair Generator.

Creates pairwise comparison tasks for a blind listening study.
Each task contains:
  • A text prompt (shown to the rater)
  • Clip A and Clip B (from different conditions, order randomised)
  • Two rating slots (filled by the rater in viewer.html):
      Q1 — "Which sounds better overall?"
      Q2 — "Which matches the prompt better?"

Output: pairs.json — consumed directly by src/human_study/viewer.html

Usage:
    python -m src.human_study.generate_pairs \
        --manifest ./results/sweep/manifest.json \
        --conditions greedy temp_0.7 temp_1.0 temp_1.5 \
        --output_file ./results/human_study/pairs.json \
        --pairs_per_prompt 3
"""

import argparse
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Optional


def _to_relative(path: str, project_root: Optional[Path]) -> str:
    """Convert absolute path to a path relative to project_root (for web serving)."""
    if project_root is None:
        return path
    try:
        return str(Path(path).relative_to(project_root))
    except ValueError:
        return path  # already relative or on a different drive


def generate_pairs(
    manifest_path: Path,
    conditions: list[str],
    output_file: Path,
    pairs_per_prompt: int = 3,
    seed: int = 42,
    project_root: Optional[Path] = None,
) -> list[dict]:
    """
    Build pairwise comparison tasks from a sweep manifest.

    For each text prompt, `pairs_per_prompt` condition-pairs are sampled.
    Presentation order (A vs B) is randomised to prevent position bias.

    Args:
        manifest_path: Path to sweep manifest.json.
        conditions: Condition names to include (minimum 2).
        output_file: Destination for pairs.json.
        pairs_per_prompt: Number of condition pairs to create per prompt.
        seed: Random seed for reproducibility.
        project_root: If given, audio paths are made relative to this directory
                      (required when viewer.html is served from project root).

    Returns:
        List of task dicts written to output_file.
    """
    if len(conditions) < 2:
        raise ValueError("Need at least 2 conditions to build pairs")

    rng = random.Random(seed)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Build index: (condition, prompt_idx) → [audio_path, …]
    index: dict[tuple[str, int], list[str]] = {}
    for record in manifest["records"]:
        if record["condition"] not in conditions:
            continue
        key = (record["condition"], record["prompt_idx"])
        index.setdefault(key, []).append(
            _to_relative(record["audio_path"], project_root)
        )

    prompts = manifest["prompts"]
    all_pairs = list(combinations(conditions, 2))

    tasks: list[dict] = []
    for p_idx, prompt in enumerate(prompts):
        available = [
            (ca, cb) for ca, cb in all_pairs
            if index.get((ca, p_idx)) and index.get((cb, p_idx))
        ]
        if not available:
            continue
        sampled = rng.sample(available, min(pairs_per_prompt, len(available)))

        for cond_a, cond_b in sampled:
            path_a = rng.choice(index[(cond_a, p_idx)])
            path_b = rng.choice(index[(cond_b, p_idx)])

            # Randomise left/right position
            if rng.random() < 0.5:
                path_a, path_b = path_b, path_a
                cond_a, cond_b = cond_b, cond_a

            tasks.append({
                "task_id": len(tasks),
                "prompt": prompt,
                "prompt_idx": p_idx,
                "clip_a": {"path": path_a, "condition": cond_a},
                "clip_b": {"path": path_b, "condition": cond_b},
                # Rating fields filled in by viewer.html
                "q1_overall": None,      # "A" | "B" | "tie"
                "q2_prompt_match": None, # "A" | "B" | "tie"
                "rater_notes": "",
            })

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "study_info": {
            "conditions_compared": conditions,
            "n_tasks": len(tasks),
            "n_prompts": len(prompts),
            "pairs_per_prompt": pairs_per_prompt,
            "seed": seed,
            "manifest": str(manifest_path),
        },
        "tasks": tasks,
    }
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Generated {len(tasks)} comparison tasks → {output_file}")
    print(f"  Conditions : {conditions}")
    print(f"  Prompts    : {len(prompts)}")
    print(f"  Run the study:")
    print(f"    cd <project-root> && python -m http.server 8080")
    print(f"    open http://localhost:8080/src/human_study/viewer.html")
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Generate pairwise listening-study tasks from a sweep manifest"
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to sweep manifest.json")
    parser.add_argument(
        "--conditions", type=str, nargs="+",
        default=["greedy", "temp_0.7", "temp_1.0", "temp_1.5"],
        help="Condition names to compare (≥2 required)",
    )
    parser.add_argument("--output_file", type=str,
                        default="./results/human_study/pairs.json")
    parser.add_argument("--pairs_per_prompt", type=int, default=3,
                        help="Condition pairs per prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root directory; audio paths will be "
                             "made relative to this for web serving")
    args = parser.parse_args()

    generate_pairs(
        manifest_path=Path(args.manifest),
        conditions=args.conditions,
        output_file=Path(args.output_file),
        pairs_per_prompt=args.pairs_per_prompt,
        seed=args.seed,
        project_root=Path(args.project_root) if args.project_root else None,
    )


if __name__ == "__main__":
    main()
