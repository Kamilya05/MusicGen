"""
Unified Evaluation Runner.

Runs all three metric families for every condition in a sweep manifest and
produces a single summary table:

  ┌──────────────┬────────┬──────────────────┬────────────┬────────────┬──────────────┐
  │ condition    │  FAD ↓ │  CLAP-sim ↑      │  rep ↓     │  loop% ↓  │  diversity ↑ │
  ├──────────────┼────────┼──────────────────┼────────────┼────────────┼──────────────┤
  │ greedy       │  …     │  …               │  …         │  …        │  …           │
  │ temp_0.7     │  …     │  …               │  …         │  …        │  …           │
  │ …            │  …     │  …               │  …         │  …        │  …           │
  └──────────────┴────────┴──────────────────┴────────────┴────────────┴──────────────┘

Usage:
    python -m src.metrics.evaluate_all \
        --manifest  ./results/sweep/manifest.json \
        --reference ./data/audio/test \
        --output_dir ./results/metrics

    # Skip FAD (no reference audio) or CLAP (no GPU / slow)
    python -m src.metrics.evaluate_all \
        --manifest  ./results/sweep/manifest.json \
        --reference ./data/audio/test \
        --skip_fad  --output_dir ./results/metrics
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from src.evaluate import compute_fad
from src.metrics.prompt_adherence import score_condition as score_adherence
from src.metrics.repetition import evaluate_condition as eval_repetition


def evaluate_all_conditions(
    manifest_path: Path,
    reference_dir: Optional[Path],
    output_dir: Path,
    device: str = "cpu",
    skip_fad: bool = False,
    skip_clap: bool = False,
) -> dict:
    """
    Run FAD, CLAP prompt-adherence, and repetition/diversity for every
    condition listed in the sweep manifest.

    Returns a summary dict; also writes summary.json and per-condition files
    under output_dir/.
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    sweep_dir = manifest_path.parent
    conditions = [c["name"] for c in manifest["conditions"]]

    summary: dict = {
        "manifest": str(manifest_path),
        "reference_dir": str(reference_dir) if reference_dir else None,
        "conditions": {},
    }

    for name in conditions:
        print(f"\n{'─'*50}")
        print(f"Condition: {name}")
        cond_dir = sweep_dir / name
        cond_out = output_dir / name
        cond_out.mkdir(parents=True, exist_ok=True)

        metrics: dict = {"condition": name}

        # FAD
        if not skip_fad and reference_dir and Path(reference_dir).exists():
            try:
                fad = compute_fad(
                    background_dir=reference_dir,
                    eval_dir=cond_dir,
                    output_file=cond_out / "fad.json",
                )
                metrics["fad"] = fad
                print(f"  FAD             : {fad:.4f}")
            except Exception as exc:
                metrics["fad"] = None
                print(f"  FAD             : FAILED — {exc}")
        else:
            metrics["fad"] = None
            if not skip_fad:
                print("  FAD             : SKIPPED (no reference_dir)")

        # CLAP prompt adherence
        if not skip_clap:
            try:
                pa = score_adherence(
                    manifest_path=manifest_path,
                    condition_name=name,
                    output_file=cond_out / "prompt_adherence.json",
                    device=device,
                )
                metrics["mean_clap_similarity"] = pa["mean_clap_similarity"]
                metrics["std_clap_similarity"] = pa["std_clap_similarity"]
                print(f"  CLAP similarity : {pa['mean_clap_similarity']:.4f} "
                      f"± {pa['std_clap_similarity']:.4f}")
            except Exception as exc:
                metrics["mean_clap_similarity"] = None
                metrics["std_clap_similarity"] = None
                print(f"  CLAP            : FAILED — {exc}")
        else:
            metrics["mean_clap_similarity"] = None

        # repetition & diversity
        try:
            rep = eval_repetition(
                audio_dir=cond_dir,
                manifest_path=manifest_path,
                condition_name=name,
                output_file=cond_out / "repetition.json",
            )
            metrics["mean_repetition_score"] = rep["mean_repetition_score"]
            metrics["mean_loop_ratio"] = rep["mean_loop_ratio"]
            metrics["diversity_score"] = rep["diversity"]["diversity_score"]
            print(f"  Repetition      : {rep['mean_repetition_score']:.4f}")
            print(f"  Loop ratio      : {rep['mean_loop_ratio']:.4f}")
            print(f"  Diversity       : {rep['diversity']['diversity_score']:.4f}")
        except Exception as exc:
            metrics["mean_repetition_score"] = None
            metrics["mean_loop_ratio"] = None
            metrics["diversity_score"] = None
            print(f"  Repetition      : FAILED — {exc}")

        summary["conditions"][name] = metrics

    # summary table
    print(f"\n{'='*70}")
    print(f"{'Condition':<14} {'FAD':>7}  {'CLAP':>7}  {'Rep':>7}  {'Loop%':>7}  {'Div':>7}")
    print("─" * 70)
    for name, m in summary["conditions"].items():
        fad_s   = f"{m['fad']:.3f}"             if m.get("fad")                    is not None else "  —  "
        clap_s  = f"{m['mean_clap_similarity']:.3f}"  if m.get("mean_clap_similarity") is not None else "  —  "
        rep_s   = f"{m['mean_repetition_score']:.3f}" if m.get("mean_repetition_score") is not None else "  —  "
        loop_s  = f"{m['mean_loop_ratio']:.3f}"       if m.get("mean_loop_ratio")       is not None else "  —  "
        div_s   = f"{m['diversity_score']:.3f}"       if m.get("diversity_score")       is not None else "  —  "
        print(f"{name:<14} {fad_s:>7}  {clap_s:>7}  {rep_s:>7}  {loop_s:>7}  {div_s:>7}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    return summary



def main():
    parser = argparse.ArgumentParser(
        description="Run all evaluation metrics for a sweep manifest"
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to sweep manifest.json")
    parser.add_argument("--reference", type=str, default=None,
                        help="Directory with real reference audio (for FAD)")
    parser.add_argument("--output_dir", type=str, default="./results/metrics")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for CLAP (cpu / cuda / mps)")
    parser.add_argument("--skip_fad", action="store_true",
                        help="Skip Fréchet Audio Distance computation")
    parser.add_argument("--skip_clap", action="store_true",
                        help="Skip CLAP prompt-adherence computation")
    args = parser.parse_args()

    evaluate_all_conditions(
        manifest_path=Path(args.manifest),
        reference_dir=Path(args.reference) if args.reference else None,
        output_dir=Path(args.output_dir),
        device=args.device,
        skip_fad=args.skip_fad,
        skip_clap=args.skip_clap,
    )


if __name__ == "__main__":
    main()
