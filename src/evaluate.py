"""
Fréchet Audio Distance (FAD) Evaluation.

Computes FAD between a set of reference (background) audio and generated (evaluation)
audio. Uses VGGish embeddings by default (matches original Google FAD paper).
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from frechet_audio_distance import FrechetAudioDistance

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def compute_fad(
    background_dir: str | Path,
    eval_dir: str | Path,
    model_name: str = "vggish",
    sample_rate: int = 16000,
    output_file: str | Path | None = None,
) -> float:
    """
    Compute Fréchet Audio Distance between background and evaluation sets.

    Args:
        background_dir: Directory containing reference/real audio (WAV).
        eval_dir: Directory containing generated audio to evaluate.
        model_name: Embedding model ("vggish", "pann", "clap", "encodec").
        sample_rate: Expected sample rate (VGGish: 16k, EnCodec: 48k).
        output_file: Optional path to save results JSON.

    Returns:
        FAD score (lower is better).
    """
    background_dir = Path(background_dir)
    eval_dir = Path(eval_dir)
    if not background_dir.exists():
        raise FileNotFoundError(f"Background dir not found: {background_dir}")
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

    # FAD scans directories indiscriminately — stage only audio files in a
    # temp directory so stray .txt / .jsonl files don't cause errors.
    tmp = tempfile.mkdtemp()
    try:
        tmp_bg = Path(tmp) / "background"
        tmp_bg.mkdir()
        for f in background_dir.iterdir():
            if f.suffix.lower() in AUDIO_EXTENSIONS:
                shutil.copy2(f, tmp_bg / f.name)

        frechet = FrechetAudioDistance(
            model_name=model_name,
            sample_rate=sample_rate,
            use_pca=False,
            use_activation=False,
            verbose=True,
        )
        fad_score = frechet.score(
            str(tmp_bg),
            str(eval_dir),
            dtype="float32",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    result = {
        "fad_score": float(fad_score),
        "background_dir": str(background_dir),
        "eval_dir": str(eval_dir),
        "model_name": model_name,
        "sample_rate": sample_rate,
    }
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {output_file}")
    return fad_score


def main():
    parser = argparse.ArgumentParser(description="Compute FAD for generated music")
    parser.add_argument(
        "--background_dir",
        type=str,
        required=True,
        help="Directory with reference/real audio",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory with generated audio",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vggish",
        choices=["vggish", "pann", "clap", "encodec"],
        help="Embedding model for FAD",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate (vggish: 16k, encodec: 48k)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    args = parser.parse_args()
    fad = compute_fad(
        background_dir=args.background_dir,
        eval_dir=args.eval_dir,
        model_name=args.model,
        sample_rate=args.sample_rate,
        output_file=args.output_file,
    )
    print(f"FAD score: {fad:.4f}")

if __name__ == "__main__":
    main()
