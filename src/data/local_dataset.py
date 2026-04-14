"""
Local Audio Dataset Pipeline.

Dead-simple manifest builder for fine-tuning on your own audio files.

Layout
------
Put audio files and label sidecars in one folder:

    my_dataset/
        track01.flac        ← audio
        track01.txt         ← label: "heavy industrial breakcore metal"
        track02.flac
        track02.txt
        ...

If a .txt sidecar is missing, --default_label is used instead.
Supported audio formats: flac, wav, mp3, ogg, m4a.

Usage
-----
    python -m src.data.local_dataset \\
        --input_dir  ./my_dataset \\
        --output_dir ./data/custom_manifests \\
        --default_label "ULTRAKILL OST, industrial metal breakcore, Heaven Pierce Her" \\
        --split 0.8 0.1 0.1

Then train with:
    python -m src.train \\
        --manifest_dir ./data/custom_manifests \\
        --audio_dir    . \\
        --output_dir   ./trained_model_lora \\
        --lora --epochs 5 --batch_size 2 --duration 10 --lr 1e-4
"""

import argparse
import json
import random
from pathlib import Path

AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}


def build_manifests(
    input_dir: Path,
    output_dir: Path,
    default_label: str = "music",
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Collect all audio files
    audio_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise RuntimeError(
            f"No audio files found in {input_dir}. "
            f"Supported formats: {', '.join(AUDIO_EXTENSIONS)}"
        )

    # Build entries
    entries = []
    for audio_path in audio_files:
        sidecar = audio_path.with_suffix(".txt")
        if sidecar.exists():
            label = sidecar.read_text(encoding="utf-8").strip()
        else:
            label = default_label

        entries.append({
            "audio": str(audio_path.resolve()),
            "text": label,
            "genre": label,
        })

    print(f"Found {len(entries)} audio files in {input_dir}")
    for e in entries:
        src = "sidecar" if Path(e["audio"]).with_suffix(".txt").exists() else "default"
        print(f"  {Path(e['audio']).name:40s}  [{src}] {e['text'][:60]}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(entries)

    n = len(entries)
    train_ratio, valid_ratio, _ = split
    n_train = max(1, int(n * train_ratio))
    n_valid = max(1, int(n * valid_ratio))
    # Ensure valid+test don't eat all data if dataset is tiny
    if n_train + n_valid >= n:
        n_valid = max(0, n - n_train - 1)

    splits = {
        "train": entries[:n_train],
        "valid": entries[n_train: n_train + n_valid],
        "test":  entries[n_train + n_valid:],
    }

    manifest_paths = {}
    for split_name, split_entries in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = split_dir / "data.jsonl"

        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in split_entries:
                f.write(json.dumps(entry) + "\n")

        manifest_paths[split_name] = manifest_path
        print(f"  {split_name:6s}: {len(split_entries):4d} entries → {manifest_path}")

    if not splits["test"]:
        print("  warning: dataset too small for a test split — test manifest is empty")

    return manifest_paths


def main():
    parser = argparse.ArgumentParser(
        description="Build train/valid/test manifests from a local audio folder"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Folder containing audio files (and optional .txt sidecars)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/custom_manifests",
        help="Where to write the JSONL manifests",
    )
    parser.add_argument(
        "--default_label", type=str, default="music",
        help="Label to use for any file that has no .txt sidecar",
    )
    parser.add_argument(
        "--split", type=float, nargs=3, default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VALID", "TEST"),
        help="Train/valid/test split ratios (default: 0.8 0.1 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_manifests(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        default_label=args.default_label,
        split=tuple(args.split),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
