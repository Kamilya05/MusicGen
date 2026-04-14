import argparse
import json
from dotenv import load_dotenv

load_dotenv()
import os
from pathlib import Path
from typing import Optional

from datasets import load_dataset


GENRE_NAMES = [
    "20th Century Classical", "Abstract Hip-Hop", "African", "Afrobeat",
    "Alternative Hip-Hop", "Ambient", "Ambient Electronic", "Americana",
    "Asia-Far East", "Audio Collage", "Avant-Garde", "Balkan", "Banter", "Be-Bop",
    "Big Band/Swing", "Bigbeat", "Black-Metal", "Bluegrass", "Blues", "Bollywood",
    "Brazilian", "Breakbeat", "Breakcore - Hard", "British Folk", "Celtic",
    "Chamber Music", "Chill-out", "Chip Music", "Chiptune", "Choral Music",
    "Christmas", "Classical", "Comedy", "Compilation", "Composed Music",
    "Contemporary Classical", "Country", "Country & Western", "Cumbia", "Dance",
    "Death-Metal", "Deep Funk", "Disco", "Downtempo", "Drone", "Drum & Bass",
    "Dubstep", "Easy Listening", "Easy Listening: Vocal", "Electro-Punk",
    "Electroacoustic", "Electronic", "Europe", "Experimental", "Experimental Pop",
    "Fado", "Field Recordings", "Flamenco", "Folk", "Freak-Folk", "Free-Folk",
    "Free-Jazz", "French", "Funk", "Garage", "Glitch", "Gospel", "Goth",
    "Grindcore", "Hardcore", "Hip-Hop", "Hip-Hop Beats", "Holiday", "House",
    "IDM", "Improv", "Indian", "Indie-Rock", "Industrial", "Instrumental",
    "International", "Interview", "Jazz", "Jazz: Out", "Jazz: Vocal", "Jungle",
    "Kid-Friendly", "Klezmer", "Krautrock", "Latin", "Latin America", "Lo-Fi",
    "Loud-Rock", "Lounge", "Metal", "Middle East", "Minimal Electronic",
    "Minimalism", "Modern Jazz", "Musical Theater", "Musique Concrete",
    "N. Indian Traditional", "Nerdcore", "New Age", "New Wave", "No Wave",
    "Noise", "Noise-Rock", "North African", "Novelty", "Nu-Jazz",
    "Old-Time / Historic", "Opera", "Pacific", "Poetry", "Polka", "Pop",
    "Post-Punk", "Post-Rock", "Power-Pop", "Progressive", "Psych-Folk",
    "Psych-Rock", "Punk", "Radio", "Radio Art", "Radio Theater", "Rap",
    "Reggae - Dancehall", "Reggae - Dub", "Rock", "Rock Opera", "Rockabilly",
    "Romany (Gypsy)", "Salsa", "Shoegaze", "Singer-Songwriter", "Skweee",
    "Sludge", "Soul-RnB", "Sound Art", "Sound Collage", "Sound Effects",
    "Sound Poetry", "Soundtrack", "South Indian Traditional", "Space-Rock",
    "Spanish", "Spoken", "Spoken Weird", "Spoken Word", "Surf", "Symphony",
    "Synth Pop", "Talk Radio", "Tango", "Techno", "Thrash", "Trip-Hop",
    "Turkish", "Unclassifiable", "Western Swing", "Wonky", "hiphop",
]

SELECTED_GENRES = [
    "Rock", "Jazz", "Hip-Hop", "Classical", "Pop", "Blues", "Country",
    "Electronic", "Reggae - Dancehall", "Soul-RnB", "Funk", "Ambient",
    "Metal", "Punk", "Folk", "Experimental"
]


def load_fma_dataset(
    dataset_name: str = "benjamin-paine/free-music-archive-medium",
    split: str = "train",
    streaming: bool = False,
    trust_remote_code: bool = False,
):
    return load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
    )


def get_genre_description(genres: list, genre_names: list) -> str:
    if not genres:
        return "instrumental music"
    names = []
    for g in genres:
        idx = int(g) if not isinstance(g, str) else g
        if isinstance(idx, int) and 0 <= idx < len(genre_names):
            names.append(genre_names[idx])
    return ", ".join(names) if names else "instrumental music"


def create_manifest_from_hf(
    output_dir: Path,
    dataset_name: str = "benjamin-paine/free-music-archive-medium",
    split_ratio: tuple[float, float, float] = (0.9, 0.05, 0.05),
    max_samples: Optional[int] = None,
    genre_filter: Optional[list[str]] = SELECTED_GENRES,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_fma_dataset(dataset_name=dataset_name)
    if "audio" in getattr(dataset, "column_names", []):
        dataset = dataset.remove_columns("audio")
    n = len(dataset)
    if max_samples:
        n = min(n, max_samples)
        indices = list(range(n))
    else:
        indices = list(range(n))

    train_ratio, valid_ratio, test_ratio = split_ratio
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = n - n_train - n_valid

    train_idx = indices[:n_train]
    valid_idx = indices[n_train : n_train + n_valid]
    test_idx = indices[n_train + n_valid :]

    splits = {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
    }

    manifest_paths = {}
    for split_name, idx_list in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        manifest_path = split_dir / "data.jsonl"

        with open(manifest_path, "w") as f:
            for i in idx_list:
                row = dataset[i]

                genres = row.get("genres", [])
                genre_desc = get_genre_description(genres, GENRE_NAMES)

                if genre_filter and genre_desc not in genre_filter:
                    if not any(g in genre_filter for g in genre_desc.split(", ")):
                        continue

                entry = {
                    "path": f"__hf_index_{i}__",
                    "duration": 30.0,
                    "genre": genre_desc,
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                }
                f.write(json.dumps(entry) + "\n")

        manifest_paths[split_name] = manifest_path

    return manifest_paths


def export_audio_to_disk(
    output_dir: Path,
    dataset_name: str = "benjamin-paine/free-music-archive-medium",
    split_ratio: tuple[float, float, float] = (0.9, 0.05, 0.05),
    max_samples: Optional[int] = None,
    genre_filter: Optional[list[str]] = None,
    audio_format: str = "flac",
) -> dict[str, Path]:
    """
    Export FMA audio to disk and create manifests with real paths.

    Downloads dataset, saves audio as FLAC (default) or WAV, writes AudioCraft manifests.
    """
    import io
    import soundfile as sf
    from datasets import Audio

    output_dir = Path(output_dir)
    audio_base = output_dir / "audio"
    audio_base.mkdir(parents=True, exist_ok=True)

    dataset = load_fma_dataset(dataset_name=dataset_name)
    # Avoid torchcodec dependency — get raw bytes and decode with soundfile
    dataset = dataset.cast_column("audio", Audio(decode=False))
    n = len(dataset)
    if max_samples:
        n = min(n, max_samples)

    train_ratio, valid_ratio, test_ratio = split_ratio
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    splits = {
        "train": (0, n_train),
        "valid": (n_train, n_train + n_valid),
        "test": (n_train + n_valid, n),
    }

    manifest_paths = {}
    for split_name, (start, end) in splits.items():
        split_audio = audio_base / split_name
        split_audio.mkdir(exist_ok=True)
        manifest_path = output_dir / split_name / "data.jsonl"
        (output_dir / split_name).mkdir(exist_ok=True)

        with open(manifest_path, "w") as f:
            for i in range(start, end):
                row = dataset[i]
                audio = row.get("audio")
                if not audio:
                    continue

                try:
                    if i % 100 == 0:
                        print(f"Processing audio file {i}/{end}...")
                    audio_bytes = audio.get("bytes")
                    if not audio_bytes:
                        continue
                    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
                    audio_path = split_audio / f"{i}.{audio_format}"
                    sf.write(str(audio_path), audio_array, sample_rate, format=audio_format.upper())

                    genres = row.get("genres", [])
                    genre_desc = get_genre_description(genres, GENRE_NAMES)
                    manifest_entry = {
                        "audio": str(audio_path),
                        "text": genre_desc,
                        "genre": genre_desc,
                    }
                    f.write(json.dumps(manifest_entry) + "\n")

                except Exception as e:
                    print(f"Error processing audio at index {i}: {e}")

        manifest_paths[split_name] = manifest_path

    return manifest_paths


def fix_manifests(
    output_dir: Path,
    dataset_name: str = "benjamin-paine/free-music-archive-medium",
    split_ratio: tuple[float, float, float] = (0.9, 0.05, 0.05),
    max_samples: Optional[int] = None,
) -> dict[str, Path]:
    """
    Re-generate manifests with correct genre/text from the cached dataset.

    Does not re-download or re-encode audio — only rewrites the JSONL files.
    Only entries whose WAV file already exists on disk are included.
    """
    output_dir = Path(output_dir)
    audio_base = output_dir / "audio"

    dataset = load_fma_dataset(dataset_name=dataset_name)
    if "audio" in dataset.column_names:
        dataset = dataset.remove_columns(["audio"])

    n = len(dataset)
    if max_samples:
        n = min(n, max_samples)

    train_ratio, valid_ratio, _ = split_ratio
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    splits = {
        "train": (0, n_train),
        "valid": (n_train, n_train + n_valid),
        "test": (n_train + n_valid, n),
    }

    manifest_paths = {}
    for split_name, (start, end) in splits.items():
        split_audio = audio_base / split_name
        manifest_path = output_dir / split_name / "data.jsonl"
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

        written = 0
        with open(manifest_path, "w") as f:
            for i in range(start, end):
                # Prefer FLAC, fall back to WAV for older exports
                audio_path = split_audio / f"{i}.flac"
                if not audio_path.exists():
                    audio_path = split_audio / f"{i}.wav"
                if not audio_path.exists():
                    continue

                row = dataset[i]
                genres = row.get("genres", [])
                genre_desc = get_genre_description(genres, GENRE_NAMES)

                manifest_entry = {
                    "audio": str(audio_path),
                    "text": genre_desc,
                    "genre": genre_desc,
                }
                f.write(json.dumps(manifest_entry) + "\n")
                written += 1

        manifest_paths[split_name] = manifest_path
        print(f"  {split_name}: {written} entries -> {manifest_path}")

    return manifest_paths


def main():
    parser = argparse.ArgumentParser(description="FMA dataset pipeline for MusicGen")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/manifests",
        help="Output directory for manifests and audio",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        nargs=3,
        default=[0.9, 0.05, 0.05],
        help="Train/valid/test split ratio",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per split (for quick testing)",
    )
    parser.add_argument(
        "--export_audio",
        action="store_true",
        help="Export audio to disk (requires significant storage)",
    )
    parser.add_argument(
        "--fix_manifests",
        action="store_true",
        help="Re-generate manifests with correct genre/text (no audio re-export)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="flac",
        choices=["flac", "wav"],
        help="Audio format for export (default: flac, saves ~50-60%% vs wav)",
    )
    args = parser.parse_args()

    split_ratio = tuple(args.split_ratio)
    output_dir = Path(args.output_dir)

    if args.fix_manifests:
        print("Re-generating manifests from existing audio...")
        paths = fix_manifests(
            output_dir=output_dir,
            split_ratio=split_ratio,
            max_samples=args.max_samples,
        )
        print("Done.")
        for split, path in paths.items():
            print(f"  {split}: {path}")
        return

    if args.export_audio:
        paths = export_audio_to_disk(
            output_dir=output_dir,
            split_ratio=split_ratio,
            max_samples=args.max_samples,
            audio_format=args.format,
        )
        print("Exported audio and created manifests:")
    else:
        paths = create_manifest_from_hf(
            output_dir=output_dir,
            split_ratio=split_ratio,
            max_samples=args.max_samples,
        )
        print("Created manifest placeholders (use --export_audio for full pipeline):")

    for split, path in paths.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()
