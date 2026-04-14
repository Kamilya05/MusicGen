"""
Repetition & Diversity Metrics.

Two complementary, directly measurable quantities:

1. Repetition Score (per file)
   ─────────────────────────────
   Measures intra-clip looping via the frame-level self-similarity matrix of
   the mel spectrogram.  Each frame is compared to every other frame using
   cosine similarity.  The mean off-diagonal value is the repetition score:

     0.0 → all frames are orthogonal  (no repetition)
     1.0 → every frame is identical  (perfect loop)

   Additionally we report `loop_ratio`: the fraction of off-diagonal frame
   pairs with similarity > 0.9, which directly counts repeating segments.

   Expected direction:
     low temperature  → high repetition score (stuck in one pattern)
     high temperature → low repetition score  (fragmented, incoherent)

2. Diversity Score (across samples of the same prompt)
   ─────────────────────────────────────────────────────
   Mean pairwise L2 distance between mean-mel embeddings computed over N
   samples generated for the same prompt.

     low diversity  → samples cluster together (greedy, low temp)
     high diversity → samples are spread out (high temp, sampling)

Usage:
    # Single directory
    python -m src.metrics.repetition \
        --audio_dir ./results/sweep/temp_0.7 \
        --output_file ./results/metrics/rep_temp_0.7.json

    # All conditions from a manifest
    python -m src.metrics.repetition \
        --manifest ./results/sweep/manifest.json \
        --output_dir ./results/metrics/
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

try:
    import librosa
    _HAVE_LIBROSA = True
except ImportError:
    _HAVE_LIBROSA = False


# audio loading

def _load_mono(path: Path, target_sr: int = 22_050) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, target_sr)
            audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
            sr = target_sr
        except Exception:
            pass  # proceed at original sr
    return audio, sr


# Mel spectrogram

def _mel_spec(audio: np.ndarray, sr: int, n_mels: int = 128) -> np.ndarray:
    """Return log-mel spectrogram [n_mels × T], using librosa when available."""
    if _HAVE_LIBROSA:
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels,
                                              n_fft=2048, hop_length=512)
        return librosa.power_to_db(mel, ref=np.max)

    # Pure numpy/scipy fallback
    n_fft, hop = 2048, 512
    frames = [
        audio[s: s + n_fft] * np.hanning(n_fft)
        for s in range(0, len(audio) - n_fft, hop)
    ]
    if not frames:
        return np.zeros((n_mels, 1))

    stft_mag = np.abs(np.fft.rfft(np.array(frames), n=n_fft)) ** 2  # [T, F]

    # Simple triangular mel filterbank
    n_freqs = n_fft // 2 + 1
    fmin, fmax = 0.0, sr / 2.0
    mel_lo = 2595 * np.log10(1 + fmin / 700)
    mel_hi = 2595 * np.log10(1 + fmax / 700)
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int).clip(0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        if mid > lo:
            fb[m - 1, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[m - 1, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)

    mel = fb @ stft_mag.T  # [n_mels, T]
    return 10 * np.log10(mel + 1e-10)


# repetition score (per file)

def compute_repetition_score(
    audio_path: Path,
    sr: int = 22_050,
    max_frames: int = 512,
) -> dict:
    """
    Compute per-file repetition and loop metrics.

    Args:
        audio_path: Path to WAV file.
        sr: Analysis sample rate.
        max_frames: Cap on number of mel frames to keep the self-similarity
                    matrix tractable (uniformly subsampled if exceeded).

    Returns:
        Dict with repetition_score, loop_ratio, n_frames, audio_path.
    """
    audio, actual_sr = _load_mono(Path(audio_path), target_sr=sr)
    mel = _mel_spec(audio, actual_sr)  # [n_mels, T]

    T = mel.shape[1]
    if T < 2:
        return {"repetition_score": 0.0, "loop_ratio": 0.0,
                "n_frames": T, "audio_path": str(audio_path)}

    # Subsample frames uniformly if too many
    if T > max_frames:
        idx = np.linspace(0, T - 1, max_frames, dtype=int)
        mel = mel[:, idx]
        T = max_frames

    # Normalise column-wise for cosine similarity
    norms = np.linalg.norm(mel, axis=0, keepdims=True) + 1e-8
    mel_n = mel / norms  # [n_mels, T]

    sim = mel_n.T @ mel_n  # [T, T]  — cosine self-similarity matrix

    off_diag = ~np.eye(T, dtype=bool)
    rep_score = float(sim[off_diag].mean())
    loop_ratio = float((sim[off_diag] > 0.90).mean())

    return {
        "repetition_score": rep_score,
        "loop_ratio": loop_ratio,
        "n_frames": int(T),
        "audio_path": str(audio_path),
    }


# diversity score (across samples)

def _mean_mel_embedding(path: Path, sr: int = 22_050) -> np.ndarray:
    """Global mean-mel embedding: [n_mels] summary vector for one clip."""
    audio, actual_sr = _load_mono(path, target_sr=sr)
    mel = _mel_spec(audio, actual_sr)
    return mel.mean(axis=1)  # [n_mels]


def compute_diversity_score(
    audio_paths: list[Path],
    sr: int = 22_050,
) -> dict:
    """
    Mean pairwise L2 distance between mean-mel embeddings.

    High value → samples are spread in mel space (diverse).
    Low value  → samples cluster together (deterministic / low entropy).
    """
    paths = [Path(p) for p in audio_paths]
    if len(paths) < 2:
        return {"diversity_score": 0.0, "diversity_std": 0.0,
                "n_pairs": 0, "n_samples": len(paths)}

    embs = [_mean_mel_embedding(p, sr) for p in paths]
    n = len(embs)

    distances = [
        float(np.linalg.norm(embs[i] - embs[j]))
        for i in range(n) for j in range(i + 1, n)
    ]

    return {
        "diversity_score": float(np.mean(distances)),
        "diversity_std": float(np.std(distances)),
        "n_pairs": len(distances),
        "n_samples": n,
    }


# directory-level evaluation

def evaluate_condition(
    audio_dir: Path,
    manifest_path: Optional[Path] = None,
    condition_name: Optional[str] = None,
    output_file: Optional[Path] = None,
    sr: int = 22_050,
) -> dict:
    """
    Compute repetition and diversity metrics for all WAV files in a directory.

    When manifest_path + condition_name are provided, per-prompt diversity is
    also computed (samples of the same prompt are compared separately).
    """
    audio_dir = Path(audio_dir)
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {audio_dir}")

    per_file = [compute_repetition_score(p, sr=sr) for p in wav_files]

    # Overall diversity across every clip in this condition
    diversity = compute_diversity_score(wav_files, sr=sr)

    # Per-prompt diversity (if manifest is available)
    per_prompt_diversity: dict[str, dict] = {}
    if manifest_path and condition_name:
        with open(manifest_path) as f:
            manifest = json.load(f)
        records = [r for r in manifest["records"] if r["condition"] == condition_name]
        groups: dict[int, list[Path]] = {}
        for r in records:
            groups.setdefault(r["prompt_idx"], []).append(Path(r["audio_path"]))
        for p_idx, paths in groups.items():
            per_prompt_diversity[str(p_idx)] = compute_diversity_score(paths, sr=sr)

    result = {
        "condition_dir": str(audio_dir),
        "n_files": len(wav_files),
        "mean_repetition_score": float(np.mean([r["repetition_score"] for r in per_file])),
        "std_repetition_score": float(np.std([r["repetition_score"] for r in per_file])),
        "mean_loop_ratio": float(np.mean([r["loop_ratio"] for r in per_file])),
        "diversity": diversity,
        "per_prompt_diversity": per_prompt_diversity,
        "per_file": per_file,
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
        description="Compute repetition and diversity metrics for generated audio"
    )
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Directory of WAV files (single-condition mode)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Sweep manifest.json (all-conditions mode)")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (all-conditions mode)")
    args = parser.parse_args()

    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        sweep_dir = Path(args.manifest).parent
        out_dir = Path(args.output_dir or "./results/metrics")
        for cond in manifest["conditions"]:
            name = cond["name"]
            cond_dir = sweep_dir / name
            result = evaluate_condition(
                audio_dir=cond_dir,
                manifest_path=Path(args.manifest),
                condition_name=name,
                output_file=out_dir / f"rep_{name}.json",
            )
            print(
                f"  {name:12s}  rep={result['mean_repetition_score']:.4f}  "
                f"loop={result['mean_loop_ratio']:.4f}  "
                f"div={result['diversity']['diversity_score']:.4f}"
            )
    elif args.audio_dir:
        result = evaluate_condition(
            audio_dir=Path(args.audio_dir),
            output_file=Path(args.output_file) if args.output_file else None,
        )
        print(f"Mean repetition score : {result['mean_repetition_score']:.4f}")
        print(f"Mean loop ratio       : {result['mean_loop_ratio']:.4f}")
        print(f"Diversity score       : {result['diversity']['diversity_score']:.4f}")
    else:
        parser.error("Supply --audio_dir or --manifest")


if __name__ == "__main__":
    main()
