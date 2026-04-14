"""
Decoding Audit.

Verifies that greedy (argmax) decoding is temperature-invariant. If
use_sampling=False is truly greedy/argmax, changing temperature should NOT
change the output sequence — temperature only scales logits before the argmax,
which is invariant to positive scaling.

Generates audio at T = 0.7, 0.9, 1.0, 1.2, 1.5 with use_sampling=False,
hashes both token sequences and audio outputs, and reports whether they match.

Usage:
    python -m src.audit_decoding \
        --prompt "upbeat jazz piano music" \
        --output_dir ./results/audit
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

AUDIT_TEMPERATURES = [0.7, 0.9, 1.0, 1.2, 1.5]
AUDIT_DURATION = 10.0  # short for speed; raise for a more thorough audit


def _md5(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).astype(np.float32).tobytes()).hexdigest()


def _generate_greedy(
    model: MusicGen,
    prompt: str,
    temperature: float,
    duration: float,
) -> dict:
    """
    Run one greedy generation and return generation args, hashes, and tokens.

    Tries AudioCraft's return_tokens=True (available in audiocraft>=1.2.0) to
    capture the raw LM token sequences alongside the audio.  Falls back to
    audio-only hashing for older builds.
    """
    model.set_generation_params(
        duration=duration,
        temperature=temperature,
        use_sampling=False,   # greedy / argmax
        top_k=0,
        top_p=0.0,
    )

    gen_args = {
        "duration": duration,
        "temperature": temperature,
        "use_sampling": False,
        "top_k": 0,
        "top_p": 0.0,
    }

    # try to get raw tokens via return_tokens; falls back to audio hash if unavailable
    tokens = None
    try:
        result = model.generate([prompt], return_tokens=True)
        if isinstance(result, tuple) and len(result) == 2:
            wav, tokens = result
        else:
            wav = result
    except TypeError:
        # older audiocraft build without return_tokens
        wav = model.generate([prompt])

    wav_arr = wav[0].cpu().numpy()
    audio_hash = _md5(wav_arr)

    if tokens is not None:
        tok_arr = tokens.cpu().numpy()
        token_hash = _md5(tok_arr)
        # first 20 tokens of codebook 0 for inspection
        first20 = tok_arr[0, 0, :20].tolist()
    else:
        token_hash = None
        first20 = None

    return {
        "gen_args": gen_args,
        "audio_hash": audio_hash,
        "token_hash": token_hash,
        "tokens_codebook0_first20": first20,
        # popped before serialising to json
        "_wav": torch.from_numpy(wav_arr),
        "_sample_rate": model.sample_rate,
    }


def audit_greedy(
    prompt: str,
    output_dir: Path,
    model_name: str = "facebook/musicgen-small",
    duration: float = AUDIT_DURATION,
) -> dict:
    """
    Generate with greedy decoding at each temperature, compare outputs.

    Returns a report dict; also writes audit_report.json to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    model = MusicGen.get_pretrained(model_name)

    records = []
    for temp in AUDIT_TEMPERATURES:
        print(f"  Generating  use_sampling=False  T={temp:.2f}  top_k=0  top_p=0.0 ...")
        entry = _generate_greedy(model, prompt, temperature=temp, duration=duration)

        stem = str(output_dir / f"greedy_t{temp:.2f}")
        audio_write(stem, entry.pop("_wav"), entry.pop("_sample_rate"),
                    format="wav", add_suffix=True)
        entry["audio_path"] = f"{stem}.wav"

        records.append(entry)
        print(f"    audio_hash : {entry['audio_hash']}")
        if entry["token_hash"]:
            print(f"    token_hash : {entry['token_hash']}")

    # check if all outputs match
    audio_hashes = [r["audio_hash"] for r in records]
    token_hashes = [r["token_hash"] for r in records if r["token_hash"] is not None]

    audio_identical = len(set(audio_hashes)) == 1
    tokens_identical = (len(set(token_hashes)) == 1) if token_hashes else None

    if audio_identical and (tokens_identical is None or tokens_identical):
        verdict = (
            "CONFIRMED — greedy decoding is temperature-invariant (expected). "
            "Temperature does not affect argmax selection; the claim holds."
        )
    elif not audio_identical:
        verdict = (
            "FAILED — audio outputs differ across temperatures with use_sampling=False. "
            "This means decoding is NOT purely greedy. Possible causes: "
            "(1) use_sampling flag is not being respected; "
            "(2) temperature is applied before argmax in this AudioCraft build; "
            "(3) non-determinism from CUDA ops. Investigate before proceeding."
        )
    else:
        verdict = (
            "PARTIAL — audio is identical but token hashes differ. "
            "Likely floating-point precision difference in encoding; "
            "treat as temperature-invariant but flag for further review."
        )

    report = {
        "prompt": prompt,
        "model": model_name,
        "duration_s": duration,
        "claim_under_test": (
            "greedy (use_sampling=False) output is identical at any temperature"
        ),
        "verdict": verdict,
        "audio_outputs_identical": audio_identical,
        "token_outputs_identical": tokens_identical,
        "temperatures_tested": AUDIT_TEMPERATURES,
        "records": records,
    }

    report_path = output_dir / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"Report  : {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Audit greedy decoding: verify temperature invariance"
    )
    parser.add_argument(
        "--prompt", type=str, default="upbeat jazz piano music",
        help="Text prompt to use for all generations",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/audit",
        help="Directory to save audio files and audit_report.json",
    )
    parser.add_argument(
        "--model", type=str, default="facebook/musicgen-small",
        help="MusicGen model checkpoint",
    )
    parser.add_argument(
        "--duration", type=float, default=AUDIT_DURATION,
        help=f"Generation length in seconds (default: {AUDIT_DURATION})",
    )
    args = parser.parse_args()

    audit_greedy(
        prompt=args.prompt,
        output_dir=Path(args.output_dir),
        model_name=args.model,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
