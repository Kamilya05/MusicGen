import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


def generate(
    prompts: list[str],
    output_dir: str | Path,
    model_name: str = "facebook/musicgen-small",
    duration: float = 30.0,
    num_samples: int = 1,
    temperature: float = 1.0,
    use_sampling: bool = True,
    top_k: int = 250,
    top_p: float = 0.0,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MusicGen.get_pretrained(model_name)
    model.set_generation_params(
        duration=duration,
        temperature=temperature,
        use_sampling=use_sampling,
        top_k=top_k,
        top_p=top_p,
    )

    outputs = []
    for i, prompt in enumerate(prompts):
        for j in range(num_samples):
            wav = model.generate([prompt])
            wav = wav[0]
            stem = str(output_dir / f"gen_{i:03d}_{j:03d}")
            path = audio_write(
                stem,
                wav.cpu(),
                model.sample_rate,
                format="wav",
                add_suffix=True,
            )
            outputs.append(Path(path))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Generate music with MusicGen")
    parser.add_argument(
        "--prompts", type=str, nargs="+",
        default=["upbeat electronic dance music"],
        help="Text prompts for generation",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--model", type=str, default="facebook/musicgen-small")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Samples per prompt")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (ignored for greedy)")
    parser.add_argument("--top_k", type=int, default=250,
                        help="Top-k truncation (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.0,
                        help="Top-p nucleus truncation (0 = disabled)")
    parser.add_argument("--no_sampling", action="store_true",
                        help="Use greedy (argmax) decoding instead of sampling")
    args = parser.parse_args()

    paths = generate(
        prompts=args.prompts,
        output_dir=args.output_dir,
        model_name=args.model,
        duration=args.duration,
        num_samples=args.num_samples,
        temperature=args.temperature,
        use_sampling=not args.no_sampling,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(f"Generated {len(paths)} samples to {args.output_dir}")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
