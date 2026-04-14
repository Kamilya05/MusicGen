"""
Fine-tuning Script for MusicGen on FMA.

AudioCraft's MusicGen is NOT a HuggingFace PreTrainedModel, so the standard
Trainer cannot be used.  Instead we:

  1. Freeze the EnCodec compression model (audio ↔ codec tokens).
  2. Freeze the T5 text-condition encoder (or optionally fine-tune it).
  3. Fine-tune only the language model (transformer) that predicts codec
     tokens conditioned on text.

Data flow per training step:
  audio WAV  ──EnCodec──►  codec tokens  [B, K, T]
  text prompt ──T5──►  conditioning attributes
  codec tokens + conditioning  ──LM forward──►  logits
  cross-entropy(logits, codec_tokens)  ──►  loss  ──►  backward

Memory note: musicgen-small needs ~8 GB VRAM for a batch of 2 at 30 s clips.
Use --duration 10 and --batch_size 1 if you are memory-constrained.

Usage:
    python -m src.train \
        --manifest_dir ./data/manifests \
        --audio_dir    ./data/audio \
        --output_dir   ./trained_model \
        --epochs       10 \
        --batch_size   2 \
        --lr           1e-4
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

load_dotenv()

from audiocraft.models import MusicGen


class FMAManifestDataset(Dataset):
    """
    Loads (audio_path, text_prompt) pairs from an AudioCraft-style JSONL manifest.

    Each line must have an "audio" field with a real file path and optionally
    a "text" or "genre" field for the conditioning prompt.
    Skips entries that use __hf_index__ placeholders (not yet exported to disk).
    """

    def __init__(
        self,
        manifest_path: Path,
        audio_dir: Path,
        sample_rate: int = 32_000,
        duration: float = 30.0,
        max_samples: int | None = None,
    ):
        self.sr = sample_rate
        self.n_samples = int(sample_rate * duration)

        entries = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                path_str = obj.get("audio") or obj.get("path", "")
                if "__hf_index__" in path_str:
                    continue  # audio not yet exported to disk
                audio_path = Path(path_str) if Path(path_str).is_absolute() \
                             else audio_dir / path_str
                if not audio_path.exists():
                    continue
                text = obj.get("text") or obj.get("genre") or ""
                entries.append({"audio_path": audio_path, "text": text})

        if max_samples:
            entries = entries[:max_samples]
        self.entries = entries

        if not entries:
            raise RuntimeError(
                f"No usable entries found in {manifest_path}. "
                "Run `python -m src.data.pipeline --export_audio` first to download "
                "audio files from FMA to disk."
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        audio, sr = sf.read(str(entry["audio_path"]))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        # Resample if needed (simple nearest-neighbour; scipy for quality)
        if sr != self.sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, self.sr)
            audio = resample_poly(audio, self.sr // g, sr // g).astype(np.float32)

        # Pad or crop to fixed length
        if len(audio) < self.n_samples:
            audio = np.pad(audio, (0, self.n_samples - len(audio)))
        else:
            start = random.randint(0, len(audio) - self.n_samples)
            audio = audio[start: start + self.n_samples]

        return {
            "audio": torch.from_numpy(audio).unsqueeze(0),  # [1, T]
            "text": entry["text"],
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "audio": torch.stack([b["audio"] for b in batch]),  # [B, 1, T]
        "text":  [b["text"] for b in batch],
    }


def train(
    manifest_dir: Path,
    audio_dir: Path,
    output_dir: Path,
    model_name: str = "facebook/musicgen-small",
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 1e-4,
    duration: float = 30.0,
    max_samples: int | None = None,
    freeze_text_encoder: bool = True,
    train_layers: int = 4,
    use_lora: bool = False,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {model_name} …")
    model = MusicGen.get_pretrained(model_name)
    model.lm.to(device).float().train()
    model.compression_model.to(device).float().eval()

    # freeze encodec
    for p in model.compression_model.parameters():
        p.requires_grad_(False)

    if use_lora:
        from peft import LoraConfig, get_peft_model
        # target the attention output projection and feedforward layers.
        # separately — out_proj + linear1/linear2 covers ~half the trainable surface.
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["out_proj", "linear1", "linear2"],
            lora_dropout=0.05,
            bias="none",
        )
        model.lm = get_peft_model(model.lm, lora_config)
        model.lm.print_trainable_parameters()
    else:
        # Freeze all LM parameters first, then selectively unfreeze
        for p in model.lm.parameters():
            p.requires_grad_(False)

        # only fine-tune the last `train_layers` transformer layers
        # musicgen-small has 24 layers; training the last 4 is ~30M params vs 419M
        transformer_layers = list(model.lm.transformer.layers)
        for layer in transformer_layers[-train_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)

        # train the output projection 
        for p in model.lm.out_norm.parameters():
            p.requires_grad_(True)
        for p in model.lm.linears.parameters():
            p.requires_grad_(True)

    train_manifest = manifest_dir / "train" / "data.jsonl"
    valid_manifest = manifest_dir / "valid" / "data.jsonl"

    train_ds = FMAManifestDataset(
        train_manifest, audio_dir,
        sample_rate=model.compression_model.sample_rate,
        duration=duration, max_samples=max_samples,
    )
    valid_ds = FMAManifestDataset(
        valid_manifest, audio_dir,
        sample_rate=model.compression_model.sample_rate,
        duration=duration, max_samples=max_samples,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    print(f"Train: {len(train_ds)} samples  |  Valid: {len(valid_ds)} samples")

    trainable = [p for p in model.lm.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs * len(train_loader)
    )
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {n_trainable:,}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.lm.train()
        train_loss_sum, train_steps = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            audio = batch["audio"].to(device)          # [B, 1, T]
            texts = batch["text"]

            with torch.no_grad():
                # Encode audio → codec tokens  [B, K, T_codec]
                tokens, _ = model.compression_model.encode(audio)

            # Build text conditioning attributes
            attributes, _ = model._prepare_tokens_and_attributes(texts, None)

            # LM forward: predict each codec step from previous ones
            # tokens[:, :, :-1] is the input, tokens[:, :, 1:] is the target
            lm_output = model.lm.compute_predictions(
                tokens,          # [B, K, T_codec]
                attributes,      # list of ConditioningAttributes
            )

            # lm_output.logits: [B, K, T, card]
            logits = lm_output.logits       # [B, K, T, card]
            mask   = lm_output.mask         # [B, K, T]

            # Flatten for cross-entropy: only masked (predicted) positions
            B, K, T, card = logits.shape
            loss = F.cross_entropy(
                logits[mask].reshape(-1, card),
                tokens[mask].reshape(-1).long(),
            )

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimiser.step()
            scheduler.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train = train_loss_sum / max(train_steps, 1)

        model.lm.eval()
        val_loss_sum, val_steps = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch}/{epochs} [valid]"):
                audio = batch["audio"].to(device)
                texts = batch["text"]
                tokens, _ = model.compression_model.encode(audio)
                attributes, _ = model._prepare_tokens_and_attributes(texts, None)
                lm_output = model.lm.compute_predictions(tokens, attributes)
                logits, mask = lm_output.logits, lm_output.mask
                B, K, T, card = logits.shape
                loss = F.cross_entropy(
                    logits[mask].reshape(-1, card),
                    tokens[mask].reshape(-1).long(),
                )
                val_loss_sum += loss.item()
                val_steps += 1

        avg_val = val_loss_sum / max(val_steps, 1)
        print(f"Epoch {epoch:3d}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        # Save best checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt = output_dir / "best_lm.pt"
            if use_lora:
                # Save only the LoRA adapter weights — much smaller file
                model.lm.save_pretrained(output_dir / "lora_adapter")
            torch.save(model.lm.state_dict(), ckpt)
            print(f"  ✓ Best checkpoint saved → {ckpt}  (val_loss={avg_val:.4f})")

    # Save final
    torch.save(model.lm.state_dict(), output_dir / "final_lm.pt")
    print(f"\nTraining complete. Checkpoints in {output_dir}/")
    print("To generate with the fine-tuned model, load the LM weights:")
    print(f"  model.lm.load_state_dict(torch.load('{output_dir}/best_lm.pt'))")



def main():
    parser = argparse.ArgumentParser(description="Fine-tune MusicGen LM on FMA")
    parser.add_argument("--manifest_dir", type=str, default="./data/manifests",
                        help="Directory containing train/valid/test JSONL manifests")
    parser.add_argument("--audio_dir", type=str, default="./data/audio",
                        help="Root directory where FMA audio WAVs are stored")
    parser.add_argument("--output_dir", type=str, default="./trained_model")
    parser.add_argument("--model", type=str, default="facebook/musicgen-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Clip length in seconds (use 10.0 to save VRAM)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap dataset size for quick testing")
    parser.add_argument("--no_freeze_text", action="store_true",
                        help="Also fine-tune the text condition encoder (more memory)")
    parser.add_argument("--train_layers", type=int, default=4,
                        help="Number of transformer layers to fine-tune from the top (default: 4)")
    parser.add_argument("--lora", action="store_true",
                        help="Use LoRA adapters instead of full layer fine-tuning (recommended)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        manifest_dir=Path(args.manifest_dir),
        audio_dir=Path(args.audio_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        duration=args.duration,
        max_samples=args.max_samples,
        freeze_text_encoder=not args.no_freeze_text,
        train_layers=args.train_layers,
        use_lora=args.lora,
        device=args.device,
    )


if __name__ == "__main__":
    main()
