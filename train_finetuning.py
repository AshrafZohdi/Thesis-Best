"""
Step 6: GPT-2 + LoRA Fine-tuning — all 10 models
==================================================
Trains 5 traditions × 2 tokenisers (REMI, EC-REMI) = 10 models.

Run full pipeline:
    python train_finetuning.py

Run single tradition/tokeniser:
    python train_finetuning.py --tradition irish_folk --tokeniser remi

References
----------
  Yao & Chen (2025) — GPT-2 + LoRA for symbolic music generation
  Hu et al. (2022) — LoRA: Low-Rank Adaptation of Large Language Models
"""

from __future__ import annotations
import sys, os, json, argparse, warnings, importlib, importlib.util as _ilu
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ── 1. Import all external ML packages BEFORE src/ is on sys.path.
#        This ensures transformers' lazy imports resolve against HF tokenizers,
#        not src/tokenizers/ (which is an empty stub package).
from miditok import REMI, TokenizerConfig
from symusic import Score as SScore

# Force-load transformers modules that have lazy imports of HF tokenizers.
# Must happen before ec_remi.py's import guard clears sys.modules['tokenizers'].
importlib.import_module("transformers.configuration_utils")
importlib.import_module("transformers.models.gpt2.modeling_gpt2")
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model, TaskType

# ── 2. Now safe to add src/ and load ECREMITokenizer via importlib
#        (importlib bypasses package-name collision with HF tokenizers)
sys.path.insert(0, str(Path(__file__).parent / "src"))

def _load_ec_remi():
    _path = Path(__file__).parent / "src" / "tokenizers" / "ec_remi.py"
    _spec = _ilu.spec_from_file_location("_ec_remi_ft", str(_path))
    _mod  = _ilu.module_from_spec(_spec)
    sys.modules["_ec_remi_ft"] = _mod
    _spec.loader.exec_module(_mod)
    return _mod.ECREMITokenizer

ECREMITokenizer = _load_ec_remi()

# ── Config ──────────────────────────────────────────────────────────────────────
MIDI_ROOT  = Path("data/processed")
META_DIR   = Path("data/metadata")
CKPT_ROOT  = Path("outputs/checkpoints")
RESULTS    = Path("results")
CKPT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(exist_ok=True)

CHUNK_SIZE = 512
STRIDE     = 256
N_EPOCHS   = 5
BATCH_SIZE = 4
LR         = 5e-4
LORA_R     = 8
LORA_ALPHA = 16
LORA_DROP  = 0.1
SEED       = 42

TRADITION_CONFIG = {
    "western_classical": {
        "label": "Western Classical",
        "meta_csv": "maestro_selected.csv",
        "label_col": None,
        "modal_default": "unknown",
    },
    "hindustani": {
        "label": "Hindustani",
        "meta_csv": "hindustani_tracks.csv",
        "label_col": "raga",
    },
    "carnatic": {
        "label": "Carnatic",
        "meta_csv": "carnatic_tracks.csv",
        "label_col": "raga",
    },
    "irish_folk": {
        "label": "Irish Folk",
        "meta_csv": "irish_folk_tunes.csv",
        "label_col": "mode",
    },
    "turkish_makam": {
        "label": "Turkish Makam",
        "meta_csv": "symbtr_selected.csv",
        "label_col": "makam",
    },
}


# ── Dataset ──────────────────────────────────────────────────────────────────────

class MusicChunkDataset(Dataset):
    """Fixed-length 512-token chunks for causal LM."""

    def __init__(self, chunks: list[list[int]]):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = torch.tensor(self.chunks[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


class FixedLenCollator:
    def __call__(self, features):
        ids = torch.stack([f["input_ids"] for f in features])
        return {"input_ids": ids, "labels": ids.clone()}


def _load_label_map(trad_key: str) -> dict[str, str]:
    tinfo = TRADITION_CONFIG[trad_key]
    if not tinfo.get("label_col"):
        return {}
    meta_path = META_DIR / tinfo["meta_csv"]
    if not meta_path.exists():
        return {}
    df = pd.read_csv(meta_path)
    midi_col = "midi_path" if "midi_path" in df.columns else "processed_filename"
    if midi_col not in df.columns:
        return {}
    label_map = {}
    for _, row in df.iterrows():
        fname = Path(str(row[midi_col])).name
        val = row.get(tinfo["label_col"])
        label_map[fname] = str(val) if pd.notna(val) else None
    return label_map


def _chunk(ids: list[int]) -> list[list[int]]:
    return [
        ids[i : i + CHUNK_SIZE]
        for i in range(0, len(ids) - CHUNK_SIZE + 1, STRIDE)
    ]


def build_remi_dataset(trad_key: str, remi_tok) -> tuple[MusicChunkDataset, dict]:
    midi_dir = MIDI_ROOT / trad_key / "midi"
    files = sorted(list(midi_dir.glob("*.midi")) + list(midi_dir.glob("*.mid")))
    all_chunks, ok, err = [], 0, 0
    for fpath in files:
        try:
            seqs = remi_tok.encode(SScore(str(fpath)))
            if seqs:
                all_chunks.extend(_chunk(seqs[0].ids))
                ok += 1
            else:
                err += 1
        except Exception:
            err += 1
    return MusicChunkDataset(all_chunks), {"ok": ok, "err": err, "chunks": len(all_chunks)}


def build_ec_remi_dataset(trad_key: str, ec_tok) -> tuple[MusicChunkDataset, dict]:
    midi_dir = MIDI_ROOT / trad_key / "midi"
    files = sorted(list(midi_dir.glob("*.midi")) + list(midi_dir.glob("*.mid")))
    label_map = _load_label_map(trad_key)
    modal_default = TRADITION_CONFIG[trad_key].get("modal_default")
    all_chunks, ok, err = [], 0, 0
    for fpath in files:
        modal_label = label_map.get(fpath.name, modal_default)
        try:
            tokens = ec_tok.tokenize(fpath, tradition=trad_key, modal_label=modal_label)
            if tokens:
                all_chunks.extend(_chunk(ec_tok.encode(tokens)))
                ok += 1
            else:
                err += 1
        except Exception:
            err += 1
    return MusicChunkDataset(all_chunks), {"ok": ok, "err": err, "chunks": len(all_chunks)}


# ── Model ────────────────────────────────────────────────────────────────────────

def create_gpt2_lora(vocab_size: int) -> torch.nn.Module:
    """GPT-2 small with resized embedding and LoRA on c_attn."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(vocab_size)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROP,
        target_modules=["c_attn"],
        bias="none",
    )
    return get_peft_model(model, lora_cfg)


# ── Training ─────────────────────────────────────────────────────────────────────

def train_one(
    trad_key: str,
    tokeniser_type: str,
    remi_tok=None,
    ec_tok=None,
) -> Optional[dict]:
    set_seed(SEED)
    label = TRADITION_CONFIG[trad_key]["label"]
    print(f"\n{'='*64}")
    print(f"  {label}  |  {tokeniser_type.upper()}")
    print(f"{'='*64}")

    if tokeniser_type == "remi":
        dataset, stats = build_remi_dataset(trad_key, remi_tok)
        vocab_size = remi_tok.vocab_size
    else:
        dataset, stats = build_ec_remi_dataset(trad_key, ec_tok)
        vocab_size = ec_tok.vocab_size

    print(f"  Files: {stats['ok']} ok, {stats['err']} err | 512-tok chunks: {stats['chunks']}")
    if not dataset:
        print("  No data — skipping.")
        return None

    val_n = max(1, int(0.1 * len(dataset)))
    train_ds = MusicChunkDataset(dataset.chunks[:-val_n])
    val_ds   = MusicChunkDataset(dataset.chunks[-val_n:])
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    model = create_gpt2_lora(vocab_size)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_train:,} / {n_total:,}  ({100*n_train/n_total:.2f}%)")

    ckpt_dir = CKPT_ROOT / f"{trad_key}_{tokeniser_type}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    warmup = max(10, len(train_ds) // (BATCH_SIZE * 10))
    log_steps = max(1, len(train_ds) // (BATCH_SIZE * 5))

    train_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_steps=warmup,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=log_steps,
        report_to="none",
        seed=SEED,
        use_cpu=True,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=FixedLenCollator(),
    )

    result = trainer.train()
    final_dir = ckpt_dir / "final"
    trainer.save_model(str(final_dir))

    row = {
        "tradition":        trad_key,
        "tokeniser":        tokeniser_type,
        "vocab_size":       vocab_size,
        "n_files_ok":       stats["ok"],
        "n_chunks_train":   len(train_ds),
        "n_chunks_val":     len(val_ds),
        "train_loss":       round(result.training_loss, 4),
        "trainable_params": n_train,
        "total_params":     n_total,
    }
    with open(ckpt_dir / "train_stats.json", "w") as f:
        json.dump(row, f, indent=2)

    print(f"  Final loss: {result.training_loss:.4f}")
    print(f"  Checkpoint: {final_dir}")
    return row


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="EC-REMI thesis Step 6 fine-tuning")
    ap.add_argument("--tradition", default=None,
                    choices=list(TRADITION_CONFIG.keys()),
                    help="Train only this tradition (default: all)")
    ap.add_argument("--tokeniser", default=None,
                    choices=["remi", "ec_remi"],
                    help="Train only this tokeniser (default: both)")
    args = ap.parse_args()

    remi_tok = REMI(TokenizerConfig())
    ec_tok   = ECREMITokenizer()
    print(f"REMI vocab: {remi_tok.vocab_size}  |  EC-REMI vocab: {ec_tok.vocab_size}")

    traditions = [args.tradition] if args.tradition else list(TRADITION_CONFIG.keys())
    tokenisers = [args.tokeniser] if args.tokeniser else ["remi", "ec_remi"]

    all_rows = []
    for trad in traditions:
        for tok_type in tokenisers:
            row = train_one(trad, tok_type, remi_tok=remi_tok, ec_tok=ec_tok)
            if row:
                all_rows.append(row)

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        out_csv = RESULTS / "finetuning_stats.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"\nAll stats → {out_csv}")
        print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
