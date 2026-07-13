"""
Music Transformer training script.

Supports two modes:
  pretrain  — train on all traditions combined (REMI only)
  finetune  — fine-tune a pre-trained checkpoint on one tradition + tokeniser

Train / val / test split is done at the FILE level:
  80% train  |  10% val  |  10% test   (stratified per tradition)

The test split is written to disk once and never used during training.
Run this script twice:
  1. python train.py --mode pretrain --output_dir checkpoints/pretrain
  2. python train.py --mode finetune --tradition western_classical --tokeniser remi
                      --pretrain_dir checkpoints/pretrain
                      --output_dir checkpoints/finetune_wc_remi
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
# Add repo root so 'music_transformer.model' is importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

# Prevent local 'datasets/' folder from shadowing HuggingFace datasets package
_ds_local = ROOT / 'datasets'
if str(_ds_local) in sys.path:
    sys.path.remove(str(_ds_local))

from music_transformer.model import MusicTransformer, MusicTransformerConfig  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
TRADITIONS = [
    'western_classical', 'hindustani', 'carnatic', 'irish_folk', 'turkish_makam'
]
REMI_VOCAB_SIZE   = 284
EC_REMI_VOCAB_SIZE = 526
CHUNK_SIZE        = 512    # tokens per training example
SPLIT_SEED        = 42
TRAIN_FRAC        = 0.80
VAL_FRAC          = 0.10
# TEST_FRAC = 1 - TRAIN_FRAC - VAL_FRAC = 0.10 — never seen during training


# ── Tokenisation helpers ──────────────────────────────────────────────────────
def load_remi_tokenizer():
    from miditok import REMI, TokenizerConfig
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=False,
        use_rests=False,
        use_tempos=True,
        use_time_signatures=False,
        use_sustain_pedals=False,
        use_pitch_bends=False,
    )
    return REMI(tokenizer_config=config)


def load_ec_remi_tokenizer(tradition: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'ec_remi_module', ROOT / 'src' / 'tokenizers' / 'ec_remi.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ECREMITokenizer(), tradition


def midi_to_ids_remi(midi_path: Path, tok) -> list[int]:
    from miditok.classes import TokSequence
    from symusic import Score as SScore
    try:
        seqs = tok.encode(SScore(str(midi_path)))
        if seqs:
            return seqs[0].ids
    except Exception:
        pass
    return []


def midi_to_ids_ec_remi(midi_path: Path, tok, tradition: str) -> list[int]:
    try:
        tokens = tok.tokenize(midi_path, tradition=tradition)
        if tokens:
            return tok.encode(tokens)
    except Exception:
        pass
    return []


# ── Data split ────────────────────────────────────────────────────────────────
def get_split_files(tradition: str, split: str, split_cache_dir: Path) -> list[Path]:
    """
    Return MIDI file paths for a given tradition and split.
    Creates/reuses a deterministic split written to split_cache_dir.
    """
    cache_file = split_cache_dir / f'{tradition}_split.json'

    if cache_file.exists():
        with open(cache_file) as f:
            splits = json.load(f)
    else:
        midi_dir = ROOT / 'data' / 'processed' / tradition / 'midi'
        all_files = sorted(
            list(midi_dir.glob('*.mid')) + list(midi_dir.glob('*.midi'))
        )
        if not all_files:
            raise FileNotFoundError(f'No MIDI files found in {midi_dir}')

        rng = random.Random(SPLIT_SEED)
        shuffled = [str(p) for p in all_files]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = math.floor(n * TRAIN_FRAC)
        n_val   = math.floor(n * VAL_FRAC)

        splits = {
            'train': shuffled[:n_train],
            'val':   shuffled[n_train:n_train + n_val],
            'test':  shuffled[n_train + n_val:],
        }
        split_cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f'  Created split for {tradition}: '
              f'{len(splits["train"])} train / {len(splits["val"])} val / '
              f'{len(splits["test"])} test')

    return [Path(p) for p in splits[split]]


# ── Dataset ───────────────────────────────────────────────────────────────────
class MIDIChunkDataset(Dataset):
    """
    Chunkifies tokenized MIDI files into fixed-length windows.
    Each item is (input_ids, target_ids) where target is input shifted by 1.
    """

    def __init__(
        self,
        midi_files: list[Path],
        tok,
        tokeniser_type: str,
        tradition: str,
        chunk_size: int = CHUNK_SIZE,
        stride: int | None = None,
    ):
        self.chunk_size = chunk_size
        self.stride     = stride or chunk_size    # non-overlapping by default
        self.chunks: list[list[int]] = []

        for fpath in midi_files:
            if tokeniser_type == 'remi':
                ids = midi_to_ids_remi(fpath, tok)
            else:
                ids = midi_to_ids_ec_remi(fpath, tok, tradition)

            if len(ids) < chunk_size + 1:
                continue

            for start in range(0, len(ids) - chunk_size, self.stride):
                chunk = ids[start: start + chunk_size + 1]   # +1 for target shift
                self.chunks.append(chunk)

        if not self.chunks:
            raise ValueError(
                f'No chunks created for tradition={tradition}, tok={tokeniser_type}. '
                'Check that MIDI files are valid and long enough.'
            )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        chunk = torch.tensor(self.chunks[idx], dtype=torch.long)
        return chunk[:-1], chunk[1:]    # input, target


# ── LR scheduler: warmup + cosine decay ──────────────────────────────────────
def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps'  if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Device: {device}')

    output_dir    = Path(args.output_dir)
    split_dir     = Path(args.split_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Choose tokeniser ──────────────────────────────────────────────────────
    is_ec_remi = (args.tokeniser == 'ec_remi')

    if is_ec_remi:
        tok, tradition_tok = load_ec_remi_tokenizer(args.tradition)
        vocab_size = EC_REMI_VOCAB_SIZE
    else:
        tok = load_remi_tokenizer()
        tradition_tok = args.tradition
        vocab_size = REMI_VOCAB_SIZE

    # ── Gather files ──────────────────────────────────────────────────────────
    if args.mode == 'pretrain':
        traditions = TRADITIONS
        print('Pre-training on all traditions (REMI only):')
    else:
        traditions = [args.tradition]
        print(f'Fine-tuning on tradition={args.tradition}, tokeniser={args.tokeniser}')

    train_files, val_files = [], []
    for trad in traditions:
        train_files.extend(get_split_files(trad, 'train', split_dir))
        val_files.extend(  get_split_files(trad, 'val',   split_dir))

    print(f'  {len(train_files)} train files | {len(val_files)} val files')

    random.shuffle(train_files)
    random.shuffle(val_files)

    # ── Build datasets ────────────────────────────────────────────────────────
    print('Tokenising and chunking (this may take a while on first run)…')
    tok_type = 'remi' if args.mode == 'pretrain' else args.tokeniser

    train_ds = MIDIChunkDataset(
        train_files, tok, tok_type,
        tradition=(args.tradition if args.mode == 'finetune' else 'western_classical'),
        chunk_size=CHUNK_SIZE,
        stride=CHUNK_SIZE // 2,   # 50% overlap for denser training signal
    )
    val_ds = MIDIChunkDataset(
        val_files, tok, tok_type,
        tradition=(args.tradition if args.mode == 'finetune' else 'western_classical'),
        chunk_size=CHUNK_SIZE,
    )

    print(f'  {len(train_ds)} train chunks | {len(val_ds)} val chunks')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.num_workers
    )

    # ── Build or load model ───────────────────────────────────────────────────
    if args.mode == 'pretrain' or args.pretrain_dir is None:
        cfg   = MusicTransformerConfig(
            vocab_size  = vocab_size,
            d_model     = args.d_model,
            n_heads     = args.n_heads,
            n_layers    = args.n_layers,
            d_ff        = args.d_ff,
            max_seq_len = CHUNK_SIZE,
            dropout     = args.dropout,
        )
        model = MusicTransformer(cfg).to(device)
        print(f'  New model: {model.n_params:,} params')
    else:
        model = MusicTransformer.load(args.pretrain_dir, map_location='cpu')
        if is_ec_remi and model.cfg.vocab_size != EC_REMI_VOCAB_SIZE:
            model.resize_vocab(EC_REMI_VOCAB_SIZE)
            print(f'  Expanded vocab from {REMI_VOCAB_SIZE} → {EC_REMI_VOCAB_SIZE}')
        model = model.to(device)
        print(f'  Loaded pre-trained model from {args.pretrain_dir} ({model.n_params:,} params)')

    # ── Optimiser & schedule ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = min(args.warmup_steps, total_steps // 10)

    # ── Gradient scaler (AMP for CUDA) ────────────────────────────────────────
    use_amp  = (device.type == 'cuda')
    scaler   = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    global_step   = 0
    history       = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)

            # Update learning rate
            lr = get_lr(global_step, warmup_steps, total_steps, args.max_lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model(x, targets=y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            global_step += 1

            if step % args.log_every == 0:
                avg = epoch_loss / step
                print(f'  epoch {epoch:02d} step {step:04d}/{steps_per_epoch}  '
                      f'loss={avg:.4f}  lr={lr:.2e}')

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    _, loss = model(x, targets=y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        train_loss = epoch_loss / steps_per_epoch
        elapsed    = time.time() - epoch_start

        print(f'Epoch {epoch:02d}/{args.epochs}  '
              f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
              f'({elapsed:.0f}s)')

        entry = {
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'lr': lr
        }
        history.append(entry)

        # Save latest checkpoint
        model.save(output_dir / 'latest')

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(output_dir / 'best')
            print(f'  ✓ New best val_loss={best_val_loss:.4f} — saved to {output_dir}/best')

    # ── Save training history ─────────────────────────────────────────────────
    with open(output_dir / 'train_history.json', 'w') as f:
        json.dump({
            'args': vars(args),
            'best_val_loss': best_val_loss,
            'history': history,
        }, f, indent=2)

    print(f'\nDone. Best val_loss={best_val_loss:.4f}')
    print(f'Best checkpoint: {output_dir}/best')


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Music Transformer trainer')

    # Mode
    p.add_argument('--mode', choices=['pretrain', 'finetune'], required=True)
    p.add_argument('--tradition', choices=TRADITIONS, default='western_classical')
    p.add_argument('--tokeniser', choices=['remi', 'ec_remi'], default='remi')

    # Paths
    p.add_argument('--output_dir',     default='music_transformer/checkpoints/pretrain')
    p.add_argument('--pretrain_dir',   default=None,
                   help='Pre-trained checkpoint for finetune mode')
    p.add_argument('--split_cache_dir', default='music_transformer/splits')

    # Architecture (ignored in finetune mode when loading from pretrain_dir)
    p.add_argument('--d_model',  type=int,   default=256)
    p.add_argument('--n_heads',  type=int,   default=4)
    p.add_argument('--n_layers', type=int,   default=4)
    p.add_argument('--d_ff',     type=int,   default=1024)
    p.add_argument('--dropout',  type=float, default=0.1)

    # Training
    p.add_argument('--epochs',       type=int,   default=30)
    p.add_argument('--batch_size',   type=int,   default=16)
    p.add_argument('--max_lr',       type=float, default=3e-4)
    p.add_argument('--min_lr',       type=float, default=3e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip',    type=float, default=1.0)
    p.add_argument('--warmup_steps', type=int,   default=500)
    p.add_argument('--log_every',    type=int,   default=50)
    p.add_argument('--num_workers',  type=int,   default=2)

    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
