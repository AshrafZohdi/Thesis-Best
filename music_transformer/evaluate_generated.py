"""
Objective evaluation of generated MIDI samples.

Metrics per model:
  pitch_entropy     — Shannon entropy of pitch-class histogram (0–12 bits, higher = more variety)
  note_density      — notes per second (musical activity)
  pitch_range       — max minus min MIDI pitch (melodic span)
  unique_bigram_ratio — unique consecutive pitch pairs / total pairs (melodic diversity)
  ioi_entropy       — entropy of inter-onset intervals binned to 50ms (rhythmic variety)

Usage:
  python music_transformer/evaluate_generated.py
"""

from __future__ import annotations
import math
from collections import Counter
from pathlib import Path

import pretty_midi
import numpy as np
import pandas as pd

GEN_DIR = Path(__file__).parent / "generated" / "generated_500tokens"

TRADITION_ORDER = [
    "western_classical",
    "irish_folk",
    "turkish_makam",
    "hindustani",
    "carnatic",
]
TOK_ORDER = ["remi", "ec_remi"]


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def analyse_midi(path: Path) -> dict | None:
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return None

    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)

    if len(notes) < 4:
        return None

    notes.sort(key=lambda n: n.start)
    pitches = [n.pitch for n in notes]
    onsets  = [n.start for n in notes]
    duration = max(n.end for n in notes) - min(n.start for n in notes)
    if duration <= 0:
        return None

    # Pitch-class entropy (12 classes)
    pc_counts = Counter(p % 12 for p in pitches)
    pc_entropy = _entropy(pc_counts)

    # Note density
    note_density = len(notes) / duration

    # Pitch range
    pitch_range = max(pitches) - min(pitches)

    # Unique pitch bigram ratio
    bigrams = list(zip(pitches[:-1], pitches[1:]))
    unique_bigram_ratio = len(set(bigrams)) / len(bigrams) if bigrams else 0.0

    # IOI entropy (50 ms bins)
    iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]
    ioi_bins = Counter(round(ioi / 0.05) for ioi in iois if ioi > 0)
    ioi_entropy = _entropy(ioi_bins)

    return dict(
        pitch_entropy=pc_entropy,
        note_density=note_density,
        pitch_range=pitch_range,
        unique_bigram_ratio=unique_bigram_ratio,
        ioi_entropy=ioi_entropy,
        n_notes=len(notes),
    )


def evaluate_folder(folder: Path) -> dict | None:
    midi_files = sorted(folder.glob("*.mid"))
    if not midi_files:
        return None
    results = [analyse_midi(f) for f in midi_files]
    results = [r for r in results if r is not None]
    if not results:
        return None
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


def main():
    rows = []
    for folder in sorted(GEN_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        name = folder.name
        # Parse tradition and tokenizer from folder name
        if name.endswith("_ec_remi"):
            tradition = name[: -len("_ec_remi")]
            tokenizer = "ec_remi"
        elif name.endswith("_remi"):
            tradition = name[: -len("_remi")]
            tokenizer = "remi"
        else:
            tradition, tokenizer = name, "unknown"

        stats = evaluate_folder(folder)
        if stats is None:
            print(f"  [skip] {name} — no valid MIDIs")
            continue

        rows.append(dict(tradition=tradition, tokenizer=tokenizer, **stats))
        print(f"  {name}: pc_entropy={stats['pitch_entropy']:.3f}  "
              f"density={stats['note_density']:.2f}n/s  "
              f"range={stats['pitch_range']:.1f}  "
              f"bigram={stats['unique_bigram_ratio']:.3f}  "
              f"ioi_h={stats['ioi_entropy']:.3f}  "
              f"n={int(stats['n_notes'])}")

    df = pd.DataFrame(rows)

    # Sort by tradition then tokenizer for readability
    trad_order = {t: i for i, t in enumerate(TRADITION_ORDER)}
    tok_order  = {t: i for i, t in enumerate(TOK_ORDER)}
    df["_to"] = df["tradition"].map(lambda t: trad_order.get(t, 99))
    df["_ko"] = df["tokenizer"].map(lambda t: tok_order.get(t, 99))
    df = df.sort_values(["_to", "_ko"]).drop(columns=["_to", "_ko"])

    out_csv = GEN_DIR.parent / "evaluation_results.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nSaved to {out_csv}")

    # REMI vs EC-REMI comparison per tradition
    print("\n── REMI vs EC-REMI delta (EC − REMI, positive = EC better) ──")
    metrics = ["pitch_entropy", "note_density", "pitch_range", "unique_bigram_ratio", "ioi_entropy"]
    for trad in TRADITION_ORDER:
        sub = df[df["tradition"] == trad]
        remi_row   = sub[sub["tokenizer"] == "remi"]
        ecremi_row = sub[sub["tokenizer"] == "ec_remi"]
        if remi_row.empty or ecremi_row.empty:
            continue
        r = remi_row.iloc[0]
        e = ecremi_row.iloc[0]
        deltas = {m: e[m] - r[m] for m in metrics}
        delta_str = "  ".join(f"{m.split('_')[0]}={v:+.3f}" for m, v in deltas.items())
        print(f"  {trad:<20} {delta_str}")


if __name__ == "__main__":
    main()
