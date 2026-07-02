"""
Generator + executor for Step 4: Baseline REMI Tokenisation notebook.
Run: python create_remi_notebook.py
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

# ---------------------------------------------------------------------------
# PATH SETUP (injected into every notebook)
# ---------------------------------------------------------------------------
PATH_SETUP = """\
import sys
from pathlib import Path

_here = Path().resolve()
PROJECT_ROOT = _here
for _ in range(5):
    if (PROJECT_ROOT / "PROGRESS.md").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
print("PROJECT_ROOT:", PROJECT_ROOT)
"""

# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------
cells = []

cells.append(new_markdown_cell("""\
# Step 4 — Baseline REMI Tokenisation
## RMCE Thesis: Bridging the Authenticity Gap in AI-Generated Music

**Purpose:** Tokenise all five musical traditions using standard REMI (via MiDiTok).
Establish baseline token statistics — sequence length, vocabulary coverage, pitch-class
distribution — that serve as the quantitative reference against which EC-REMI is compared
in Step 5.

**Reference:** Huang & Yang (2020) REMI; Yang & Lerch (2020) objective MIR metrics.
"""))

cells.append(new_code_cell(PATH_SETUP))

cells.append(new_code_cell("""\
import warnings, collections, sys
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# miditok internally imports `tokenizers` (HuggingFace).
# Temporarily remove src/ from sys.path to avoid shadowing it with src/tokenizers/.
_src_paths = [p for p in sys.path if p.endswith("/src")]
for p in _src_paths:
    sys.path.remove(p)

from miditok import REMI, TokenizerConfig
from symusic import Score

for p in _src_paths:
    sys.path.insert(0, p)

from utils.midi_utils import analyse_midi

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("Imports OK")
"""))

cells.append(new_markdown_cell("## 1. Tradition Configuration"))

cells.append(new_code_cell("""\
TRADITIONS = {
    "western_classical": {"label": "Western Classical", "color": "#2166ac"},
    "hindustani":        {"label": "Hindustani",        "color": "#d73027"},
    "carnatic":          {"label": "Carnatic",           "color": "#fc8d59"},
    "irish_folk":        {"label": "Irish Folk",         "color": "#1a9850"},
    "turkish_makam":     {"label": "Turkish Makam",      "color": "#762a83"},
}
MIDI_ROOT = PROJECT_ROOT / "data" / "processed"

# Verify all traditions present
for key, info in TRADITIONS.items():
    midi_dir = MIDI_ROOT / key / "midi"
    files = sorted(list(midi_dir.glob("*.midi")) + list(midi_dir.glob("*.mid")))
    status = f"{len(files)} files" if files else "⚠ NO FILES"
    print(f"  {info['label']:22s} {status}")
"""))

cells.append(new_markdown_cell("## 2. REMI Tokeniser Initialisation"))

cells.append(new_code_cell("""\
# Use MiDiTok default REMI config — covers MIDI pitch range 21-108, beat_res {(0,4):8,(4,12):4},
# 32 velocities.  Default is well-tested and matches the original REMI paper.
tokenizer = REMI(TokenizerConfig())

vocab      = tokenizer.vocab
vocab_size = len(vocab)
pitch_tokens = sorted([k for k in vocab if k.startswith("Pitch_")])
pitch_values = [int(k.split("_")[1]) for k in pitch_tokens]

print(f"REMI vocabulary size : {vocab_size}")
print(f"Pitch tokens         : {len(pitch_tokens)}  (MIDI {min(pitch_values)}–{max(pitch_values)})")
print(f"Token types in vocab : {sorted(set(k.split('_')[0] for k in vocab))}")
"""))

cells.append(new_markdown_cell("## 3. Tokenise All Traditions"))

cells.append(new_code_cell("""\
records = []
type_counter = defaultdict(Counter)   # tradition_key -> token_type -> count
pitch_usage  = defaultdict(Counter)   # tradition_key -> pitch_midi -> count

for trad_key, trad_info in TRADITIONS.items():
    midi_dir = MIDI_ROOT / trad_key / "midi"
    files = sorted(list(midi_dir.glob("*.midi")) + list(midi_dir.glob("*.mid")))
    if not files:
        print(f"⚠  {trad_info['label']}: no MIDI files — skipped")
        continue

    ok, err = 0, 0
    for fpath in files:
        try:
            score  = Score(str(fpath))
            seqs   = tokenizer.encode(score)
            if not seqs:
                continue
            seq    = seqs[0]
            ids    = seq.ids
            tokens = seq.tokens

            tc = Counter(t.split("_")[0] for t in tokens)
            for k, v in tc.items():
                type_counter[trad_key][k] += v

            for t in tokens:
                if t.startswith("Pitch_"):
                    pitch_usage[trad_key][int(t.split("_")[1])] += 1

            records.append({
                "tradition"       : trad_key,
                "label"           : trad_info["label"],
                "file"            : fpath.name,
                "seq_len"         : len(ids),
                "n_pitch_tokens"  : tc.get("Pitch", 0),
                "n_velocity_tokens": tc.get("Velocity", 0),
                "n_duration_tokens": tc.get("Duration", 0),
                "n_position_tokens": tc.get("Position", 0),
                "n_bar_tokens"    : tc.get("Bar", 0),
                "unique_pitches"  : len(set(t for t in tokens if t.startswith("Pitch_"))),
                "unique_token_ids": len(set(ids)),
            })
            ok += 1
        except Exception as e:
            err += 1

    print(f"{trad_info['label']:22s}: {ok} tokenised, {err} errors")

df = pd.DataFrame(records)
print(f"\\nTotal files tokenised: {len(df)}")
"""))

cells.append(new_markdown_cell("## 4. Per-Tradition Token Statistics"))

cells.append(new_code_cell("""\
stats = (
    df.groupby("label")
    .agg(
        n_files          =("file",            "count"),
        mean_seq_len     =("seq_len",         "mean"),
        median_seq_len   =("seq_len",         "median"),
        std_seq_len      =("seq_len",         "std"),
        mean_pitch_tokens=("n_pitch_tokens",  "mean"),
        mean_bars        =("n_bar_tokens",     "mean"),
        mean_unique_pitch=("unique_pitches",  "mean"),
        vocab_coverage_pct=("unique_token_ids","mean"),
    )
    .round(1)
    .reset_index()
)

# Convert vocab_coverage to percentage of full vocab
stats["vocab_coverage_pct"] = (stats["vocab_coverage_pct"] / vocab_size * 100).round(1)

order = [TRADITIONS[k]["label"] for k in TRADITIONS]
stats["label"] = pd.Categorical(stats["label"], categories=order, ordered=True)
stats = stats.sort_values("label")

print(stats.to_string(index=False))
"""))

cells.append(new_markdown_cell("## 5. Pitch-Class Coverage Analysis\n\n*Which of the 88 REMI pitch tokens does each tradition actually use?  This directly motivates EC-REMI: REMI can only represent 12-TET pitches 21–108; any microtonal content in Turkish/Indian traditions is already lost at the transcription→tokenisation step.*"))

cells.append(new_code_cell("""\
# Build pitch coverage table: fraction of 88 piano pitches used per tradition
PIANO_PITCHES = set(range(21, 109))  # full REMI pitch range
coverage_rows = []
for trad_key, trad_info in TRADITIONS.items():
    if trad_key not in pitch_usage:
        continue
    used = set(pitch_usage[trad_key].keys()) & PIANO_PITCHES
    total_pitch_events = sum(pitch_usage[trad_key].values())
    coverage_rows.append({
        "label"              : trad_info["label"],
        "pitches_used"       : len(used),
        "pitch_coverage_pct" : round(len(used) / len(PIANO_PITCHES) * 100, 1),
        "total_pitch_events" : total_pitch_events,
        "min_pitch"          : min(used) if used else None,
        "max_pitch"          : max(used) if used else None,
        "pitch_span"         : (max(used) - min(used)) if used else 0,
    })

cov_df = pd.DataFrame(coverage_rows)
order = [TRADITIONS[k]["label"] for k in TRADITIONS if k in pitch_usage]
cov_df["label"] = pd.Categorical(cov_df["label"], categories=order, ordered=True)
cov_df = cov_df.sort_values("label")
print(cov_df.to_string(index=False))
"""))

cells.append(new_markdown_cell("## 6. Pitch-Class Entropy from Token Stream\n\n*Recompute PC entropy directly from REMI Pitch tokens — confirms alignment with the MIDI-level metric computed in EDA (Step 3).*"))

cells.append(new_code_cell("""\
entropy_rows = []
for trad_key, trad_info in TRADITIONS.items():
    if trad_key not in pitch_usage:
        continue
    hist = np.zeros(12, dtype=float)
    for pitch_midi, count in pitch_usage[trad_key].items():
        hist[pitch_midi % 12] += count
    if hist.sum() > 0:
        hist /= hist.sum()
    nonzero = hist[hist > 0]
    h = float(-np.sum(nonzero * np.log2(nonzero))) if len(nonzero) > 0 else 0.0
    entropy_rows.append({"label": trad_info["label"], "remi_pc_entropy": round(h, 4)})

ent_df = pd.DataFrame(entropy_rows)
print(ent_df.to_string(index=False))
print("\\n(Compare with EDA PC entropy: Western 3.356 | Turkish 2.719 | Irish 2.542)")
"""))

cells.append(new_markdown_cell("## 7. Visualisations"))

cells.append(new_code_cell("""\
# --- 7a: Sequence length distributions (box plot per tradition) ---
trad_order  = [TRADITIONS[k]["label"] for k in TRADITIONS if k in df["tradition"].values]
trad_colors = [TRADITIONS[k]["color"]  for k in TRADITIONS if k in df["tradition"].values]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("REMI Tokenisation — Sequence Length and Pitch Coverage", fontsize=13, fontweight="bold")

# Box plot: sequence lengths
box_data  = [df[df["label"] == lab]["seq_len"].values for lab in trad_order]
bp = axes[0].boxplot(box_data, patch_artist=True, medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], trad_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_xticklabels([lab.replace(" ", "\\n") for lab in trad_order], fontsize=8)
axes[0].set_ylabel("Token sequence length")
axes[0].set_title("REMI Sequence Length Distribution")
axes[0].set_yscale("log")
axes[0].grid(axis="y", alpha=0.3)

# Bar chart: pitch coverage %
bar_labels  = cov_df["label"].tolist()
bar_colors2 = [TRADITIONS[k]["color"] for k in TRADITIONS if k in pitch_usage]
bars = axes[1].bar(range(len(bar_labels)), cov_df["pitch_coverage_pct"], color=bar_colors2, alpha=0.8)
axes[1].set_xticks(range(len(bar_labels)))
axes[1].set_xticklabels([lab.replace(" ", "\\n") for lab in bar_labels], fontsize=8)
axes[1].set_ylabel("% of 88 REMI pitch tokens used")
axes[1].set_title("Pitch Token Coverage (MIDI 21–108)")
axes[1].set_ylim(0, 100)
axes[1].grid(axis="y", alpha=0.3)
for bar, val in zip(bars, cov_df["pitch_coverage_pct"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "remi_seqlen_pitch_coverage.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: remi_seqlen_pitch_coverage.png")
"""))

cells.append(new_code_cell("""\
# --- 7b: Token type composition stacked bar per tradition ---
type_order = ["Pitch", "Velocity", "Duration", "Position", "Bar"]
type_colors_map = {
    "Pitch":    "#e41a1c",
    "Velocity": "#377eb8",
    "Duration": "#4daf4a",
    "Position": "#ff7f00",
    "Bar":      "#984ea3",
}

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(trad_order))
bottom = np.zeros(len(trad_order))

for tok_type in type_order:
    vals = []
    for trad_key in [k for k in TRADITIONS if k in type_counter]:
        total = sum(type_counter[trad_key].values()) or 1
        vals.append(type_counter[trad_key].get(tok_type, 0) / total * 100)
    bars = ax.bar(x, vals, bottom=bottom, label=tok_type,
                  color=type_colors_map[tok_type], alpha=0.85)
    bottom += np.array(vals)

ax.set_xticks(x)
ax.set_xticklabels([lab.replace(" ", "\\n") for lab in trad_order], fontsize=9)
ax.set_ylabel("% of total tokens")
ax.set_title("REMI Token Type Composition by Tradition", fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "remi_token_type_composition.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: remi_token_type_composition.png")
"""))

cells.append(new_code_cell("""\
# --- 7c: Pitch usage heatmap (pitch vs tradition) ---
all_traditions_with_data = [k for k in TRADITIONS if k in pitch_usage]
mat = np.zeros((len(all_traditions_with_data), 88))  # 88 piano pitches

for i, trad_key in enumerate(all_traditions_with_data):
    total = sum(pitch_usage[trad_key].values()) or 1
    for pitch_midi, cnt in pitch_usage[trad_key].items():
        if 21 <= pitch_midi <= 108:
            mat[i, pitch_midi - 21] = cnt / total

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax.set_yticks(range(len(all_traditions_with_data)))
ax.set_yticklabels([TRADITIONS[k]["label"] for k in all_traditions_with_data], fontsize=9)

# X-axis: octave labels at C notes
c_positions = [i for i in range(88) if (i + 21) % 12 == 0]
c_labels    = [f"C{((i+21)//12)-1}" for i in c_positions]
ax.set_xticks(c_positions)
ax.set_xticklabels(c_labels, fontsize=8)
ax.set_xlabel("MIDI Pitch (C notes labelled)")
ax.set_title("Pitch Usage Density by Tradition (REMI tokens)", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, label="Relative frequency")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "remi_pitch_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: remi_pitch_heatmap.png")
"""))

cells.append(new_markdown_cell("## 8. Cross-Tradition Summary Table"))

cells.append(new_code_cell("""\
# Merge stats + coverage + entropy into one master table
summary = stats[["label","n_files","mean_seq_len","median_seq_len","mean_unique_pitch",
                  "vocab_coverage_pct"]].copy()
summary = summary.merge(cov_df[["label","pitch_coverage_pct","pitch_span"]], on="label", how="left")
summary = summary.merge(ent_df, on="label", how="left")
summary.columns = [
    "Tradition", "N", "Mean seq len", "Median seq len",
    "Mean unique pitches", "Vocab coverage %",
    "Pitch coverage %", "Pitch span (st)", "REMI PC entropy"
]
print(summary.to_string(index=False))
"""))

cells.append(new_markdown_cell("""\
## 9. Key Findings for Thesis

**What REMI captures well:**
- Western Classical (piano): full pitch range, high sequence length, vocabulary well-utilised
- Irish Folk: compact sequences, narrow pitch range consistent with melodic instrument

**What REMI loses — motivation for EC-REMI:**
1. **Turkish Makam:** SymbTr Koma53 encodes 53-TET deviations; REMI rounds to nearest semitone.
   All microtonal inflection (±0–22 cents per note) is discarded.
2. **Hindustani / Carnatic:** Gamakas, meends, and shruti deviations are mapped to nearest
   12-TET pitch by Basic-Pitch. REMI has no token for "microtonal bend" or "ornament type".
3. **Irish Folk:** Rolls, cuts, and slides appear as rapid note sequences; REMI tokenises
   them identically to regular notes with no semantic ornament token.
4. **Modal / scale identity:** REMI has no Raga, Makam, or Mode token.
   EC-REMI Step 5 adds all four missing categories.
"""))

cells.append(new_code_cell("""\
# Save master CSV
out_path = RESULTS_DIR / "remi_tokenisation_stats.csv"
summary.to_csv(out_path, index=False)
print(f"Saved: {out_path}")

# Also save raw per-file stats
raw_path = RESULTS_DIR / "remi_per_file_stats.csv"
df.to_csv(raw_path, index=False)
print(f"Saved: {raw_path}")

print("\\n=== Step 4 Complete ===")
print(f"Charts  : {RESULTS_DIR}/remi_*.png  (3 files)")
print(f"Tables  : remi_tokenisation_stats.csv, remi_per_file_stats.csv")
"""))

# ---------------------------------------------------------------------------
# Assemble and write notebook
# ---------------------------------------------------------------------------
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.9.0"}

out_path = Path("notebooks/02_baseline_remi/02_baseline_remi_tokenisation.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
import nbformat
nbformat.write(nb, str(out_path))
print(f"Written: {out_path}  ({len(cells)} cells)")
