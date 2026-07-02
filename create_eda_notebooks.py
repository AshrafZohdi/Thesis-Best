#!/usr/bin/env python3
"""
Generate all EDA notebooks for the RMCE thesis.
Run once from the project root: python3 create_eda_notebooks.py
"""

import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
NB_DIR = PROJECT_ROOT / "notebooks" / "01_eda"


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def make_nb(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.9"}
    return nb


PATH_SETUP = """
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

_here = Path().resolve()
PROJECT_ROOT = _here
for _ in range(5):
    if (PROJECT_ROOT / "PROGRESS.md").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.midi_utils import (
    load_midi, analyse_midi, get_pitch_class_histogram,
    pitch_class_entropy, piano_roll_plot
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PC_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
print(f"Project root: {PROJECT_ROOT}")
"""

STATS_HELPER = """
def build_stats_df(midi_dir, meta_df=None, id_col=None):
    \"\"\"Analyse all MIDI files in midi_dir; optionally merge metadata.\"\"\"
    midi_files = sorted(midi_dir.glob("*.mid")) + sorted(midi_dir.glob("*.midi"))
    print(f"Analysing {len(midi_files)} MIDI files ...")
    records = [analyse_midi(p) for p in midi_files]
    df = pd.DataFrame(records)
    if "error" in df.columns:
        bad = df["error"].notna().sum()
        if bad:
            print(f"  Warning: {bad} files failed to load.")
        df = df[df["error"].isna()].drop(columns=["error"])
    df["filename"] = [Path(p).name for p in df["path"]]
    if meta_df is not None and id_col is not None:
        df = df.merge(meta_df, left_on="filename", right_on=id_col, how="left")
    return df
"""

PLOT_HELPER = """
def summary_panel(df, tradition_name, save_path):
    \"\"\"4-panel summary figure: duration, note density, pitch range, PC entropy.\"\"\"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{tradition_name} — EDA Summary (n={len(df)})", fontsize=13, y=1.01)

    # Duration
    ax = axes[0, 0]
    df["duration_s"].div(60).plot.hist(bins=25, ax=ax, color="steelblue", edgecolor="white")
    ax.axvline(df["duration_s"].mean()/60, color="red", linestyle="--", label=f"mean={df['duration_s'].mean()/60:.1f} min")
    ax.set_xlabel("Duration (minutes)")
    ax.set_title("Duration Distribution")
    ax.legend(fontsize=8)

    # Note density
    ax = axes[0, 1]
    df["note_density"].plot.hist(bins=25, ax=ax, color="seagreen", edgecolor="white")
    ax.axvline(df["note_density"].mean(), color="red", linestyle="--", label=f"mean={df['note_density'].mean():.2f}")
    ax.set_xlabel("Notes per second")
    ax.set_title("Note Density Distribution")
    ax.legend(fontsize=8)

    # Pitch range
    ax = axes[1, 0]
    df["pitch_range"].plot.hist(bins=25, ax=ax, color="darkorange", edgecolor="white")
    ax.axvline(df["pitch_range"].mean(), color="red", linestyle="--", label=f"mean={df['pitch_range'].mean():.1f}")
    ax.set_xlabel("Pitch range (semitones)")
    ax.set_title("Pitch Range Distribution")
    ax.legend(fontsize=8)

    # PC entropy
    ax = axes[1, 1]
    df["pc_entropy"].plot.hist(bins=25, ax=ax, color="mediumpurple", edgecolor="white")
    ax.axvline(df["pc_entropy"].mean(), color="red", linestyle="--", label=f"mean={df['pc_entropy'].mean():.3f}")
    ax.set_xlabel("Pitch Class Entropy (bits)")
    ax.set_title("PC Entropy Distribution\\n(Yang & Lerch, 2020)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path.relative_to(PROJECT_ROOT)}")


def mean_pc_histogram_plot(df_midi_paths, tradition_name, save_path):
    \"\"\"Plot the mean pitch class histogram across all pieces.\"\"\"
    hists = []
    for p in df_midi_paths:
        pm = load_midi(p)
        if pm:
            hists.append(get_pitch_class_histogram(pm))
    if not hists:
        return
    mean_hist = np.mean(hists, axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(PC_LABELS, mean_hist, color="steelblue", edgecolor="white")
    ax.set_xlabel("Pitch class")
    ax.set_ylabel("Mean relative frequency")
    ax.set_title(f"{tradition_name} — Mean Pitch Class Histogram (n={len(hists)})")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path.relative_to(PROJECT_ROOT)}")
"""

# ============================================================
# 01a — MAESTRO EDA
# ============================================================

maestro_eda = [
    md("""# 01a — MAESTRO Western Classical: Exploratory Data Analysis

This notebook characterises the processed MAESTRO subset (150 files) across five
objective dimensions:

1. Duration distribution
2. Note density (notes/sec)
3. Pitch range
4. Pitch Class (PC) Entropy — primary metric from Yang & Lerch (2020)
5. Mean pitch class histogram

Results are saved to `results/` for use in the cross-tradition comparison (Step 7).
"""),
    code(PATH_SETUP),
    code(STATS_HELPER),
    code(PLOT_HELPER),
    code("""
MIDI_DIR = PROJECT_ROOT / "data" / "processed" / "western_classical" / "midi"
META_CSV = PROJECT_ROOT / "data" / "metadata" / "maestro_selected.csv"

if not MIDI_DIR.exists() or not any(MIDI_DIR.iterdir()):
    raise FileNotFoundError(
        "No MIDI files found. Run notebook 00a_maestro_prep.ipynb first."
    )

meta = pd.read_csv(META_CSV)
print(f"Metadata rows: {len(meta)}")
meta.head(3)
"""),
    md("## 1. Load and compute statistics"),
    code("""
stats = build_stats_df(MIDI_DIR)
print(f"Files analysed: {len(stats)}")
print("\\nDescriptive statistics:")
print(stats[["duration_s", "note_count", "note_density", "pitch_range", "pc_entropy"]].describe().round(3))
"""),
    md("## 2. Distribution plots"),
    code("""
summary_panel(stats, "Western Classical (MAESTRO)", RESULTS_DIR / "eda_maestro_summary.png")
"""),
    md("## 3. Mean pitch class histogram"),
    code("""
mean_pc_histogram_plot(stats["path"].tolist(), "Western Classical (MAESTRO)",
                       RESULTS_DIR / "eda_maestro_pc_histogram.png")
"""),
    md("## 4. Composer and year breakdown"),
    code("""
# Merge with metadata for labelled analysis
stats_meta = stats.copy()
stats_meta["processed_filename"] = stats_meta["filename"]
stats_meta = stats_meta.merge(
    meta[["processed_filename", "canonical_composer", "year", "duration"]],
    on="processed_filename", how="left"
)

print("Top 10 composers by file count:")
print(stats_meta["canonical_composer"].value_counts().head(10).to_string())
print("\\nMean PC entropy by year:")
print(stats_meta.groupby("year")["pc_entropy"].mean().round(4).to_string())
"""),
    md("## 5. Sample piano roll"),
    code("""
sample_path = stats["path"].iloc[0]
pm = load_midi(sample_path)
sample_name = Path(sample_path).stem[:60]

fig, ax = plt.subplots(figsize=(14, 4))
piano_roll_plot(pm, ax, time_start=30, time_end=60,
                title=f"Piano roll sample — {sample_name}")
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "eda_maestro_piano_roll.png"), dpi=150)
plt.show()
"""),
    md("## 6. Save summary statistics"),
    code("""
stats["tradition"] = "western_classical"
stats.to_csv(RESULTS_DIR / "eda_maestro_stats.csv", index=False)
print("Saved → results/eda_maestro_stats.csv")

print("\\n=== MAESTRO EDA Summary ===")
print(f"Files          : {len(stats)}")
print(f"Duration       : {stats['duration_s'].mean()/60:.1f} min mean  "
      f"({stats['duration_s'].min()/60:.1f} – {stats['duration_s'].max()/60:.1f})")
print(f"Note density   : {stats['note_density'].mean():.2f} notes/s mean")
print(f"Pitch range    : {stats['pitch_range'].mean():.1f} semitones mean")
print(f"PC entropy     : {stats['pc_entropy'].mean():.3f} bits mean")
"""),
]

# ============================================================
# 01b — Hindustani EDA
# ============================================================

hindustani_eda = [
    md("""# 01b — Saraga Hindustani: Exploratory Data Analysis

This notebook characterises the processed Hindustani subset (60 Basic-Pitch
transcribed MIDI files). It covers the same five dimensions as 01a plus
tradition-specific analysis: raga and taal label distributions.

**Prerequisite:** Run `00b_hindustani_prep.ipynb` first (overnight transcription job).
A check cell below will raise a clear error if the MIDI files are not yet present.
"""),
    code(PATH_SETUP),
    code(STATS_HELPER),
    code(PLOT_HELPER),
    code("""
MIDI_DIR = PROJECT_ROOT / "data" / "processed" / "hindustani" / "midi"
META_CSV = PROJECT_ROOT / "data" / "metadata" / "hindustani_tracks.csv"

midi_files = list(MIDI_DIR.glob("*.mid"))
if not midi_files:
    raise FileNotFoundError(
        "No MIDI files found in data/processed/hindustani/midi/. "
        "Run notebook 00b_hindustani_prep.ipynb first (overnight job)."
    )
print(f"Found {len(midi_files)} MIDI files.")

meta = pd.read_csv(META_CSV)
print(f"Metadata rows: {len(meta)}")
meta[["title", "raga", "taal", "metadata_source"]].head(5)
"""),
    md("## 1. Metadata quality"),
    code("""
print("Metadata source breakdown:")
print(meta["metadata_source"].value_counts().to_string())
print(f"\\nUnique ragas : {meta['raga'].nunique()}")
print(f"Unique taals : {meta['taal'].nunique()}")
print("\\nTop 15 ragas in selection:")
print(meta["raga"].value_counts().head(15).to_string())
"""),
    md("## 2. Compute statistics"),
    code("""
stats = build_stats_df(MIDI_DIR)
print(f"Files analysed: {len(stats)}")
print("\\nDescriptive statistics:")
print(stats[["duration_s", "note_count", "note_density", "pitch_range", "pc_entropy"]].describe().round(3))
"""),
    code("""
# Note: Hindustani recordings are long concerts; durations will be large.
print(f"Duration range: {stats['duration_s'].min()/60:.1f} – {stats['duration_s'].max()/60:.1f} minutes")
print(f"Mean duration : {stats['duration_s'].mean()/60:.1f} minutes")
"""),
    md("""## 3. Distribution plots

Note: High note density relative to Western Classical is expected — Basic-Pitch
transcribes all pitched sources in the mix (vocal + harmonium), producing denser
note sequences than single-instrument piano MIDI.
"""),
    code("""
summary_panel(stats, "Hindustani Classical (Saraga)", RESULTS_DIR / "eda_hindustani_summary.png")
"""),
    md("## 4. Mean pitch class histogram"),
    code("""
mean_pc_histogram_plot(stats["path"].tolist(), "Hindustani Classical (Saraga)",
                       RESULTS_DIR / "eda_hindustani_pc_histogram.png")
"""),
    md("""## 5. Raga label distribution

The raga distribution reflects the coverage in the Saraga Hindustani 1.5 dataset;
we have not artificially balanced raga counts — the distribution is as-recorded.
"""),
    code("""
meta_valid = meta[meta["midi_path"].notna()].copy() if "midi_path" in meta.columns else meta.copy()
raga_counts = meta_valid["raga"].value_counts()

fig, ax = plt.subplots(figsize=(12, 5))
raga_counts.head(20).plot(kind="bar", ax=ax, color="indianred", edgecolor="white")
ax.set_xlabel("Raga")
ax.set_ylabel("Number of tracks")
ax.set_title(f"Hindustani — Raga Distribution (top 20, n={len(meta_valid)})")
ax.tick_params(axis="x", rotation=45, labelsize=8)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "eda_hindustani_raga_dist.png"), dpi=150)
plt.show()
"""),
    md("## 6. Sample piano roll"),
    code("""
sample_path = stats["path"].iloc[0]
pm = load_midi(sample_path)
sample_name = Path(sample_path).stem[:60]

fig, ax = plt.subplots(figsize=(14, 4))
piano_roll_plot(pm, ax, time_start=60, time_end=90,
                title=f"Piano roll sample (60–90 s) — {sample_name}")
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "eda_hindustani_piano_roll.png"), dpi=150)
plt.show()
"""),
    md("## 7. Save summary statistics"),
    code("""
stats["tradition"] = "hindustani"
stats.to_csv(RESULTS_DIR / "eda_hindustani_stats.csv", index=False)
print("Saved → results/eda_hindustani_stats.csv")
print(f"\\nPC entropy mean : {stats['pc_entropy'].mean():.3f} bits")
print(f"Note density mean: {stats['note_density'].mean():.2f} notes/s")
"""),
]

# ============================================================
# 01c — Carnatic EDA
# ============================================================

carnatic_eda = [
    md("""# 01c — Saraga Carnatic: Exploratory Data Analysis

This notebook characterises the processed Carnatic subset (60 transcribed MIDI files).
Carnatic music features a distinct melodic grammar (gamaka — continuous pitch inflections,
oscillations) and rhythmic structure (tala cycles with subdivisions called laghu, drutam,
anudrutam) that standard REMI tokenisation cannot fully represent.

**Prerequisite:** Run `00c_carnatic_prep.ipynb` first.
"""),
    code(PATH_SETUP),
    code(STATS_HELPER),
    code(PLOT_HELPER),
    code("""
MIDI_DIR = PROJECT_ROOT / "data" / "processed" / "carnatic" / "midi"
META_CSV = PROJECT_ROOT / "data" / "metadata" / "carnatic_tracks.csv"

midi_files = list(MIDI_DIR.glob("*.mid"))
if not midi_files:
    raise FileNotFoundError(
        "No MIDI files in data/processed/carnatic/midi/. "
        "Run notebook 00c_carnatic_prep.ipynb first."
    )
print(f"Found {len(midi_files)} MIDI files.")

meta = pd.read_csv(META_CSV)
print(f"Metadata rows: {len(meta)}")
meta[["title", "raga", "taal", "metadata_source", "concert_folder"]].head(5)
"""),
    md("## 1. Metadata quality"),
    code("""
print("Metadata source breakdown:")
print(meta["metadata_source"].value_counts().to_string())
print(f"\\nUnique ragas (incl. transliterated): {meta['raga'].nunique()}")
print(f"Concert folders (performers)         : {meta['concert_folder'].nunique()}")
print("\\nNote: many raga names are Unicode transliterations from the JSON 'name' field.")
print("Where raaga array was empty, the piece title was used as raga proxy.")
"""),
    md("## 2. Compute statistics"),
    code("""
stats = build_stats_df(MIDI_DIR)
print(f"Files analysed: {len(stats)}")
print("\\nDescriptive statistics:")
print(stats[["duration_s", "note_count", "note_density", "pitch_range", "pc_entropy"]].describe().round(3))
"""),
    md("## 3. Distribution plots"),
    code("""
summary_panel(stats, "Carnatic Classical (Saraga)", RESULTS_DIR / "eda_carnatic_summary.png")
"""),
    md("## 4. Mean pitch class histogram"),
    code("""
mean_pc_histogram_plot(stats["path"].tolist(), "Carnatic Classical (Saraga)",
                       RESULTS_DIR / "eda_carnatic_pc_histogram.png")
"""),
    md("""## 5. Performer (concert) distribution

Carnatic tracks are drawn from 27 concert recordings. The distribution below shows
how many tracks each performer contributes to the selected subset.
"""),
    code("""
perf_counts = meta["concert_folder"].value_counts()
fig, ax = plt.subplots(figsize=(14, 5))
perf_counts.plot(kind="bar", ax=ax, color="teal", edgecolor="white")
ax.set_xlabel("Concert / Performer")
ax.set_ylabel("Tracks")
ax.set_title(f"Carnatic — Tracks per Performer (n={len(meta)})")
ax.tick_params(axis="x", rotation=60, labelsize=7)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "eda_carnatic_performer_dist.png"), dpi=150)
plt.show()
"""),
    md("## 6. Sample piano roll"),
    code("""
sample_path = stats["path"].iloc[0]
pm = load_midi(sample_path)
sample_name = Path(sample_path).stem[:60]

fig, ax = plt.subplots(figsize=(14, 4))
piano_roll_plot(pm, ax, time_start=30, time_end=60,
                title=f"Piano roll sample (30–60 s) — {sample_name}")
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "eda_carnatic_piano_roll.png"), dpi=150)
plt.show()
"""),
    md("## 7. Save summary statistics"),
    code("""
stats["tradition"] = "carnatic"
stats.to_csv(RESULTS_DIR / "eda_carnatic_stats.csv", index=False)
print("Saved → results/eda_carnatic_stats.csv")
print(f"\\nPC entropy mean : {stats['pc_entropy'].mean():.3f} bits")
print(f"Note density mean: {stats['note_density'].mean():.2f} notes/s")
"""),
]

# ============================================================
# 01d — Irish Folk EDA
# ============================================================

irish_eda = [
    md("""# 01d — TheSession Irish Folk: Exploratory Data Analysis

This notebook characterises the Irish folk subset (200 tunes converted from ABC notation).
Irish traditional music is primarily monophonic or heterophonic (melody instruments playing
in unison with ornamentation), which should produce distinctly different note density and
pitch range distributions compared to the polyphonic Western Classical subset.

Key EDA dimensions:
- Tune type distribution (reel, jig, hornpipe, etc.) and their rhythmic signatures
- Key/mode distribution (major, dorian, mixolydian — common modes in Irish trad)
- Duration and note density (short tunes — typically 16–32 bars, repeated)
- Pitch class entropy (expected lower than Western Classical due to modal focus)
"""),
    code(PATH_SETUP),
    code(STATS_HELPER),
    code(PLOT_HELPER),
    code("""
MIDI_DIR = PROJECT_ROOT / "data" / "processed" / "irish_folk" / "midi"
META_CSV = PROJECT_ROOT / "data" / "metadata" / "irish_folk_tunes.csv"

midi_files = list(MIDI_DIR.glob("*.mid"))
if not midi_files:
    raise FileNotFoundError(
        "No MIDI files found. Run notebook 00d_irish_folk_prep.ipynb first."
    )
print(f"Found {len(midi_files)} MIDI files.")

meta = pd.read_csv(META_CSV)
print(f"Metadata rows: {len(meta)}")
meta[["name", "type", "meter", "mode", "tunebooks"]].head(5)
"""),
    md("## 1. Tune type and mode distributions"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

type_counts = meta["type"].value_counts()
type_counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title(f"Tune Type Distribution (n={len(meta)})")
axes[0].set_xlabel("Type")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)

mode_counts = meta["mode"].value_counts().head(15)
mode_counts.plot(kind="bar", ax=axes[1], color="darkorange", edgecolor="white")
axes[1].set_title("Top 15 Key/Mode Distribution")
axes[1].set_xlabel("Mode")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=45, labelsize=8)

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "eda_irish_type_mode.png"), dpi=150)
plt.show()
"""),
    md("## 2. Compute statistics"),
    code("""
stats = build_stats_df(MIDI_DIR)
print(f"Files analysed: {len(stats)}")
print("\\nDescriptive statistics:")
print(stats[["duration_s", "note_count", "note_density", "pitch_range", "pc_entropy"]].describe().round(3))
"""),
    md("## 3. Distribution plots"),
    code("""
summary_panel(stats, "Irish Folk (TheSession)", RESULTS_DIR / "eda_irish_summary.png")
"""),
    md("## 4. Mean pitch class histogram"),
    code("""
mean_pc_histogram_plot(stats["path"].tolist(), "Irish Folk (TheSession)",
                       RESULTS_DIR / "eda_irish_pc_histogram.png")
"""),
    md("""## 5. PC entropy by tune type

We expect reels (4/4) and hornpipes to have similar pitch entropy, while
slipjigs (9/8) and slides (12/8) may differ due to their distinctive melodic patterns.
"""),
    code("""
# Map filename back to tune metadata for grouping
stats["tune_id"] = stats["filename"].str.extract(r"irish_\\d+_(\\d+)_").astype(float)
stats_meta = stats.merge(meta[["tune_id", "type", "mode"]], on="tune_id", how="left")

if stats_meta["type"].notna().any():
    entropy_by_type = stats_meta.groupby("type")["pc_entropy"].agg(["mean","std","count"])
    print("PC Entropy by tune type:")
    print(entropy_by_type.sort_values("mean", ascending=False).round(3).to_string())
"""),
    md("## 6. Sample piano roll"),
    code("""
sample_path = stats["path"].iloc[0]
pm = load_midi(sample_path)
sample_name = Path(sample_path).stem[:60]

fig, ax = plt.subplots(figsize=(14, 4))
piano_roll_plot(pm, ax, time_start=0, time_end=30,
                title=f"Piano roll — {sample_name}")
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "eda_irish_piano_roll.png"), dpi=150)
plt.show()
"""),
    md("## 7. Save summary statistics"),
    code("""
stats["tradition"] = "irish_folk"
stats.to_csv(RESULTS_DIR / "eda_irish_stats.csv", index=False)
print("Saved → results/eda_irish_stats.csv")
print(f"\\nPC entropy mean : {stats['pc_entropy'].mean():.3f} bits")
print(f"Note density mean: {stats['note_density'].mean():.2f} notes/s")
print(f"Duration mean    : {stats['duration_s'].mean():.1f} s  ({stats['duration_s'].mean()/60:.2f} min)")
"""),
]

# ============================================================
# 01e — Turkish Makam EDA
# ============================================================

turkish_eda = [
    md("""# 01e — SymbTr Turkish Makam: Exploratory Data Analysis

This notebook characterises the Turkish Makam subset (200 MIDI files from SymbTr).
Turkish Makam music uses a microtonal pitch system based on 53-tone equal temperament
(53-TET), where pitches can deviate from the 12-TET semitone grid by multiples of a
*comma* (~22.6 cents). This is the core musical feature that EC-REMI must encode.

This notebook covers:
1. Standard MIDI-level statistics (duration, note density, pitch range, PC entropy)
2. Makam, usul, and form distributions
3. A preview of the 53-TET Koma data from one SymbTr TXT file — the microtonal
   ground truth that will calibrate EC-REMI deviation tokens in Step 5.
"""),
    code(PATH_SETUP),
    code(STATS_HELPER),
    code(PLOT_HELPER),
    code("""
MIDI_DIR = PROJECT_ROOT / "data" / "processed" / "turkish_makam" / "midi"
META_CSV = PROJECT_ROOT / "data" / "metadata" / "symbtr_selected.csv"
TXT_DIR  = PROJECT_ROOT / "datasets" / "SymbTr" / "txt"

midi_files = list(MIDI_DIR.glob("*.mid"))
if not midi_files:
    raise FileNotFoundError(
        "No MIDI files found. Run notebook 00e_turkish_makam_prep.ipynb first."
    )
print(f"Found {len(midi_files)} MIDI files.")

meta = pd.read_csv(META_CSV)
print(f"Metadata rows: {len(meta)}")
meta[["makam", "form", "usul", "composer"]].head(5)
"""),
    md("## 1. Makam, form, and usul distributions"),
    code("""
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

meta["makam"].value_counts().head(20).plot(
    kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title("Top 20 Makams")
axes[0].tick_params(axis="x", rotation=60, labelsize=8)
axes[0].set_ylabel("Count")

meta["usul"].value_counts().head(15).plot(
    kind="bar", ax=axes[1], color="seagreen", edgecolor="white")
axes[1].set_title("Top 15 Usuls (rhythmic cycles)")
axes[1].tick_params(axis="x", rotation=60, labelsize=8)

meta["form"].value_counts().head(12).plot(
    kind="bar", ax=axes[2], color="darkorange", edgecolor="white")
axes[2].set_title("Top 12 Forms")
axes[2].tick_params(axis="x", rotation=60, labelsize=8)

plt.suptitle("SymbTr Turkish Makam — Metadata Distribution (n=200)", fontsize=12)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "eda_turkish_metadata_dist.png"), dpi=150)
plt.show()
"""),
    md("## 2. Compute statistics"),
    code("""
stats = build_stats_df(MIDI_DIR)
print(f"Files analysed: {len(stats)}")
print("\\nDescriptive statistics:")
print(stats[["duration_s", "note_count", "note_density", "pitch_range", "pc_entropy"]].describe().round(3))
"""),
    md("## 3. Distribution plots"),
    code("""
summary_panel(stats, "Turkish Makam (SymbTr)", RESULTS_DIR / "eda_turkish_summary.png")
"""),
    md("## 4. Mean pitch class histogram"),
    code("""
mean_pc_histogram_plot(stats["path"].tolist(), "Turkish Makam (SymbTr)",
                       RESULTS_DIR / "eda_turkish_pc_histogram.png")
"""),
    md("""## 5. PC entropy by makam (top 10)

Different makams have characteristic modal frameworks and may show distinct pitch
class distributions. High-entropy makams use more pitch classes; low-entropy ones
have a strong tonal focus.
"""),
    code("""
stats["filename"] = stats["filename"].astype(str)
# Extract makam from processed filename: symbtr_NNN_makam_form.mid
stats["makam"] = stats["filename"].str.extract(r"symbtr_\\d+_([^_]+)_")

entropy_by_makam = (
    stats.groupby("makam")["pc_entropy"]
    .agg(["mean", "std", "count"])
    .sort_values("mean", ascending=False)
)
print("PC Entropy by makam (top 15):")
print(entropy_by_makam.head(15).round(3).to_string())
"""),
    md("""## 6. Microtonal preview — 53-TET Koma data

The SymbTr TXT files encode pitch in 53-TET commas via the `Koma53` column.
This is the only tradition in this study with native microtonal notation in the
source dataset. The distribution of Koma53 values within a makam shows its
characteristic pitch inflections — which standard REMI (12-TET only) cannot represent,
and which EC-REMI deviation tokens are designed to encode.

This analysis directly motivates the EC-REMI design in Step 5.
"""),
    code("""
# Load Koma53 data from the original TXT files for the selected pieces
koma_records = []
for _, row in meta.head(50).iterrows():  # sample 50 to keep analysis fast
    # Original filename (before renaming)
    orig_name = row["filename"].replace(".mid", ".txt")
    txt_path  = TXT_DIR / orig_name
    if not txt_path.exists():
        continue
    try:
        txt_df = pd.read_csv(txt_path, sep="\\t", encoding="utf-8")
        # Keep only rows that are actual notes (Koma53 is numeric)
        txt_df = txt_df[pd.to_numeric(txt_df["Koma53"], errors="coerce").notna()].copy()
        txt_df["Koma53"] = pd.to_numeric(txt_df["Koma53"])
        txt_df["makam"]  = row["makam"]
        koma_records.append(txt_df[["Koma53", "NotaAE", "makam"]])
    except Exception:
        continue

if koma_records:
    koma_df = pd.concat(koma_records, ignore_index=True)
    print(f"Total note events with Koma53 data: {len(koma_df):,}")
    print(f"Koma53 range: {koma_df['Koma53'].min():.0f} – {koma_df['Koma53'].max():.0f}")
    print(f"Unique Koma53 values: {koma_df['Koma53'].nunique()}")
    print("\\nTop 15 pitch names (Western equivalent):")
    print(koma_df["NotaAE"].value_counts().head(15).to_string())
else:
    print("No TXT files found — check that the SymbTr TXT directory is correct.")
"""),
    code("""
if koma_records and len(koma_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Koma53 distribution (all sampled pieces)
    koma_df["Koma53"].plot.hist(bins=53, ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_xlabel("53-TET Koma value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Koma53 Distribution (all sampled makams)\\n"
                       "Non-zero = microtonal deviation from 12-TET")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.5, label="12-TET ref")
    axes[0].legend()

    # Deviation from nearest 12-TET semitone
    # In 53-TET: one semitone = ~4.42 commas. Deviation = Koma53 mod 4.42
    koma_df["deviation_commas"] = koma_df["Koma53"] % (53/12)
    koma_df["deviation_commas"].plot.hist(
        bins=30, ax=axes[1], color="indianred", edgecolor="white")
    axes[1].set_xlabel("Microtonal deviation from nearest 12-TET pitch (commas)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Microtonal Deviation Distribution\\n"
                       "Motivates EC-REMI deviation tokens")

    plt.suptitle("SymbTr 53-TET Koma Analysis", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "eda_turkish_koma53.png"), dpi=150)
    plt.show()
    print("Saved → results/eda_turkish_koma53.png")
"""),
    md("## 7. Sample piano roll"),
    code("""
sample_path = stats["path"].iloc[0]
pm = load_midi(sample_path)
sample_name = Path(sample_path).stem[:60]

fig, ax = plt.subplots(figsize=(14, 4))
piano_roll_plot(pm, ax, time_start=0, time_end=30,
                title=f"Piano roll — {sample_name}")
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "eda_turkish_piano_roll.png"), dpi=150)
plt.show()
"""),
    md("## 8. Save summary statistics + cross-tradition preview"),
    code("""
stats["tradition"] = "turkish_makam"
stats.to_csv(RESULTS_DIR / "eda_turkish_stats.csv", index=False)
print("Saved → results/eda_turkish_stats.csv")
print(f"\\nPC entropy mean : {stats['pc_entropy'].mean():.3f} bits")
print(f"Note density mean: {stats['note_density'].mean():.2f} notes/s")
"""),
    code("""
# Cross-tradition comparison preview (full table built in 05_evaluation)
csv_files = {
    "Western Classical": "eda_maestro_stats.csv",
    "Turkish Makam"    : "eda_turkish_stats.csv",
    "Irish Folk"       : "eda_irish_stats.csv",
    "Hindustani"       : "eda_hindustani_stats.csv",
    "Carnatic"         : "eda_carnatic_stats.csv",
}

rows = []
for tradition, fname in csv_files.items():
    fpath = RESULTS_DIR / fname
    if fpath.exists():
        d = pd.read_csv(fpath)
        rows.append({
            "Tradition"      : tradition,
            "n"              : len(d),
            "Duration (min)" : f"{d['duration_s'].mean()/60:.1f}",
            "Note density"   : f"{d['note_density'].mean():.2f}",
            "Pitch range"    : f"{d['pitch_range'].mean():.1f}",
            "PC entropy"     : f"{d['pc_entropy'].mean():.3f}",
        })

if rows:
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "eda_cross_tradition_summary.csv", index=False)
    print("\\n=== Cross-Tradition EDA Summary ===")
    print(comparison.to_string(index=False))
    print("\\nSaved → results/eda_cross_tradition_summary.csv")
else:
    print("Run remaining EDA notebooks to populate the cross-tradition table.")
"""),
]

# ============================================================
# Write all notebooks
# ============================================================

notebooks = [
    ("01a_maestro_eda.ipynb",    maestro_eda),
    ("01b_hindustani_eda.ipynb", hindustani_eda),
    ("01c_carnatic_eda.ipynb",   carnatic_eda),
    ("01d_irish_folk_eda.ipynb", irish_eda),
    ("01e_turkish_makam_eda.ipynb", turkish_eda),
]

for filename, cells in notebooks:
    nb   = make_nb(cells)
    path = NB_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Written: {path.relative_to(PROJECT_ROOT)}")

print("\nAll 5 EDA notebooks created.")
