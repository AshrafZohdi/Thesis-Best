#!/usr/bin/env python3
"""
Generate all data preparation notebooks for the RMCE thesis.
Run once from the project root: python3 create_notebooks.py
"""

import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
NB_DIR = PROJECT_ROOT / "notebooks" / "00_data_preparation"


def md(text: str) -> nbf.notebooknode.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.notebooknode.NotebookNode:
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


# ---------------------------------------------------------------------------
# Shared path-resolution snippet inserted at top of every notebook
# ---------------------------------------------------------------------------
PATH_SETUP = """
import sys
from pathlib import Path

# Locate project root regardless of where Jupyter was launched from.
# Searches upward for PROGRESS.md — the root marker.
_here = Path().resolve()
PROJECT_ROOT = _here
for _ in range(5):
    if (PROJECT_ROOT / "PROGRESS.md").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))
print(f"Project root: {PROJECT_ROOT}")
"""

# ============================================================
# 00a — MAESTRO Western Classical
# ============================================================

maestro_cells = [
    md("""# 00a — MAESTRO Western Classical: Data Preparation

**Thesis context:** This notebook selects and copies a representative subset of MIDI files from
the MAESTRO v3.0.0 dataset (Anderson et al., 2019) for use in baseline tokenisation and
fine-tuning experiments. MAESTRO contains piano recordings with aligned MIDI from the
International Piano-e-Competition, 2004–2018.

**Output:**
- `data/processed/western_classical/midi/` — 150 selected MIDI files
- `data/metadata/maestro_selected.csv` — per-file metadata (composer, title, year, duration)

**Sampling strategy:** Stratified by year to preserve the temporal distribution of the dataset.
Files are drawn from the pre-defined `train` split provided in the MAESTRO metadata CSV.
"""),
    code(PATH_SETUP),
    code("""
import pandas as pd
import shutil

RAW_DIR   = PROJECT_ROOT / "datasets" / "western_classical" / "maestro-v3.0.0"
OUT_MIDI  = PROJECT_ROOT / "data" / "processed" / "western_classical" / "midi"
META_DIR  = PROJECT_ROOT / "data" / "metadata"

OUT_MIDI.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
"""),
    md("## 1. Load and inspect MAESTRO metadata"),
    code("""
df = pd.read_csv(RAW_DIR / "maestro-v3.0.0.csv")
print(f"Total records : {len(df)}")
print(f"Columns       : {df.columns.tolist()}")
print(f"Split counts  :\\n{df['split'].value_counts()}")
print(f"Year range    : {df['year'].min()} – {df['year'].max()}")
df.head(3)
"""),
    code("""
# Duration stats (seconds)
print("Duration statistics (seconds):")
print(df[df['split'] == 'train']['duration'].describe().round(1))
"""),
    md("## 2. Filter to training split"),
    code("""
train_df = df[df['split'] == 'train'].copy()
print(f"Train-split records: {len(train_df)}")
print("\\nFiles per year (train split):")
print(train_df['year'].value_counts().sort_index().to_string())
"""),
    md("""## 3. Stratified sample — 150 files, proportional across years

We sample proportionally so every competition year is represented in roughly its
original ratio. `random_state=42` makes the selection reproducible.
"""),
    code("""
N_TARGET = 150

sample = (
    train_df
    .groupby("year", group_keys=False)
    .apply(lambda x: x.sample(
        min(len(x), max(1, round(N_TARGET * len(x) / len(train_df)))),
        random_state=42
    ))
)

# Rounding may produce slightly more or fewer than N_TARGET — trim / top-up
if len(sample) > N_TARGET:
    sample = sample.sample(N_TARGET, random_state=42)
elif len(sample) < N_TARGET:
    remaining = train_df[~train_df.index.isin(sample.index)]
    topup = remaining.sample(min(N_TARGET - len(sample), len(remaining)), random_state=42)
    sample = pd.concat([sample, topup])

print(f"Selected {len(sample)} files")
print("\\nFiles per year in selection:")
print(sample['year'].value_counts().sort_index().to_string())
"""),
    code("""
print("\\nTop 15 composers in selection:")
print(sample['canonical_composer'].value_counts().head(15).to_string())
"""),
    md("## 4. Copy MIDI files to processed directory"),
    code("""
copied_names = []
missing = []

for _, row in sample.iterrows():
    src = RAW_DIR / row['midi_filename']
    # Flatten the year/filename path into a single name separated by underscore
    flat_name = row['midi_filename'].replace('/', '_')
    dst = OUT_MIDI / flat_name
    if src.exists():
        shutil.copy2(src, dst)
        copied_names.append(flat_name)
    else:
        missing.append(str(src))
        copied_names.append(None)

print(f"Copied : {len([n for n in copied_names if n])} files")
if missing:
    print(f"Missing: {missing}")
"""),
    md("## 5. Save metadata and verify"),
    code("""
sample = sample.copy()
sample['processed_filename'] = copied_names
sample.to_csv(META_DIR / "maestro_selected.csv", index=False)
print(f"Metadata saved → data/metadata/maestro_selected.csv")

# Verification
midi_files = list(OUT_MIDI.glob("*.midi")) + list(OUT_MIDI.glob("*.mid"))
print(f"\\nMIDI files in output dir : {len(midi_files)}")
print(f"Duration range (min)     : {sample['duration'].min()/60:.1f} – {sample['duration'].max()/60:.1f}")
print(f"Total duration (hours)   : {sample['duration'].sum()/3600:.2f}")
print("\\n✓ MAESTRO preparation complete.")
"""),
]

# ============================================================
# 00b — Saraga Hindustani
# ============================================================

hindustani_cells = [
    md("""# 00b — Saraga Hindustani: Data Preparation & Transcription

**Thesis context:** This notebook processes the Hindustani subset of the Saraga 1.5 dataset
(Srinivasamurthy et al., 2021). Recordings are full concert MP3s; we transcribe them to MIDI
using Basic-Pitch (Bitteur et al., 2022) to obtain symbolic representations suitable for
tokenisation.

**Key decisions:**
- **Backend:** Basic-Pitch is called with the explicit TF SavedModel path (`ICASSP_2022_MODEL_PATH`)
  to avoid the macOS CoreML floating-point exception that occurs when the CoreML backend is
  selected implicitly.
- **Metadata:** Per-track `.json` files provide raga label (`raags[0]['common_name']`), taal,
  and form. Where the JSON `raags` array is empty, the raga name is inferred from the concert
  folder name (e.g. "Raag Shree by Artist" → "Shree"). This fallback is flagged in the output.
- **File naming quirk:** Hindustani audio files carry a double `.mp3.mp3` extension (download
  artefact). The code strips the extra extension before constructing output paths.
- **Track selection:** 60 tracks are selected from the 108 available, stratified across ragas to
  maximise diversity.

**Output:**
- `data/processed/hindustani/midi/` — 60 transcribed MIDI files
- `data/metadata/hindustani_tracks.csv` — per-track metadata with raga labels and metadata source

**Time estimate:** ~5–8 min per file on CPU × 60 files ≈ **5–8 hours total**.
Run this notebook overnight or in a background terminal.
"""),
    code("""
# IMPORTANT: Import TensorFlow before basic_pitch to ensure the TF SavedModel
# backend is loaded rather than CoreML (macOS). This must be the first import.
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
"""),
    code(PATH_SETUP),
    code("""
import pandas as pd
import json
import shutil
import time

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

print(f"Basic-Pitch model: {ICASSP_2022_MODEL_PATH}")

RAW_DIR   = PROJECT_ROOT / "datasets" / "indian_classical" / "saraga1.5_hindustani"
OUT_MIDI  = PROJECT_ROOT / "data" / "processed" / "hindustani" / "midi"
META_DIR  = PROJECT_ROOT / "data" / "metadata"

OUT_MIDI.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
"""),
    md("## 1. Scan all tracks and load metadata"),
    code("""
# Load the file_paths.csv — the curated track list provided by the Saraga dataset.
# filepath column: "{concert_folder}/{raga_folder}/{piece_name}" (no extension)
fpc = pd.read_csv(RAW_DIR / "file_paths.csv", index_col=0)
print(f"file_paths.csv entries: {len(fpc)}")
fpc.head()
"""),
    code("""
def find_audio_file(raw_dir, filepath_stem):
    \"\"\"Find the audio file for a given stem path from file_paths.csv.\"\"\"
    # Hindustani: double extension .mp3.mp3
    for ext in [".mp3.mp3", ".mp3"]:
        p = raw_dir / (filepath_stem + ext)
        if p.exists():
            return p
    return None


def parse_hindustani_json(json_path):
    \"\"\"Extract raga, taal, form from a Saraga Hindustani JSON metadata file.\"\"\"
    try:
        meta = json.load(open(json_path, encoding="utf-8"))
        raags  = meta.get("raags", [])
        taals  = meta.get("taals", [])
        forms  = meta.get("forms", [])
        return {
            "title"  : meta.get("title", ""),
            "mbid"   : meta.get("mbid", ""),
            "raga"   : raags[0]["common_name"] if raags else None,
            "taal"   : taals[0]["common_name"] if taals else None,
            "form"   : forms[0]["common_name"] if forms else None,
            "length_ms": meta.get("length", None),
        }
    except Exception as e:
        return {"error": str(e)}


records = []
for _, row in fpc.iterrows():
    fp_stem = row["filepath"]
    audio   = find_audio_file(RAW_DIR, fp_stem)
    json_p  = RAW_DIR / (fp_stem + ".json")

    meta = parse_hindustani_json(json_p) if json_p.exists() else {}

    # Fallback: infer raga from concert folder name ("Raag Shree by Artist" → "Shree")
    raga = meta.get("raga")
    meta_src = "json"
    if not raga:
        concert_folder = fp_stem.split("/")[0]
        # Strip "Raag " prefix and artist suffix
        raga_raw = concert_folder.split(" by ")[0].replace("Raag ", "").strip()
        raga = raga_raw if raga_raw else "Unknown"
        meta_src = "folder_name"

    records.append({
        "filepath_stem": fp_stem,
        "audio_path"   : str(audio) if audio else None,
        "json_exists"  : json_path.exists() if (json_path := json_p) else False,
        "title"        : meta.get("title", ""),
        "mbid"         : row.get("mbid", ""),
        "raga"         : raga,
        "taal"         : meta.get("taal", ""),
        "form"         : meta.get("form", ""),
        "length_ms"    : meta.get("length_ms", None),
        "metadata_source": meta_src,
    })

tracks_df = pd.DataFrame(records)
print(f"Total tracks found: {len(tracks_df)}")
print(f"\\nMetadata source distribution:")
print(tracks_df["metadata_source"].value_counts().to_string())
print(f"\\nTracks with audio file: {tracks_df['audio_path'].notna().sum()}")
print(f"Unique ragas           : {tracks_df['raga'].nunique()}")
"""),
    code("""
# Show raga distribution across available tracks
print("Raga distribution (top 20):")
print(tracks_df["raga"].value_counts().head(20).to_string())
"""),
    md("""## 2. Select 60 tracks — stratified across ragas

We sample proportionally across ragas to represent the breadth of the Hindustani
tradition in the dataset. Tracks with JSON metadata are preferred; folder-name fallbacks
are used only to reach the target count.
"""),
    code("""
N = 60

# Prefer JSON-sourced tracks
json_tracks = tracks_df[
    (tracks_df["metadata_source"] == "json") & tracks_df["audio_path"].notna()
].copy()

sampled = (
    json_tracks
    .groupby("raga", group_keys=False)
    .apply(lambda x: x.sample(
        min(len(x), max(1, round(N * len(x) / len(json_tracks)))),
        random_state=42
    ))
)

if len(sampled) > N:
    sampled = sampled.sample(N, random_state=42)
elif len(sampled) < N:
    remaining = tracks_df[
        ~tracks_df["filepath_stem"].isin(sampled["filepath_stem"]) &
        tracks_df["audio_path"].notna()
    ]
    topup = remaining.sample(min(N - len(sampled), len(remaining)), random_state=42)
    sampled = pd.concat([sampled, topup])

sampled = sampled.reset_index(drop=True)
print(f"Selected {len(sampled)} tracks")
print(f"\\nRaga distribution in selection:")
print(sampled["raga"].value_counts().to_string())
"""),
    code("""
# Save track selection to metadata CSV before starting transcription
# (transcription can be safely interrupted and resumed)
sampled.to_csv(META_DIR / "hindustani_tracks.csv", index=False)
print("Selection saved → data/metadata/hindustani_tracks.csv")
"""),
    md("""## 3. Basic-Pitch transcription

**Runtime warning:** Each full concert recording (~35–50 minutes of audio) takes ~5–8 minutes
on CPU. For 60 files this is approximately **5–8 hours**. The loop is idempotent — already
transcribed files are skipped, so it is safe to interrupt and re-run.

**Transcription note:** Basic-Pitch transcribes all audible pitched content from the full mix.
For Hindustani recordings this captures the lead vocal/instrument plus harmonium and tabla.
This is a known limitation of automatic transcription from mixed audio; it is documented as
a methodological consideration in the thesis (Chapter 3).
"""),
    code("""
midi_paths = []
failed     = []

for i, row in sampled.iterrows():
    audio_path = Path(row["audio_path"])
    raga_safe  = row["raga"].replace(" ", "_").replace("/", "_")[:30]
    out_name   = f"hindustani_{i:03d}_{raga_safe}.mid"
    out_path   = OUT_MIDI / out_name

    # Skip if already transcribed
    if out_path.exists():
        print(f"[{i+1:3d}/{len(sampled)}] SKIP  {out_name}")
        midi_paths.append(str(out_path))
        continue

    print(f"[{i+1:3d}/{len(sampled)}] Transcribing: {audio_path.name[:60]} ...", end="", flush=True)
    t0 = time.time()
    try:
        _, midi_data, _ = predict(str(audio_path), ICASSP_2022_MODEL_PATH)
        midi_data.write(str(out_path))
        elapsed = time.time() - t0
        print(f" {elapsed/60:.1f} min")
        midi_paths.append(str(out_path))
    except Exception as e:
        print(f" FAILED: {e}")
        failed.append({"index": i, "audio": str(audio_path), "error": str(e)})
        midi_paths.append(None)

print(f"\\n=== Transcription complete ===")
print(f"Succeeded : {sum(p is not None for p in midi_paths)}")
print(f"Failed    : {len(failed)}")
if failed:
    for f in failed:
        print(f"  [{f['index']}] {f['audio']}: {f['error']}")
"""),
    md("## 4. Update metadata and verify"),
    code("""
sampled = sampled.copy()
sampled["midi_path"] = midi_paths
sampled.to_csv(META_DIR / "hindustani_tracks.csv", index=False)

# Summary
midi_files = list(OUT_MIDI.glob("*.mid"))
print(f"MIDI files in output dir          : {len(midi_files)}")
print(f"Tracks with MIDI path recorded    : {sampled['midi_path'].notna().sum()}")
print(f"Raga coverage                     : {sampled['raga'].nunique()} unique ragas")
print("\\n✓ Hindustani preparation complete.")
"""),
]

# ============================================================
# 00c — Saraga Carnatic
# ============================================================

carnatic_cells = [
    md("""# 00c — Saraga Carnatic: Data Preparation & Transcription

**Thesis context:** This notebook processes the Carnatic subset of the Saraga 1.5 dataset.
Carnatic classical music differs from Hindustani in its rhythmic organisation (tala system),
melodic structures (raga with different characteristic phrases), and ensemble interaction
patterns (primary vocal with violin, mridangam, and ghatam).

**Metadata note:** Unlike the Hindustani subset, many Carnatic JSON files have empty `raaga`
arrays. This appears to be a known metadata completeness issue in Saraga Carnatic 1.5.
Where `raaga` is empty, the raga name is inferred from the piece title or concert folder name.
This limitation is documented and affects a majority of the 249 curated tracks.

**Track selection:** The Saraga Carnatic collection contains 991 MP3 files — 249 are full
concert mixes (listed in file_paths.csv); the remainder are separated audio stems (vocal,
violin, mridangam). We use only the full mixes. 60 tracks are selected across the 27
available concert folders for performer diversity.

**Output:**
- `data/processed/carnatic/midi/` — 60 transcribed MIDI files
- `data/metadata/carnatic_tracks.csv` — per-track metadata

**Time estimate:** ~5–8 hours on CPU.
"""),
    code("""
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
"""),
    code(PATH_SETUP),
    code("""
import pandas as pd
import json
import shutil
import time

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

RAW_DIR  = PROJECT_ROOT / "datasets" / "indian_classical" / "saraga1.5_carnatic"
OUT_MIDI = PROJECT_ROOT / "data" / "processed" / "carnatic" / "midi"
META_DIR = PROJECT_ROOT / "data" / "metadata"

OUT_MIDI.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
"""),
    md("## 1. Scan tracks from file_paths.csv"),
    code("""
fpc = pd.read_csv(RAW_DIR / "file_paths.csv", index_col=0)
print(f"file_paths.csv entries (full mixes): {len(fpc)}")
fpc.head()
"""),
    code("""
def find_carnatic_audio(raw_dir, filepath_stem):
    \"\"\"Locate the Carnatic audio file; Carnatic files may use single .mp3 extension.\"\"\"
    for ext in [".mp3", ".mp3.mp3"]:
        p = raw_dir / (filepath_stem + ext)
        if p.exists():
            return p
    return None


def parse_carnatic_json(json_path):
    \"\"\"
    Extract raga/taal/form from a Saraga Carnatic JSON file.

    Key difference from Hindustani: the field is 'raaga' (singular),
    and the raga object often has only 'name' (Unicode transliteration),
    not 'common_name'. Many files have an empty 'raaga' list.
    \"\"\"
    try:
        meta = json.load(open(json_path, encoding="utf-8"))
        raaga_list = meta.get("raaga", [])
        taala_list = meta.get("taala", [])
        form_list  = meta.get("form", [])
        return {
            "title"    : meta.get("title", ""),
            "mbid"     : meta.get("mbid", ""),
            # Prefer 'common_name'; fall back to 'name' (Unicode)
            "raga"     : (raaga_list[0].get("common_name") or raaga_list[0].get("name", ""))
                         if raaga_list else None,
            "taal"     : (taala_list[0].get("common_name") or taala_list[0].get("name", ""))
                         if taala_list else None,
            "form"     : (form_list[0].get("name", "")) if form_list else None,
            "length_ms": meta.get("length", None),
        }
    except Exception as e:
        return {"error": str(e)}


records = []
for _, row in fpc.iterrows():
    fp_stem = row["filepath"]
    audio   = find_carnatic_audio(RAW_DIR, fp_stem)
    json_p  = RAW_DIR / (fp_stem + ".json")
    meta    = parse_carnatic_json(json_p) if json_p.exists() else {}

    raga     = meta.get("raga")
    meta_src = "json"
    if not raga:
        # Fallback: use the piece title (middle component of path)
        parts    = fp_stem.split("/")
        raga_raw = parts[1] if len(parts) >= 2 else parts[-1]
        raga     = raga_raw.strip() or "Unknown"
        meta_src = "title_fallback"

    records.append({
        "filepath_stem"   : fp_stem,
        "audio_path"      : str(audio) if audio else None,
        "title"           : meta.get("title", ""),
        "mbid"            : row.get("mbid", ""),
        "raga"            : raga,
        "taal"            : meta.get("taal", ""),
        "form"            : meta.get("form", ""),
        "length_ms"       : meta.get("length_ms", None),
        "metadata_source" : meta_src,
        "concert_folder"  : fp_stem.split("/")[0],
    })

tracks_df = pd.DataFrame(records)
print(f"Total tracks: {len(tracks_df)}")
print(f"\\nMetadata source breakdown:")
print(tracks_df["metadata_source"].value_counts().to_string())
print(f"\\nTracks with audio: {tracks_df['audio_path'].notna().sum()}")
print(f"Concert folders  : {tracks_df['concert_folder'].nunique()}")
"""),
    md("""## 2. Select 60 tracks across concert folders

We prioritise tracks with JSON-sourced raga labels and sample to cover all 27 concert
folders (i.e. performers), ensuring performer diversity in the selected subset.
"""),
    code("""
N = 60
available = tracks_df[tracks_df["audio_path"].notna()].copy()
print(f"Tracks with audio available: {len(available)}")

# Stratify by concert folder (performer) for diversity
sampled = (
    available
    .groupby("concert_folder", group_keys=False)
    .apply(lambda x: x.sample(
        min(len(x), max(1, round(N * len(x) / len(available)))),
        random_state=42
    ))
)

if len(sampled) > N:
    sampled = sampled.sample(N, random_state=42)
elif len(sampled) < N:
    remaining = available[~available["filepath_stem"].isin(sampled["filepath_stem"])]
    topup = remaining.sample(min(N - len(sampled), len(remaining)), random_state=42)
    sampled = pd.concat([sampled, topup])

sampled = sampled.reset_index(drop=True)
print(f"Selected: {len(sampled)} tracks across {sampled['concert_folder'].nunique()} performers")
print(f"\\nMetadata source in selection:")
print(sampled["metadata_source"].value_counts().to_string())
"""),
    code("""
sampled.to_csv(META_DIR / "carnatic_tracks.csv", index=False)
print("Selection saved → data/metadata/carnatic_tracks.csv")
"""),
    md("""## 3. Basic-Pitch transcription

Same configuration as Hindustani (TF backend, ICASSP_2022 model). Idempotent —
re-run safely after interruption.
"""),
    code("""
midi_paths = []
failed     = []

for i, row in sampled.iterrows():
    audio_path = Path(row["audio_path"])
    raga_safe  = str(row["raga"]).replace(" ", "_").replace("/", "_")[:30]
    out_name   = f"carnatic_{i:03d}_{raga_safe}.mid"
    out_path   = OUT_MIDI / out_name

    if out_path.exists():
        print(f"[{i+1:3d}/{len(sampled)}] SKIP  {out_name}")
        midi_paths.append(str(out_path))
        continue

    print(f"[{i+1:3d}/{len(sampled)}] Transcribing: {audio_path.name[:60]} ...", end="", flush=True)
    t0 = time.time()
    try:
        _, midi_data, _ = predict(str(audio_path), ICASSP_2022_MODEL_PATH)
        midi_data.write(str(out_path))
        elapsed = time.time() - t0
        print(f" {elapsed/60:.1f} min")
        midi_paths.append(str(out_path))
    except Exception as e:
        print(f" FAILED: {e}")
        failed.append({"index": i, "audio": str(audio_path), "error": str(e)})
        midi_paths.append(None)

print(f"\\n=== Transcription complete ===")
print(f"Succeeded : {sum(p is not None for p in midi_paths)}")
print(f"Failed    : {len(failed)}")
"""),
    md("## 4. Update metadata and verify"),
    code("""
sampled = sampled.copy()
sampled["midi_path"] = midi_paths
sampled.to_csv(META_DIR / "carnatic_tracks.csv", index=False)

midi_files = list(OUT_MIDI.glob("*.mid"))
print(f"MIDI files in output dir    : {len(midi_files)}")
print(f"Tracks with MIDI path       : {sampled['midi_path'].notna().sum()}")
print(f"Performer (concert) coverage: {sampled['concert_folder'].nunique()}")
print("\\n✓ Carnatic preparation complete.")
"""),
]

# ============================================================
# 00d — TheSession Irish Folk
# ============================================================

irish_cells = [
    md("""# 00d — TheSession Irish Folk: Data Preparation

**Thesis context:** TheSession.org is a community database of Irish traditional music
published under CC BY-NC 4.0. The data export contains ABC notation for over 267,000
tune *settings* (individual notations of tunes). A setting is one person's written version
of a tune; many settings exist for the same underlying tune.

**Pipeline:**
1. Deduplicate by `tune_id` (keep one setting per unique tune)
2. Join with `tune_popularity.csv` to rank by community popularity
3. Stratify selection across tune types (reel, jig, hornpipe, etc.) for repertoire diversity
4. Convert ABC notation → MIDI via `music21`

**Output:**
- `data/processed/irish_folk/midi/` — 200 MIDI files
- `data/metadata/irish_folk_tunes.csv` — per-tune metadata

**Key metadata used:**
- `type`: tune category (reel, jig, hornpipe, polka, slide, waltz, mazurka, strathspey, ...)
- `mode`: key and mode (e.g. "Gmajor", "Ador", "Dmixolydian")
- `meter`: time signature (4/4, 6/8, 9/8, 3/4, ...)
- `tune_popularity.csv`: tunebooks count — proxy for community canonical status
"""),
    code(PATH_SETUP),
    code("""
import pandas as pd
import re
import time
import warnings

from music21 import converter, midi as m21_midi

RAW_CSV  = PROJECT_ROOT / "datasets" / "irish_folk" / "TheSession-data" / "csv"
OUT_MIDI = PROJECT_ROOT / "data" / "processed" / "irish_folk" / "midi"
META_DIR = PROJECT_ROOT / "data" / "metadata"

OUT_MIDI.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
"""),
    md("## 1. Load tunes.csv and inspect"),
    code("""
tunes = pd.read_csv(RAW_CSV / "tunes.csv")
print(f"Total rows (settings): {len(tunes):,}")
print(f"Columns: {tunes.columns.tolist()}")
print(f"Unique tune_ids: {tunes['tune_id'].nunique():,}")
tunes.head(3)
"""),
    code("""
print("Tune type distribution (all settings):")
print(tunes["type"].value_counts().head(15).to_string())
print("\\nMode distribution (top 10):")
print(tunes["mode"].value_counts().head(10).to_string())
print("\\nMeter distribution:")
print(tunes["meter"].value_counts().to_string())
"""),
    md("## 2. Load popularity and deduplicate"),
    code("""
popularity = pd.read_csv(RAW_CSV / "tune_popularity.csv")
print(f"Popularity entries: {len(popularity):,}")
popularity.head(3)
"""),
    code("""
# Keep one setting per tune_id — the lowest setting_id is typically the
# oldest/original notation. This gives us one ABC string per unique tune.
unique_tunes = (
    tunes
    .sort_values("setting_id")
    .drop_duplicates(subset="tune_id", keep="first")
    .copy()
)
print(f"Unique tunes after deduplication: {len(unique_tunes):,}")

# Join with popularity
unique_tunes = unique_tunes.merge(
    popularity[["tune_id", "tunebooks"]],
    on="tune_id",
    how="left"
)
unique_tunes["tunebooks"] = unique_tunes["tunebooks"].fillna(0).astype(int)
print(f"Tunes with popularity data: {unique_tunes['tunebooks'].gt(0).sum():,}")
print(f"\\nTop 5 tunes by popularity:")
print(unique_tunes.nlargest(5, "tunebooks")[["name", "type", "mode", "tunebooks"]].to_string())
"""),
    md("""## 3. Stratified selection — 200 tunes across type categories

We sample proportionally from each tune type so the selection reflects the natural
distribution of the Irish tradition (which is dominated by reels and jigs).
"""),
    code("""
N = 200

# Focus on the main tune types with enough data
main_types = unique_tunes["type"].value_counts()
print("Type distribution in unique tunes:")
print(main_types.to_string())
"""),
    code("""
# Sort by popularity within each type, then sample proportionally
sampled_parts = []
type_counts = unique_tunes["type"].value_counts()
total_unique = len(unique_tunes)

for tune_type, count in type_counts.items():
    n_select = max(1, round(N * count / total_unique))
    subset = (
        unique_tunes[unique_tunes["type"] == tune_type]
        .sort_values("tunebooks", ascending=False)
        .head(n_select)
    )
    sampled_parts.append(subset)

sampled = pd.concat(sampled_parts)
if len(sampled) > N:
    sampled = sampled.nlargest(N, "tunebooks")
elif len(sampled) < N:
    remaining = unique_tunes[~unique_tunes["tune_id"].isin(sampled["tune_id"])]
    topup = remaining.nlargest(N - len(sampled), "tunebooks")
    sampled = pd.concat([sampled, topup])

sampled = sampled.reset_index(drop=True)
print(f"Selected {len(sampled)} tunes")
print("\\nType distribution in selection:")
print(sampled["type"].value_counts().to_string())
"""),
    md("## 4. ABC → MIDI conversion"),
    code("""
def mode_to_abc_key(mode_str):
    \"\"\"Convert TheSession mode strings to ABC K: field notation.

    Examples: 'Gmajor' → 'G', 'Aminor' → 'Am', 'Dmixolydian' → 'Dmix'
    Reference: ABC notation standard v2.1
    \"\"\"
    mode_map = [
        ("major", ""),
        ("minor", "m"),
        ("mixolydian", "mix"),
        ("dorian", "dor"),
        ("lydian", "lyd"),
        ("phrygian", "phr"),
        ("locrian", "loc"),
    ]
    s = str(mode_str)
    for suffix, abc_suffix in mode_map:
        if s.lower().endswith(suffix):
            key_letter = s[:len(s) - len(suffix)]
            return key_letter + abc_suffix
    return s  # fallback: use as-is


def build_abc_string(row):
    \"\"\"Construct a complete ABC notation string from a TheSession tunes.csv row.\"\"\"
    key = mode_to_abc_key(row["mode"])
    meter = str(row["meter"])
    note_len = "1/8"  # standard default
    title = str(row["name"]).replace('"', "'")
    composer = str(row.get("composer", "")) if pd.notna(row.get("composer", "")) else ""

    header  = f"X:{row['tune_id']}\\n"
    header += f"T:{title}\\n"
    if composer:
        header += f"C:{composer}\\n"
    header += f"M:{meter}\\n"
    header += f"L:{note_len}\\n"
    header += f"K:{key}\\n"
    return header + str(row["abc"])


# Test on first tune
test_row = sampled.iloc[0]
test_abc = build_abc_string(test_row)
print("Sample ABC string:")
print(test_abc[:300])
"""),
    code("""
midi_paths = []
failed     = []
warnings.filterwarnings("ignore")  # suppress music21 verbose output

for i, row in sampled.iterrows():
    type_safe = str(row["type"]).replace(" ", "_")
    out_name  = f"irish_{i:03d}_{row['tune_id']}_{type_safe}.mid"
    out_path  = OUT_MIDI / out_name

    if out_path.exists():
        print(f"[{i+1:3d}/{len(sampled)}] SKIP  {out_name}")
        midi_paths.append(str(out_path))
        continue

    try:
        abc_str = build_abc_string(row)
        score   = converter.parse(abc_str, format="abc")
        score.write("midi", fp=str(out_path))
        midi_paths.append(str(out_path))
        if (i + 1) % 25 == 0:
            print(f"[{i+1:3d}/{len(sampled)}] Converted {i+1} tunes...")
    except Exception as e:
        failed.append({"index": i, "tune_id": row["tune_id"], "error": str(e)})
        midi_paths.append(None)

print(f"\\n=== Conversion complete ===")
print(f"Converted : {sum(p is not None for p in midi_paths)}/{len(sampled)}")
print(f"Failed    : {len(failed)}")
if failed:
    for f in failed[:10]:
        print(f"  tune_id={f['tune_id']}: {f['error']}")
"""),
    md("## 5. Save metadata and verify"),
    code("""
sampled = sampled.copy()
sampled["midi_path"] = midi_paths
sampled.to_csv(META_DIR / "irish_folk_tunes.csv", index=False)

midi_files = list(OUT_MIDI.glob("*.mid"))
print(f"MIDI files in output dir : {len(midi_files)}")
print(f"Conversion success rate  : {sampled['midi_path'].notna().sum()}/{len(sampled)}")
print(f"\\nMode distribution in selection:")
print(sampled["mode"].value_counts().head(10).to_string())
print("\\n✓ Irish folk preparation complete.")
"""),
]

# ============================================================
# 00e — SymbTr Turkish Makam
# ============================================================

symbtr_cells = [
    md("""# 00e — SymbTr Turkish Makam: Data Preparation

**Thesis context:** SymbTr (Karaosmanoğlu, 2012) is the largest machine-readable collection
of Turkish Makam music, containing 2,200 pieces in 155 makams. MIDI files are available
directly — no transcription is needed. This makes Turkish Makam the only tradition in this
study with native symbolic representation.

**Metadata:** Every filename encodes four fields separated by `--`:
`{makam}--{form}--{usul}--{title}--{composer}.mid`
- **makam**: tonal/modal system (e.g. `rast`, `hicaz`, `saba`, `ussak`) — equivalent to raga
- **usul**: rhythmic cycle (e.g. `duyek`, `aksak`, `sofyan`) — equivalent to tala
- **form**: compositional form (e.g. `sarki`, `pesrev`, `sazsemaisi`)

Additionally, the SymbTr TXT files contain a `Koma53` column encoding pitch in 53-tone equal
temperament commas — the microtonal ground truth that will inform EC-REMI deviation tokens
in Step 5.

**Sampling strategy:** Stratified across makams. The top 20 most frequent makams receive
proportional representation; rarer makams receive at least 1 sample.

**Output:**
- `data/processed/turkish_makam/midi/` — 200 MIDI files
- `data/metadata/symbtr_selected.csv` — per-piece metadata (makam, usul, form, composer)
"""),
    code(PATH_SETUP),
    code("""
import pandas as pd
import shutil
import re

RAW_MIDI = PROJECT_ROOT / "datasets" / "SymbTr" / "midi"
RAW_TXT  = PROJECT_ROOT / "datasets" / "SymbTr" / "txt"
OUT_MIDI = PROJECT_ROOT / "data" / "processed" / "turkish_makam" / "midi"
META_DIR = PROJECT_ROOT / "data" / "metadata"

OUT_MIDI.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
"""),
    md("## 1. Parse all filenames → metadata DataFrame"),
    code("""
all_midi = sorted(RAW_MIDI.glob("*.mid"))
print(f"Total SymbTr MIDI files: {len(all_midi)}")

records = []
for p in all_midi:
    stem   = p.stem
    parts  = stem.split("--")
    # Filename pattern: makam--form--usul--title--composer
    # Some pieces have fewer components; handle gracefully
    makam    = parts[0] if len(parts) > 0 else "unknown"
    form     = parts[1] if len(parts) > 1 else "unknown"
    usul     = parts[2] if len(parts) > 2 else "unknown"
    title    = parts[3].replace("_", " ") if len(parts) > 3 else stem
    composer = parts[4].replace("_", " ") if len(parts) > 4 else "unknown"

    records.append({
        "filename": p.name,
        "makam"   : makam,
        "form"    : form,
        "usul"    : usul,
        "title"   : title,
        "composer": composer,
        "midi_path": str(p),
    })

df = pd.DataFrame(records)
print(f"Parsed {len(df)} pieces")
print(f"\\nUnique makams : {df['makam'].nunique()}")
print(f"Unique usuls  : {df['usul'].nunique()}")
print(f"Unique forms  : {df['form'].nunique()}")
df.head(5)
"""),
    md("## 2. Makam frequency distribution"),
    code("""
makam_counts = df["makam"].value_counts()
print(f"Makam distribution (top 30 of {len(makam_counts)}):")
print(makam_counts.head(30).to_string())
"""),
    code("""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt

top_makams = makam_counts.head(25)
fig, ax = plt.subplots(figsize=(12, 5))
top_makams.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_xlabel("Makam")
ax.set_ylabel("Number of pieces")
ax.set_title("SymbTr: Top 25 Makams by Piece Count (full collection, n=2200)")
ax.tick_params(axis="x", rotation=45, labelsize=9)
plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / "results" / "symbtr_makam_distribution.png"), dpi=150)
plt.show()
print("Chart saved → results/symbtr_makam_distribution.png")
"""),
    md("""## 3. Stratified sample — 200 pieces across makams

Each makam receives at least 1 piece. Makams with more pieces receive proportionally
more, capped to avoid over-representing any single makam.
"""),
    code("""
N = 200

sampled_parts = []
for makam, count in makam_counts.items():
    n_select = max(1, round(N * count / len(df)))
    subset   = df[df["makam"] == makam].sample(
        min(n_select, count), random_state=42
    )
    sampled_parts.append(subset)

sampled = pd.concat(sampled_parts)
if len(sampled) > N:
    sampled = sampled.sample(N, random_state=42)
elif len(sampled) < N:
    remaining = df[~df["filename"].isin(sampled["filename"])]
    topup = remaining.sample(min(N - len(sampled), len(remaining)), random_state=42)
    sampled = pd.concat([sampled, topup])

sampled = sampled.reset_index(drop=True)
print(f"Selected: {len(sampled)} pieces across {sampled['makam'].nunique()} makams")
print(f"\\nMakam distribution in selection (top 20):")
print(sampled["makam"].value_counts().head(20).to_string())
"""),
    md("## 4. Copy MIDI files to processed directory"),
    code("""
copied_names = []

for i, row in sampled.iterrows():
    src      = Path(row["midi_path"])
    out_name = f"symbtr_{i:03d}_{row['makam']}_{row['form']}.mid"
    dst      = OUT_MIDI / out_name
    shutil.copy2(src, dst)
    copied_names.append(out_name)

print(f"Copied {len(copied_names)} files → data/processed/turkish_makam/midi/")
"""),
    md("""## 5. Note on 53-TET Koma data in TXT files

The SymbTr TXT files contain a `Koma53` column: pitch expressed in 53-tone equal
temperament (53-TET) commas. This is the theoretical pitch representation used in
Ottoman/Turkish Makam music theory (Arel-Ezgi-Uzdilek system).

This data will be used in Step 5 (EC-REMI design) to calibrate the microtonal deviation
tokens for Turkish Makam — specifically to set the Koma offset values for each makam's
characteristic pitch inflections. A brief preview is shown below.
"""),
    code("""
# Preview the 53-TET data from the TXT file for one selected piece
sample_piece = sampled.iloc[0]
txt_filename = sample_piece["filename"].replace(".mid", ".txt")
txt_path     = RAW_TXT / txt_filename

if txt_path.exists():
    txt_df = pd.read_csv(txt_path, sep="\\t", nrows=20, encoding="utf-8")
    print(f"TXT file: {txt_filename}")
    print(f"Columns: {txt_df.columns.tolist()}")
    print("\\nFirst 10 note rows:")
    note_rows = txt_df[txt_df["Koma53"].notna() & (txt_df["Koma53"] != "")].head(10)
    print(note_rows[["Sira", "NotaAE", "Koma53", "KomaAE", "Pay", "Payda", "Bas"]].to_string())
else:
    print(f"TXT file not found: {txt_path}")
"""),
    md("## 6. Save metadata and verify"),
    code("""
sampled = sampled.copy()
sampled["processed_filename"] = copied_names
sampled.to_csv(META_DIR / "symbtr_selected.csv", index=False)
print("Metadata saved → data/metadata/symbtr_selected.csv")

midi_files = list(OUT_MIDI.glob("*.mid"))
print(f"\\nMIDI files in output dir : {len(midi_files)}")
print(f"Makam coverage           : {sampled['makam'].nunique()} makams")
print(f"Form coverage            : {sampled['form'].nunique()} forms")
print(f"Usul coverage            : {sampled['usul'].nunique()} usuls")
print("\\n✓ Turkish Makam preparation complete.")
"""),
]

# ============================================================
# Write all notebooks
# ============================================================

notebooks = [
    ("00a_maestro_prep.ipynb",    maestro_cells),
    ("00b_hindustani_prep.ipynb", hindustani_cells),
    ("00c_carnatic_prep.ipynb",   carnatic_cells),
    ("00d_irish_folk_prep.ipynb", irish_cells),
    ("00e_turkish_makam_prep.ipynb", symbtr_cells),
]

for filename, cells in notebooks:
    nb   = make_nb(cells)
    path = NB_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Written: {path.relative_to(PROJECT_ROOT)}")

print("\nAll 5 data preparation notebooks created.")
