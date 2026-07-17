"""
Transcribe remaining Hindustani and Carnatic audio files using Basic-Pitch.

Run from the repo root:
    nohup python scripts/transcribe_indian.py > transcription.log 2>&1 &

Progress is checkpointed per-file. Safe to kill and restart — already-done
files are detected by glob so both old and new naming conventions are handled.
"""

import json
import os
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
HIND_RAW  = REPO_ROOT / "datasets" / "indian_classical" / "saraga1.5_hindustani"
CARN_RAW  = REPO_ROOT / "datasets" / "indian_classical" / "saraga1.5_carnatic"
HIND_MIDI = REPO_ROOT / "data" / "processed" / "hindustani" / "midi"
CARN_MIDI = REPO_ROOT / "data" / "processed" / "carnatic"   / "midi"
META_DIR  = REPO_ROOT / "data" / "metadata"
HIND_META = META_DIR / "hindustani_tracks.csv"
CARN_META = META_DIR / "carnatic_tracks.csv"

HIND_MIDI.mkdir(parents=True, exist_ok=True)
CARN_MIDI.mkdir(parents=True, exist_ok=True)

print(f"Repo root       : {REPO_ROOT}")
print(f"Hindustani raw  : {HIND_RAW.exists()}  ({HIND_RAW})")
print(f"Carnatic raw    : {CARN_RAW.exists()}  ({CARN_RAW})")
if not HIND_RAW.exists() or not CARN_RAW.exists():
    print("ERROR: raw audio directories not found. Aborting.")
    sys.exit(1)

# ── Basic-Pitch ───────────────────────────────────────────────────────────────
print("\nLoading Basic-Pitch...")
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
print(f"Model: {ICASSP_2022_MODEL_PATH}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_stem(path: Path) -> str:
    name = path.name
    for ext in [".mp3.mp3", ".mp3", ".wav"]:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return re.sub(r"[^\w\-]", "_", name)[:60]


def is_concert_mix(audio_path: Path) -> bool:
    stem = audio_path.name
    for ext in [".mp3.mp3", ".mp3", ".wav"]:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    return stem == audio_path.parent.name


def transcribe_audio(audio_path: Path, out_path: Path) -> bool:
    try:
        _, midi_data, _ = predict(str(audio_path), ICASSP_2022_MODEL_PATH)
        midi_data.write(str(out_path))
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def transcribe_with_timeout(audio_path: Path, out_path: Path, timeout: int = 300) -> bool:
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(transcribe_audio, audio_path, out_path)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            print(f"    TIMEOUT after {timeout}s — skipping")
            if out_path.exists():
                out_path.unlink()
            return False


def any_midi_exists(midi_dir: Path, prefix: str) -> bool:
    """True if any file matching prefix_*.mid already exists (handles old + new naming)."""
    return any(midi_dir.glob(f"{prefix}_*.mid"))


def run_tradition(
    label: str,
    all_audio: list,
    midi_dir: Path,
    meta_path: Path,
    out_name_fn,
    extract_meta_fn,
):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Total audio files : {len(all_audio)}")

    # Detect already-done by glob so both old and new filenames are recognised
    already_done = {
        i for i, f in enumerate(all_audio)
        if any_midi_exists(midi_dir, f"{label.lower().replace(' ', '_')}_{i:03d}")
    }
    todo = [(i, f) for i, f in enumerate(all_audio) if i not in already_done]
    print(f"Already done      : {len(already_done)}")
    print(f"To transcribe     : {len(todo)}")

    existing_meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    done_paths = (
        set(existing_meta["audio_path"].dropna())
        if "audio_path" in existing_meta.columns
        else set()
    )

    succeeded = failed = skipped = 0
    session_start = time.time()

    for n, (i, audio_path) in enumerate(todo, 1):
        out_name = out_name_fn(i, audio_path)
        out_path = midi_dir / out_name

        if out_path.exists():
            skipped += 1
            continue

        t0 = time.time()
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [{n:3d}/{len(todo)}] {audio_path.name[:55]} ...", flush=True)
        ok = transcribe_with_timeout(audio_path, out_path, timeout=300)
        elapsed = time.time() - t0

        if ok:
            meta_row = extract_meta_fn(audio_path, out_path)
            row_df = pd.DataFrame([meta_row])
            row_df.to_csv(
                meta_path,
                mode="a",
                header=not meta_path.exists(),
                index=False,
            )
            succeeded += 1
            remaining = len(todo) - n
            avg = (time.time() - session_start) / n
            eta_min = remaining * avg / 60
            print(f"    done ({elapsed:.0f}s) — {remaining} left, ETA ~{eta_min:.0f} min")
        else:
            failed += 1
            print(f"    FAILED ({elapsed:.0f}s)")

    print(f"\n{label} done: {succeeded} new, {skipped} skipped, {failed} failed")
    print(f"Total MIDI files : {len(list(midi_dir.glob('*.mid')))}")


# ── Hindustani ────────────────────────────────────────────────────────────────
all_hind_audio = sorted(
    list(HIND_RAW.rglob("*.mp3.mp3"))
    + [f for f in HIND_RAW.rglob("*.mp3") if not str(f).endswith(".mp3.mp3")]
)


def hind_out_name(i, audio_path):
    return f"hindustani_{i:03d}_{safe_stem(audio_path)}.mid"


def hind_meta(audio_path, out_path):
    raga = None
    json_path = audio_path.parent / (audio_path.parent.name + ".json")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            ragas = data.get("raags") or data.get("ragas") or []
            raga = ragas[0].get("name") if ragas else None
        except Exception:
            pass
    return {
        "audio_path": str(audio_path),
        "midi_path": str(out_path),
        "raga": raga,
        "source": "saraga_hindustani",
    }


run_tradition(
    label="Hindustani",
    all_audio=all_hind_audio,
    midi_dir=HIND_MIDI,
    meta_path=HIND_META,
    out_name_fn=hind_out_name,
    extract_meta_fn=hind_meta,
)

# ── Carnatic ──────────────────────────────────────────────────────────────────
all_carn_audio = sorted(
    list(CARN_RAW.rglob("*.mp3.mp3"))
    + [f for f in CARN_RAW.rglob("*.mp3") if not str(f).endswith(".mp3.mp3")]
)
concert_mixes = [f for f in all_carn_audio if is_concert_mix(f)]


def carn_out_name(i, audio_path):
    return f"carnatic_{i:03d}_{safe_stem(audio_path)}.mid"


def carn_meta(audio_path, out_path):
    raaga = None
    json_path = audio_path.parent / (audio_path.parent.name + ".json")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            raagas = data.get("raaga") or data.get("ragas") or []
            raaga = raagas[0].get("name") if raagas else None
        except Exception:
            pass
    return {
        "audio_path": str(audio_path),
        "midi_path": str(out_path),
        "raga": raaga,
        "source": "saraga_carnatic",
    }


run_tradition(
    label="Carnatic",
    all_audio=concert_mixes,
    midi_dir=CARN_MIDI,
    meta_path=CARN_META,
    out_name_fn=carn_out_name,
    extract_meta_fn=carn_meta,
)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
for label, midi_dir, meta_path in [
    ("Hindustani", HIND_MIDI, HIND_META),
    ("Carnatic",   CARN_MIDI, CARN_META),
]:
    midi_files = list(midi_dir.glob("*.mid"))
    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    print(f"{label}: {len(midi_files)} MIDI files, {len(meta)} metadata rows")
print("\nDone. Next step: run the Music Transformer training notebook.")
