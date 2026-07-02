"""
Shared MIDI analysis utilities for the RMCE thesis pipeline.

Used by: EDA notebooks (01_eda/), baseline tokenisation (02_baseline_remi/),
         and comparative evaluation (05_evaluation/).

References:
  Yang & Lerch (2020) — objective MIR metrics, Neural Computing and Applications.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_midi(path: str | Path) -> Optional[pretty_midi.PrettyMIDI]:
    """Load a MIDI file and return a PrettyMIDI object, or None on failure."""
    try:
        return pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------

def get_all_notes(pm: pretty_midi.PrettyMIDI) -> list:
    """Return a flat list of all Note objects from every instrument track."""
    notes = []
    for inst in pm.instruments:
        notes.extend(inst.notes)
    return notes


def get_duration(pm: pretty_midi.PrettyMIDI) -> float:
    """Total duration of the MIDI file in seconds."""
    return pm.get_end_time()


def get_note_count(pm: pretty_midi.PrettyMIDI) -> int:
    """Total number of notes across all instrument tracks."""
    return sum(len(inst.notes) for inst in pm.instruments)


def get_note_density(pm: pretty_midi.PrettyMIDI) -> float:
    """Notes per second (averaged over the full duration)."""
    dur = get_duration(pm)
    if dur <= 0:
        return 0.0
    return get_note_count(pm) / dur


def get_pitch_range(pm: pretty_midi.PrettyMIDI) -> tuple[int, int]:
    """Return (min_pitch, max_pitch) as MIDI note numbers (0-127)."""
    notes = get_all_notes(pm)
    if not notes:
        return (0, 0)
    pitches = [n.pitch for n in notes]
    return int(min(pitches)), int(max(pitches))


def get_avg_pitch(pm: pretty_midi.PrettyMIDI) -> float:
    """Mean MIDI pitch across all notes."""
    notes = get_all_notes(pm)
    if not notes:
        return 0.0
    return float(np.mean([n.pitch for n in notes]))


def get_avg_velocity(pm: pretty_midi.PrettyMIDI) -> float:
    """Mean MIDI velocity across all notes."""
    notes = get_all_notes(pm)
    if not notes:
        return 0.0
    return float(np.mean([n.velocity for n in notes]))


def get_avg_note_duration(pm: pretty_midi.PrettyMIDI) -> float:
    """Mean note duration in seconds."""
    notes = get_all_notes(pm)
    if not notes:
        return 0.0
    return float(np.mean([n.end - n.start for n in notes]))


# ---------------------------------------------------------------------------
# Pitch class histogram and entropy  (Yang & Lerch, 2020)
# ---------------------------------------------------------------------------

def get_pitch_class_histogram(
    pm: pretty_midi.PrettyMIDI, normalize: bool = True
) -> np.ndarray:
    """
    Compute a 12-bin pitch class histogram (C=0 … B=11).
    Normalized to sum to 1.0 when normalize=True.
    """
    hist = np.zeros(12, dtype=float)
    for note in get_all_notes(pm):
        hist[note.pitch % 12] += 1.0
    if normalize and hist.sum() > 0:
        hist /= hist.sum()
    return hist


def pitch_class_entropy(pm: pretty_midi.PrettyMIDI) -> float:
    """
    Shannon entropy of the pitch class histogram (bits).

    Higher entropy = more equal pitch-class usage (less tonal focus).
    Lower entropy  = strong tonal centre or modal focus.
    Used as a cultural diversity indicator: Yang & Lerch (2020).

        H_PC = -sum_i ( p_i * log2(p_i) )   for p_i > 0
    """
    hist = get_pitch_class_histogram(pm, normalize=True)
    nonzero = hist[hist > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

def analyse_midi(path: str | Path) -> dict:
    """
    Load one MIDI file and return a flat dict of analysis metrics.
    Returns {'path': ..., 'error': 'load_failed'} if the file cannot be read.
    """
    pm = load_midi(path)
    if pm is None:
        return {"path": str(path), "error": "load_failed"}

    pitch_min, pitch_max = get_pitch_range(pm)
    return {
        "path"           : str(path),
        "duration_s"     : round(get_duration(pm), 2),
        "note_count"     : get_note_count(pm),
        "note_density"   : round(get_note_density(pm), 3),
        "pitch_min"      : pitch_min,
        "pitch_max"      : pitch_max,
        "pitch_range"    : pitch_max - pitch_min,
        "avg_pitch"      : round(get_avg_pitch(pm), 1),
        "avg_velocity"   : round(get_avg_velocity(pm), 1),
        "avg_note_dur_s" : round(get_avg_note_duration(pm), 3),
        "pc_entropy"     : round(pitch_class_entropy(pm), 4),
        "n_instruments"  : len(pm.instruments),
    }


# ---------------------------------------------------------------------------
# Piano roll visualisation
# ---------------------------------------------------------------------------

def piano_roll_plot(
    pm: pretty_midi.PrettyMIDI,
    ax,
    time_start: float = 0.0,
    time_end: float = 30.0,
    title: str = "",
    fs: int = 100,
) -> None:
    """
    Draw a piano roll for a time window of a piece.

    Parameters
    ----------
    pm         : PrettyMIDI object
    ax         : matplotlib Axes to draw on
    time_start : window start in seconds
    time_end   : window end in seconds
    title      : plot title string
    fs         : frames per second for piano roll resolution
    """
    roll = pm.get_piano_roll(fs=fs)
    start_frame = int(time_start * fs)
    end_frame   = int(min(time_end * fs, roll.shape[1]))
    roll_clip   = roll[:, start_frame:end_frame]

    ax.imshow(
        roll_clip,
        aspect="auto",
        origin="lower",
        cmap="Blues",
        interpolation="nearest",
    )
    ax.set_xlabel(f"Time (frames @ {fs} fps)")
    ax.set_ylabel("MIDI pitch (0-127)")

    # Light octave boundary lines
    for octave in range(11):
        ax.axhline(y=octave * 12, color="gray", linewidth=0.3, alpha=0.5)

    if title:
        ax.set_title(title, fontsize=9)
