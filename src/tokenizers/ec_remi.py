"""
EC-REMI: Extended Cultural REMI Tokenisation
=============================================
Central technical contribution of TP093781.

Extends standard REMI (Huang & Yang, 2020) with three additional token categories:

  1. Modal tokens     — Raga (Indian), Makam (Turkish), Mode/Key (Western, Irish)
  2. Microtonal tokens— Indian shruti-system offset (MicroOffset_I) and
                        Turkish 53-TET comma-system offset (MicroOffset_T).
                        These are two DISTINCT systems, never conflated.
  3. Ornament tokens  — Irish ornaments (Roll, Cut, Slide) detected from note
                        timing patterns; Turkish articulation (Vibrato, Trill).

Vocabulary layout
-----------------
  IDs 0 – (REMI_VOCAB_SIZE-1) : standard MiDiTok REMI tokens (unchanged)
  IDs REMI_VOCAB_SIZE + ...    : EC-REMI extension tokens

References
----------
  Huang & Yang (2020) "Pop Music Transformer" — REMI baseline.
  Yang & Lerch (2020) — objective MIR metrics.
  Arel (2006) "An overview of the Turkish maqam system" — 53-TET.
  Loy (2006) "Musimathics" Vol.1 Ch.5 — Indian shruti system.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi

# ---------------------------------------------------------------------------
# Import miditok safely: src/tokenizers/ conflicts with HuggingFace tokenizers.
# Solution: clear our package stub from sys.modules and sys.path before loading
# miditok, then restore sys.path.  HuggingFace tokenizers stays cached for
# subsequent miditok internal calls; our empty __init__.py is never needed.
# ---------------------------------------------------------------------------
_stale = [k for k in list(sys.modules) if k == "tokenizers" or
          (k.startswith("tokenizers.") and "ec_remi" not in k)]
for _k in _stale:
    sys.modules.pop(_k, None)

_src_paths = [p for p in sys.path if p == "src" or p.endswith("/src")]
for _p in _src_paths:
    sys.path.remove(_p)

from miditok import REMI, TokenizerConfig
from symusic import Score as _Score

for _p in _src_paths:
    sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CENTS_PER_COMMA_53TET = 1200.0 / 53          # ≈ 22.642 cents
CENTS_PER_SHRUTI_HALF = 1200.0 / (22 * 2)    # ≈ 27.3 cents — half a shruti step

# Traditions that receive MicroOffset tokens
MICRO_TRADITIONS = {"hindustani", "carnatic", "turkish_makam"}

# Traditions that receive Ornament tokens
ORNAMENT_TRADITIONS = {"irish_folk", "turkish_makam"}

# ---------------------------------------------------------------------------
# Vocabulary lists  (populated from actual dataset metadata)
# ---------------------------------------------------------------------------

HINDUSTANI_RAGAS = [
    "Abhogi", "Bahar", "Bairagi", "Basanti kedar", "Bhairabi", "Bhairav",
    "Bhatiyar", "Bhimpalas", "Bhoop", "Bibhas", "Bilaskhani todi",
    "Chandrakauns", "Dagori", "Des", "Dhani", "Gawti", "Hindol Pancham",
    "Jait Kalyan", "Jog", "Jogia", "Kalavati", "Kalyan", "Kedar", "Khamaj",
    "Khat", "Khokar", "Kirwani", "Komal rishabh asavari", "Lagan Gandhar",
    "Lalat", "Lalit pancham", "Maajh khamaj", "Madhukauns", "Malkauns",
    "Marubihag", "Marwa", "Megh", "Mishra kalingada", "Mishra piloo",
    "Miya malhar", "Nat Kamod", "Nat bhairav", "Puriya dhanashree",
    "Rageshri", "Ramdasi malhar", "Saraswati", "Shree", "Shuddh sarang",
    "Shuddha kalyan", "Sohini", "Todi", "Triveni gauri", "Yaman kalyan",
]

CARNATIC_RAGAS = [
    "Budham Aashrayami", "Dharini Telusukonti", "Dorakuna", "Gam Ganapate",
    "Gamakakriya", "Ganamuda Panam", "Gange Maam Pahi", "Gopi Gopala Bala",
    "Janakipathe", "Kailasapathe", "Kanakangaka Guha", "Karuna Ela Gante",
    "Lalitha Lavanga", "Neene Ballideno", "Nera Nammiti", "Paragu Matada",
    "RTP Andholika", "Sami Dayajuda", "Sarasamukhi Sakala",
    "Shloka Sri Ramachandra Shrita Parijata", "Shlokham", "Tamburi Mitidava",
    "Tillana", "Tolinenu Jesina", "Vinakayunna Della", "behag", "bhairavi",
    "chakravakam", "chittaranjani", "gambhira nata", "hindolam",
    "kambhoji", "karaharapriya", "madhyamavati", "mohanam", "sankarabharanam",
    "sindhubhairavi", "todi", "vasanta",
]

TURKISH_MAKAMS = [
    "acem", "acemasiran", "acembuselik", "acemkurdi", "araban", "arazbar",
    "asiranmaye", "askefza", "bendihisar", "besteisfahan", "bestenigar",
    "beyati", "beyati_araban", "beyati_arabanbuselik", "beyatibuselik",
    "buselik", "buselikasiran", "buzurk", "dilkeshaveran", "dilkeside",
    "dilkusa", "dilnisin", "dugah", "dugah_maye", "evic", "evicbuselik",
    "ferahfeza", "gerdaniye", "gerdaniyebuselik", "goncairana", "guldeste",
    "gulizar", "gulzar", "hicaz", "hicaz_humayun", "hicaz_uzzal",
    "hicaz_zemzeme", "hicaz_zirgule", "hicazasiran", "hicazbuselik",
    "hicazkar", "hicazkarkurdi", "hisarbuselik", "huseyni", "huzi",
    "huzzam", "huzzam_icedid", "isfahanek", "karcigar", "kucek",
    "kurdi_gerdaniye", "kurdilihicazkar", "mahur", "mahurbuselik",
    "maveraunnehr", "muhayyer", "muhayyerbuselik", "muhayyerkurdi",
    "muhayyersunbule", "murekkep_isfahan", "mustear", "neva", "neva_kurdi",
    "nevabuselik", "neveser", "nihavendikebir", "nihavendirumi", "nihavent",
    "nikriz", "nisabur", "nisaburek", "nisabureyn", "pencgah", "peykinesat",
    "rahatfeza", "rahatulervah", "rast", "rasticedid", "rastmaye", "rehavi",
    "rengidil", "revnaknuma", "reyya", "ruhnuvaz", "ruyidilara", "ruyiirak",
    "saba", "sabaasiran", "sababuselik", "sabazemzeme", "sazkar", "sedaraban",
    "segah", "segah_maye", "sehnaz", "sehnazbuselik", "selmek", "serefnuma",
    "sevk_icedid", "sevkefza", "sipihr", "sivenuma", "sultaniirak",
    "sultaniyegah", "suzidil", "suzidilara", "suzinak", "suzinak_zirgule",
    "tahir", "tarzinevin", "ussak", "vecdidil", "vechisehnaz", "yegah",
    "yeni_cargah", "zavil", "zirefkend",
]

IRISH_MODES = [
    "Adorian", "Amajor", "Aminor", "Amixolydian",
    "Bminor",
    "Ddorian", "Dmajor", "Dminor", "Dmixolydian",
    "Edorian", "Emajor", "Eminor",
    "Gmajor", "Gminor",
]

WESTERN_MODES = ["major", "minor", "unknown"]


# ---------------------------------------------------------------------------
# ECREMIVocabulary
# ---------------------------------------------------------------------------

class ECREMIVocabulary:
    """
    Builds and manages the EC-REMI vocabulary extension.

    The base REMI vocabulary is kept intact at IDs 0..REMI_VOCAB_SIZE-1.
    EC-REMI extension tokens start immediately after.
    """

    def __init__(self, remi_vocab: dict):
        # id_to_token / token_to_id for the *full* combined vocabulary
        self._base: dict[str, int] = dict(remi_vocab)
        self._ext: dict[str, int] = {}
        self._next_id = len(remi_vocab)
        self._build()

    # ---- Build EC-REMI extension tokens ----------------------------------

    def _add(self, token: str) -> None:
        if token not in self._base and token not in self._ext:
            self._ext[token] = self._next_id
            self._next_id += 1

    def _build(self) -> None:
        # 1. Modal tokens — Hindustani ragas
        for raga in HINDUSTANI_RAGAS:
            self._add(f"Modal_Raga_H_{raga.replace(' ', '_')}")

        # 2. Modal tokens — Carnatic ragas
        for raga in CARNATIC_RAGAS:
            self._add(f"Modal_Raga_C_{raga.replace(' ', '_')}")

        # 3. Modal tokens — Turkish makams
        for makam in TURKISH_MAKAMS:
            self._add(f"Modal_Makam_{makam}")

        # 4. Modal tokens — Irish modes
        for mode in IRISH_MODES:
            self._add(f"Modal_Mode_{mode}")

        # 5. Modal tokens — Western Classical
        for mode in WESTERN_MODES:
            self._add(f"Modal_WC_{mode}")

        # 6. Microtonal offset tokens — Indian shruti system
        for dev in (-2, -1, 0, 1, 2):
            self._add(f"MicroOffset_I_{dev:+d}")

        # 7. Microtonal offset tokens — Turkish 53-TET comma system
        for dev in (-2, -1, 0, 1, 2):
            self._add(f"MicroOffset_T_{dev:+d}")

        # 8. Ornament tokens — Irish folk
        for orn in ("Roll", "Cut", "Slide"):
            self._add(f"Ornament_{orn}")

        # 9. Ornament tokens — Turkish Makam articulation
        for orn in ("Vibrato", "Trill"):
            self._add(f"Ornament_{orn}")

        # 10. Unknown modal placeholder (fallback for unlabelled files)
        self._add("Modal_Unknown")

    # ---- Public interface -------------------------------------------------

    @property
    def full_vocab(self) -> dict[str, int]:
        """Combined REMI + EC-REMI token → id mapping."""
        return {**self._base, **self._ext}

    @property
    def ext_vocab(self) -> dict[str, int]:
        """EC-REMI extension tokens only."""
        return dict(self._ext)

    @property
    def remi_vocab_size(self) -> int:
        return len(self._base)

    @property
    def ext_vocab_size(self) -> int:
        return len(self._ext)

    @property
    def total_vocab_size(self) -> int:
        return self._next_id

    def token_to_id(self, token: str) -> int:
        if token in self._base:
            return self._base[token]
        if token in self._ext:
            return self._ext[token]
        return self._base.get("MASK_None", 0)

    def id_to_token(self, id_: int) -> str:
        rev = {v: k for k, v in self.full_vocab.items()}
        return rev.get(id_, "MASK_None")


# ---------------------------------------------------------------------------
# Utility: pitch-bend cents at a note's onset
# ---------------------------------------------------------------------------

def _build_pb_timeline(pm: pretty_midi.PrettyMIDI) -> tuple:
    """Pre-build sorted (times, cents) arrays for O(log n) lookup per note."""
    import bisect
    times, cents = [], []
    for inst in pm.instruments:
        for pb in inst.pitch_bends:
            times.append(pb.time)
            cents.append(pb.pitch / 8192.0 * 200.0)
    if not times:
        return [], []
    paired = sorted(zip(times, cents))
    return [p[0] for p in paired], [p[1] for p in paired]


def _pitch_bend_cents_at(times: list, cents: list, onset_s: float) -> float:
    """Binary-search the pre-built timeline for the pitch bend active at onset_s."""
    if not times:
        return 0.0
    import bisect
    idx = bisect.bisect_right(times, onset_s + 0.02) - 1
    return cents[idx] if idx >= 0 else 0.0


def _cents_to_comma_bin(cents: float, tradition: str) -> int:
    """
    Quantise pitch-bend cents to a discrete deviation bin in [-2, +2].

    Turkish Makam  — 53-TET: one comma ≈ 22.6 cents
    Indian (Hindustani / Carnatic) — shruti half-step ≈ 27.3 cents
    """
    if tradition == "turkish_makam":
        val = cents / CENTS_PER_COMMA_53TET
    else:
        val = cents / CENTS_PER_SHRUTI_HALF
    return int(max(-2, min(2, round(val))))


# ---------------------------------------------------------------------------
# Utility: Irish ornament detection from note timing patterns
# ---------------------------------------------------------------------------

def _detect_irish_ornaments(notes: list) -> dict[int, str]:
    """
    Scan a sorted note list for Irish ornamentation patterns.
    Returns {note_index: ornament_type} for detected ornaments.

    Heuristics (tuned for ABC-sourced MIDI at typical Irish tempos):
      Roll   — 3+ consecutive notes within 180 ms where first and last pitch
               are within 2 semitones (repeated-note figure with upper/lower neighbours)
      Cut    — a note < 70 ms that is a step above the immediately following note
      Slide  — 2+ ascending or descending semitone steps each < 100 ms
    """
    ornaments: dict[int, str] = {}
    n = len(notes)
    i = 0
    while i < n:
        note = notes[i]
        dur = note.end - note.start

        # --- Roll ---------------------------------------------------------
        if i + 2 < n:
            n2, n3 = notes[i + 1], notes[i + 2]
            window = n3.start - note.start
            if window < 0.18:
                # Check repeated-note pattern: start ≈ end pitch
                if abs(note.pitch - n3.pitch) <= 2:
                    ornaments[i] = "Roll"
                    i += 1
                    continue

        # --- Cut ----------------------------------------------------------
        if dur < 0.07 and i + 1 < n:
            n2 = notes[i + 1]
            gap = n2.start - note.end
            if gap < 0.02 and (note.pitch - n2.pitch) in (1, 2):
                ornaments[i] = "Cut"
                i += 1
                continue

        # --- Slide --------------------------------------------------------
        if dur < 0.10 and i + 1 < n:
            n2 = notes[i + 1]
            if n2.end - n2.start < 0.10:
                diff = n2.pitch - note.pitch
                if diff in (1, -1):  # chromatic step
                    # Confirm there's a landing note after
                    if i + 2 < n:
                        ornaments[i] = "Slide"
                        i += 1
                        continue

        i += 1

    return ornaments


# ---------------------------------------------------------------------------
# Utility: Turkish Makam vibrato/trill detection
# ---------------------------------------------------------------------------

def _detect_turkish_ornaments(notes: list) -> dict[int, str]:
    """
    Heuristics for Turkish Makam ornaments:
      Vibrato — note with many pitch-bend oscillations (handled via pitch-bend data,
                not note timing; approximated here as a longer note with high PB count)
      Trill   — rapid alternation between two pitches within 1-2 semitones
    """
    ornaments: dict[int, str] = {}
    n = len(notes)
    i = 0
    while i < n:
        # Trill: 4+ rapid alternations between two pitches
        if i + 3 < n:
            p = [notes[i + k].pitch for k in range(4)]
            dur_sum = notes[i + 3].start - notes[i].start
            if (
                dur_sum < 0.30
                and abs(p[0] - p[1]) <= 2
                and p[0] == p[2]
                and p[1] == p[3]
            ):
                ornaments[i] = "Trill"
                i += 1
                continue
        i += 1
    return ornaments


# ---------------------------------------------------------------------------
# ECREMITokenizer
# ---------------------------------------------------------------------------

class ECREMITokenizer:
    """
    EC-REMI tokeniser for the five musical traditions in TP093781.

    Usage
    -----
    >>> tok = ECREMITokenizer()
    >>> tokens = tok.tokenize("file.mid", tradition="turkish_makam",
    ...                        modal_label="hicaz")
    >>> ids = tok.encode(tokens)
    >>> print(tok.vocab_size)
    """

    def __init__(self) -> None:
        self._remi = REMI(TokenizerConfig())
        self._vocab = ECREMIVocabulary(self._remi.vocab)
        self._tok2id = self._vocab.full_vocab
        self._id2tok = {v: k for k, v in self._tok2id.items()}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._vocab.total_vocab_size

    @property
    def remi_vocab_size(self) -> int:
        return self._vocab.remi_vocab_size

    @property
    def ec_extension_size(self) -> int:
        return self._vocab.ext_vocab_size

    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._tok2id)

    # ------------------------------------------------------------------
    # Core tokenisation
    # ------------------------------------------------------------------

    def tokenize(
        self,
        midi_path: str | Path,
        tradition: str,
        modal_label: Optional[str] = None,
    ) -> list[str]:
        """
        Full EC-REMI tokenisation of one MIDI file.

        Parameters
        ----------
        midi_path    : path to MIDI file
        tradition    : one of 'western_classical', 'hindustani', 'carnatic',
                       'irish_folk', 'turkish_makam'
        modal_label  : raga name / makam name / mode string / key string
                       (used to look up the Modal token; falls back to Unknown)

        Returns
        -------
        List of EC-REMI token strings.
        """
        midi_path = Path(midi_path)

        # 1. Base REMI tokens
        try:
            seqs = self._remi.encode(_Score(str(midi_path)))
        except Exception:
            return []
        if not seqs:
            return []
        base_tokens: list[str] = seqs[0].tokens  # list of "Type_Value" strings

        # 2. Modal token (prepended)
        modal_token = self._resolve_modal_token(tradition, modal_label)

        # 3. Load MIDI for pitch-bend and ornament analysis
        pm: Optional[pretty_midi.PrettyMIDI] = None
        pb_times: list = []
        pb_cents: list = []
        if tradition in MICRO_TRADITIONS or tradition in ORNAMENT_TRADITIONS:
            try:
                pm = pretty_midi.PrettyMIDI(str(midi_path))
                if tradition in MICRO_TRADITIONS:
                    pb_times, pb_cents = _build_pb_timeline(pm)
            except Exception:
                pm = None

        # 4. Get all notes sorted by (onset, pitch) — same order as REMI
        notes_sorted: list = []
        if pm is not None:
            all_notes = [n for inst in pm.instruments for n in inst.notes]
            notes_sorted = sorted(all_notes, key=lambda n: (n.start, n.pitch))

        # 5. Ornament map {note_idx -> ornament_type}
        ornament_map: dict[int, str] = {}
        if pm is not None and tradition == "irish_folk":
            ornament_map = _detect_irish_ornaments(notes_sorted)
        elif pm is not None and tradition == "turkish_makam":
            ornament_map = _detect_turkish_ornaments(notes_sorted)

        # 6. Build EC-REMI sequence
        ec_tokens: list[str] = [modal_token]
        note_idx = 0

        for tok in base_tokens:
            ec_tokens.append(tok)

            if tok.startswith("Pitch_") and note_idx < len(notes_sorted):
                note = notes_sorted[note_idx]

                # 6a. Microtonal offset token (Indian / Turkish)
                if tradition in MICRO_TRADITIONS and pb_times:
                    cents = _pitch_bend_cents_at(pb_times, pb_cents, note.start)
                    dev = _cents_to_comma_bin(cents, tradition)
                    prefix = "MicroOffset_T_" if tradition == "turkish_makam" else "MicroOffset_I_"
                    ec_tokens.append(f"{prefix}{dev:+d}")

                # 6b. Ornament token (Irish / Turkish)
                if note_idx in ornament_map:
                    ec_tokens.append(f"Ornament_{ornament_map[note_idx]}")

                note_idx += 1

        return ec_tokens

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert token strings to integer IDs."""
        return [self._tok2id.get(t, self._tok2id.get("MASK_None", 0)) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """Convert integer IDs back to token strings."""
        return [self._id2tok.get(i, "MASK_None") for i in ids]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_modal_token(self, tradition: str, label: Optional[str]) -> str:
        """Map a free-text label to a canonical Modal_* token string."""
        if label is None:
            return "Modal_Unknown"

        label_clean = str(label).strip()

        if tradition == "hindustani":
            safe = label_clean.replace(" ", "_")
            tok = f"Modal_Raga_H_{safe}"
            return tok if tok in self._tok2id else "Modal_Unknown"

        if tradition == "carnatic":
            safe = label_clean.replace(" ", "_")
            tok = f"Modal_Raga_C_{safe}"
            return tok if tok in self._tok2id else "Modal_Unknown"

        if tradition == "turkish_makam":
            tok = f"Modal_Makam_{label_clean.lower()}"
            return tok if tok in self._tok2id else "Modal_Unknown"

        if tradition == "irish_folk":
            tok = f"Modal_Mode_{label_clean}"
            return tok if tok in self._tok2id else "Modal_Unknown"

        if tradition == "western_classical":
            key = label_clean.lower()
            if "minor" in key:
                tok = "Modal_WC_minor"
            elif "major" in key:
                tok = "Modal_WC_major"
            else:
                tok = "Modal_WC_unknown"
            return tok if tok in self._tok2id else "Modal_Unknown"

        return "Modal_Unknown"

    # ------------------------------------------------------------------
    # Vocabulary inspection helpers
    # ------------------------------------------------------------------

    def modal_tokens(self) -> list[str]:
        return [t for t in self._tok2id if t.startswith("Modal_")]

    def micro_tokens(self) -> list[str]:
        return [t for t in self._tok2id if t.startswith("MicroOffset_")]

    def ornament_tokens(self) -> list[str]:
        return [t for t in self._tok2id if t.startswith("Ornament_")]
