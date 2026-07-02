"""
Standalone script: EC-REMI tokenisation of all 666 processed MIDI files.
Saves per-file results to results/ec_remi_per_file_stats.csv.

Run once before opening the Step 5 notebook:
    python run_ec_remi_tokenise.py
"""

import sys, warnings, collections
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import pandas as pd
from pathlib import Path
from tokenizers.ec_remi import ECREMITokenizer

MIDI_ROOT = Path("data/processed")
META_DIR  = Path("data/metadata")
RESULTS   = Path("results")
RESULTS.mkdir(exist_ok=True)
OUT_FILE  = RESULTS / "ec_remi_per_file_stats.csv"

TRADITIONS = {
    "western_classical": {"label": "Western Classical", "meta_csv": "maestro_selected.csv",
                          "label_col": None, "modal_default": "unknown"},
    "hindustani":        {"label": "Hindustani",        "meta_csv": "hindustani_tracks.csv",
                          "label_col": "raga"},
    "carnatic":          {"label": "Carnatic",           "meta_csv": "carnatic_tracks.csv",
                          "label_col": "raga"},
    "irish_folk":        {"label": "Irish Folk",         "meta_csv": "irish_folk_tunes.csv",
                          "label_col": "mode"},
    "turkish_makam":     {"label": "Turkish Makam",      "meta_csv": "symbtr_selected.csv",
                          "label_col": "makam"},
}

tok = ECREMITokenizer()
print(f"EC-REMI vocab: {tok.vocab_size} tokens ({tok.remi_vocab_size} REMI + {tok.ec_extension_size} ext)")

records = []
micro_counters = collections.defaultdict(collections.Counter)

for trad_key, tinfo in TRADITIONS.items():
    midi_dir = MIDI_ROOT / trad_key / "midi"
    files = sorted(list(midi_dir.glob("*.midi")) + list(midi_dir.glob("*.mid")))
    if not files:
        print(f"  {tinfo['label']}: no MIDI files")
        continue

    meta_path = META_DIR / tinfo["meta_csv"]
    label_map = {}
    if tinfo.get("label_col") and meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        midi_col = "midi_path" if "midi_path" in meta_df.columns else "processed_filename"
        if midi_col in meta_df.columns and tinfo["label_col"] in meta_df.columns:
            for _, row in meta_df.iterrows():
                fname = Path(str(row[midi_col])).name
                val = row[tinfo["label_col"]]
                label_map[fname] = str(val) if pd.notna(val) else None

    ok, err = 0, 0
    for fpath in files:
        modal_label = label_map.get(fpath.name, tinfo.get("modal_default"))
        try:
            tokens = tok.tokenize(fpath, tradition=trad_key, modal_label=modal_label)
            if not tokens:
                err += 1
                continue

            n_micro = sum(1 for t in tokens if t.startswith("MicroOffset"))
            n_orn   = sum(1 for t in tokens if t.startswith("Ornament"))
            modal_tok = tokens[0] if tokens[0].startswith("Modal") else "Modal_Unknown"

            for t in tokens:
                if t.startswith("MicroOffset"):
                    micro_counters[trad_key][t] += 1

            records.append({
                "tradition":  trad_key,
                "label":      tinfo["label"],
                "file":       fpath.name,
                "modal_tok":  modal_tok,
                "seq_len_ec": len(tokens),
                "n_micro":    n_micro,
                "n_ornament": n_orn,
            })
            ok += 1
        except Exception as e:
            err += 1

    print(f"  {tinfo['label']:22s}: {ok} ok, {err} err")

df = pd.DataFrame(records)
df.to_csv(OUT_FILE, index=False)
print(f"\nSaved {len(df)} rows → {OUT_FILE}")

# Also save microtonal distribution summary
micro_rows = []
for trad_key, counter in micro_counters.items():
    total = sum(counter.values())
    for token, cnt in sorted(counter.items()):
        micro_rows.append({"tradition": trad_key, "token": token,
                           "count": cnt, "pct": round(cnt / total * 100, 2)})
micro_df = pd.DataFrame(micro_rows)
micro_df.to_csv(RESULTS / "ec_remi_micro_distribution.csv", index=False)
print(f"Saved microtonal distribution → results/ec_remi_micro_distribution.csv")
