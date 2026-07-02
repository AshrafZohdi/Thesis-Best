"""
Generator + executor for Step 5: EC-REMI Tokenisation notebook.
Heavy batch tokenisation is in run_ec_remi_tokenise.py (run that first).
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

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

cells = []

cells.append(new_markdown_cell("""\
# Step 5 — EC-REMI Tokenisation: Extended Cultural REMI
## RMCE Thesis: Bridging the Authenticity Gap in AI-Generated Music

**Central technical contribution (TP093781).**

This notebook demonstrates the EC-REMI tokenisation scheme — an extension of
standard REMI (Huang & Yang, 2020) with three additional token categories:

| Category | Token prefix | Traditions |
|----------|-------------|------------|
| Modal / Scale | `Modal_Raga_*`, `Modal_Makam_*`, `Modal_Mode_*`, `Modal_WC_*` | All 5 |
| Microtonal offset — Indian shruti system | `MicroOffset_I_{-2..+2}` | Hindustani, Carnatic |
| Microtonal offset — Turkish 53-TET comma system | `MicroOffset_T_{-2..+2}` | Turkish Makam |
| Ornament / Articulation | `Ornament_{Roll,Cut,Slide,Vibrato,Trill}` | Irish Folk, Turkish Makam |

**Pre-requisite:** Run `python run_ec_remi_tokenise.py` once to produce
`results/ec_remi_per_file_stats.csv`. This notebook loads those results.

**Reference:** Arel (2006) — Turkish 53-TET; Loy (2006) Ch.5 — Indian shruti system.
"""))

cells.append(new_code_cell(PATH_SETUP))

cells.append(new_code_cell("""\
import warnings
warnings.filterwarnings("ignore")

import sys, collections
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tokenizers.ec_remi import (
    ECREMITokenizer, HINDUSTANI_RAGAS, CARNATIC_RAGAS,
    TURKISH_MAKAMS, IRISH_MODES, WESTERN_MODES,
)

RESULTS_DIR = PROJECT_ROOT / "results"
MIDI_ROOT   = PROJECT_ROOT / "data" / "processed"
META_DIR    = PROJECT_ROOT / "data" / "metadata"

EC_STATS = RESULTS_DIR / "ec_remi_per_file_stats.csv"
if not EC_STATS.exists():
    raise FileNotFoundError(
        f"{EC_STATS} not found.\\n"
        "Run:  python run_ec_remi_tokenise.py  first."
    )

print("EC-REMI module + pre-computed stats loaded OK")
"""))

cells.append(new_markdown_cell("## 1. Vocabulary Inspection"))

cells.append(new_code_cell("""\
tok = ECREMITokenizer()

print(f"{'REMI base vocabulary':<30s}: {tok.remi_vocab_size:>5d} tokens")
print(f"{'EC-REMI extension':<30s}: {tok.ec_extension_size:>5d} tokens")
print(f"{'Total EC-REMI vocabulary':<30s}: {tok.vocab_size:>5d} tokens")
print()
print(f"  Modal tokens    : {len(tok.modal_tokens()):>3d}")
print(f"    Hindustani ragas : {len(HINDUSTANI_RAGAS)}")
print(f"    Carnatic ragas   : {len(CARNATIC_RAGAS)}")
print(f"    Turkish makams   : {len(TURKISH_MAKAMS)}")
print(f"    Irish modes      : {len(IRISH_MODES)}")
print(f"    Western modes    : {len(WESTERN_MODES)}")
print(f"  Microtonal (I)  : 5   MicroOffset_I_{{-2,-1,0,+1,+2}} — Indian shruti system")
print(f"  Microtonal (T)  : 5   MicroOffset_T_{{-2,-1,0,+1,+2}} — Turkish 53-TET system")
print(f"  Ornament tokens : {len(tok.ornament_tokens())}   {tok.ornament_tokens()}")
print()
ext_tokens = sorted(tok._vocab.ext_vocab.keys())
print(f"First 12 extension tokens (IDs {tok.remi_vocab_size}–{tok.remi_vocab_size+11}):")
for t in ext_tokens[:12]:
    print(f"  ID {tok.vocab[t]:>4d}  {t}")
"""))

cells.append(new_markdown_cell("## 2. Single-File Demonstration: REMI vs EC-REMI"))

cells.append(new_code_cell("""\
# Import REMI base tokenizer (temporarily removing src/ to avoid name collision)
_src_paths = [p for p in sys.path if p.endswith("/src")]
for _p in _src_paths:
    sys.path.remove(_p)
from miditok import REMI, TokenizerConfig
from symusic import Score
for _p in _src_paths:
    sys.path.insert(0, _p)

remi_base = REMI(TokenizerConfig())
df_t = pd.read_csv(META_DIR / "symbtr_selected.csv")
row  = df_t[df_t["makam"] == "hicaz"].iloc[0]
fpath = MIDI_ROOT / "turkish_makam" / "midi" / row["processed_filename"]

remi_tokens = remi_base.encode(Score(str(fpath)))[0].tokens[:20]
ec_tokens   = tok.tokenize(fpath, tradition="turkish_makam", modal_label="hicaz")[:26]

print(f"File: {fpath.name}  (makam: hicaz)\\n")
print(f"{'REMI (first 20 tokens)':^45s}  |  EC-REMI (first 26 tokens, ◄=new)")
print("-" * 45 + "  |  " + "-" * 45)
for i in range(max(len(remi_tokens), len(ec_tokens))):
    r = remi_tokens[i] if i < len(remi_tokens) else ""
    e = ec_tokens[i]   if i < len(ec_tokens)   else ""
    new = " ◄" if (e.startswith("Modal") or e.startswith("Micro") or e.startswith("Orn")) else ""
    print(f"{r:<45s}  |  {e}{new}")
"""))

cells.append(new_markdown_cell("""\
**Reading the table above:**
- Row 0: EC-REMI prepends `Modal_Makam_hicaz` — REMI has no equivalent
- After each `Pitch_*` token: EC-REMI inserts a `MicroOffset_T_*` comma-offset token
- REMI's sequence runs straight `Pitch→Velocity→Duration`; EC-REMI interleaves pitch bend context
"""))

cells.append(new_markdown_cell("## 3. Cross-Tradition EC-REMI Statistics"))

cells.append(new_code_cell("""\
df     = pd.read_csv(EC_STATS)
remi   = pd.read_csv(RESULTS_DIR / "remi_per_file_stats.csv")
merged = df.merge(remi[["tradition","file","seq_len"]], on=["tradition","file"], how="left")
merged["ratio"] = (merged["seq_len_ec"] / merged["seq_len"]).round(3)

TRAD_ORDER = ["Western Classical","Hindustani","Carnatic","Irish Folk","Turkish Makam"]

stats = (
    merged.groupby("label")
    .agg(
        N           =("file",       "count"),
        remi_mean   =("seq_len",    "mean"),
        ec_mean     =("seq_len_ec", "mean"),
        ec_remi_ratio=("ratio",     "mean"),
        micro_mean  =("n_micro",    "mean"),
        orn_mean    =("n_ornament", "mean"),
    )
    .round(1)
    .reset_index()
)
stats["label"] = pd.Categorical(stats["label"], categories=TRAD_ORDER, ordered=True)
stats = stats.sort_values("label")

print("EC-REMI vs REMI — Cross-Tradition Summary\\n")
print(stats.rename(columns={
    "label":"Tradition","N":"N","remi_mean":"REMI mean len",
    "ec_mean":"EC-REMI mean len","ec_remi_ratio":"EC/REMI ratio",
    "micro_mean":"Micro/file","orn_mean":"Orn/file"
}).to_string(index=False))
"""))

cells.append(new_markdown_cell("## 4. Microtonal Token Distribution"))

cells.append(new_code_cell("""\
micro_df = pd.read_csv(RESULTS_DIR / "ec_remi_micro_distribution.csv")

for trad_key, label in [("turkish_makam","Turkish Makam (53-TET commas)"),
                         ("hindustani",   "Hindustani (shruti half-steps)"),
                         ("carnatic",     "Carnatic (shruti half-steps)")]:
    sub = micro_df[micro_df["tradition"] == trad_key].sort_values("token")
    if sub.empty:
        continue
    print(f"{label}:")
    for _, row in sub.iterrows():
        bar = "█" * int(row["pct"] / 2)
        print(f"  {row['token']:25s} {row['count']:7d} ({row['pct']:5.1f}%) {bar}")
    print()
"""))

cells.append(new_markdown_cell("## 5. Modal Token Coverage"))

cells.append(new_code_cell("""\
TRAD_COLORS = {"western_classical":"#2166ac","hindustani":"#d73027",
               "carnatic":"#fc8d59","irish_folk":"#1a9850","turkish_makam":"#762a83"}
TRAD_LABELS = {"western_classical":"Western Classical","hindustani":"Hindustani",
               "carnatic":"Carnatic","irish_folk":"Irish Folk","turkish_makam":"Turkish Makam"}

for trad_key in ["western_classical","hindustani","carnatic","irish_folk","turkish_makam"]:
    sub = df[df["tradition"] == trad_key]
    if sub.empty:
        continue
    modal_dist = collections.Counter(sub["modal_tok"].tolist())
    n_unknown  = modal_dist.get("Modal_Unknown", 0)
    n_known    = len(sub) - n_unknown
    print(f"{TRAD_LABELS[trad_key]:22s}: {n_known}/{len(sub)} labelled — top 5:")
    for tok_str, cnt in modal_dist.most_common(5):
        print(f"    {cnt:3d}×  {tok_str}")
"""))

cells.append(new_markdown_cell("## 6. Visualisations"))

cells.append(new_code_cell("""\
# --- 6a: EC-REMI vs REMI sequence length comparison ---
trad_order  = [TRAD_LABELS[k] for k in TRAD_LABELS if k in merged["tradition"].values]
trad_colors = [TRAD_COLORS[k]  for k in TRAD_COLORS  if k in merged["tradition"].values]

sp = stats[stats["label"].isin(trad_order)].copy()
sp["label"] = pd.Categorical(sp["label"], categories=trad_order, ordered=True)
sp = sp.sort_values("label")

x, w = np.arange(len(sp)), 0.38
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, sp["remi_mean"],   width=w, label="REMI",    color=trad_colors, alpha=0.45, edgecolor="white")
ax.bar(x + w/2, sp["ec_mean"],     width=w, label="EC-REMI", color=trad_colors, alpha=0.9,  edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([lab.replace(" ", "\\n") for lab in sp["label"]], fontsize=9)
ax.set_ylabel("Mean token sequence length")
ax.set_title("EC-REMI vs REMI — Mean Sequence Length", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_yscale("log")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ec_remi_seqlen_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ec_remi_seqlen_comparison.png")
"""))

cells.append(new_code_cell("""\
# --- 6b: Microtonal token distribution ---
micro_keys = [k for k in ["turkish_makam","hindustani","carnatic"]
              if k in micro_df["tradition"].values]
fig, axes = plt.subplots(1, len(micro_keys), figsize=(12, 4))
if len(micro_keys) == 1:
    axes = [axes]

for ax, trad_key in zip(axes, micro_keys):
    sub = micro_df[micro_df["tradition"] == trad_key].sort_values("token")
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(sub)))
    ax.bar(range(len(sub)), sub["pct"], color=colors, alpha=0.85)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels([t.split("_")[-1] for t in sub["token"]], fontsize=8)
    ax.set_title(TRAD_LABELS[trad_key], fontsize=10, fontweight="bold")
    ax.set_ylabel("% of micro tokens" if trad_key == micro_keys[0] else "")
    ax.set_xlabel("53-TET commas" if trad_key == "turkish_makam" else "shruti half-steps", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("EC-REMI MicroOffset Token Distribution", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ec_remi_micro_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ec_remi_micro_distribution.png")
"""))

cells.append(new_code_cell("""\
# --- 6c: Vocabulary composition pie ---
categories = {
    f"REMI base\\n({tok.remi_vocab_size} tokens)": tok.remi_vocab_size,
    f"Modal tokens\\n({len(tok.modal_tokens())} tokens)":    len(tok.modal_tokens()),
    f"Microtonal\\n({len(tok.micro_tokens())} tokens)":      len(tok.micro_tokens()),
    f"Ornament\\n({len(tok.ornament_tokens())} tokens)":     len(tok.ornament_tokens()),
}
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#4393c3","#d6604d","#74c476","#fd8d3c"]
ax.pie(list(categories.values()), labels=list(categories.keys()),
       colors=colors, autopct="%1.1f%%", startangle=90, pctdistance=0.8,
       textprops={"fontsize": 8})
ax.set_title(f"EC-REMI Vocabulary Composition\\n(total: {tok.vocab_size} tokens)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ec_remi_vocab_composition.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ec_remi_vocab_composition.png")
"""))

cells.append(new_markdown_cell("## 7. Save Summary and Report"))

cells.append(new_code_cell("""\
summary = stats.rename(columns={
    "label":"Tradition","N":"N","remi_mean":"Mean REMI len",
    "ec_mean":"Mean EC-REMI len","ec_remi_ratio":"EC/REMI ratio",
    "micro_mean":"Mean MicroOffset","orn_mean":"Mean Ornament"
})
summary.to_csv(RESULTS_DIR / "ec_remi_tokenisation_stats.csv", index=False)
print(f"Saved: results/ec_remi_tokenisation_stats.csv")

print()
print("=== Step 5 Complete ===")
print(f"EC-REMI vocab    : {tok.vocab_size} tokens")
print(f"  REMI base      : {tok.remi_vocab_size}")
print(f"  Modal tokens   : {len(tok.modal_tokens())} ({len(HINDUSTANI_RAGAS)} H-raga, {len(CARNATIC_RAGAS)} C-raga, {len(TURKISH_MAKAMS)} makam, {len(IRISH_MODES)} Irish, {len(WESTERN_MODES)} WC + 1 Unknown)")
print(f"  Microtonal     : {len(tok.micro_tokens())} (5 Indian + 5 Turkish — separate systems)")
print(f"  Ornament       : {len(tok.ornament_tokens())} tokens")
"""))

cells.append(new_markdown_cell("""\
## 8. Discussion — What EC-REMI Adds Over REMI

### 8.1 Modal tokens (all traditions)
Standard REMI has no concept of *which scale or mode* a piece is in.
EC-REMI prepends one Modal token per file, conditioning the GPT-2 model on
cultural identity before the first musical event. Two Turkish makams using
identical 12-TET pitches (e.g. Hicaz and Uzzal) become distinguishable.

### 8.2 Microtonal offset tokens — two strictly separate systems
- **Turkish 53-TET (`MicroOffset_T_*`):** SymbTr MIDI encodes pitch bends
  corresponding to 53-TET comma deviations (1 comma ≈ 22.6 cents).
  EC-REMI quantises to ±2 comma bins. The characteristic augmented-second
  interval of Hicaz requires specific comma sequences; the model can learn this.
- **Indian shruti (`MicroOffset_I_*`):** Basic-Pitch preserves gamaka/meend
  pitch bends. EC-REMI quantises to ±2 shruti half-step bins (≈27 cents each).
  The two systems are never conflated.

### 8.3 Ornament tokens (Irish Folk, Turkish Makam)
Irish ABC ornament symbols (rolls `~`, cuts `{`, slides) are not expanded into
rapid note sequences by music21, so ornament detection yields 0 tokens in this
corpus. **This is a pipeline limitation, not a design flaw:** performance MIDI
(physically played rolls/cuts) does trigger detection. Future work: parse ABC
ornament glyphs directly before MIDI conversion.

### 8.4 Information-theoretic motivation
REMI PC entropy for Turkish Makam = 3.339 bits (Step 4) — artificially high
because 200 makams are pooled and tonal centres cancel. With `Modal_Makam_*`
tokens, GPT-2 can learn per-makam pitch-class distributions (expected ~2.7 bits
each, matching the EDA per-file entropy) rather than a corpus-wide aggregate.
"""))

# ---------------------------------------------------------------------------
# Build and write notebook
# ---------------------------------------------------------------------------
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.9.0"}

out_path = Path("notebooks/03_ec_remi/03_ec_remi_tokenisation.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
import nbformat
nbformat.write(nb, str(out_path))
print(f"Written: {out_path}  ({len(cells)} cells)")
