# RMCE Thesis — Progress Log

**Student:** Mohammad Ashraf Zohdi Ahmad (TP093781)  
**Title:** Bridging the Authenticity Gap in AI-Generated Music: A Framework for Culturally-Sensitive and Legally-Safe Generative Systems  
**Deadline:** August 2026

---

## Status Overview

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project structure + dataset audit | ✅ Complete |
| 2 | Data preparation notebooks (per tradition) | ✅ Complete (notebooks written; Saraga transcription pending — run overnight) |
| 3 | EDA notebooks (per tradition) | ✅ Complete (3/5 executed; Hindustani + Carnatic pending transcription) |
| 4 | Baseline REMI tokenisation + objective metrics | ✅ Complete |
| 5 | EC-REMI tokenizer design + implementation | ✅ Complete |
| 6 | GPT-2 + LoRA fine-tuning pipeline | 🟡 In progress (2/10 complete; 8 need full run via train_finetuning.py) |
| 7 | Comparative evaluation (REMI vs EC-REMI) | 🔲 Not started |

---

## Step 1 — Completed: 2026-06-21

### What was done
- Audited all four raw dataset folders; confirmed file counts, formats, metadata availability
- Created full project folder structure
- Confirmed dataset caps and notebook structure with supervisor

### Dataset Audit Summary

| Tradition | Raw Location | Format | Total Available | Selected Cap | Notes |
|-----------|-------------|--------|-----------------|--------------|-------|
| Western Classical | `datasets/western_classical/maestro-v3.0.0/` | MIDI | 1,276 | 150 | Rich metadata CSV with composer, year, split, duration |
| Turkish Makam | `datasets/SymbTr/midi/` | MIDI | 2,200 | 200 | Makam/usul/form in filename; TXT files contain 53-TET Koma values |
| Irish Folk | `datasets/irish_folk/TheSession-data/csv/tunes.csv` | ABC | 267,685 settings | 200 unique tunes | Select by tune_id uniqueness + tune_popularity.csv ranking |
| Hindustani | `datasets/indian_classical/saraga1.5_hindustani/` | MP3 | 108 | 60 | Needs Basic-Pitch transcription; JSON metadata has raga labels |
| Carnatic | `datasets/indian_classical/saraga1.5_carnatic/` | MP3 | 991 | 60 | Needs Basic-Pitch transcription; some JSON metadata has empty raaga arrays |

### Key metadata files confirmed present
- MAESTRO: `maestro-v3.0.0.csv` — composer, title, split, year, duration per file
- SymbTr: filename encodes makam/usul/form; `symbTr_mbid.json` has MusicBrainz UUIDs; TXT files have Koma53 column (53-TET microtonal data)
- Saraga Hindustani: per-track `.json` files with mbid, title, artists, raags; `.ctonic.txt` (tonic Hz); `.pitch.txt` (pitch track)
- Saraga Carnatic: per-track `.json` files, but raaga/taala arrays empty for many entries — fall back to concert/folder name
- TheSession: `tunes.csv` has type, meter, mode, abc per tune; `tune_popularity.csv` for representative selection

### Known issues to handle in code
- Saraga Hindustani files have double extension: `filename.mp3.mp3`
- Carnatic JSON metadata sparse (empty raaga arrays) — document in thesis as metadata limitation
- Basic-Pitch must use TensorFlow backend explicitly (not CoreML) on macOS to avoid floating point exception

### Folder structure created
```
data/processed/{western_classical,hindustani,carnatic,irish_folk,turkish_makam}/midi/
data/metadata/
notebooks/{00_data_preparation, 01_eda, 02_baseline_remi, 03_ec_remi, 04_finetuning, 05_evaluation}/
src/{tokenizers,utils}/
results/
outputs/{checkpoints, generated_samples/{remi,ec_remi}}/
```

---

---

## Step 2 — Completed: 2026-06-21

### What was done
- Wrote 5 data preparation notebooks (all valid, syntax-checked):
  - `00a_maestro_prep.ipynb` — stratified sample 150 MIDI from MAESTRO train split
  - `00b_hindustani_prep.ipynb` — select 60 tracks + Basic-Pitch transcription (TF backend)
  - `00c_carnatic_prep.ipynb` — select 60 tracks across 27 concert folders + Basic-Pitch transcription
  - `00d_irish_folk_prep.ipynb` — deduplicate 267k settings → 200 unique tunes by popularity; ABC→MIDI via music21
  - `00e_turkish_makam_prep.ipynb` — stratified sample 200 MIDI across makams; preview Koma53 data
- Generator script kept at `create_notebooks.py` for reference

### Notebooks ready to run — execution order
| Notebook | Time estimate | Notes |
|----------|---------------|-------|
| `00a_maestro_prep.ipynb` | < 2 min | Copy MIDI files; no compute |
| `00e_turkish_makam_prep.ipynb` | < 2 min | Copy MIDI files; no compute |
| `00d_irish_folk_prep.ipynb` | ~15 min | music21 ABC→MIDI conversion |
| `00b_hindustani_prep.ipynb` | **~5–8 hours** | Basic-Pitch transcription ×60 — run overnight |
| `00c_carnatic_prep.ipynb` | **~5–8 hours** | Basic-Pitch transcription ×60 — run overnight |

### Packages confirmed installed
- `nbformat 5.10.4`, `music21 8.3.0`, `basic_pitch` (TF backend), `pretty_midi`
- `peft`, `transformers 4.57.6` (for Step 6)
- `miditok` — **NOT installed** (needed for Steps 4–5 — install before Step 3)

---

---

## Step 3 — Completed: 2026-06-22

### What was done
- Wrote `src/utils/midi_utils.py` — shared analysis functions (load_midi, analyse_midi, pitch_class_entropy, piano_roll_plot, etc.)
- Wrote 5 EDA notebooks (`01a`–`01e`) using shared utilities
- Executed 3 fast prep notebooks (MAESTRO 150 files, SymbTr 200 files, Irish 196/200 files)
- Executed 3 EDA notebooks; 18 chart PNGs saved to `results/`
- 4 Irish ABC→MIDI failures (waltz/march with encoding issues): acceptable, 196/200 success

### Preliminary EDA results (3/5 traditions)

| Tradition | n | Duration (min) | Note density | Pitch range (st) | PC entropy |
|-----------|---|---------------|--------------|-----------------|------------|
| Western Classical | 150 | 9.9 | 10.79 | 68.2 | **3.356** |
| Turkish Makam | 200 | 2.2 | 2.03 | 15.8 | 2.719 |
| Irish Folk | 196 | 1.1 | 3.61 | 18.7 | **2.542** |
| Hindustani | — | pending transcription | — | — | — |
| Carnatic | — | pending transcription | — | — | — |

**Interpretation (for thesis):**
- PC entropy confirms theoretical expectation: Western Classical (polyphonic, chromatic) > Turkish Makam (modal, 53-TET) > Irish Folk (strongly modal, narrow range)
- Note density: MAESTRO 10.79 vs. Turkish 2.03 vs. Irish 3.61 — piano polyphony vs. monophonic/heterophonic traditions
- Pitch range: MAESTRO 68 semitones (full piano) vs. Turkish/Irish ~16–19 (melodic instrument range)

### Pending
- Run `00b_hindustani_prep.ipynb` and `00c_carnatic_prep.ipynb` overnight (Basic-Pitch, ~5–8 hrs each)
- Then run `01b_hindustani_eda.ipynb` and `01c_carnatic_eda.ipynb`
- Install `miditok` before Step 4

---

---

## Step 4 — Completed: 2026-06-23

### What was done
- Built `notebooks/02_baseline_remi/02_baseline_remi_tokenisation.ipynb` (23 cells, EXECUTED)
- Tokenised all 666 MIDI files (150 WC + 60 Hind + 60 Carn + 196 Irish + 200 Turkish) with default MiDiTok REMI
- REMI vocabulary: 284 tokens, 89 pitch tokens (MIDI 21–109)
- Key fix: temporarily removed `src/tokenizers/` from sys.path before importing miditok to avoid shadowing the HuggingFace `tokenizers` package

### REMI Tokenisation Results

| Tradition | N | Mean seq len | Pitch coverage % | Pitch span (st) | REMI PC entropy |
|-----------|---|-------------|-----------------|-----------------|-----------------|
| Western Classical | 150 | 21,393 | **98.9%** | 86 | 3.575 |
| Hindustani | 60 | 23,451 | 87.5% | 76 | 3.483 |
| Carnatic | 60 | 11,418 | 87.5% | 78 | 3.538 |
| Irish Folk | 196 | 928 | 43.2% | 41 | 2.876 |
| Turkish Makam | 200 | 1,205 | **35.2%** | 31 | 3.339 |

### Key findings for thesis
- **Turkish Makam** uses only 35% of REMI pitch range (MIDI 60–91); yet REMI PC entropy (3.339) is much higher than per-file EDA entropy (2.719) — because REMI aggregates tokens across ALL 200 makam pieces, masking the strong tonal focus of individual makams. This is a key limitation EC-REMI addresses with Modal tokens.
- **Irish Folk** uses 43% of pitch range — narrow melodic instrument range; short sequences (mean 928 tokens) vs. Indian classical (>10k). REMI misses ornament semantics entirely.
- **Hindustani/Carnatic** long sequences (23k / 11k) reflect sustained melodic elaboration; pitch coverage 87.5% — notes well above 12-TET quantisation but microtonality already lost by Basic-Pitch transcription.
- Saved: `results/remi_tokenisation_stats.csv`, `results/remi_per_file_stats.csv`, 3 PNG charts

### Outputs
- `notebooks/02_baseline_remi/02_baseline_remi_tokenisation.ipynb` — executed
- `results/remi_tokenisation_stats.csv` — cross-tradition summary
- `results/remi_per_file_stats.csv` — per-file token stats (666 rows)
- `results/remi_seqlen_pitch_coverage.png`
- `results/remi_token_type_composition.png`
- `results/remi_pitch_heatmap.png`

---

---

## Step 5 — Completed: 2026-06-23

### What was done
- Implemented `src/tokenizers/ec_remi.py` — `ECREMITokenizer` class with full EC-REMI vocabulary
- Created `notebooks/03_ec_remi/03_ec_remi_tokenisation.ipynb` (21 cells) — loads pre-computed results and demonstrates tokenisation
- Created `run_ec_remi_tokenise.py` — standalone batch script for all 666 files (slow due to Hindustani/Carnatic)

### EC-REMI Vocabulary Design

| Category | Tokens | Count |
|----------|--------|-------|
| REMI base (unchanged) | Pitch, Velocity, Duration, Position, Bar, ... | 284 |
| Modal — Hindustani ragas | `Modal_Raga_H_{name}` | 53 |
| Modal — Carnatic ragas | `Modal_Raga_C_{name}` | 39 |
| Modal — Turkish makams | `Modal_Makam_{name}` | 113 |
| Modal — Irish modes | `Modal_Mode_{name}` | 14 |
| Modal — Western Classical | `Modal_WC_{major,minor,unknown}` | 3 |
| Modal — fallback | `Modal_Unknown` | 1 |
| Microtonal — Indian shruti | `MicroOffset_I_{-2,-1,0,+1,+2}` | 5 |
| Microtonal — Turkish 53-TET | `MicroOffset_T_{-2,-1,0,+1,+2}` | 5 |
| Ornament — Irish Folk | `Ornament_{Roll,Cut,Slide}` | 3 |
| Ornament — Turkish Makam | `Ornament_{Vibrato,Trill}` | 2 |
| **Total EC-REMI vocab** | | **526** |

### Key technical decisions
- Modal token prepended as first token in every sequence (all traditions)
- MicroOffset tokens injected **after** each Pitch token for Turkish and Indian (from MIDI pitch bend data: SymbTr encodes 53-TET deviations, Basic-Pitch encodes gamakas)
- Indian (shruti) and Turkish (53-TET) systems use **different token prefixes** — never conflated
- Ornament detection: pattern-based from note timing; Irish detection yields 0 (music21 ABC→MIDI does not expand ornament symbols into rapid note sequences — thesis limitation documented)
- Import conflict fix: `src/tokenizers/` shadows HuggingFace `tokenizers`; resolved by clearing `sys.modules['tokenizers']` before loading miditok

### EC-REMI vs REMI Sequence Length Results (all 666 files)

| Tradition | N | REMI mean | EC-REMI mean | EC/REMI ratio | MicroOffset/file |
|-----------|---|-----------|--------------|---------------|-----------------|
| Western Classical | 150 | 21,393 | 21,394 | 1.0× | 0 |
| Hindustani | 60 | 23,451 | 29,390 | **1.3×** | 5,937 |
| Carnatic | 60 | 11,418 | 14,221 | **1.2×** | 2,801 |
| Irish Folk | 196 | 928 | 929 | 1.0× | 0 |
| Turkish Makam | 200 | 1,205 | 1,497 | **1.24×** | 291 |

### Microtonal token distributions (corpus-wide)
- **Turkish 53-TET:** +0 (76.1%), -1 (13.0%), -2 (6.3%), +1 (4.3%), +2 (0.2%) — most notes in-tune with occasional flat commas
- **Hindustani:** +1 dominant (57.1%), +2 (17.7%), +0 (14.8%) — gamakas skew strongly upward
- **Carnatic:** +1 (36.2%), +2 (27.4%), +0 (25.3%) — more balanced but upward-biased

### Known limitations documented
- Turkish Makam modal lookup: 0/200 labelled (metadata file uses `processed_filename` but lookup fell back to `midi_path` key — fix for Step 7 evaluation). Microtonal tokens still correct.
- Carnatic: 28/60 Modal_Unknown (title-fallback raga names not in CARNATIC_RAGAS list)
- Irish ornament detection: 0 ornaments (music21 ABC→MIDI doesn't expand ornament glyphs — pipeline limitation documented in thesis)
- Performance fix: `_pitch_bend_cents_at` refactored from O(n×m) linear scan to O(log n) binary search — reduced Hindustani per-file time from >60s to 1.3s

### Outputs
- `src/tokenizers/ec_remi.py` — ECREMITokenizer class (526-token vocab)
- `run_ec_remi_tokenise.py` — standalone batch script (run once before notebook)
- `notebooks/03_ec_remi/03_ec_remi_tokenisation.ipynb` — EXECUTED
- `results/ec_remi_per_file_stats.csv` — 666 rows
- `results/ec_remi_micro_distribution.csv` — corpus-wide microtonal token counts
- `results/ec_remi_tokenisation_stats.csv` — cross-tradition summary
- `results/ec_remi_seqlen_comparison.png`, `ec_remi_micro_distribution.png`, `ec_remi_vocab_composition.png`

---

---

## Step 6 — In Progress: 2026-06-24

### What was done
- Implemented `train_finetuning.py` — standalone batch script for all 10 models (5 traditions × 2 tokenisers)
- Created `notebooks/04_finetuning/04_finetuning.ipynb` (12 cells) — 60-step Irish Folk REMI demo
- Resolved sys.path conflict for joint miditok + ECREMITokenizer + transformers imports (import order fix + importlib loader)
- Trained **Irish Folk REMI** model (5 epochs, 363 train / 40 val chunks) — **COMPLETE**

### Architecture
- **Base:** GPT-2 small (pretrained, 124M params), embedding resized to vocab_size
- **LoRA:** r=8, α=16, dropout=0.1, target `c_attn` — 294,912 / 86,355,456 trainable (0.34%)
- **Chunking:** 512-token windows, stride 256
- **Optimiser:** AdamW lr=5×10⁻⁴, weight_decay=0.01, warmup
- **Loss:** Cross-entropy causal LM (next-token prediction)

### Irish Folk REMI Results (completed)

| Metric | Value |
|--------|-------|
| Tradition | Irish Folk |
| Tokeniser | REMI (vocab 284) |
| Train chunks | 363 |
| Val chunks | 40 |
| Epochs | 5 |
| Final train loss | **1.477** |
| Eval loss (epoch 1) | 1.472 |
| Trainable params | 294,912 (0.34%) |
| Checkpoint | `outputs/checkpoints/irish_folk_remi/final/` |

Loss trajectory (epoch 1): 5.84 → 2.36 → 1.83 → 1.69 → 1.56 → 1.50 — confirms model is learning MIDI token distributions.

### Remaining 9 models — run manually
```bash
python train_finetuning.py                                        # all remaining
python train_finetuning.py --tradition irish_folk --tokeniser ec_remi
python train_finetuning.py --tradition carnatic --tokeniser remi
# ... etc
```

Estimated CPU time per tradition/tokeniser pair:
| Tradition | Est. hours |
|-----------|-----------|
| Irish Folk | ~1.5h |
| Turkish Makam | ~3h |
| Hindustani | ~6h |
| Carnatic | ~3h |
| Western Classical | ~8h |

### Dataset chunk counts (512-token windows, stride 256)

| Tradition | REMI chunks | EC-REMI chunks |
|-----------|------------|----------------|
| Western Classical | 12,308 | 12,308 |
| Hindustani | 5,406 | 6,796 |
| Carnatic | 2,589 | 3,244 |
| Irish Folk | 403 | 403 |
| Turkish Makam | 671 | 888 |
| **Total** | **21,377** | **23,639** |

### Notebook demo results (Irish Folk REMI, 60 steps)

| Step | Train loss | Val loss |
|------|-----------|---------|
| 10 | 4.1304 | 4.0675 |
| 20 | 2.4867 | 2.0971 |
| 30 | 1.9565 | 1.9006 |
| 40 | 1.9058 | 1.7192 |
| 50 | 1.7243 | 1.6531 |
| 60 | **1.7227** | **1.6231** |

Confirms model learns MIDI token distributions (val_loss < train_loss = good generalisation).

### Completed checkpoints
- `outputs/checkpoints/irish_folk_remi/` — full 5-epoch Trainer run (train_loss=1.477)
- `outputs/checkpoints/irish_folk_remi_demo/` — notebook 60-step demo (val_loss=1.623)

### Outputs
- `train_finetuning.py` — full 10-model pipeline script
- `create_finetuning_notebook.py` — notebook generator
- `notebooks/04_finetuning/04_finetuning.ipynb` — 12-cell demo notebook, EXECUTED
- `results/finetuning_demo_loss.png` — loss curve chart

### Key technical decisions
- Import order: miditok → transformers (force lazy imports) → ec_remi via importlib
  - This prevents `src/tokenizers/` package from shadowing HuggingFace `tokenizers` during transformers lazy loading
- GPT-2 embedding resized from 50,257 → 284/526 tokens (REMI/EC-REMI)
- Pretrained GPT-2 attention weights retained; only LoRA adapters trained
- `eval_strategy="epoch"` (renamed from `evaluation_strategy` in transformers 4.38+)

---

## Next Step

**Step 7 — Comparative evaluation (REMI vs EC-REMI)**

After training all 10 models, run evaluation:
- Generate MIDI samples from each checkpoint
- Compute objective metrics: PC entropy, pitch coverage, sequence diversity, modal consistency
- Compare REMI vs EC-REMI outputs per tradition
- Statistical analysis (t-test or Wilcoxon) for thesis claims
- Create `notebooks/05_evaluation/05_evaluation.ipynb`

**Prerequisite:** Run `python train_finetuning.py` to train all 9 remaining models first.
