"""
Generate thesis-ready evaluation plots from evaluation_results.csv.

Produces two figures:
  1. evaluation_metrics.png  — grouped bar charts (REMI vs EC-REMI) per metric
  2. evaluation_delta.png    — delta bar chart (EC-REMI minus REMI) per tradition

Usage:
  python music_transformer/plot_evaluation.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CSV  = Path(__file__).parent / "generated" / "evaluation_results.csv"
FIGS = Path(__file__).parent / "generated" / "figures"
FIGS.mkdir(exist_ok=True)

TRADITIONS = ["western_classical", "irish_folk", "turkish_makam", "hindustani", "carnatic"]
TRAD_LABELS = {
    "western_classical": "Western\nClassical",
    "irish_folk":        "Irish\nFolk",
    "turkish_makam":     "Turkish\nMakam",
    "hindustani":        "Hindustani",
    "carnatic":          "Carnatic",
}

METRICS = {
    "pitch_entropy":       "Pitch-Class Entropy (bits)",
    "note_density":        "Note Density (notes/s)",
    "pitch_range":         "Pitch Range (semitones)",
    "unique_bigram_ratio": "Unique Bigram Ratio",
    "ioi_entropy":         "IOI Entropy (bits)",
}

REMI_COLOR   = "#4C72B0"
ECREMI_COLOR = "#DD8452"

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":     150,
})


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    trad_order = {t: i for i, t in enumerate(TRADITIONS)}
    df["_o"] = df["tradition"].map(lambda t: trad_order.get(t, 99))
    return df.sort_values("_o").drop(columns="_o")


# ── Figure 1: grouped bars per metric ────────────────────────────────────────
def plot_metrics(df: pd.DataFrame):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 4.5))
    fig.suptitle("Objective Evaluation: REMI vs EC-REMI", fontsize=13, fontweight="bold", y=1.01)

    bar_w = 0.35

    for ax, (metric, ylabel) in zip(axes, METRICS.items()):
        # Traditions that have both tokenizers
        paired = [t for t in TRADITIONS
                  if len(df[(df["tradition"] == t)]) > 0]

        remi_vals   = []
        ecremi_vals = []
        labels      = []

        for trad in paired:
            sub = df[df["tradition"] == trad]
            r = sub[sub["tokenizer"] == "remi"]
            e = sub[sub["tokenizer"] == "ec_remi"]
            if r.empty:
                continue
            labels.append(TRAD_LABELS[trad])
            remi_vals.append(r.iloc[0][metric])
            ecremi_vals.append(e.iloc[0][metric] if not e.empty else float("nan"))

        x = np.arange(len(labels))
        ax.bar(x - bar_w/2, remi_vals,   bar_w, color=REMI_COLOR,   label="REMI",    zorder=3)
        ax.bar(x + bar_w/2, ecremi_vals, bar_w, color=ECREMI_COLOR, label="EC-REMI", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel.split(" (")[0], fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=REMI_COLOR,   label="REMI"),
        mpatches.Patch(color=ECREMI_COLOR, label="EC-REMI"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06), fontsize=10, frameon=False)

    fig.tight_layout()
    out = FIGS / "evaluation_metrics.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── Figure 2: delta bars (EC-REMI − REMI) ────────────────────────────────────
def plot_delta(df: pd.DataFrame):
    metrics_list  = list(METRICS.keys())
    metric_labels = [v.split(" (")[0] for v in METRICS.values()]

    # Only traditions that have both tokenizers
    paired_trads = [t for t in TRADITIONS
                    if not df[(df["tradition"] == t) & (df["tokenizer"] == "remi")].empty
                    and not df[(df["tradition"] == t) & (df["tokenizer"] == "ec_remi")].empty]

    n_trads   = len(paired_trads)
    n_metrics = len(metrics_list)
    bar_w     = 0.14
    x         = np.arange(n_metrics)

    cmap   = plt.cm.tab10
    colors = [cmap(i) for i in range(n_trads)]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("EC-REMI vs REMI: Metric Deltas (EC-REMI − REMI)",
                 fontsize=13, fontweight="bold")

    for i, trad in enumerate(paired_trads):
        r = df[(df["tradition"] == trad) & (df["tokenizer"] == "remi")].iloc[0]
        e = df[(df["tradition"] == trad) & (df["tokenizer"] == "ec_remi")].iloc[0]
        deltas = [e[m] - r[m] for m in metrics_list]
        offset = (i - n_trads / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, deltas, bar_w, color=colors[i],
                      label=TRAD_LABELS[trad].replace("\n", " "), zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Delta (EC-REMI − REMI)", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False, bbox_to_anchor=(1.01, 0.5), loc="center left")

    # Shade positive/negative regions lightly
    ymin, ymax = ax.get_ylim()
    ax.axhspan(0, ymax, alpha=0.04, color="green", zorder=0)
    ax.axhspan(ymin, 0, alpha=0.04, color="red",   zorder=0)
    ax.text(0.01, 0.97, "← EC-REMI better", transform=ax.transAxes,
            fontsize=8, color="green", alpha=0.7, va="top")
    ax.text(0.01, 0.03, "← REMI better",    transform=ax.transAxes,
            fontsize=8, color="red",   alpha=0.7, va="bottom")

    fig.tight_layout()
    out = FIGS / "evaluation_delta.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    df = load()
    plot_metrics(df)
    plot_delta(df)
    print("Done.")
