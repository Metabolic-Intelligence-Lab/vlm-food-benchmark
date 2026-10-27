"""Paper figures (Figs. 4-7) from the official configuration (clean mode for the
first campaign + metrics_v2_all for the 2026 campaign), saved in
analysis/outputs/figures/:
  f1_mosaic_open_{1,2}.png/pdf     T x eps heatmaps, open models, split in two
  nmae_mosaic_open_{1,2}.png/pdf
  f1_mosaic_proprietary.png/pdf    gpt-4o/4.1/5.2 (T axis) + gpt-5.5/claude (replicas)
  nmae_mosaic_proprietary.png/pdf

Usage: python analysis/14_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import OUTPUTS_DIR, THRESHOLDS

PLOTS = os.path.join(OUTPUTS_DIR, "figures")
os.makedirs(PLOTS, exist_ok=True)

TEMPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
OPENAI_OLD = ["gpt-4.1", "gpt-4o", "gpt-5.2"]
PROP_NEW = ["gpt-5.5", "claude-sonnet-5"]


def load_official():
    """DataFrame model/temperature/run/epsilon/f1/nmae for both campaigns."""
    clean = json.load(open(os.path.join(OUTPUTS_DIR, "metrics_clean.json")))
    rows = []
    for v in clean.values():
        rows.append({"model": v["model"], "temperature": float(v["temperature"]),
                     "run": "r1", "epsilon": v["epsilon"], "f1": v["f1"],
                     "nmae": v["nmae_legacy"]})
    v2 = pd.read_csv(os.path.join(OUTPUTS_DIR, "metrics_v2_all.csv"))
    for r in v2.itertuples():
        rows.append({"model": r.model, "temperature": r.temperature,
                     "run": r.run, "epsilon": r.epsilon, "f1": r.f1,
                     "nmae": r.nmae_legacy})
    df = pd.DataFrame(rows)
    # first campaign models keep their r1; replicate runs are used only for
    # the proprietary replicate panels and spreads
    return df


def mosaic(df, models, value, fname, vmin, vmax, cmap, cbar_label, fmt,
           x_field="temperature", x_values=TEMPS, x_label="Temperature"):
    """Reproduces plot_performance_2D.py's panel style exactly."""
    n = len(models)
    ncols = min(5, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    for i, model in enumerate(models):
        ax = axes[i]
        sub = df[df.model == model]
        hm = sub.pivot_table(index="epsilon", columns=x_field, values=value,
                             aggfunc="mean") \
                .reindex(index=sorted(THRESHOLDS, reverse=True), columns=x_values)
        if value == "nmae":
            hm = hm.round(0)
        sns.heatmap(hm, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, annot=True,
                    fmt=fmt, cbar=(i == n - 1),
                    cbar_ax=cbar_ax if i == n - 1 else None,
                    cbar_kws={"label": cbar_label})
        ax.set_title(model, fontsize=15)
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel("CrossEnc Threshold", fontsize=13)
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    cbar_ax.set_ylabel(cbar_label, fontsize=15)
    cbar_ax.tick_params(labelsize=12)
    plt.subplots_adjust(left=0.07, right=0.9, top=0.93, bottom=0.07,
                        wspace=0.3, hspace=0.35)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PLOTS, f"{fname}.{ext}"),
                    dpi=300 if ext == "png" else None,
                    bbox_inches="tight" if ext == "pdf" else None)
    plt.close(fig)
    print(f"salvata {fname} ({n} pannelli)")


def version_a(df):
    open_models = sorted(m for m in df.model.unique()
                         if not m.startswith("gpt") and m != "claude-sonnet-5")
    half = -(-len(open_models) // 2)
    for value, vmin, vmax, cmap, lab, fmt in [
            ("f1", 0, 1, "inferno", "F1 Score", ".2f"),
            ("nmae", 60, 125, "inferno_r", "NMAE [g]", ".0f")]:
        r1 = df[df.run == "r1"]
        mosaic(r1, open_models[:half], value,
               f"{value}_mosaic_open_1", vmin, vmax, cmap, lab, fmt)
        mosaic(r1, open_models[half:], value,
               f"{value}_mosaic_open_2", vmin, vmax, cmap, lab, fmt)
        # proprietari: UNICA figura a 5 pannelli (3 con asse T, 2 con asse replica)
        prop_mosaic(df, value, f"{value}_mosaic_proprietary",
                    vmin, vmax, cmap, lab, fmt)


def prop_mosaic(df, value, fname, vmin, vmax, cmap, cbar_label, fmt):
    """Single-row proprietary mosaic: gpt-4o/4.1/5.2 on the T axis, gpt-5.5 and
    claude-sonnet-5 on the replica axis. Same panel style as the original."""
    panels = [("gpt-4o", "temperature", TEMPS, "Temperature"),
              ("gpt-4.1", "temperature", TEMPS, "Temperature"),
              ("gpt-5.2", "temperature", TEMPS, "Temperature"),
              ("gpt-5.5", "run", ["r1", "r2", "r3"], "Replica"),
              ("claude-sonnet-5", "run", ["r1", "r2", "r3"], "Replica")]
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(5 * 5, 4.6))
    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    for i, (model, x_field, x_values, x_label) in enumerate(panels):
        ax = axes[i]
        sub = df[df.model == model]
        if x_field == "temperature":
            sub = sub[sub.run == "r1"]
        hm = sub.pivot_table(index="epsilon", columns=x_field, values=value,
                             aggfunc="mean")                 .reindex(index=sorted(THRESHOLDS, reverse=True), columns=x_values)
        if value == "nmae":
            hm = hm.round(0)
        sns.heatmap(hm, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, annot=True,
                    fmt=fmt, cbar=(i == 4),
                    cbar_ax=cbar_ax if i == 4 else None,
                    cbar_kws={"label": cbar_label})
        ax.set_title(model, fontsize=15)
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel("CrossEnc Threshold" if i == 0 else "", fontsize=13)
    cbar_ax.set_ylabel(cbar_label, fontsize=15)
    cbar_ax.tick_params(labelsize=12)
    plt.subplots_adjust(left=0.05, right=0.9, top=0.88, bottom=0.15, wspace=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PLOTS, f"{fname}.{ext}"),
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"salvata {fname} (5 pannelli misti)")


if __name__ == "__main__":
    df = load_official()
    print(f"{df.model.nunique()} modelli, {len(df)} righe")
    version_a(df)
