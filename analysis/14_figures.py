"""B5 - Paper figures, BOTH candidate versions, from the official configuration
(clean mode for the first campaign + metrics_v2_all for the 2026 campaign).

Version A (identical style to the submitted paper, extended):
  A_f1_mosaic_open_{1,2}.png/pdf   T x eps heatmaps, open models, split in two
  A_nmae_mosaic_open_{1,2}.png/pdf
  A_f1_mosaic_proprietary.png/pdf  gpt-4o/4.1/5.2 (T axis) + gpt-5.5/claude (replicas)
  A_nmae_mosaic_proprietary.png/pdf

Version B (new proposal):
  B_f1_models_by_eps.png/pdf       single annotated heatmap, open models x eps (T=0.0)
  B_proprietary_lines.png/pdf      F1 vs eps, one line per proprietary model,
                                   band = spread across T (or replicas)

Usage: analysis_env/bin/python analysis/14_figures.py
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

PLOTS = os.path.join(OUTPUTS_DIR, "plots_v2")
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
               f"A_{value}_mosaic_open_1", vmin, vmax, cmap, lab, fmt)
        mosaic(r1, open_models[half:], value,
               f"A_{value}_mosaic_open_2", vmin, vmax, cmap, lab, fmt)
        # proprietari: UNICA figura a 5 pannelli (3 con asse T, 2 con asse replica)
        prop_mosaic(df, value, f"A_{value}_mosaic_proprietary",
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


def version_b(df):
    # B1: heatmap open models x eps at T=0.0
    open_models = [m for m in df.model.unique()
                   if not m.startswith("gpt") and m != "claude-sonnet-5"]
    sub = df[(df.temperature == 0.0) & (df.run == "r1")
             & df.model.isin(open_models)]
    hm = sub.pivot_table(index="model", columns="epsilon", values="f1")
    hm = hm.loc[hm[0.5].sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(hm) + 1.2))
    sns.heatmap(hm, ax=ax, cmap="inferno", vmin=0, vmax=1, annot=True,
                fmt=".2f", annot_kws={"fontsize": 8},
                cbar_kws={"label": "F1 Score"})
    ax.set_xlabel("CrossEnc Threshold ε", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Open-source VLMs — F1 at T=0.0", fontsize=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PLOTS, f"B_f1_models_by_eps.{ext}"), dpi=300)
    plt.close(fig)
    print("salvata B_f1_models_by_eps")

    # B2: proprietari, linee F1 vs eps con banda di variabilita'
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {"gpt-4o": "#4053d3", "gpt-4.1": "#ddb310", "gpt-5.2": "#b51d14",
              "gpt-5.5": "#00beff", "claude-sonnet-5": "#fb49b0"}
    for model in ["gpt-4o", "gpt-4.1", "gpt-5.2", "gpt-5.5", "claude-sonnet-5"]:
        sub = df[df.model == model]
        agg = sub.groupby("epsilon")["f1"].agg(["mean", "min", "max"])
        c = colors[model]
        ax.plot(agg.index, agg["mean"], "-o", color=c, lw=2, ms=5, label=model)
        ax.fill_between(agg.index, agg["min"], agg["max"], color=c, alpha=0.15)
    ax.set_xlabel("CrossEnc Threshold ε", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.set_xlim(0.48, 0.92)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_title("Proprietary baselines — band: spread across T (or replicas)",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PLOTS, f"B_proprietary_lines.{ext}"), dpi=300)
    plt.close(fig)
    print("salvata B_proprietary_lines")


if __name__ == "__main__":
    df = load_official()
    print(f"{df.model.nunique()} modelli, {len(df)} righe")
    version_a(df)
    version_b(df)
