import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

# --- Load JSON files ---
json_files = sorted(glob("model_performance/evaluation_metrics_CrossEncThr*.json"))
records = []

for file in json_files:
    with open(file, "r") as f:
        metrics = json.load(f)

    thr_str = os.path.basename(file).replace("evaluation_metrics_CrossEncThr", "").replace(".json", "")
    try:
        threshold = float(thr_str)
    except ValueError:
        continue

    for key, vals in metrics.items():
        if not isinstance(vals, dict) or 'f1' not in vals:
            continue
        model, temp = key.split("_T")
        records.append({
            "model": model,
            "temperature": float(temp),
            "threshold": threshold,
            "f1": vals["f1"],
            "mae": vals["mae"],
            "nmae": vals["normalized_mae"]
        })

# --- Create DataFrame ---
df = pd.DataFrame(records)

# --- Plot config ---
all_models = sorted(df['model'].unique())
gpt_models = [m for m in all_models if m.lower().startswith("gpt")]
other_models = [m for m in all_models if not m.lower().startswith("gpt")]
models = other_models + gpt_models
n_models = len(models)
ncols = 5
nrows = (n_models + ncols - 1) // ncols



########################## F1 ##############################


fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4.2 * nrows), constrained_layout=False)
axes = axes.flatten()

# Colorbar shared range
vmin, vmax = 0, 1
all_temps = sorted(df['temperature'].unique())
all_thrs = sorted(df['threshold'].unique(), reverse=True)

# Create colorbar axis
cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height] # type:ignore

# Plot each model heatmap
for i, model in enumerate(models):
    ax = axes[i]
    model_df = df[df['model'] == model]
    heatmap_data = model_df.pivot(index="threshold", columns="temperature", values="f1").reindex(index=all_thrs, columns=all_temps)

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        cbar=(i == n_models - 1),
        cbar_ax=cbar_ax if i == n_models - 1 else None,
        cbar_kws={"label": "F1 Score"}  # Add label here
    )

    ax.set_title(model, fontsize=15)
    ax.set_xlabel("Temperature", fontsize=13)
    ax.set_ylabel("CrossEnc Threshold", fontsize=13)

if n_models > 0:
    cbar_ax.set_ylabel("F1 Score", fontsize=15)
    cbar_ax.tick_params(labelsize=12)

# Hide any unused subplots
for j in range(n_models, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.subplots_adjust(
    left=0.07, right=0.9, top=0.93, bottom=0.07,
    wspace=0.3, hspace=0.35
)

# Save figure
os.makedirs("model_performance/plots", exist_ok=True)
plt.savefig("model_performance/plots/f1_mosaic_with_labels_and_shared_cbar.png", dpi=300)
plt.savefig("model_performance/plots/f1_mosaic_with_labels_and_shared_cbar.pdf", bbox_inches='tight')
plt.show()


########################## NMAE ##############################


fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4.2 * nrows), constrained_layout=False)
axes = axes.flatten()

# Colorbar shared range
vmin, vmax = 60, 125
all_temps = sorted(df['temperature'].unique())
all_thrs = sorted(df['threshold'].unique(), reverse=True)

# Create colorbar axis
cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height] # type:ignore

# Plot each model heatmap
for i, model in enumerate(models):
    ax = axes[i]
    model_df = df[df['model'] == model]
    heatmap_data = model_df.pivot(index="threshold", columns="temperature", values="nmae").reindex(index=all_thrs, columns=all_temps).round(0) # round data to the nearest integer

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="inferno_r",
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".0f",
        cbar=(i == n_models - 1),
        cbar_ax=cbar_ax if i == n_models - 1 else None,
        cbar_kws={"label": "NMAE [g]"}  # Add label here
    )

    ax.set_title(model, fontsize=15)
    ax.set_xlabel("Temperature", fontsize=13)
    ax.set_ylabel("CrossEnc Threshold", fontsize=13)

# Hide any unused subplots
for j in range(n_models, len(axes)):
    fig.delaxes(axes[j])

if n_models > 0:
    cbar_ax.set_ylabel("NMAE [g]", fontsize=15)
    cbar_ax.tick_params(labelsize=12)

# Adjust layout
plt.subplots_adjust(
    left=0.07, right=0.9, top=0.93, bottom=0.07,
    wspace=0.3, hspace=0.35
)

# Save figure
os.makedirs("model_performance/plots", exist_ok=True)
plt.savefig("model_performance/plots/NMAE_mosaic_with_labels_and_shared_cbar.png", dpi=300)
plt.savefig("model_performance/plots/NMAE_mosaic_with_labels_and_shared_cbar.pdf", bbox_inches='tight')
plt.show()
