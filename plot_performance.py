import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

CrossEnc_thr = sys.argv[1]

# Load metrics from file
with open(f"model_performance/evaluation_metrics_CrossEncThr{CrossEnc_thr}.json", "r") as f:
    metrics = json.load(f)


# Convert to DataFrame
df = pd.DataFrame(metrics).T.reset_index()
df[['model', 'temperature']] = df['index'].str.extract(r'(.*)_T(.*)')
df['temperature'] = df['temperature'].astype(float)

palette = sns.color_palette("tab20", n_colors=12)  # tab20 has 20 distinct colors

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="temperature", y="f1", hue="model", marker="o", palette=palette)
plt.title("F1 Score by Model and Temperature")
plt.xlabel("Temperature")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.grid(True)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(f"./model_performance/plots/F1_CrossEncThr{CrossEnc_thr}.png")
plt.close()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="temperature", y="mae", hue="model", marker="o", palette=palette)
plt.title("MAE by Model and Temperature")
plt.xlabel("Temperature")
plt.ylabel("MAE")
plt.grid(True)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(f"./model_performance/plots/MAE_CrossEncThr{CrossEnc_thr}.png")
plt.close()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="temperature", y="normalized_mae", hue="model", marker="o", palette=palette)
plt.title("Normalized MAE by Model and Temperature")
plt.xlabel("Temperature")
plt.ylabel("MAE")
plt.grid(True)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(f"./model_performance/plots/MAEnorm_CrossEncThr{CrossEnc_thr}.png")
plt.close()