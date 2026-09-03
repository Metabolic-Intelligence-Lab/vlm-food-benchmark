import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd


def calibrate(pred, GT, default_w0=20.0, fallback_cov=1.0):
    """
    Fit linear calibration: GT = k * pred + w0
    Returns:
        w0, k, sigma_w0, sigma_k, cov_w0_k
    Fallback is used if fit is impossible (too few points or singular).
    """
    # Remove NaNs/Infs
    mask = ~np.isnan(pred) & ~np.isnan(GT)
    pred = np.array(pred)[mask]
    GT = np.array(GT)[mask]

    # Check for enough points and variability
    if len(pred) < 2 or np.all(pred == pred[0]):
        # fallback
        k = 1.0 if np.mean(pred) != 0 else 0.0
        w0 = default_w0
        sigma_k = 0.0
        sigma_w0 = 0.0
        cov_w0_k = fallback_cov
        return w0, k, sigma_w0, sigma_k, cov_w0_k

    try:
        # Only request covariance if enough points for reliable estimation
        if len(pred) >= 3:
            (k, w0), pcov = np.polyfit(pred, GT, 1, cov=True)
            sigma_k = np.sqrt(pcov[0, 0])
            sigma_w0 = np.sqrt(pcov[1, 1])
            cov_w0_k = pcov[0, 1]
        else:
            # Fallback slope for 2 points
            k = (GT[-1] - GT[0]) / (pred[-1] - pred[0] + 1e-6)
            w0 = GT[0] - k * pred[0]
            sigma_k = 0.0
            sigma_w0 = 0.0
            cov_w0_k = fallback_cov
    except Exception:
        # Generic fallback
        k = np.mean(GT) / (np.mean(pred) + 1e-6)
        w0 = default_w0
        sigma_k = 0.0
        sigma_w0 = 0.0
        cov_w0_k = fallback_cov

    return w0, k, sigma_w0, sigma_k, cov_w0_k

files = glob.glob("model_performance/GT_PRED_*.csv")

fit_results = []

for f in files:

    data = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)

    GT = data[:, 0]
    pred = data[:, 1]

    MAE = np.median(np.abs(GT-pred))

    # Extract info from filename
    fname = os.path.basename(f)
    _, _, model, temp, thresh = fname.split("_")  

    out_path = "model_performance/plots/{}.png".format(fname)

    pred_corr = lambda pred, w0, k: k * pred + w0

    w0, k, sigma_w0, sigma_k, cov_w0_k = calibrate(pred, GT)
    corrected_pred = pred_corr(pred, w0, k)

    MAE_corr = np.median(np.abs(GT - corrected_pred))

    fit_results.append({
        "file": fname,
        "model": model,
        "temperature": temp,
        "threshold": thresh.replace(".csv", ""),
        "slope": k,
        "e_slope": sigma_k,
        "intercept": w0,
        "e_intercept": sigma_w0,
        "cov_slope_intercept": cov_w0_k,
        "MAE": round(MAE,2),
        "MAE_corr": round(MAE_corr,2),
        "N": len(pred)
    })

    x_vals = np.array([0, 350])
    y_vals = k * x_vals + w0


    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(pred, GT, 'o', alpha=0.6, label="N = %i"%len(pred))
    plt.plot([0, 350], [0, 350], 'k--')
    plt.plot(x_vals, y_vals, 'r-', label=f"Fit: y={k:.2f}x+{w0:.2f}")
    plt.xlabel("Predicted portion size [g]")
    plt.ylabel("GT portion size [g]")
    #plt.title(f"Model: {model}, T={temp}, Th={thresh}")
    plt.xlim(0, 350)
    plt.ylim(0, 350)
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


df = pd.DataFrame(fit_results)
df.to_csv("model_performance/fit_parameters.csv", index=False)