# Exported from select_prediction_horizon_windows.ipynb

from pathlib import Path
from collections import OrderedDict, defaultdict
import math
import re
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

ROOT = Path.cwd().parent if Path.cwd().name == "PredictionPlot" else Path.cwd()
DRAW_DIR = ROOT / "draw"
OUT_DIR = ROOT / "PredictionPlot"
CUSTOM_FIG_DIR = OUT_DIR / "custom_horizon_figures"
CUSTOM_DATA_DIR = OUT_DIR / "custom_horizon_data"
CUSTOM_FIG_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["Abilene", "Geant", "Seattle"]
PRED_LENS = [5, 10, 15, 20]
D_MODEL = 256
TARGET_COL = "TC0"

MODEL_DIR_TO_LABEL = OrderedDict([
    ("net", "DARNet"),
    ("PMDformer", "PMDformer"),
    ("iTransformer", "iTransformer"),
    ("FEDformer", "FEDformer"),
    ("FeTS", "FeTS"),
    ("HMformer", "HMformer"),
    ("PatchTST", "PatchTST"),
    ("timesnet", "TimesNet"),
    ("WPMixer", "WPMixer"),
    ("P_sLSTM", "P_sLSTM"),
    ("xLSTMTime", "xLSTMTime"),
    ("xlstm_mixer", "xLSTM-Mixer"),
])
MODEL_ORDER = list(MODEL_DIR_TO_LABEL.values())
DISPLAY_ORDER = ["True"] + MODEL_ORDER

LINE_STYLE = {
    "True": dict(color="black", linewidth=2.8, linestyle="-", marker="o", markersize=5.0, alpha=0.95, zorder=30),
    "DARNet": dict(color="#d62728", linewidth=2.4, linestyle="-", marker="s", markersize=4.8, alpha=0.96, zorder=25),
    "PMDformer": dict(color="#1f77b4", linewidth=1.35, linestyle="-", marker=".", markersize=4.5, alpha=0.88),
    "iTransformer": dict(color="#ff7f0e", linewidth=1.35, linestyle="-", marker=".", markersize=4.5, alpha=0.88),
    "FEDformer": dict(color="#2ca02c", linewidth=1.35, linestyle="-", marker=".", markersize=4.5, alpha=0.88),
    "FeTS": dict(color="#9467bd", linewidth=1.35, linestyle="-", marker=".", markersize=4.5, alpha=0.88),
    "HMformer": dict(color="#8c564b", linewidth=1.35, linestyle="-", marker=".", markersize=4.5, alpha=0.88),
    "PatchTST": dict(color="#e377c2", linewidth=1.35, linestyle="-", marker=".", markersize=4.5, alpha=0.88),
    "TimesNet": dict(color="#7f7f7f", linewidth=1.35, linestyle="-", marker=".", markersize=4.5, alpha=0.88),
    "WPMixer": dict(color="#bcbd22", linewidth=1.35, linestyle="-", marker=".", markersize=4.5, alpha=0.88),
    "P_sLSTM": dict(color="#17becf", linewidth=1.35, linestyle="-", marker=".", markersize=4.5, alpha=0.88),
    "xLSTMTime": dict(color="#00429d", linewidth=1.35, linestyle="--", marker=".", markersize=4.5, alpha=0.90),
    "xLSTM-Mixer": dict(color="#93003a", linewidth=1.35, linestyle="--", marker=".", markersize=4.5, alpha=0.90),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 180,
})


# %%
def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def get_raw_path(dataset, model_dir, pred_len, d_model=D_MODEL, target_col=TARGET_COL):
    return DRAW_DIR / dataset / f"{dataset}_single" / model_dir / f"PL{pred_len}_DM{d_model}" / target_col / "test_raw.csv"


def read_raw_windows(path, pred_len):
    df = pd.read_csv(path)
    required = {"true", "pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    true = pd.to_numeric(df["true"], errors="coerce").to_numpy(dtype=np.float64)
    pred = pd.to_numeric(df["pred"], errors="coerce").to_numpy(dtype=np.float64)
    usable = (len(true) // pred_len) * pred_len
    if usable == 0:
        raise ValueError(f"{path} has no usable rows for pred_len={pred_len}")
    return true[:usable].reshape(-1, pred_len), pred[:usable].reshape(-1, pred_len)


def load_windows(dataset, pred_len):
    windows = {}
    missing = []
    for model_dir, label in MODEL_DIR_TO_LABEL.items():
        path = get_raw_path(dataset, model_dir, pred_len)
        if not path.exists():
            missing.append(str(path))
            continue
        true_w, pred_w = read_raw_windows(path, pred_len)
        windows[label] = {
            "true": true_w,
            "pred": pred_w,
            "path": path,
            "model_dir": model_dir,
        }
    if missing:
        raise FileNotFoundError("Missing raw files:\n" + "\n".join(missing))
    return windows


def curve_dynamic_score(curve):
    curve = np.asarray(curve, dtype=np.float64)
    finite = np.isfinite(curve)
    if not finite.any():
        return -np.inf
    vals = curve[finite]
    return float((np.nanmax(vals) - np.nanmin(vals)) + 0.10 * np.nanmax(np.abs(vals)))


def scale_for_curve(curve):
    curve = np.asarray(curve, dtype=np.float64)
    finite = np.isfinite(curve)
    if not finite.any():
        return 1.0
    vals = curve[finite]
    return max(
        float(np.nanmax(np.abs(vals))),
        float(np.nanmax(vals) - np.nanmin(vals)),
        float(np.nanstd(vals)),
        1e-12,
    )


def match_window_by_true(true_windows, reference_true):
    valid = np.isfinite(true_windows) & np.isfinite(reference_true[None, :])
    diff = np.where(valid, true_windows - reference_true[None, :], 0.0)
    valid_count = valid.sum(axis=1)
    mse = np.full(true_windows.shape[0], np.inf, dtype=np.float64)
    ok = valid_count > 0
    mse[ok] = (diff[ok] ** 2).sum(axis=1) / valid_count[ok]
    idx = int(np.argmin(mse))
    rmse = float(np.sqrt(mse[idx])) if np.isfinite(mse[idx]) else math.inf
    max_abs = float(np.nanmax(np.abs(true_windows[idx] - reference_true))) if np.isfinite(rmse) else math.inf
    return idx, rmse, max_abs


def candidate_windows(dataset, pred_len, top_n=30):
    """Return DARNet reference-window candidates sorted by dynamic score."""
    windows = load_windows(dataset, pred_len)
    true_w = windows["DARNet"]["true"]
    pred_w = windows["DARNet"]["pred"]
    rows = []
    for idx, true_curve in enumerate(true_w):
        pred_curve = pred_w[idx]
        err = pred_curve - true_curve
        rows.append({
            "dataset": dataset,
            "pred_len": pred_len,
            "window": idx,
            "dynamic_score": curve_dynamic_score(true_curve),
            "true_min": float(np.nanmin(true_curve)),
            "true_max": float(np.nanmax(true_curve)),
            "true_range": float(np.nanmax(true_curve) - np.nanmin(true_curve)),
            "true_first": float(true_curve[0]),
            "true_last": float(true_curve[-1]),
            "darnet_mae": float(np.nanmean(np.abs(err))),
            "darnet_rmse": float(np.sqrt(np.nanmean(err ** 2))),
        })
    return pd.DataFrame(rows).sort_values("dynamic_score", ascending=False).head(top_n).reset_index(drop=True)


def build_selected_curve(dataset, pred_len, reference_window, align_by_true=True):
    """
    Build one horizon curve dataframe.

    reference_window is the DARNet window index.
    If align_by_true=True, each baseline uses the window whose true curve best matches DARNet's reference true curve.
    """
    windows = load_windows(dataset, pred_len)
    ref_true_windows = windows["DARNet"]["true"]
    if not 0 <= int(reference_window) < ref_true_windows.shape[0]:
        raise IndexError(f"reference_window={reference_window} out of range for {dataset} PL{pred_len}; valid 0..{ref_true_windows.shape[0]-1}")

    reference_window = int(reference_window)
    reference_true = ref_true_windows[reference_window]
    df = pd.DataFrame({"horizon": np.arange(1, pred_len + 1), "True": reference_true})

    match_rows = []
    for model in MODEL_ORDER:
        if model == "DARNet" or not align_by_true:
            idx = min(reference_window, windows[model]["true"].shape[0] - 1)
            rmse = 0.0 if model == "DARNet" else np.nan
            max_abs = 0.0 if model == "DARNet" else np.nan
        else:
            idx, rmse, max_abs = match_window_by_true(windows[model]["true"], reference_true)
        df[model] = windows[model]["pred"][idx]
        match_rows.append({
            "dataset": dataset,
            "pred_len": pred_len,
            "reference_window": reference_window,
            "model": model,
            "matched_window": idx,
            "true_match_rmse": rmse,
            "true_match_max_abs": max_abs,
            "window_count": windows[model]["true"].shape[0],
        })
    return df, pd.DataFrame(match_rows)


def axis_margin(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    vmin, vmax = float(np.min(values)), float(np.max(values))
    pad = max((vmax - vmin) * 0.08, abs(vmax) * 0.02, 1e-12)
    return vmin - pad, vmax + pad


def plot_one_window(dataset, pred_len, reference_window, align_by_true=True, save=True, show=True):
    df, matches = build_selected_curve(dataset, pred_len, reference_window, align_by_true=align_by_true)
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    x = df["horizon"].to_numpy()
    plotted = []
    for label in DISPLAY_ORDER:
        y = df[label].to_numpy(dtype=np.float64)
        plotted.append(y)
        ax.plot(x, y, label=label, **LINE_STYLE[label])

    ax.set_title(f"{dataset} | PredLen={pred_len} | RefWindow={reference_window}", fontsize=13, pad=10)
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Value")
    ax.set_xticks(np.arange(1, pred_len + 1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.38)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    lim = axis_margin(np.concatenate(plotted))
    if lim is not None:
        ax.set_ylim(*lim)
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=9, handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    if save:
        stem = f"{safe_name(dataset)}_PL{pred_len}_window{reference_window}_horizon"
        pdf_path = CUSTOM_FIG_DIR / f"{stem}.pdf"
        png_path = CUSTOM_FIG_DIR / f"{stem}.png"
        csv_path = CUSTOM_DATA_DIR / f"{stem}.csv"
        match_path = CUSTOM_DATA_DIR / f"{stem}_matches.csv"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        matches.to_csv(match_path, index=False, encoding="utf-8-sig")
        print(f"saved: {pdf_path}")
        print(f"saved: {png_path}")
        print(f"saved: {csv_path}")
        print(f"saved: {match_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return df, matches


def plot_selected_grid(selections, align_by_true=True, save_name="custom_12_subplots_horizon"):
    """
    selections format:
        {("Abilene", 5): 376, ("Abilene", 10): 376, ...}

    Missing combinations are skipped and shown as blank panels.
    """
    fig, axes = plt.subplots(len(DATASETS), len(PRED_LENS), figsize=(21.0, 11.8), sharex=False, sharey=False)
    handles = OrderedDict()
    index_rows = []
    match_rows_all = []

    for r, dataset in enumerate(DATASETS):
        for c, pred_len in enumerate(PRED_LENS):
            ax = axes[r, c]
            key = (dataset, pred_len)
            if key not in selections:
                ax.axis("off")
                ax.set_title(f"{dataset} | PredLen={pred_len} | not selected")
                continue

            reference_window = int(selections[key])
            df, matches = build_selected_curve(dataset, pred_len, reference_window, align_by_true=align_by_true)
            data_path = CUSTOM_DATA_DIR / f"{safe_name(dataset)}_PL{pred_len}_window{reference_window}_grid_curve.csv"
            df.to_csv(data_path, index=False, encoding="utf-8-sig")
            matches.to_csv(CUSTOM_DATA_DIR / f"{safe_name(dataset)}_PL{pred_len}_window{reference_window}_grid_matches.csv", index=False, encoding="utf-8-sig")
            match_rows_all.extend(matches.to_dict("records"))
            index_rows.append({"dataset": dataset, "pred_len": pred_len, "reference_window": reference_window, "curve_csv": str(data_path.relative_to(OUT_DIR)).replace("\\", "/")})

            x = df["horizon"].to_numpy()
            plotted = []
            for label in DISPLAY_ORDER:
                y = df[label].to_numpy(dtype=np.float64)
                plotted.append(y)
                line, = ax.plot(x, y, label=label, **LINE_STYLE[label])
                handles.setdefault(label, line)
            ax.set_title(f"{dataset} | PredLen={pred_len} | W={reference_window}", fontsize=12, pad=7)
            ax.set_xticks(np.arange(1, pred_len + 1))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(pred_len, 6)))
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
            ax.tick_params(axis="both", labelsize=8)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            lim = axis_margin(np.concatenate(plotted))
            if lim is not None:
                ax.set_ylim(*lim)
            if r == len(DATASETS) - 1:
                ax.set_xlabel("Prediction Horizon", fontsize=10)
            if c == 0:
                ax.set_ylabel("Value", fontsize=10)

    fig.suptitle("Selected Forecast Horizon Windows", fontsize=16, y=0.985)
    fig.legend([handles[x] for x in DISPLAY_ORDER if x in handles], [x for x in DISPLAY_ORDER if x in handles], loc="lower center", ncol=7, frameon=False, fontsize=10, handlelength=2.4, columnspacing=1.2)
    fig.tight_layout(rect=[0, 0.095, 1, 0.965])

    pdf_path = CUSTOM_FIG_DIR / f"{save_name}.pdf"
    png_path = CUSTOM_FIG_DIR / f"{save_name}.png"
    index_path = CUSTOM_DATA_DIR / f"{save_name}_index.csv"
    match_path = CUSTOM_DATA_DIR / f"{save_name}_matches.csv"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    pd.DataFrame(index_rows).to_csv(index_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(match_rows_all).to_csv(match_path, index=False, encoding="utf-8-sig")
    plt.show()
    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")
    print(f"saved: {index_path}")
    print(f"saved: {match_path}")
    return pdf_path, png_path


# %%
dataset = "Abilene"
pred_len = 20
top_n = 20

candidates = candidate_windows(dataset, pred_len, top_n=top_n)
candidates


# %%
dataset = "Abilene"
pred_len = 20
reference_window = 376

single_curve_df, single_match_df = plot_one_window(
    dataset=dataset,
    pred_len=pred_len,
    reference_window=reference_window,
    align_by_true=True,
    save=True,
    show=True,
)

single_match_df


# %%
SELECTED_WINDOWS = {
    ("Abilene", 5): 376,
    ("Abilene", 10): 376,
    ("Abilene", 15): 368,
    ("Abilene", 20): 368,

    ("Geant", 5): 448,
    ("Geant", 10): 384,
    ("Geant", 15): 424,
    ("Geant", 20): 192,

    ("Seattle", 5): 76,
    ("Seattle", 10): 32,
    ("Seattle", 15): 32,
    ("Seattle", 20): 32,
}

grid_index_df, grid_match_df = plot_selected_grid(
    SELECTED_WINDOWS,
    align_by_true=True,
    save_name="custom_selected_12_subplots_horizon",
)

grid_index_df


# %%
for (dataset, pred_len), window in SELECTED_WINDOWS.items():
    plot_one_window(
        dataset=dataset,
        pred_len=pred_len,
        reference_window=window,
        align_by_true=True,
        save=True,
        show=False,
    )

print("done")
print("figure directory:", CUSTOM_FIG_DIR)
print("data directory:", CUSTOM_DATA_DIR)


# %%
figures = sorted(CUSTOM_FIG_DIR.glob("*.pdf"))
data_files = sorted(CUSTOM_DATA_DIR.glob("*.csv"))

print("PDF figures:", len(figures))
for path in figures[-10:]:
    print(path)

print("CSV files:", len(data_files))
for path in data_files[-10:]:
    print(path)


# %%
import subprocess
import sys

ALL_WINDOW_ID_DIR = OUT_DIR / "all_window_id_figures"
cmd = [
    sys.executable,
    str(OUT_DIR / "export_all_window_id_figures.py"),
    "--output-dir",
    str(ALL_WINDOW_ID_DIR),
    "--progress-every",
    "100",
]

# Optional examples for partial export:
# cmd += ["--datasets", "Abilene", "--pred-lens", "20"]
# cmd += ["--max-windows", "20"]
# cmd += ["--overwrite"]

print("running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("saved to:", ALL_WINDOW_ID_DIR)
