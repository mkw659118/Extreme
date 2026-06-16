from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


ROOT = Path.cwd().parent if Path.cwd().name == "PredictionPlot" else Path.cwd()
PLOT_DATA_DIR = ROOT / "PredictionPlot" / "plot_data"
FIG_DIR = ROOT / "PredictionPlot" / "figures"

DATASETS = ["Abilene", "Geant"]
PRED_LENS = [5, 10, 15, 20]
MODEL_COLUMN = "DARNet"
MODEL_LABEL = "DATP-Net"
STYLE_SUFFIX = "distribution_true_vs_DATPNet_v1"

COLORS = {
    "True": "#6E6E6E",
    MODEL_LABEL: "#F4A261",
}

TOKENS = {
    "surface": "#FFFFFF",
    "panel": "#FFFFFF",
    "ink": "#2F3A4A",
    "grid": "#E2E5EA",
    "axis": "#AEB4BE",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 180,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


def finite_values(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def load_curve(dataset: str, pred_len: int) -> pd.DataFrame:
    path = PLOT_DATA_DIR / f"{dataset}_PL{pred_len}_merged_prediction_curves.csv"
    df = pd.read_csv(path)
    missing = [col for col in ["True", MODEL_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def kde_curve(values: np.ndarray, xs: np.ndarray) -> np.ndarray | None:
    values = finite_values(values)
    if values.size < 2:
        return None
    std = float(np.std(values, ddof=1))
    if not np.isfinite(std) or std <= 0:
        return None
    bw = 1.06 * std * (values.size ** (-1 / 5))
    if not np.isfinite(bw) or bw <= 0:
        return None
    z = (xs[:, None] - values[None, :]) / bw
    density = np.exp(-0.5 * z * z).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    return density


def axis_range(*arrays: np.ndarray) -> tuple[float, float]:
    merged = np.concatenate([finite_values(arr) for arr in arrays])
    if merged.size == 0:
        return -1.0, 1.0
    lo, hi = np.quantile(merged, [0.005, 0.995])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(merged)), float(np.max(merged))
    pad = max((hi - lo) * 0.08, abs(hi) * 0.015, 1e-12)
    return float(lo - pad), float(hi + pad)


def style_axis(ax, title: str | None = None) -> None:
    ax.set_facecolor(TOKENS["panel"])
    if title:
        ax.set_title(title, fontsize=16, color=TOKENS["ink"], pad=8)
    ax.set_xlabel("Value", color=TOKENS["ink"], fontsize=14, labelpad=6)
    ax.set_ylabel("Density", color=TOKENS["ink"], fontsize=14, labelpad=6)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
    ax.grid(True, linestyle="--", linewidth=0.7, color=TOKENS["grid"], alpha=0.82)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(axis="both", colors=TOKENS["ink"], labelsize=11, width=0.9, length=4.0)
    ax.xaxis.get_offset_text().set_fontsize(11)
    ax.xaxis.get_offset_text().set_color(TOKENS["ink"])


def plot_distribution_on_axis(ax, true_values: np.ndarray, pred_values: np.ndarray, title: str) -> None:
    xmin, xmax = axis_range(true_values, pred_values)
    bins = np.linspace(xmin, xmax, 32)
    xs = np.linspace(xmin, xmax, 360)

    series = [
        ("Ground Truth", finite_values(true_values), COLORS["True"]),
        (MODEL_LABEL, finite_values(pred_values), COLORS[MODEL_LABEL]),
    ]
    for label, values, color in series:
        clipped = values[(values >= xmin) & (values <= xmax)]
        ax.hist(
            clipped,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.18,
            color=color,
            edgecolor=color,
            linewidth=1.15,
            label=f"{label} Hist.",
        )
        density = kde_curve(clipped, xs)
        if density is not None:
            ax.plot(xs, density, color=color, linewidth=2.2, label=f"{label} KDE")

    style_axis(ax, title)
    ax.set_xlim(xmin, xmax)


def distribution_stats(dataset: str, pred_len: int, label: str, values: np.ndarray) -> dict[str, object]:
    values = finite_values(values)
    return {
        "dataset": dataset,
        "pred_len": pred_len,
        "series": label,
        "count": int(values.size),
        "mean": float(np.mean(values)) if values.size else np.nan,
        "std": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
        "median": float(np.median(values)) if values.size else np.nan,
        "q05": float(np.quantile(values, 0.05)) if values.size else np.nan,
        "q95": float(np.quantile(values, 0.95)) if values.size else np.nan,
        "min": float(np.min(values)) if values.size else np.nan,
        "max": float(np.max(values)) if values.size else np.nan,
    }


def plot_abilene_geant_pl5() -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), facecolor=TOKENS["surface"])
    for ax, dataset in zip(axes, ["Abilene", "Geant"]):
        df = load_curve(dataset, 5)
        plot_distribution_on_axis(
            ax,
            df["True"].to_numpy(dtype=np.float64),
            df[MODEL_COLUMN].to_numpy(dtype=np.float64),
            dataset,
        )
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        facecolor=TOKENS["panel"],
        edgecolor=TOKENS["axis"],
        framealpha=0.92,
        fontsize=10.5,
    )
    fig.tight_layout(w_pad=2.6)
    stem = "Abilene_Geant_PL5_distribution_true_vs_DATPNet_1x2_v1"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}


def plot_all_grid() -> dict[str, str]:
    fig, axes = plt.subplots(
        len(DATASETS),
        len(PRED_LENS),
        figsize=(18.4, 10.2),
        facecolor=TOKENS["surface"],
    )
    for r, dataset in enumerate(DATASETS):
        for c, pred_len in enumerate(PRED_LENS):
            ax = axes[r, c]
            df = load_curve(dataset, pred_len)
            plot_distribution_on_axis(
                ax,
                df["True"].to_numpy(dtype=np.float64),
                df[MODEL_COLUMN].to_numpy(dtype=np.float64),
                f"{dataset} PL{pred_len}",
            )
            if r < len(DATASETS) - 1:
                ax.set_xlabel("")
            if c > 0:
                ax.set_ylabel("")

    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.012),
        ncol=4,
        frameon=False,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985], w_pad=1.8, h_pad=1.8)
    stem = f"all_datasets_predlens_{STYLE_SUFFIX}"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}


def write_stats() -> str:
    rows = []
    for dataset in DATASETS:
        for pred_len in PRED_LENS:
            df = load_curve(dataset, pred_len)
            rows.append(distribution_stats(dataset, pred_len, "True", df["True"].to_numpy(dtype=np.float64)))
            rows.append(distribution_stats(dataset, pred_len, MODEL_LABEL, df[MODEL_COLUMN].to_numpy(dtype=np.float64)))
    path = PLOT_DATA_DIR / f"{STYLE_SUFFIX}_stats.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [plot_abilene_geant_pl5(), plot_all_grid()]
    stats_path = write_stats()
    for output in outputs:
        print(output["pdf"])
        print(output["png"])
    print(stats_path)


if __name__ == "__main__":
    main()
