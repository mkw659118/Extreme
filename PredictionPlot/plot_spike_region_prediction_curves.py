from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter


ROOT = Path.cwd().parent if Path.cwd().name == "PredictionPlot" else Path.cwd()
PLOT_DATA_DIR = ROOT / "PredictionPlot" / "plot_data"
FIG_DIR = ROOT / "PredictionPlot" / "figures"
OUT_DIR = ROOT / "PredictionPlot"

DATASETS = ["Abilene", "Geant", "Seattle"]
PRED_LENS = [5, 10, 15, 20]
SELECTED_BASELINES = ["iTransformer", "HMformer", "TimesNet", "WPMixer", "P_sLSTM"]
DISPLAY_ORDER = ["True", "DARNet"] + SELECTED_BASELINES
STYLE_SUFFIX = "spike_region_selected_baselines_v1"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

LINE_STYLE = {
    "True": dict(color="#4A4F5A", linewidth=2.55, linestyle="-", alpha=0.98, zorder=30),
    "DARNet": dict(color="#B23A48", linewidth=2.20, linestyle="-", alpha=0.98, zorder=25),
    "iTransformer": dict(color="#5477C4", linewidth=1.24, linestyle="-", alpha=0.68),
    "HMformer": dict(color="#8A3A6F", linewidth=1.24, linestyle="-", alpha=0.68),
    "TimesNet": dict(color="#7A828F", linewidth=1.18, linestyle="-", alpha=0.70),
    "WPMixer": dict(color="#386411", linewidth=1.22, linestyle="-", alpha=0.68),
    "P_sLSTM": dict(color="#CC6F47", linewidth=1.22, linestyle="-", alpha=0.68),
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 180,
    }
)


def axis_margin(values: np.ndarray) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    vmin, vmax = float(np.min(values)), float(np.max(values))
    pad = max((vmax - vmin) * 0.08, abs(vmax) * 0.015, 1e-12)
    return vmin - pad, vmax + pad


def load_curve(dataset: str, pred_len: int) -> pd.DataFrame:
    path = PLOT_DATA_DIR / f"{dataset}_PL{pred_len}_merged_prediction_curves.csv"
    df = pd.read_csv(path)
    missing = [col for col in ["step", "True"] + DISPLAY_ORDER[1:] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def choose_spike_region(df: pd.DataFrame) -> dict[str, int | float]:
    true = df["True"].to_numpy(dtype=np.float64)
    peak_pos = int(np.nanargmax(true))
    n = len(df)
    radius = min(max(int(round(n * 0.08)), 12), 45)
    target_len = min(n, radius * 2 + 1)
    start = max(0, peak_pos - radius)
    end = min(n, peak_pos + radius + 1)
    if end - start < target_len:
        if start == 0:
            end = min(n, target_len)
        elif end == n:
            start = max(0, n - target_len)
    return {
        "peak_pos": peak_pos,
        "peak_step": int(df["step"].iloc[peak_pos]),
        "peak_true": float(true[peak_pos]),
        "start_pos": int(start),
        "end_pos": int(end),
        "start_step": int(df["step"].iloc[start]),
        "end_step": int(df["step"].iloc[end - 1]),
        "region_points": int(end - start),
    }


def style_axis(ax) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.set_xlabel("Test Time Step", color=TOKENS["ink"])
    ax.set_ylabel("Value", color=TOKENS["ink"])
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    ax.grid(True, linestyle="--", linewidth=0.6, color=TOKENS["grid"], alpha=0.82)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=TOKENS["ink"])


def add_inside_legend(ax, fontsize: float = 8.5) -> None:
    legend = ax.legend(
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        facecolor=TOKENS["panel"],
        edgecolor=TOKENS["axis"],
        framealpha=0.86,
        fontsize=fontsize,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.34,
        borderpad=0.48,
        borderaxespad=0.25,
    )
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color(TOKENS["ink"])
        text.set_ha("right")


def plot_region_on_axis(ax, df: pd.DataFrame, region: dict[str, int | float], show_legend: bool = True) -> None:
    local = df.iloc[int(region["start_pos"]) : int(region["end_pos"])].copy()
    x = local["step"].to_numpy(dtype=np.float64)
    plotted = []
    for label in DISPLAY_ORDER:
        y = local[label].to_numpy(dtype=np.float64)
        plotted.append(y)
        ax.plot(x, y, label=label, **LINE_STYLE[label])
    style_axis(ax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    lim = axis_margin(np.concatenate(plotted))
    if lim is not None:
        ax.set_ylim(*lim)
    if show_legend:
        add_inside_legend(ax)


def plot_single(dataset: str, pred_len: int) -> dict[str, object]:
    df = load_curve(dataset, pred_len)
    region = choose_spike_region(df)
    fig, ax = plt.subplots(figsize=(10.4, 4.8), facecolor=TOKENS["surface"])
    plot_region_on_axis(ax, df, region, show_legend=True)
    fig.tight_layout()

    stem = f"{dataset}_PL{pred_len}_prediction_curves_{STYLE_SUFFIX}"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    curve_path = OUT_DIR / "plot_data" / f"{dataset}_PL{pred_len}_{STYLE_SUFFIX}.csv"

    local = df.iloc[int(region["start_pos"]) : int(region["end_pos"])].copy()
    local.to_csv(curve_path, index=False, encoding="utf-8-sig")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return {
        "dataset": dataset,
        "pred_len": pred_len,
        **region,
        "pdf": str(pdf_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "png": str(png_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "curve_csv": str(curve_path.relative_to(OUT_DIR)).replace("\\", "/"),
    }


def plot_grid(index_rows: list[dict[str, object]]) -> dict[str, object]:
    row_by_key = {(r["dataset"], int(r["pred_len"])): r for r in index_rows}
    fig, axes = plt.subplots(len(DATASETS), len(PRED_LENS), figsize=(19.5, 10.2), facecolor=TOKENS["surface"])
    handles = OrderedDict()

    for r, dataset in enumerate(DATASETS):
        for c, pred_len in enumerate(PRED_LENS):
            ax = axes[r, c]
            df = load_curve(dataset, pred_len)
            region = row_by_key[(dataset, pred_len)]
            local = df.iloc[int(region["start_pos"]) : int(region["end_pos"])].copy()
            x = local["step"].to_numpy(dtype=np.float64)
            plotted = []
            for label in DISPLAY_ORDER:
                y = local[label].to_numpy(dtype=np.float64)
                plotted.append(y)
                line, = ax.plot(x, y, label=label, **LINE_STYLE[label])
                handles.setdefault(label, line)
            style_axis(ax)
            ax.set_title(f"{dataset} PL{pred_len}", fontsize=10.5, color=TOKENS["ink"], pad=4)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.tick_params(axis="both", labelsize=7.6)
            lim = axis_margin(np.concatenate(plotted))
            if lim is not None:
                ax.set_ylim(*lim)
            if r < len(DATASETS) - 1:
                ax.set_xlabel("")
            if c > 0:
                ax.set_ylabel("")

    legend = fig.legend(
        [handles[x] for x in DISPLAY_ORDER if x in handles],
        [x for x in DISPLAY_ORDER if x in handles],
        loc="upper right",
        bbox_to_anchor=(0.986, 0.985),
        ncol=1,
        frameon=True,
        facecolor=TOKENS["panel"],
        edgecolor=TOKENS["axis"],
        framealpha=0.88,
        fontsize=8.8,
        handlelength=2.3,
        labelspacing=0.34,
        borderpad=0.48,
    )
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color(TOKENS["ink"])
        text.set_ha("right")

    fig.tight_layout(rect=[0, 0, 0.995, 1])
    stem = f"main_12_subplots_prediction_curves_{STYLE_SUFFIX}"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return {
        "dataset": "ALL",
        "pred_len": "ALL",
        "pdf": str(pdf_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "png": str(png_path.relative_to(OUT_DIR)).replace("\\", "/"),
    }


def plot_pred5_three_subplots(index_rows: list[dict[str, object]]) -> dict[str, object]:
    pred_len = 5
    row_by_key = {(r["dataset"], int(r["pred_len"])): r for r in index_rows}
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16.6, 4.45), facecolor=TOKENS["surface"])
    fig.patch.set_alpha(0.0)

    for c, dataset in enumerate(DATASETS):
        ax = axes[c]
        df = load_curve(dataset, pred_len)
        region = row_by_key[(dataset, pred_len)]
        local = df.iloc[int(region["start_pos"]) : int(region["end_pos"])].copy()
        x = local["step"].to_numpy(dtype=np.float64)
        plotted = []
        for label in DISPLAY_ORDER:
            y = local[label].to_numpy(dtype=np.float64)
            plotted.append(y)
            ax.plot(x, y, label=label, **LINE_STYLE[label])
        style_axis(ax)
        ax.patch.set_alpha(0.0)
        ax.set_title(dataset, fontsize=11.0, color=TOKENS["ink"], pad=5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.tick_params(axis="both", labelsize=8.5)
        lim = axis_margin(np.concatenate(plotted))
        if lim is not None:
            ax.set_ylim(*lim)
        if c > 0:
            ax.set_ylabel("")
        if c == len(DATASETS) - 1:
            add_inside_legend(ax, fontsize=8.2)

    fig.tight_layout(w_pad=2.0)
    stem = "main_3_subplots_prediction_curves_spike_region_selected_baselines_pred5_no_pred_label_v1"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    nobg_pdf_path = FIG_DIR / f"{stem}_nobg.pdf"
    nobg_png_path = FIG_DIR / f"{stem}_nobg.png"
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    fig.savefig(png_path, bbox_inches="tight", dpi=300, transparent=True)
    fig.savefig(nobg_pdf_path, bbox_inches="tight", transparent=True)
    fig.savefig(nobg_png_path, bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)

    return {
        "dataset": "ALL",
        "pred_len": pred_len,
        "pdf": str(pdf_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "png": str(png_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "nobg_pdf": str(nobg_pdf_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "nobg_png": str(nobg_png_path.relative_to(OUT_DIR)).replace("\\", "/"),
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        for pred_len in PRED_LENS:
            print(f"plot spike region {dataset} PL{pred_len}", flush=True)
            rows.append(plot_single(dataset, pred_len))
    grid_row = plot_grid(rows)
    pred5_grid_row = plot_pred5_three_subplots(rows)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"{STYLE_SUFFIX}_figure_index.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([grid_row]).to_csv(OUT_DIR / f"{STYLE_SUFFIX}_grid_index.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([pred5_grid_row]).to_csv(
        OUT_DIR / "spike_region_selected_baselines_pred5_three_subplots_no_pred_label_v1_grid_index.csv",
        index=False,
        encoding="utf-8-sig",
    )

    palette = pd.DataFrame(
        [
            {"series": series, "color": style["color"], "line_style": str(style["linestyle"]), "line_width": style["linewidth"]}
            for series, style in LINE_STYLE.items()
        ]
    )
    palette.to_csv(OUT_DIR / f"{STYLE_SUFFIX}_palette.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
