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
SELECTED_BASELINES = ["iTransformer", "TimesNet", "WPMixer", "P_sLSTM"]
DISPLAY_ORDER = ["True", "DARNet"] + SELECTED_BASELINES
DISPLAY_LABEL = {
    "DARNet": "DATP-Net",
}
STYLE_SUFFIX = "spike_region_selected_baselines_v1"

TOKENS = {
    "surface": "#FFFFFF",
    "panel": "#FFFFFF",
    "ink": "#2F3A4A",
    "grid": "#E2E5EA",
    "axis": "#AEB4BE",
}
MATCHED_AXIS_COLOR = "#444444"

LINE_STYLE = {
    "True": dict(color="#6E6E6E", linewidth=1.95, linestyle="-", alpha=0.96, zorder=30),
    "DARNet": dict(color="#F4A261", linewidth=1.95, linestyle="-", alpha=0.96, zorder=25),
    "iTransformer": dict(color="#6FA8DC", linewidth=1.95, linestyle="-", alpha=0.90),
    "TimesNet": dict(color="#7BC8A4", linewidth=1.95, linestyle="-", alpha=0.90),
    "WPMixer": dict(color="#C9A0DC", linewidth=1.95, linestyle="-", alpha=0.90),
    "P_sLSTM": dict(color="#B08968", linewidth=1.95, linestyle="-", alpha=0.90),
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 180,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
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


def display_label(label: str) -> str:
    return DISPLAY_LABEL.get(label, label)


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
    ax.set_xlabel("Test Time Step", color=TOKENS["ink"], fontsize=17, labelpad=8)
    ax.set_ylabel("Value", color=TOKENS["ink"], fontsize=17, labelpad=8)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    ax.grid(True, linestyle="--", linewidth=0.75, color=TOKENS["grid"], alpha=0.86)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(1.05)
    ax.tick_params(axis="both", colors=TOKENS["ink"], labelsize=15, width=1.0, length=4.5)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.yaxis.get_offset_text().set_color(TOKENS["ink"])
    ax.xaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_color(TOKENS["ink"])


def match_hyperparameter_border(ax) -> None:
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color(MATCHED_AXIS_COLOR)


def add_inside_legend(ax, fontsize: float = 11.5) -> None:
    legend = ax.legend(
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        facecolor=TOKENS["panel"],
        edgecolor=TOKENS["axis"],
        framealpha=0.92,
        fontsize=fontsize,
        handlelength=2.35,
        handletextpad=0.62,
        labelspacing=0.42,
        borderpad=0.58,
        borderaxespad=0.32,
    )
    legend.get_frame().set_linewidth(0.9)
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
        ax.plot(x, y, label=display_label(label), **LINE_STYLE[label])
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
    fig, ax = plt.subplots(figsize=(10.8, 5.15), facecolor=TOKENS["surface"])
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
                line, = ax.plot(x, y, label=display_label(label), **LINE_STYLE[label])
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
        [display_label(x) for x in DISPLAY_ORDER if x in handles],
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
            style = LINE_STYLE[label].copy()
            if label in {"True", "DARNet"}:
                style["linewidth"] = 2.4
                style["solid_capstyle"] = "round"
            ax.plot(x, y, label=display_label(label), **style)
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


def plot_pred5_abilene_geant_two_subplots() -> dict[str, object]:
    pred_len = 5
    datasets = ["Abilene", "Geant"]
    dataset_display = {"Abilene": "Abilene", "Geant": "GÉANT"}
    label_color = "#000000"
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.6), facecolor=TOKENS["surface"])

    for c, dataset in enumerate(datasets):
        ax = axes[c]
        df = load_curve(dataset, pred_len)
        region = choose_spike_region(df)
        local = df.iloc[int(region["start_pos"]) : int(region["end_pos"])].copy()
        x = local["step"].to_numpy(dtype=np.float64)
        plotted = []
        for label in DISPLAY_ORDER:
            y = local[label].to_numpy(dtype=np.float64)
            plotted.append(y)
            style = LINE_STYLE[label].copy()
            style["linewidth"] = 4.2 if label in {"True", "DARNet"} else 2.15
            if label in {"True", "DARNet"}:
                style["solid_capstyle"] = "round"
            ax.plot(x, y, label=display_label(label), **style)

        style_axis(ax)
        match_hyperparameter_border(ax)
        ax.set_title(dataset_display.get(dataset, dataset), fontsize=22, color=label_color, pad=10)
        ax.set_xlabel("Test Time Step", color=label_color, fontsize=20, labelpad=8)
        ax.set_ylabel("Value", color=label_color, fontsize=20, labelpad=8)
        ax.tick_params(axis="both", colors=label_color, labelsize=18, width=1.0, length=4.5)
        ax.yaxis.get_offset_text().set_fontsize(18)
        ax.yaxis.get_offset_text().set_color(label_color)
        ax.xaxis.get_offset_text().set_fontsize(18)
        ax.xaxis.get_offset_text().set_color(label_color)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        lim = axis_margin(np.concatenate(plotted))
        if lim is not None:
            ax.set_ylim(*lim)
        if c > 0:
            ax.set_ylabel("")
            add_inside_legend(ax, fontsize=15)
            legend = ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color(label_color)

    fig.tight_layout(w_pad=2.4)

    stem = "Abilene_Geant_PL5_prediction_curves_spike_region_selected_baselines_1x2_v1"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    alias_pdf_path = FIG_DIR / "Abilene_Geant_PL5_prediction.pdf"
    alias_png_path = FIG_DIR / "Abilene_Geant_PL5_prediction.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(alias_pdf_path, bbox_inches="tight")
    fig.savefig(alias_png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return {
        "dataset": "Abilene_Geant",
        "pred_len": pred_len,
        "pdf": str(pdf_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "png": str(png_path.relative_to(OUT_DIR)).replace("\\", "/"),
    }


def plot_pred5_abilene_geant_full_test_two_subplots() -> dict[str, object]:
    pred_len = 5
    datasets = ["Abilene", "Geant"]
    dataset_display = {"Abilene": "Abilene", "Geant": "GÉANT"}
    label_color = "#000000"
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.6), facecolor=TOKENS["surface"])

    for c, dataset in enumerate(datasets):
        ax = axes[c]
        df = load_curve(dataset, pred_len)
        local = df.copy()
        x = local["step"].to_numpy(dtype=np.float64)
        plotted = []
        for label in DISPLAY_ORDER:
            y = local[label].to_numpy(dtype=np.float64)
            plotted.append(y)
            style = LINE_STYLE[label].copy()
            style["linewidth"] = 4.2 if label in {"True", "DARNet"} else 2.15
            if label in {"True", "DARNet"}:
                style["solid_capstyle"] = "round"
            ax.plot(x, y, label=display_label(label), **style)

        style_axis(ax)
        match_hyperparameter_border(ax)
        ax.set_title(dataset_display.get(dataset, dataset), fontsize=22, color=label_color, pad=10)
        ax.set_xlabel("Test Time Step", color=label_color, fontsize=20, labelpad=8)
        ax.set_ylabel("Value", color=label_color, fontsize=20, labelpad=8)
        ax.tick_params(axis="both", colors=label_color, labelsize=18, width=1.0, length=4.5)
        ax.yaxis.get_offset_text().set_fontsize(18)
        ax.yaxis.get_offset_text().set_color(label_color)
        ax.xaxis.get_offset_text().set_fontsize(18)
        ax.xaxis.get_offset_text().set_color(label_color)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        lim = axis_margin(np.concatenate(plotted))
        if lim is not None:
            ax.set_ylim(*lim)
        if c > 0:
            ax.set_ylabel("")
            add_inside_legend(ax, fontsize=15)
            legend = ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color(label_color)

    fig.tight_layout(w_pad=2.4)

    stem = "Abilene_Geant_PL5_prediction_full_test"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return {
        "dataset": "Abilene_Geant",
        "pred_len": pred_len,
        "region": "full_test",
        "pdf": str(pdf_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "png": str(png_path.relative_to(OUT_DIR)).replace("\\", "/"),
    }


def plot_pred5_abilene_geant_full_test_no_bold_two_subplots() -> dict[str, object]:
    pred_len = 5
    datasets = ["Abilene", "Geant"]
    dataset_display = {"Abilene": "Abilene", "Geant": "GÉANT"}
    label_color = "#000000"
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.6), facecolor=TOKENS["surface"])

    for c, dataset in enumerate(datasets):
        ax = axes[c]
        df = load_curve(dataset, pred_len)
        local = df.copy()
        x = local["step"].to_numpy(dtype=np.float64)
        plotted = []
        for label in DISPLAY_ORDER:
            y = local[label].to_numpy(dtype=np.float64)
            plotted.append(y)
            style = LINE_STYLE[label].copy()
            style["linewidth"] = 2.15
            ax.plot(x, y, label=display_label(label), **style)

        style_axis(ax)
        match_hyperparameter_border(ax)
        ax.set_title(dataset_display.get(dataset, dataset), fontsize=22, color=label_color, pad=10)
        ax.set_xlabel("Test Time Step", color=label_color, fontsize=20, labelpad=8)
        ax.set_ylabel("Value", color=label_color, fontsize=20, labelpad=8)
        ax.tick_params(axis="both", colors=label_color, labelsize=18, width=1.0, length=4.5)
        ax.yaxis.get_offset_text().set_fontsize(18)
        ax.yaxis.get_offset_text().set_color(label_color)
        ax.xaxis.get_offset_text().set_fontsize(18)
        ax.xaxis.get_offset_text().set_color(label_color)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        lim = axis_margin(np.concatenate(plotted))
        if lim is not None:
            ax.set_ylim(*lim)
        if c > 0:
            ax.set_ylabel("")
            add_inside_legend(ax, fontsize=15)
            legend = ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color(label_color)

    fig.tight_layout(w_pad=2.4)

    stem = "Abilene_Geant_PL5_prediction_full_test_no_bold"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return {
        "dataset": "Abilene_Geant",
        "pred_len": pred_len,
        "region": "full_test",
        "style": "no_bold_true_datp",
        "pdf": str(pdf_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "png": str(png_path.relative_to(OUT_DIR)).replace("\\", "/"),
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
    pred5_abilene_geant_row = plot_pred5_abilene_geant_two_subplots()
    pd.DataFrame([pred5_abilene_geant_row]).to_csv(
        OUT_DIR / "spike_region_selected_baselines_pred5_abilene_geant_1x2_v1_index.csv",
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
