from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter


ROOT = Path.cwd().parent if Path.cwd().name == "PredictionPlot" else Path.cwd()
OUT_DIR = ROOT / "PredictionPlot"
FIG_DIR = OUT_DIR / "figures"
PLOT_DATA_DIR = OUT_DIR / "plot_data"
DISCOVERED_PATH = OUT_DIR / "discovered_prediction_files.csv"

STEM = "Abilene_Geant_PL5_prediction_full_test_all_baselines_bold_true_datp_cos_from_log_legend_upper_right_600pts"
COS_PATH = PLOT_DATA_DIR / f"{STEM}_cos_values.csv"

DATASETS = ["Abilene", "Geant"]
DATASET_DISPLAY = {"Abilene": "Abilene", "Geant": "GÉANT"}
PRED_LEN = 5
MODELS = [
    "DARNet",
    "PMDformer",
    "HMformer",
    "FeTS",
    "TimesNet",
    "iTransformer",
    "PatchTST",
    "WPMixer",
    "P_sLSTM",
    "xLSTMTime",
    "xLSTM-Mixer",
    "FEDformer",
]
DISPLAY_LABEL = {"DARNet": "DATP-Net"}

TOKENS = {
    "surface": "#FFFFFF",
    "panel": "#FFFFFF",
    "ink": "#000000",
    "grid": "#E2E5EA",
    "axis": "#444444",
}

LINE_STYLE = {
    "True": dict(color="#3F3F3F", linewidth=3.9, linestyle="-", alpha=0.96, zorder=30),
    "DARNet": dict(color="#FF8C29", linewidth=3.6, linestyle="-", alpha=0.96, zorder=28),
    "PMDformer": dict(color="#8FB1D4", linewidth=1.65, linestyle="-", alpha=0.78),
    "HMformer": dict(color="#85BC7C", linewidth=1.65, linestyle="-", alpha=0.78),
    "FeTS": dict(color="#FF9DA1", linewidth=1.65, linestyle="-", alpha=0.78),
    "TimesNet": dict(color="#A7D7D7", linewidth=1.65, linestyle="-", alpha=0.78),
    "iTransformer": dict(color="#B6D6F5", linewidth=1.65, linestyle="-", alpha=0.78),
    "PatchTST": dict(color="#F4D166", linewidth=1.65, linestyle="-", alpha=0.78),
    "WPMixer": dict(color="#C9A3BA", linewidth=1.65, linestyle="-", alpha=0.78),
    "P_sLSTM": dict(color="#C3A795", linewidth=1.65, linestyle="-", alpha=0.78),
    "xLSTMTime": dict(color="#CFCAC0", linewidth=1.65, linestyle="-", alpha=0.78),
    "xLSTM-Mixer": dict(color="#A9DCA3", linewidth=1.65, linestyle="-", alpha=0.78),
    "FEDformer": dict(color="#FFB3B8", linewidth=1.65, linestyle="-", alpha=0.78),
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


def axis_margin(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    pad = max((vmax - vmin) * 0.08, abs(vmax) * 0.015, 1e-12)
    return vmin - pad, vmax + pad


def model_label(model: str, cos_by_dataset: dict[tuple[str, str], float], dataset_display: str) -> str:
    label = DISPLAY_LABEL.get(model, model)
    cos = cos_by_dataset.get((dataset_display, label))
    if cos is None:
        cos = cos_by_dataset.get((dataset_display, model))
    return f"{label} ({cos:.4f})" if cos is not None else label


def load_dataset_curves(dataset: str) -> pd.DataFrame:
    discovered = pd.read_csv(DISCOVERED_PATH)
    rows = discovered[(discovered["dataset"] == dataset) & (discovered["pred_len"] == PRED_LEN)]

    curves: pd.DataFrame | None = None
    for model in MODELS:
        row = rows[rows["model"] == model]
        if row.empty:
            raise ValueError(f"Missing {dataset} PL{PRED_LEN} prediction path for {model}")
        data_path = Path(str(row.iloc[0]["path"]))
        raw = pd.read_csv(data_path)
        model_df = pd.DataFrame(
            {
                "step": np.arange(len(raw), dtype=np.int64),
                "True": raw["true"].to_numpy(dtype=np.float64),
                model: raw["pred"].to_numpy(dtype=np.float64),
            }
        )
        if curves is None:
            curves = model_df
        else:
            if len(curves) != len(model_df):
                raise ValueError(f"{dataset} {model} length mismatch")
            curves[model] = model_df[model]

    if curves is None:
        raise ValueError(f"No curves loaded for {dataset}")
    return curves


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cos_df = pd.read_csv(COS_PATH)
    cos_by_dataset = {
        (str(row["dataset"]), str(row["model"])): float(row["COS_from_log"])
        for _, row in cos_df.iterrows()
    }

    fig, axes = plt.subplots(1, 2, figsize=(16.69, 6.5), facecolor=TOKENS["surface"])
    for ax, dataset in zip(axes, DATASETS):
        display_dataset = DATASET_DISPLAY.get(dataset, dataset)
        df = load_dataset_curves(dataset)
        x = df["step"].to_numpy(dtype=np.float64)
        plotted = [df["True"].to_numpy(dtype=np.float64)]

        ax.plot(x, plotted[0], label="True", **LINE_STYLE["True"])
        for model in MODELS:
            y = df[model].to_numpy(dtype=np.float64)
            plotted.append(y)
            ax.plot(x, y, label=model_label(model, cos_by_dataset, display_dataset), **LINE_STYLE[model])

        ax.set_facecolor(TOKENS["panel"])
        ax.set_title(display_dataset, fontsize=28, color=TOKENS["ink"], pad=10)
        ax.set_xlabel("Test Time Step", color=TOKENS["ink"], fontsize=25, labelpad=9)
        ax.set_ylabel("Value", color=TOKENS["ink"], fontsize=25, labelpad=10)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        ax.yaxis.get_offset_text().set_fontsize(22)
        ax.xaxis.get_offset_text().set_fontsize(22)
        ax.tick_params(axis="both", colors=TOKENS["ink"], labelsize=22, width=1.0, length=5.5)
        ax.grid(True, linestyle="--", linewidth=0.75, color=TOKENS["grid"], alpha=0.86)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_xlim(-30, 630)
        ax.set_ylim(*axis_margin(np.concatenate(plotted)))
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(TOKENS["axis"])
            ax.spines[spine].set_linewidth(1.05)

        legend = ax.legend(
            ncol=2,
            loc="upper right",
            bbox_to_anchor=(0.998, 0.998),
            frameon=True,
            facecolor=TOKENS["panel"],
            edgecolor="#AEB4BE",
            framealpha=0.98,
            fontsize=16.2,
            handlelength=2.2,
            handletextpad=0.55,
            labelspacing=0.24,
            borderpad=0.35,
            borderaxespad=0.25,
        )
        legend.get_frame().set_linewidth(0.9)
        for text in legend.get_texts():
            text.set_color(TOKENS["ink"])

    fig.tight_layout(w_pad=2.4)
    pdf_path = FIG_DIR / f"{STEM}.pdf"
    png_path = FIG_DIR / f"{STEM}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
