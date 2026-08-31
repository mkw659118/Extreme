from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


RESULTS_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "Abilene_GEANT_PL5_DM256_mlu_ratio_all_baselines_bold_true_datp_mean_legend_1x2_updated_600pts_long_true_legend"
MEAN_PATH = RESULTS_DIR / f"{OUTPUT_STEM}_mean_values.csv"
LEGEND_FRAME_ALPHA = 0.98

DATASETS = ["Abilene", "GEANT"]
DATASET_DISPLAY = {"Abilene": "Abilene", "GEANT": "GÉANT"}
PRED_DIR = "PL5_DM256"
MODELS = [
    "DATP-Net",
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
MODEL_FILE = {
    "DATP-Net": {"Abilene": "DATP-Net_pred.csv", "GEANT": "datp_net_multi_pred.csv"},
    "PMDformer": "PMDformer_pred.csv",
    "HMformer": "HMformer_pred.csv",
    "FeTS": "FeTS_pred.csv",
    "TimesNet": "timesnet_pred.csv",
    "iTransformer": "iTransformer_pred.csv",
    "PatchTST": "PatchTST_pred.csv",
    "WPMixer": "WPMixer_pred.csv",
    "P_sLSTM": "P_sLSTM_pred.csv",
    "xLSTMTime": "xLSTMTime_pred.csv",
    "xLSTM-Mixer": "xlstm_mixer_pred.csv",
    "FEDformer": "FEDformer_pred.csv",
}

TOKENS = {
    "surface": "#FFFFFF",
    "panel": "#FFFFFF",
    "ink": "#000000",
    "grid": "#E2E5EA",
    "axis": "#444444",
}

LINE_STYLE = {
    "True": dict(color="#3F3F3F", linewidth=3.9, linestyle="--", alpha=0.96, zorder=30),
    "DATP-Net": dict(color="#FF8C29", linewidth=3.6, linestyle="-", alpha=0.96, zorder=28),
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


def data_file(dataset: str, model: str) -> Path:
    filename = MODEL_FILE[model]
    if isinstance(filename, dict):
        filename = filename[dataset]
    return RESULTS_DIR / dataset / PRED_DIR / "mlu" / filename


def axis_margin(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    pad = max((vmax - vmin) * 0.08, 0.015)
    return vmin - pad, vmax + pad


def main(output_stem: str = OUTPUT_STEM, legend_frame_alpha: float = LEGEND_FRAME_ALPHA) -> None:
    mean_df = pd.read_csv(MEAN_PATH)
    mean_lookup = {
        (str(row["dataset"]), str(row["model"])): float(row["mean_mlu_ratio"])
        for _, row in mean_df.iterrows()
    }

    fig, axes = plt.subplots(1, 2, figsize=(16.69, 6.51), facecolor=TOKENS["surface"])
    for ax, dataset in zip(axes, DATASETS):
        display_dataset = DATASET_DISPLAY[dataset]
        true_df = pd.read_csv(RESULTS_DIR / dataset / PRED_DIR / "mlu" / "true.csv")
        true_mlu = true_df["mlu"].to_numpy(dtype=np.float64)
        x = np.arange(len(true_mlu), dtype=np.float64)
        plotted = [np.ones_like(true_mlu)]

        ax.plot(x, plotted[0], label="True", **LINE_STYLE["True"])
        for model in MODELS:
            pred_df = pd.read_csv(data_file(dataset, model))
            ratio = pred_df["mlu"].to_numpy(dtype=np.float64) / true_mlu
            plotted.append(ratio)
            mean_value = mean_lookup[(display_dataset, model)]
            ax.plot(x, ratio, label=f"{model} ({mean_value:.3f})", **LINE_STYLE[model])

        ax.set_facecolor(TOKENS["panel"])
        ax.set_title(display_dataset, fontsize=28, color=TOKENS["ink"], pad=10)
        ax.set_xlabel("Test Time Step", color=TOKENS["ink"], fontsize=25, labelpad=9)
        ax.set_ylabel("MLU Ratio", color=TOKENS["ink"], fontsize=25, labelpad=10)
        ax.tick_params(axis="both", colors=TOKENS["ink"], labelsize=22, width=1.0, length=5.5)
        ax.grid(True, linestyle="--", linewidth=0.75, color=TOKENS["grid"], alpha=0.86)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_xlim(-5, len(true_mlu) + 5)
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
            framealpha=legend_frame_alpha,
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
    pdf_path = RESULTS_DIR / f"{output_stem}.pdf"
    png_path = RESULTS_DIR / f"{output_stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
