from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from regenerate_full_test_all_baselines_bigfont import (
    COS_PATH,
    DATASET_DISPLAY,
    DATASETS,
    FIG_DIR,
    LINE_STYLE,
    MODELS,
    TOKENS,
    axis_margin,
    load_dataset_curves,
    model_label,
)


OUTPUT_STEM = (
    "Abilene_Geant_PL5_prediction_full_test_all_baselines_bold_true_datp_cos_from_log_"
    "legend_inside_upper_center_biglegend_600pts"
)


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
            label = model_label(model, cos_by_dataset, display_dataset)
            ax.plot(x, y, label=label, **LINE_STYLE[model])

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
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            frameon=True,
            facecolor=TOKENS["panel"],
            edgecolor="#AEB4BE",
            framealpha=0.98,
            fontsize=16.2,
            handlelength=1.6,
            handletextpad=0.42,
            columnspacing=0.65,
            labelspacing=0.24,
            borderpad=0.32,
            borderaxespad=0.2,
        )
        legend.get_frame().set_linewidth(0.9)
        for text in legend.get_texts():
            text.set_color(TOKENS["ink"])

    fig.tight_layout(w_pad=2.4)
    pdf_path = FIG_DIR / f"{OUTPUT_STEM}.pdf"
    png_path = FIG_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
