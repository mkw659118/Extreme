from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "MoEExpertAnalysis"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

BASE_STEM = (
    "moe_sample_soft_routing_DATPNetStepRouterMoEWeightBalanced_clean_no_header_"
    "with_annot_1sample_1x2_no_sample_text_PL5_1samples_5steps"
)

PATCH_VALUES_PATH = DATA_DIR / f"{BASE_STEM}_patch_values.csv"
OUTPUT_STEM = f"{BASE_STEM}_transparent"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#B8C0CC",
}

DISPLAY_NAMES = {
    "Abilene": "Abilene",
    "Geant": "G\u00c9ANT",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": TOKENS["surface"],
        "axes.facecolor": TOKENS["panel"],
        "figure.dpi": 180,
    }
)


def main() -> None:
    patch_values = pd.read_csv(PATCH_VALUES_PATH)
    datasets = ["Abilene", "Geant"]
    expert_cols = ["E1", "E2", "E3", "E4"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.5, 3.7),
        squeeze=False,
        constrained_layout=False,
    )

    heatmap_values = []
    for dataset in datasets:
        dataset_values = patch_values[patch_values["dataset"] == dataset]
        sample_id = int(dataset_values["sample_id"].iloc[0])
        sample_values = dataset_values[dataset_values["sample_id"] == sample_id]
        heatmap_values.append(sample_values[expert_cols].to_numpy())

    vmax = max(0.5, float(np.max([arr.max() for arr in heatmap_values])))
    vmax = min(1.0, np.ceil(vmax * 10) / 10)
    cmap = sns.color_palette("YlGnBu", as_cmap=True)

    cbar_ax = fig.add_axes([0.92, 0.16, 0.018, 0.68])
    first_heatmap = True

    for col_idx, dataset in enumerate(datasets):
        ax = axes[0, col_idx]
        dataset_values = patch_values[patch_values["dataset"] == dataset]
        sample_id = int(dataset_values["sample_id"].iloc[0])
        sample_values = dataset_values[dataset_values["sample_id"] == sample_id].reset_index(drop=True)
        matrix = sample_values[expert_cols]

        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            annot_kws={"fontsize": 8.5},
            linewidths=0.4,
            linecolor="#F3F5FA",
            cbar=first_heatmap,
            cbar_ax=cbar_ax if first_heatmap else None,
            cbar_kws={"label": ""},
        )
        first_heatmap = False

        ax.set_xticklabels(expert_cols, rotation=0, fontsize=10)
        ax.set_yticklabels(sample_values["patch"], rotation=0, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(
            f"{DISPLAY_NAMES.get(dataset, dataset)} Expert Weight",
            fontsize=12,
            fontweight="bold",
            color=TOKENS["ink"],
            pad=8,
        )
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(TOKENS["axis"])
            ax.spines[spine].set_linewidth(0.8)

    fig.subplots_adjust(left=0.08, right=0.89, top=0.94, bottom=0.08, wspace=0.22, hspace=0.36)
    cbar_ax.tick_params(labelsize=9, colors=TOKENS["ink"])
    cbar_ax.set_ylabel("")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{OUTPUT_STEM}.pdf"
    png_path = FIG_DIR / f"{OUTPUT_STEM}.png"
    savefig_kwargs = {
        "bbox_inches": "tight",
        "transparent": True,
        "facecolor": "none",
        "edgecolor": "none",
    }
    fig.savefig(pdf_path, **savefig_kwargs)
    fig.savefig(png_path, dpi=300, **savefig_kwargs)
    plt.close(fig)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
