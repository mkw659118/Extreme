from pathlib import Path
import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

TOKENS = {
    "ink": "#2F3746",
    "axis": "#B8C0CC",
    "grid": "#E6EAF0",
    "panel": "#FFFFFF",
}

COLORS = {
    "topk": "#F4A261",
    "weighted": "#7BC8A4",
}


def style_axis(ax) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(True, axis="y", linestyle="--", color=TOKENS["grid"], linewidth=0.7, alpha=0.85)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(axis="both", colors=TOKENS["ink"], labelsize=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MoE expert usage without Top-1 bars.")
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--suffix", default="DATPNet")
    parser.add_argument("--usage_csv", default=None)
    parser.add_argument("--output_suffix", default="no_top1")
    return parser.parse_args()


def plot_usage_no_top1(
    usage_csv: Path,
    pred_len: int = 5,
    suffix: str = "DATPNet",
    output_suffix: str = "no_top1",
) -> List[Path]:
    usage = pd.read_csv(usage_csv)
    usage = usage[usage["pred_len"] == pred_len].copy()
    datasets = list(usage["dataset"].drop_duplicates())
    num_experts = int(usage["expert"].max()) + 1

    fig, axes = plt.subplots(1, len(datasets), figsize=(11.2, 4.1), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(num_experts)
    width = 0.34

    for ax, dataset in zip(axes, datasets):
        part = usage[usage["dataset"] == dataset].sort_values("expert")
        ax.bar(x - width / 2, part["topk_usage"], width, label="Top-K", color=COLORS["topk"])
        ax.bar(x + width / 2, part["weighted_usage"], width, label="Weighted", color=COLORS["weighted"])
        ax.set_title(dataset, fontsize=15, color=TOKENS["ink"])
        ax.set_xlabel("Expert", fontsize=12, color=TOKENS["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_ylim(0, 1.06)
        style_axis(ax)

    axes[0].set_ylabel("Usage Ratio", fontsize=12, color=TOKENS["ink"])
    axes[-1].legend(frameon=True, fontsize=10)
    fig.tight_layout(w_pad=2.0)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"moe_expert_usage_{suffix}_PL{pred_len}_{output_suffix}" if suffix else f"moe_expert_usage_PL{pred_len}_{output_suffix}"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [pdf_path, png_path]


def main() -> None:
    args = parse_args()
    usage_csv = Path(args.usage_csv) if args.usage_csv else DATA_DIR / f"moe_expert_usage_{args.suffix}.csv"
    for path in plot_usage_no_top1(
        usage_csv=usage_csv,
        pred_len=args.pred_len,
        suffix=args.suffix,
        output_suffix=args.output_suffix,
    ):
        print(path)


if __name__ == "__main__":
    main()
