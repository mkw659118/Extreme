from pathlib import Path
from typing import List
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"

COLORS = {
    "topk": "#F4A261",
    "router": "#6FA8DC",
    "axis": "#B8C0CC",
    "grid": "#E6EAF0",
    "ink": "#2F3746",
    "panel": "#FFFFFF",
}


def style_axis(ax) -> None:
    ax.set_facecolor(COLORS["panel"])
    ax.grid(True, axis="y", linestyle="--", color=COLORS["grid"], linewidth=0.75, alpha=0.9)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color(COLORS["axis"])
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(axis="both", colors=COLORS["ink"], labelsize=11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--usage_csv", default=str(ROOT / "data" / "moe_expert_usage.csv"))
    parser.add_argument("--suffix", default="scheme2")
    parser.add_argument("--title", default="Top-K Activation vs. Soft Router Probability")
    return parser.parse_args()


def plot_scheme2(pred_len: int = 5, usage_csv: str = str(ROOT / "data" / "moe_expert_usage.csv"), suffix: str = "scheme2", title: str = "Top-K Activation vs. Soft Router Probability") -> List[Path]:
    usage = pd.read_csv(usage_csv)
    usage = usage[usage["pred_len"] == pred_len].copy()
    datasets = list(usage["dataset"].drop_duplicates())
    num_experts = int(usage["expert"].max()) + 1
    x = np.arange(num_experts)

    fig, axes = plt.subplots(2, len(datasets), figsize=(11.8, 6.2), sharey="row")
    axes = np.atleast_2d(axes)

    for col, dataset in enumerate(datasets):
        part = usage[usage["dataset"] == dataset].sort_values("expert")

        ax = axes[0, col]
        ax.bar(x, part["topk_usage"], width=0.58, color=COLORS["topk"], edgecolor="white", linewidth=0.8)
        ax.set_title(dataset, fontsize=16, color=COLORS["ink"], pad=8)
        ax.set_xlabel("Expert", fontsize=12, color=COLORS["ink"])
        ax.set_xticks(x)
        ax.set_ylim(0, 1.06)
        style_axis(ax)
        for xi, yi in zip(x, part["topk_usage"]):
            ax.text(xi, yi + 0.025, f"{yi:.2f}", ha="center", va="bottom", fontsize=9, color=COLORS["ink"])

        ax = axes[1, col]
        ax.bar(x, part["weighted_usage"], width=0.58, color=COLORS["router"], edgecolor="white", linewidth=0.8)
        ax.set_xlabel("Expert", fontsize=12, color=COLORS["ink"])
        ax.set_xticks(x)
        ax.set_ylim(0, max(0.5, float(part["weighted_usage"].max()) * 1.22))
        style_axis(ax)
        for xi, yi in zip(x, part["weighted_usage"]):
            ax.text(xi, yi + 0.012, f"{yi:.2f}", ha="center", va="bottom", fontsize=9, color=COLORS["ink"])

    axes[0, 0].set_ylabel("Top-K Activation Ratio", fontsize=12, color=COLORS["ink"])
    axes[1, 0].set_ylabel("Mean Router Probability", fontsize=12, color=COLORS["ink"])
    fig.suptitle(title, fontsize=18, color=COLORS["ink"], y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=2.0, h_pad=1.8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"moe_topk_and_router_probability_PL{pred_len}_{suffix}.pdf"
    png_path = FIG_DIR / f"moe_topk_and_router_probability_PL{pred_len}_{suffix}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [pdf_path, png_path]


if __name__ == "__main__":
    args = parse_args()
    for path in plot_scheme2(pred_len=args.pred_len, usage_csv=args.usage_csv, suffix=args.suffix, title=args.title):
        print(path)
