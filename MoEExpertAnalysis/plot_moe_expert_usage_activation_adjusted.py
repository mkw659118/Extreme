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
    "adjusted_weight": "#7BC8A4",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MoE Top-K usage with activation-adjusted weights.")
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--suffix", default="DATPNet")
    parser.add_argument("--usage_csv", default=None)
    parser.add_argument("--output_suffix", default="activation_adjusted_weight")
    parser.add_argument("--adjust_strength", type=float, default=0.18)
    return parser.parse_args()


def style_axis(ax) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(True, axis="y", linestyle="--", color=TOKENS["grid"], linewidth=0.7, alpha=0.85)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(axis="both", colors=TOKENS["ink"], labelsize=10)


def display_dataset_name(dataset: str) -> str:
    return "G\u00c9ANT" if dataset == "Geant" else dataset


def add_activation_adjusted_weight(usage: pd.DataFrame, adjust_strength: float) -> pd.DataFrame:
    rows = []
    for (dataset, pred_len), part in usage.groupby(["dataset", "pred_len"], sort=False):
        part = part.sort_values("expert").copy()
        topk_center = float(part["topk_usage"].mean())
        part["activation_adjusted_weight"] = (
            part["weighted_usage"] + adjust_strength * (part["topk_usage"] - topk_center)
        ).clip(lower=0.0, upper=1.0)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def plot_activation_adjusted(
    usage_csv: Path,
    pred_len: int = 5,
    suffix: str = "DATPNet",
    output_suffix: str = "activation_adjusted_weight",
    adjust_strength: float = 0.18,
) -> List[Path]:
    usage = pd.read_csv(usage_csv)
    usage = usage[usage["pred_len"] == pred_len].copy()
    usage = add_activation_adjusted_weight(usage, adjust_strength=adjust_strength)
    datasets = list(usage["dataset"].drop_duplicates())
    num_experts = int(usage["expert"].max()) + 1

    fig, axes = plt.subplots(1, len(datasets), figsize=(11.2, 4.1), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(num_experts)
    width = 0.34

    for ax, dataset in zip(axes, datasets):
        part = usage[usage["dataset"] == dataset].sort_values("expert")
        ax.bar(x - width / 2, part["topk_usage"], width, label="Top-K", color=COLORS["topk"])
        ax.bar(
            x + width / 2,
            part["activation_adjusted_weight"],
            width,
            label="Activation-Adjusted Weight",
            color=COLORS["adjusted_weight"],
        )
        ax.set_title(display_dataset_name(dataset), fontsize=15, color=TOKENS["ink"])
        ax.set_xlabel("Expert", fontsize=12, color=TOKENS["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_ylim(0, 1.06)
        style_axis(ax)

    axes[0].set_ylabel("Usage Ratio", fontsize=12, color=TOKENS["ink"])
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=True,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=2.0)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"moe_expert_usage_{suffix}_PL{pred_len}_{output_suffix}" if suffix else f"moe_expert_usage_PL{pred_len}_{output_suffix}"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    csv_path = FIG_DIR / f"{stem}.csv"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    usage.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return [pdf_path, png_path, csv_path]


def main() -> None:
    args = parse_args()
    usage_csv = Path(args.usage_csv) if args.usage_csv else DATA_DIR / f"moe_expert_usage_{args.suffix}.csv"
    for path in plot_activation_adjusted(
        usage_csv=usage_csv,
        pred_len=args.pred_len,
        suffix=args.suffix,
        output_suffix=args.output_suffix,
        adjust_strength=args.adjust_strength,
    ):
        print(path)


if __name__ == "__main__":
    main()
