from pathlib import Path
import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

TOKENS = {
    "surface": "#FFFFFF",
    "panel": "#FFFFFF",
    "ink": "#2F3746",
    "axis": "#B8C0CC",
    "grid": "#E6EAF0",
}

COLORS = {
    "selection": "#F4A261",
    "selected_mix": "#6FA8DC",
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
    parser = argparse.ArgumentParser(description="Plot consistent MoE Top-K selection and selected-mix shares.")
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--suffix", default="DATPNet")
    parser.add_argument("--datasets", nargs="+", default=["Abilene", "Geant"])
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--output_suffix", default="consistent_selection_share")
    return parser.parse_args()


def style_axis(ax) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(True, axis="y", linestyle="--", color=TOKENS["grid"], linewidth=0.75, alpha=0.9)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(axis="both", colors=TOKENS["ink"], labelsize=11)


def display_dataset_name(dataset: str) -> str:
    return "G\u00c9ANT" if dataset == "Geant" else dataset


def load_routing(dataset: str, pred_len: int, suffix: str) -> pd.DataFrame:
    path = DATA_DIR / f"{dataset}_PL{pred_len}_moe_routing_samples_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def compute_selection_summary(df: pd.DataFrame, dataset: str, pred_len: int, num_experts: int) -> pd.DataFrame:
    topk_cols = [f"topk_contains_e{i}" for i in range(num_experts)]
    prob_cols = [f"router_prob_e{i}" for i in range(num_experts)]
    missing = [col for col in topk_cols + prob_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset} PL{pred_len} missing columns: {missing}")

    selected = df[topk_cols].to_numpy(dtype=np.float64)
    router_prob = df[prob_cols].to_numpy(dtype=np.float64)
    selected_prob = selected * router_prob
    denom = selected_prob.sum(axis=1, keepdims=True)
    selected_mix = np.divide(
        selected_prob,
        np.maximum(denom, 1e-12),
        out=np.zeros_like(selected_prob),
        where=denom > 0,
    )

    total_selections = selected.sum()
    total_mix = selected_mix.sum()
    rows = []
    for expert in range(num_experts):
        selection_share = selected[:, expert].sum() / total_selections if total_selections > 0 else 0.0
        selected_mix_share = selected_mix[:, expert].sum() / total_mix if total_mix > 0 else 0.0
        rows.append(
            {
                "dataset": dataset,
                "pred_len": pred_len,
                "expert": expert,
                "selection_share": float(selection_share),
                "selected_mix_share": float(selected_mix_share),
                "topk_activation_ratio": float(selected[:, expert].mean()),
                "mean_router_prob": float(router_prob[:, expert].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_selection_share(summary: pd.DataFrame, pred_len: int, suffix: str, output_suffix: str) -> List[Path]:
    datasets = list(summary["dataset"].drop_duplicates())
    num_experts = int(summary["expert"].max()) + 1
    x = np.arange(num_experts)
    width = 0.34

    fig, axes = plt.subplots(1, len(datasets), figsize=(11.2, 4.15), sharey=True, facecolor=TOKENS["surface"])
    axes = np.atleast_1d(axes)

    for ax, dataset in zip(axes, datasets):
        part = summary[summary["dataset"] == dataset].sort_values("expert")
        bars1 = ax.bar(
            x - width / 2,
            part["selection_share"],
            width,
            label="Selection Share",
            color=COLORS["selection"],
            edgecolor="white",
            linewidth=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            part["selected_mix_share"],
            width,
            label="Selected Mix Share",
            color=COLORS["selected_mix"],
            edgecolor="white",
            linewidth=0.8,
        )
        for bars, values in [(bars1, part["selection_share"]), (bars2, part["selected_mix_share"])]:
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.012,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=TOKENS["ink"],
                )
        ax.set_title(display_dataset_name(dataset), fontsize=16, color=TOKENS["ink"], pad=8)
        ax.set_xlabel("Expert", fontsize=12, color=TOKENS["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_ylim(0, 0.58)
        style_axis(ax)

    axes[0].set_ylabel("Share Across Selected Experts", fontsize=12, color=TOKENS["ink"])
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
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return [pdf_path, png_path, csv_path]


def main() -> None:
    args = parse_args()
    frames = [
        compute_selection_summary(
            load_routing(dataset, args.pred_len, args.suffix),
            dataset,
            args.pred_len,
            args.num_experts,
        )
        for dataset in args.datasets
    ]
    summary = pd.concat(frames, ignore_index=True)
    for path in plot_selection_share(summary, args.pred_len, args.suffix, args.output_suffix):
        print(path)


if __name__ == "__main__":
    main()
