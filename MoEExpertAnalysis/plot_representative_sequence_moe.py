from pathlib import Path
import argparse
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
    "Top-1": "#6FA8DC",
    "Top-K": "#F4A261",
    "Weighted": "#7BC8A4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MoE routing on one representative continuous sequence.")
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--suffix", default="DATPNet")
    parser.add_argument("--datasets", nargs="+", default=["Abilene", "Geant"])
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--num_experts", type=int, default=4)
    return parser.parse_args()


def style_axis(ax) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(True, axis="y", linestyle="--", color=TOKENS["grid"], linewidth=0.75, alpha=0.9)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(axis="both", colors=TOKENS["ink"], labelsize=10)


def load_dataset(dataset: str, pred_len: int, suffix: str) -> pd.DataFrame:
    path = DATA_DIR / f"{dataset}_PL{pred_len}_moe_routing_samples_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df.sort_values("sample_id").reset_index(drop=True)


def select_representative_window(df: pd.DataFrame, window: int, num_experts: int) -> pd.DataFrame:
    if len(df) <= window:
        return df.copy()

    best_score = None
    best_start = 0
    topk_cols = [f"topk_contains_e{i}" for i in range(num_experts)]

    for start in range(0, len(df) - window + 1):
        part = df.iloc[start : start + window]
        coverage = (part[topk_cols].sum(axis=0) > 0).sum()
        coverage_bonus = 1000.0 * coverage
        change_score = float(part["future_abs_diff_max"].mean()) if "future_abs_diff_max" in part else 0.0
        entropy_score = float(part["routing_entropy"].mean()) if "routing_entropy" in part else 0.0
        score = coverage_bonus + change_score + 0.01 * entropy_score
        if best_score is None or score > best_score:
            best_score = score
            best_start = start

    return df.iloc[best_start : best_start + window].copy()


def summarize_usage(part: pd.DataFrame, num_experts: int) -> pd.DataFrame:
    rows = []
    top1 = part["top1_expert"].astype(int).to_numpy()
    for expert in range(num_experts):
        rows.append(
            {
                "expert": expert,
                "top1_usage": float(np.mean(top1 == expert)),
                "topk_usage": float(part[f"topk_contains_e{expert}"].mean()),
                "weighted_usage": float(part[f"router_prob_e{expert}"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_usage(selected: Dict[str, pd.DataFrame], pred_len: int, suffix: str, num_experts: int) -> Path:
    datasets = list(selected)
    fig, axes = plt.subplots(1, len(datasets), figsize=(11.6, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(num_experts)

    for ax, dataset in zip(axes, datasets):
        part = selected[dataset]
        usage = summarize_usage(part, num_experts)
        bars = ax.bar(x, usage["topk_usage"], width=0.56, label="Top-K Activation", color=COLORS["Top-K"])
        for bar, value in zip(bars, usage["topk_usage"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=TOKENS["ink"],
            )
        start_id = int(part["sample_id"].min())
        end_id = int(part["sample_id"].max())
        ax.set_title(f"{dataset} [{start_id}-{end_id}]", fontsize=14, color=TOKENS["ink"])
        ax.set_xlabel("Expert", fontsize=12, color=TOKENS["ink"])
        ax.set_xticks(x)
        ax.set_ylim(0, 1.08)
        style_axis(ax)

    axes[0].set_ylabel("Top-K Activation Ratio", fontsize=12, color=TOKENS["ink"])
    axes[-1].legend(frameon=True, fontsize=10)
    fig.suptitle("Representative Sequence Top-K Expert Activation", fontsize=17, color=TOKENS["ink"], y=1.02)
    fig.tight_layout(w_pad=2.0)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"moe_representative_sequence_topk_usage_{suffix}_PL{pred_len}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def plot_top1_timeline(selected: Dict[str, pd.DataFrame], pred_len: int, suffix: str, num_experts: int) -> Path:
    datasets = list(selected)
    fig, axes = plt.subplots(len(datasets), 1, figsize=(11.6, 4.8), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, dataset in zip(axes, datasets):
        part = selected[dataset]
        x = part["sample_id"].to_numpy()
        top1 = part["top1_expert"].astype(int).to_numpy()
        ax.step(x, top1, where="post", color="#0072B2", linewidth=2.0)
        ax.scatter(x, top1, c=top1, cmap="tab10", s=18, alpha=0.85)
        ax.set_title(dataset, fontsize=14, color=TOKENS["ink"])
        ax.set_ylabel("Top-1 Expert", fontsize=11, color=TOKENS["ink"])
        ax.set_yticks(np.arange(num_experts))
        ax.set_ylim(-0.5, num_experts - 0.5)
        style_axis(ax)

    axes[-1].set_xlabel("Sample ID in Selected Sequence", fontsize=12, color=TOKENS["ink"])
    fig.suptitle("Representative Sequence Top-1 Expert Timeline", fontsize=17, color=TOKENS["ink"], y=1.02)
    fig.tight_layout(h_pad=1.4)

    path = FIG_DIR / f"moe_representative_sequence_top1_timeline_{suffix}_PL{pred_len}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def plot_topk_timeline(selected: Dict[str, pd.DataFrame], pred_len: int, suffix: str, num_experts: int) -> Path:
    datasets = list(selected)
    fig, axes = plt.subplots(len(datasets), 1, figsize=(11.6, 4.8), sharex=False)
    axes = np.atleast_1d(axes)
    cmap = ListedColormap(["#FFF8D6", "#D55E00"])

    for ax, dataset in zip(axes, datasets):
        part = selected[dataset].reset_index(drop=True)
        mat = np.vstack([part[f"topk_contains_e{expert}"].to_numpy(dtype=float) for expert in range(num_experts)])
        ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
        tick_positions = np.linspace(0, len(part) - 1, num=6, dtype=int)
        tick_labels = part.loc[tick_positions, "sample_id"].astype(int).astype(str).tolist()
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(np.arange(num_experts))
        ax.set_yticklabels([f"Expert {i}" for i in range(num_experts)])
        ax.set_title(dataset, fontsize=14, color=TOKENS["ink"])
        ax.set_ylabel("Top-K Expert", fontsize=11, color=TOKENS["ink"])
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color(TOKENS["axis"])
            ax.spines[spine].set_linewidth(1.0)
        ax.tick_params(axis="both", colors=TOKENS["ink"], labelsize=10)

    axes[-1].set_xlabel("Sample ID in Selected Sequence", fontsize=12, color=TOKENS["ink"])
    fig.suptitle("Representative Sequence Top-K Activation Timeline", fontsize=17, color=TOKENS["ink"], y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=1.4)

    path = FIG_DIR / f"moe_representative_sequence_topk_timeline_{suffix}_PL{pred_len}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    selected = {}
    summary_rows = []

    for dataset in args.datasets:
        df = load_dataset(dataset, args.pred_len, args.suffix)
        part = select_representative_window(df, args.window, args.num_experts)
        selected[dataset] = part
        usage = summarize_usage(part, args.num_experts)
        for _, row in usage.iterrows():
            summary_rows.append(
                {
                    "dataset": dataset,
                    "pred_len": args.pred_len,
                    "window": len(part),
                    "sample_start": int(part["sample_id"].min()),
                    "sample_end": int(part["sample_id"].max()),
                    **row.to_dict(),
                }
            )

    summary = pd.DataFrame(summary_rows)
    out_csv = DATA_DIR / f"moe_representative_sequence_usage_{args.suffix}_PL{args.pred_len}.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    paths = [
        plot_usage(selected, args.pred_len, args.suffix, args.num_experts),
        plot_top1_timeline(selected, args.pred_len, args.suffix, args.num_experts),
        plot_topk_timeline(selected, args.pred_len, args.suffix, args.num_experts),
    ]
    print(out_csv)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
