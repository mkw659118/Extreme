from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator, MultipleLocator


ROOT = Path.cwd().parent if Path.cwd().name == "Ablation" else Path.cwd()
ABLATION_DIR = ROOT / "Ablation"
DATA_PATH = ABLATION_DIR / "parsed_hyperparameter_ablation_results_aggregated.csv"
OUT_DIR = ABLATION_DIR / "figures" / "num_experts" / "Abilene"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUT_DIR / "num_experts_Abilene_pred5_pred20_NMAE_combined.pdf"
PNG_PATH = OUT_DIR / "num_experts_Abilene_pred5_pred20_NMAE_combined.png"
CSV_PATH = OUT_DIR / "num_experts_Abilene_pred5_pred20_NMAE_combined.csv"
BROKEN_PDF_PATH = OUT_DIR / "num_experts_Abilene_pred5_pred20_NMAE_combined_broken_axis.pdf"
BROKEN_PNG_PATH = OUT_DIR / "num_experts_Abilene_pred5_pred20_NMAE_combined_broken_axis.png"
NORMAL_TICK_PDF_PATH = OUT_DIR / "num_experts_Abilene_pred5_pred20_NMAE_combined_normal_axis_ytick010.pdf"
NORMAL_TICK_PNG_PATH = OUT_DIR / "num_experts_Abilene_pred5_pred20_NMAE_combined_normal_axis_ytick010.png"
RELATIVE_PDF_PATH = OUT_DIR / "num_experts_Abilene_pred5_pred20_NMAE_relative_change_vs_1expert.pdf"
RELATIVE_PNG_PATH = OUT_DIR / "num_experts_Abilene_pred5_pred20_NMAE_relative_change_vs_1expert.png"
RELATIVE_CSV_PATH = OUT_DIR / "num_experts_Abilene_pred5_pred20_NMAE_relative_change_vs_1expert.csv"


TOKENS = {
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

SERIES_STYLE = {
    5: {"label": "PredLen=5", "color": "#5477C4", "marker": "o"},
    20: {"label": "PredLen=20", "color": "#B23A48", "marker": "^"},
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 180,
        "savefig.transparent": True,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
    }
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    sub = df[
        (df["experiment"] == "num_experts")
        & (df["Dataset"] == "Abilene")
        & (df["Pred_Len"].isin([5, 20]))
    ].copy()
    if sub.empty:
        raise ValueError("No matching rows found for num_experts / Abilene / Pred_Len in {5, 20}.")
    sub["Num_Experts"] = pd.to_numeric(sub["Num_Experts"])
    sub["NMAE"] = pd.to_numeric(sub["NMAE"])
    return sub.sort_values(["Pred_Len", "Num_Experts"])


def style_axis(ax) -> None:
    ax.grid(True, linestyle="--", linewidth=0.6, color=TOKENS["grid"], alpha=0.9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=TOKENS["ink"])
    ax.patch.set_alpha(0.0)


def plot_original(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8), facecolor="none")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    for pred_len in [5, 20]:
        part = data[data["Pred_Len"] == pred_len]
        style = SERIES_STYLE[pred_len]
        ax.plot(
            part["Num_Experts"],
            part["NMAE"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.9,
            markersize=6.2,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

    ax.set_xlabel("Number of Experts", color=TOKENS["ink"])
    ax.set_ylabel("NMAE", color=TOKENS["ink"])
    ax.set_xticks(sorted(data["Num_Experts"].unique()))
    ax.set_ylim(0.49, 0.75)
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.margins(x=0.08, y=0.14)
    style_axis(ax)
    ax.legend(frameon=False, loc="best", fontsize=9.2, handlelength=2.2)

    fig.tight_layout()
    fig.savefig(PDF_PATH, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(PNG_PATH, bbox_inches="tight", dpi=300, transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(NORMAL_TICK_PDF_PATH, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(NORMAL_TICK_PNG_PATH, bbox_inches="tight", dpi=300, transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)


def plot_broken_axis(data: pd.DataFrame) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(5.8, 4.25),
        facecolor="none",
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
    )
    fig.patch.set_alpha(0.0)

    for ax in [ax_top, ax_bottom]:
        for pred_len in [5, 20]:
            part = data[data["Pred_Len"] == pred_len]
            style = SERIES_STYLE[pred_len]
            ax.plot(
                part["Num_Experts"],
                part["NMAE"],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.9,
                markersize=6.2,
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
        style_axis(ax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    y_low = data[data["Pred_Len"] == 5]["NMAE"]
    y_high = data[data["Pred_Len"] == 20]["NMAE"]
    low_pad = max((float(y_low.max()) - float(y_low.min())) * 0.45, 0.003)
    high_pad = max((float(y_high.max()) - float(y_high.min())) * 0.45, 0.003)
    ax_bottom.set_ylim(float(y_low.min()) - low_pad, float(y_low.max()) + low_pad)
    ax_top.set_ylim(float(y_high.min()) - high_pad, float(y_high.max()) + high_pad)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_bottom.set_xlabel("Number of Experts", color=TOKENS["ink"])
    fig.text(0.025, 0.52, "NMAE", va="center", rotation="vertical", color=TOKENS["ink"])
    ax_bottom.set_xticks(sorted(data["Num_Experts"].unique()))

    d = 0.010
    break_style = dict(color=TOKENS["ink"], clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **break_style)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **break_style)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **break_style)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **break_style)

    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend(handles[:2], labels[:2], frameon=False, loc="upper right", fontsize=9.0, handlelength=2.2)

    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.16, top=0.96, hspace=0.08)
    fig.savefig(BROKEN_PDF_PATH, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(BROKEN_PNG_PATH, bbox_inches="tight", dpi=300, transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)


def plot_relative_change(data: pd.DataFrame) -> None:
    rows = []
    for pred_len in [5, 20]:
        part = data[data["Pred_Len"] == pred_len].copy()
        baseline = float(part.loc[part["Num_Experts"] == 1, "NMAE"].iloc[0])
        part["relative_change_percent"] = (part["NMAE"] - baseline) / baseline * 100.0
        rows.append(part)
    rel = pd.concat(rows, ignore_index=True)
    rel[["Dataset", "Pred_Len", "Num_Experts", "NMAE", "relative_change_percent"]].to_csv(
        RELATIVE_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(5.6, 3.8), facecolor="none")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    for pred_len in [5, 20]:
        part = rel[rel["Pred_Len"] == pred_len]
        style = SERIES_STYLE[pred_len]
        ax.plot(
            part["Num_Experts"],
            part["relative_change_percent"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.9,
            markersize=6.2,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

    y_min = float(rel["relative_change_percent"].min())
    y_max = float(rel["relative_change_percent"].max())
    pad = max((y_max - y_min) * 0.18, 0.35)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.axhline(0, color=TOKENS["muted"], linewidth=0.9, alpha=0.85)
    ax.set_xlabel("Number of Experts", color=TOKENS["ink"])
    ax.set_ylabel("Relative NMAE Change (%)", color=TOKENS["ink"])
    ax.set_xticks(sorted(data["Num_Experts"].unique()))
    ax.margins(x=0.08)
    style_axis(ax)
    ax.legend(frameon=False, loc="best", fontsize=9.2, handlelength=2.2)

    fig.tight_layout()
    fig.savefig(RELATIVE_PDF_PATH, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(RELATIVE_PNG_PATH, bbox_inches="tight", dpi=300, transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)


def main() -> None:
    data = load_data()
    data[["Dataset", "Pred_Len", "Num_Experts", "NMAE"]].to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    plot_original(data)
    plot_relative_change(data)

    print(f"saved: {PDF_PATH}")
    print(f"saved: {PNG_PATH}")
    print(f"saved: {NORMAL_TICK_PDF_PATH}")
    print(f"saved: {NORMAL_TICK_PNG_PATH}")
    print(f"saved: {RELATIVE_PDF_PATH}")
    print(f"saved: {RELATIVE_PNG_PATH}")
    print(f"saved: {RELATIVE_CSV_PATH}")
    print(f"saved: {CSV_PATH}")


if __name__ == "__main__":
    main()
