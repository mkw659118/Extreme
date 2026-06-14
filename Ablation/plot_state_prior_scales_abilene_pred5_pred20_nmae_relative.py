from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator


ROOT = Path.cwd().parent if Path.cwd().name == "Ablation" else Path.cwd()
ABLATION_DIR = ROOT / "Ablation"
DATA_PATH = ABLATION_DIR / "parsed_hyperparameter_ablation_results_aggregated.csv"
OUT_DIR = ABLATION_DIR / "figures" / "state_prior_scales" / "Abilene"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUT_DIR / "state_prior_scales_Abilene_pred5_pred20_NMAE_relative_change_vs_scale1.pdf"
PNG_PATH = OUT_DIR / "state_prior_scales_Abilene_pred5_pred20_NMAE_relative_change_vs_scale1.png"
CSV_PATH = OUT_DIR / "state_prior_scales_Abilene_pred5_pred20_NMAE_relative_change_vs_scale1.csv"
RAW_PDF_PATH = OUT_DIR / "state_prior_scales_Abilene_pred5_pred20_NMAE_combined.pdf"
RAW_PNG_PATH = OUT_DIR / "state_prior_scales_Abilene_pred5_pred20_NMAE_combined.png"

STATE_ORDER = [
    "1",
    "1,4",
    "1,4,8",
    "1,4,8,16",
    "1,4,8,16,32",
    "1,4,8,16_no_seq",
]

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
        (df["experiment"] == "state_prior_scales")
        & (df["Dataset"] == "Abilene")
        & (df["Pred_Len"].isin([5, 20]))
    ].copy()
    if sub.empty:
        raise ValueError("No matching rows found for state_prior_scales / Abilene / Pred_Len in {5, 20}.")

    sub["State_Prior_Setting"] = sub["State_Prior_Setting"].astype(str)
    sub["NMAE"] = pd.to_numeric(sub["NMAE"])
    missing = sorted(set(sub["State_Prior_Setting"]) - set(STATE_ORDER))
    if missing:
        raise ValueError(f"Unexpected state-prior settings: {missing}")

    sub["State_Order"] = sub["State_Prior_Setting"].map({name: i for i, name in enumerate(STATE_ORDER)})
    return sub.sort_values(["Pred_Len", "State_Order"])


def style_axis(ax) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, color=TOKENS["grid"], alpha=0.9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=TOKENS["ink"])
    ax.patch.set_alpha(0.0)


def build_relative_data(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pred_len in [5, 20]:
        part = data[data["Pred_Len"] == pred_len].copy()
        baseline_setting = STATE_ORDER[0]
        baseline = float(part.loc[part["State_Prior_Setting"] == baseline_setting, "NMAE"].iloc[0])
        part["baseline_state_prior_setting"] = baseline_setting
        part["baseline_nmae"] = baseline
        part["relative_change_percent"] = (part["NMAE"] - baseline) / baseline * 100.0
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def plot_raw(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.9), facecolor="none")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    x_positions = list(range(len(STATE_ORDER)))
    for pred_len in [5, 20]:
        part = data[data["Pred_Len"] == pred_len]
        style = SERIES_STYLE[pred_len]
        ax.plot(
            part["State_Order"],
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

    ax.set_ylim(0.50, 0.74)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel("State prior scales", color=TOKENS["ink"])
    ax.set_ylabel("NMAE", color=TOKENS["ink"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(STATE_ORDER, rotation=18, ha="right")
    ax.margins(x=0.04)
    style_axis(ax)
    ax.legend(frameon=False, loc="best", fontsize=9.2, handlelength=2.2)

    fig.tight_layout()
    fig.savefig(RAW_PDF_PATH, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(RAW_PNG_PATH, bbox_inches="tight", dpi=300, transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)


def main() -> None:
    data = load_data()
    rel = build_relative_data(data)
    rel[
        [
            "Dataset",
            "Pred_Len",
            "State_Prior_Setting",
            "NMAE",
            "baseline_state_prior_setting",
            "baseline_nmae",
            "relative_change_percent",
        ]
    ].to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6.8, 3.9), facecolor="none")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    x_positions = list(range(len(STATE_ORDER)))
    for pred_len in [5, 20]:
        part = rel[rel["Pred_Len"] == pred_len]
        style = SERIES_STYLE[pred_len]
        ax.plot(
            part["State_Order"],
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
    pad = max((y_max - y_min) * 0.2, 0.12)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.axhline(0, color=TOKENS["muted"], linewidth=0.9, alpha=0.85)
    ax.set_xlabel("State prior scales", color=TOKENS["ink"])
    ax.set_ylabel("Relative NMAE Change (%)", color=TOKENS["ink"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(STATE_ORDER, rotation=18, ha="right")
    ax.margins(x=0.04)
    style_axis(ax)
    ax.legend(frameon=False, loc="best", fontsize=9.2, handlelength=2.2)

    fig.tight_layout()
    fig.savefig(PDF_PATH, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(PNG_PATH, bbox_inches="tight", dpi=300, transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)

    plot_raw(data)

    print(f"saved: {PDF_PATH}")
    print(f"saved: {PNG_PATH}")
    print(f"saved: {CSV_PATH}")
    print(f"saved: {RAW_PDF_PATH}")
    print(f"saved: {RAW_PNG_PATH}")


if __name__ == "__main__":
    main()
