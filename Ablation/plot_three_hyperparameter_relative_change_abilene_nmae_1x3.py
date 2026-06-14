from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator


ROOT = Path.cwd().parent if Path.cwd().name == "Ablation" else Path.cwd()
ABLATION_DIR = ROOT / "Ablation"
FIG_DIR = ABLATION_DIR / "figures"

INPUTS = {
    "num_experts": FIG_DIR
    / "num_experts"
    / "Abilene"
    / "num_experts_Abilene_pred5_pred20_NMAE_relative_change_vs_1expert.csv",
    "retrieval_topk": FIG_DIR
    / "retrieval_topk"
    / "Abilene"
    / "retrieval_topk_Abilene_pred5_pred20_NMAE_relative_change_vs_topk1.csv",
    "state_prior_scales": FIG_DIR
    / "state_prior_scales"
    / "Abilene"
    / "state_prior_scales_Abilene_pred5_pred20_NMAE_relative_change_vs_scale1.csv",
}

OUT_DIR = FIG_DIR / "combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUT_DIR / "Abilene_three_hyperparameter_relative_change_NMAE_1x3_coded_scales.pdf"
PNG_PATH = OUT_DIR / "Abilene_three_hyperparameter_relative_change_NMAE_1x3_coded_scales.png"
CSV_PATH = OUT_DIR / "Abilene_three_hyperparameter_relative_change_NMAE_1x3_coded_scales.csv"
SCALE_CODE_MAP_PATH = OUT_DIR / "Abilene_state_prior_scale_code_mapping.csv"

STATE_ORDER = [
    "1",
    "1,4",
    "1,4,8",
    "1,4,8,16",
    "1,4,8,16,32",
    "1,4,8,16_no_seq",
]

STATE_CODE_LABELS = {setting: chr(ord("A") + idx) for idx, setting in enumerate(STATE_ORDER)}

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

PANELS = [
    {
        "key": "num_experts",
        "title": "(a) Number of experts",
        "x_col": "Num_Experts",
        "x_label": "Number of experts",
        "x_tick_rotation": 0,
        "y_step": 1.0,
    },
    {
        "key": "retrieval_topk",
        "title": "(b) Retrieval topK",
        "x_col": "Retrieval_Num",
        "x_label": "Retrieval topK",
        "x_tick_rotation": 0,
        "y_step": 0.1,
    },
    {
        "key": "state_prior_scales",
        "title": "(c) State prior scales",
        "x_col": "State_Prior_Setting",
        "x_label": "Scale setting code",
        "x_tick_rotation": 0,
        "y_step": 1.0,
    },
]

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


def load_panel_data(panel: dict) -> pd.DataFrame:
    path = INPUTS[panel["key"]]
    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(path)
    data["experiment"] = panel["key"]
    data["panel_title"] = panel["title"]
    data["relative_change_percent"] = pd.to_numeric(data["relative_change_percent"])
    data["Pred_Len"] = pd.to_numeric(data["Pred_Len"])

    x_col = panel["x_col"]
    if panel["key"] == "state_prior_scales":
        data[x_col] = data[x_col].astype(str)
        data["x_order"] = data[x_col].map({name: i for i, name in enumerate(STATE_ORDER)})
        data["x_label"] = data[x_col].map(STATE_CODE_LABELS)
        data["state_prior_scale_code"] = data["x_label"]
    else:
        data[x_col] = pd.to_numeric(data[x_col])
        values = sorted(data[x_col].unique())
        data["x_order"] = data[x_col].map({value: i for i, value in enumerate(values)})
        data["x_label"] = data[x_col].map(lambda value: f"{int(value)}")

    return data.sort_values(["Pred_Len", "x_order"])


def style_axis(ax) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, color=TOKENS["grid"], alpha=0.9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=TOKENS["ink"], labelsize=8.5)
    ax.patch.set_alpha(0.0)


def set_panel_ylim(ax, data: pd.DataFrame, step: float) -> None:
    y_min = float(data["relative_change_percent"].min())
    y_max = float(data["relative_change_percent"].max())
    pad = max((y_max - y_min) * 0.18, step * 0.35)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.yaxis.set_major_locator(MultipleLocator(step))


def main() -> None:
    panel_data = {panel["key"]: load_panel_data(panel) for panel in PANELS}
    combined = pd.concat(panel_data.values(), ignore_index=True)
    combined.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "state_prior_scale_code": [STATE_CODE_LABELS[setting] for setting in STATE_ORDER],
            "State_Prior_Setting": STATE_ORDER,
        }
    ).to_csv(SCALE_CODE_MAP_PATH, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6), facecolor="none")
    fig.patch.set_alpha(0.0)

    legend_handles = []
    legend_labels = []
    for ax, panel in zip(axes, PANELS):
        data = panel_data[panel["key"]]
        for pred_len in [5, 20]:
            part = data[data["Pred_Len"] == pred_len]
            style = SERIES_STYLE[pred_len]
            (line,) = ax.plot(
                part["x_order"],
                part["relative_change_percent"],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.9,
                markersize=5.8,
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
            if panel["key"] == PANELS[0]["key"]:
                legend_handles.append(line)
                legend_labels.append(style["label"])

        tick_data = data.drop_duplicates("x_order").sort_values("x_order")
        ax.set_xticks(tick_data["x_order"].tolist())
        ax.set_xticklabels(
            tick_data["x_label"].tolist(),
            rotation=panel["x_tick_rotation"],
            ha="right" if panel["x_tick_rotation"] else "center",
        )
        ax.set_title(panel["title"], color=TOKENS["ink"], fontsize=10.8, pad=7)
        ax.set_xlabel(panel["x_label"], color=TOKENS["ink"], fontsize=9.5)
        ax.axhline(0, color=TOKENS["muted"], linewidth=0.9, alpha=0.85)
        ax.margins(x=0.08)
        set_panel_ylim(ax, data, panel["y_step"])
        style_axis(ax)

    axes[0].set_ylabel("Relative NMAE Change (%)", color=TOKENS["ink"], fontsize=9.8)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=9.4,
        handlelength=2.2,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), w_pad=1.5)
    fig.savefig(PDF_PATH, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(PNG_PATH, bbox_inches="tight", dpi=300, transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)

    print(f"saved: {PDF_PATH}")
    print(f"saved: {PNG_PATH}")
    print(f"saved: {CSV_PATH}")
    print(f"saved: {SCALE_CODE_MAP_PATH}")


if __name__ == "__main__":
    main()
