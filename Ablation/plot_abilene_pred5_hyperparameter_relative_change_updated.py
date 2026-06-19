import csv
import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Ablation" / "figures" / "combined"

LOGS = {
    "num_experts": ROOT / "CPU-x86_64_GPU-NVIDIA-GeForce-RTX-4090_DARNet_hp_num_experts.log",
    "retrieval_topk": ROOT / "CPU-x86_64_GPU-NVIDIA-GeForce-RTX-4090_DARNet_hp_retrieval_topk.log",
    "state_prior_scales": ROOT / "CPU-x86_64_GPU-NVIDIA-GeForce-RTX-4090_DARNet_hp_state_prior_scales.log",
}

DATASET = "Abilene"
PRED_LEN = 5

MAIN_NUM_EXPERTS = 4
MAIN_RETRIEVAL_NUM = 2
MAIN_STATE_PRIOR = ("1,4,8,16", True)

STATE_PRIOR_CODES = [
    ("A", "1", True),
    ("B", "1,4", True),
    ("C", "1,4,8", True),
    ("D", "1,4,8,16", True),
    ("E", "1,4,8,16,32", True),
    ("F", "1,4,8,16", False),
]


def extract_value(pattern, text, cast=str, default=None):
    match = re.search(pattern, text)
    if not match:
        return default
    return cast(match.group(1).strip())


def parse_log(path):
    records = []
    current = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Dataset :" in line and "Pred_Len :" in line:
                current = {
                    "Dataset": extract_value(r"Dataset\s*:\s*([^,]+)", line),
                    "Pred_Len": extract_value(r"Pred_Len\s*:\s*(\d+)", line, int),
                    "Retrieval_Num": extract_value(r"Retrieval_Num\s*:\s*(\d+)", line, int),
                    "Num_Experts": extract_value(r"Num_Experts\s*:\s*(\d+)", line, int),
                    "Top_K_Experts": extract_value(r"Top_K_Experts\s*:\s*(\d+)", line, int),
                    "State_Prior_Scales": extract_value(
                        r"State_Prior_Scales\s*:\s*(.*?),\s*State_Prior_Include_Seq_Level",
                        line,
                    ),
                    "State_Prior_Include_Seq_Level": extract_value(
                        r"State_Prior_Include_Seq_Level\s*:\s*(True|False)",
                        line,
                        lambda x: x == "True",
                    ),
                }
                continue

            if current and "NMAE -" in line:
                nmae = extract_value(r"NMAE\s*-\s*([^\s]+)", line, float)
                if nmae is not None and math.isfinite(nmae):
                    records.append({**current, "NMAE": nmae, "source_log": path.name})
                current = None

    return records


def require_one(df, description):
    if len(df) != 1:
        raise ValueError(f"Expected exactly one row for {description}, got {len(df)}")
    return df.iloc[0]


def build_plot_data():
    parsed = {
        name: pd.DataFrame(parse_log(path))
        for name, path in LOGS.items()
    }
    for name, df in parsed.items():
        if df.empty:
            raise ValueError(f"No records parsed from {LOGS[name]}")
        parsed[name] = df[(df["Dataset"] == DATASET) & (df["Pred_Len"] == PRED_LEN)].copy()

    rows = []

    num_df = parsed["num_experts"]
    num_order = [1, 2, 4, 6, 8]
    num_baseline = require_one(num_df[num_df["Num_Experts"] == num_order[0]], "Num_Experts baseline")["NMAE"]
    for idx, value in enumerate(num_order):
        record = require_one(num_df[num_df["Num_Experts"] == value], f"Num_Experts={value}")
        rows.append({
            "Dataset": DATASET,
            "Pred_Len": PRED_LEN,
            "experiment": "num_experts",
            "panel_title": "Number of Experts",
            "x_order": idx,
            "x_label": str(value),
            "setting": str(value),
            "NMAE": record["NMAE"],
            "baseline_setting": str(num_order[0]),
            "baseline_nmae": num_baseline,
            "relative_change_percent": (record["NMAE"] - num_baseline) / num_baseline * 100.0,
            "is_main_setting": value == MAIN_NUM_EXPERTS,
            "source_log": record["source_log"],
        })

    topk_df = parsed["retrieval_topk"]
    topk_order = [1, 2, 3, 5, 8]
    topk_baseline = require_one(topk_df[topk_df["Retrieval_Num"] == topk_order[0]], "Retrieval_Num baseline")["NMAE"]
    for idx, value in enumerate(topk_order):
        record = require_one(topk_df[topk_df["Retrieval_Num"] == value], f"Retrieval_Num={value}")
        rows.append({
            "Dataset": DATASET,
            "Pred_Len": PRED_LEN,
            "experiment": "retrieval_topk",
            "panel_title": "Retrieval Top-K",
            "x_order": idx,
            "x_label": str(value),
            "setting": str(value),
            "NMAE": record["NMAE"],
            "baseline_setting": str(topk_order[0]),
            "baseline_nmae": topk_baseline,
            "relative_change_percent": (record["NMAE"] - topk_baseline) / topk_baseline * 100.0,
            "is_main_setting": value == MAIN_RETRIEVAL_NUM,
            "source_log": record["source_log"],
        })

    scale_df = parsed["state_prior_scales"]
    scale_baseline_record = require_one(
        scale_df[
            (scale_df["State_Prior_Scales"] == STATE_PRIOR_CODES[0][1])
            & (scale_df["State_Prior_Include_Seq_Level"] == STATE_PRIOR_CODES[0][2])
        ],
        "State_Prior baseline",
    )
    scale_baseline = scale_baseline_record["NMAE"]
    for idx, (code, scales, include_seq) in enumerate(STATE_PRIOR_CODES):
        record = require_one(
            scale_df[
                (scale_df["State_Prior_Scales"] == scales)
                & (scale_df["State_Prior_Include_Seq_Level"] == include_seq)
            ],
            f"State_Prior={code}",
        )
        setting = scales if include_seq else f"{scales}_no_seq"
        rows.append({
            "Dataset": DATASET,
            "Pred_Len": PRED_LEN,
            "experiment": "state_prior_scales",
            "panel_title": "State Prior Scales",
            "x_order": idx,
            "x_label": code,
            "setting": setting,
            "state_prior_scale_code": code,
            "NMAE": record["NMAE"],
            "baseline_setting": STATE_PRIOR_CODES[0][1],
            "baseline_nmae": scale_baseline,
            "relative_change_percent": (record["NMAE"] - scale_baseline) / scale_baseline * 100.0,
            "is_main_setting": (scales, include_seq) == MAIN_STATE_PRIOR,
            "source_log": record["source_log"],
        })

    return pd.DataFrame(rows)


def build_plot_data_for(dataset):
    global DATASET
    previous_dataset = DATASET
    DATASET = dataset
    try:
        return build_plot_data()
    finally:
        DATASET = previous_dataset


def save_mapping():
    mapping_path = OUT_DIR / f"{DATASET}_state_prior_scale_code_mapping_PL5_updated.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state_prior_scale_code", "State_Prior_Setting"])
        for code, scales, include_seq in STATE_PRIOR_CODES:
            writer.writerow([code, scales if include_seq else f"{scales}_no_seq"])
    return mapping_path


def draw(df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = f"{DATASET}_three_hyperparameter_relative_change_NMAE_1x3_coded_scales_PL5_updated"
    csv_path = OUT_DIR / f"{output_stem}.csv"
    pdf_path = OUT_DIR / f"{output_stem}.pdf"
    png_path = OUT_DIR / f"{output_stem}.png"

    df.to_csv(csv_path, index=False)
    mapping_path = save_mapping()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.7), sharey=True)
    experiments = ["num_experts", "retrieval_topk", "state_prior_scales"]
    line_color = "#3C6E71"
    main_color = "#F4A261"
    baseline_color = "#4A4A4A"

    ymin = df["relative_change_percent"].min()
    ymax = df["relative_change_percent"].max()
    pad = max((ymax - ymin) * 0.18, 0.35)
    ylim = (min(ymin - pad, -1.2), max(ymax + pad, 2.8))

    for ax, exp in zip(axes, experiments):
        sub = df[df["experiment"] == exp].sort_values("x_order")
        x = sub["x_order"].to_numpy()
        y = sub["relative_change_percent"].to_numpy()

        ax.axhline(0, color="#8A8A8A", linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
        ax.plot(x, y, color=line_color, linewidth=2.2, marker="o", markersize=5.5, zorder=2)

        baseline = sub.iloc[0]
        ax.scatter(
            [baseline["x_order"]],
            [baseline["relative_change_percent"]],
            s=54,
            color=baseline_color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        main = sub[sub["is_main_setting"]]
        ax.scatter(
            main["x_order"],
            main["relative_change_percent"],
            s=86,
            color=main_color,
            edgecolor="#3A3A3A",
            linewidth=0.8,
            zorder=4,
        )

        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['relative_change_percent']:+.2f}%",
                (row["x_order"], row["relative_change_percent"]),
                textcoords="offset points",
                xytext=(0, 8 if row["relative_change_percent"] <= 0 else 7),
                ha="center",
                va="bottom",
                fontsize=8.4,
                color="#303030",
            )

        ax.set_title(sub.iloc[0]["panel_title"], pad=7, fontweight="semibold")
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['NMAE']:.4f}",
                (row["x_order"], row["NMAE"]),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                va="bottom",
                fontsize=8.4,
                color="#303030",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(sub["x_label"].tolist())
        ax.set_xlabel("Configuration")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#D6D6D6", linewidth=0.7, alpha=0.7)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("#444444")

    axes[0].set_ylabel("Relative NMAE Change (%)")
    fig.suptitle("Abilene Hyperparameter Sensitivity (pred_len = 5)", y=1.03, fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        "Relative change is computed against the first setting in each panel; lower is better. Orange marker denotes the default DATP-Net setting.",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#444444",
    )

    fig.tight_layout(w_pad=1.1)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return csv_path, mapping_path, pdf_path, png_path


def draw_nmae(df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = f"{DATASET}_three_hyperparameter_NMAE_1x3_coded_scales_PL5_updated_no_outer_text_transparent_value_labels"
    csv_path = OUT_DIR / f"{output_stem}.csv"
    pdf_path = OUT_DIR / f"{output_stem}.pdf"
    png_path = OUT_DIR / f"{output_stem}.png"

    df.to_csv(csv_path, index=False)
    mapping_path = save_mapping()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.7), sharey=True)
    experiments = ["num_experts", "retrieval_topk", "state_prior_scales"]
    line_color = "#3C6E71"
    main_color = "#F4A261"
    baseline_color = "#4A4A4A"

    ymin = df["NMAE"].min()
    ymax = df["NMAE"].max()
    pad = max((ymax - ymin) * 0.18, 0.0025)
    ylim = (ymin - pad, ymax + pad)

    for ax, exp in zip(axes, experiments):
        sub = df[df["experiment"] == exp].sort_values("x_order")
        x = sub["x_order"].to_numpy()
        y = sub["NMAE"].to_numpy()

        ax.plot(x, y, color=line_color, linewidth=2.2, marker="o", markersize=5.5, zorder=2)

        baseline = sub.iloc[0]
        ax.scatter(
            [baseline["x_order"]],
            [baseline["NMAE"]],
            s=54,
            color=baseline_color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        main = sub[sub["is_main_setting"]]
        ax.scatter(
            main["x_order"],
            main["NMAE"],
            s=86,
            color=main_color,
            edgecolor="#3A3A3A",
            linewidth=0.8,
            zorder=4,
        )

        ax.set_title(sub.iloc[0]["panel_title"], pad=7, fontweight="semibold")
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['NMAE']:.4f}",
                (row["x_order"], row["NMAE"]),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                va="bottom",
                fontsize=8.4,
                color="#303030",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(sub["x_label"].tolist())
        ax.set_xlim(x.min() - 0.35, x.max() + 0.35)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#D6D6D6", linewidth=0.7, alpha=0.7)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("#444444")

    axes[0].set_ylabel("NMAE")

    fig.tight_layout(w_pad=1.1)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return csv_path, mapping_path, pdf_path, png_path


def draw_combined_nmae(datasets=("Abilene", "Geant")):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = "Abilene_Geant_three_hyperparameter_NMAE_2x3_coded_scales_PL5_updated_row_labels_bottom_columns_light_yellow_regular_text"
    csv_path = OUT_DIR / f"{output_stem}.csv"
    pdf_path = OUT_DIR / f"{output_stem}.pdf"
    png_path = OUT_DIR / f"{output_stem}.png"

    all_data = pd.concat([build_plot_data_for(dataset) for dataset in datasets], ignore_index=True)
    all_data.to_csv(csv_path, index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    experiments = ["num_experts", "retrieval_topk", "state_prior_scales"]
    panel_titles = ["Number of Experts", "Retrieval Top-K", "State Prior Scales"]
    line_color = "#E9C46A"
    main_color = "#F4A261"
    baseline_color = line_color

    fig, axes = plt.subplots(len(datasets), 3, figsize=(9.6, 5.45), sharex=False, sharey=False)
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, dataset in enumerate(datasets):
        row_data = all_data[all_data["Dataset"] == dataset]
        ymin = row_data["NMAE"].min()
        ymax = row_data["NMAE"].max()
        pad = max((ymax - ymin) * 0.18, 0.0025)
        ylim = (ymin - pad, ymax + pad)

        for col_idx, (exp, panel_title) in enumerate(zip(experiments, panel_titles)):
            ax = axes[row_idx, col_idx]
            sub = row_data[row_data["experiment"] == exp].sort_values("x_order")
            x = sub["x_order"].to_numpy()
            y = sub["NMAE"].to_numpy()

            ax.set_box_aspect(1)
            ax.plot(x, y, color=line_color, linewidth=2.2, marker="o", markersize=5.5, zorder=2)

            baseline = sub.iloc[0]
            ax.scatter(
                [baseline["x_order"]],
                [baseline["NMAE"]],
                s=54,
                color=baseline_color,
                edgecolor=baseline_color,
                linewidth=0.0,
                zorder=3,
            )

            main = sub[sub["is_main_setting"]]
            ax.scatter(
                main["x_order"],
                main["NMAE"],
                s=86,
                color=main_color,
                edgecolor="#3A3A3A",
                linewidth=0.8,
                zorder=4,
            )

            ax.set_xticks(x)
            ax.set_xticklabels(sub["x_label"].tolist())
            if row_idx == 0:
                ax.set_title(panel_title, pad=7, fontweight="normal")
            ax.set_xlim(x.min() - 0.35, x.max() + 0.35)
            ax.set_ylim(*ylim)
            ax.grid(axis="y", color="#D6D6D6", linewidth=0.7, alpha=0.7)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            for spine in ax.spines.values():
                spine.set_linewidth(0.9)
                spine.set_color("#444444")

            if col_idx == 0:
                ax.set_ylabel("NMAE")

    fig.tight_layout(w_pad=1.85, h_pad=0.35)
    fig.subplots_adjust(left=0.155, bottom=0.08)
    fig.canvas.draw()
    for row_idx, dataset in enumerate(datasets):
        row_axes = axes[row_idx, :]
        y0 = min(ax.get_position().y0 for ax in row_axes)
        y1 = max(ax.get_position().y1 for ax in row_axes)
        display_dataset = "GÉANT" if dataset == "Geant" else dataset
        fig.text(
            0.075,
            (y0 + y1) / 2,
            display_dataset,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="normal",
            color="#1F1F1F",
        )

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return csv_path, pdf_path, png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET, choices=["Abilene", "Geant", "Seattle"])
    parser.add_argument("--combined", action="store_true")
    args = parser.parse_args()

    if args.combined:
        outputs = draw_combined_nmae()
    else:
        DATASET = args.dataset
        data = build_plot_data()
        outputs = draw_nmae(data)
    for path in outputs:
        print(path)
