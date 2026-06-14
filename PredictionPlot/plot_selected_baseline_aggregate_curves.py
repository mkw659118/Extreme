from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


ROOT = Path.cwd().parent if Path.cwd().name == "PredictionPlot" else Path.cwd()
PLOT_DATA_DIR = ROOT / "PredictionPlot" / "plot_data"
FIG_DIR = ROOT / "PredictionPlot" / "figures"
OUT_DIR = ROOT / "PredictionPlot"

DATASETS = ["Abilene", "Geant", "Seattle"]
PRED_LENS = [5, 10, 15, 20]

# One representative per broad baseline family, shared by every dataset.
SELECTED_BASELINES = [
    "iTransformer",  # Transformer-style baseline; weaker than PMDformer/PatchTST on these aggregate curves.
    "HMformer",      # Heterogeneous/missing-related baseline family.
    "TimesNet",      # Periodicity / 2D temporal variation family.
    "WPMixer",       # MLP / mixer-style baseline.
    "P_sLSTM",       # Recurrent/LSTM-style baseline; consistently among the weaker baselines here.
]

DISPLAY_ORDER = ["True", "DARNet"] + SELECTED_BASELINES
STYLE_SUFFIX = "selected_baselines_v6"

LINE_STYLE = {
    "True": dict(color="#4A4F5A", linewidth=2.55, linestyle="-", alpha=0.98, zorder=30),
    "DARNet": dict(color="#B23A48", linewidth=2.20, linestyle="-", alpha=0.98, zorder=25),
    "iTransformer": dict(color="#5477C4", linewidth=1.24, linestyle="-", alpha=0.68),
    "HMformer": dict(color="#8A3A6F", linewidth=1.24, linestyle="-", alpha=0.68),
    "TimesNet": dict(color="#7A828F", linewidth=1.18, linestyle="-", alpha=0.70),
    "WPMixer": dict(color="#386411", linewidth=1.22, linestyle="-", alpha=0.68),
    "P_sLSTM": dict(color="#CC6F47", linewidth=1.22, linestyle="-", alpha=0.68),
}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
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


def metric_rows() -> pd.DataFrame:
    rows = []
    all_baselines = [
        "PMDformer",
        "iTransformer",
        "FEDformer",
        "FeTS",
        "HMformer",
        "PatchTST",
        "TimesNet",
        "WPMixer",
        "P_sLSTM",
        "xLSTMTime",
        "xLSTM-Mixer",
    ]
    for dataset in DATASETS:
        for pred_len in PRED_LENS:
            path = PLOT_DATA_DIR / f"{dataset}_PL{pred_len}_merged_prediction_curves.csv"
            df = pd.read_csv(path)
            true = df["True"].to_numpy(dtype=np.float64)
            denom = max(float(np.nanmean(np.abs(true))), float(np.nanstd(true)), 1e-12)
            for model in all_baselines:
                pred = df[model].to_numpy(dtype=np.float64)
                err = pred - true
                rows.append(
                    {
                        "dataset": dataset,
                        "pred_len": pred_len,
                        "model": model,
                        "mae": float(np.nanmean(np.abs(err))),
                        "rmse": float(np.sqrt(np.nanmean(err**2))),
                        "nmae": float(np.nanmean(np.abs(err)) / denom),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics["mae_rank"] = metrics.groupby(["dataset", "pred_len"])["mae"].rank(method="average")
    return metrics


def write_selection_summary() -> None:
    metrics = metric_rows()
    summary = (
        metrics.groupby("model")
        .agg(
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            mean_nmae=("nmae", "mean"),
            mean_rank=("mae_rank", "mean"),
            median_rank=("mae_rank", "median"),
        )
        .sort_values(["mean_rank", "mean_nmae"], ascending=[False, False])
        .reset_index()
    )
    summary["selected_for_plot"] = summary["model"].isin(SELECTED_BASELINES)
    summary.to_csv(OUT_DIR / "selected_baseline_aggregate_metric_summary.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT_DIR / "selected_baseline_aggregate_metric_detail.csv", index=False, encoding="utf-8-sig")

    rationale = pd.DataFrame(
        [
            {
                "selected_model": "iTransformer",
                "representative_family": "Transformer / attention",
                "reason": "Keeps a classic Transformer-style baseline while avoiding stronger/denser choices such as PMDformer and PatchTST.",
            },
            {
                "selected_model": "HMformer",
                "representative_family": "Heterogeneous or missing-related modeling",
                "reason": "Worse average rank than FeTS on the aggregate curves, so it gives a clearer contrast.",
            },
            {
                "selected_model": "TimesNet",
                "representative_family": "Periodicity / temporal 2D variation",
                "reason": "Different inductive bias from attention and recurrent baselines; middle-to-weak aggregate performance.",
            },
            {
                "selected_model": "WPMixer",
                "representative_family": "MLP / mixer",
                "reason": "Represents the mixer family without adding another attention-style baseline.",
            },
            {
                "selected_model": "P_sLSTM",
                "representative_family": "Recurrent / LSTM",
                "reason": "Consistently the weakest baseline by average rank in these aggregate plots.",
            },
        ]
    )
    rationale.to_csv(OUT_DIR / "selected_baseline_aggregate_choice_rationale.csv", index=False, encoding="utf-8-sig")

    palette = pd.DataFrame(
        [
            {"series": series, "color": style["color"], "line_style": str(style["linestyle"]), "line_width": style["linewidth"]}
            for series, style in LINE_STYLE.items()
        ]
    )
    palette.to_csv(OUT_DIR / f"selected_baseline_aggregate_palette_{STYLE_SUFFIX}.csv", index=False, encoding="utf-8-sig")


def axis_margin(values: np.ndarray) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    vmin, vmax = float(np.min(values)), float(np.max(values))
    pad = max((vmax - vmin) * 0.06, abs(vmax) * 0.015, 1e-12)
    return vmin - pad, vmax + pad


def plot_one(dataset: str, pred_len: int) -> dict[str, object]:
    csv_path = PLOT_DATA_DIR / f"{dataset}_PL{pred_len}_merged_prediction_curves.csv"
    df = pd.read_csv(csv_path)
    missing = [model for model in DISPLAY_ORDER if model not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    x = df["step"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12.8, 5.6), facecolor=TOKENS["surface"])
    ax.set_facecolor(TOKENS["panel"])

    plotted = []
    for label in DISPLAY_ORDER:
        y = df[label].to_numpy(dtype=np.float64)
        plotted.append(y)
        ax.plot(x, y, label=label, **LINE_STYLE[label])

    ax.set_xlabel("Test Time Step", color=TOKENS["ink"])
    ax.set_ylabel("Value", color=TOKENS["ink"])
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    ax.grid(True, linestyle="--", linewidth=0.6, color=TOKENS["grid"], alpha=0.82)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=TOKENS["ink"])
    lim = axis_margin(np.concatenate(plotted))
    if lim is not None:
        ax.set_ylim(*lim)
    legend = ax.legend(
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        facecolor=TOKENS["panel"],
        edgecolor=TOKENS["axis"],
        framealpha=0.86,
        fontsize=8.5,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.34,
        borderpad=0.48,
        borderaxespad=0.25,
    )
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color(TOKENS["ink"])
        text.set_ha("right")
    fig.tight_layout()

    stem = f"{dataset}_PL{pred_len}_prediction_curves_{STYLE_SUFFIX}"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return {
        "dataset": dataset,
        "pred_len": pred_len,
        "pdf": str(pdf_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "png": str(png_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "source_csv": str(csv_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "selected_baselines": ",".join(SELECTED_BASELINES),
        "point_count": int(len(df)),
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_selection_summary()
    rows = []
    for dataset in DATASETS:
        for pred_len in PRED_LENS:
            print(f"plot {dataset} PL{pred_len}", flush=True)
            rows.append(plot_one(dataset, pred_len))
    pd.DataFrame(rows).to_csv(
        OUT_DIR / f"selected_baseline_aggregate_figure_index_{STYLE_SUFFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
