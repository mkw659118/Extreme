from __future__ import annotations

import argparse
import csv
import math
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, ScalarFormatter


ROOT = Path.cwd().parent if Path.cwd().name == "PredictionPlot" else Path.cwd()
DRAW_DIR = ROOT / "draw"
OUT_DIR = ROOT / "PredictionPlot"

DATASETS = ["Abilene", "Geant", "Seattle"]
PRED_LENS = [5, 10, 15, 20]
D_MODEL = 256
TARGET_COL = "TC0"

MODEL_DIR_TO_LABEL = OrderedDict(
    [
        ("net", "DARNet"),
        ("PMDformer", "PMDformer"),
        ("iTransformer", "iTransformer"),
        ("FEDformer", "FEDformer"),
        ("FeTS", "FeTS"),
        ("HMformer", "HMformer"),
        ("PatchTST", "PatchTST"),
        ("timesnet", "TimesNet"),
        ("WPMixer", "WPMixer"),
        ("P_sLSTM", "P_sLSTM"),
        ("xLSTMTime", "xLSTMTime"),
        ("xlstm_mixer", "xLSTM-Mixer"),
    ]
)
MODEL_ORDER = list(MODEL_DIR_TO_LABEL.values())
DISPLAY_ORDER = ["True"] + MODEL_ORDER

LINE_STYLE = {
    "True": dict(color="black", linewidth=2.20, linestyle="-", marker="o", markersize=3.5, alpha=0.96, zorder=30),
    "DARNet": dict(color="#d62728", linewidth=1.85, linestyle="-", marker="s", markersize=3.2, alpha=0.94, zorder=25),
    "PMDformer": dict(color="#1f77b4", linewidth=1.00, linestyle="-", marker=".", markersize=3.2, alpha=0.84),
    "iTransformer": dict(color="#ff7f0e", linewidth=1.00, linestyle="-", marker=".", markersize=3.2, alpha=0.84),
    "FEDformer": dict(color="#2ca02c", linewidth=1.00, linestyle="-", marker=".", markersize=3.2, alpha=0.84),
    "FeTS": dict(color="#9467bd", linewidth=1.00, linestyle="-", marker=".", markersize=3.2, alpha=0.84),
    "HMformer": dict(color="#8c564b", linewidth=1.00, linestyle="-", marker=".", markersize=3.2, alpha=0.84),
    "PatchTST": dict(color="#e377c2", linewidth=1.00, linestyle="-", marker=".", markersize=3.2, alpha=0.84),
    "TimesNet": dict(color="#7f7f7f", linewidth=1.00, linestyle="-", marker=".", markersize=3.2, alpha=0.84),
    "WPMixer": dict(color="#bcbd22", linewidth=1.00, linestyle="-", marker=".", markersize=3.2, alpha=0.84),
    "P_sLSTM": dict(color="#17becf", linewidth=1.00, linestyle="-", marker=".", markersize=3.2, alpha=0.84),
    "xLSTMTime": dict(color="#00429d", linewidth=1.00, linestyle="--", marker=".", markersize=3.2, alpha=0.86),
    "xLSTM-Mixer": dict(color="#93003a", linewidth=1.00, linestyle="--", marker=".", markersize=3.2, alpha=0.86),
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 120,
    }
)


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def index_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(OUT_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def get_raw_path(dataset: str, model_dir: str, pred_len: int, d_model: int = D_MODEL, target_col: str = TARGET_COL) -> Path:
    return DRAW_DIR / dataset / f"{dataset}_single" / model_dir / f"PL{pred_len}_DM{d_model}" / target_col / "test_raw.csv"


def read_raw_windows(path: Path, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    required = {"true", "pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    true = pd.to_numeric(df["true"], errors="coerce").to_numpy(dtype=np.float64)
    pred = pd.to_numeric(df["pred"], errors="coerce").to_numpy(dtype=np.float64)
    usable = (len(true) // pred_len) * pred_len
    if usable == 0:
        raise ValueError(f"{path} has no usable rows for pred_len={pred_len}")
    return true[:usable].reshape(-1, pred_len), pred[:usable].reshape(-1, pred_len)


def load_windows(dataset: str, pred_len: int) -> dict[str, dict[str, object]]:
    windows: dict[str, dict[str, object]] = {}
    missing = []
    for model_dir, label in MODEL_DIR_TO_LABEL.items():
        path = get_raw_path(dataset, model_dir, pred_len)
        if not path.exists():
            missing.append(str(path))
            continue
        true_w, pred_w = read_raw_windows(path, pred_len)
        windows[label] = {"true": true_w, "pred": pred_w, "path": path, "model_dir": model_dir}
    if missing:
        raise FileNotFoundError("Missing raw files:\n" + "\n".join(missing))
    return windows


def curve_dynamic_score(curve: np.ndarray) -> float:
    curve = np.asarray(curve, dtype=np.float64)
    finite = np.isfinite(curve)
    if not finite.any():
        return -math.inf
    vals = curve[finite]
    return float((np.nanmax(vals) - np.nanmin(vals)) + 0.10 * np.nanmax(np.abs(vals)))


def axis_margin(values: np.ndarray) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    vmin, vmax = float(np.min(values)), float(np.max(values))
    pad = max((vmax - vmin) * 0.08, abs(vmax) * 0.02, 1e-12)
    return vmin - pad, vmax + pad


def match_all_windows_by_true(reference_true: np.ndarray, candidate_true: np.ndarray, chunk_size: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every DARNet reference window, find the closest true curve in another model output."""
    reference_true = np.asarray(reference_true, dtype=np.float64)
    candidate_true = np.asarray(candidate_true, dtype=np.float64)
    n_ref = reference_true.shape[0]
    match_idx = np.zeros(n_ref, dtype=np.int64)
    rmse = np.full(n_ref, np.inf, dtype=np.float64)
    max_abs = np.full(n_ref, np.inf, dtype=np.float64)

    for start in range(0, n_ref, chunk_size):
        end = min(start + chunk_size, n_ref)
        ref_chunk = reference_true[start:end]
        valid = np.isfinite(ref_chunk[:, None, :]) & np.isfinite(candidate_true[None, :, :])
        diff = np.where(valid, ref_chunk[:, None, :] - candidate_true[None, :, :], 0.0)
        valid_count = valid.sum(axis=2)
        mse = np.full(valid_count.shape, np.inf, dtype=np.float64)
        ok = valid_count > 0
        sse = (diff**2).sum(axis=2)
        mse[ok] = sse[ok] / valid_count[ok]

        best = np.argmin(mse, axis=1)
        row = np.arange(end - start)
        match_idx[start:end] = best
        rmse[start:end] = np.sqrt(mse[row, best])
        max_abs[start:end] = np.nanmax(np.abs(candidate_true[best] - ref_chunk), axis=1)

    return match_idx, rmse, max_abs


def compute_match_maps(windows: dict[str, dict[str, object]], align_by_true: bool = True) -> dict[str, dict[str, np.ndarray]]:
    reference_true = windows["DARNet"]["true"]
    n_ref = reference_true.shape[0]
    maps: dict[str, dict[str, np.ndarray]] = {}
    for model in MODEL_ORDER:
        model_true = windows[model]["true"]
        if model == "DARNet" or not align_by_true:
            idx = np.minimum(np.arange(n_ref), model_true.shape[0] - 1).astype(np.int64)
            maps[model] = {
                "idx": idx,
                "rmse": np.zeros(n_ref, dtype=np.float64) if model == "DARNet" else np.full(n_ref, np.nan),
                "max_abs": np.zeros(n_ref, dtype=np.float64) if model == "DARNet" else np.full(n_ref, np.nan),
            }
        else:
            idx, rmse, max_abs = match_all_windows_by_true(reference_true, model_true)
            maps[model] = {"idx": idx, "rmse": rmse, "max_abs": max_abs}
    return maps


def build_curve_from_maps(
    dataset: str,
    pred_len: int,
    reference_window: int,
    windows: dict[str, dict[str, object]],
    match_maps: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    reference_window = int(reference_window)
    reference_true = windows["DARNet"]["true"][reference_window]
    df = pd.DataFrame({"horizon": np.arange(1, pred_len + 1), "True": reference_true})
    match_rows = []

    for model in MODEL_ORDER:
        matched_window = int(match_maps[model]["idx"][reference_window])
        df[model] = windows[model]["pred"][matched_window]
        match_rows.append(
            {
                "dataset": dataset,
                "pred_len": pred_len,
                "reference_window": reference_window,
                "model": model,
                "matched_window": matched_window,
                "true_match_rmse": float(match_maps[model]["rmse"][reference_window]),
                "true_match_max_abs": float(match_maps[model]["max_abs"][reference_window]),
                "window_count": int(windows[model]["true"].shape[0]),
            }
        )
    return df, match_rows


def plot_window_id_figure(
    df: pd.DataFrame,
    dataset: str,
    pred_len: int,
    reference_window: int,
    output_path: Path,
    dpi: int = 140,
    show_legend: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = df["horizon"].to_numpy()
    plotted = []
    handles = []
    labels = []
    for label in DISPLAY_ORDER:
        y = df[label].to_numpy(dtype=np.float64)
        plotted.append(y)
        line, = ax.plot(x, y, label=label, **LINE_STYLE[label])
        handles.append(line)
        labels.append(label)

    ax.set_title(f"{dataset} | PredLen={pred_len} | Window ID={reference_window}", fontsize=11, pad=9)
    ax.text(
        0.015,
        0.955,
        f"ID {reference_window}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#333333", linewidth=0.7, alpha=0.86),
        zorder=40,
    )
    ax.set_xlabel("Prediction Horizon", fontsize=9)
    ax.set_ylabel("Value", fontsize=9)
    ax.set_xticks(np.arange(1, pred_len + 1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(pred_len, 6)))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.34)
    ax.tick_params(axis="both", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    lim = axis_margin(np.concatenate(plotted))
    if lim is not None:
        ax.set_ylim(*lim)
    if show_legend:
        ax.legend(
            handles,
            labels,
            ncol=6,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            frameon=False,
            fontsize=6.7,
            handlelength=1.6,
            columnspacing=0.8,
        )
        fig.tight_layout(rect=[0, 0.09, 1, 1])
    else:
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def export_combo(
    dataset: str,
    pred_len: int,
    output_dir: Path,
    max_windows: int | None = None,
    overwrite: bool = False,
    align_by_true: bool = True,
    dpi: int = 140,
    progress_every: int = 100,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    windows = load_windows(dataset, pred_len)
    match_maps = compute_match_maps(windows, align_by_true=align_by_true)
    n_ref = int(windows["DARNet"]["true"].shape[0])
    if max_windows is not None:
        n_ref = min(n_ref, int(max_windows))

    combo_dir = output_dir / dataset / f"PL{pred_len}"
    combo_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    match_rows_all: list[dict[str, object]] = []
    for reference_window in range(n_ref):
        stem = f"{safe_name(dataset)}_PL{pred_len}_window_{reference_window:04d}"
        png_path = combo_dir / f"{stem}.png"
        csv_path = combo_dir / f"{stem}.csv"
        df, match_rows = build_curve_from_maps(dataset, pred_len, reference_window, windows, match_maps)
        true_curve = df["True"].to_numpy(dtype=np.float64)
        pred_curve = df["DARNet"].to_numpy(dtype=np.float64)
        err = pred_curve - true_curve

        if overwrite or not png_path.exists():
            plot_window_id_figure(df, dataset, pred_len, reference_window, png_path, dpi=dpi)
        if overwrite or not csv_path.exists():
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        index_rows.append(
            {
                "dataset": dataset,
                "pred_len": pred_len,
                "window": reference_window,
                "dynamic_score": curve_dynamic_score(true_curve),
                "true_min": float(np.nanmin(true_curve)),
                "true_max": float(np.nanmax(true_curve)),
                "true_range": float(np.nanmax(true_curve) - np.nanmin(true_curve)),
                "darnet_mae": float(np.nanmean(np.abs(err))),
                "darnet_rmse": float(np.sqrt(np.nanmean(err**2))),
                "figure_path": index_path(png_path),
                "curve_csv": index_path(csv_path),
            }
        )
        match_rows_all.extend(match_rows)

        done = reference_window + 1
        if progress_every and (done % progress_every == 0 or done == n_ref):
            print(f"{dataset} PL{pred_len}: {done}/{n_ref}", flush=True)

    combo_index_path = combo_dir / f"{safe_name(dataset)}_PL{pred_len}_window_index.csv"
    combo_match_path = combo_dir / f"{safe_name(dataset)}_PL{pred_len}_window_matches.csv"
    pd.DataFrame(index_rows).to_csv(combo_index_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(match_rows_all).to_csv(combo_match_path, index=False, encoding="utf-8-sig")
    return index_rows, match_rows_all


def export_all(
    datasets: list[str],
    pred_lens: list[int],
    output_dir: Path,
    max_windows: int | None = None,
    overwrite: bool = False,
    align_by_true: bool = True,
    dpi: int = 140,
    progress_every: int = 100,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_index_rows: list[dict[str, object]] = []
    all_match_rows: list[dict[str, object]] = []

    for dataset in datasets:
        for pred_len in pred_lens:
            print(f"Start {dataset} PL{pred_len}", flush=True)
            index_rows, match_rows = export_combo(
                dataset=dataset,
                pred_len=int(pred_len),
                output_dir=output_dir,
                max_windows=max_windows,
                overwrite=overwrite,
                align_by_true=align_by_true,
                dpi=dpi,
                progress_every=progress_every,
            )
            all_index_rows.extend(index_rows)
            all_match_rows.extend(match_rows)

    index_path = output_dir / "all_window_id_figure_index.csv"
    match_path = output_dir / "all_window_id_matches.csv"
    pd.DataFrame(all_index_rows).to_csv(index_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(all_match_rows).to_csv(match_path, index=False, encoding="utf-8-sig")

    summary_path = output_dir / "README.txt"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["This directory stores one PNG per DARNet reference window."])
        writer.writerow(["Use the Window ID shown in the title/badge as reference_window in select_prediction_horizon_windows.ipynb."])
        writer.writerow(["Index CSV", str(index_path)])
        writer.writerow(["Match CSV", str(match_path)])
        writer.writerow(["Datasets", " ".join(datasets)])
        writer.writerow(["Prediction lengths", " ".join(map(str, pred_lens))])
    print(f"Saved master index: {index_path}", flush=True)
    print(f"Saved match table: {match_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one horizon prediction figure for every DARNet reference window.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--pred-lens", nargs="+", type=int, default=PRED_LENS, choices=PRED_LENS)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR / "all_window_id_figures")
    parser.add_argument("--max-windows", type=int, default=None, help="Debug option: export only the first N windows for each dataset/pred_len.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing PNG and CSV files.")
    parser.add_argument("--no-align-by-true", action="store_true", help="Use the same raw window index for all models instead of matching by true curve.")
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_all(
        datasets=args.datasets,
        pred_lens=args.pred_lens,
        output_dir=args.output_dir,
        max_windows=args.max_windows,
        overwrite=args.overwrite,
        align_by_true=not args.no_align_by_true,
        dpi=args.dpi,
        progress_every=args.progress_every,
    )
