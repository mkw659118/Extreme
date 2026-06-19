from __future__ import annotations

import argparse
import math
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path.cwd().parent if Path.cwd().name == "PredictionPlot" else Path.cwd()
DRAW_DIR = ROOT / "draw"
OUT_DIR = ROOT / "PredictionPlot" / "plot_data"

DATASETS = ["Abilene", "Geant"]
DATASET_DISPLAY = {"Abilene": "Abilene", "Geant": "G\u00c9ANT"}

MODEL_DIR_TO_LABEL = OrderedDict(
    [
        ("datp_net", "DATP-Net"),
        ("net", "DARNet"),
        ("DLinear", "DLinear"),
        ("FEDformer", "FEDformer"),
        ("FeTS", "FeTS"),
        ("HMformer", "HMformer"),
        ("informer", "Informer"),
        ("iTransformer", "iTransformer"),
        ("NLinear", "NLinear"),
        ("P_sLSTM", "P_sLSTM"),
        ("PatchTST", "PatchTST"),
        ("PMDformer", "PMDformer"),
        ("timesnet", "TimesNet"),
        ("WPMixer", "WPMixer"),
        ("xLSTMTime", "xLSTMTime"),
        ("xlstm_mixer", "xLSTM-Mixer"),
    ]
)

VARIANT_MODEL_DIR_TO_LABEL = OrderedDict(
    [
        ("datp_net_step", "DATP-Net Step Router"),
        ("datp_net_horizon", "DATP-Net Horizon Router"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PL5 prediction-vs-ground-truth distribution similarity metrics."
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--target_col", default="TC0")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument(
        "--reference_model_dir",
        default="datp_net",
        help="Model directory whose true column is used as the common ground-truth distribution.",
    )
    parser.add_argument(
        "--include_variants",
        action="store_true",
        help="Also include DATP-Net step/horizon router analysis variants.",
    )
    parser.add_argument(
        "--suffix",
        default="PL5_distribution_similarity_metrics_Abilene_Geant",
        help="Output filename stem.",
    )
    return parser.parse_args()


def raw_path(dataset: str, model_dir: str, pred_len: int, d_model: int, target_col: str) -> Path:
    return (
        DRAW_DIR
        / dataset
        / f"{dataset}_single"
        / model_dir
        / f"PL{pred_len}_DM{d_model}"
        / target_col
        / "test_raw.csv"
    )


def finite_1d(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def read_true_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    required = {"true", "pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return finite_1d(df["true"]), finite_1d(df["pred"])


def common_bin_edges(values: list[np.ndarray], bins: int) -> np.ndarray:
    merged = np.concatenate([finite_1d(v) for v in values if finite_1d(v).size])
    if merged.size == 0:
        raise ValueError("No finite values for histogram bins.")
    lo = float(np.min(merged))
    hi = float(np.max(merged))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite histogram range.")
    if lo == hi:
        pad = max(abs(lo) * 1e-6, 1e-12)
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, bins + 1)


def hist_prob(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(finite_1d(values), bins=edges)
    probs = counts.astype(np.float64)
    total = probs.sum()
    if total <= 0:
        return np.full(len(edges) - 1, 1.0 / (len(edges) - 1), dtype=np.float64)
    return probs / total


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(math.sqrt(max(0.0, 0.5 * (kl_pm + kl_qm))))


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / math.sqrt(2.0))


def scale_for_normalization(true_values: np.ndarray) -> float:
    true_values = finite_1d(true_values)
    if true_values.size == 0:
        return np.nan
    q25, q75 = np.quantile(true_values, [0.25, 0.75])
    iqr = float(q75 - q25)
    if np.isfinite(iqr) and iqr > 0:
        return iqr
    std = float(np.std(true_values, ddof=1)) if true_values.size > 1 else np.nan
    if np.isfinite(std) and std > 0:
        return std
    value_range = float(np.max(true_values) - np.min(true_values))
    if np.isfinite(value_range) and value_range > 0:
        return value_range
    return 1.0


def distribution_metrics(
    dataset: str,
    pred_len: int,
    model_dir: str,
    model_label: str,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    edges: np.ndarray,
    source_path: Path,
    reference_path: Path,
) -> dict[str, object]:
    true_values = finite_1d(true_values)
    pred_values = finite_1d(pred_values)
    true_hist = hist_prob(true_values, edges)
    pred_hist = hist_prob(pred_values, edges)

    scale = scale_for_normalization(true_values)
    wasserstein = float(stats.wasserstein_distance(true_values, pred_values))
    ks = stats.ks_2samp(true_values, pred_values)

    true_mean = float(np.mean(true_values))
    pred_mean = float(np.mean(pred_values))
    true_std = float(np.std(true_values, ddof=1)) if true_values.size > 1 else np.nan
    pred_std = float(np.std(pred_values, ddof=1)) if pred_values.size > 1 else np.nan

    return {
        "dataset": DATASET_DISPLAY.get(dataset, dataset),
        "dataset_key": dataset,
        "pred_len": pred_len,
        "model": model_label,
        "model_dir": model_dir,
        "n_true": int(true_values.size),
        "n_pred": int(pred_values.size),
        "wasserstein": wasserstein,
        "wasserstein_norm_iqr": wasserstein / scale if np.isfinite(scale) and scale > 0 else np.nan,
        "js_distance": js_distance(true_hist, pred_hist),
        "hellinger": hellinger_distance(true_hist, pred_hist),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "true_mean": true_mean,
        "pred_mean": pred_mean,
        "mean_abs_diff": abs(pred_mean - true_mean),
        "true_std": true_std,
        "pred_std": pred_std,
        "std_abs_diff": abs(pred_std - true_std),
        "source_path": str(source_path.relative_to(ROOT)),
        "reference_true_path": str(reference_path.relative_to(ROOT)),
    }


def model_map(include_variants: bool) -> OrderedDict[str, str]:
    models = OrderedDict(MODEL_DIR_TO_LABEL)
    if include_variants:
        models.update(VARIANT_MODEL_DIR_TO_LABEL)
    return models


def collect_dataset_rows(dataset: str, args: argparse.Namespace) -> list[dict[str, object]]:
    models = model_map(args.include_variants)
    reference_path = raw_path(dataset, args.reference_model_dir, args.pred_len, args.d_model, args.target_col)
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference true file not found: {reference_path}")
    reference_true, _ = read_true_pred(reference_path)

    loaded: list[tuple[str, str, Path, np.ndarray]] = []
    all_for_bins = [reference_true]
    for model_dir, label in models.items():
        path = raw_path(dataset, model_dir, args.pred_len, args.d_model, args.target_col)
        if not path.exists():
            print(f"Skip missing file: {path}")
            continue
        _, pred = read_true_pred(path)
        loaded.append((model_dir, label, path, pred))
        all_for_bins.append(pred)

    edges = common_bin_edges(all_for_bins, args.bins)
    return [
        distribution_metrics(
            dataset=dataset,
            pred_len=args.pred_len,
            model_dir=model_dir,
            model_label=label,
            true_values=reference_true,
            pred_values=pred,
            edges=edges,
            source_path=path,
            reference_path=reference_path,
        )
        for model_dir, label, path, pred in loaded
    ]


def format_metric(value: object, sci: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    if sci:
        return f"{number:.3e}"
    return f"{number:.6f}"


def markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "model",
        "n_pred",
        "wasserstein_norm_iqr",
        "wasserstein",
        "js_distance",
        "hellinger",
        "ks_stat",
        "mean_abs_diff",
        "std_abs_diff",
    ]
    view = df[columns].copy()
    for col in ["wasserstein", "mean_abs_diff", "std_abs_diff"]:
        view[col] = view[col].map(lambda x: format_metric(x, sci=True))
    for col in ["wasserstein_norm_iqr", "js_distance", "hellinger", "ks_stat"]:
        view[col] = view[col].map(format_metric)
    headers = {
        "model": "Model",
        "n_pred": "N",
        "wasserstein_norm_iqr": "Wasserstein/IQR",
        "wasserstein": "Wasserstein",
        "js_distance": "JS Dist.",
        "hellinger": "Hellinger",
        "ks_stat": "KS Stat.",
        "mean_abs_diff": "Abs Mean Diff",
        "std_abs_diff": "Abs Std Diff",
    }
    view = view.rename(columns=headers)
    labels = list(view.columns)
    string_rows = [[str(row[col]) for col in labels] for _, row in view.iterrows()]
    widths = [
        max(len(str(label)), *(len(row[idx]) for row in string_rows))
        for idx, label in enumerate(labels)
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(labels), separator, *(fmt_row(row) for row in string_rows)])


def write_markdown(df: pd.DataFrame, path: Path, args: argparse.Namespace) -> None:
    lines = [
        "# PL5 Prediction Distribution Similarity",
        "",
        "Lower values indicate the prediction distribution is closer to the ground-truth distribution.",
        "",
        "Metrics:",
        "- `Wasserstein/IQR`: Earth Mover's Distance normalized by the ground-truth IQR. This is usually the most interpretable cross-dataset distance.",
        "- `JS Dist.`: Jensen-Shannon distance on common histogram bins, bounded from 0 to 1.",
        "- `Hellinger`: histogram probability distance, bounded from 0 to 1.",
        "- `KS Stat.`: maximum gap between empirical CDFs.",
        "- `Abs Mean Diff` and `Abs Std Diff`: location and spread mismatch.",
        "",
        f"Scope: datasets = {', '.join(args.datasets)}, pred_len = {args.pred_len}, d_model = {args.d_model}, target_col = {args.target_col}.",
        f"Common ground truth source per dataset: `{args.reference_model_dir}` test_raw true column.",
        "",
    ]
    for dataset in args.datasets:
        display = DATASET_DISPLAY.get(dataset, dataset)
        part = df[df["dataset_key"] == dataset].sort_values(["wasserstein_norm_iqr", "js_distance"])
        lines.extend([f"## {display}", "", markdown_table(part), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        rows.extend(collect_dataset_rows(dataset, args))
    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset_key", "wasserstein_norm_iqr", "js_distance"]).reset_index(drop=True)
    df.insert(0, "rank_by_dataset", df.groupby("dataset_key").cumcount() + 1)

    safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.suffix).strip("_")
    csv_path = OUT_DIR / f"{safe_suffix}.csv"
    md_path = OUT_DIR / f"{safe_suffix}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_markdown(df, md_path, args)

    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
