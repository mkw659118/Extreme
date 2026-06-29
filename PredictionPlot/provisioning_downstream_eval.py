from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


DATASETS = {
    "Abilene": Path("draw/Abilene/Abilene_single"),
    "Geant": Path("draw/Geant/Geant_single"),
}

DATASET_LABELS = {
    "Abilene": "Abilene",
    "Geant": "GÉANT",
}

MODEL_LABELS = {
    "datp_net": "DATP-Net",
    "datp_net_horizon": "DATP-Net-H",
    "datp_net_step": "DATP-Net-S",
    "net": "Net",
    "DLinear": "DLinear",
    "NLinear": "NLinear",
    "PatchTST": "PatchTST",
    "timesnet": "TimesNet",
    "iTransformer": "iTransformer",
    "informer": "Informer",
    "FEDformer": "FEDformer",
    "WPMixer": "WPMixer",
    "P_sLSTM": "P-sLSTM",
    "xLSTMTime": "xLSTMTime",
    "xlstm_mixer": "xLSTM-Mixer",
    "HMformer": "HMformer",
    "PMDformer": "PMDformer",
    "FeTS": "FeTS",
}

PREFERRED_MODELS = [
    "datp_net",
    "PMDformer",
    "HMformer",
    "FeTS",
    "timesnet",
    "iTransformer",
    "PatchTST",
    "WPMixer",
    "P_sLSTM",
    "xLSTMTime",
    "xlstm_mixer",
    "FEDformer",
]

DEFAULT_MODEL_FILTER = set(PREFERRED_MODELS)


@dataclass(frozen=True)
class PredictionSeries:
    dataset: str
    model: str
    label: str
    path: Path
    data: pd.DataFrame


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def output_dataset_slug(dataset: str) -> str:
    return dataset.lower().replace("é", "e")


def find_test_agg_files(dataset_root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    if not dataset_root.exists():
        return files

    for model_dir in dataset_root.iterdir():
        if not model_dir.is_dir():
            continue
        if model_dir.name not in DEFAULT_MODEL_FILTER:
            continue
        candidates = sorted(model_dir.glob("PL*/TC*/test_agg.csv"))
        if not candidates:
            continue
        preferred = [p for p in candidates if any(part.startswith("PL5_") for part in p.parts)]
        files[model_dir.name] = preferred[0] if preferred else candidates[0]
    return files


def load_prediction_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"true", "pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    keep_cols = [c for c in ["time_idx", "count", "index", "true", "pred"] if c in df.columns]
    df = df[keep_cols].copy()
    df["true"] = pd.to_numeric(df["true"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["true", "pred"])
    df["true"] = df["true"].clip(lower=0.0)
    df["pred"] = df["pred"].clip(lower=0.0)
    if "time_idx" not in df.columns:
        df["time_idx"] = np.arange(len(df), dtype=np.int64)
    return df.reset_index(drop=True)


def add_persistence(series: dict[str, PredictionSeries], dataset: str) -> None:
    if not series:
        return
    base = next(iter(series.values())).data[["time_idx", "true"]].copy()
    base["pred"] = base["true"].shift(1)
    base = base.dropna(subset=["pred"]).reset_index(drop=True)
    series["persistence"] = PredictionSeries(
        dataset=dataset,
        model="persistence",
        label="Persistence",
        path=Path("<constructed:true[t-1]>"),
        data=base,
    )


def align_to_reference(series: dict[str, PredictionSeries]) -> dict[str, PredictionSeries]:
    if not series:
        return {}
    ref_model = "datp_net" if "datp_net" in series else next(iter(series))
    ref = series[ref_model].data[["time_idx", "true"]].rename(columns={"true": "true_ref"})
    aligned: dict[str, PredictionSeries] = {}

    for model, item in series.items():
        merged = ref.merge(
            item.data[["time_idx", "true", "pred"]],
            on="time_idx",
            how="inner",
        )
        if merged.empty:
            continue
        true_delta = np.abs(merged["true_ref"].to_numpy() - merged["true"].to_numpy())
        denom = np.maximum(np.abs(merged["true_ref"].to_numpy()), 1e-12)
        mismatch = float(np.mean(true_delta / denom > 1e-4))
        if mismatch > 0.01:
            print(
                f"Warning: {item.dataset}/{item.label} true values differ from reference "
                f"on {mismatch:.1%} aligned rows."
            )
        out = merged[["time_idx", "true_ref", "pred"]].rename(columns={"true_ref": "true"})
        aligned[model] = PredictionSeries(
            dataset=item.dataset,
            model=item.model,
            label=item.label,
            path=item.path,
            data=out.reset_index(drop=True),
        )
    return aligned


def compute_masks(true: np.ndarray, high_load_q: float, high_change_q: float) -> dict[str, np.ndarray]:
    high_load_threshold = np.quantile(true, high_load_q)
    load_mask = true >= high_load_threshold
    diff = np.abs(np.diff(true, prepend=true[0]))
    change_threshold = np.quantile(diff, high_change_q)
    change_mask = diff >= change_threshold
    return {
        "all": np.ones_like(true, dtype=bool),
        f"high_load_q{int(high_load_q * 100)}": load_mask,
        f"high_change_q{int(high_change_q * 100)}": change_mask,
        f"burst_union_q{int(high_load_q * 100)}": load_mask | change_mask,
    }


def provisioning_metrics(true: np.ndarray, pred: np.ndarray, alpha: float, mask: np.ndarray) -> dict[str, float]:
    true_m = true[mask]
    pred_m = pred[mask]
    capacity = pred_m * (1.0 + alpha)
    under = np.maximum(0.0, true_m - capacity)
    over = np.maximum(0.0, capacity - true_m)
    demand = float(np.sum(true_m) + 1e-12)
    reserved = float(np.sum(capacity))
    return {
        "n": int(mask.sum()),
        "alpha": float(alpha),
        "violation_rate": float(np.mean(true_m > capacity)),
        "under_cost": float(np.sum(under)),
        "over_cost": float(np.sum(over)),
        "normalized_under_cost": float(np.sum(under) / demand),
        "normalized_over_cost": float(np.sum(over) / demand),
        "reserved_to_demand": float(reserved / demand),
    }


def mae_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "NMAE": float(np.sum(np.abs(err)) / (np.sum(np.abs(true)) + 1e-12)),
    }


def build_results(
    dataset: str,
    series: dict[str, PredictionSeries],
    alphas: np.ndarray,
    high_load_q: float,
    high_change_q: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for model in ordered_models(series):
        item = series[model]
        true = item.data["true"].to_numpy(dtype=np.float64)
        pred = item.data["pred"].to_numpy(dtype=np.float64)
        masks = compute_masks(true, high_load_q=high_load_q, high_change_q=high_change_q)
        base_metrics = mae_metrics(true, pred)
        metric_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "label": item.label,
                "source_path": str(item.path),
                "n": len(item.data),
                **base_metrics,
            }
        )
        for mask_name, mask in masks.items():
            if not np.any(mask):
                continue
            for alpha in alphas:
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "label": item.label,
                        "subset": mask_name,
                        **provisioning_metrics(true, pred, float(alpha), mask),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(metric_rows)


def ordered_models(series: dict[str, PredictionSeries]) -> list[str]:
    known = [model for model in PREFERRED_MODELS if model in series]
    extra = sorted(model for model in series if model not in known)
    return known + extra


def summarize_at_targets(curves: pd.DataFrame, targets: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, model, label, subset), group in curves.groupby(["dataset", "model", "label", "subset"]):
        group = group.sort_values("alpha")
        for target in targets:
            feasible = group[group["violation_rate"] <= target]
            if feasible.empty:
                chosen = group.iloc[group["violation_rate"].argmin()]
                status = "not_reached"
            else:
                chosen = feasible.iloc[feasible["normalized_over_cost"].argmin()]
                status = "reached"
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "label": label,
                    "subset": subset,
                    "target_violation_rate": target,
                    "status": status,
                    "alpha": chosen["alpha"],
                    "violation_rate": chosen["violation_rate"],
                    "normalized_over_cost": chosen["normalized_over_cost"],
                    "normalized_under_cost": chosen["normalized_under_cost"],
                    "reserved_to_demand": chosen["reserved_to_demand"],
                }
            )
    return pd.DataFrame(rows)


def plot_tradeoff(dataset: str, curves: pd.DataFrame, output_dir: Path, subset: str = "all") -> None:
    df = curves[(curves["dataset"] == dataset) & (curves["subset"] == subset)].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for model in ordered_models({m: None for m in df["model"].unique()}):
        group = df[df["model"] == model].sort_values("alpha")
        if group.empty:
            continue
        label = group["label"].iloc[0]
        lw = 2.8 if model.startswith("datp_net") else 1.35
        alpha = 0.95 if model.startswith("datp_net") else 0.78
        zorder = 3 if model.startswith("datp_net") else 2
        color = "tab:orange" if model.startswith("datp_net") else None
        ax.plot(
            group["normalized_over_cost"],
            group["violation_rate"],
            marker="o" if model.startswith("datp_net") else None,
            markersize=3.2,
            linewidth=lw,
            alpha=alpha,
            label=label,
            color=color,
            zorder=zorder,
        )

    ax.set_xlabel("Normalized over-provisioning cost")
    ax.set_ylabel("SLA violation rate")
    title = f"{dataset_label(dataset)} capacity provisioning trade-off"
    if subset != "all":
        title = f"{title} ({subset})"
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"{output_dataset_slug(dataset)}_{subset}_provisioning_tradeoff.{suffix}", dpi=300)
    plt.close(fig)


def plot_combined_tradeoff(curves: pd.DataFrame, output_dir: Path, subset: str = "all") -> None:
    datasets = [dataset for dataset in ["Abilene", "GÉANT"] if dataset in set(curves["dataset"])]
    if len(datasets) < 2:
        datasets = list(curves["dataset"].drop_duplicates())
    if not datasets:
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(16.8, 6.6), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        df = curves[(curves["dataset"] == dataset) & (curves["subset"] == subset)].copy()
        if df.empty:
            continue

        for model in ordered_models({m: None for m in df["model"].unique()}):
            group = df[df["model"] == model].sort_values("alpha")
            if group.empty:
                continue
            label = group["label"].iloc[0]
            lw = 4.2 if model.startswith("datp_net") else 2.15
            alpha = 0.95 if model.startswith("datp_net") else 0.78
            zorder = 3 if model.startswith("datp_net") else 2
            color = "tab:orange" if model.startswith("datp_net") else None
            ax.plot(
                group["normalized_over_cost"],
                group["violation_rate"],
                marker="o" if model.startswith("datp_net") else None,
                markersize=5.0,
                linewidth=lw,
                alpha=alpha,
                label=label,
                color=color,
                zorder=zorder,
            )

        title = f"{dataset_label(dataset)} capacity provisioning trade-off"
        if subset != "all":
            title = f"{title} ({subset})"
        ax.set_title(title, fontsize=22)
        ax.set_xlabel("Cost", fontsize=20)
        ax.set_ylabel("SLA", fontsize=20)
        ax.tick_params(axis="both", labelsize=18)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, pos: "" if abs(value) < 1e-12 else f"{value:.1f}")
        )
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.legend(ncol=2, fontsize=15, frameon=False)

    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"abilene_geant_{subset}_provisioning_tradeoff_1x2.{suffix}", dpi=300)
    plt.close(fig)


def plot_mae_vs_downstream(summary: pd.DataFrame, errors: pd.DataFrame, output_dir: Path, target: float) -> None:
    df = summary[
        (summary["subset"] == "all")
        & (np.isclose(summary["target_violation_rate"].astype(float), target))
        & (summary["status"] == "reached")
    ].merge(errors[["dataset", "model", "NMAE"]], on=["dataset", "model"], how="left")
    if df.empty:
        return

    for dataset, group in df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        for _, row in group.iterrows():
            color = "tab:orange" if str(row["model"]).startswith("datp_net") else "tab:blue"
            ax.scatter(row["NMAE"], row["normalized_over_cost"], s=46, color=color, alpha=0.85)
            ax.annotate(
                row["label"],
                (row["NMAE"], row["normalized_over_cost"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
        ax.set_xlabel("Prediction NMAE")
        ax.set_ylabel(f"Over-provisioning cost at <= {target:.0%} violations")
        ax.set_title(f"{dataset_label(dataset)}: prediction error vs downstream cost")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        fig.tight_layout()
        for suffix in ["png", "pdf"]:
            fig.savefig(output_dir / f"{output_dataset_slug(dataset)}_nmae_vs_cost_target_{target:g}.{suffix}", dpi=300)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capacity provisioning downstream evaluation.")
    parser.add_argument("--draw-root", type=Path, default=Path("draw"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/provisioning_downstream"))
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument("--high-load-q", type=float, default=0.9)
    parser.add_argument("--high-change-q", type=float, default=0.9)
    parser.add_argument("--targets", type=float, nargs="+", default=[0.10, 0.05, 0.01])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_roots = {
        "Abilene": args.draw_root / "Abilene" / "Abilene_single",
        "Geant": args.draw_root / "Geant" / "Geant_single",
    }
    alphas = np.round(np.arange(0.0, args.alpha_max + args.alpha_step / 2.0, args.alpha_step), 10)

    all_curves: list[pd.DataFrame] = []
    all_errors: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, str]] = []

    for dataset, root in dataset_roots.items():
        files = find_test_agg_files(root)
        loaded: dict[str, PredictionSeries] = {}
        for model, path in files.items():
            try:
                loaded[model] = PredictionSeries(
                    dataset=dataset,
                    model=model,
                    label=model_label(model),
                    path=path,
                    data=load_prediction_csv(path),
                )
            except Exception as exc:
                print(f"Skipping {dataset}/{model}: {exc}")
        aligned = align_to_reference(loaded)
        if not aligned:
            print(f"No usable predictions found for {dataset} under {root}")
            continue

        curves, errors = build_results(
            dataset,
            aligned,
            alphas=alphas,
            high_load_q=args.high_load_q,
            high_change_q=args.high_change_q,
        )
        all_curves.append(curves)
        all_errors.append(errors)
        for model in ordered_models(aligned):
            item = aligned[model]
            manifest_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "label": item.label,
                    "path": str(item.path),
                    "rows": str(len(item.data)),
                }
            )
        print(f"{dataset}: evaluated {len(aligned)} models over {len(alphas)} alpha values.")

    if not all_curves:
        raise SystemExit("No provisioning results were generated.")

    curves_df = pd.concat(all_curves, ignore_index=True)
    errors_df = pd.concat(all_errors, ignore_index=True)
    summary_df = summarize_at_targets(curves_df, args.targets)
    burst_df = summary_df[summary_df["subset"].str.contains("burst|high_", regex=True)].copy()
    manifest_df = pd.DataFrame(manifest_rows)

    for df in [curves_df, errors_df, summary_df, burst_df, manifest_df]:
        if "dataset" in df.columns:
            df["dataset"] = df["dataset"].map(dataset_label)

    curves_df.to_csv(args.output_dir / "tradeoff_curves.csv", index=False)
    errors_df.to_csv(args.output_dir / "prediction_error_summary.csv", index=False)
    summary_df.to_csv(args.output_dir / "summary_at_targets.csv", index=False)
    burst_df.to_csv(args.output_dir / "burst_summary_at_targets.csv", index=False)
    manifest_df.to_csv(args.output_dir / "input_manifest.csv", index=False)

    for dataset in curves_df["dataset"].unique():
        plot_tradeoff(dataset, curves_df, args.output_dir, subset="all")
        for subset in curves_df[curves_df["dataset"] == dataset]["subset"].unique():
            if subset != "all":
                plot_tradeoff(dataset, curves_df, args.output_dir, subset=subset)
    plot_combined_tradeoff(curves_df, args.output_dir, subset="all")
    if args.targets:
        plot_mae_vs_downstream(summary_df, errors_df, args.output_dir, target=args.targets[0])

    print(f"Saved provisioning outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
