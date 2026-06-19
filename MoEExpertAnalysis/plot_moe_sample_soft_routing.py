from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

ROOT = Path.cwd().parent if Path.cwd().name == "MoEExpertAnalysis" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.DATPNetHorizonConfig import DATPNetHorizonConfig
from configs.DATPNetStepConfig import DATPNetStepConfig
from data_provider.DS_abilene_diff_single_mask import DS
from exp.exp_model_net_diff import Model


OUT_DIR = ROOT / "MoEExpertAnalysis"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
CKPT_ROOT = ROOT / "checkpoints"

DATA_FILES = {
    "Abilene": "Abilene_single.csv",
    "Geant": "Geant_single.csv",
}

DISPLAY_NAMES = {
    "Abilene": "Abilene",
    "Geant": "GÉANT",
}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#B8C0CC",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": TOKENS["surface"],
        "axes.facecolor": TOKENS["panel"],
        "figure.dpi": 180,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected-sample soft routing weights for step-level DATP-Net MoE."
    )
    parser.add_argument("--datasets", nargs="+", default=["Abilene", "Geant"])
    parser.add_argument("--config", default="DATPNetStepConfig")
    parser.add_argument("--model", default=None)
    parser.add_argument("--router_granularity", default=None)
    parser.add_argument("--samples_per_dataset", type=int, default=2)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--round_id", type=int, default=0)
    parser.add_argument(
        "--checkpoint_contains",
        default=None,
        help="Optional substring that the checkpoint filename must contain, e.g. a hash.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k_experts", type=int, default=2)
    parser.add_argument(
        "--patch_size",
        type=int,
        default=12,
        help="Average this many history-step tokens into one patch. Use 1 for token-level rows.",
    )
    parser.add_argument(
        "--display_steps",
        type=int,
        default=None,
        help="If set, split the router sequence into this many displayed rows. For PL5 figures, use 5.",
    )
    parser.add_argument(
        "--low_weight_threshold",
        type=float,
        default=0.01,
        help="Sample-selection penalty threshold for near-zero soft routing cells.",
    )
    parser.add_argument(
        "--selection_mode",
        default="balanced",
        choices=["balanced", "max_horizon_variation"],
        help="How to choose representative samples for each dataset.",
    )
    parser.add_argument("--annot_fmt", default=".3f", help="Heatmap annotation number format.")
    parser.add_argument("--tokens_csv", default=None, help="Reuse an existing token-level routing CSV.")
    parser.add_argument("--selected_csv", default=None, help="Reuse an existing selected-samples CSV.")
    parser.add_argument("--hide_annotations", action="store_true", help="Hide heatmap cell values.")
    parser.add_argument("--hide_figure_header", action="store_true", help="Hide the figure title and subtitle.")
    parser.add_argument("--hide_colorbar_label", action="store_true", help="Hide the colorbar label.")
    parser.add_argument(
        "--title_style",
        default="default",
        choices=["default", "expert_weight", "expert_weight_dataset"],
        help="Subplot title style.",
    )
    parser.add_argument(
        "--layout",
        default="dataset_rows",
        choices=["dataset_rows", "dataset_columns"],
        help="Arrange datasets by row or by column.",
    )
    parser.add_argument("--suffix", default="DATPNetStepRouterMoEWeightBalanced")
    parser.add_argument("--allow_overwrite", action="store_true")
    return parser.parse_args()


def unique_path(path: Path, allow_overwrite: bool = False) -> Path:
    if allow_overwrite or not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(2, 1000):
        candidate = path.with_name(f"{stem}_v{idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find a non-overwriting path for {path}")


def checkpoint_filter_for_dataset(dataset: str, args: argparse.Namespace) -> str | None:
    spec = args.checkpoint_contains
    if not spec:
        return None
    if ":" not in spec:
        return spec
    parts = {}
    for item in spec.split(","):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        parts[key.strip()] = value.strip()
    return parts.get(dataset)


def build_config(dataset: str, args: argparse.Namespace):
    config_cls = {
        "DATPNetStepConfig": DATPNetStepConfig,
        "DATPNetHorizonConfig": DATPNetHorizonConfig,
    }.get(args.config)
    if config_cls is None:
        raise ValueError(f"Unsupported config for routing analysis: {args.config}")

    cfg = config_cls()
    cfg.dataset = dataset
    cfg.data_file = DATA_FILES[dataset]
    cfg.pred_len = args.pred_len
    cfg.seq_len = args.seq_len
    cfg.label_len = 0
    cfg.d_model = args.d_model
    cfg.bs = args.batch_size
    cfg.num_experts = args.num_experts
    cfg.top_k_experts = args.top_k_experts
    cfg.retrieval_num = 2
    cfg.state_prior_scales = "1,4,8,16"
    cfg.state_prior_include_seq_level = True
    cfg.use_retrieval = True
    cfg.use_state_prior = True
    cfg.use_missing_aware_encoding = True
    if args.model:
        cfg.model = args.model
    if args.router_granularity:
        cfg.router_granularity = args.router_granularity
    cfg.ensure_all_experts_in_topk = False
    cfg.gate_epochs = 0
    cfg.pretrain_epochs = 0
    cfg.retrain = False
    cfg.device = args.device
    cfg.use_amp = False
    cfg.num_workers = 0
    cfg.mask_zero_as_missing = True
    cfg.artificial_missing_rate = 0.0
    cfg.artificial_missing_splits = "train,val,test"
    cfg.artificial_missing_seed = 2026
    return cfg


def find_checkpoint(dataset: str, args: argparse.Namespace, model_name: str) -> Path:
    ckpt_dir = CKPT_ROOT / model_name
    pattern = (
        f"Dataset{dataset}_Model{model_name}_PL{args.pred_len}_"
        f"DM{args.d_model}_BS{args.batch_size}_*_round_{args.round_id}.pt"
    )
    candidates = [p for p in ckpt_dir.glob(pattern) if p.is_file() and p.stat().st_size > 0]
    checkpoint_filter = checkpoint_filter_for_dataset(dataset, args)
    if checkpoint_filter:
        candidates = [p for p in candidates if checkpoint_filter in p.name]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No step-router checkpoint found with pattern: {ckpt_dir / pattern}")
    return candidates[0]


def safe_torch_load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def infer_checkpoint_input_dim(state: dict[str, torch.Tensor]) -> int | None:
    key = "model.enc_embedding.value_embedding.tokenConv.weight"
    if key in state:
        return int(state[key].shape[1])
    for name, value in state.items():
        if name.endswith("enc_embedding.value_embedding.tokenConv.weight") and hasattr(value, "shape"):
            return int(value.shape[1])
    return None


def collect_sample_router_probs(dataset: str, args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    cfg = build_config(dataset, args)
    ckpt = find_checkpoint(dataset, args, cfg.model)
    state = safe_torch_load(ckpt, cfg.device)

    checkpoint_input_dim = infer_checkpoint_input_dim(state)
    if checkpoint_input_dim is not None:
        cfg.enc_in = checkpoint_input_dim
        cfg.c_in = checkpoint_input_dim
        cfg.dec_in = 1
        cfg.out_dim = 1
        cfg.use_missing_aware_encoding = checkpoint_input_dim % 6 == 0
        cfg.missing_aware_groups = 6 if cfg.use_missing_aware_encoding else 3

    data_module = DS(cfg)
    model = Model(cfg).to(cfg.device)
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in data_module.test_data_loader:
            x, x_mark, label, sample_ids = batch
            x = x.to(cfg.device)
            x_mark = x_mark.to(cfg.device)
            label = label.to(cfg.device)
            sample_ids = sample_ids.to(cfg.device).long()
            dec_input = torch.zeros_like(label[:, -cfg.pred_len :, :]).float()

            output, _, aux = model(
                x,
                x_mark,
                dec_input,
                None,
                sample_ids,
                mode="valid",
                return_aux=True,
            )

            router_prob = aux["router_prob"].detach().cpu().numpy()
            if router_prob.ndim == 2:
                router_prob = router_prob[:, None, :]

            batch_size, step_count, num_experts = router_prob.shape
            prob_flat = router_prob.reshape(-1, num_experts)
            sample_ids_np = sample_ids.detach().cpu().numpy()

            block = {
                "dataset": np.repeat(dataset, batch_size * step_count),
                "sample_id": np.repeat(sample_ids_np, step_count),
                "history_step": np.tile(np.arange(step_count), batch_size),
                "top1_expert": np.argmax(prob_flat, axis=1),
                "top1_prob": np.max(prob_flat, axis=1),
            }
            entropy = -np.sum(prob_flat * np.log(prob_flat + 1e-12), axis=1) / math.log(num_experts)
            block["router_entropy"] = entropy
            for expert in range(num_experts):
                block[f"router_prob_e{expert}"] = prob_flat[:, expert]
            rows.append(pd.DataFrame(block))

    return pd.concat(rows, ignore_index=True), ckpt


def aggregate_router_matrix(
    sample_tokens: pd.DataFrame,
    num_experts: int,
    patch_size: int,
    display_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    expert_cols = [f"router_prob_e{i}" for i in range(num_experts)]
    ordered = sample_tokens.sort_values("history_step")
    steps = ordered["history_step"].to_numpy()
    values = ordered[expert_cols].to_numpy()

    if display_steps is not None and int(display_steps) > 0:
        value_chunks = np.array_split(values, int(display_steps))
        step_chunks = np.array_split(steps, int(display_steps))
    else:
        patch_size = max(1, int(patch_size))
        value_chunks = [values[start : start + patch_size] for start in range(0, len(values), patch_size)]
        step_chunks = [steps[start : start + patch_size] for start in range(0, len(steps), patch_size)]

    matrix = np.asarray([chunk.mean(axis=0) for chunk in value_chunks], dtype=float)
    spans = np.asarray([[int(chunk[0]), int(chunk[-1])] for chunk in step_chunks], dtype=int)
    return matrix, spans


def sample_patch_vector(
    sample_tokens: pd.DataFrame,
    num_experts: int,
    patch_size: int,
    display_steps: int | None = None,
) -> np.ndarray:
    matrix, _ = aggregate_router_matrix(sample_tokens, num_experts, patch_size, display_steps)
    return matrix.reshape(-1)


def choose_samples(
    tokens: pd.DataFrame,
    num_experts: int,
    samples_per_dataset: int,
    patch_size: int,
    display_steps: int | None,
    low_weight_threshold: float,
    selection_mode: str = "balanced",
) -> pd.DataFrame:
    expert_cols = [f"router_prob_e{i}" for i in range(num_experts)]
    rows = []
    for (dataset, sample_id), part in tokens.groupby(["dataset", "sample_id"]):
        mean_prob = part[expert_cols].mean().to_numpy()
        top1_counts = part["top1_expert"].value_counts(normalize=True)
        dominant_expert = int(mean_prob.argmax())
        patch_vector = sample_patch_vector(part, num_experts, patch_size, display_steps)
        horizon_matrix = patch_vector.reshape(-1, num_experts)
        rows.append(
            {
                "dataset": dataset,
                "sample_id": int(sample_id),
                "mean_entropy": float(part["router_entropy"].mean()),
                "specialization": float(1.0 - part["router_entropy"].mean()),
                "top1_mean": float(part["top1_prob"].mean()),
                "dominant_expert": dominant_expert,
                "dominant_share": float(top1_counts.get(dominant_expert, 0.0)),
                "expert_spread": float(mean_prob.max() - mean_prob.min()),
                "min_cell": float(patch_vector.min()),
                "low_weight_fraction": float((patch_vector < low_weight_threshold).mean()),
                "horizon_weight_std": float(horizon_matrix.std(axis=0).mean()),
                "horizon_weight_range": float(horizon_matrix.max() - horizon_matrix.min()),
                "patch_vector": patch_vector,
            }
        )

    summary = pd.DataFrame(rows)
    selected_rows = []
    for dataset, part in summary.groupby("dataset"):
        if selection_mode == "max_horizon_variation":
            ranked = part.sort_values(
                [
                    "low_weight_fraction",
                    "horizon_weight_std",
                    "horizon_weight_range",
                    "min_cell",
                    "specialization",
                ],
                ascending=[True, False, False, False, False],
            ).reset_index(drop=True)
        else:
            ranked = part.sort_values(
                ["low_weight_fraction", "min_cell", "specialization", "expert_spread", "top1_mean"],
                ascending=[True, False, False, False, False],
            ).reset_index(drop=True)
        selected = []
        selected.append(ranked.iloc[0].copy())

        if len(selected) < samples_per_dataset:
            picked_ids = {int(row["sample_id"]) for row in selected}
            pool_size = min(len(ranked), max(80, samples_per_dataset * 40))
            pool = ranked.head(pool_size).copy()
            spec = pool["specialization"].to_numpy(dtype=float)
            spec_range = float(spec.max() - spec.min())
            norm_spec = (spec - spec.min()) / spec_range if spec_range > 1e-12 else np.ones_like(spec)
            min_cell = pool["min_cell"].to_numpy(dtype=float)
            min_range = float(min_cell.max() - min_cell.min())
            norm_min = (min_cell - min_cell.min()) / min_range if min_range > 1e-12 else np.ones_like(min_cell)
            low_frac = pool["low_weight_fraction"].to_numpy(dtype=float)
            low_range = float(low_frac.max() - low_frac.min())
            norm_low = (low_frac - low_frac.min()) / low_range if low_range > 1e-12 else np.zeros_like(low_frac)

            while len(selected) < samples_per_dataset:
                selected_vectors = [np.asarray(row["patch_vector"], dtype=float) for row in selected]
                distances = []
                for _, row in pool.iterrows():
                    if int(row["sample_id"]) in picked_ids:
                        distances.append(-np.inf)
                        continue
                    vector = np.asarray(row["patch_vector"], dtype=float)
                    distances.append(min(float(np.linalg.norm(vector - prev)) for prev in selected_vectors))

                distances = np.asarray(distances, dtype=float)
                finite = np.isfinite(distances)
                if not finite.any():
                    break

                finite_dist = distances[finite]
                dist_range = float(finite_dist.max() - finite_dist.min())
                norm_dist = np.zeros_like(distances)
                if dist_range > 1e-12:
                    norm_dist[finite] = (finite_dist - finite_dist.min()) / dist_range
                else:
                    norm_dist[finite] = 1.0

                if selection_mode == "max_horizon_variation":
                    horizon_std = pool["horizon_weight_std"].to_numpy(dtype=float)
                    std_range = float(horizon_std.max() - horizon_std.min())
                    norm_std = (
                        (horizon_std - horizon_std.min()) / std_range
                        if std_range > 1e-12
                        else np.ones_like(horizon_std)
                    )
                    score = 0.40 * norm_dist + 0.35 * norm_std + 0.15 * norm_min - 0.35 * norm_low
                else:
                    score = 0.45 * norm_dist + 0.25 * norm_min + 0.20 * norm_spec - 0.50 * norm_low
                score[~finite] = -np.inf
                chosen_idx = int(np.argmax(score))
                row = pool.iloc[chosen_idx].copy()
                row["diversity_distance"] = float(distances[chosen_idx])
                selected.append(row)
                picked_ids.add(int(row["sample_id"]))

        for row in selected:
            if "diversity_distance" not in row:
                row["diversity_distance"] = 0.0
            row = row.drop(labels=["patch_vector"], errors="ignore")
            selected_rows.append(row)

        if len(selected) < samples_per_dataset:
            picked_ids = {int(row["sample_id"]) for row in selected}
            for _, row in ranked.iterrows():
                if len(selected) >= samples_per_dataset:
                    break
                if int(row["sample_id"]) not in picked_ids:
                    row = row.copy()
                    row["diversity_distance"] = 0.0
                    row = row.drop(labels=["patch_vector"], errors="ignore")
                    selected_rows.append(row)
                    picked_ids.add(int(row["sample_id"]))
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def aggregate_patches(sample_tokens: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    matrix, spans = aggregate_router_matrix(
        sample_tokens,
        args.num_experts,
        args.patch_size,
        args.display_steps,
    )
    patch_rows = []
    for patch_idx, patch_values in enumerate(matrix, start=1):
        row = {
            "patch": f"S{patch_idx}" if args.display_steps else f"P{patch_idx}",
            "history_start": int(spans[patch_idx - 1, 0]),
            "history_end": int(spans[patch_idx - 1, 1]),
        }
        for expert in range(args.num_experts):
            row[f"E{expert + 1}"] = float(patch_values[expert])
        patch_rows.append(row)
    return pd.DataFrame(patch_rows)


def plot_selected_samples(tokens: pd.DataFrame, selected: pd.DataFrame, args: argparse.Namespace) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = list(args.datasets)
    samples_per_dataset = int(args.samples_per_dataset)
    if args.layout == "dataset_columns":
        nrows = samples_per_dataset
        ncols = len(datasets)
        fig_width = max(9.5, 4.6 * ncols)
        fig_height = max(3.7, 3.3 * nrows)
    else:
        nrows = len(datasets)
        ncols = samples_per_dataset
        fig_width = max(9.5, 4.2 * ncols)
        fig_height = max(6.2, 3.3 * nrows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=False,
    )

    expert_cols = [f"E{i + 1}" for i in range(args.num_experts)]
    patch_frames = []
    heatmap_values = []

    for dataset in datasets:
        dataset_selected = selected[selected["dataset"] == dataset].head(samples_per_dataset)
        for _, row in dataset_selected.iterrows():
            sample_id = int(row["sample_id"])
            sample_tokens = tokens[
                (tokens["dataset"] == dataset) & (tokens["sample_id"] == sample_id)
            ]
            patch_df = aggregate_patches(sample_tokens, args)
            patch_df.insert(0, "dataset", dataset)
            patch_df.insert(1, "sample_id", sample_id)
            patch_frames.append(patch_df)
            heatmap_values.append(patch_df[expert_cols].to_numpy())

    vmax = max(0.5, float(np.max([arr.max() for arr in heatmap_values])))
    vmax = min(1.0, math.ceil(vmax * 10) / 10)
    cmap = sns.color_palette("YlGnBu", as_cmap=True)

    cbar_ax = fig.add_axes([0.92, 0.16, 0.018, 0.68])
    first_heatmap = True

    for row_idx, dataset in enumerate(datasets):
        dataset_selected = selected[selected["dataset"] == dataset].head(samples_per_dataset).reset_index(drop=True)
        for col_idx in range(samples_per_dataset):
            if args.layout == "dataset_columns":
                ax = axes[col_idx, row_idx]
            else:
                ax = axes[row_idx, col_idx]
            if col_idx >= len(dataset_selected):
                ax.axis("off")
                continue

            sample_id = int(dataset_selected.loc[col_idx, "sample_id"])
            sample_tokens = tokens[
                (tokens["dataset"] == dataset) & (tokens["sample_id"] == sample_id)
            ]
            patch_df = aggregate_patches(sample_tokens, args)
            matrix = patch_df[expert_cols]

            sns.heatmap(
                matrix,
                ax=ax,
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                annot=not args.hide_annotations,
                fmt=args.annot_fmt,
                annot_kws={"fontsize": 8.5},
                linewidths=0.4,
                linecolor="#F3F5FA",
                cbar=first_heatmap,
                cbar_ax=cbar_ax if first_heatmap else None,
                cbar_kws={"label": "" if args.hide_colorbar_label else "Soft Routing Weight"},
            )
            first_heatmap = False

            ax.set_xticklabels(expert_cols, rotation=0, fontsize=10)
            ax.set_yticklabels(patch_df["patch"], rotation=0, fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel("")
            if args.title_style == "expert_weight_dataset":
                title = f"{DISPLAY_NAMES.get(dataset, dataset)} Expert Weight"
            elif args.title_style == "expert_weight":
                title = f"{DISPLAY_NAMES.get(dataset, dataset)} Expert Weight of Sample {col_idx + 1}"
            else:
                title = f"{DISPLAY_NAMES.get(dataset, dataset)} Sample {col_idx + 1} (id={sample_id})"
            ax.set_title(
                title,
                fontsize=12,
                fontweight="bold",
                color=TOKENS["ink"],
                pad=8,
            )
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color(TOKENS["axis"])
                ax.spines[spine].set_linewidth(0.8)

    if not args.hide_figure_header:
        fig.suptitle(
            "Soft Routing Weight Distribution Across Experts",
            fontsize=15,
            fontweight="bold",
            color=TOKENS["ink"],
            y=0.975,
        )
    if args.display_steps:
        if (args.router_granularity or "").lower() == "horizon" or args.config == "DATPNetHorizonConfig":
            step_scope = f"future horizon step t+1 to t+{args.display_steps}"
        else:
            step_scope = f"one of {args.display_steps} displayed routing steps"
        subtitle = (
            f"PL{args.pred_len}: each row is {step_scope}; "
            f"columns are dense router probabilities for E1-E{args.num_experts}."
        )
        row_suffix = f"{args.display_steps}steps"
    else:
        subtitle = (
            f"Each row is a patch averaged over {args.patch_size} history-step token(s); "
            f"columns are dense router probabilities for E1-E{args.num_experts}."
        )
        row_suffix = f"patch{args.patch_size}"
    if not args.hide_figure_header:
        fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=10.5, color=TOKENS["muted"])
    top_margin = 0.94 if args.hide_figure_header else 0.89
    fig.subplots_adjust(left=0.08, right=0.89, top=top_margin, bottom=0.08, wspace=0.22, hspace=0.36)

    cbar_ax.tick_params(labelsize=9, colors=TOKENS["ink"])
    if args.hide_colorbar_label:
        cbar_ax.set_ylabel("")
    else:
        cbar_ax.yaxis.label.set_size(10)
        cbar_ax.yaxis.label.set_color(TOKENS["ink"])

    stem = (
        f"moe_sample_soft_routing_{args.suffix}_PL{args.pred_len}_"
        f"{args.samples_per_dataset}samples_{row_suffix}"
    )
    pdf_path = unique_path(FIG_DIR / f"{stem}.pdf", args.allow_overwrite)
    png_path = pdf_path.with_suffix(".png")
    if not args.allow_overwrite:
        png_path = unique_path(png_path, args.allow_overwrite)

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    patch_table = pd.concat(patch_frames, ignore_index=True)
    selected_path = unique_path(DATA_DIR / f"{stem}_selected_samples.csv", args.allow_overwrite)
    patch_path = unique_path(DATA_DIR / f"{stem}_patch_values.csv", args.allow_overwrite)
    selected.to_csv(selected_path, index=False)
    patch_table.to_csv(patch_path, index=False)

    return [pdf_path, png_path, selected_path, patch_path]


def main() -> None:
    args = parse_args()
    if bool(args.tokens_csv) != bool(args.selected_csv):
        raise ValueError("--tokens_csv and --selected_csv must be provided together.")

    checkpoints = {}
    if args.tokens_csv and args.selected_csv:
        tokens_path = Path(args.tokens_csv)
        selected_path = Path(args.selected_csv)
        if not tokens_path.is_absolute():
            tokens_path = ROOT / tokens_path
        if not selected_path.is_absolute():
            selected_path = ROOT / selected_path
        tokens = pd.read_csv(tokens_path)
        selected = pd.read_csv(selected_path)
        print(f"Reuse token CSV: {tokens_path}")
        print(f"Reuse selected-samples CSV: {selected_path}")
    else:
        all_tokens = []
        for dataset in args.datasets:
            print(f"Collect soft routing weights: {dataset} PL{args.pred_len}")
            tokens, checkpoint = collect_sample_router_probs(dataset, args)
            all_tokens.append(tokens)
            checkpoints[dataset] = checkpoint

        tokens = pd.concat(all_tokens, ignore_index=True)
        selected = choose_samples(
            tokens,
            args.num_experts,
            args.samples_per_dataset,
            args.patch_size,
            args.display_steps,
            args.low_weight_threshold,
            args.selection_mode,
        )
    paths = plot_selected_samples(tokens, selected, args)

    token_path = unique_path(
        DATA_DIR
        / f"moe_sample_soft_routing_{args.suffix}_PL{args.pred_len}_tokens.csv",
        args.allow_overwrite,
    )
    tokens.to_csv(token_path, index=False)

    if checkpoints:
        print("Checkpoints:")
        for dataset, checkpoint in checkpoints.items():
            print(f"  {dataset}: {checkpoint}")
    print("Selected samples:")
    print(selected.to_string(index=False))
    print(token_path)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
