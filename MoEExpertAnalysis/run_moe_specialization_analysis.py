from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd().parent if Path.cwd().name == "MoEExpertAnalysis" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.NetConfig import NetConfig
from data_provider.DS_abilene_diff_single_mask import DS
from exp.exp_model_net_diff import Model


OUT_DIR = ROOT / "MoEExpertAnalysis"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"
CKPT_DIR = ROOT / "checkpoints" / "net"

DATA_FILES = {
    "Abilene": "Abilene_single.csv",
    "Geant": "Geant_single.csv",
}

COLORS = {
    "Expert 0": "#6FA8DC",
    "Expert 1": "#F4A261",
    "Expert 2": "#7BC8A4",
    "Expert 3": "#C9A0DC",
}

TOKENS = {
    "ink": "#2F3A4A",
    "grid": "#E2E5EA",
    "axis": "#AEB4BE",
    "panel": "#FFFFFF",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Top-K MoE expert specialization.")
    parser.add_argument("--datasets", nargs="+", default=["Abilene", "Geant"])
    parser.add_argument("--pred_lens", nargs="+", type=int, default=[5])
    parser.add_argument("--round_id", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k_experts", type=int, default=2)
    parser.add_argument("--max_rep_per_expert", type=int, default=3)
    return parser.parse_args()


def build_config(dataset: str, pred_len: int, args: argparse.Namespace) -> NetConfig:
    cfg = NetConfig()
    cfg.model = "net"
    cfg.dataset = dataset
    cfg.data_file = DATA_FILES[dataset]
    cfg.pred_len = pred_len
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
    cfg.retrain = False
    cfg.device = args.device
    cfg.use_amp = False
    cfg.num_workers = 0
    cfg.mask_zero_as_missing = True
    cfg.artificial_missing_rate = 0.0
    cfg.artificial_missing_splits = "train,val,test"
    cfg.artificial_missing_seed = 2026
    return cfg


def find_checkpoint(dataset: str, pred_len: int, args: argparse.Namespace) -> Path:
    pattern = f"Dataset{dataset}_Modelnet_PL{pred_len}_DM{args.d_model}_BS{args.batch_size}_*_round_{args.round_id}.pt"
    candidates = [
        p for p in CKPT_DIR.glob(pattern)
        if p.is_file() and p.stat().st_size > 0
    ]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    legacy = (
        f"Dataset{dataset}_batchsize{args.batch_size}_Modelnet_dmodel{args.d_model}_"
        f"epochs200_patience40_SeqLen{args.seq_len}_PredLen{pred_len}_*.pt"
    )
    candidates = [
        p for p in CKPT_DIR.glob(legacy)
        if p.is_file() and p.stat().st_size > 0
    ]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [p for p in candidates if f"round_{args.round_id}" in p.stem]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No checkpoint found for {dataset} PL{pred_len} round {args.round_id}.")


def safe_torch_load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def masked_mean(values: np.ndarray, mask: np.ndarray, axis=None) -> np.ndarray:
    weights = (mask > 0.5).astype(np.float64)
    denom = np.maximum(weights.sum(axis=axis), 1.0)
    return (values * weights).sum(axis=axis) / denom


def masked_max_abs(values: np.ndarray, mask: np.ndarray, axis=None) -> np.ndarray:
    valid = mask > 0.5
    arr = np.where(valid, np.abs(values), np.nan)
    out = np.nanmax(arr, axis=axis)
    return np.where(np.isfinite(out), out, 0.0)


def infer_checkpoint_input_dim(state: dict[str, torch.Tensor]) -> int | None:
    key = "model.enc_embedding.value_embedding.tokenConv.weight"
    if key in state:
        return int(state[key].shape[1])
    for name, value in state.items():
        if name.endswith("enc_embedding.value_embedding.tokenConv.weight") and hasattr(value, "shape"):
            return int(value.shape[1])
    return None


def extract_input_features(x: np.ndarray, label: np.ndarray, label_mask: np.ndarray, num_vars: int) -> dict[str, np.ndarray]:
    c = num_vars
    if x.shape[-1] == 6 * c:
        diff_norm = x[:, :, 0:c]
        diff_mask = x[:, :, c : 2 * c]
        raw = x[:, :, 4 * c : 5 * c]
        raw_mask = x[:, :, 5 * c : 6 * c]
    elif x.shape[-1] == 3 * c:
        diff_norm = x[:, :, 0:c]
        diff_mask = np.ones_like(diff_norm, dtype=np.float32)
        raw = x[:, :, 2 * c : 3 * c]
        raw_mask = np.isfinite(raw).astype(np.float32)
    else:
        diff_norm = x
        diff_mask = np.ones_like(diff_norm, dtype=np.float32)
        raw = x
        raw_mask = np.isfinite(raw).astype(np.float32)

    raw_mean = masked_mean(raw, raw_mask, axis=(1, 2))
    raw_centered = raw - raw_mean[:, None, None]
    raw_std = np.sqrt(masked_mean(raw_centered ** 2, raw_mask, axis=(1, 2)))
    raw_last = raw[:, -1, :].mean(axis=1)
    missing_rate = 1.0 - raw_mask.mean(axis=(1, 2))
    abs_diff_mean = masked_mean(np.abs(diff_norm), diff_mask, axis=(1, 2))
    abs_diff_max = masked_max_abs(diff_norm, diff_mask, axis=(1, 2))
    future_abs_diff_mean = masked_mean(np.abs(label), label_mask, axis=(1, 2))
    future_abs_diff_max = masked_max_abs(label, label_mask, axis=(1, 2))

    return {
        "input_mean": raw_mean,
        "input_std": raw_std,
        "last_value": raw_last,
        "missing_rate": missing_rate,
        "abs_diff_mean": abs_diff_mean,
        "abs_diff_max": abs_diff_max,
        "future_abs_diff_mean": future_abs_diff_mean,
        "future_abs_diff_max": future_abs_diff_max,
    }


def collect_routing(dataset: str, pred_len: int, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cfg = build_config(dataset, pred_len, args)
    ckpt = find_checkpoint(dataset, pred_len, args)
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
    representative = {
        "dataset": dataset,
        "pred_len": pred_len,
        "x_raw": [],
        "sample_id": [],
        "top1_expert": [],
        "top1_prob": [],
        "router_prob": [],
    }

    num_experts = int(cfg.num_experts)
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
            topk_experts = aux["topk_experts"].detach().cpu().numpy()
            state_probs = aux["state_probs"].detach().cpu().numpy()
            top1_expert = np.argmax(router_prob, axis=1)
            top1_prob = np.max(router_prob, axis=1)
            state_id = np.argmax(state_probs, axis=1)
            routing_entropy = -np.sum(router_prob * np.log(router_prob + 1e-12), axis=1) / math.log(num_experts)

            x_np = x.detach().cpu().numpy()
            label_np = label.detach().cpu().numpy()
            mask_np = x_mark.detach().cpu().numpy()
            features = extract_input_features(x_np, label_np, mask_np, data_module.num_vars)

            c = data_module.num_vars
            if x_np.shape[-1] == 6 * c:
                x_raw = x_np[:, :, 4 * c : 5 * c].mean(axis=2)
            elif x_np.shape[-1] == 3 * c:
                x_raw = x_np[:, :, 2 * c : 3 * c].mean(axis=2)
            else:
                x_raw = x_np.mean(axis=2)

            for i in range(x_np.shape[0]):
                row = {
                    "dataset": dataset,
                    "pred_len": pred_len,
                    "sample_id": int(sample_ids[i].detach().cpu().item()),
                    "top1_expert": int(top1_expert[i]),
                    "top1_prob": float(top1_prob[i]),
                    "state_id": int(state_id[i]),
                    "routing_entropy": float(routing_entropy[i]),
                }
                for e in range(num_experts):
                    row[f"router_prob_e{e}"] = float(router_prob[i, e])
                    row[f"state_prob_s{e}"] = float(state_probs[i, e]) if e < state_probs.shape[1] else np.nan
                    row[f"topk_contains_e{e}"] = int(e in set(topk_experts[i].tolist()))
                for key, value in features.items():
                    row[key] = float(value[i])
                rows.append(row)

                representative["x_raw"].append(x_raw[i])
                representative["sample_id"].append(row["sample_id"])
                representative["top1_expert"].append(row["top1_expert"])
                representative["top1_prob"].append(row["top1_prob"])
                representative["router_prob"].append(router_prob[i].copy())

    df = pd.DataFrame(rows)
    for key in ["x_raw", "sample_id", "top1_expert", "top1_prob", "router_prob"]:
        representative[key] = np.asarray(representative[key], dtype=object if key == "x_raw" else None)
    representative["checkpoint"] = str(ckpt)
    return df, representative


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    n = len(x)
    if n == 0:
        return np.nan
    mi = 0.0
    for xi in np.unique(x):
        for yi in np.unique(y):
            pxy = np.mean((x == xi) & (y == yi))
            if pxy <= 0:
                continue
            px = np.mean(x == xi)
            py = np.mean(y == yi)
            mi += pxy * math.log(pxy / (px * py + 1e-12) + 1e-12)
    return float(mi)


def summarize(df: pd.DataFrame, num_experts: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    usage_rows = []
    align_rows = []
    for (dataset, pred_len), part in df.groupby(["dataset", "pred_len"]):
        n = len(part)
        top1 = part["top1_expert"].to_numpy()
        state = part["state_id"].to_numpy()
        summary_rows.append(
            {
                "dataset": dataset,
                "pred_len": pred_len,
                "sample_count": n,
                "mean_routing_entropy": part["routing_entropy"].mean(),
                "expert_state_mutual_information": max(0.0, mutual_information(state, top1)),
                "dominant_expert_ratio": part["top1_expert"].value_counts(normalize=True).max(),
            }
        )
        for e in range(num_experts):
            usage_rows.append(
                {
                    "dataset": dataset,
                    "pred_len": pred_len,
                    "expert": e,
                    "top1_usage": float(np.mean(top1 == e)),
                    "topk_usage": float(part[f"topk_contains_e{e}"].mean()),
                    "weighted_usage": float(part[f"router_prob_e{e}"].mean()),
                }
            )
        for s in range(num_experts):
            state_part = part[part["state_id"] == s]
            for e in range(num_experts):
                align_rows.append(
                    {
                        "dataset": dataset,
                        "pred_len": pred_len,
                        "state": s,
                        "expert": e,
                        "mean_router_prob": float(state_part[f"router_prob_e{e}"].mean()) if len(state_part) else 0.0,
                        "top1_ratio": float(np.mean(state_part["top1_expert"].to_numpy() == e)) if len(state_part) else 0.0,
                        "state_count": int(len(state_part)),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(usage_rows), pd.DataFrame(align_rows)


def style_axis(ax) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(True, axis="y", linestyle="--", color=TOKENS["grid"], linewidth=0.7, alpha=0.85)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(axis="both", colors=TOKENS["ink"], labelsize=10)


def plot_usage(usage: pd.DataFrame, pred_len: int, num_experts: int) -> Path:
    datasets = list(usage["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(11.2, 4.1), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.24
    x = np.arange(num_experts)
    for ax, dataset in zip(axes, datasets):
        part = usage[(usage["dataset"] == dataset) & (usage["pred_len"] == pred_len)].sort_values("expert")
        ax.bar(x - width, part["top1_usage"], width, label="Top-1", color="#6FA8DC")
        ax.bar(x, part["topk_usage"], width, label="Top-K", color="#F4A261")
        ax.bar(x + width, part["weighted_usage"], width, label="Weighted", color="#7BC8A4")
        ax.set_title(dataset, fontsize=15, color=TOKENS["ink"])
        ax.set_xlabel("Expert", fontsize=12, color=TOKENS["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        style_axis(ax)
    axes[0].set_ylabel("Usage Ratio", fontsize=12, color=TOKENS["ink"])
    axes[-1].legend(frameon=True, fontsize=10)
    fig.tight_layout(w_pad=2.0)
    path = FIG_DIR / f"moe_expert_usage_PL{pred_len}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def plot_alignment(align: pd.DataFrame, pred_len: int, num_experts: int) -> Path:
    datasets = list(align["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(10.6, 4.25))
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets):
        part = align[(align["dataset"] == dataset) & (align["pred_len"] == pred_len)]
        mat = part.pivot(index="state", columns="expert", values="mean_router_prob").reindex(
            index=range(num_experts),
            columns=range(num_experts),
            fill_value=0.0,
        )
        im = ax.imshow(mat.to_numpy(), cmap="YlGnBu", vmin=0.0, vmax=max(0.5, mat.to_numpy().max()))
        ax.set_title(dataset, fontsize=15, color=TOKENS["ink"])
        ax.set_xlabel("Expert", fontsize=12, color=TOKENS["ink"])
        ax.set_ylabel("State Component", fontsize=12, color=TOKENS["ink"])
        ax.set_xticks(range(num_experts))
        ax.set_yticks(range(num_experts))
        for i in range(num_experts):
            for j in range(num_experts):
                ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9, color="#1F2430")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="Mean Router Probability")
    fig.tight_layout(w_pad=2.0)
    path = FIG_DIR / f"moe_state_expert_alignment_PL{pred_len}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def plot_features(df: pd.DataFrame, pred_len: int, num_experts: int) -> Path:
    features = [
        ("input_std", "Input Std"),
        ("abs_diff_max", "Max |Input Diff|"),
        ("future_abs_diff_max", "Max |Future Diff|"),
        ("routing_entropy", "Routing Entropy"),
    ]
    datasets = list(df["dataset"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), len(features), figsize=(15.4, 6.8), sharex=True)
    for r, dataset in enumerate(datasets):
        part = df[(df["dataset"] == dataset) & (df["pred_len"] == pred_len)]
        for c, (feature, title) in enumerate(features):
            ax = axes[r, c]
            values = [part.loc[part[f"topk_contains_e{e}"] == 1, feature].to_numpy() for e in range(num_experts)]
            bp = ax.boxplot(values, patch_artist=True, showfliers=False)
            for e, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(COLORS[f"Expert {e}"])
                patch.set_alpha(0.55)
                patch.set_edgecolor(TOKENS["axis"])
            for median in bp["medians"]:
                median.set_color("#2F3A4A")
                median.set_linewidth(1.5)
            ax.set_title(title if r == 0 else "", fontsize=12, color=TOKENS["ink"])
            if c == 0:
                ax.set_ylabel(dataset, fontsize=13, color=TOKENS["ink"])
            ax.set_xlabel("Expert", fontsize=11, color=TOKENS["ink"])
            ax.set_xticks(range(1, num_experts + 1))
            ax.set_xticklabels([str(i) for i in range(num_experts)])
            style_axis(ax)
    fig.tight_layout(w_pad=1.4, h_pad=1.2)
    path = FIG_DIR / f"moe_feature_by_topk_expert_PL{pred_len}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def plot_representatives(rep_by_key: dict[tuple[str, int], dict[str, np.ndarray]], pred_len: int, num_experts: int, max_rep: int) -> Path:
    datasets = [key[0] for key in rep_by_key if key[1] == pred_len]
    fig, axes = plt.subplots(len(datasets), num_experts, figsize=(15.4, 6.6), sharex=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]
    for r, dataset in enumerate(datasets):
        rep = rep_by_key[(dataset, pred_len)]
        top1 = rep["top1_expert"].astype(int)
        router_prob = np.vstack(rep["router_prob"]).astype(float)
        x_raw = rep["x_raw"]
        for e in range(num_experts):
            ax = axes[r, e]
            idx = np.arange(router_prob.shape[0])
            if idx.size:
                chosen = idx[np.argsort(router_prob[idx, e])[-max_rep:]][::-1]
                for j, sample_idx in enumerate(chosen):
                    y = np.asarray(x_raw[sample_idx], dtype=np.float64)
                    ax.plot(y, linewidth=1.6, alpha=0.78, label=f"p={router_prob[sample_idx, e]:.2f}")
            ax.set_title(f"{dataset} E{e}", fontsize=12, color=TOKENS["ink"])
            if e == 0:
                ax.set_ylabel("Input Raw", fontsize=11, color=TOKENS["ink"])
            ax.set_xlabel("History Step", fontsize=10, color=TOKENS["ink"])
            style_axis(ax)
            if idx.size:
                ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout(w_pad=1.2, h_pad=1.3)
    path = FIG_DIR / f"moe_representative_samples_PL{pred_len}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def write_markdown(summary: pd.DataFrame, paths: list[Path], pred_lens: list[int], args: argparse.Namespace) -> Path:
    def df_to_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            values = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.6g}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "# Top-K MoE Expert Specialization Visualization",
        "",
        "## Objective",
        "",
        "This experiment checks whether the Top-K MoE router in DATP-Net forms expert specialization on Abilene and Geant.",
        "",
        "## Setup",
        "",
        f"- Datasets: {', '.join(args.datasets)}",
        f"- Prediction lengths: {', '.join(map(str, pred_lens))}",
        f"- Number of experts: {args.num_experts}",
        f"- Top-K experts: {args.top_k_experts}",
        f"- Checkpoint round: {args.round_id}",
        "- Split used for analysis: test set",
        "- Routing source: Student-T state prior responsibilities -> router -> Top-K experts",
        "",
        "## Collected Routing Signals",
        "",
        "- `router_prob`: dense probability over experts for each test sample.",
        "- `topk_experts`: experts selected by Top-K routing.",
        "- `state_probs`: Student-T state prior responsibility vector.",
        "- `top1_expert`: expert with the largest router probability.",
        "- `routing_entropy`: normalized entropy of expert probabilities; lower means sharper routing.",
        "",
        "## Visualizations",
        "",
    ]
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        lines.append(f"- [{path.stem}]({rel})")
    lines.extend(
        [
            "",
            "## Summary Metrics",
            "",
            df_to_markdown(summary),
            "",
            "## Interpretation Guide",
            "",
            "- Expert usage shows whether the router collapses to a small number of experts or uses multiple experts.",
            "- State-expert alignment shows whether Student-T latent states map preferentially to different experts.",
            "- Feature-by-expert boxplots show whether experts receive samples with different volatility, future change, or routing certainty.",
            "- Representative samples show the input patterns most confidently assigned to each expert.",
            "",
            "Stronger specialization is indicated by non-uniform state-expert heatmaps, distinct feature distributions by expert, and clear representative-pattern differences.",
        ]
    )
    path = OUT_DIR / "TopK_MoE_expert_specialization_experiment.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    rep_by_key = {}
    for dataset in args.datasets:
        if dataset not in DATA_FILES:
            raise ValueError(f"Unsupported dataset: {dataset}")
        for pred_len in args.pred_lens:
            print(f"Collect routing: {dataset} PL{pred_len}", flush=True)
            df, rep = collect_routing(dataset, pred_len, args)
            df.to_csv(DATA_DIR / f"{dataset}_PL{pred_len}_moe_routing_samples.csv", index=False, encoding="utf-8-sig")
            all_rows.append(df)
            rep_by_key[(dataset, pred_len)] = rep

    routing_df = pd.concat(all_rows, ignore_index=True)
    routing_df.to_csv(DATA_DIR / "moe_routing_samples_all.csv", index=False, encoding="utf-8-sig")
    summary, usage, align = summarize(routing_df, args.num_experts)
    summary.to_csv(DATA_DIR / "moe_specialization_summary.csv", index=False, encoding="utf-8-sig")
    usage.to_csv(DATA_DIR / "moe_expert_usage.csv", index=False, encoding="utf-8-sig")
    align.to_csv(DATA_DIR / "moe_state_expert_alignment.csv", index=False, encoding="utf-8-sig")

    figure_paths = []
    for pred_len in args.pred_lens:
        figure_paths.append(plot_usage(usage, pred_len, args.num_experts))
        figure_paths.append(plot_alignment(align, pred_len, args.num_experts))
        figure_paths.append(plot_features(routing_df, pred_len, args.num_experts))
        figure_paths.append(plot_representatives(rep_by_key, pred_len, args.num_experts, args.max_rep_per_expert))

    md_path = write_markdown(summary, figure_paths, args.pred_lens, args)
    print(md_path)
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
