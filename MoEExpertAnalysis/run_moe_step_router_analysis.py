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
import torch

ROOT = Path.cwd().parent if Path.cwd().name == "MoEExpertAnalysis" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

COLORS = {
    "topk": "#F4A261",
    "weight": "#6FA8DC",
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
    parser = argparse.ArgumentParser(description="Analyze step-level Top-K MoE routing.")
    parser.add_argument("--datasets", nargs="+", default=["Abilene", "Geant"])
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--round_id", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k_experts", type=int, default=2)
    parser.add_argument("--suffix", default="DATPNetStepRouter")
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


def style_axis(ax) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(True, axis="y", linestyle="--", color=TOKENS["grid"], linewidth=0.75, alpha=0.9)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(TOKENS["axis"])
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(axis="both", colors=TOKENS["ink"], labelsize=10)


def build_config(dataset: str, args: argparse.Namespace) -> DATPNetStepConfig:
    cfg = DATPNetStepConfig()
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
    cfg.router_granularity = "step"
    cfg.router_balance_weight = 0.08
    cfg.topk_coverage_weight = 0.0
    cfg.router_entropy_weight = 0.04
    cfg.topk_min_usage = 0.10
    cfg.router_temperature = 1.2
    cfg.router_train_noise_std = 0.05
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


def find_checkpoint(dataset: str, args: argparse.Namespace) -> Path:
    ckpt_dir = CKPT_ROOT / "datp_net_step"
    pattern = (
        f"Dataset{dataset}_Modeldatp_net_step_PL{args.pred_len}_"
        f"DM{args.d_model}_BS{args.batch_size}_*_round_{args.round_id}.pt"
    )
    candidates = [p for p in ckpt_dir.glob(pattern) if p.is_file() and p.stat().st_size > 0]
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


def collect_step_routing(dataset: str, args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    cfg = build_config(dataset, args)
    ckpt = find_checkpoint(dataset, args)
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
            topk_experts = aux["topk_experts"].detach().cpu().numpy()
            topk_weights = output["topk_probs"].detach().cpu().numpy()

            if router_prob.ndim == 2:
                router_prob = router_prob[:, None, :]
                topk_experts = topk_experts[:, None, :]
                topk_weights = topk_weights[:, None, :]

            batch_size, step_count, num_experts = router_prob.shape
            sparse_moe_weight = np.zeros_like(router_prob)
            np.put_along_axis(sparse_moe_weight, topk_experts, topk_weights, axis=-1)

            prob_flat = router_prob.reshape(-1, num_experts)
            moe_weight_flat = sparse_moe_weight.reshape(-1, num_experts)
            topk_flat = topk_experts.reshape(-1, topk_experts.shape[-1])

            block = {
                "dataset": np.repeat(dataset, batch_size * step_count),
                "pred_len": np.repeat(args.pred_len, batch_size * step_count),
                "sample_id": np.repeat(sample_ids.detach().cpu().numpy(), step_count),
                "history_step": np.tile(np.arange(step_count), batch_size),
                "top1_expert": np.argmax(prob_flat, axis=1),
                "top1_prob": np.max(prob_flat, axis=1),
            }

            entropy = -np.sum(prob_flat * np.log(prob_flat + 1e-12), axis=1) / math.log(num_experts)
            block["router_entropy"] = entropy

            for expert in range(num_experts):
                block[f"router_prob_e{expert}"] = prob_flat[:, expert]
                block[f"moe_weight_e{expert}"] = moe_weight_flat[:, expert]
                block[f"topk_contains_e{expert}"] = (topk_flat == expert).any(axis=1).astype(np.int64)

            rows.append(pd.DataFrame(block))

    return pd.concat(rows, ignore_index=True), ckpt


def summarize_usage(tokens: pd.DataFrame, num_experts: int) -> pd.DataFrame:
    rows = []
    for (dataset, pred_len), part in tokens.groupby(["dataset", "pred_len"]):
        for expert in range(num_experts):
            rows.append(
                {
                    "dataset": dataset,
                    "pred_len": pred_len,
                    "expert": expert,
                    "topk_usage": float(part[f"topk_contains_e{expert}"].mean()),
                    "mean_router_weight": float(part[f"router_prob_e{expert}"].mean()),
                    "mean_moe_weight": float(part[f"moe_weight_e{expert}"].mean()),
                    "top1_usage": float((part["top1_expert"] == expert).mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_by_step(tokens: pd.DataFrame, num_experts: int) -> pd.DataFrame:
    rows = []
    for (dataset, pred_len, step), part in tokens.groupby(["dataset", "pred_len", "history_step"]):
        for expert in range(num_experts):
            rows.append(
                {
                    "dataset": dataset,
                    "pred_len": pred_len,
                    "history_step": step,
                    "expert": expert,
                    "topk_usage": float(part[f"topk_contains_e{expert}"].mean()),
                    "mean_router_weight": float(part[f"router_prob_e{expert}"].mean()),
                    "mean_moe_weight": float(part[f"moe_weight_e{expert}"].mean()),
                }
            )
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, stem: str, args: argparse.Namespace) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = unique_path(DATA_DIR / f"{stem}_{args.suffix}_PL{args.pred_len}.csv", args.allow_overwrite)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_figure(fig, stem: str, args: argparse.Namespace) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = unique_path(FIG_DIR / f"{stem}_{args.suffix}_PL{args.pred_len}.pdf", args.allow_overwrite)
    png_path = pdf_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [pdf_path, png_path]


def plot_usage(usage: pd.DataFrame, args: argparse.Namespace) -> list[Path]:
    datasets = list(usage["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(11.4, 4.25), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(args.num_experts)
    width = 0.34

    for ax, dataset in zip(axes, datasets):
        part = usage[usage["dataset"] == dataset].sort_values("expert")
        ax.bar(
            x - width / 2,
            part["topk_usage"],
            width,
            label="Top-K Activation",
            color=COLORS["topk"],
            edgecolor="#804126",
            linewidth=0.8,
        )
        ax.bar(
            x + width / 2,
            part["mean_moe_weight"],
            width,
            label="Mean MoE Weight",
            color=COLORS["weight"],
            edgecolor="#2E4780",
            linewidth=0.8,
        )
        ax.set_title(DISPLAY_NAMES.get(dataset, dataset), fontsize=16, color=TOKENS["ink"], pad=10)
        ax.set_xlabel("Expert", fontsize=12, color=TOKENS["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_ylim(0, 1.04)
        style_axis(ax)

    axes[0].set_ylabel("Usage / Mean Weight", fontsize=12, color=TOKENS["ink"])
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=True, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=2.0)
    return save_figure(fig, "moe_step_router_usage", args)


def plot_step_heatmap(step_usage: pd.DataFrame, args: argparse.Namespace, value_col: str, stem: str, cbar_label: str) -> list[Path]:
    datasets = list(step_usage["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(12.0, 4.25), sharey=True)
    axes = np.atleast_1d(axes)
    last_im = None

    vmax = 1.0 if value_col == "topk_usage" else max(0.35, float(step_usage[value_col].max()))
    for ax, dataset in zip(axes, datasets):
        part = step_usage[step_usage["dataset"] == dataset]
        matrix = part.pivot(index="expert", columns="history_step", values=value_col).reindex(
            index=range(args.num_experts),
            columns=range(args.seq_len),
            fill_value=0.0,
        )
        last_im = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest", cmap="YlGnBu", vmin=0.0, vmax=vmax)
        ax.set_title(DISPLAY_NAMES.get(dataset, dataset), fontsize=16, color=TOKENS["ink"], pad=10)
        ax.set_xlabel("History Step", fontsize=12, color=TOKENS["ink"])
        ax.set_yticks(np.arange(args.num_experts))
        ax.set_yticklabels([f"Expert {i}" for i in range(args.num_experts)])
        tick_positions = np.linspace(0, args.seq_len - 1, 6, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(i) for i in tick_positions])
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color(TOKENS["axis"])
            ax.spines[spine].set_linewidth(1.0)
        ax.tick_params(axis="both", colors=TOKENS["ink"], labelsize=10)

    axes[0].set_ylabel("Expert", fontsize=12, color=TOKENS["ink"])
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.03, shrink=0.9)
    cbar.set_label(cbar_label, fontsize=11, color=TOKENS["ink"])
    cbar.ax.tick_params(labelsize=9, colors=TOKENS["ink"])
    fig.tight_layout(w_pad=2.0)
    return save_figure(fig, stem, args)


def write_markdown(paths: dict[str, Path], checkpoints: dict[str, Path], usage: pd.DataFrame, args: argparse.Namespace) -> Path:
    def df_to_markdown(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            values = []
            for col in cols:
                value = row[col]
                if isinstance(value, (float, np.floating)):
                    values.append(f"{float(value):.6f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "# Step-Level MoE Routing Analysis",
        "",
        "## Setup",
        "",
        f"- Model: DATP-Net step-level router (`datp_net_step`)",
        f"- Prediction length: {args.pred_len}",
        f"- Datasets: {', '.join(DISPLAY_NAMES.get(d, d) for d in args.datasets)}",
        f"- Experts: {args.num_experts}",
        f"- Top-K: {args.top_k_experts}",
        "- Routing grain: each history step has its own router distribution and Top-K expert set.",
        "- Split for analysis: test set.",
        "",
        "## Checkpoints",
        "",
    ]
    for dataset, path in checkpoints.items():
        lines.append(f"- {DISPLAY_NAMES.get(dataset, dataset)}: `{path.relative_to(ROOT).as_posix()}`")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for label, path in paths.items():
        lines.append(f"- {label}: `{path.relative_to(ROOT).as_posix()}`")

    display_usage = usage.copy()
    display_usage["dataset"] = display_usage["dataset"].map(lambda x: DISPLAY_NAMES.get(x, x))
    lines.extend(
        [
            "",
            "## Expert Usage Summary",
            "",
            df_to_markdown(display_usage),
            "",
            "## Reading Guide",
            "",
            "- `Top-K Activation` is the fraction of history-step tokens where an expert appears in the Top-K set.",
            "- `Mean Router Weight` is the dense router probability averaged over all history-step tokens.",
            "- `Mean MoE Weight` is the sparse normalized Top-K weight actually applied to expert outputs in the MoE fusion.",
            "- If step-level routing is working, the heatmaps should vary across history steps instead of showing one constant sample-level choice.",
            "- The auxiliary balance loss encourages all experts to receive traffic, while the entropy term keeps router weights from becoming exactly identical.",
        ]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = unique_path(OUT_DIR / f"MoE_step_router_experiment_{args.suffix}_PL{args.pred_len}.md", args.allow_overwrite)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    all_tokens = []
    checkpoints = {}

    for dataset in args.datasets:
        if dataset not in DATA_FILES:
            raise ValueError(f"Unsupported dataset: {dataset}")
        print(f"Collect step routing: {dataset} PL{args.pred_len}", flush=True)
        tokens, checkpoint = collect_step_routing(dataset, args)
        all_tokens.append(tokens)
        checkpoints[dataset] = checkpoint

    tokens = pd.concat(all_tokens, ignore_index=True)
    usage = summarize_usage(tokens, args.num_experts)
    step_usage = summarize_by_step(tokens, args.num_experts)

    output_paths: dict[str, Path] = {}
    output_paths["token CSV"] = save_csv(tokens, "moe_step_router_tokens", args)
    output_paths["usage CSV"] = save_csv(usage, "moe_step_router_usage", args)
    output_paths["step CSV"] = save_csv(step_usage, "moe_step_router_by_history_step", args)

    usage_paths = plot_usage(usage, args)
    topk_heatmap_paths = plot_step_heatmap(
        step_usage,
        args,
        value_col="topk_usage",
        stem="moe_step_router_topk_heatmap",
        cbar_label="Top-K Activation",
    )
    weight_heatmap_paths = plot_step_heatmap(
        step_usage,
        args,
        value_col="mean_moe_weight",
        stem="moe_step_router_weight_heatmap",
        cbar_label="Mean MoE Weight",
    )

    output_paths["usage figure"] = usage_paths[0]
    output_paths["Top-K heatmap"] = topk_heatmap_paths[0]
    output_paths["weight heatmap"] = weight_heatmap_paths[0]

    md_path = write_markdown(output_paths, checkpoints, usage, args)
    print(md_path)
    for path in output_paths.values():
        print(path)


if __name__ == "__main__":
    main()
