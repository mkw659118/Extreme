"""Standalone training entrypoint for DATP-Net variants.

This module intentionally does not import ``run_train_DARNet.py``. The old
DARNet entrypoint can therefore be removed without breaking DATP-Net runs.
"""

import argparse
import collections
import hashlib

import numpy as np
import torch

from data_provider.DS_abilene_diff_single_mask import DS
from exp.exp_model_DATPNet import Model
from utils.exp_config import get_config
from utils.exp_logger import Logger
from utils.exp_metrics_plotter import MetricsPlotter
from utils.model_efficiency import get_efficiency
from utils.utils import set_seed, set_settings


torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")


DATP_MODEL_NAMES = {
    "datp_net",
    "datp_net_step",
    "datp_net_horizon",
}


def validate_datp_config(config, config_name):
    """Reject accidental use of a non-DATP model config."""
    if config.model not in DATP_MODEL_NAMES:
        allowed = ", ".join(sorted(DATP_MODEL_NAMES))
        raise ValueError(
            "run_train_DATPNet.py only accepts DATP-Net model configs. "
            f"Got config={config_name!r}, model={config.model!r}; "
            f"allowed model values: {allowed}."
        )


def get_experiment_name(config):
    detail_fields = {
        "Dataset": config.dataset,
        "batchsize": config.bs,
        "Model": config.model,
        "d_model": config.d_model,
        "epochs": config.epochs,
        "patience": config.patience,
    }

    optional_fields = [
        "seq_len",
        "pred_len",
        "enc_in",
        "out_dim",
        "use_missing_aware_encoding",
        "use_state_prior",
        "use_retrieval",
        "retrieval_num",
        "state_num",
        "num_experts",
        "top_k_experts",
        "state_prior_distribution",
        "state_prior_scales",
        "state_prior_include_seq_level",
        "artificial_missing_rate",
        "artificial_missing_seed",
        "artificial_missing_splits",
        "artificial_missing_target_only",
        "artificial_missing_pattern",
        "artificial_missing_block_length",
        "artificial_missing_column_rate",
        "experiment_tag",
        "router_granularity",
        "router_balance_weight",
        "topk_coverage_weight",
        "router_entropy_weight",
        "topk_min_usage",
        "moe_weight_balance_weight",
        "moe_min_weight",
        "moe_max_weight",
        "topk_weight_floor_weight",
        "topk_min_head_weight",
        "horizon_diversity_weight",
        "horizon_min_weight_std",
        "horizon_max_cosine",
        "router_temperature",
        "router_train_noise_std",
        "ensure_all_experts_in_topk",
        "horizon_router_use_state_context",
        "use_memory",
        "share_weights",
        "mem_size",
        "pretrain_epochs",
        "gate_epochs",
    ]

    for field in optional_fields:
        if hasattr(config, field):
            key = field.replace("_", " ").title().replace(" ", "_")
            value = getattr(config, field)
            if field == "state_num" and int(value) == 0:
                value = getattr(config, "num_experts", value)
            detail_fields[key] = value

    experiment_detail = ", ".join(
        f"{key} : {value}" for key, value in detail_fields.items()
    )
    filename = "_".join(
        f"{key.replace('_', '')}{value}"
        for key, value in detail_fields.items()
    )
    if len(filename) > 150:
        digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:8]
        filename = (
            f"Dataset{config.dataset}_"
            f"Model{config.model}_"
            f"PL{config.pred_len}_"
            f"DM{config.d_model}_"
            f"BS{config.bs}_"
            f"{digest}"
        )

    return filename, experiment_detail


def run_experiments(log, config):
    log("*" * 20 + "Experiment Start" + "*" * 20)
    metrics = collections.defaultdict(list)
    model = None
    datamodule = None

    for run_id in range(config.rounds):
        set_seed(config.seed + run_id)
        datamodule = DS(config)
        model = Model(config)
        log.plotter.reset_round()

        results = model.RunOnce(
            config,
            run_id,
            model,
            datamodule,
            log,
        )
        for key, value in results.items():
            metrics[key].append(value)
        log.plotter.append_round()

    if model is None or datamodule is None:
        raise ValueError("config.rounds must be at least 1.")

    log("*" * 20 + "Experiment Results:" + "*" * 20)
    log(log.exper_detail)
    log(
        f"Train_length : {len(datamodule.train_data_loader.dataset)} "
        f"Valid_length : {len(datamodule.val_data_loader.dataset)} "
        f"Test_length : {len(datamodule.test_data_loader.dataset)}"
    )

    for key, values in metrics.items():
        log(f"{key}: {np.mean(values):.8f} +/- {np.std(values):.8f}")

    try:
        flops, params, inference_time = get_efficiency(
            datamodule,
            Model(config),
            config,
        )
        log(f"Flops: {flops:.0f}")
        log(f"Params: {params:.0f}")
        log(f"Inference time: {inference_time:.2f} ms")
    except Exception:
        log("Skip the efficiency calculation")

    log.save_in_log(metrics)
    if config.record:
        log.save_result(metrics)
        log.plotter.record_metric(metrics)

    log("*" * 20 + "Experiment Success" + "*" * 20)
    log.end_the_experiment(model)
    return metrics


def run(config):
    set_settings(config)
    filename, experiment_detail = get_experiment_name(config)
    plotter = MetricsPlotter(filename, config)
    log = Logger(filename, experiment_detail, plotter, config)
    return run_experiments(log, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="DATPNetConfig",
        help="DATP-Net config class under configs/ (default: DATPNetConfig).",
    )
    args, _ = parser.parse_known_args()

    config = get_config(args.config)
    validate_datp_config(config, args.config)
    return run(config)


if __name__ == "__main__":
    main()
