"""Train an independent ``linear interpolation -> forecast`` baseline."""

import collections
import hashlib

import numpy as np
import torch

from data_provider.DS_abilene_linear_imputation import DS
from exp.exp_model_net_baseline_linear import Model
import utils.model_efficiency
import utils.utils


torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")


def get_experiment_name(config):
    detail_fields = {
        "Dataset": config.dataset,
        "batchsize": config.bs,
        "Model": config.model,
        "Pipeline": "linear_imputation",
        "Missing_pattern": config.artificial_missing_pattern,
        "Missing_rate": config.artificial_missing_rate,
        "Missing_seed": config.artificial_missing_seed,
        "Seq_len": config.seq_len,
        "Pred_len": config.pred_len,
        "d_model": config.d_model,
        "epochs": config.epochs,
        "patience": config.patience,
        "rounds": config.rounds,
    }

    if str(config.artificial_missing_pattern).lower() in {
        "block",
        "time_block",
        "structured_block",
    }:
        detail_fields["Block_length"] = config.artificial_missing_block_length
        detail_fields["Column_rate"] = config.artificial_missing_column_rate

    exper_detail = ", ".join(f"{key} : {value}" for key, value in detail_fields.items())
    raw_filename = "_".join(
        f"{key.replace('_', '')}{value}" for key, value in detail_fields.items()
    )
    digest = hashlib.sha1(raw_filename.encode("utf-8")).hexdigest()[:10]
    filename = (
        f"TwoStageLinear_Dataset{config.dataset}_Model{config.model}_"
        f"Pattern{config.artificial_missing_pattern}_"
        f"Rate{config.artificial_missing_rate}_PL{config.pred_len}_"
        f"DM{config.d_model}_{digest}"
    )
    return filename, exper_detail


def run_experiments(log, config):
    log("*" * 20 + "Two-Stage Linear Experiment Start" + "*" * 20)
    metrics = collections.defaultdict(list)
    datamodule = None
    model = None

    for run_id in range(config.rounds):
        utils.utils.set_seed(config.seed + run_id)
        datamodule = DS(config)
        if config.model.lower() in {"fedformer", "autoformer"}:
            config.dec_in = config.enc_in

        model = Model(config)
        log.plotter.reset_round()
        results = model.RunOnce(config, run_id, model, datamodule, log)

        for key, value in results.items():
            metrics[key].append(value)
        log.plotter.append_round()

    log("*" * 20 + "Two-Stage Linear Experiment Results" + "*" * 20)
    log(log.exper_detail)
    log(
        f"Train_length : {len(datamodule.train_data_loader.dataset)} "
        f"Valid_length : {len(datamodule.val_data_loader.dataset)} "
        f"Test_length : {len(datamodule.test_data_loader.dataset)}"
    )
    for key, values in metrics.items():
        log(f"{key}: {np.mean(values):.8f} +/- {np.std(values):.8f}")

    try:
        flops, params, inference_time = utils.model_efficiency.get_efficiency(
            datamodule,
            Model(config),
            config,
        )
        log(f"Flops: {flops:.0f}")
        log(f"Params: {params:.0f}")
        log(f"Inference time: {inference_time:.2f} ms")
    except Exception as exc:
        log(f"Skip the efficiency calculation: {exc}")

    log.save_in_log(metrics)
    if config.record:
        log.save_result(metrics)
        log.plotter.record_metric(metrics)

    log("*" * 20 + "Two-Stage Linear Experiment Success" + "*" * 20)
    log.end_the_experiment(model)
    return metrics


def run(config):
    from utils.exp_logger import Logger
    from utils.exp_metrics_plotter import MetricsPlotter
    from utils.utils import set_settings

    config.input_imputation = "linear"
    config.two_stage_forecasting = True
    config.experiment_tag = getattr(
        config,
        "experiment_tag",
        "two_stage_linear_baseline",
    )
    set_settings(config)

    filename, exper_detail = get_experiment_name(config)
    plotter = MetricsPlotter(filename, config)
    log = Logger(filename, exper_detail, plotter, config)
    return run_experiments(log, config)


if __name__ == "__main__":
    import argparse

    from utils.exp_config import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="PMDformerConfig")
    args, _ = parser.parse_known_args()
    run(get_config(args.config))

