import numpy as np
import torch
import collections
import pandas as pd
import hashlib

from data_provider.DS_abilene_diff_single_mask import DS
from exp.exp_model_net_diff import Model
import utils.model_efficiency
import utils.utils

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')


def get_experiment_name(config):
    detail_fields = {
        'Dataset': config.dataset,
        'batchsize': config.bs,
        'Model': config.model,
        'd_model': config.d_model,
        'epochs': config.epochs,
        'patience': config.patience,
        # 'dropout': config.dropout,
    }

    optional_fields = [
        'seq_len',
        'pred_len',
        'enc_in',
        'out_dim',
        'use_missing_aware_encoding',
        'use_state_prior',
        'use_retrieval',
        'retrieval_num',
        'state_num',
        'num_experts',
        'top_k_experts',
        'use_memory',
        'share_weights',
        'mem_size',
        'pretrain_epochs',
    ]

    for field in optional_fields:
        if hasattr(config, field):
            key = field.replace('_', ' ').title().replace(' ', '_')
            value = getattr(config, field)
            if field == 'state_num' and int(value) == 0:
                value = getattr(config, 'num_experts', value)
            detail_fields[key] = value

    exper_detail = ', '.join(f"{k} : {v}" for k, v in detail_fields.items())
    filename = '_'.join(f"{k.replace('_', '')}{v}" for k, v in detail_fields.items())
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

    return filename, exper_detail


def RunExperiments(log, config):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(config.rounds):
        utils.utils.set_seed(config.seed + runId)

        datamodule = DS(config)

        model = Model(config)
        log.plotter.reset_round()

        results = model.RunOnce(config, runId, model, datamodule, log)

        for key in results:
            metrics[key].append(results[key])

        log.plotter.append_round()

    log('*' * 20 + 'Experiment Results:' + '*' * 20)
    log(log.exper_detail)

    log(
        f'Train_length : {len(datamodule.train_data_loader.dataset)} '
        f'Valid_length : {len(datamodule.val_data_loader.dataset)} '
        f'Test_length : {len(datamodule.test_data_loader.dataset)}'
    )

    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.8f} ± {np.std(metrics[key]):.8f}')
    try:
        flops, params, inference_time = utils.model_efficiency.get_efficiency(
            datamodule,
            Model(config),
            config
        )
        log(f'Flops: {flops:.0f}')
        log(f'Params: {params:.0f}')
        log(f'Inference time: {inference_time:.2f} ms')
    except Exception:
        log('Skip the efficiency calculation')

    log.save_in_log(metrics)

    if config.record:
        log.save_result(metrics)
        log.plotter.record_metric(metrics)

    log('*' * 20 + 'Experiment Success' + '*' * 20)
    log.end_the_experiment(model)

    return metrics


def run(config):
    from utils.exp_logger import Logger
    from utils.exp_metrics_plotter import MetricsPlotter
    from utils.utils import set_settings

    set_settings(config)

    filename, exper_detail = get_experiment_name(config)

    plotter = MetricsPlotter(filename, config)
    log = Logger(filename, exper_detail, plotter, config)

    metrics = RunExperiments(log, config)

    return metrics


if __name__ == '__main__':
    import argparse
    from utils.exp_config import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='NetConfig'
    )

    args, _ = parser.parse_known_args()

    config = get_config(args.config)
    run(config)
