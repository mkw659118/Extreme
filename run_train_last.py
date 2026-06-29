import numpy as np
import pandas as pd
import torch
import collections
from data_provider.DS3 import DS
from exp.exp_model_last import Model
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
        'reservoir': config.reservoir_sensor,
        'epochs': config.epochs,
        'patience': config.patience,
        'dropout': config.dropout,
    }

    optional_fields = [
        'use_memory',
        'seq_len',
        'pred_len',
        'share_weights',
        'mem_size',
        'pretrain_epochs',
        'state_prior_distribution',
        'state_prior_scales',
        'state_prior_include_seq_level',
        'num_experts',
        'top_k_experts',
        'retrieval_num',
        'experiment_tag',
    ]
    for field in optional_fields:
        if hasattr(config, field):
            key = field.replace('_', ' ').title().replace(' ', '_')
            detail_fields[key] = getattr(config, field)

    exper_detail = ', '.join(f"{k} : {v}" for k, v in detail_fields.items())
    filename = '_'.join(f"{k.replace('_', '')}{v}" for k, v in detail_fields.items())
    return filename, exper_detail


def prepare_data(config):
    trainX = pd.read_csv(
        './datasets/' + config.dataset + '/' + config.reservoir_sensor + '.tsv', sep='\t'
    )
    trainX.columns = ['datetime', 'value']
    trainX.sort_values('datetime', inplace=True)
    return trainX


def RunExperiments(log, config):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(config.rounds):
        utils.utils.set_seed(config.seed + runId)
        trainX = prepare_data(config)
        datamodule = DS(config, trainX)

        model = Model(config)
        log.plotter.reset_round()
        results = model.RunOnce(config, runId, model, datamodule, log)
        for key in results:
            metrics[key].append(results[key])
        log.plotter.append_round()

    log('*' * 20 + 'Experiment Results:' + '*' * 20)
    log(log.exper_detail)
    log(f'Train_length : {len(datamodule.train_data_loader.dataset)} Valid_length : {len(datamodule.val_data_loader.dataset)} Test_length : {len(datamodule.test_data_loader.dataset)}')

    for key in metrics:
        if 'COS' in key:
            log(f'{key}: {np.mean(metrics[key]):.8f} ± {np.std(metrics[key]):.8f}')
        else:
            log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')

    try:
        flops, params, inference_time = utils.model_efficiency.get_efficiency(datamodule, Model(config), config)
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
        default='ExtremeLSTMMemoConfig'
    )

    args, _ = parser.parse_known_args()

    config = get_config(args.config)
    run(config)
