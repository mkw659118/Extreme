# coding : utf-8
# Author : Yuxiang Zeng
import collections
import numpy as np

from data_center import DataModule
from utils.exp_run_once import RunOnce
from utils.model_efficiency import *
from utils.utils import set_seed


def RunExperiments(log, config):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(config.rounds):
        set_seed(config.seed + runId)
        datamodule = DataModule(config)
        model = Model(datamodule, config)
        datamodule = DataModule(config)
        log.plotter.reset_round()
        results = RunOnce(config, runId, model, datamodule, log)
        for key in results:
            metrics[key].append(results[key])
        log.plotter.append_round()

    log('*' * 20 + 'Experiment Results:' + '*' * 20)
    log(log.exper_detail)
    log(f'Train_length : {len(datamodule.train_loader.dataset)} Valid_length : {len(datamodule.valid_loader.dataset)} Test_length : {len(datamodule.test_loader.dataset)}')

    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')
    try:
        flops, params, inference_time = get_efficiency(config)
        log(f"Flops: {flops:.0f}")
        log(f"Params: {params:.0f}")
        log(f"Inference time: {inference_time:.2f} ms")
    except Exception as e:
        log('Skip the efficiency calculation')

    log.save_in_log(metrics)

    if config.record:
        log.save_result(metrics)
        log.plotter.record_metric(metrics)
    log('*' * 20 + 'Experiment Success' + '*' * 20)
    return metrics