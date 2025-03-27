# coding : utf-8
# Author : Yuxiang Zeng
import collections
import numpy as np

from utils.exp_run_once import RunOnce
from utils.model_efficiency import *

def RunExperiments(log, config):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(config.rounds):
        log.plotter.reset_round()
        try:
            results = RunOnce(config, runId, log)
            for key in results:
                metrics[key].append(results[key])
            log.plotter.append_round()
        except Exception as e:
            raise Exception
            log(f'Run {runId + 1} Error: {e}, This run will be skipped.')
        except KeyboardInterrupt as e:
            raise KeyboardInterrupt

    log('*' * 20 + 'Experiment Results:' + '*' * 20)
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