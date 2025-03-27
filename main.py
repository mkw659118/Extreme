# coding : utf-8
# Author : Yuxiang Zeng
import torch
from utils.exp_run import RunExperiments
torch.set_default_dtype(torch.float32)

def get_experiment_name(config):
    log_filename = f'Model_{config.model}_Dataset_{config.dataset}_{config.idx}_R{config.rank}'
    exper_detail = f"Dataset : {config.dataset.upper()}, Model : {config.model}, Density : {config.density:.3f}, Bs : {config.bs}, Rank : {config.rank}, "
    return log_filename, exper_detail

def run(config):
    from utils.exp_logger import Logger
    from utils.exp_metrics_plotter import MetricsPlotter
    from utils.utils import set_settings
    set_settings(config)
    log_filename, exper_detail = get_experiment_name(config)
    plotter = MetricsPlotter(log_filename, config)
    log = Logger(log_filename, exper_detail, plotter, config)
    metrics = RunExperiments(log, config)
    log.end_the_experiment()
    return metrics


if __name__ == '__main__':
    # Experiment Settings, logger, plotter
    from utils.exp_config import get_config
    config = get_config()
    run(config)
