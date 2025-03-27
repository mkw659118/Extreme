# coding : utf-8
# Author : Yuxiang Zeng
import os
import torch
from tqdm import *
import pickle

from model import Model
from utils.model_monitor import EarlyStopping
from utils.utils import set_seed
from data_center import DataModule


def RunOnce(config, runId, log):
    # Set seed of this round
    set_seed(config.seed + runId)

    # Initialize the data and the model
    datamodule = DataModule(config, True)
    model = Model(datamodule, config)
    try:
        model.compile()
    except Exception as e:
        print(f'Skip the model.compile() because {e}')

    # Setting
    monitor = EarlyStopping(config)
    os.makedirs(f'./checkpoints/{config.model}', exist_ok=True)
    model_path = f'./checkpoints/{config.model}/{log.filename}_round_{runId}.pt'

    # Check if retrain is required or if model file exists
    retrain_required = config.retrain == 1 or not os.path.exists(model_path) and config.continue_train

    if not retrain_required:
        try:
            sum_time = pickle.load(open(f'./results/metrics/' + log.filename + '.pkl', 'rb'))['train_time'][runId]
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
            model.setup_optimizer(config)
            results = model.evaluate_one_epoch(datamodule, 'test')
            if not config.classification:
                log(f'MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} time={sum_time:.1f} s ')
            else:
                log(f'Ac={results["AC"]:.4f} Pr={results["PR"]:.4f} Rc={results["RC"]:.4f} F1={results["F1"]:.4f} time={sum_time:.1f} s ')
            config.record = False
        except Exception as e:
            log.only_print(f'Error: {str(e)}')
            retrain_required = True

    if config.continue_train:
        log.only_print(f'Continue training...')
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))

    if retrain_required:
        model.setup_optimizer(config)
        train_time = []
        for epoch in trange(config.epochs):
            if monitor.early_stop:
                break
            train_loss, time_cost = model.train_one_epoch(datamodule)
            valid_error = model.evaluate_one_epoch(datamodule, 'valid')
            monitor.track_one_epoch(epoch, model, valid_error, config.monitor_metrics)
            log.show_epoch_error(runId, epoch, monitor, train_loss, valid_error, train_time)
            train_time.append(time_cost)
            log.plotter.append_epochs(train_loss, valid_error)
            torch.save(model.state_dict(), model_path)
        model.load_state_dict(monitor.best_model)
        sum_time = sum(train_time[: monitor.best_epoch])
        results = model.evaluate_one_epoch(datamodule, 'test')
        log.show_test_error(runId, monitor, results, sum_time)
        torch.save(monitor.best_model, model_path)
        log.only_print(f'Model parameters saved to {model_path}')

    results['train_time'] = sum_time
    return results