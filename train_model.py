# coding : utf-8
# Author : Yuxiang Zeng
import os
import sys
import time
import torch
import pickle
import collections
import numpy as np
from tqdm import *

from baselines.encoder_seq import SeqEncoder
from baselines.mlp import MLP
from baselines.cross_former import Crossformer
from baselines.timesnet import TimesNet
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import set_seed
from utils.efficiency import get_efficiency
from modules.backbone import Backbone
torch.set_default_dtype(torch.float32)

# 每次开展新实验都改一下这里
def get_experiment_name(config):
    log_filename = f'Model_{config.model}_Dataset_{config.dataset}_R{config.rank}'
    return log_filename

class Model(torch.nn.Module):
    def __init__(self, datamodule, config):
        super().__init__()
        self.config = config
        self.input_size = datamodule.train_x.shape[-1]
        self.hidden_size = config.rank

        if config.model == 'ours':
            self.model = Backbone(self.input_size, config)
        elif config.model == 'mlp':
            self.model = MLP(input_dim=self.input_size * config.seq_len, hidden_dim=self.hidden_size, output_dim=config.pred_len, n_layer=config.num_layers, init_method='xavier')
        elif config.model in ['rnn', 'lstm', 'gru']:
            self.model = SeqEncoder(input_size=self.input_size, d_model=self.hidden_size, seq_len=config.seq_len, pred_len=config.seq_len, num_layers=config.num_layers, seq_method=config.model, bidirectional=True)
        elif config.model == 'crossformer':  # 添加 Crossformer 支持
            self.model = Crossformer(
                data_dim=self.input_size,
                in_len=config.seq_len,
                out_len=config.pred_len,
                seg_len=config.seg_len,  # 使用 config.seg_len
                win_size=4,
                d_model=self.hidden_size,
                n_heads=8,
                e_layers=2,
                dropout=0.1,
                device=config.device
            )
        elif config.model == 'timesnet':
            self.model = TimesNet(enc_in=self.input_size, configs=config)

        else:
            raise ValueError(f"Unsupported model type: {config.model}")


    def forward(self, x):
        y = self.model(x)
        return y

    def setup_optimizer(self, config):
        self.to(config.device)
        self.loss_function = get_loss_function(config).to(config.device)
        self.optimizer = get_optimizer(self.parameters(), lr=config.lr, decay=config.decay, config=config)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min' if config.classification else 'max', factor=0.5, patience=config.patience // 1.5, threshold=0.0)

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_batch in (dataModule.train_loader):
            all_item = [item.to(self.config.device) for item in train_batch]
            inputs, label = all_item[:-1], all_item[-1]
            pred = self.forward(*inputs)
            loss = self.loss_function(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        return loss, t2 - t1

    def evaluate_one_epoch(self, dataModule, mode='valid'):
        dataloader = dataModule.valid_loader if mode == 'valid' and len(dataModule.valid_loader.dataset) != 0 else dataModule.test_loader
        preds, reals, val_loss = [], [], 0.
        for batch in (dataloader):
            all_item = [item.to(self.config.device) for item in batch]
            inputs, label = all_item[:-1], all_item[-1]
            pred = self.forward(*inputs)
            if mode == 'valid':
                val_loss += self.loss_function(pred, label)
            if self.config.classification:
                pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
            reals.append(label)
            preds.append(pred)
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        if mode == 'valid':
            self.scheduler.step(val_loss)
        metrics_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value, self.config)
        return metrics_error


def RunOnce(config, runId, log):
    # Set seed of this round
    set_seed(config.seed + runId)

    # Initialize the data and the model
    from data import experiment, DataModule
    exper = experiment(config)
    datamodule = DataModule(exper, config)
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
            monitor.track_one_epoch(epoch, model, valid_error, 'NMAE' if not config.classification else 'AC')
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


def run(config):
    from utils.logger import Logger
    from utils.plotter import MetricsPlotter
    from utils.utils import set_settings
    set_settings(config)
    log_filename = get_experiment_name(config)
    plotter = MetricsPlotter(log_filename, config)
    log = Logger(log_filename, plotter, config)
    try:
        metrics = RunExperiments(log, config)
        # log.send_email(log_filename, metrics, 'zengyuxiang@hnu.edu.cn')
        log.end_the_experiment()
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(error_details)
        sys.exit(1)  # 终止程序，并返回一个非零的退出状态码，表示程序出错
    return metrics


if __name__ == '__main__':
    # Experiment Settings, logger, plotter
    from utils.config import get_config
    config = get_config()
    run(config)
