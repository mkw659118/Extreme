# coding : utf-8
# Author : Yuxiang Zeng
import os
import torch
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import *
import numpy as np
from data import TensorDataset, custom_collate_fn
from train_model import Model, get_experiment_name
from utils.utils import set_seed
from concurrent.futures import ThreadPoolExecutor
torch.set_default_dtype(torch.float32)

def data_to_dataloader(data_input, label):
    if label is None:
        data_set = TensorDataset(data_input.reshape(-1, 96, 1), torch.randn(data_input.shape[0], 1), 'pred', config)
        print(data_input.reshape(-1, 96, 1).shape)
        bs = 1
        flag = 'pred'
    else:
        data_set = TensorDataset(data_input, label, 'pred', config)
        bs = 256
        flag = 'test'

    pred_dataloader = DataLoader(
        data_set,
        batch_size=bs,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, config),
    )
    return pred_dataloader, flag


def predict(model, data_input, label, config):
    dataloader, flag = data_to_dataloader(data_input, label)
    os.makedirs(f'./figs/{config.model}', exist_ok=True)
    model.setup_optimizer(config)
    cnt = 0
    for batch in tqdm(dataloader):
        all_item = [item.to(config.device) for item in batch]
        inputs, label = all_item[:-1], all_item[-1]
        pred = model.forward(*inputs)
        for j in range(len(pred)):
            if flag == 'test':
                save_figure(inputs[0][j], label[j], pred[j], cnt, config)
            elif flag == 'pred':
                save_figure(inputs[0][j], None, pred[j], cnt, config)
            cnt += 1
    return True


def save_figure(inputs, label, pred, cnt, config):
    plt.figure(figsize=(12, 6), dpi=300)

    if inputs.shape[-1] != 1:
        inputs = inputs[:, -1]

    # exit()
    # 确保inputs和label/pred都是1维
    input_seq = inputs.cpu().reshape(-1).numpy()
    # print(input_seq)
    # print(pred)
    # exit()
    if label is not None:
        real_seq = label.cpu().reshape(-1).numpy()
    pred_seq = pred.cpu().reshape(-1).detach().numpy()

    # 计算x轴时间索引
    input_len = len(input_seq)
    future_len = len(pred_seq)

    input_time = np.arange(input_len)  # inputs对应的时间
    future_time = np.arange(input_len, input_len + future_len)  # 预测区间时间
    # 画图：前面是inputs，后面是label和pred
    plt.plot(input_time, input_seq, label='Input', linestyle='-.', marker='s', markersize=3)
    if label is not None:
        plt.plot(future_time, real_seq, label='Real', linestyle='--', marker='o', markersize=3)
    plt.plot(future_time, pred_seq, label='Pred', linestyle='-', marker='x', markersize=3)

    plt.legend()
    plt.title(f'Prediction vs Real - Sample {cnt}')
    plt.xlabel('Time Index')
    plt.ylabel('Value' if not config.classification else 'Class Label')
    plt.grid(True)
    plt.savefig(f'./figs/{config.model}/{cnt}.jpg')
    plt.close()

    print(f"Figure {cnt} has done!")


def RunOnce(config, runId, log):
    set_seed(config.seed + runId)

    from data import experiment, DataModule
    exper = experiment(config)
    datamodule = DataModule(exper, config)
    model = Model(datamodule, config)
    model_path = f'./checkpoints/{config.model}/{log.filename}_round_{runId}.pt'
    try:
        sum_time = pickle.load(open(f'./results/metrics/' + log.filename + '.pkl', 'rb'))['train_time'][runId]
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
        model.setup_optimizer(config)
        results = model.evaluate_one_epoch(datamodule, 'test')
        if not config.classification:
            log(f'MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} time={sum_time:.1f} s ')
        else:
            log(f'Ac={results["AC"]:.4f} Pr={results["PR"]:.4f} Rc={results["RC"]:.4f} F1={results["F1"]:.4f} time={sum_time:.1f} s ')
        results['train_time'] = sum_time
        config.record = False
    except Exception as e:
        log.only_print(f'Error: {str(e)}')
    # results = model.evaluate_one_epoch(datamodule, 'test')
    results = predict(model, datamodule.test_set.x, datamodule.test_set.y, config)
    # results = predict(model, datamodule.test_set.x[0], None, config)
    return results


def run(config):
    from utils.logger import Logger
    from utils.plotter import MetricsPlotter
    from utils.utils import set_settings
    set_settings(config)
    log_filename = get_experiment_name(config)
    plotter = MetricsPlotter(log_filename, config)
    log = Logger(log_filename, plotter, config)
    metrics = RunOnce(config, 0, log)
    return metrics


if __name__ == '__main__':
    # Experiment Settings, logger, plotter
    from utils.config import get_config
    config = get_config()
    # config = get_config('MLPConfig')
    # config = get_config('CrossformerConfig')
    # config = get_config('TimesNetConfig')
    run(config)
