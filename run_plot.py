# coding : utf-8
# Author : Yuxiang Zeng
import os
import torch
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import *
import numpy as np

from data_provider.data_loader import DataModule
from data_provider.data_getitem import TensorDataset
from exp.exp_model import Model
from run_train import get_experiment_name
from utils.utils import set_seed
from utils.exp_logger import Logger
from utils.exp_metrics_plotter import MetricsPlotter
from utils.utils import set_settings
torch.set_default_dtype(torch.float32)


def data_to_dataloader(data_input, label):
    data_set = TensorDataset(data_input, label, 'pred', config)
    bs = 256
    flag = 'test'
    pred_dataloader = DataLoader(
        data_set,
        batch_size=bs,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: data_set.custom_collate_fn(batch, config),
    )
    return pred_dataloader, flag


def save_figure(inputs, label, pred, cnt, scaler, code_idx, config):
    plt.figure(figsize=(12, 6), dpi=300)
    os.makedirs(f'./figs/{config.model}/{code_idx}', exist_ok=True)
    inputs, label, pred = scaler.inverse_transform(inputs), scaler.inverse_transform(label), scaler.inverse_transform(pred)
    input_seq = inputs.reshape(-1)
    real_seq = label.reshape(-1)
    pred_seq = pred.reshape(-1)

    # 计算x轴时间索引
    input_len = len(input_seq)
    future_len = len(pred_seq)

    input_time = np.arange(input_len)  # inputs对应的时间
    future_time = np.arange(input_len, input_len + future_len)  # 预测区间时间
    # 画图：前面是inputs，后面是label和pred
    plt.plot(input_time, input_seq, label='Input', linestyle='-.', marker='s', markersize=3)
    plt.plot(future_time, real_seq, label='Real', linestyle='--', marker='o', markersize=3)
    plt.plot(future_time, pred_seq, label='Pred', linestyle='-', marker='x', markersize=3)
    plt.legend()
    plt.title(f'Prediction vs Real - Sample {cnt}')
    # plt.ylim([0, 1.5])
    plt.xlabel('Time Index')
    plt.ylabel('Value' if not config.classification else 'Class Label')
    plt.grid(True)
    plt.savefig(f'./figs/{config.model}/{code_idx}/{cnt}.jpg')
    plt.close()
    # print(f"Figure {cnt} has done!")


def predict(model, data_input, label, scaler, config):
    dataloader, flag = data_to_dataloader(data_input, label)
    model.setup_optimizer(config)
    cnt = 0
    print(len(dataloader.dataset))
    for batch in (dataloader):
        all_item = [item.to(config.device) for item in batch]
        inputs, label = all_item[:-1], all_item[-1]
        pred = model.forward(*inputs)

        history_value = inputs[0][:, :, :, -1].detach().cpu().numpy()
        pred_value = pred.detach().cpu().numpy()
        real_value = label.detach().cpu().numpy()

        for k in range(history_value.shape[-1]):
            now_idx = cnt
            for i in trange(history_value.shape[0]):
                now_idx += 1
                save_figure(history_value[i, :, k], real_value[i, :, k], pred_value[i, :, k], now_idx, scaler, k, config)

        cnt += label.shape[0]

    return True


def RunOnce(config, runId, log):
    set_seed(config.seed + runId)
    datamodule = DataModule(config)
    model = Model(datamodule, config)
    model_path = f'./checkpoints/{config.model}/{log.filename}_round_{runId}.pt'
    sum_time = pickle.load(open(f'./results/metrics/' + log.filename + '.pkl', 'rb'))['train_time'][runId]
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    except:
        pass
    model.setup_optimizer(config)
    results = model.evaluate_one_epoch(datamodule, 'test')
    if not config.classification:
        log(f'MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} time={sum_time:.1f} s ')
    else:
        log(f'Ac={results["AC"]:.4f} Pr={results["PR"]:.4f} Rc={results["RC"]:.4f} F1={results["F1"]:.4f} time={sum_time:.1f} s ')
    results['train_time'] = sum_time
    config.record = False
    results = predict(model, datamodule.test_set.x, datamodule.test_set.y, datamodule.scaler, config)
    return results


def run(config):
    set_settings(config)
    log_filename, exper_detail = get_experiment_name(config)
    plotter = MetricsPlotter(log_filename, config)
    log = Logger(log_filename, exper_detail, plotter, config)
    metrics = RunOnce(config, 0, log)
    return metrics

if __name__ == '__main__':
    from utils.exp_config import get_config
    config = get_config()
    # config = get_config('MLPConfig')
    # config = get_config('CrossformerConfig')
    # config = get_config('TimesNetConfig')
    run(config)
