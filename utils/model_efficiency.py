# coding : utf-8
# Author : yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年1月17日19:47:38）

import sys
import os
import torch
from exp.exp_model import Model
import run_train

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)


def _prepare_model_inputs(model, sample_input, config):
    """Convert a training batch into forecasting-model arguments."""
    batch = tuple(sample_input)
    if len(batch) < 3:
        raise ValueError(
            f"Expected at least x, x_mark and label, got {len(batch)} batch items"
        )

    x, x_mark, label = (item.to(config.device) for item in batch[:3])
    if hasattr(model, "_prepare_model_x_mark"):
        x_mark = model._prepare_model_x_mark(x, x_mark)
        pred_len = int(config.pred_len)
        label_len = int(config.label_len)
        dec_input = torch.zeros_like(label[:, -pred_len:, :]).float()
        dec_input = torch.cat(
            [label[:, :label_len, :], dec_input],
            dim=1,
        ).to(config.device)
        return x, x_mark, dec_input, None

    return tuple(item.to(config.device) for item in batch[:-1])


def _get_train_loader(datamodule):
    loader = getattr(datamodule, "train_loader", None)
    if loader is None:
        loader = getattr(datamodule, "train_data_loader", None)
    if loader is None:
        raise AttributeError(
            "Data module must provide train_loader or train_data_loader"
        )
    return loader

def calculate_flops_params(model, sample_input, config):
    model_inputs = _prepare_model_inputs(model, sample_input, config)
    model = model.to(config.device)
    params = sum(parameter.numel() for parameter in model.parameters())
    try:
        from thop import profile
    except ModuleNotFoundError as exc:
        if exc.name != "thop":
            raise
        return float("nan"), float(params)

    flops, params = profile(model, inputs=model_inputs, verbose=False)
    # config.log.only_print(f"Flops: {flops} Params: {params}")
    return flops, params


def calculate_inference_time(model, sample_input, config):
    from time import time
    import numpy as np
    step = 100
    all_time = []
    inputs = _prepare_model_inputs(model, sample_input, config)
    for i in range(step):
        t1 = time()
        model(*inputs)  # 动态解包传递所有输入到模型
        t2 = time()
        all_time.append(t2 - t1)
    inference_time = np.mean(all_time)
    # config.log.only_print(f"Average Inference Time: {inference_time * 1000:.2f} ms")
    return inference_time * 1000


def get_efficiency(datamodule, model, config):
    sample_inputs = next(iter(_get_train_loader(datamodule)))
    flops, params = calculate_flops_params(model, sample_inputs, config)
    inference_time = calculate_inference_time(model, sample_inputs, config)
    return flops, params, inference_time


def only_run():
    from utils.exp_config import get_config
    from utils.exp_logger import Logger
    from utils.exp_metrics_plotter import MetricsPlotter
    from utils.utils import set_settings
    config = get_config()
    set_settings(config)
    log_filename = f'Model_{config.model}_{config.dataset}_S{config.train_size}_R{config.rank}_Ablation{config.Ablation}'
    plotter = MetricsPlotter(log_filename, config)
    # filename, exper_detail, plotter, config,
    log_filename, exper_detail = run_train.get_experiment_name(config)
    log = Logger(log_filename, exper_detail, plotter, config, show_params=False)

    datamodule = DataModule(config)
    model = Model(datamodule, config).to(config.device)

    sample_inputs = next(iter(datamodule.train_loader))
    flops, params = calculate_flops_params(model, sample_inputs, config)
    inference_time = calculate_inference_time(model, sample_inputs, config)
    print(f"Flops: {flops:.0f}")
    print(f"Params: {params:.0f}")
    print(f"Inference time: {inference_time:.2f} ms")
    return flops, params, inference_time


if __name__ == '__main__':
    only_run()
