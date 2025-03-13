# coding : utf-8
# Author : yuxiang Zeng
import time
import subprocess
import numpy as np
from datetime import datetime
from model_train import get_experiment_name
import pickle
from itertools import product
from utils.exp_config import get_config


# 搜索最佳超参数然后取最佳
def add_parameter(command: str, params: dict) -> str:
    for param_name, param_value in params.items():
        command += f" --{param_name} {param_value}"
    return command

def once_experiment(exper_name, hyper_dict, grid_search=0, retrain=1, debug=0):
    # 先进行超参数探索
    best_hyper = hyper_search(exper_name, hyper_dict, grid_search=grid_search, retrain=retrain, debug=debug)

    # 再跑最佳参数实验
    commands = []
    command = f"python model_train.py --exp_name {exper_name} --retrain 1"
    commands.append(command)

    commands = [add_parameter(command, best_hyper) for command in commands]

    # 执行所有命令
    for command in commands:
        run_command(command, './run.log')
    return True


def hyper_search(exp_name, hyper_dict, grid_search=0, retrain=1, debug=0):
    """
    入口函数：选择使用网格搜索还是逐步搜索
    """
    if grid_search:
        return grid_search_hyperparameters(exp_name, hyper_dict, retrain, debug)
    else:
        return sequential_hyper_search(exp_name, hyper_dict, retrain, debug)


def run_and_get_metric(cmd_str, config, chosen_hyper, debug=False):
    """
    运行训练命令，并提取 metric
    """
    print(cmd_str)
    config.__dict__.update(chosen_hyper)
    log_filename = get_experiment_name(config)

    if debug:
        print(log_filename, chosen_hyper)
    else:
        subprocess.run(cmd_str, shell=True)

    metric_file_address = f'./results/metrics/' + get_experiment_name(config)
    this_expr_metrics = pickle.load(open(metric_file_address + '.pkl', 'rb'))

    # 选择最优 metric
    classification_task = getattr(config, 'classification', False)
    metric_name = 'AC' if classification_task else 'MAE'
    best_value = np.mean(this_expr_metrics[metric_name])
    return best_value


def grid_search_hyperparameters(exp_name, hyper_dict, retrain, debug):
    """
    进行网格搜索（笛卡尔积搜索所有超参数组合）
    """
    config = get_config(exp_name)
    classification_task = getattr(config, 'classification', False)

    log_file = f'./run.log'
    hyper_keys = list(hyper_dict.keys())
    hyper_values_list = [hyper_dict[k] for k in hyper_keys]

    best_metric = 0 if classification_task else 1e9
    best_combo = None

    with open(log_file, 'a') as f:
        f.write("================== Grid Search ==================\n")
        for combo in product(*hyper_values_list):
            combo_dict = dict(zip(hyper_keys, combo))

            command = f"python train_model.py --exp_name {exp_name} --hyper_search 1 --retrain {retrain} "
            command += " ".join([f"--{k} {v}" for k, v in combo_dict.items()])

            # 运行并获取结果
            current_metric = run_and_get_metric(command, config, combo_dict, debug)

            # 选择最优参数
            if classification_task:
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_combo = combo_dict
            else:
                if current_metric < best_metric:
                    best_metric = current_metric
                    best_combo = combo_dict

            print(f"Combo: {combo_dict}, Metric= {current_metric}\n")
            f.write(f"Combo: {combo_dict}, Metric= {current_metric}\n")

        f.write(f"Best combo: {best_combo}, Best metric: {best_metric}\n")
    return best_combo


def sequential_hyper_search(exp_name, hyper_dict, retrain, debug):
    """
    逐步搜索超参数，每次调整一个参数，并保持其他最优值
    """
    config = get_config(exp_name)
    classification_task = getattr(config, 'classification', False)

    log_file = f'./run.log'
    best_hyper = {}

    with open(log_file, 'a') as f:
        f.write("================== Sequential Hyper Search ==================\n")

        for hyper_name, hyper_values in hyper_dict.items():
            if len(hyper_values) == 1:
                best_hyper[hyper_name] = hyper_values[0]
                continue

            local_best_metric = 0 if classification_task else 1e9
            current_best_value = None

            for value in hyper_values:
                command = f"python train_model.py --exp_name {exp_name} --hyper_search 1 --retrain {retrain} "
                command += " ".join([f"--{k} {v}" for k, v in best_hyper.items()])
                command += f" --{hyper_name} {value}"

                # 运行并获取 metric
                chosen_dict = best_hyper.copy()
                chosen_dict[hyper_name] = value
                current_metric = run_and_get_metric(command, config, chosen_dict, debug)

                if classification_task:
                    if current_metric > local_best_metric:
                        local_best_metric = current_metric
                        current_best_value = value
                else:
                    if current_metric < local_best_metric:
                        local_best_metric = current_metric
                        current_best_value = value

            # 更新最优超参数
            best_hyper[hyper_name] = current_best_value

        f.write(f"The Best Hyperparameters: {best_hyper}\n")
    return best_hyper



def run_command(command, log_file, retry_count=0):
    success = False
    while not success:
        # 获取当前时间并格式化
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 如果是重试的命令，标记为 "Retrying"
        if retry_count > 0:
            retry_message = "Retrying"
        else:
            retry_message = "Running"

        # 将执行的命令和时间写入日志文件
        with open(log_file, 'a') as f:
            f.write(f"{retry_message} at {current_time}: {command}\n")

        # 直接执行命令，将输出和错误信息打印到终端
        process = subprocess.run(f'ulimit -s unlimited; ulimit -c unlimited&& ulimit -a && echo {command} &&' + command, shell=True)

        # 根据返回码判断命令是否成功执行
        if process.returncode == 0:
            success = True
        else:
            with open(log_file, 'a') as f:
                f.write(f"Command failed, retrying in 3 seconds: {command}\n")
            retry_count += 1
            time.sleep(3)  # 等待一段时间后重试