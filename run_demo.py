# coding : utf-8
# Author : yuxiang Zeng
from utils.exp_sh import once_experiment
from datetime import datetime

# 在这里写下超参数探索空间
hyper_dict = {
    'rank': [128],
    'num_layers': [3],
    'dataset': ['weather'],  # weather electricity
}

######################################################################################################
# 这里是总执行实验顺序！！！！！！！！
def experiment_run():
    # Baselines()
    # Ablation()
    Our_model()
    return True

def Baselines():
    # once_experiment('MLPConfig', hyper_dict)
    once_experiment('RNNConfig', hyper_dict)
    # once_experiment('LSTMConfig', hyper_dict)
    # once_experiment('GRUConfig', hyper_dict)
    # once_experiment('CrossformerConfig', hyper_dict)
    # once_experiment('TimesNetConfig', hyper_dict)
    return True

def Ablation():
    hyper_dict = {
        # 'dataset': ['a', 'b', 'c'],
        'ablation': [1, 2, 3],
    }
    once_experiment('TestConfig', hyper_dict, grid_search=1, retrain=1)
    return True


def Our_model(hyper=None):
    once_experiment('TestConfig', hyper_dict)
    return True


def log_message(message):
    log_file = "run.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

if __name__ == "__main__":
    log_message("Experiment Start!!!")
    experiment_run()
    log_message("All commands executed successfully.\n")


