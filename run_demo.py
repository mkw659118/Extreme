# coding : utf-8
# Author : yuxiang Zeng
import subprocess

from utils.exp_sh import once_experiment, log_message

# 在这里写下超参数探索空间
hyper_dict = {
    'rounds': [1],
    'rank': [64],
    # 'num_layers': [4],
    # 'dataset': ['financial'],  # weather financial lottery
    # 'att': ['self'],
    # 'norm': ['rms'],
    # 'ffn': ['moe'],
    # 'loss_coef': [0.001],
    # 'fft': [True],
    # 'revin': [False],
    # 'dis_method': ['cosine'],  #
    # 'dis_method': ['none', 'cosine', 'euclidean', 'manhattan', 'minkowski', 'linear', 'kl', 'mahalanobis' 'dtw'],  #
    'idx': [i for i in range(33)]
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
    # once_experiment('RNNConfig', hyper_dict)
    # once_experiment('LSTMConfig', hyper_dict)
    # once_experiment('GRUConfig', hyper_dict)
    # once_experiment('CrossformerConfig', hyper_dict)
    # once_experiment('TimesNetConfig', hyper_dict)
    # once_experiment('timeLLMConfig', hyper_dict)
    return True


def Ablation():
    hyper_dict = {
        # 'dataset': ['a', 'b', 'c'],
        'ablation': [1, 2, 3],
    }
    once_experiment('TestConfig', hyper_dict, grid_search=1, retrain=1)
    return True


def Our_model(hyper=None):
    once_experiment('TestConfig', hyper_dict, grid_search=0)
    subprocess.run(f'python model_pred.py', shell=True)
    return True


if __name__ == "__main__":
    log_message("\nExperiment Start!!!")
    experiment_run()
    log_message("All commands executed successfully.\n")


