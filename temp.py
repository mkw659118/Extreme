import os
import pickle
import shutil


def get_benchmark_code():
    with open('./datasets/benchmark.pkl', 'rb') as f:
        group = pickle.load(f)
        group = group['stock-270000']
    return group

if __name__ == '__main__':
    all_list = os.listdir('./datasets/financial')
    with open('./datasets/benchmark.pkl', 'rb') as f:
        group = pickle.load(f)
        group = group['stock-270000']
        group.remove('013869')
        group.remove('013870')
    print(group)
    print(len(group))
    os.makedirs('./datasets/financial/benchmark', exist_ok=True)
    for i in range(len(group)):
        # print(group[i])
        try:
            shutil.copy(os.path.join('./datasets/financial', group[i] + '.pkl'), './datasets/financial/benchmark')
        except Exception as e:
            print(e)