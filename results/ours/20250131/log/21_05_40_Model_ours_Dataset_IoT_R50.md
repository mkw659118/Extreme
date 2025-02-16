```python
|2025-01-31 21:05:40| {
     'ablation': 0, 'bs': 32, 'classification': True, 'continue_train': False,
     'dataset': IoT, 'debug': False, 'decay': 0.0001, 'density': 0.8,
     'device': cpu, 'epochs': 200, 'eval_set': True, 'hyper_search': False,
     'log': <utils.logger.Logger object at 0x7fccc91596d0>, 'logger': None, 'loss_func': CrossEntropyLoss, 'lr': 0.001,
     'model': ours, 'optim': AdamW, 'path': ./datasets/, 'patience': 50,
     'rank': 50, 'record': True, 'retrain': True, 'rounds': 2,
     'seed': 0, 'time_interval': 10, 'train_size': 500, 'try_exp': 1,
     'verbose': 10,
}
|2025-01-31 21:05:40| ********************Experiment Start********************
|2025-01-31 21:05:47| Round=1 BestEpoch=158 Ac=0.9833 Pr=0.9836 Rc=0.9833 F1=0.9833 Training_time=4.1 s 

|2025-01-31 21:05:52| Round=2 BestEpoch=147 Ac=0.9800 Pr=0.9800 Rc=0.9800 F1=0.9800 Training_time=3.7 s 

|2025-01-31 21:05:52| ********************Experiment Results:********************
|2025-01-31 21:05:52| AC: 0.9817 ± 0.0017
|2025-01-31 21:05:52| PR: 0.9818 ± 0.0018
|2025-01-31 21:05:52| RC: 0.9817 ± 0.0017
|2025-01-31 21:05:52| F1: 0.9817 ± 0.0017
|2025-01-31 21:05:52| train_time: 3.9132 ± 0.1880
|2025-01-31 21:05:52| Flops: 2560
|2025-01-31 21:05:52| Params: 100
|2025-01-31 21:05:52| Inference time: 0.01 ms
|2025-01-31 21:05:53| ********************Experiment Success********************
```

