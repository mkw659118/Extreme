```python
|2025-01-31 20:59:45| {
     'ablation': 0, 'bs': 128, 'classification': True, 'continue_train': False,
     'dataset': IoT, 'debug': False, 'decay': 0.0001, 'density': 0.8,
     'device': cpu, 'epochs': 200, 'eval_set': True, 'hyper_search': False,
     'log': <utils.logger.Logger object at 0x7fede8481b50>, 'logger': None, 'loss_func': CrossEntropyLoss, 'lr': 0.001,
     'model': ours, 'optim': AdamW, 'path': ./datasets/, 'patience': 50,
     'rank': 16, 'record': True, 'retrain': True, 'rounds': 2,
     'seed': 0, 'time_interval': 10, 'train_size': 500, 'try_exp': 1,
     'verbose': 10,
}
|2025-01-31 20:59:45| ********************Experiment Start********************
|2025-01-31 20:59:48| Round=1 BestEpoch=194 Ac=0.9267 Pr=0.9354 Rc=0.9267 F1=0.9261 Training_time=1.8 s 

|2025-01-31 20:59:51| Round=2 BestEpoch=184 Ac=0.9000 Pr=0.9198 Rc=0.9000 F1=0.8980 Training_time=1.7 s 

|2025-01-31 20:59:51| ********************Experiment Results:********************
|2025-01-31 20:59:51| AC: 0.9133 ± 0.0133
|2025-01-31 20:59:51| PR: 0.9276 ± 0.0078
|2025-01-31 20:59:51| RC: 0.9133 ± 0.0133
|2025-01-31 20:59:51| F1: 0.9120 ± 0.0140
|2025-01-31 20:59:51| train_time: 1.7463 ± 0.0770
|2025-01-31 20:59:51| Flops: 10240
|2025-01-31 20:59:51| Params: 100
|2025-01-31 20:59:51| Inference time: 0.01 ms
|2025-01-31 20:59:51| ********************Experiment Success********************
```

