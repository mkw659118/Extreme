```python
|2025-01-31 20:59:26| {
     'ablation': 0, 'bs': 128, 'classification': True, 'continue_train': False,
     'dataset': IoT, 'debug': False, 'decay': 0.0001, 'density': 0.8,
     'device': cpu, 'epochs': 200, 'eval_set': True, 'hyper_search': False,
     'log': <utils.logger.Logger object at 0x7f7c682f1b80>, 'logger': None, 'loss_func': CrossEntropyLoss, 'lr': 0.001,
     'model': ours, 'optim': AdamW, 'path': ./datasets/, 'patience': 50,
     'rank': 32, 'record': True, 'retrain': True, 'rounds': 2,
     'seed': 0, 'time_interval': 10, 'train_size': 500, 'try_exp': 1,
     'verbose': 10,
}
|2025-01-31 20:59:26| ********************Experiment Start********************
|2025-01-31 20:59:30| Round=1 BestEpoch=194 Ac=0.9267 Pr=0.9354 Rc=0.9267 F1=0.9261 Training_time=2.1 s 

|2025-01-31 20:59:32| Round=2 BestEpoch=184 Ac=0.9000 Pr=0.9198 Rc=0.9000 F1=0.8980 Training_time=1.7 s 

|2025-01-31 20:59:32| ********************Experiment Results:********************
|2025-01-31 20:59:32| AC: 0.9133 ± 0.0133
|2025-01-31 20:59:32| PR: 0.9276 ± 0.0078
|2025-01-31 20:59:32| RC: 0.9133 ± 0.0133
|2025-01-31 20:59:32| F1: 0.9120 ± 0.0140
|2025-01-31 20:59:32| train_time: 1.8884 ± 0.1755
|2025-01-31 20:59:32| Flops: 10240
|2025-01-31 20:59:32| Params: 100
|2025-01-31 20:59:32| Inference time: 0.01 ms
|2025-01-31 20:59:33| ********************Experiment Success********************
```

