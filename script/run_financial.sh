#!/bin/bash
# python run_cluster.py
for i in {0..10}
do
    python run_train.py --idx $i --seq_len 17 --pred_len 7
    python run_train.py --idx $i --seq_len 36 --pred_len 30
    python run_train.py --idx $i --seq_len 36 --pred_len 60
    python run_train.py --idx $i --seq_len 36 --pred_len 90
done
# python run_service.py
