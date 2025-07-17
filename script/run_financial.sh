#!/bin/bash
# python run_cluster.py
for i in {0..10}
do
    python run_train.py --idx $i --seq_len 36 --pred_len 7 --constraint True
    python run_train.py --idx $i --seq_len 36 --pred_len 30 --constraint True
    python run_train.py --idx $i --seq_len 36 --pred_len 60 --constraint True
    python run_train.py --idx $i --seq_len 36 --pred_len 90 --constraint True
done
# python run_plot.py --constraint False
python run_plot.py --constraint True
python run_service.py


