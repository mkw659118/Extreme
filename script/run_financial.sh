#!/bin/bash
# python run_cluster.py
for i in {0..1}
do
    python run_train.py --idx $i --seq_len 36 --pred_len 7 --topelize False
    python run_train.py --idx $i --seq_len 36 --pred_len 30 --topelize False
    python run_train.py --idx $i --seq_len 36 --pred_len 60 --topelize False
    python run_train.py --idx $i --seq_len 36 --pred_len 90 --topelize False
done
python run_plot.py
# python run_service.py
