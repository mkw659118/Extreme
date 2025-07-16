#!/bin/bash
# python run_cluster.py
for i in {0..0}
do
    python run_train.py --idx $i --seq_len 36 --pred_len 7 --Constraint False
    python run_train.py --idx $i --seq_len 36 --pred_len 30 --Constraint False
    python run_train.py --idx $i --seq_len 36 --pred_len 60 --Constraint False
    python run_train.py --idx $i --seq_len 36 --pred_len 90 --Constraint False
done
python run_plot.py

for i in {0..0}
do
    python run_train.py --idx $i --seq_len 36 --pred_len 7 --Constraint True
    python run_train.py --idx $i --seq_len 36 --pred_len 30 --Constraint True
    python run_train.py --idx $i --seq_len 36 --pred_len 60 --Constraint True
    python run_train.py --idx $i --seq_len 36 --pred_len 90 --Constraint True
done
python run_plot.py
# python run_service.py
