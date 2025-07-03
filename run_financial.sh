#!/bin/bash
#python run_cluster.py

#all_idx=(0 23 28 30 36 37 55 64 66 72 79 120)
#for i in "${all_idx[@]}"
#for i in {0..149}
#do
#    python run_train.py --idx $i --pred_len 7
#done

python run_service.py --pred_len 7

# for i in {0..149}
# do
#     python run_train.py --idx $i --seq_len 36 --pred_len 30
# done


# for i in {0..149}
# do
#     python run_train.py --idx $i --seq_len 96 --pred_len 60
# done

# for i in {0..149}
# do
    # python run_train.py --idx $i --seq_len 96 --pred_len 90
# done
# python run_service.py
