#!/bin/bash
pred_lens=(96 192 336 720)
diffusion_flags=(True)

# noise_steps_arr=(100 200 1000)
lamda_arr=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# for len in "${pred_lens[@]}"
# do
#   for diffusion in "${diffusion_flags[@]}"
#   do
#     for noise_steps in "${noise_steps_arr[@]}"
#     do
#       echo "Running with pred_len=$len, diffusion=$diffusion, noise_steps=$noise_steps"
#       python run_train.py \
#         --exp_name "Transformer2Config" \
#         --rounds 3 \
#         --retrain 1 \
#         --pred_len "$len" \
#         --revin True \
#         --diffusion "$diffusion" \
#         --noise_steps "$noise_steps"
#     done
#   done
# done



for len in "${pred_lens[@]}"
do
  for diffusion in "${diffusion_flags[@]}"
  do
    for lamda in "${lamda_arr[@]}"
    do
      echo "Running with pred_len=$len, diffusion=$diffusion,lamda=$lamda"
      python run_train.py \
        --exp_name "Transformer2Config" \
        --rounds 3 \
        --retrain 1 \
        --pred_len "$len" \
        --revin True \
        --diffusion "$diffusion"\
        --lamda "$lamda"
    done
  done
done