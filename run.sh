#!/bin/bash

export py=/users/hert5217/anaconda3/envs/gputorch/bin/python3

# addqueue --gputype rtx2080with12gb --gpus 1 -q gpushort -n 1x4 -m 7 -s $py train_data_agd_transformer.py rtx4090with24gb
# addqueue --gputype rtx4090with24gb --gpus 1 -q gpulong -n 1x4 -m 7 -s $py train_data_agd_transformer.py --batchsize 5000 
addqueue --gputype rtx4090with24gb --gpus 1 -q gpulong -n 1x4 -m 7 -s $py train_data_agd_transformer.py --beta 0.9 --lr 0.00005 --wd 0.00001 --numskills 5 --alpha 1.9

# Add linear layer at the front
# potentially move down to 16
# python3 train_data_agd_transformer.py --beta 0.9 --lr 0.00005 --wd 0.00001