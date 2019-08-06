#!/bin/bash
n=1
part=AD
name=v2_gpu1_3DCNNs
srun --mpi=pmi2 --gres=gpu:${n}  \
        -p $part -n1 \
        --ntasks-per-node=1 \
        -J $name -K \
        python -u train.py \
        --regularization='3DCNNs' \
        --dtu_data_root=/mnt/lustre/yihongwei/yihongwei/dataset/dtu \
        --log_dir=../tf_log/$name \
        --model_dir=../model/$name &
