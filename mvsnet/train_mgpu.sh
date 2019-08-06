#!/bin/bash
n=8
part=AD
data=0805
model=GRU
name=${data}_${n}_${model}
now=$(date +"%Y%m%d_%H%M%S")
echo $name
srun --mpi=pmi2 --gres=gpu:${n}  \
        -p $part -n1 \
        --ntasks-per-node=1 \
        -J $name -K \
        python -u train.py \
        --num_gpus=$n \
        --regularization=$model \
        --dtu_data_root=/mnt/lustre/yihongwei/yihongwei/dataset/dtu \
        --log_dir=../tf_log/${name} \
        --model_dir=../model/${name} \
        2>&1|tee ./logs/${name}-${now}.log &
