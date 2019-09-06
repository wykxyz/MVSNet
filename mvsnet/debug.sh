#!/bin/bash
n=1
part=AD
data=$(date +"%m%d")
model=GRU_NONLOCALHW
view=3
max_d=128
name=debug_${model}
now=$(date +"%Y%m%d_%H%M%S")
echo $name
echo $now
srun --mpi=pmi2 --gres=gpu:${n}  \
        -p $part -n1 \
        --ntasks-per-node=1 \
        -J $name -K \
        python2 -u train.py \
        --num_gpus=$n \
        --regularization=$model \
        --dtu_data_root=/mnt/lustre/share/yihongwei/dataset/dtu \
        --view_num=${view} \
        --max_d=${max_d} \
        --log_dir=../tf_log/${name} \
        --model_dir=../model/${name} \
        2>&1|tee ../logs/${name}-${now}.log &
