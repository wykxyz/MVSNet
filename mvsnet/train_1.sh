#!/bin/bash
n=4
part=AD
data=$(date +"%m%d")
model=GRU_W
view=3
max_d=128
name=${data}_${n}_${model}_v${view}_d${max_d}
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
        2>&1|tee ./logs/${name}-${now}.log &
