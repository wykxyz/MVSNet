#!/bin/bash
n=4
part=AD
data=$(date +"%m%d")
#model=GRU_NONLOCALVIEWNUM
#model=GRU_WGATE
model=GRU_NONLOCALHW
view=3
max_d=128
interval_scale=1.06
#interval_scale=1.59
name=${data}_${n}_${model}_v${view}_d${max_d}_i${interval_scale}_fixresnet
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
        --interval_scale=${interval_scale} \
        --log_dir=../tf_log/${name} \
        --model_dir=../model/${name} \
        2>&1|tee ../logs/${name}-${now}.log &
