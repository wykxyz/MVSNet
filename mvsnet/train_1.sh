#!/bin/bash
#view2d64:8.8G, view3d64:1.04G, view3d128:1.08G
n=1
part=AD
now=$(date +"%Y%m%d_%H%M%S")
model=GRU_W
view=3
max_d=128
data=$(date +"%m%d")
name=${data}_${n}_${model}_v${view}_d${max_d}
echo $name
srun --mpi=pmi2 --gres=gpu:${n}  \
        -p $part -w SH-IDC1-10-5-30-208 -n1 \
        --ntasks-per-node=1 \
        -J $name -K \
        python -u train.py \
        --num_gpus=$n \
        --regularization=$model \
        --dtu_data_root=/mnt/lustre/yihongwei/yihongwei/dataset/dtu \
        --view_num=$view \
        --max_d=$max_d \
        --log_dir=../tf_log/${name} \
        --model_dir=../model/${name} \
        2>&1|tee ./logs/${name}-${now}.log &
