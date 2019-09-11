#!/bin/bash
n=1
part=AD
data=$(date +"%m%d")
#model=GRU_WORI
#model=3DCNNs
#model=GRU
model=GRU_NONLOCALVIEWNUM
view=5
max_w=640
max_h=512
max_d=256
stepvalue=100000
interval_scale=0.8
mode=validation
name=${data}_${mode}_${n}_${model}_v${view}_w${max_w}_h${max_h}_d${max_d}_i${interval_scale}_wori
now=$(date +"%Y%m%d_%H%M%S")
echo $name
echo $now
#        --ckpt_step=160000 \
#--upsampling=False \
#        --model_dir=/mnt/lustre/zhouhui/yihongwei/workspace/MVSNet/model/0905_4_GRU_NONLOCALVIEWNUM_v3_d128_i1.59 \
#        --model_dir=/mnt/lustre/zhouhui/yihongwei/workspace/MVSNet/model/tf_model \
srun --mpi=pmi2 --gres=gpu:${n}  \
        -p $part -n1 \
        --ntasks-per-node=1 \
        -J $name -K \
        python2 -u validate.py \
        --regularization=${model} \
        --dtu_data_root=/mnt/lustre/share/yihongwei/dataset/dtu \
        --model_dir=/mnt/lustre/zhouhui/yihongwei/workspace/MVSNet/model/0905_4_GRU_NONLOCALVIEWNUM_v3_d128_i1.59 \
        --log_dirs=../tf_log/${name} \
        --ckpt_step=${stepvalue} \
        --mode=${mode} \
        --max_w=${max_w} \
        --max_h=${max_h} \
        --max_d=${max_d} \
        --interval_scale=${interval_scale} \
        2>&1|tee ../logs/${name}-${now}.log &
