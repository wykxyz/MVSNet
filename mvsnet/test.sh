#!/bin/bash
n=1
part=AD
name=evaluate
#        --ckpt_step=160000 \
srun --mpi=pmi2 --gres=gpu:${n}  \
        -p $part -n1 \
        --ntasks-per-node=1 \
        -J $name -K \
        python2 -u test.py \
        --regularization='GRU_NONLOCALVIEWNUM' \
        --model_dir=../model/0905_4_GRU_NONLOCALVIEWNUM_v3_d128_i1.06 \
        --dense_folder=/mnt/lustre/share/yihongwei/dataset/mvs-test/family \
        --ckpt_step=135000 \
        --max_w=1920 \
        --max_h=1040 \
        --max_d=128 \
        --interval_scale=1.06 \
        --upsampling=True &
